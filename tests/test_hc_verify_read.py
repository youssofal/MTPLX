"""The hyper-connection read at verify widths (mtplx.kernels.hc_verify_read).

Its claim is bit identity with the compiled stock chain it replaces inside a
compiled verify body. Every comparison here is on the raw 16-bit patterns, so
a moved sign of zero counts as a difference.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.compile_state import compiled_step_body
from mtplx.kernels import hc_verify_read
from mtplx.models.qwen4_exp import GatedResidual

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the kernels are Metal kernels"
)

FAMILY = SimpleNamespace(hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(hc_verify_read.ENV, raising=False)
    monkeypatch.delenv("MTPLX_FUSED_HC", raising=False)
    monkeypatch.delenv("MTPLX_FUSED_HC_V3", raising=False)
    hc_verify_read.reset_for_tests()
    yield
    hc_verify_read.reset_for_tests()


def _residual(args=FAMILY, *, dtype=mx.bfloat16, combine=True, seed=0) -> GatedResidual:
    module = GatedResidual(args, use_combine=combine)
    hcd = args.hc_count * args.hidden_size
    keys = mx.random.split(mx.random.key(seed), 4)
    module.hc_norm.weight = (1.0 + 0.1 * mx.random.normal((hcd,), key=keys[0])).astype(dtype)
    module.input_mix_weight_down.weight = (
        0.01 * mx.random.normal((args.hc_lowrank, hcd), key=keys[1])
    ).astype(dtype)
    module.input_mix_weight_up.weight = (
        0.06 * mx.random.normal((hcd, args.hc_lowrank), key=keys[2])
    ).astype(dtype)
    if combine:
        module.block_inject_weight.weight = (
            0.01 * mx.random.normal((args.hc_count, hcd), key=keys[3])
        ).astype(dtype)
    mx.eval(module.parameters())
    return module


def _stream(args, rows, *, dtype=mx.bfloat16, seed=1):
    keys = mx.random.split(mx.random.key(seed), 3)
    hcd = args.hc_count * args.hidden_size
    x = (3.0 * mx.random.normal((1, rows, hcd), key=keys[0])).astype(dtype)
    block = (2.0 * mx.random.normal((1, rows, args.hidden_size), key=keys[1])).astype(dtype)
    gates = mx.random.uniform(0.0, 2.0, (1, rows, args.hc_count), key=keys[2]).astype(dtype)
    mx.eval(x, block, gates)
    return x, block, gates


def _bits(a: mx.array) -> np.ndarray:
    return np.array(mx.view(a, mx.uint16))


def _stock(module, x, pending):
    """The parent: the stock chain as the compiled verify body runs it."""

    if pending is None:
        return mx.compile(lambda s: module(s))(x)
    return mx.compile(lambda s, b, g: module(s, pending=(b, g)))(x, *pending)


def _fused(module, x, pending):
    rows = x.shape[1]
    hcd = x.shape[-1]
    combine = "block_inject_weight" in module
    return hc_verify_read.read_rows(
        x.reshape(rows, hcd),
        module.hc_norm.weight,
        module.input_mix_weight_down.weight,
        module.input_mix_weight_up.weight,
        module.block_inject_weight.weight if combine else None,
        None if pending is None else pending[0].reshape(rows, module.hidden_size),
        None if pending is None else pending[1].reshape(rows, module.hc_count),
        hc=module.hc_count,
        eps=module.hc_norm.eps,
        sigmoid=hc_verify_read.sigmoid_table(x.dtype),
    )


def _assert_same(module, x, pending):
    rows = x.shape[1]
    want = _stock(module, x, pending)
    got_mixed, got_written, got_inject = _fused(module, x, pending)
    if "block_inject_weight" not in module:
        want = (want, None, None)
    want_mixed, want_written, want_inject = want
    mx.eval(*[a for a in (want_mixed, want_written, want_inject, got_mixed, got_written) if a is not None])
    np.testing.assert_array_equal(
        _bits(got_mixed), _bits(want_mixed.reshape(rows, module.hidden_size))
    )
    if want_inject is not None:
        np.testing.assert_array_equal(
            _bits(got_inject), _bits(want_inject.reshape(rows, module.hc_count))
        )
        np.testing.assert_array_equal(
            _bits(got_written), _bits(want_written.reshape(rows, x.shape[-1]))
        )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows", range(2, 9))
def test_every_width_matches_the_compiled_stock_chain_bit_for_bit(dtype, rows):
    reader = _residual(dtype=dtype, seed=rows)
    mixer = _residual(dtype=dtype, combine=False, seed=100 + rows)
    x, block, gates = _stream(FAMILY, rows, dtype=dtype, seed=200 + rows)
    _assert_same(reader, x, None)
    _assert_same(reader, x, (block, gates))
    _assert_same(mixer, x, None)


def test_signed_zero_products_leave_the_stream_sum_as_the_stock_reduce_does():
    # A column whose normed value is -0 in every stream makes every product
    # -0; the stock column reduce starts each stream from +0, so the mean is
    # +0. Summing the products directly would return -0.
    reader = _residual(seed=3)
    x, _block, _gates = _stream(FAMILY, 4, seed=4)
    width = FAMILY.hidden_size
    column = mx.zeros((1, 4, 1), dtype=mx.bfloat16) * -1.0
    pieces = []
    for stream in range(FAMILY.hc_count):
        start = stream * width
        pieces += [column, x[..., start + 1 : start + width]]
    x = mx.concatenate(pieces, axis=-1)
    assert int(_bits(x)[0, 0, 0]) == 0x8000  # the planted value is -0
    want = _stock(reader, x, None)[0]
    got = _fused(reader, x, None)[0]
    mx.eval(want, got)
    assert int(_bits(want)[0, 0, 0]) == 0x0000
    np.testing.assert_array_equal(_bits(got), _bits(want.reshape(4, width)))


DIVERGENT = -6.84375  # the one bfloat16 input where MLX's two sigmoid lowerings differ


def _bf16_index(value: float) -> int:
    return int(_bits(mx.array([value], dtype=mx.bfloat16))[0])


def test_the_tables_are_the_stock_lowerings_at_every_input():
    # Each row equals, at every finite input, the lowering its sites take in
    # the running MLX: row 0 the fused one (silu, inject gate), row 1 the
    # standalone primitive (mix gate).
    for dtype in (mx.bfloat16, mx.float16):
        assert hc_verify_read._table_check(dtype) is None
    table = _bits(hc_verify_read.sigmoid_table(mx.bfloat16))
    at = _bf16_index(DIVERGENT)
    value = mx.array([DIVERGENT], dtype=mx.bfloat16)
    standalone = int(_bits(mx.sigmoid(value))[0])
    assert int(table[65536 + at]) == standalone
    # MLX 0.32.2 parts the two lowerings at this input, so one table would be
    # wrong at one of the sites there. MLX 0.32.3 gives both the same value;
    # the rows follow whichever build is running.
    from importlib.metadata import version

    if version("mlx") == "0.32.2":
        assert table[at] != table[65536 + at]


def test_the_mix_gate_takes_the_standalone_sigmoid_where_the_lowerings_part():
    # Scaled up-projection weights put a mix-gate input exactly on -6.84375
    # with this seed (asserted, so the case cannot silently stop covering the
    # divergent input); the fused sigmoid there would move one output by one ulp.
    import mlx.nn as nn

    reader = _residual(seed=5)
    reader.input_mix_weight_up.weight = reader.input_mix_weight_up.weight * 400.0
    mx.eval(reader.parameters())
    x, block, gates = _stream(FAMILY, 4, seed=6)
    written = x + (block[..., None, :] * gates[..., :, None]).reshape(*x.shape)
    normed = reader.hc_norm(written)
    mix = nn.silu(reader.input_mix_weight_down(normed) / FAMILY.hc_count)
    gate_inputs = reader.input_mix_weight_up(mix)
    mx.eval(gate_inputs)
    assert (_bits(gate_inputs) == _bf16_index(DIVERGENT)).any()
    _assert_same(reader, x, (block, gates))


def test_each_site_reads_its_own_table_row():
    reader = _residual(seed=15)
    x, block, gates = _stream(FAMILY, 4, seed=16)
    table = hc_verify_read.sigmoid_table(mx.bfloat16)
    n = 1 << 16
    zeros = mx.zeros((n,), dtype=mx.bfloat16)

    def run(sigmoid):
        rows = x.shape[1]
        out = hc_verify_read.read_rows(
            x.reshape(rows, -1),
            reader.hc_norm.weight,
            reader.input_mix_weight_down.weight,
            reader.input_mix_weight_up.weight,
            reader.block_inject_weight.weight,
            None,
            None,
            hc=FAMILY.hc_count,
            eps=reader.hc_norm.eps,
            sigmoid=sigmoid,
        )
        mx.eval(out[0], out[2])
        return _bits(out[0]), _bits(out[2])

    real_mixed, real_inject = run(table)
    # No mix gate: every product is a zero, the mean is +0; inject untouched.
    mixed, inject = run(mx.concatenate([table[:n], zeros]))
    assert (mixed == 0).all()
    np.testing.assert_array_equal(inject, real_inject)
    # No fused sigmoid: the inject gate is zero and the mixed output moves.
    mixed, inject = run(mx.concatenate([zeros, table[n:]]))
    assert (inject == 0).all()
    assert (mixed != real_mixed).any()


def _graph_primitives(outputs, tmp_path) -> list[str]:
    """Primitive names in the lazy graph behind ``outputs``."""

    path = tmp_path / "graph.dot"
    mx.export_to_dot(str(path), *outputs)
    names = []
    for line in path.read_text().splitlines():
        if "shape=rectangle" in line and 'label ="' in line:
            names.append(line.split('label ="', 1)[1].split('"', 1)[0])
    return names


def _served_here() -> bool:
    """The read's device route on the GPU running the tests."""

    from mtplx.nax_detect import gpu_architecture

    return bool(hc_verify_read.device_route(gpu_architecture())[0])


needs_the_read = pytest.mark.skipif(
    not _served_here(),
    reason="this GPU's route keeps the stock chain (M1, M2, an M3 Ultra): nothing to engage",
)


def _model(module_seed=7):
    reader = _residual(seed=module_seed)
    mixer = _residual(combine=False, seed=module_seed + 1)
    return SimpleNamespace(
        layers=[SimpleNamespace(attn_hyper_connection=reader)],
        hyper_connection_mixer=mixer,
    )


def _installed_model(module_seed=7):
    """The read installed the way the route installs it: on this GPU's own
    architecture (never a forced one)."""

    model = _model(module_seed)
    report = hc_verify_read.install(model, rows=(4,))
    return model, report


@needs_the_read
def test_inside_a_compiled_body_one_read_is_three_kernels(tmp_path):
    model, report = _installed_model()
    assert report["installed"], report
    reader = model.layers[0].attn_hyper_connection
    x, block, gates = _stream(FAMILY, 4, seed=8)
    with compiled_step_body():
        mixed, written, inject = reader(x, pending=(block, gates))
    names = _graph_primitives([mixed, written, inject], tmp_path)
    compute = [name for name in names if name not in {"Reshape"}]
    assert compute.count("CustomKernel") == 3, names
    assert set(compute) == {"CustomKernel"}, names
    assert hc_verify_read.engagement()["traces"] == 1


def test_outside_a_compiled_body_the_stock_chain_runs(tmp_path):
    model, _report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    x, block, gates = _stream(FAMILY, 4, seed=9)
    mixed, written, inject = reader(x, pending=(block, gates))
    names = _graph_primitives([mixed, written, inject], tmp_path)
    assert "CustomKernel" not in names
    assert "RMSNorm" in names
    assert hc_verify_read.engagement()["traces"] == 0


@pytest.mark.parametrize("rows", [1, 9])
def test_widths_outside_the_verify_window_keep_their_own_paths(rows):
    model, _report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    x, _block, _gates = _stream(FAMILY, rows, seed=10)
    with compiled_step_body():
        out = reader(x)
    mx.eval(out)
    assert hc_verify_read.engagement()["traces"] == 0


def test_a_width_the_probe_did_not_prove_keeps_the_stock_chain():
    model, _report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    x, _block, _gates = _stream(FAMILY, 3, seed=11)
    with compiled_step_body():
        mx.eval(reader(x))
    assert hc_verify_read.engagement()["traces"] == 0


def test_a_pending_write_on_the_stock_path_is_written_before_the_read():
    # The hand-over must not change the stock path: reading with a pending
    # write equals writing first and then reading.
    from mtplx.models.qwen4_exp import _hyper_residual_write

    reader = _residual(seed=12)
    x, block, gates = _stream(FAMILY, 4, seed=13)
    handed = reader(x, pending=(block, gates))
    written_first = reader(_hyper_residual_write(x, block, gates))
    mx.eval(*handed, *written_first)
    for a, b in zip(handed, written_first):
        np.testing.assert_array_equal(_bits(a), _bits(b))


@needs_the_read
def test_the_probe_turns_the_read_off_on_any_difference(monkeypatch, capsys):
    real = hc_verify_read.read_rows

    def one_ulp_off(*args, **kwargs):
        mixed, written, inject = real(*args, **kwargs)
        moved = mx.view(mx.view(mixed, mx.uint16) ^ mx.array(1, dtype=mx.uint16), mixed.dtype)
        return moved, written, inject

    monkeypatch.setattr(hc_verify_read, "read_rows", one_ulp_off)
    model, report = _installed_model()
    assert not report["installed"]
    assert "mixed differs" in report["disabled_reason"]
    assert report["probe_failures"] == 1
    assert "hyper-connection verify read off" in capsys.readouterr().out
    reader = model.layers[0].attn_hyper_connection
    x, _block, _gates = _stream(FAMILY, 4, seed=14)
    with compiled_step_body():
        mx.eval(reader(x))
    assert hc_verify_read.engagement()["traces"] == 0


def _zero_columns(x, hc, width, columns=8):
    """``x`` with the first ``columns`` values of every stream set to -0."""

    rows = x.shape[1]
    grouped = x.reshape(1, rows, hc, width)
    zeros = mx.zeros((1, rows, hc, columns), dtype=x.dtype) * -1.0
    return mx.concatenate([zeros, grouped[..., columns:]], axis=-1).reshape(x.shape)


@needs_the_read
def test_a_sign_only_difference_turns_the_read_off_and_the_stock_chain_serves(monkeypatch, capsys):
    # A read right in every bit but the sign of a zero (a stream sum started
    # from its first product instead of +0 gives -0 where the stock mean is
    # +0) must fail the probe; the compiled read then equals the stock chain.
    real = hc_verify_read.read_rows

    def zero_sign_slip(*args, **kwargs):
        mixed, written, inject = real(*args, **kwargs)
        bits = mx.view(mixed, mx.uint16)
        slipped = mx.where((bits & 0x7FFF) == 0, bits | 0x8000, bits)
        return mx.view(slipped, mixed.dtype), written, inject

    monkeypatch.setattr(hc_verify_read, "read_rows", zero_sign_slip)
    model, report = _installed_model()
    assert not report["installed"]
    assert "mixed differs" in report["disabled_reason"]
    assert "hyper-connection verify read off" in capsys.readouterr().out
    reader = model.layers[0].attn_hyper_connection
    x, _block, _gates = _stream(FAMILY, 4, seed=18)
    x = _zero_columns(x, FAMILY.hc_count, FAMILY.hidden_size)
    mx.eval(x)
    with compiled_step_body():
        got = mx.compile(lambda s: reader(s))(x)
    want = _stock(reader, x, None)
    mx.eval(*got, *want)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(_bits(a), _bits(b))
    assert (_bits(got[0])[..., :8] == 0).all()  # +0, not -0
    assert hc_verify_read.engagement()["traces"] == 0


@needs_the_read
def test_with_the_production_v3_flag_one_row_keeps_the_v3_read(monkeypatch, tmp_path):
    # The server arms MTPLX_FUSED_HC_V3=1 for this family, so single-row
    # decode reads take the 8-bit v3 read. The verify-width read serves 2 to
    # 8 rows only and leaves that path as it was, output bits included.
    from mtplx.kernels import hyper_connection_v3

    monkeypatch.setenv("MTPLX_FUSED_HC_V3", "1")
    model, report = _installed_model()
    assert report["installed"], report
    reader = model.layers[0].attn_hyper_connection
    # The device probe dispatches the v3 kernels once itself; take it first.
    served_v3 = hyper_connection_v3.device_supports_hyper_v3()
    calls = []
    real = hyper_connection_v3.fused_hyper_read_v3

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hyper_connection_v3, "fused_hyper_read_v3", counted)
    x1, block1, gates1 = _stream(FAMILY, 1, seed=19)
    with compiled_step_body():
        one = reader(x1, pending=(block1, gates1))
    mx.eval(*one)
    assert len(calls) == (1 if served_v3 else 0)
    assert hc_verify_read.engagement()["traces"] == 0

    hc_verify_read.reset_state()  # the same read with the verify read off
    with compiled_step_body():
        again = reader(x1, pending=(block1, gates1))
    mx.eval(*again)
    for a, b in zip(one, again):
        np.testing.assert_array_equal(_bits(a), _bits(b))

    model, report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    before = len(calls)
    x4, block4, gates4 = _stream(FAMILY, 4, seed=20)
    with compiled_step_body():
        four = reader(x4, pending=(block4, gates4))
    names = [name for name in _graph_primitives(list(four), tmp_path) if name != "Reshape"]
    assert names.count("CustomKernel") == 3, names
    assert len(calls) == before
    assert hc_verify_read.engagement()["traces"] == 1


def test_a_pending_write_of_another_dtype_keeps_the_stock_promotion():
    # A float32 block output promotes the stock write to a float32 stream:
    # bf16 1 + float32 2^-10 * 1 = float32 1.0009765625. The kernels would
    # write the stream in bfloat16 (1), so such a write keeps the stock path.
    model, report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    x, block, gates = _stream(FAMILY, 4, seed=21)
    x = mx.ones_like(x)
    block = mx.full(block.shape, 2.0**-10, dtype=mx.float32)
    gates = mx.ones_like(gates)
    mx.eval(x, block, gates)
    before = hc_verify_read.engagement()["traces"]
    with compiled_step_body():
        mixed, written, inject = mx.compile(lambda s, b, g: reader(s, pending=(b, g)))(x, block, gates)
    mx.eval(mixed, written, inject)
    assert hc_verify_read.engagement()["traces"] == before
    assert written.dtype == mx.float32
    assert float(written[0, 0, 0].item()) == 1.0009765625
    want = _stock(reader, x, (block, gates))
    mx.eval(*want)
    for a, b in zip((mixed, written, inject), want):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(np.array(a.astype(mx.float32)).view(np.uint32),
                                      np.array(b.astype(mx.float32)).view(np.uint32))


def test_the_projections_call_the_installed_library_template():
    # The down, inject and up projections run MLX's own GemvWide template, read
    # from the installed package's header when the kernels are built; nothing
    # of it is kept in this tree.
    source, reason = hc_verify_read._gemv_template()
    assert source is not None, reason
    assert "struct GemvWide" in source and "static METAL_FUNC void run(" in source
    assert "gemv_wide_do_axpby" not in source  # the bias branch is fixed off
    module_text = open(hc_verify_read.__file__).read()
    assert "simd_shuffle_down" not in module_text
    assert "GemvWide<T, NV, KL_DOWN>::run(" in module_text


@needs_the_read
def test_an_unreadable_library_header_keeps_the_stock_chain(monkeypatch, capsys):
    monkeypatch.setattr(
        hc_verify_read, "_gemv_template", lambda: (None, "the installed MLX gemv header is not readable here")
    )
    model, report = _installed_model()
    assert not report["installed"]
    assert "gemv header is not readable" in report["disabled_reason"]
    assert "hyper-connection verify read off" in capsys.readouterr().out
    reader = model.layers[0].attn_hyper_connection
    x, _block, _gates = _stream(FAMILY, 4, seed=22)
    with compiled_step_body():
        got = mx.compile(lambda s: reader(s))(x)
    want = _stock(reader, x, None)
    mx.eval(*got, *want)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(_bits(a), _bits(b))
    assert hc_verify_read.engagement()["traces"] == 0


@needs_the_read
def test_a_kernel_that_fails_to_build_leaves_the_stock_chain(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("pipeline refused")

    monkeypatch.setattr(hc_verify_read, "_norm_kernel", refuse)
    _model, report = _installed_model()
    assert not report["installed"]
    assert "pipeline refused" in report["disabled_reason"]


def test_the_switch_keeps_the_stock_chain(monkeypatch):
    monkeypatch.setenv(hc_verify_read.ENV, "0")
    _model, report = _installed_model()
    assert not report["installed"]
    assert report["probe_cases"] == 0


@needs_the_read
def test_a_biased_or_replaced_projection_keeps_the_stock_chain():
    # The kernels read the projection weights only; a bias (the parity test
    # plants one to put an exact value on the inject gate) must send the read
    # back to the stock chain, which applies it.
    import mlx.nn as nn

    model, _report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    assert hc_verify_read.serves(reader, mx.bfloat16, 4)
    biased = nn.Linear(FAMILY.hc_count * FAMILY.hidden_size, FAMILY.hc_count, bias=True)
    biased.weight = reader.block_inject_weight.weight
    biased.bias = mx.full((FAMILY.hc_count,), -27.375, dtype=mx.bfloat16)
    reader.block_inject_weight = biased
    assert not hc_verify_read.serves(reader, mx.bfloat16, 4)
    x, _block, _gates = _stream(FAMILY, 4, seed=17)
    with compiled_step_body():
        out = reader(x)
    mx.eval(out)
    assert hc_verify_read.engagement()["traces"] == 0


@needs_the_read
def test_quantized_mixers_keep_the_stock_chain():
    import mlx.nn as nn

    model, _report = _installed_model()
    reader = model.layers[0].attn_hyper_connection
    reader.input_mix_weight_down = nn.QuantizedLinear.from_linear(
        reader.input_mix_weight_down, group_size=64, bits=8
    )
    assert not hc_verify_read.serves(reader, mx.bfloat16, 4)


@pytest.mark.parametrize(
    "architecture, served",
    [
        ("applegpu_g13g", False),  # M1: tiled GEMM parent
        ("applegpu_g14s", False),  # M2 Max: tiled GEMM parent
        ("applegpu_g15s", True),  # M3 Max
        ("applegpu_g16s", True),  # M4 Max, and the M1-M4 rehearsal string
        ("applegpu_g17s", True),  # M5 Max
        ("applegpu_g17g", True),  # M5, M5 Pro
        ("applegpu_g17d", True),  # M5 Ultra: measured
        ("applegpu_g15d", False),  # M3 Ultra: not measured
        ("applegpu_g18p", False),  # phone class
        (None, False),
        ("mystery", False),
    ],
)
def test_the_device_route_follows_the_library_wide_gemv(architecture, served):
    assert hc_verify_read.device_route(architecture)[0] is served


@pytest.mark.parametrize("architecture", ["applegpu_g15d", "applegpu_g18p", "applegpu_g14s", "mystery", None])
def test_a_refused_route_leaves_the_suffix_to_install(architecture):
    # install() prints every reason with "; the stock chain serves" appended.
    ok, why = hc_verify_read.device_route(architecture)
    assert not ok and why and "stock chain serves" not in why


@pytest.mark.parametrize(
    "rows, outputs, lanes",
    [(2, 320, 32), (5, 320, 32), (6, 320, 16), (8, 10240, 16), (8, 4, 32), (7, 64, 32)],
)
def test_lanes_per_row_follow_the_library_pass_plan(rows, outputs, lanes):
    assert hc_verify_read._k_lanes(rows, outputs) == lanes


def test_the_rehearsal_architecture_is_what_mlx_reports_when_set():
    # Under MLX_METAL_GPU_ARCH=applegpu_g16s the library takes its generation
    # 16 kernels and the read must follow it (the M1-M4 rehearsal switch).
    from mtplx.nax_detect import gpu_architecture

    forced = os.environ.get("MLX_METAL_GPU_ARCH")
    if not forced:
        pytest.skip("run under the rehearsal switch to exercise this")
    assert gpu_architecture() == forced
    assert hc_verify_read.device_route(forced)[0]
