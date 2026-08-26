import inspect
import math

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx import deepseek_v4_nvfp4_kv as nvfp4_kv  # noqa: E402
from mtplx.kernels import deepseek_v4_qkv_prologue as prologue  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _capture_kernel(builder, *args, **kwargs) -> dict:
    captured = {}
    original = mx.fast.metal_kernel

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    builder.cache_clear()
    mx.fast.metal_kernel = capture
    try:
        builder(*args, **kwargs)
    finally:
        builder.cache_clear()
        mx.fast.metal_kernel = original
    return captured


class _ShapedOutput:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


def _zero_output_kernel(**kwargs):
    return tuple(
        (
            mx.zeros(shape, dtype=dtype)
            if math.prod(shape) < 10_000_000
            else _ShapedOutput(shape, dtype)
        )
        for shape, dtype in zip(
            kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
        )
    )


def _bf16_roundtrip(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _e4m3_positive(value: np.float32) -> int:
    value = np.float32(max(float(value), 0.0))
    if not value > 0.0:
        return 0
    value = np.float32(min(float(value), 448.0))
    if value < np.float32(0.015625):
        return min(int(np.rint(value / np.float32(0.001953125))), 8)
    exponent = int(np.floor(np.log2(value)))
    step = np.float32(2.0 ** (exponent - 3))
    significand = int(np.rint(value / step))
    if significand >= 16:
        exponent += 1
        significand = 8
    stored_exponent = min(exponent + 7, 15)
    if stored_exponent == 15:
        significand = min(significand, 14)
    return (stored_exponent << 3) | (significand - 8)


def _e4m3_decode(raw: int) -> np.float32:
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    if exponent == 0:
        return np.float32(mantissa * 2.0**-9)
    return np.float32((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))


def _e2m1(value: np.float32) -> int:
    magnitude = abs(float(value))
    if magnitude <= 0.25:
        code = 0
    elif magnitude < 0.75:
        code = 1
    elif magnitude <= 1.25:
        code = 2
    elif magnitude < 1.75:
        code = 3
    elif magnitude <= 2.5:
        code = 4
    elif magnitude < 3.5:
        code = 5
    elif magnitude <= 5.0:
        code = 6
    else:
        code = 7
    sign = int(np.asarray(value, dtype=np.float32).view(np.uint32) >> 31)
    return code | (sign << 3)


def _pack_stock432(latent: np.ndarray, rope: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    rope = np.asarray(rope, dtype=np.float32)
    records = np.zeros((latent.shape[0], 432), dtype=np.uint8)
    for row_index, row in enumerate(latent):
        for group in range(32):
            group_values = row[group * 16 : (group + 1) * 16]
            scale_byte = _e4m3_positive(np.max(np.abs(group_values)) / 6.0)
            scale = _e4m3_decode(scale_byte)
            inverse = np.float32(0.0 if scale == 0.0 else 1.0 / scale)
            for packed in range(8):
                low = _e2m1(group_values[packed * 2] * inverse)
                high = _e2m1(group_values[packed * 2 + 1] * inverse)
                records[row_index, group * 8 + packed] = low | (high << 4)
            records[row_index, 256 + group] = scale_byte
        rope_bf16 = (
            _bf16_roundtrip(rope[row_index]).view(np.uint32) >> np.uint32(16)
        ).astype(np.uint16)
        records[row_index, 304:] = rope_bf16.view(np.uint8)
    return records


def test_prologue_sources_pin_source_order_and_stock432_bytes() -> None:
    learned = _capture_kernel(prologue._learned_qkv_norm_kernel)
    kv_learned = _capture_kernel(prologue._learned_kv_norm_kernel)
    qkv = _capture_kernel(prologue._qkv_record_kernel, prefill=False)
    prefill = _capture_kernel(prologue._qkv_record_kernel, prefill=True)
    context = _capture_kernel(prologue._kv_record_kernel)

    assert learned["input_names"] == [
        "projection",
        "q_weight",
        "kv_weight",
        "rows",
        "rms_eps",
    ]
    assert learned["output_names"] == ["q_rank_norm", "kv_norm"]
    assert learned["ensure_row_contiguous"] is False
    assert "projection + size_t(row) * 1536u" in learned["source"]
    assert "float normalized = float(input[dim]) * rrms" in learned["source"]
    assert "* float(weight[dim]);" in learned["source"]
    assert "output[dim] = T(normalized);" in learned["source"]
    assert kv_learned["input_names"] == [
        "projection",
        "kv_weight",
        "rows",
        "rms_eps",
    ]
    assert kv_learned["output_names"] == ["kv_norm"]
    assert kv_learned["ensure_row_contiguous"] is False
    assert "projection + size_t(row) * 1536u + 1024u" in kv_learned["source"]

    source = qkv["source"]
    assert qkv["output_names"] == ["q_out", "records"]
    assert qkv["ensure_row_contiguous"] is False
    assert prefill["ensure_row_contiguous"] is False
    assert "float rms_rcp = rsqrt(simd_sum(local_sq) / 512.0f + float(rms_eps));" in source
    assert "float normalized = elements[i] * rms_rcp;" in source
    assert "float rotated = normalized;" in source
    assert "q_out[q_offset + dim] = T(rotated);" in source
    assert "mtplx_bf16_roundtrip(normalized)" not in source
    for record_source in (source, prefill["source"]):
        pair_loop = record_source.index("for (uint i = 0u; i < 16u; i += 2u)")
        even_read = record_source.index("float even = elements[i];", pair_loop)
        odd_read = record_source.index("float odd = elements[i + 1u];", pair_loop)
        even_store = record_source.index(
            "elements[i] = mtplx_bf16_roundtrip(even * c - odd * s);",
            pair_loop,
        )
        odd_store = record_source.index(
            "elements[i + 1u] = mtplx_bf16_roundtrip(even * s + odd * c);",
            pair_loop,
        )
        rope_copy = record_source.index("rope_elements[i] = elements[i];")
        quantization = record_source.index("float group_max = 0.0f;")
        assert pair_loop < even_read < odd_read < even_store < odd_store
        assert odd_store < rope_copy < quantization
    assert "record[256u + lane] = scale_byte;" in source
    assert "record[304u + rope_byte]" in source
    assert "record[288u + lane] = uchar(0);" in source
    assert "threadgroup_position_in_grid.x * 8u" in source
    assert "simdgroup_index_in_threadgroup" in source
    assert "for (uint slot = simdgroup_index_in_threadgroup;" in prefill["source"]
    assert "slot < 65u; slot += 8u" in prefill["source"]
    assert context["output_names"] == ["records"]
    assert context["ensure_row_contiguous"] is False
    assert context["input_names"] == ["kv_norm", "rope_cos", "rope_sin", "rows"]
    assert "q_out" not in context["source"]
    context_pair_loop = context["source"].index(
        "for (uint i = 0u; i < 16u; i += 2u)"
    )
    context_even_read = context["source"].index(
        "float even = elements[i];", context_pair_loop
    )
    context_odd_read = context["source"].index(
        "float odd = elements[i + 1u];", context_pair_loop
    )
    context_even_store = context["source"].index(
        "elements[i] = mtplx_bf16_roundtrip(even * c - odd * s);",
        context_pair_loop,
    )
    context_odd_store = context["source"].index(
        "elements[i + 1u] = mtplx_bf16_roundtrip(even * s + odd * c);",
        context_pair_loop,
    )
    context_rope_copy = context["source"].index("rope_elements[i] = elements[i];")
    context_quantization = context["source"].index("float group_max = 0.0f;")
    assert (
        context_pair_loop
        < context_even_read
        < context_odd_read
        < context_even_store
        < context_odd_store
        < context_rope_copy
        < context_quantization
    )
    assert "threadgroup_position_in_grid.x * 8u" in context["source"]

    dimensions = np.arange(512, dtype=np.float32)
    kv = _bf16_roundtrip(
        np.sin(dimensions * np.float32(0.173)) * np.float32(1.7)
        + np.linspace(-0.4, 0.6, 512, dtype=np.float32)
    )
    angles = (np.arange(32, dtype=np.float32) + np.float32(0.37)) * np.float32(
        0.217
    )
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    rotated = kv.copy()
    tail = kv[-64:].reshape(32, 2)
    rotated[-64::2] = tail[:, 0] * cosine - tail[:, 1] * sine
    rotated[-63::2] = tail[:, 0] * sine + tail[:, 1] * cosine
    source_boundary = _bf16_roundtrip(rotated)
    expected = _pack_stock432(
        source_boundary[None], source_boundary[-64:][None]
    )[0]
    unrotated_value = _pack_stock432(kv[None], source_boundary[-64:][None])[0]

    q = _bf16_roundtrip(
        np.cos(dimensions * np.float32(0.113)) * np.float32(2.1)
        + np.linspace(-0.7, 0.2, 512, dtype=np.float32)
    )
    rms_rcp = np.float32(
        1.0 / np.sqrt(np.mean(q * q, dtype=np.float32) + np.float32(1.0e-6))
    )
    q_normalized = q * rms_rcp
    q_exact = q_normalized.copy()
    q_tail = q_normalized[-64:].reshape(32, 2)
    q_exact[-64::2] = q_tail[:, 0] * cosine - q_tail[:, 1] * sine
    q_exact[-63::2] = q_tail[:, 0] * sine + q_tail[:, 1] * cosine
    q_exact = _bf16_roundtrip(q_exact)
    q_early = _bf16_roundtrip(q_normalized)
    q_early_tail = q_early[-64:].reshape(32, 2)
    q_early[-64::2] = q_early_tail[:, 0] * cosine - q_early_tail[:, 1] * sine
    q_early[-63::2] = q_early_tail[:, 0] * sine + q_early_tail[:, 1] * cosine
    q_early = _bf16_roundtrip(q_early)

    assert expected.shape == (432,)
    np.testing.assert_array_equal(expected[288:304], np.zeros(16, dtype=np.uint8))
    np.testing.assert_array_equal(
        expected[304:],
        (source_boundary[-64:].view(np.uint32) >> np.uint32(16))
        .astype(np.uint16)
        .view(np.uint8),
    )
    np.testing.assert_array_equal(expected[:224], unrotated_value[:224])
    np.testing.assert_array_equal(expected[256:284], unrotated_value[256:284])
    assert not np.array_equal(expected[224:256], unrotated_value[224:256])
    assert not np.array_equal(q_exact[-64:], q_early[-64:])


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is unavailable")
def test_real_kv_record_kernels_match_pairwise_rope_stock432_bytes() -> None:
    mx.set_default_device(mx.gpu)
    rows = 2
    dimensions = np.arange(rows * 512, dtype=np.float32).reshape(rows, 512)
    kv = _bf16_roundtrip(
        np.sin(dimensions * np.float32(0.173)) * np.float32(1.7)
        + np.linspace(-0.4, 0.6, 512, dtype=np.float32)[None, :]
    )
    pair = np.arange(rows * 32, dtype=np.float32).reshape(rows, 32)
    angles = (pair + np.float32(0.37)) * np.float32(0.217)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    rotated = kv.copy()
    tail = kv[:, -64:].reshape(rows, 32, 2)
    rotated[:, -64::2] = tail[:, :, 0] * cosine - tail[:, :, 1] * sine
    rotated[:, -63::2] = tail[:, :, 0] * sine + tail[:, :, 1] * cosine
    source_boundary = _bf16_roundtrip(rotated)
    expected = _pack_stock432(source_boundary, source_boundary[:, -64:])

    kv_input = mx.array(kv.reshape(1, rows, 512)).astype(mx.bfloat16)
    rope_cos = mx.array(cosine.reshape(1, rows, 32))
    rope_sin = mx.array(sine.reshape(1, rows, 32))
    context_records = prologue._run_context_kv_records(
        kv_input,
        rope_cos,
        rope_sin,
        kernel=prologue._kv_record_kernel(),
    )
    _q_out, target_records = prologue._run_target_qkv_records(
        mx.zeros((1, rows, 64, 512), dtype=mx.bfloat16),
        kv_input,
        rope_cos,
        rope_sin,
        decode_kernel=prologue._qkv_record_kernel(prefill=False),
        rms_eps=1.0e-6,
    )
    mx.eval(context_records, target_records)

    np.testing.assert_array_equal(np.array(context_records)[0], expected)
    np.testing.assert_array_equal(np.array(target_records)[0], expected)


def test_installed_prologue_prebinds_norm_and_phase_finalizer_kernels(
    monkeypatch,
) -> None:
    launches = {
        "norm": [],
        "kv_norm": [],
        "qkv_decode": [],
        "qkv_prefill": [],
        "context": [],
    }

    def kernel_for(label):
        def kernel(**kwargs):
            launches[label].append(kwargs)
            return _zero_output_kernel(**kwargs)

        return kernel

    norm_kernel = kernel_for("norm")
    kv_norm_kernel = kernel_for("kv_norm")
    qkv_decode_kernel = kernel_for("qkv_decode")
    qkv_prefill_kernel = kernel_for("qkv_prefill")
    context_kernel = kernel_for("context")
    monkeypatch.setattr(prologue, "_learned_qkv_norm_kernel", lambda: norm_kernel)
    monkeypatch.setattr(prologue, "_learned_kv_norm_kernel", lambda: kv_norm_kernel)
    monkeypatch.setattr(
        prologue,
        "_qkv_record_kernel",
        lambda *, prefill: qkv_prefill_kernel if prefill else qkv_decode_kernel,
    )
    monkeypatch.setattr(prologue, "_kv_record_kernel", lambda: context_kernel)
    monkeypatch.setattr(prologue.mx.metal, "is_available", lambda: True)

    installed = prologue.install_mia_qkv_prologue(
        q_rank=1024,
        heads=64,
        head_dim=512,
        rope_dim=64,
        proposal_rows=5,
        context_rows=128,
        prefill_tile_rows=1024,
    )

    def fail_factory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("installed prologue re-entered a kernel factory")

    monkeypatch.setattr(prologue, "_learned_qkv_norm_kernel", fail_factory)
    monkeypatch.setattr(prologue, "_learned_kv_norm_kernel", fail_factory)
    monkeypatch.setattr(prologue, "_qkv_record_kernel", fail_factory)
    monkeypatch.setattr(prologue, "_kv_record_kernel", fail_factory)

    fused_projection = mx.zeros((1, 6, 1536), dtype=mx.bfloat16)
    with monkeypatch.context() as no_copy:
        no_copy.setattr(
            prologue.mx,
            "contiguous",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("fused projection path materialized a split copy")
            ),
        )
        q_rank, kv = installed.learned_norm(
            fused_projection,
            mx.ones((1024,), dtype=mx.bfloat16),
            mx.ones((512,), dtype=mx.bfloat16),
            rms_eps=1.0e-6,
        )
        kv_only = installed.kv_norm(
            fused_projection,
            mx.ones((512,), dtype=mx.bfloat16),
            rms_eps=1.0e-6,
        )
    assert launches["norm"][0]["inputs"][0] is fused_projection
    assert launches["kv_norm"][0]["inputs"][0] is fused_projection
    q_out, target_records = installed.target_records(
        mx.zeros((1, 6, 64, 512), dtype=mx.bfloat16),
        kv,
        mx.ones((1, 6, 32), dtype=mx.float32),
        mx.zeros((1, 6, 32), dtype=mx.float32),
        rms_eps=1.0e-6,
    )
    proposal_q, proposal_records = installed.proposal_records(
        mx.zeros((1, 5, 64, 512), dtype=mx.bfloat16),
        mx.zeros((1, 5, 512), dtype=mx.bfloat16),
        mx.ones((1, 5, 32), dtype=mx.float32),
        mx.zeros((1, 5, 32), dtype=mx.float32),
        rms_eps=1.0e-6,
    )
    context_records = installed.context_records(
        mx.zeros((1, 7, 512), dtype=mx.bfloat16),
        mx.ones((1, 7, 32), dtype=mx.float32),
        mx.zeros((1, 7, 32), dtype=mx.float32),
    )

    monkeypatch.setattr(prologue.mx, "contiguous", lambda value: value)
    installed.target_records(
        _ShapedOutput((1, 1, 64, 512), mx.bfloat16),
        _ShapedOutput((1, 1, 512), mx.bfloat16),
        _ShapedOutput((1, 1, 32), mx.float32),
        _ShapedOutput((1, 1, 32), mx.float32),
        rms_eps=1.0e-6,
    )
    prefill_q_rank, prefill_kv = installed.learned_norm(
        _ShapedOutput((1, 1024, 1536), mx.bfloat16),
        _ShapedOutput((1024,), mx.bfloat16),
        _ShapedOutput((512,), mx.bfloat16),
        rms_eps=1.0e-6,
    )
    partial_prefill_q, partial_prefill_records = installed.prefill_records(
        _ShapedOutput((1, 7, 64, 512), mx.bfloat16),
        _ShapedOutput((1, 7, 512), mx.bfloat16),
        _ShapedOutput((1, 7, 32), mx.float32),
        _ShapedOutput((1, 7, 32), mx.float32),
        rms_eps=1.0e-6,
    )
    prefill_q, prefill_records = installed.prefill_records(
        _ShapedOutput((1, 1024, 64, 512), mx.bfloat16),
        prefill_kv,
        _ShapedOutput((1, 1024, 32), mx.float32),
        _ShapedOutput((1, 1024, 32), mx.float32),
        rms_eps=1.0e-6,
    )

    assert tuple(q_rank.shape) == (1, 6, 1024)
    assert tuple(kv.shape) == (1, 6, 512)
    assert tuple(kv_only.shape) == (1, 6, 512)
    assert tuple(q_out.shape) == (1, 6, 64, 512)
    assert tuple(target_records.shape) == (1, 6, 432)
    assert tuple(proposal_q.shape) == (1, 5, 64, 512)
    assert tuple(proposal_records.shape) == (1, 5, 432)
    assert tuple(context_records.shape) == (1, 7, 432)
    assert tuple(prefill_q_rank.shape) == (1, 1024, 1024)
    assert tuple(prefill_q.shape) == (1, 1024, 64, 512)
    assert tuple(prefill_records.shape) == (1, 1024, 432)
    assert tuple(partial_prefill_q.shape) == (1, 7, 64, 512)
    assert tuple(partial_prefill_records.shape) == (1, 7, 432)
    assert len(launches["norm"]) == 2
    assert len(launches["kv_norm"]) == 1
    assert len(launches["qkv_decode"]) == 4
    assert len(launches["qkv_prefill"]) == 1
    assert len(launches["context"]) == 1
    assert launches["norm"][0]["grid"] == (6 * 2 * 256, 1, 1)
    assert launches["norm"][1]["grid"] == (1024 * 2 * 256, 1, 1)
    assert launches["kv_norm"][0]["grid"] == (6 * 256, 1, 1)
    assert launches["qkv_decode"][0]["grid"] == (
        math.ceil(6 * 65 / 8) * 256,
        1,
        1,
    )
    assert launches["qkv_decode"][1]["grid"] == (
        math.ceil(5 * 65 / 8) * 256,
        1,
        1,
    )
    assert launches["qkv_decode"][2]["grid"] == (
        math.ceil(65 / 8) * 256,
        1,
        1,
    )
    assert launches["qkv_decode"][3]["grid"] == (
        math.ceil(7 * 65 / 8) * 256,
        1,
        1,
    )
    assert launches["qkv_prefill"][0]["grid"] == (1024 * 256, 1, 1)
    assert launches["context"][0]["grid"] == (math.ceil(7 / 8) * 256, 1, 1)
    assert launches["context"][0]["inputs"][-1] == 7
    for label in ("qkv_decode", "qkv_prefill", "context"):
        assert launches[label][0]["threadgroup"] == (256, 1, 1)
    assert installed.geometry == {
        "q_rank": 1024,
        "heads": 64,
        "head_dim": 512,
        "rope_dim": 64,
        "target_decode_rows": (1, 6),
        "proposal_rows": 5,
        "context_rows": 128,
        "prefill_tile_rows": 1024,
    }
    for run in (
        prologue._run_learned_qkv_norm,
        prologue._run_learned_kv_norm,
        prologue._run_target_qkv_records,
        prologue._run_prefill_qkv_records,
        prologue._run_k5_proposal_records,
        prologue._run_context_kv_records,
    ):
        source = inspect.getsource(run)
        assert "_kernel()" not in source
        assert "kernel" in source


def test_bound_prologue_owns_projection_weights_eps_and_hot_callables(
    monkeypatch,
) -> None:
    events = []

    class Owner:
        split = 1024

        def project_fused(self, hidden):
            events.append(("project", hidden))
            return "projection"

    raw = prologue.MiaQKVPrologue(
        learned_norm=lambda projection, q_weight, kv_weight, *, rms_eps: (
            events.append(
                ("learned", projection, q_weight, kv_weight, rms_eps)
            )
            or ("qrank", "kv")
        ),
        kv_norm=lambda projection, kv_weight, *, rms_eps: (
            events.append(("kv_norm", projection, kv_weight, rms_eps)) or "kv"
        ),
        target_records=lambda *args, rms_eps: (
            events.append(("target", *args, rms_eps)) or ("q", "records")
        ),
        prefill_records=lambda *args, rms_eps: (
            events.append(("prefill", *args, rms_eps)) or ("q", "records")
        ),
        proposal_records=lambda *args, rms_eps: (
            events.append(("proposal", *args, rms_eps)) or ("q", "records")
        ),
        context_records=lambda *args: (
            events.append(("context", *args)) or "records"
        ),
        q_rank=1024,
        heads=64,
        head_dim=512,
        rope_dim=64,
        proposal_rows=5,
        context_rows=128,
        prefill_tile_rows=1024,
    )
    q_weight = mx.ones((1024,), dtype=mx.bfloat16)
    kv_weight = mx.ones((512,), dtype=mx.bfloat16)
    owner = Owner()
    bound = prologue.bind_mia_qkv_prologue(
        raw,
        projection_owner=owner,
        q_weight=q_weight,
        kv_weight=kv_weight,
        rms_eps=1.0e-6,
    )

    assert bound.project_learned("hidden") == ("qrank", "kv")
    assert bound.project_kv("context") == "kv"
    assert bound.target_records(1, 2, 3, 4) == ("q", "records")
    assert bound.prefill_records(1, 2, 3, 4) == ("q", "records")
    assert bound.proposal_records(1, 2, 3, 4) == ("q", "records")
    assert bound.context_records(1, 2, 3) == "records"
    assert bound.projection_owner is owner
    assert bound.q_weight is q_weight
    assert bound.kv_weight is kv_weight
    assert events == [
        ("project", "hidden"),
        ("learned", "projection", q_weight, kv_weight, 1.0e-6),
        ("project", "context"),
        ("kv_norm", "projection", kv_weight, 1.0e-6),
        ("target", 1, 2, 3, 4, 1.0e-6),
        ("prefill", 1, 2, 3, 4, 1.0e-6),
        ("proposal", 1, 2, 3, 4, 1.0e-6),
        ("context", 1, 2, 3),
    ]


def test_record_writers_scatter_once_and_proposals_never_mutate_persistent_ring(
    monkeypatch,
) -> None:
    window = nvfp4_kv.FixedMiaNVFP4Window(capacity_rows=8, block_size=4)
    window._pages[:] = mx.full(window._pages.shape, 0xA5, dtype=mx.uint8)
    target_records = mx.full((1, 3, 432), 0x17, dtype=mx.uint8)
    target_writer = nvfp4_kv.install_fixed_window_record_writer(window)
    with monkeypatch.context() as no_hot_mapping:
        no_hot_mapping.setattr(
            nvfp4_kv.mx,
            "arange",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("record writer rebuilt its slot mapping")
            ),
        )
        target_writer(target_records, absolute_start=6)
        visible = window.slice(6, 9)
        mx.eval(visible)
    mx.eval(window._pages)

    physical = np.array(window._pages)
    changed = {0, 6, 7}
    for position in range(8):
        block, offset = divmod(position, 4)
        expected = 0x17 if position in changed else 0xA5
        assert np.all(physical[block, offset] == expected)
    assert window.end == 9

    restored_pages = mx.full(window._pages.shape, 0xA5, dtype=mx.uint8)
    window.replace_state(
        (
            restored_pages,
            mx.array([1, 0], dtype=mx.int32),
            0,
            0,
        )
    )
    target_writer(
        mx.full((1, 1, 432), 0x51, dtype=mx.uint8),
        absolute_start=0,
    )
    mx.eval(window._pages)
    restored_physical = np.array(window._pages)
    assert np.all(restored_physical[1, 0] == 0x51)
    assert np.all(restored_physical[0, 0] == 0xA5)

    append_ring = nvfp4_kv.FixedMiaNVFP4Ring(capacity_rows=128)
    append_calls = []
    original_write_tail = append_ring._pool._write_installed_tail

    def counted_write_tail(*args, **kwargs):
        append_calls.append((args, kwargs))
        return original_write_tail(*args, **kwargs)

    monkeypatch.setattr(
        append_ring._pool,
        "_write_installed_tail",
        counted_write_tail,
    )
    append_ring._append_installed_records(
        mx.full((1, 7, 432), 0x28, dtype=mx.uint8),
        prefix=(1,),
    )
    assert len(append_ring) == 7
    assert [call[1]["count"] for call in append_calls] == [7]

    ring = nvfp4_kv.FixedMiaNVFP4Ring(capacity_rows=128)
    ring._pages[:] = mx.full(ring._pages.shape, 0xA5, dtype=mx.uint8)
    before = np.array(ring.records)
    proposal_records = mx.full((1, 5, 432), 0x33, dtype=mx.uint8)
    assert proposal_records.shape == (1, 5, 432)
    assert len(ring) == 0
    np.testing.assert_array_equal(np.array(ring.records), before)

    context_writer = nvfp4_kv.install_fixed_ring_context_writer(ring)
    with monkeypatch.context() as no_hot_mapping:
        no_hot_mapping.setattr(
            nvfp4_kv.mx,
            "arange",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("context writer rebuilt its slot mapping")
            ),
        )
        context_writer(
            mx.full((1, 7, 432), 0x29, dtype=mx.uint8),
            absolute_start=0,
        )
    mx.eval(ring.records)

    assert len(ring) == 128
    physical = np.array(ring.records)[0]
    assert np.all(physical[:7] == 0x29)
    assert np.all(physical[7:] == 0)

    wrapped_ring = nvfp4_kv.FixedMiaNVFP4Ring(capacity_rows=128)
    wrapped_writer = nvfp4_kv.install_fixed_ring_context_writer(wrapped_ring)
    row_ids = mx.broadcast_to(
        mx.arange(128, dtype=mx.uint8)[:, None],
        (128, 432),
    )[None]
    wrapped_writer(row_ids, absolute_start=5)
    mx.eval(wrapped_ring.records)
    wrapped = np.array(wrapped_ring.records)[0, :, 0]
    np.testing.assert_array_equal(wrapped, (np.arange(128) - 5) % 128)
    assert len(wrapped_ring) == 128

    commit_writer = nvfp4_kv.install_fixed_ring_commit_writer(ring)
    with monkeypatch.context() as no_hot_mapping:
        no_hot_mapping.setattr(
            nvfp4_kv.mx,
            "arange",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("commit writer rebuilt its slot mapping")
            ),
        )
        commit_writer(
            mx.full((1, 3, 432), 0x44, dtype=mx.uint8),
            absolute_start=127,
        )
    mx.eval(ring.records)

    assert len(ring) == 128
    physical = np.array(ring.records)[0]
    assert np.all(physical[[127, 0, 1]] == 0x44)
    assert np.all(physical[2:7] == 0x29)
    assert np.all(physical[7:127] == 0)
    decoded_key, decoded_value = ring.decode(127, 128)
    assert tuple(decoded_key.shape) == (1, 1, 512)
    assert tuple(decoded_value.shape) == (1, 1, 512)
