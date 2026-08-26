import hashlib
import inspect
import struct
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mtplx import deepseek_v4_nvfp4_kv as nvfp4_kv  # noqa: E402
from mtplx.attention_context import attention_phase  # noqa: E402
from mtplx.kernels import deepseek_v4_compressor as compressor_kernels  # noqa: E402
from mtplx.kernels import deepseek_v4_qkv_prologue as qkv_prologue  # noqa: E402
from mtplx.deepseek_v4_mia_engine import MiaM6RatioTables  # noqa: E402
from mtplx.models import deepseek_v4 as target_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    Compressor,
    CompressorState,
    FixedMiaCompressorState,
    MiaRoPETableProvider,
    ModelArgs,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _bf16_roundtrip(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


@pytest.mark.parametrize("input_dtype", [mx.bfloat16, mx.float32])
def test_mia_stacked_mxfp8_projection_preserves_named_rows_and_one_gemm(
    monkeypatch,
    input_dtype,
) -> None:
    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            first = nn.Linear(64, 48, bias=False)
            second = nn.Linear(64, 16, bias=False)
            first.weight = (
                mx.arange(48 * 64, dtype=mx.float32).reshape(48, 64) - 311.0
            ) / 257.0
            second.weight = (
                mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64) - 97.0
            ) / 193.0
            self.first = nn.QuantizedLinear.from_linear(
                first,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
            self.second = nn.QuantizedLinear.from_linear(
                second,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )

    pair = Pair()
    inputs = (
        (mx.arange(2 * 64, dtype=mx.float32) - 43.0) / 71.0
    ).reshape(1, 2, 64).astype(input_dtype)
    expected = (pair.first(inputs), pair.second(inputs))
    mx.eval(*expected)
    names_before = tuple(name for name, _ in tree_flatten(pair.parameters()))

    owner = target_module.MiaStackedMXFP8Projection(
        pair.first,
        pair.second,
    )
    names_after = tuple(name for name, _ in tree_flatten(pair.parameters()))
    original_quantized_matmul = target_module.mx.quantized_matmul
    calls = []

    def counted_quantized_matmul(*args, **kwargs):
        calls.append((args, kwargs))
        return original_quantized_matmul(*args, **kwargs)

    monkeypatch.setattr(
        target_module.mx,
        "quantized_matmul",
        counted_quantized_matmul,
    )
    monkeypatch.setattr(
        target_module.mx,
        "contiguous",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stacked output split materialized a copy")
        ),
    )
    fused = owner.project_fused(inputs)
    actual = owner.split_fused(fused)
    mx.eval(*actual)

    assert names_after == names_before == (
        "first.weight",
        "first.scales",
        "second.weight",
        "second.scales",
    )
    assert len(calls) == 1
    assert calls[0][1]["mode"] == "mxfp8"
    assert owner.split == 48
    np.testing.assert_array_equal(
        np.array(actual[0].astype(mx.float32)),
        np.array(expected[0].astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(actual[1].astype(mx.float32)),
        np.array(expected[1].astype(mx.float32)),
    )


def test_mia_stacked_dense_projection_preserves_fp32_compressor_arithmetic(
    monkeypatch,
) -> None:
    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(8, 6, bias=False)
            self.second = nn.Linear(8, 4, bias=False)
            self.first.weight = (
                (mx.arange(48, dtype=mx.float32) - 19.0) / 23.0
            ).reshape(6, 8).astype(mx.bfloat16)
            self.second.weight = (
                (mx.arange(32, dtype=mx.float32) - 7.0) / 17.0
            ).reshape(4, 8).astype(mx.bfloat16)

    pair = Pair()
    values = ((mx.arange(24, dtype=mx.float32) - 5.0) / 13.0).reshape(1, 3, 8)
    control_weight = mx.contiguous(
        mx.concatenate((pair.first.weight, pair.second.weight), axis=0)
    )
    control_fused = mx.matmul(values, mx.swapaxes(control_weight, -1, -2))
    expected = (control_fused[..., :6], control_fused[..., 6:])
    mx.eval(*expected)
    names_before = tuple(name for name, _ in tree_flatten(pair.parameters()))
    owner = target_module.MiaStackedDenseProjection(pair.first, pair.second)
    assert owner.weight.dtype == mx.float32
    assert pair.first.weight.dtype == mx.float32
    assert pair.second.weight.dtype == mx.float32
    original_matmul = target_module.mx.matmul
    calls = []

    def counted_matmul(*args, **kwargs):
        calls.append((args, kwargs))
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(target_module.mx, "matmul", counted_matmul)
    monkeypatch.setattr(
        target_module.mx,
        "contiguous",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stacked output split materialized a copy")
        ),
    )
    fused = owner.project_fused(values)
    actual = owner.split_fused(fused)
    mx.eval(*actual)

    assert tuple(name for name, _ in tree_flatten(pair.parameters())) == names_before
    assert names_before == ("first.weight", "second.weight")
    assert len(calls) == 1
    assert owner.split == 6
    np.testing.assert_array_equal(
        np.array(actual[0]),
        np.array(expected[0]),
    )
    np.testing.assert_array_equal(
        np.array(actual[1]),
        np.array(expected[1]),
    )


def test_exact_attention_and_compressor_routes_poison_old_projection_calls() -> None:
    projection_calls = []

    class StackedAttention:
        split = 1024

        def project_fused(self, hidden):
            projection_calls.append("attention")
            return mx.zeros(
                (*hidden.shape[:2], 1536),
                dtype=hidden.dtype,
            )

    raw_prologue = qkv_prologue.MiaQKVPrologue(
        learned_norm=lambda fused, **_kwargs: (
            fused[..., :1024],
            fused[..., 1024:],
        ),
        kv_norm=lambda fused, **_kwargs: fused[..., 1024:],
        target_records=lambda query, _latent, _cos, _sin, **_kwargs: (
            query,
            mx.zeros((*query.shape[:2], 432), dtype=mx.uint8),
        ),
        prefill_records=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("decode route called the prefill finalizer")
        ),
        proposal_records=lambda *_args, **_kwargs: None,
        context_records=lambda *_args, **_kwargs: None,
        q_rank=1024,
        heads=64,
        head_dim=512,
        rope_dim=64,
        proposal_rows=5,
        context_rows=128,
        prefill_tile_rows=1024,
    )
    bound_prologue = qkv_prologue.bind_mia_qkv_prologue(
        raw_prologue,
        projection_owner=StackedAttention(),
        q_weight=mx.ones((1024,), dtype=mx.bfloat16),
        kv_weight=mx.ones((512,), dtype=mx.bfloat16),
        rms_eps=1.0e-6,
    )

    def fail_old_projection(*_args, **_kwargs):
        raise AssertionError("exact route called an unpacked projection")

    attention = target_module.DeepseekV4Attention.__new__(
        target_module.DeepseekV4Attention
    )
    attention.n_heads = 64
    attention.head_dim = 512
    attention._mia_qkv_plan = None
    attention._mia_token_rope_tables = lambda start, count: (
        mx.arange(start, start + count, dtype=mx.int32),
        mx.ones((count, 32), dtype=mx.float32),
        mx.zeros((count, 32), dtype=mx.float32),
    )
    attention.wq_b = lambda values: mx.zeros(
        (*values.shape[:2], 64 * 512), dtype=values.dtype
    )
    attention.wq_a = fail_old_projection
    attention.wkv = fail_old_projection
    attention.install_mia_qkv_prologue(bound_prologue)
    with attention_phase("ar_decode"):
        query, query_rank, records, *_ = attention._mia_qkv_impl(
            mx.zeros((1, 2, 8), dtype=mx.bfloat16),
            SimpleNamespace(offset=7),
        )

    compressor = Compressor(
        ModelArgs(hidden_size=8, qk_rope_head_dim=64),
        compress_ratio=4,
        head_dim=128,
    )

    def stacked_compressor(values):
        projection_calls.append("compressor")
        shape = (*values.shape[:2], 256)
        return mx.zeros(shape, dtype=values.dtype), mx.zeros(
            shape, dtype=values.dtype
        )

    compressor._project_rows_impl = stacked_compressor
    compressor.wkv = fail_old_projection
    compressor.wgate = fail_old_projection
    kv, score, windows, _dtype = compressor._whole_projected_windows(
        mx.zeros((1, 4, 8), dtype=mx.bfloat16)
    )

    assert isinstance(attention._mia_qkv_plan, qkv_prologue.MiaBoundQKVPrologue)
    assert (
        attention._mia_qkv_impl.__func__
        is target_module.DeepseekV4Attention._mia_cached_qkv_records
    )
    assert projection_calls == ["attention", "compressor"]
    assert tuple(query.shape) == (1, 2, 64, 512)
    assert tuple(query_rank.shape) == (1, 2, 1024)
    assert tuple(records.shape) == (1, 2, 432)
    assert tuple(kv.shape) == (1, 1, 4, 256)
    assert tuple(score.shape) == (1, 1, 4, 256)
    assert windows == 1


def _e4m3_encode(value: float) -> int:
    sign = 0x80 if np.signbit(np.float32(value)) else 0
    magnitude = min(abs(float(value)), 448.0)
    if not magnitude > 0.0:
        code = 0
    elif magnitude < 0.015625:
        mantissa = int(np.rint(magnitude / 0.001953125))
        code = 0x08 if mantissa >= 8 else mantissa
    else:
        exponent = int(np.floor(np.log2(magnitude)))
        step = 2.0 ** (exponent - 3)
        significand = int(np.rint(magnitude / step))
        if significand >= 16:
            exponent += 1
            significand = 8
        stored_exponent = exponent + 7
        if stored_exponent >= 15:
            stored_exponent = 15
            significand = min(significand, 14)
        code = (stored_exponent << 3) | (significand - 8)
    return sign | code


def _e4m3_positive_decode(raw: int) -> float:
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    if exponent == 0:
        return mantissa * 0.001953125
    return (1.0 + mantissa * 0.125) * 2.0 ** (exponent - 7)


def _e2m1_magnitude_code(value: float) -> int:
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
    return code


def _raw_e2m1_encode(value: float) -> int:
    code = _e2m1_magnitude_code(value)
    return code | (0x8 if np.signbit(np.float32(value)) else 0)


def _compressor_e2m1_encode(value: float) -> int:
    code = _e2m1_magnitude_code(value)
    return code | (0x8 if np.float32(value) < np.float32(0.0) else 0)


def _pack_stock432(rows: np.ndarray) -> np.ndarray:
    records = np.zeros((rows.shape[0], 432), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        for group in range(32):
            values = row[group * 16 : (group + 1) * 16]
            scale_byte = _e4m3_encode(float(np.max(np.abs(values))) / 6.0)
            scale = _e4m3_positive_decode(scale_byte)
            inverse = 0.0 if scale == 0.0 else 1.0 / scale
            codes = [_compressor_e2m1_encode(value * inverse) for value in values]
            for pair in range(8):
                records[row_index, group * 8 + pair] = (
                    codes[2 * pair] | (codes[2 * pair + 1] << 4)
                )
            records[row_index, 256 + group] = scale_byte
        rope_bits = (row[-64:].view(np.uint32) >> 16).astype(np.uint16)
        records[row_index, 304:432] = rope_bits.view(np.uint8)
    return records


def _pack_mia132(rows: np.ndarray) -> np.ndarray:
    records = np.empty((rows.shape[0], 132), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        amax = max(float(np.max(np.abs(row))), 1.0e-4)
        scale = np.float32(2.0 ** np.ceil(np.log2(amax / 448.0)))
        records[row_index, :128] = np.array(
            [_e4m3_encode(value / scale) for value in row], dtype=np.uint8
        )
        records[row_index, 128:132] = np.frombuffer(
            struct.pack("<f", float(scale)), dtype=np.uint8
        )
    return records


def _source_rows(
    compressor: Compressor,
    kv_windows: mx.array,
    score_windows: mx.array,
    previous_kv: mx.array | None,
    previous_score: mx.array | None,
    first_window: int,
    has_previous: bool,
    output_rows: int,
) -> np.ndarray:
    kv = np.array(kv_windows.astype(mx.float32))
    score = np.array(score_windows.astype(mx.float32))
    previous_kv_np = (
        None if previous_kv is None else np.array(previous_kv.astype(mx.float32))
    )
    previous_score_np = (
        None
        if previous_score is None
        else np.array(previous_score.astype(mx.float32))
    )
    weight = np.array(compressor.norm.weight.astype(mx.float32))
    inv_freq = np.array(compressor._inv_freq.astype(mx.float32))
    rows = []
    for window in range(output_rows):
        if compressor.overlap:
            current_kv = kv[0, window, :, compressor.head_dim :]
            current_score = score[0, window, :, compressor.head_dim :]
            if window > 0:
                prior_kv = kv[0, window - 1, :, : compressor.head_dim]
                prior_score = score[0, window - 1, :, : compressor.head_dim]
            elif has_previous:
                assert previous_kv_np is not None and previous_score_np is not None
                prior_kv = previous_kv_np[0, :, : compressor.head_dim]
                prior_score = previous_score_np[0, :, : compressor.head_dim]
            else:
                prior_kv = np.zeros_like(current_kv)
                prior_score = np.full_like(current_score, -np.inf)
            current_kv = np.concatenate([prior_kv, current_kv], axis=0)
            current_score = np.concatenate([prior_score, current_score], axis=0)
        else:
            current_kv = kv[0, window]
            current_score = score[0, window]

        maximum = np.max(current_score, axis=0, keepdims=True)
        probability = np.exp(current_score - maximum)
        probability /= np.sum(probability, axis=0, keepdims=True)
        pooled = np.sum(current_kv * probability, axis=0, dtype=np.float32)
        rrms = np.float32(
            1.0
            / np.sqrt(
                np.mean(pooled * pooled, dtype=np.float32)
                + compressor.rms_norm_eps
            )
        )
        normed = pooled * rrms * weight

        position = (first_window + window) * compressor.compress_ratio
        angle = np.float32(position) * inv_freq
        cosine = np.cos(angle).astype(np.float32)
        sine = np.sin(angle).astype(np.float32)
        tail = normed[-64:].reshape(32, 2)
        rotated_tail = np.empty_like(tail)
        rotated_tail[:, 0] = tail[:, 0] * cosine - tail[:, 1] * sine
        rotated_tail[:, 1] = tail[:, 0] * sine + tail[:, 1] * cosine
        rotated = normed.copy()
        rotated[-64:] = rotated_tail.reshape(64)
        rows.append(_bf16_roundtrip(rotated))
    return np.asarray(rows, dtype=np.float32)


def _install_cpu_record_oracle(compressor: Compressor, mode: str) -> None:
    record_bytes = 432 if mode == "stock432" else 132

    def finalize(
        kv_windows,
        score_windows,
        previous_kv,
        previous_score,
        first_window,
        has_previous,
        output_rows,
    ):
        if output_rows == 0:
            return mx.zeros((1, 0, record_bytes), dtype=mx.uint8)
        rows = _source_rows(
            compressor,
            kv_windows,
            score_windows,
            previous_kv,
            previous_score,
            first_window,
            has_previous,
            output_rows,
        )
        packed = _pack_stock432(rows) if mode == "stock432" else _pack_mia132(rows)
        return mx.array(packed[None], dtype=mx.uint8)

    compressor._mia_record_impl = finalize


def _make_compressor(head_dim: int, mode: str, ratio: int) -> Compressor:
    args = ModelArgs(hidden_size=8, qk_rope_head_dim=64, rms_norm_eps=1.0e-6)
    compressor = Compressor(args, compress_ratio=ratio, head_dim=head_dim)
    rng = np.random.default_rng(1900 + head_dim + ratio)
    compressor.wkv.weight = mx.array(
        rng.normal(0.0, 0.18, compressor.wkv.weight.shape).astype(np.float32)
    )
    compressor.wgate.weight = mx.array(
        rng.normal(0.0, 0.13, compressor.wgate.weight.shape).astype(np.float32)
    )
    compressor.ape = mx.array(
        rng.normal(0.0, 0.07, compressor.ape.shape).astype(np.float32)
    )
    compressor.norm.weight = mx.array(
        rng.normal(1.0, 0.05, compressor.norm.weight.shape).astype(np.float32)
    )
    _install_cpu_record_oracle(compressor, mode)
    return compressor


def test_shared_mia_rope_provider_reuses_exact_compressor_tables_after_poison(
    monkeypatch,
) -> None:
    args = ModelArgs(hidden_size=8, qk_rope_head_dim=64, rms_norm_eps=1.0e-6)
    first = Compressor(args, compress_ratio=4, head_dim=512)
    second = Compressor(args, compress_ratio=4, head_dim=128)
    provider = MiaRoPETableProvider(
        first._inv_freq,
        max_positions=384_000,
    )
    first.install_mia_rope_provider(provider)
    second.install_mia_rope_provider(provider)
    provider.begin_forward()

    first_cos, first_sin = first._mia_rope_tables_for_windows(7, 3)
    mx.eval(first_cos, first_sin)
    expected_positions = (np.arange(3, dtype=np.float32) + 7.0) * 4.0
    inv_freq = np.array(first._inv_freq)
    np.testing.assert_allclose(
        np.array(first_cos),
        np.cos(expected_positions[:, None] * inv_freq[None]),
        rtol=1e-6,
        atol=1e-6,
    )

    def fail_trig(*_args, **_kwargs):
        raise AssertionError("shared compressor RoPE tables were rebuilt")

    monkeypatch.setattr(target_module.mx, "cos", fail_trig)
    monkeypatch.setattr(target_module.mx, "sin", fail_trig)
    second_cos, second_sin = second._mia_rope_tables_for_windows(7, 3)

    assert second_cos is first_cos
    assert second_sin is first_sin
    assert first._mia_rope_provider is provider
    assert second._mia_rope_provider is provider


def test_shared_mia_rope_provider_reuses_exact_target_tables_after_poison(
    monkeypatch,
) -> None:
    inv_freq = target_module._yarn_inv_freq(64, 10_000.0, 0, 1.0, 32, 1)
    provider = MiaRoPETableProvider(inv_freq, max_positions=384_000)
    provider.begin_forward()

    positions, first_cos, first_sin = provider.token_tables(17, 3)
    mx.eval(positions, first_cos, first_sin)
    np.testing.assert_array_equal(np.array(positions), np.arange(17, 20))
    expected_angles = np.arange(17, 20, dtype=np.float32)[:, None] * np.array(
        inv_freq
    )[None]
    np.testing.assert_allclose(
        np.array(first_cos),
        np.cos(expected_angles),
        rtol=1e-6,
        atol=1e-6,
    )

    def fail_trig(*_args, **_kwargs):
        raise AssertionError("shared target RoPE tables were rebuilt")

    monkeypatch.setattr(target_module.mx, "cos", fail_trig)
    monkeypatch.setattr(target_module.mx, "sin", fail_trig)
    second_positions, second_cos, second_sin = provider.token_tables(17, 3)

    assert second_positions is positions
    assert second_cos is first_cos
    assert second_sin is first_sin
    with pytest.raises(ValueError, match="384k"):
        provider.token_tables(383_999, 2)

    target_source = inspect.getsource(
        target_module.DeepseekV4Attention._mia_cached_qkv_records
    )
    assert "_mia_token_rope_tables" in target_source
    assert "self._rope_tables(" not in target_source
    for record_impl in (
        target_module.Compressor._nvfp4_record_impl,
        target_module.Compressor._indexer_record_impl,
    ):
        record_source = inspect.getsource(record_impl)
        assert "_mia_rope_tables_for_windows" in record_source
        assert "self._rope_tables_for_windows(" not in record_source


def _captured_kernel_definition(builder, *args) -> dict:
    captured = {}
    original = mx.fast.metal_kernel

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    builder.cache_clear()
    mx.fast.metal_kernel = capture
    try:
        builder(*args)
    finally:
        builder.cache_clear()
        mx.fast.metal_kernel = original
    return captured


def _captured_kernel_source(builder, *args) -> str:
    return _captured_kernel_definition(builder, *args)["source"]


def _zero_output_kernel(**kwargs):
    return tuple(
        mx.zeros(shape, dtype=dtype)
        for shape, dtype in zip(
            kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
        )
    )


def test_stock432_installer_prebinds_raw_pack_kernel(monkeypatch) -> None:
    launches = []

    def fake_kernel(**kwargs):
        launches.append(kwargs)
        return _zero_output_kernel(**kwargs)

    monkeypatch.setattr(nvfp4_kv, "_stock432_pack_kernel", lambda: fake_kernel)
    installed = nvfp4_kv.install_stock432_record_packer(
        head_dim=512,
        rope_dim=64,
    )

    def fail_factory():
        raise AssertionError("installed stock432 path re-entered the kernel factory")

    monkeypatch.setattr(nvfp4_kv, "_stock432_pack_kernel", fail_factory)
    records = installed(
        mx.zeros((1, 1, 512), dtype=mx.bfloat16),
        mx.zeros((1, 1, 64), dtype=mx.bfloat16),
    )

    assert installed.keywords["kernel"] is fake_kernel
    assert installed.func is nvfp4_kv._run_pack_stock432
    assert tuple(records.shape) == (1, 1, 432)
    assert len(launches) == 1
    installed_source = inspect.getsource(nvfp4_kv._run_pack_stock432)
    assert "_stock432_pack_kernel" not in installed_source
    assert "records = kernel(" in installed_source


@pytest.mark.parametrize(
    "mode,head_dim,ratio,factory_name",
    [
        ("nvfp4", 512, 4, "_nvfp4_finalize_kernel"),
        ("nvfp4", 512, 128, "_nvfp4_finalize_kernel"),
        ("indexer", 128, 4, "_indexer_finalize_kernel"),
    ],
)
def test_compressor_installer_prebinds_record_finalize_kernel(
    monkeypatch,
    mode: str,
    head_dim: int,
    ratio: int,
    factory_name: str,
) -> None:
    launches = []

    def fake_kernel(**kwargs):
        launches.append(kwargs)
        return _zero_output_kernel(**kwargs)

    if factory_name == "_nvfp4_finalize_kernel":
        monkeypatch.setattr(
            compressor_kernels,
            factory_name,
            lambda _ratio: fake_kernel,
        )
    else:
        monkeypatch.setattr(compressor_kernels, factory_name, lambda: fake_kernel)

    args = ModelArgs(hidden_size=8, qk_rope_head_dim=64, rms_norm_eps=1.0e-6)
    compressor = Compressor(args, compress_ratio=ratio, head_dim=head_dim)
    compressor.install_mia_rope_provider(
        MiaRoPETableProvider(
            compressor._inv_freq,
            max_positions=384_000,
        )
    )
    compressor.install_mia_record_packer(mode)

    def fail_factory(*_args):
        raise AssertionError("installed compressor path re-entered the kernel factory")

    monkeypatch.setattr(compressor_kernels, factory_name, fail_factory)
    width = (2 if ratio == 4 else 1) * head_dim
    records = compressor._mia_record_impl(
        mx.zeros((1, 1, ratio, width), dtype=mx.float32),
        mx.zeros((1, 1, ratio, width), dtype=mx.float32),
        None,
        None,
        0,
        False,
        1,
    )

    assert compressor._mia_record_impl.keywords["kernel"] is fake_kernel
    assert compressor._mia_record_impl.func.__name__ == (
        "_nvfp4_record_impl" if mode == "nvfp4" else "_indexer_record_impl"
    )
    assert tuple(records.shape) == (1, 1, 432 if mode == "nvfp4" else 132)
    assert len(launches) == 1
    launch = (
        compressor_kernels.fused_nvfp4_records
        if mode == "nvfp4"
        else compressor_kernels.fused_indexer_records
    )
    launch_source = inspect.getsource(launch)
    assert factory_name not in launch_source
    assert "records = kernel(" in launch_source


def _metal_helper_body(header: str, helper: str) -> str:
    start = header.index(f"inline uchar {helper}")
    return header[start : header.index("\n    }", start)]


def _uses_ieee_sign(body: str) -> bool:
    return (
        "as_type<uint>(value) >> 31" in body
        and "value < 0.0f" not in body
    )


def _uses_numeric_sign(body: str) -> bool:
    return (
        "value < 0.0f" in body
        and "as_type<uint>(value) >> 31" not in body
    )


def test_stock432_signed_zero_preserves_the_two_pinned_source_contracts() -> None:
    assert _raw_e2m1_encode(np.float32(0.0)) == 0
    assert _raw_e2m1_encode(np.float32(-0.0)) == 0x8
    assert _compressor_e2m1_encode(np.float32(0.0)) == 0
    assert _compressor_e2m1_encode(np.float32(-0.0)) == 0
    assert (
        _raw_e2m1_encode(np.float32(-0.0))
        | (_raw_e2m1_encode(np.float32(0.0)) << 4)
    ) == 0x08
    assert (
        _compressor_e2m1_encode(np.float32(-0.0))
        | (_compressor_e2m1_encode(np.float32(0.0)) << 4)
    ) == 0x00

    finite_nonzero = np.array(
        [
            -7.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            7.0,
        ],
        dtype=np.float32,
    )
    assert [_raw_e2m1_encode(value) for value in finite_nonzero] == [
        _compressor_e2m1_encode(value) for value in finite_nonzero
    ]

    raw_kernel = _captured_kernel_definition(nvfp4_kv._stock432_pack_kernel)
    raw_body = _metal_helper_body(raw_kernel["header"], "mtplx_e2m1_encode")
    assert _uses_ieee_sign(raw_body)
    assert "mtplx_mia_compressor_e2m1_encode" not in raw_kernel["header"]
    assert raw_kernel["source"].count("mtplx_e2m1_encode(") == 2

    compressor_kernel = _captured_kernel_definition(
        compressor_kernels._nvfp4_finalize_kernel, 4
    )
    compressor_body = _metal_helper_body(
        compressor_kernel["header"], "mtplx_mia_compressor_e2m1_encode"
    )
    assert _uses_numeric_sign(compressor_body)
    assert "uchar code = mtplx_e2m1_encode(value);" in compressor_body
    assert "code & uchar(0x07)" in compressor_body
    assert (
        compressor_kernel["source"].count("mtplx_mia_compressor_e2m1_encode(")
        == 2
    )
    assert "mtplx_e2m1_encode(" not in compressor_kernel["source"]

    raw_numeric_mutation = raw_body.replace(
        "as_type<uint>(value) >> 31", "value < 0.0f ? 1u : 0u"
    )
    compressor_ieee_mutation = compressor_body.replace(
        "value < 0.0f ? 1u : 0u", "as_type<uint>(value) >> 31"
    )
    assert not _uses_ieee_sign(raw_numeric_mutation)
    assert not _uses_numeric_sign(compressor_ieee_mutation)


def test_mia132_indexer_compressor_is_e4m3_not_e2m1() -> None:
    indexer_kernel = _captured_kernel_definition(
        compressor_kernels._indexer_finalize_kernel
    )
    assert "mtplx_indexer_e4m3_encode" in indexer_kernel["header"]
    assert "mtplx_indexer_e4m3_encode(" in indexer_kernel["source"]
    assert "e2m1" not in indexer_kernel["header"].lower()
    assert "e2m1" not in indexer_kernel["source"].lower()


def test_record_kernels_rope_fp32_before_the_bf16_record_boundary() -> None:
    stock432 = _captured_kernel_source(
        compressor_kernels._nvfp4_finalize_kernel, 4
    )
    mia132 = _captured_kernel_source(compressor_kernels._indexer_finalize_kernel)

    assert "normed[tid] = normalized;" in stock432
    assert "float record_value = mtplx_bf16_roundtrip(rotated);" in stock432
    assert "float even = normed[even_dim];" in stock432
    assert "normed[tid] = normalized;" in mia132
    assert "normalized = mtplx_bf16_roundtrip(normalized);" in mia132


@pytest.mark.parametrize(
    "width,expected_digest,legacy_digest",
    [
        (
            512,
            "92d272bd2bf7c20996acb8c46dad044216bf2e8c3a42afdb1ee1437de19ec7e8",
            "76c7eb5a1c0b4f915cbeba1fd2281c4e4625bb096f9c7448b7096462ce24897a",
        ),
        (
            128,
            "2b1ab38462f78a84f457685b1c46b4aeb3d6ed7a767091bb43c8fe4781d8d100",
            "9011be43f422e645c6afc74492395598e8a5d3fde0e99de31a3b0626d2dfcf44",
        ),
    ],
)
def test_post_rope_bf16_boundary_has_deterministic_record_bytes(
    width: int, expected_digest: str, legacy_digest: str
) -> None:
    dimensions = np.arange(width, dtype=np.float32)
    normed = (
        np.sin(dimensions * np.float32(0.173)) * np.float32(1.7)
        + np.linspace(-0.4, 0.6, width, dtype=np.float32)
    )
    angles = (
        np.arange(32, dtype=np.float32) + np.float32(0.37)
    ) * np.float32(0.217)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)

    def rotate(row: np.ndarray) -> np.ndarray:
        result = row.copy()
        tail = row[-64:].reshape(32, 2)
        result[-64::2] = tail[:, 0] * cosine - tail[:, 1] * sine
        result[-63::2] = tail[:, 0] * sine + tail[:, 1] * cosine
        return result

    pack = _pack_stock432 if width == 512 else _pack_mia132
    expected = pack(_bf16_roundtrip(rotate(normed))[None])[0]

    pre_rope_bf16 = _bf16_roundtrip(normed)
    hybrid = _bf16_roundtrip(rotate(pre_rope_bf16))
    legacy = pack(hybrid[None])[0]
    if width == 512:
        # The old Metal path packed the pre-RoPE BF16 latent, then stored a
        # separately rotated BF16 tail.
        legacy = pack(pre_rope_bf16[None])[0]
        legacy[304:] = pack(hybrid[None])[0, 304:]

    assert hashlib.sha256(expected).hexdigest() == expected_digest
    assert hashlib.sha256(legacy).hexdigest() == legacy_digest
    assert expected_digest != legacy_digest


@pytest.mark.parametrize(
    "head_dim,mode,ratio",
    [
        (512, "stock432", 4),
        (512, "stock432", 128),
        (128, "mia132", 4),
    ],
)
def test_full_and_incremental_completed_windows_emit_the_same_source_records(
    head_dim: int, mode: str, ratio: int
) -> None:
    compressor = _make_compressor(head_dim, mode, ratio)
    rng = np.random.default_rng(731)
    x = mx.array(rng.normal(0.0, 0.4, (1, 3 * ratio, 8)).astype(np.float32))

    whole = compressor.mia_records(x)
    state = CompressorState(ratio=ratio, overlap=ratio == 4)
    pieces = [
        compressor.step_records(x[:, : ratio - 1], state, 0),
        compressor.step_records(x[:, ratio - 1 : ratio + 1], state, ratio - 1),
        compressor.step_records(x[:, ratio + 1 :], state, ratio + 1),
    ]
    incremental = mx.concatenate(pieces, axis=1)
    mx.eval(whole, incremental)

    np.testing.assert_array_equal(np.array(incremental), np.array(whole))
    assert whole.shape == (1, 3, 432 if mode == "stock432" else 132)


def test_fixed_compressor_uses_retained_frontier_without_journal_readback() -> None:
    class TracedFixedState(FixedMiaCompressorState):
        def __init__(self) -> None:
            super().__init__(
                ratio=128,
                overlap=False,
                rollback_capacity=8,
                state_width=512,
            )
            self.latest_calls: list[tuple[int, int]] = []

        def _latest(self, count: int) -> tuple[mx.array, mx.array]:
            self.latest_calls.append((int(count), self._journal_end))
            return super()._latest(count)

    compressor = _make_compressor(512, "stock432", 128)
    inputs = mx.array(
        np.random.default_rng(128).normal(0.0, 0.4, (1, 128, 8)).astype(
            np.float32
        )
    )
    expected = compressor.mia_records(inputs)
    state = TracedFixedState()
    before_boundary = compressor.step_records(inputs[:, :127], state, 0)
    actual = compressor.step_records(inputs[:, 127:], state, 127)
    mx.eval(expected, before_boundary, actual)

    assert state.latest_calls == []
    assert before_boundary.shape == (1, 0, 432)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("offset", [3, 127, 191])
@pytest.mark.parametrize(
    "head_dim,mode,ratio",
    [
        (512, "stock432", 4),
        (512, "stock432", 128),
        (128, "mia132", 4),
    ],
)
def test_m6_scheduled_compressor_matches_current_records_and_frontiers(
    offset: int,
    head_dim: int,
    mode: str,
    ratio: int,
) -> None:
    compressor = _make_compressor(head_dim, mode, ratio)
    overlap = ratio == 4
    state_width = (2 if overlap else 1) * head_dim
    expected_state = FixedMiaCompressorState(
        ratio=ratio,
        overlap=overlap,
        rollback_capacity=8,
        state_width=state_width,
    )
    actual_state = FixedMiaCompressorState(
        ratio=ratio,
        overlap=overlap,
        rollback_capacity=8,
        state_width=state_width,
    )
    inputs = mx.array(
        np.random.default_rng(4_000 + head_dim + ratio + offset).normal(
            0.0,
            0.4,
            (1, offset + 6, 8),
        ).astype(np.float32)
    )
    if offset:
        expected_prefix = compressor._step_records_installed(
            inputs[:, :offset],
            expected_state,
            0,
        )
        actual_prefix = compressor._step_records_installed(
            inputs[:, :offset],
            actual_state,
            0,
        )
        mx.eval(expected_prefix, actual_prefix)

    compressed_capacity = (256 + ratio - 1) // ratio
    block_size = max(1, 256 // ratio)
    tables = MiaM6RatioTables.allocate(
        ratio=ratio,
        rollback_rows=actual_state.rollback_rows,
        capacity_tokens=256,
        compressed_capacity=compressed_capacity,
        compressed_block_size=block_size,
        block_table=mx.arange(
            (compressed_capacity + block_size - 1) // block_size,
            dtype=mx.int32,
        ),
    )
    schedule = tables.slice(offset)
    expected = compressor._step_records_installed(
        inputs[:, offset:],
        expected_state,
        offset,
    )
    actual = compressor._step_m6_records_installed(
        inputs[:, offset:],
        actual_state,
        schedule,
    )
    expected_arrays = [
        expected,
        *expected_state.journal_buffers,
        *(
            value
            for value in (
                expected_state.cur_kv,
                expected_state.cur_score,
                expected_state.prev_kv,
                expected_state.prev_score,
            )
            if value is not None
        ),
    ]
    actual_arrays = [
        actual,
        *actual_state.journal_buffers,
        *(
            value
            for value in (
                actual_state.cur_kv,
                actual_state.cur_score,
                actual_state.prev_kv,
                actual_state.prev_score,
            )
            if value is not None
        ),
    ]
    mx.eval(*expected_arrays, *actual_arrays)

    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    for actual_journal, expected_journal in zip(
        actual_state.journal_buffers,
        expected_state.journal_buffers,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.array(actual_journal),
            np.array(expected_journal),
        )
    for name in ("cur_kv", "cur_score", "prev_kv", "prev_score"):
        actual_frontier = getattr(actual_state, name)
        expected_frontier = getattr(expected_state, name)
        assert (actual_frontier is None) == (expected_frontier is None)
        if actual_frontier is not None:
            np.testing.assert_array_equal(
                np.array(actual_frontier),
                np.array(expected_frontier),
            )
    assert actual_state.n_emitted == expected_state.n_emitted
    assert actual_state._journal_end == expected_state._journal_end
    assert actual_state._journal_length == expected_state._journal_length
