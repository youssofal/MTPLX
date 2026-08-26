"""q6 paged-KV quantization: mode plumbing, bit packing, and error bounds.

q6 stores four 6-bit codes per three bytes. The pack/unpack pair must be
exact on integer codes; the float path is lossy by construction and is only
asserted to stay inside the q6 grid step and to sit between q4 and q8.
"""

from __future__ import annotations

import pytest

from mtplx.kv_quant import PagedKVQuantConfig, compression_ratio, packed_dim
from mtplx.runtime_options import KV_QUANT_MODES, normalize_paged_kv_quantization

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# A. Mode parsing


def test_q6_is_a_canonical_mode() -> None:
    assert KV_QUANT_MODES == ("off", "q8", "q6", "q4")


@pytest.mark.parametrize("spelling", ["q6", "Q6", " q6 ", "6", "6bit", "int6", "uint6"])
def test_q6_spellings_normalize(spelling: str) -> None:
    assert normalize_paged_kv_quantization(spelling) == "q6"


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("8", "q8"),
        ("8bit", "q8"),
        ("uint8", "q8"),
        ("int8", "q8"),
        ("q8_0", "q8"),
        ("q8", "q8"),
        ("4", "q4"),
        ("4bit", "q4"),
        ("uint4", "q4"),
        ("int4", "q4"),
        ("q4_0", "q4"),
        ("q4", "q4"),
        ("off", "off"),
        ("none", "off"),
        ("", "off"),
    ],
)
def test_existing_spellings_are_unchanged(spelling: str, expected: str) -> None:
    assert normalize_paged_kv_quantization(spelling) == expected


@pytest.mark.parametrize("spelling", ["q3", "q5", "q6_k", "q6_0", "6_bit", "nonsense"])
def test_unsupported_modes_still_raise(spelling: str) -> None:
    # q6_k in particular must not be accepted: this packing is MTPLX's own and
    # is not the llama.cpp Q6_K format.
    with pytest.raises(ValueError, match="unsupported paged KV quantization"):
        normalize_paged_kv_quantization(spelling)


@pytest.mark.parametrize(
    ("mode", "bits"),
    [("q8", 8), ("int8", 8), ("q6", 6), ("int6", 6), ("q4", 4), ("int4", 4)],
)
def test_config_reports_the_right_bit_width(mode: str, bits: int) -> None:
    assert PagedKVQuantConfig(mode).bits == bits


def test_q6_config_is_not_silently_q8() -> None:
    # Guards the `4 if q4 else 8` shape, which would allocate q8 pages for q6.
    assert PagedKVQuantConfig("q6").bits != PagedKVQuantConfig("q8").bits


# ---------------------------------------------------------------------------
# B. packed_dim


@pytest.mark.parametrize(
    ("head_dim", "bits", "expected"),
    [
        (256, 8, 256),
        (256, 6, 192),
        (256, 4, 128),
        (64, 6, 48),
        (128, 6, 96),
        (4, 6, 3),
    ],
)
def test_packed_dim(head_dim: int, bits: int, expected: int) -> None:
    assert packed_dim(head_dim, bits) == expected


@pytest.mark.parametrize("head_dim", [2, 6, 10, 62, 255])
def test_q6_rejects_head_dims_not_divisible_by_four(head_dim: int) -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        packed_dim(head_dim, 6)


def test_unsupported_bit_width_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported paged KV quantization bits"):
        packed_dim(256, 5)


# ---------------------------------------------------------------------------
# C. Exact integer pack/unpack round trip


def test_q6_pack_unpack_is_exact_on_boundary_codes() -> None:
    from mtplx.kv_quant import pack_q6, unpack_q6

    signed = [-31, -30, -1, 0, 1, 30, 31, -31]
    unsigned = mx.array([[v + 32 for v in signed]], dtype=mx.uint8)
    packed = pack_q6(unsigned)
    mx.eval(packed)
    assert packed.dtype == mx.uint8
    assert packed.shape == (1, 6)  # 8 codes -> 2 groups -> 6 bytes

    restored = unpack_q6(packed, len(signed))
    mx.eval(restored)
    assert restored.tolist() == unsigned.tolist()


def test_q6_pack_unpack_is_exact_over_the_whole_code_range() -> None:
    from mtplx.kv_quant import pack_q6, unpack_q6

    # Every representable 6-bit code, in every lane position within a group.
    codes = mx.arange(64, dtype=mx.uint8).reshape(1, 64)
    packed = pack_q6(codes)
    restored = unpack_q6(packed, 64)
    mx.eval(packed, restored)
    assert packed.shape == (1, 48)
    assert restored.tolist() == codes.tolist()


def test_q6_pack_unpack_is_exact_on_random_codes() -> None:
    from mtplx.kv_quant import pack_q6, unpack_q6

    mx.random.seed(6060)
    codes = mx.random.randint(1, 64, (3, 5, 256)).astype(mx.uint8)
    packed = pack_q6(codes)
    restored = unpack_q6(packed, 256)
    mx.eval(packed, restored)
    assert packed.shape == (3, 5, 192)
    assert restored.shape == codes.shape
    assert bool(mx.array_equal(restored, codes).item())


def test_q6_packing_preserves_element_order() -> None:
    from mtplx.kv_quant import pack_q6, unpack_q6

    codes = mx.array([[0, 1, 2, 3, 60, 61, 62, 63]], dtype=mx.uint8)
    restored = unpack_q6(pack_q6(codes), 8)
    mx.eval(restored)
    assert restored.tolist() == [[0, 1, 2, 3, 60, 61, 62, 63]]


def test_q6_byte_layout_matches_the_documented_bit_packing() -> None:
    from mtplx.kv_quant import pack_q6

    # u0=1, u1=2, u2=3, u3=4 with the documented little-endian layout:
    #   byte0 = 1 | (2 & 3) << 6      = 0x81
    #   byte1 = (2 >> 2) | (3 & 15) << 4 = 0x30
    #   byte2 = (3 >> 4) | 4 << 2     = 0x10
    packed = pack_q6(mx.array([[1, 2, 3, 4]], dtype=mx.uint8))
    mx.eval(packed)
    assert packed.tolist() == [[0x81, 0x30, 0x10]]


# ---------------------------------------------------------------------------
# D. Float quantize / dequantize


def test_q6_quantize_shapes_and_dtypes() -> None:
    from mtplx.kv_quant import quantize_symmetric

    mx.random.seed(17)
    x = mx.random.normal((2, 7, 256), dtype=mx.float16)
    q, scale = quantize_symmetric(x, bits=6)
    mx.eval(q, scale)

    assert q.dtype == mx.uint8
    assert q.shape == (2, 7, 192)
    assert scale.dtype == mx.float32
    assert scale.shape == (2, 7, 1)


def test_q6_codes_never_exceed_the_representable_range() -> None:
    from mtplx.kv_quant import quantize_symmetric, unpack_q6

    mx.random.seed(4242)
    x = 8.0 * mx.random.normal((4, 256), dtype=mx.float32)
    q, _ = quantize_symmetric(x, bits=6)
    codes = unpack_q6(q, 256).astype(mx.int32) - 32
    mx.eval(codes)
    assert int(mx.max(codes).item()) <= 31
    assert int(mx.min(codes).item()) >= -31


def test_q6_dequantized_error_stays_within_one_grid_step() -> None:
    from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric

    mx.random.seed(991)
    x = mx.random.normal((3, 256), dtype=mx.float32)
    q, scale = quantize_symmetric(x, bits=6)
    back = dequantize_symmetric(q, scale, bits=6, head_dim=256)
    mx.eval(back)

    assert back.shape == x.shape
    assert bool(mx.all(mx.isfinite(back)).item())

    # On a scale = max|x| / 31 grid, round-to-nearest contributes half a step
    # and floating-point evaluation adds a small relative term on top, so
    # bound at well under one step rather than exactly half of one.
    step = mx.max(mx.abs(x), axis=-1, keepdims=True) / 31.0
    err = mx.abs(back - x)
    assert bool(mx.all(err <= 0.6 * step + 1e-3).item())


def test_q6_dequantize_is_finite_on_an_all_zero_row() -> None:
    from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric

    x = mx.zeros((1, 256), dtype=mx.float32)
    q, scale = quantize_symmetric(x, bits=6)
    back = dequantize_symmetric(q, scale, bits=6, head_dim=256)
    mx.eval(back)
    assert bool(mx.all(mx.isfinite(back)).item())
    assert float(mx.max(mx.abs(back)).item()) == 0.0


# ---------------------------------------------------------------------------
# E. q6 error sits between q4 and q8


def test_q6_reconstruction_error_lies_between_q4_and_q8() -> None:
    from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric

    def rms_error(bits: int, x) -> float:
        q, scale = quantize_symmetric(x, bits=bits)
        back = dequantize_symmetric(q, scale, bits=bits, head_dim=int(x.shape[-1]))
        err = mx.sqrt(mx.mean((back - x) ** 2))
        mx.eval(err)
        return float(err.item())

    # Deterministic seed and a large sample: with 64 rows of 256 normal
    # samples the per-bit grid steps (1/127, 1/31, 1/7 of max|x|) are far
    # enough apart that the ordering is stable, not a coin flip.
    mx.random.seed(20260816)
    x = mx.random.normal((64, 256), dtype=mx.float32)

    err8 = rms_error(8, x)
    err6 = rms_error(6, x)
    err4 = rms_error(4, x)

    assert err8 < err6 < err4
    # The grid steps differ by ~4x each way; require a clear margin rather
    # than a hair's-breadth inequality.
    assert err6 > 2.0 * err8
    assert err4 > 2.0 * err6


# ---------------------------------------------------------------------------
# F. Storage / compression


@pytest.mark.parametrize(
    ("bits", "expected_bytes"),
    [
        # K + V packed rows plus one fp32 scale each, head_dim = 256.
        (8, 2 * 256 + 8),  # 520
        (6, 2 * 192 + 8),  # 392
        (4, 2 * 128 + 8),  # 264
    ],
)
def test_calculated_kv_bytes_per_token_per_kv_head(bits: int, expected_bytes: int) -> None:
    assert 2 * packed_dim(256, bits) + 8 == expected_bytes


def test_fp16_reference_bytes_per_token_per_kv_head() -> None:
    assert 2 * 256 * 2 == 1024


def test_compression_ratio_orders_q4_then_q6_then_q8() -> None:
    r8 = compression_ratio(head_dim=256, bits=8)
    r6 = compression_ratio(head_dim=256, bits=6)
    r4 = compression_ratio(head_dim=256, bits=4)
    assert r4 > r6 > r8 > 1.0
    assert r6 == pytest.approx(1024 / 392)


def test_calculated_full_cache_size_for_the_qwen3_8_full_attention_layout() -> None:
    # 16 full-attention layers x 4 KV heads x 262144 tokens. Storage
    # arithmetic only -- not a measurement of process RAM.
    layers, kv_heads, tokens = 16, 4, 262_144
    gib = float(layers * kv_heads * tokens)

    def gibibytes(bytes_per_token: int) -> float:
        return gib * bytes_per_token / (1024**3)

    assert gibibytes(1024) == pytest.approx(16.0)
    assert gibibytes(520) == pytest.approx(8.125)
    assert gibibytes(392) == pytest.approx(6.125)
    assert gibibytes(264) == pytest.approx(4.125)
