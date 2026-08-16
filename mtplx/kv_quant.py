"""Plain paged-KV quantization helpers.

This is intentionally separate from TurboQuant.  TurboQuant depends on the
external vLLM-Metal encode/attention kernels; this module provides an in-tree
q8/q6/q4 storage mode that can always fall back through MLX SDPA after dequant.

``q6`` is an in-tree MTPLX packing, not any external "Q6" on-disk format: it
stores four 6-bit codes per three bytes and sits between q8 and q4 on both
storage size and quantization step.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


MODES = {"q8", "int8", "q6", "int6", "q4", "int4"}

#: Symmetric quantizer clip bound per bit width. Explicit rather than derived
#: so a new width cannot silently borrow another width's range: q6 read
#: through a ``127 if bits == 8 else 7`` expression would quantize to the q4
#: grid while allocating q6-sized pages.
_QMAX = {8: 127, 6: 31, 4: 7}

#: Storage bias q6 applies before packing: ``unsigned = q + 32`` maps the
#: symmetric -31..31 range onto codes 1..63 inside a 6-bit field. (q4 uses the
#: same idea with a bias of 8, inline in its own branch.)
_Q6_ZERO_POINT = 32


@dataclass(frozen=True)
class PagedKVQuantConfig:
    mode: str = "q8"

    @property
    def normalized_mode(self) -> str:
        raw = self.mode.strip().lower().replace("-", "_")
        if raw in {"int8", "q8_0"}:
            return "q8"
        if raw == "int6":
            return "q6"
        if raw in {"int4", "q4_0"}:
            return "q4"
        return raw

    @property
    def bits(self) -> int:
        mode = self.normalized_mode
        if mode == "q4":
            return 4
        if mode == "q6":
            return 6
        return 8


def env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def paged_kv_quant_mode_from_env() -> str:
    """The single parse of the paged-KV-quant env pair, canonicalized.

    Returns one of ``off``/``q8``/``q6``/``q4``. Every reader goes through
    :func:`~mtplx.runtime_options.normalize_paged_kv_quantization`, so
    spellings like ``8``/``8bit``/``uint8`` cannot normalize in one reader,
    raise in a second, and fall through to the wrong KV layout in a third.
    """

    from mtplx.runtime_options import normalize_paged_kv_quantization

    raw = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or ""
    )
    return str(normalize_paged_kv_quantization(raw))


def config_from_env() -> PagedKVQuantConfig | None:
    mode = paged_kv_quant_mode_from_env()
    if mode == "off":
        return None
    return PagedKVQuantConfig(mode=mode)


def packed_dim(head_dim: int, bits: int) -> int:
    head_dim = int(head_dim)
    bits = int(bits)
    if bits == 8:
        return head_dim
    if bits == 6:
        # Four 6-bit codes share three bytes, so the packed row is 3/4 of the
        # head dim (256 -> 192). Groups may not straddle the row end.
        if head_dim % 4:
            raise ValueError(
                f"q6 paged KV quantization requires head_dim divisible by 4, got {head_dim}"
            )
        return head_dim // 4 * 3
    if bits == 4:
        if head_dim % 2:
            raise ValueError(f"q4 paged KV quantization requires even head_dim, got {head_dim}")
        return head_dim // 2
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def pack_q6(unsigned: Any) -> Any:
    """Pack trailing-axis 6-bit codes (0..63) into 3 bytes per 4 values.

    ``unsigned`` is uint8 with a trailing axis divisible by 4. Each group of
    four codes ``u0..u3`` becomes three little-endian bytes::

        byte0 = u0        | (u1 & 0x03) << 6
        byte1 = u1 >> 2   | (u2 & 0x0F) << 4
        byte2 = u2 >> 4   | (u3 & 0x3F) << 2

    Every partial field is masked before shifting, so no term exceeds 8 bits
    and the whole thing stays in uint8 without intermediate widening.
    """

    import mlx.core as mx

    group_count = int(unsigned.shape[-1]) // 4
    groups = unsigned.reshape(*unsigned.shape[:-1], group_count, 4)
    u0 = mx.bitwise_and(groups[..., 0], 0x3F)
    u1 = mx.bitwise_and(groups[..., 1], 0x3F)
    u2 = mx.bitwise_and(groups[..., 2], 0x3F)
    u3 = mx.bitwise_and(groups[..., 3], 0x3F)
    byte0 = mx.bitwise_or(u0, mx.left_shift(mx.bitwise_and(u1, 0x03), 6))
    byte1 = mx.bitwise_or(
        mx.bitwise_and(mx.right_shift(u1, 2), 0x0F),
        mx.left_shift(mx.bitwise_and(u2, 0x0F), 4),
    )
    byte2 = mx.bitwise_or(
        mx.bitwise_and(mx.right_shift(u2, 4), 0x03),
        mx.left_shift(u3, 2),
    )
    packed = mx.stack([byte0, byte1, byte2], axis=-1)
    return packed.reshape(*unsigned.shape[:-1], group_count * 3).astype(mx.uint8)


def unpack_q6(packed: Any, head_dim: int) -> Any:
    """Inverse of :func:`pack_q6`, returning uint8 codes in 0..63."""

    import mlx.core as mx

    head_dim = int(head_dim)
    group_count = head_dim // 4
    triples = packed.reshape(*packed.shape[:-1], group_count, 3)
    b0 = triples[..., 0]
    b1 = triples[..., 1]
    b2 = triples[..., 2]
    u0 = mx.bitwise_and(b0, 0x3F)
    u1 = mx.bitwise_or(
        mx.bitwise_and(mx.right_shift(b0, 6), 0x03),
        mx.left_shift(mx.bitwise_and(b1, 0x0F), 2),
    )
    u2 = mx.bitwise_or(
        mx.bitwise_and(mx.right_shift(b1, 4), 0x0F),
        mx.left_shift(mx.bitwise_and(b2, 0x03), 4),
    )
    u3 = mx.bitwise_and(mx.right_shift(b2, 2), 0x3F)
    codes = mx.stack([u0, u1, u2, u3], axis=-1)
    return codes.reshape(*packed.shape[:-1], head_dim).astype(mx.uint8)


def quantize_symmetric(x: Any, *, bits: int) -> tuple[Any, Any]:
    import mlx.core as mx

    bits = int(bits)
    if bits not in _QMAX:
        raise ValueError(f"unsupported paged KV quantization bits={bits}")
    qmax = _QMAX[bits]
    max_abs = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
    scale = mx.maximum(max_abs / float(qmax), mx.array(1.0e-6, dtype=mx.float32))
    q = mx.round(x.astype(mx.float32) / scale)
    q = mx.clip(q, -float(qmax), float(qmax))
    # Scales stay fp32 end-to-end: they are computed here in fp32, stored
    # fp32 (cache_state scale caches), and consumed in fp32 by
    # dequantize_symmetric and the paged q8 kernel. The former fp16
    # round-trip added avoidable error on top of the int quantization for a
    # ~1.5% (q8) memory saving on the scale sidecar only.
    if bits == 8:
        return q.astype(mx.int8), scale
    if bits == 6:
        # -31..31 -> 1..63, then four codes per three bytes.
        unsigned = (q + _Q6_ZERO_POINT).astype(mx.uint8)
        return pack_q6(unsigned), scale
    if bits == 4:
        unsigned = (q + 8).astype(mx.uint8)
        even = unsigned[..., 0::2]
        odd = unsigned[..., 1::2]
        packed = mx.bitwise_or(even, mx.left_shift(odd, 4)).astype(mx.uint8)
        return packed, scale
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def dequantize_symmetric(q: Any, scale: Any, *, bits: int, head_dim: int) -> Any:
    import mlx.core as mx

    bits = int(bits)
    if bits == 8:
        return q.astype(mx.float32) * scale.astype(mx.float32)
    if bits == 6:
        signed = unpack_q6(q, int(head_dim)).astype(mx.int16) - _Q6_ZERO_POINT
        return signed.astype(mx.float32) * scale.astype(mx.float32)
    if bits == 4:
        low = mx.bitwise_and(q, 0x0F)
        high = mx.bitwise_and(mx.right_shift(q, 4), 0x0F)
        stacked = mx.stack([low, high], axis=-1).reshape(*q.shape[:-1], int(head_dim))
        signed = stacked.astype(mx.int16) - 8
        return signed.astype(mx.float32) * scale.astype(mx.float32)
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def compression_ratio(*, head_dim: int, bits: int) -> float:
    head_dim = int(head_dim)
    bits = int(bits)
    # Two fp16 tensors, key + value.
    fp16_bytes = 2 * head_dim * 2
    # Two quantized tensors plus one fp32 scale for K and one for V.
    quant_bytes = 2 * packed_dim(head_dim, bits) + 2 * 4
    return float(fp16_bytes) / float(quant_bytes)
