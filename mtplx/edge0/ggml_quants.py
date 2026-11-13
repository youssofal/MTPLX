"""Vectorised NumPy decoders for the ggml block formats in Edge0 packs.

Layouts verified against ``edge0-strata``: ``third_party/ggml/ggml-common.h``
(``block_nvfp4``, ``kvalues_fp4``), ``src/kernels/cuda/iq_kernels.cu`` and
``src/kernels/cpu/nvfp4_avx2.cpp`` (UE4M3 scale, nibble order), and ggml's
Q4_0 / Q8_0. The ``unpack_*`` functions expose the stored integers and scales
exactly so they can be repacked losslessly; ``decode_*`` give float32 values.
All inputs are flat byte buffers (bytes, memoryview, uint8 arrays, memmaps).
"""

from __future__ import annotations

import numpy as np

NVFP4_BLOCK_VALUES = 64
NVFP4_SUB_VALUES = 16
NVFP4_BLOCK_BYTES = 36  # 4 UE4M3 scale bytes + 32 code bytes
Q4_0_BLOCK_VALUES = 32
Q4_0_BLOCK_BYTES = 18  # f16 d + 16 code bytes
Q8_0_BLOCK_VALUES = 32
Q8_0_BLOCK_BYTES = 34  # f16 d + 32 int8

# E2M1 code -> doubled value; the real value is this * ue4m3(scale) * 0.5
KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def _as_bytes(buf) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else np.asarray(buf).reshape(-1).view(np.uint8)


def _blocks(buf, block_bytes: int, what: str) -> np.ndarray:
    a = _as_bytes(buf)
    if a.size % block_bytes:
        raise ValueError(f"{what}: {a.size} bytes is not a multiple of the {block_bytes}-byte block")
    return a.reshape(-1, block_bytes)


def _ue4m3_table() -> np.ndarray:
    x = np.arange(256, dtype=np.int32)
    e = x >> 3
    m = (x & 7).astype(np.float64)
    v = np.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64)))
    v[0] = 0.0
    v[0x7F] = 0.0
    return v.astype(np.float32)


_UE4M3 = _ue4m3_table()


def ue4m3_to_float(b) -> np.ndarray:
    """Unsigned E4M3 scale byte(s) to float32; 0x00 and 0x7F (NaN) decode to 0."""
    return _UE4M3[np.asarray(b, dtype=np.uint8)]


def unpack_nvfp4(buf) -> tuple[np.ndarray, np.ndarray]:
    """NVFP4 blocks -> (codes uint8[n], scales uint8[n // 16]) in value order.

    Within a 16-value sub-block s, byte ``qs[8s + j]`` holds value j in its low
    nibble and value j + 8 in its high nibble.
    """
    blocks = _blocks(buf, NVFP4_BLOCK_BYTES, "NVFP4")
    nb = blocks.shape[0]
    scales = np.ascontiguousarray(blocks[:, :4]).reshape(nb * 4)
    qs = blocks[:, 4:].reshape(nb, 4, 8)
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)  # [nb, 4, 16]
    return codes.reshape(nb * NVFP4_BLOCK_VALUES), scales


def decode_nvfp4(buf) -> np.ndarray:
    codes, scales = unpack_nvfp4(buf)
    vals = KVALUES_FP4[codes].reshape(-1, NVFP4_SUB_VALUES) * (ue4m3_to_float(scales) * np.float32(0.5))[:, None]
    return vals.reshape(-1)


def _f16_scales(blocks: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(blocks[:, :2]).view("<f2").reshape(-1)


def unpack_q4_0(buf) -> tuple[np.ndarray, np.ndarray]:
    """Q4_0 blocks -> (codes uint8[n] in 0..15, d float16[n // 32]).

    Element j of a block is the low nibble of ``qs[j]``, element j + 16 the high one.
    """
    blocks = _blocks(buf, Q4_0_BLOCK_BYTES, "Q4_0")
    qs = blocks[:, 2:]
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    return codes.reshape(-1), _f16_scales(blocks)


def decode_q4_0(buf) -> np.ndarray:
    codes, d = unpack_q4_0(buf)
    vals = (codes.reshape(-1, Q4_0_BLOCK_VALUES).astype(np.float32) - 8.0) * d.astype(np.float32)[:, None]
    return vals.reshape(-1)


def unpack_q8_0(buf) -> tuple[np.ndarray, np.ndarray]:
    """Q8_0 blocks -> (q int8[n], d float16[n // 32])."""
    blocks = _blocks(buf, Q8_0_BLOCK_BYTES, "Q8_0")
    q = np.ascontiguousarray(blocks[:, 2:]).view(np.int8).reshape(-1)
    return q, _f16_scales(blocks)


def decode_q8_0(buf) -> np.ndarray:
    q, d = unpack_q8_0(buf)
    return (q.reshape(-1, Q8_0_BLOCK_VALUES).astype(np.float32) * d.astype(np.float32)[:, None]).reshape(-1)


def decode_bf16(buf) -> np.ndarray:
    """Little-endian bfloat16 bytes -> float32 (the 16 bits are the top half of an f32)."""
    a = _as_bytes(buf)
    if a.size % 2:
        raise ValueError(f"BF16: odd byte count {a.size}")
    return (np.ascontiguousarray(a).view("<u2").astype(np.uint32) << 16).view(np.float32)


def _e4m3_table() -> np.ndarray:
    x = np.arange(256, dtype=np.int32)
    sign = np.where(x & 0x80, -1.0, 1.0)
    e = (x >> 3) & 0xF
    m = (x & 7).astype(np.float64)
    mag = np.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64)))
    v = sign * mag
    v[(x & 0x7F) == 0x7F] = np.nan  # E4M3FN: only S.1111.111 is NaN
    return v.astype(np.float32)


_E4M3 = _e4m3_table()


def decode_fp8_e4m3(buf, scale: float = 1.0) -> np.ndarray:
    """Signed FP8 E4M3 (FN variant, bias 7, no infinities) bytes -> float32 * scale."""
    return _E4M3[_as_bytes(buf)] * np.float32(scale)
