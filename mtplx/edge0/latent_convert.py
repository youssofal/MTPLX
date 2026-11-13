"""Lossless repack of an Edge0 layer's latent MoE weights into MLX containers.

Nothing is requantized: every stored integer and scale is moved, never
rounded, so ``mx.dequantize`` of the output equals the NumPy decoders in
:mod:`mtplx.edge0.ggml_quants` exactly in float32 (bit for bit for the NVFP4
experts; for D/U up to the sign of zero, see below).

MLX 0.32.2 containers, as verified against the installed kernels
(``backend/metal/kernels/fp_quantized.h``, ``fp4.h``, ``fp8.h``) and by
round trips:

- ``nvfp4``: ``weight`` uint32 ``[..., out, in / 8]``, value ``k`` of a row in
  nibble ``k % 8`` of word ``k // 8``, least significant nibble first (so byte
  ``k // 2``, low nibble for even ``k``); ``scales`` uint8 ``[..., out, in / 16]``,
  one per 16 consecutive values, decoded as *signed* E4M3 with no NaN (``0x7F``
  reads as 480, bit 7 as a sign); E2M1 code 8 reads as ``-0.0``. No biases.
- ``affine``: ``weight`` uint32 packed the same way (8 four-bit or 4 eight-bit
  codes per word, lowest bits first), ``w = scale * code + bias`` per group of
  32. With float16 scales ``mx.dequantize`` rounds ``scale * code`` in float16,
  so scales and biases are stored float32, which makes every value exact.

ggml differs from that in three places, each handled here:

- NVFP4 nibble order: ggml byte ``j`` of a 16-value sub-block holds values
  ``j`` and ``j + 8``; repacked to MLX's pairwise order.
- NVFP4 scale bytes are *unsigned* E4M3 in ggml, ``0x7F`` meaning zero. ``0x7F``
  is rewritten to ``0x00`` (zero in both). Bytes ``>= 0x80`` (values of 512 and
  above in ggml, negative in MLX, outside E4M3's range of 448) cannot be
  carried by an MLX scale and raise :class:`Edge0ConvertError`. Neither code
  point occurs in the NVFP4 build (scanned layers 0, 23 and 47: largest scale
  byte 0x7E).
- E2M1 code 8 is ``+0`` in ggml and ``-0`` in MLX; it is rewritten to code 0
  so the repacked weights match bit for bit, sign of zero included.

Q4_0 ``(code - 8) * d`` becomes affine 4-bit ``code * d + (-8 d)``; Q8_0 ``q * d``
becomes affine 8-bit ``(q + 128) * d + (-128 d)``. Both products and sums are
exact in float32 (at most 19 significant bits), so every value matches; the one
bit that can differ is the sign of a zero (ggml gives ``-0`` for a zero code
under a negative ``d``, MLX's ``x + (-x)`` gives ``+0``), which no product or
sum can observe.

D keeps its ggml layout ``[G * L, H]``. U ``[G * H, L]`` is regrouped to
``[H, G * L]`` (row ``h`` is ``U[0, h, :], U[1, h, :], ...``) so the per-group
combine maps back to the hidden size with a single matmul over the
concatenated group latents; whole 32-value blocks move, nothing is split.

Import cost is NumPy only; MLX is imported inside the functions that need it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .ggml_quants import (
    NVFP4_BLOCK_BYTES,
    NVFP4_SUB_VALUES,
    Q4_0_BLOCK_VALUES,
    unpack_nvfp4,
    unpack_q4_0,
    unpack_q8_0,
)

FORMAT_VERSION = 1
NVFP4_GROUP = NVFP4_SUB_VALUES  # 16
DENSE_GROUP = Q4_0_BLOCK_VALUES  # 32 (Q8_0 too)
DENSE_BITS = {"q4": 4, "q8": 8}

EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_KEY_PREFIX = "model.layers.{layer}.mlp."
DEFAULT_FILE_PATTERN = "latent-moe-{layer:03d}.safetensors"
INDEX_NAME = "latent-moe.index.json"

_E2M1_NEG_ZERO = 8
_UE4M3_ZERO_NAN = 0x7F


class Edge0ConvertError(ValueError):
    """Raised when a stored value has no exact MLX representation."""


# --------------------------------------------------------------------------- bit packing


def _pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """uint8 codes ``[..., n]`` (0..15) -> uint32 ``[..., n / 8]``, value ``k`` in nibble ``k % 8``."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    packed = np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8))
    return packed.view("<u4").astype(np.uint32, copy=False)


def _pack_bytes(codes: np.ndarray) -> np.ndarray:
    """uint8 codes ``[..., n]`` -> uint32 ``[..., n / 4]``, value ``k`` in byte ``k % 4``."""
    return np.ascontiguousarray(codes, dtype=np.uint8).view("<u4").astype(np.uint32, copy=False)


# --------------------------------------------------------------------------- NVFP4 -> mlx nvfp4


def nvfp4_codes_to_mlx(codes: np.ndarray, scales: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Value-order E2M1 codes and UE4M3 scale bytes -> MLX ``nvfp4`` ``(weight, scales)``.

    ``shape`` is the matrix shape ``(..., out, in)``; ``codes`` has its element
    count and ``scales`` one byte per 16 values along ``in``.
    """
    *lead, cols = shape
    if cols % NVFP4_GROUP:
        raise Edge0ConvertError(f"NVFP4 row length {cols} is not a multiple of {NVFP4_GROUP}")
    codes = np.asarray(codes, dtype=np.uint8).reshape(*lead, cols)
    scales = np.asarray(scales, dtype=np.uint8).reshape(*lead, cols // NVFP4_GROUP)
    if scales.size and int(scales.max()) >= 0x80:
        bad = int(np.count_nonzero(scales >= 0x80))
        raise Edge0ConvertError(
            f"{bad} NVFP4 scale byte(s) >= 0x80: ggml reads them as unsigned E4M3 (>= 512), MLX's nvfp4 scale "
            "is signed E4M3 (max 448), so no MLX scale byte carries them; these blocks need a different container"
        )
    scales = np.where(scales == _UE4M3_ZERO_NAN, np.uint8(0), scales)
    codes = np.where(codes == _E2M1_NEG_ZERO, np.uint8(0), codes)
    return _pack_nibbles(codes), np.ascontiguousarray(scales)


def nvfp4_to_mlx(buf, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """ggml NVFP4 bytes of a row-major ``[rows, cols]`` matrix -> MLX ``nvfp4`` ``(weight, scales)``."""
    codes, scales = unpack_nvfp4(buf)
    if codes.size != rows * cols:
        raise Edge0ConvertError(f"NVFP4 buffer holds {codes.size} values, expected {rows} x {cols}")
    return nvfp4_codes_to_mlx(codes, scales, (rows, cols))


# --------------------------------------------------------------------------- Q4_0 / Q8_0 -> mlx affine


def _dense_unpack(buf, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (unsigned codes uint8 in value order, d float32 per 32 values)."""
    if fmt == "q4":
        codes, d = unpack_q4_0(buf)
        return codes, d.astype(np.float32)
    if fmt == "q8":
        q, d = unpack_q8_0(buf)
        return (q.astype(np.int16) + 128).astype(np.uint8), d.astype(np.float32)
    raise ValueError(f"dense format must be 'q4' or 'q8', got {fmt!r}")


def dense_to_affine(buf, fmt: str, rows: int, cols: int) -> dict[str, np.ndarray]:
    """ggml Q4_0 / Q8_0 bytes of a row-major ``[rows, cols]`` matrix -> MLX affine ``weight/scales/biases``.

    Scales and biases are float32: ``code * d + (-2**(bits - 1) * d)`` then equals
    the ggml value exactly (``mx.dequantize`` in float32 and the quantized
    matmuls, which promote to float32).
    """
    bits = DENSE_BITS[fmt]
    codes, d = _dense_unpack(buf, fmt)
    if codes.size != rows * cols or cols % DENSE_GROUP:
        raise Edge0ConvertError(f"{fmt} buffer holds {codes.size} values, expected {rows} x {cols} (cols multiple of 32)")
    if not np.all(np.isfinite(d)):
        raise Edge0ConvertError(f"{fmt} block scale is not finite")
    codes = codes.reshape(rows, cols)
    weight = _pack_nibbles(codes) if bits == 4 else _pack_bytes(codes)
    scales = d.reshape(rows, cols // DENSE_GROUP)
    biases = scales * np.float32(-(2 ** (bits - 1)))
    return {"weight": weight, "scales": np.ascontiguousarray(scales), "biases": biases}


def regroup_u(affine: dict[str, np.ndarray], n_groups: int) -> dict[str, np.ndarray]:
    """``U`` affine arrays ``[G * H, L / pf]`` -> ``[H, G * L / pf]`` (group-major columns)."""
    out = {}
    for name, a in affine.items():
        gh, c = a.shape
        out[name] = np.ascontiguousarray(a.reshape(n_groups, gh // n_groups, c).transpose(1, 0, 2).reshape(gh // n_groups, n_groups * c))
    return out


# --------------------------------------------------------------------------- one layer from a pack


def _dense_raw(pack, layer: int, fmt: str) -> tuple[np.memmap, np.memmap]:
    from .pack import _DENSE_FILES

    d_bytes, u_bytes = pack._dense_layer_bytes(fmt, pack.n_groups)
    path = pack.path / _DENSE_FILES[fmt]
    if not path.exists():
        raise Edge0ConvertError(f"{path}: file is missing")
    start = layer * (d_bytes + u_bytes)
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=start, shape=(d_bytes + u_bytes,))
    return raw[:d_bytes], raw[d_bytes:]


def _expert_rows(pack, layer: int) -> np.memmap:
    off = pack._layer_offsets[layer]
    mm = np.memmap(pack.path / "experts.bin", dtype=np.uint8, mode="r", offset=off, shape=(pack.n_experts * pack.blob_bytes,))
    return mm.reshape(pack.n_experts, pack.blob_bytes)


def convert_experts(pack, layer: int, *, chunk: int = 64) -> dict[str, np.ndarray]:
    """All experts of one layer -> ``experts.{gate,up,down}_proj.{weight,scales}`` and ``experts.scalars``.

    Gate and up are ``[E, I, L]``, down ``[E, L, I]`` (out, in), as MLX
    ``nvfp4`` arrays; ``experts.scalars`` is float32 ``[E, 3]`` (s_gate, s_up,
    s_down from each blob's tail). Reads ``chunk`` experts at a time.
    """
    pack._check_ids(layer)
    n_exp, ff, lat = pack.n_experts, pack.expert_ff, pack.latent_dim
    proj = pack._proj_bytes
    shapes = {"gate_proj": (ff, lat), "up_proj": (ff, lat), "down_proj": (lat, ff)}
    out: dict[str, np.ndarray] = {}
    for name, (r, c) in shapes.items():
        out[f"experts.{name}.weight"] = np.empty((n_exp, r, c // 8), np.uint32)
        out[f"experts.{name}.scales"] = np.empty((n_exp, r, c // NVFP4_GROUP), np.uint8)
    scalars = np.empty((n_exp, 3), np.float32)
    rows = _expert_rows(pack, layer)
    for e0 in range(0, n_exp, chunk):
        e1 = min(e0 + chunk, n_exp)
        blobs = np.asarray(rows[e0:e1])
        for j, name in enumerate(EXPERT_PROJECTIONS):
            r, c = shapes[name]
            part = np.ascontiguousarray(blobs[:, j * proj : (j + 1) * proj])
            codes, sc = unpack_nvfp4(part)
            w, s = nvfp4_codes_to_mlx(codes, sc, (e1 - e0, r, c))
            out[f"experts.{name}.weight"][e0:e1] = w
            out[f"experts.{name}.scales"][e0:e1] = s
        tails = np.ascontiguousarray(blobs[:, 3 * proj : 3 * proj + 12]).view("<f4")
        scalars[e0:e1] = tails
    out["experts.scalars"] = scalars
    return out


def convert_dense(pack, layer: int, fmt: str | None = None) -> dict[str, np.ndarray]:
    """One layer's D ``[G * L, H]`` and regrouped U ``[H, G * L]`` as MLX affine arrays."""
    pack._check_ids(layer)
    fmt = fmt or pack.default_dense_format
    if fmt not in DENSE_BITS:
        raise ValueError(f"dense format must be 'q4' or 'q8', got {fmt!r}")
    g, lat, hid = pack.n_groups, pack.latent_dim, pack.hidden
    d_raw, u_raw = _dense_raw(pack, layer, fmt)
    out = {f"dense_d.{k}": v for k, v in dense_to_affine(d_raw, fmt, g * lat, hid).items()}
    u = regroup_u(dense_to_affine(u_raw, fmt, g * hid, lat), g)
    out.update({f"dense_u.{k}": v for k, v in u.items()})
    return out


def convert_layer(pack, layer: int, *, dense_format: str | None = None) -> dict[str, np.ndarray]:
    """Every latent-MoE array of one layer, keyed by its name in ``LatentMoeBlock``.

    ``dense_d.{weight,scales,biases}``, ``dense_u.{weight,scales,biases}``,
    ``experts.{gate,up,down}_proj.{weight,scales}``, ``experts.scalars`` and
    ``labels`` (int32 ``[E]``). The router and the shared expert are not here;
    they come from the skeleton like any other dense weight.
    """
    out = convert_experts(pack, layer)
    out.update(convert_dense(pack, layer, dense_format))
    out["labels"] = np.ascontiguousarray(pack.labels[layer], dtype=np.int32)
    return out


# --------------------------------------------------------------------------- the whole model


def _to_mlx(arrays: dict[str, np.ndarray]):
    import mlx.core as mx

    return {k: mx.array(v) for k, v in arrays.items()}


def write_latent_safetensors(
    pack,
    out_dir,
    *,
    layers: Iterable[int] | None = None,
    dense_format: str | None = None,
    key_prefix: str = DEFAULT_KEY_PREFIX,
    file_pattern: str = DEFAULT_FILE_PATTERN,
) -> Path:
    """Convert layers one at a time into ``latent-moe-LLL.safetensors`` files plus an index.

    ``file_pattern`` names the files (``.format(layer=...)``; the MTPLX pack
    forge uses ``model-latent-LLL.safetensors`` so mlx-lm's ``model*.safetensors``
    glob picks them up). Each file holds one layer's arrays under ``key_prefix.format(layer=...)``;
    only that layer is in memory (about 0.8 GB host plus the same again as MLX
    arrays while it is written for the 120B geometry). Files are written to a
    temporary name and renamed, so an interrupted run leaves no partial file.
    The index (``latent-moe.index.json``) maps every key to its file and records
    the geometry; it is rewritten after each layer. Returns the index path.
    """
    import mlx.core as mx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = dense_format or pack.default_dense_format
    layers = list(range(pack.n_layers)) if layers is None else [int(x) for x in layers]
    index_path = out_dir / INDEX_NAME
    index = {"weight_map": {}}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    meta = {
        "format": "edge0-latent-moe",
        "format_version": str(FORMAT_VERSION),
        "model_name": pack.model_name,
        "hidden": str(pack.hidden),
        "latent_dim": str(pack.latent_dim),
        "expert_ff": str(pack.expert_ff),
        "n_groups": str(pack.n_groups),
        "n_experts": str(pack.n_experts),
        "dense_format": fmt,
        "dense_bits": str(DENSE_BITS[fmt]),
    }
    for layer in layers:
        arrays = convert_layer(pack, layer, dense_format=fmt)
        prefix = key_prefix.format(layer=layer)
        named = _to_mlx({prefix + k: v for k, v in arrays.items()})
        del arrays
        name = file_pattern.format(layer=layer)
        tmp = out_dir / f".{name}.partial.safetensors"
        mx.save_safetensors(str(tmp), named, metadata={**meta, "layer": str(layer)})
        os.replace(tmp, out_dir / name)
        index["weight_map"].update({k: name for k in named})
        del named
        index["metadata"] = meta
        tmp_index = out_dir / f".{INDEX_NAME}.partial"
        tmp_index.write_text(json.dumps(index, indent=1, sort_keys=True))
        os.replace(tmp_index, index_path)
    return index_path


def scan_code_points(pack, layers: Iterable[int], *, chunk: int = 64) -> dict[int, dict[str, int]]:
    """Count the NVFP4 code points MLX and ggml read differently, per layer.

    Returns ``{layer: {"scale_00": n, "scale_7f": n, "scale_ge_80": n, "max_scale": b, "code_8": n}}``.
    """
    res = {}
    for layer in layers:
        rows = _expert_rows(pack, layer)
        proj3 = 3 * pack._proj_bytes
        hist = np.zeros(256, np.int64)
        code8 = 0
        for e0 in range(0, pack.n_experts, chunk):
            b = np.ascontiguousarray(np.asarray(rows[e0 : e0 + chunk, :proj3])).reshape(-1, NVFP4_BLOCK_BYTES)
            hist += np.bincount(b[:, :4].reshape(-1), minlength=256)
            q = b[:, 4:]
            code8 += int(np.count_nonzero((q & 0x0F) == _E2M1_NEG_ZERO) + np.count_nonzero((q >> 4) == _E2M1_NEG_ZERO))
        res[layer] = {
            "scale_00": int(hist[0]),
            "scale_7f": int(hist[_UE4M3_ZERO_NAN]),
            "scale_ge_80": int(hist[0x80:].sum()),
            "max_scale": int(np.nonzero(hist)[0].max()) if hist.any() else 0,
            "code_8": code8,
        }
    return res


__all__ = [
    "DEFAULT_KEY_PREFIX",
    "DENSE_BITS",
    "DENSE_GROUP",
    "Edge0ConvertError",
    "FORMAT_VERSION",
    "NVFP4_GROUP",
    "convert_dense",
    "convert_experts",
    "convert_layer",
    "dense_to_affine",
    "nvfp4_codes_to_mlx",
    "nvfp4_to_mlx",
    "regroup_u",
    "scan_code_points",
    "write_latent_safetensors",
]
