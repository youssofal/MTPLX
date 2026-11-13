"""Edge0 FP8 n-gram table as a second backend for the qwen4_exp sidecar gather.

Edge0's ``ple-fp8.gguf`` holds the upstream n-gram table as raw FP8 E4M3 bytes:
one I8 tensor of ``rows x dim`` (the 128 upstream shards concatenated in shard
order, no padding or reshaping beyond the 128-row round-up MTPLX also applies)
and one global ``strata.ple.scale``. MTPLX's own sidecar quantizes the same
rows to 4 bits; this module lets ``_SidecarGather`` read the FP8 rows in place
instead, so nothing is converted and nothing extra is lost.

Selected from the pack (see ``resolve_source``): either a ``config.json`` entry
``"mtplx_ngram_source": {"backend": "edge0_fp8_gguf", "file": "ple-fp8.gguf"}``
or the same two values (``ngram_backend`` / ``ngram_gguf``) in the header
metadata of a stub ``ngram-table.safetensors``.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from .ggml_quants import decode_fp8_e4m3
from .gguf_lite import GGML_I8, GGUFFile

BACKEND = "edge0_fp8_gguf"
TENSOR = "per_layer_token_embd.weight"
FP8_DTYPE = "F8_E4M3"


class Edge0NgramError(ValueError):
    """The GGUF is not an FP8 n-gram table of the shape the model expects."""


def resolve_source(sidecar_path: Path, header_meta: dict | None = None) -> Path | None:
    """The GGUF this pack asks for, or None when it uses the stock 4-bit sidecar."""
    sidecar_path = Path(sidecar_path)
    spec = None
    meta = header_meta or {}
    if meta.get("ngram_backend") == BACKEND:
        spec = {"backend": BACKEND, "file": meta.get("ngram_gguf", "ple-fp8.gguf")}
    else:
        cfg = sidecar_path.parent / "config.json"
        try:
            found = json.loads(cfg.read_text()).get("mtplx_ngram_source")
        except (OSError, ValueError):
            found = None
        if isinstance(found, dict) and found.get("backend") == BACKEND:
            spec = found
    if spec is None:
        return None
    target = Path(str(spec.get("file") or "ple-fp8.gguf")).expanduser()
    return target if target.is_absolute() else sidecar_path.parent / target


def open_table(path: Path, rows: int, dim: int):
    """Validate the GGUF and return ``(entries, scale)`` for ``_SidecarGather``.

    ``entries`` maps ``"weight"`` to a safetensors-style info dict plus the
    absolute data offset, so the gather's existing memmap, pread, hot-row and
    prewarm code addresses the GGUF bytes unchanged (row ``i`` is at
    ``data_offset + i * dim``).
    """
    gguf = GGUFFile(path)
    fmt = gguf.metadata.get("strata.ple.format")
    if fmt != "f8_e4m3":
        raise Edge0NgramError(f"{path}: strata.ple.format is {fmt!r}, expected 'f8_e4m3'")
    if TENSOR not in gguf.tensors:
        raise Edge0NgramError(f"{path}: no tensor {TENSOR!r}")
    t = gguf.tensors[TENSOR]
    if t.ggml_type != GGML_I8 or tuple(t.shape) != (rows, dim):
        raise Edge0NgramError(
            f"{path}: {TENSOR} is type {t.ggml_type} shape {tuple(t.shape)}, "
            f"model expects I8 ({rows}, {dim})"
        )
    scale = float(np.float32(gguf.metadata["strata.ple.scale"]))
    info = {"dtype": FP8_DTYPE, "shape": [rows, dim], "data_offsets": [0, rows * dim]}
    return {"weight": (info, int(t.offset))}, scale


@functools.lru_cache(maxsize=4)
def _f32_lut(scale: float):
    import mlx.core as mx

    table = decode_fp8_e4m3(np.arange(256, dtype=np.uint8), scale)
    return mx.array(np.asarray(table, dtype=np.float32))


def decode_rows(rows_u8, scale: float):
    """uint8 mx.array of FP8 bytes -> float32 ``fp8 * scale``, exact.

    An E4M3 value has at most 4 significant bits and the GGUF scale is a
    float32, so the product is exact in float32 but generally not in bfloat16
    (216 of the 254 finite codes round under the Edge0-120B scale, by up to
    0.29%). The caller (``PLELayer``) casts to the residual-stream dtype, so a
    bfloat16 model still sees one rounding of the exact product and a float32
    model sees no rounding at all.
    """
    import mlx.core as mx

    return mx.take(_f32_lut(float(scale)), rows_u8.astype(mx.uint32))


def skeleton_layout(skeleton_path: Path) -> dict:
    """The n-gram hashing constants Edge0 records in ``model-skeleton.gguf``."""
    md = GGUFFile(skeleton_path).metadata
    return {
        "layers": list(md["qwen4exp.ple.layers"]),
        "ngram_size": int(md["qwen4exp.ple.ngram_size"]),
        "heads_per_ngram": int(md["qwen4exp.ple.heads_per_ngram"]),
        "eos": int(md["qwen4exp.ple.eos_token_id"]),
        "multipliers": [int(v) for v in md["qwen4exp.ple.layer_multipliers"]],
        "offsets": [int(v) for v in md["qwen4exp.ple.head_offsets"]],
        "sizes": [int(v) for v in md["qwen4exp.ple.head_vocab_sizes"]],
    }
