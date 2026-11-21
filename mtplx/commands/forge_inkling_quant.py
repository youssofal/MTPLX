"""Inkling 2 Mini consumer packs: affine projections and vocabulary, native experts, whole vision tower and drafter.

Recipes (``matrix_layout`` is the single source of which matrix gets which bits):

- ``quality``: the published checkpoint, byte for byte (``stage_native_pack``).
- ``bare-speed``: every large 16-bit matrix a token reads (the six attention projections of all
  24 layers and the vocabulary head) as 4-bit affine codes in groups of 64; experts keep their
  trained NVFP4 and MXFP8 formats; routers, norms, clip scales, convolutions, controllers and the
  relative basis keep their source values.
- ``speed``: the mixed recipe (4 bits where the model does not notice, 8 where it does).
- ``native-quality``: the vocabulary head at 8 bits, everything else published (a research control).

The writer streams: unchanged tensors are copied byte ranges of the source file, and one matrix at a
time is converted, so no recipe ever holds the checkpoint in memory. Every pack is audited against
the pinned upstream before it is moved into place: the vision tower (count, dtype, shape and bytes of
every tensor, ``vision_config``) and the drafter file. A failed audit leaves no pack behind.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .forge_inkling import (
    SIDECARS,
    WEIGHTS,
    _copy_weight,
    stage_native_pack,
    verify_source,
    write_pack_metadata,
)

HEAD = "model.llm.unembed.weight"
EMBED = "model.llm.embed.weight"
PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud")
_PROJECTION = re.compile(
    r"model\.llm\.layers\.(\d+)\.attn\.(" + "|".join(PROJECTIONS) + r")\.weight"
)
_EXPERTS = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)")
RECIPES = ("bare-speed", "bare-speed-e4", "speed", "quality", "native-quality")
SHARD_BYTES = 5 * 1000**3
# A recipe may move the four trained-MXFP8 expert layers (20 to 23) to affine codes.
_FP8_EXPERT_LAYERS = range(20, 24)


def matrix_layout(name: str, recipe: str) -> tuple[int, int] | None:
    """(bits, group size) of the affine codes a recipe stores for ``name``; None keeps the source tensor."""
    if recipe in {"speed", "native-quality"}:
        return (8, 32) if name == HEAD else None
    if recipe in {"bare-speed", "bare-speed-e4"}:
        if name == HEAD or _PROJECTION.fullmatch(name):
            return (4, 64)
        expert = _EXPERTS.fullmatch(name)
        if recipe == "bare-speed-e4" and expert and int(expert.group(1)) in _FP8_EXPERT_LAYERS:
            return (4, 64)
        return None
    if recipe == "quality":
        return None
    raise ValueError(f"Unknown Inkling recipe {recipe!r}")


# --------------------------------------------------------------------------- number formats
def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round to nearest, ties to even, as MLX casts float32 to bfloat16 (finite inputs)."""
    words = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def _round_bf16(values: np.ndarray) -> np.ndarray:
    return bf16_to_f32(f32_to_bf16(values))


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """MLX's affine layout: each uint32 holds 32 // bits codes, the first in the lowest bits."""
    if 32 % bits:
        raise ValueError("packing supports 2, 4 and 8 bits")
    per = 32 // bits
    words = codes.reshape(*codes.shape[:-1], codes.shape[-1] // per, per).astype(np.uint32)
    shifts = np.arange(per, dtype=np.uint32) * np.uint32(bits)
    return np.bitwise_or.reduce(words << shifts, axis=-1)


def affine_groups(x: np.ndarray, bits: int, rounding: str = "mse"):
    """Codes, scales and biases for rows of one group each, scales and biases rounded to bfloat16.

    ``mlx`` reproduces ``mx.quantize``'s choice (range end points, the larger-magnitude end exact).
    ``mse`` starts from five ranges shrunk about the centre, refines each by alternating the nearest
    codes and the least-squares scale and bias, and keeps the candidate with the smallest squared
    error after bfloat16 rounding; the full range is always a candidate. Returns float32 squared
    error per group as well.
    """
    levels = float((1 << bits) - 1)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    if rounding == "mlx":
        mask = np.abs(lo) > np.abs(hi)
        scale = np.maximum((hi - lo) / levels, 1e-7)
        scale = np.where(mask, scale, -scale)
        edge = np.where(mask, lo, hi)
        q0 = np.rint(edge / scale)
        scale = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scale)
        bias = np.where(q0 == 0, 0.0, edge).astype(np.float32)
        codes = np.clip(np.rint((x - bias) / scale), 0, levels)
        scale16, bias16 = _round_bf16(scale), _round_bf16(bias)
        err = np.square(scale16 * codes + bias16 - x).sum(axis=1, keepdims=True)
        return codes.astype(np.uint8), scale16, bias16, err
    if rounding != "mse":
        raise ValueError(f"Unknown rounding {rounding!r}")
    centre, half = (hi + lo) * 0.5, (hi - lo) * 0.5
    best_err = best_scale = best_bias = best_codes = None
    for shrink in (1.0, 0.95, 0.9, 0.85, 0.8):
        scale = np.maximum(2.0 * shrink * half / levels, 1e-12).astype(np.float32)
        bias = (centre - shrink * half).astype(np.float32)
        for _ in range(3):
            codes = np.clip(np.rint((x - bias) / scale), 0, levels)
            mean_codes = codes.mean(axis=1, keepdims=True)
            mean_x = x.mean(axis=1, keepdims=True)
            spread = codes - mean_codes
            var = np.square(spread).sum(axis=1, keepdims=True)
            cov = (spread * (x - mean_x)).sum(axis=1, keepdims=True)
            fit = cov / np.where(var > 0, var, 1.0)
            ok = (var > 0) & (fit > 1e-12)
            scale = np.where(ok, fit, scale).astype(np.float32)
            bias = np.where(ok, mean_x - fit * mean_codes, bias).astype(np.float32)
        scale16, bias16 = _round_bf16(scale), _round_bf16(bias)
        scale16 = np.where(scale16 > 0, scale16, np.float32(1e-12))
        codes = np.clip(np.rint((x - bias16) / scale16), 0, levels)
        err = np.square(scale16 * codes + bias16 - x).sum(axis=1, keepdims=True)
        if best_err is None:
            best_err, best_scale, best_bias, best_codes = err, scale16, bias16, codes
            continue
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_scale = np.where(better, scale16, best_scale)
        best_bias = np.where(better, bias16, best_bias)
        best_codes = np.where(better, codes, best_codes)
    return best_codes.astype(np.uint8), best_scale, best_bias, best_err


def affine_quantize(w: np.ndarray, bits: int, group_size: int, rounding: str = "mse", rows_per_chunk: int = 4096):
    """``w`` (..., rows, columns) float32 -> (codes uint32, scales bf16 bits, biases bf16 bits, stats)."""
    shape = w.shape
    columns = shape[-1]
    if columns % group_size or group_size % (32 // bits):
        raise ValueError(f"{columns} columns do not divide into groups of {group_size}")
    flat = w.reshape(-1, columns)
    codes = np.empty((flat.shape[0], columns * bits // 32), np.uint32)
    scales = np.empty((flat.shape[0], columns // group_size), np.uint16)
    biases = np.empty_like(scales)
    sq_err = sq_norm = 0.0
    max_err = 0.0
    for start in range(0, flat.shape[0], rows_per_chunk):
        block = np.ascontiguousarray(flat[start : start + rows_per_chunk], np.float32)
        rows = block.shape[0]
        groups = block.reshape(-1, group_size)
        q, scale, bias, err = affine_groups(groups, bits, rounding)
        codes[start : start + rows] = pack_codes(q.reshape(rows, columns), bits)
        scales[start : start + rows] = f32_to_bf16(scale).reshape(rows, -1)
        biases[start : start + rows] = f32_to_bf16(bias).reshape(rows, -1)
        recon = scale * q + bias
        max_err = max(max_err, float(np.abs(recon - groups).max()))
        sq_err += float(err.sum(dtype=np.float64))
        sq_norm += float(np.square(groups, dtype=np.float64).sum())
    lead = shape[:-1]
    stats = {
        "relative_rms_error": (sq_err / sq_norm) ** 0.5 if sq_norm else 0.0,
        "max_abs_error": max_err,
    }
    return (
        codes.reshape(*lead, -1),
        scales.reshape(*lead, -1),
        biases.reshape(*lead, -1),
        stats,
    )


# --------------------------------------------------------------------------- safetensors streaming
_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "U32": 4, "I32": 4, "I64": 8}


def _nbytes(dtype: str, shape) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * _SIZES[dtype]


def _read_tensor(source: Path, data_start: int, entry: dict) -> bytes:
    begin, end = entry["data_offsets"]
    with source.open("rb") as handle:
        handle.seek(data_start + begin)
        raw = handle.read(end - begin)
    if len(raw) != end - begin:
        raise ValueError("Truncated source tensor")
    return raw


def _write_shard(path: Path, entries: list, metadata: dict) -> None:
    """``entries``: (name, dtype, shape, producer) where producer() yields bytes; streamed in order."""
    header, offset = {"__metadata__": metadata}, 0
    for name, dtype, shape, _ in entries:
        size = _nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as out:
        out.write(len(raw).to_bytes(8, "little"))
        out.write(raw)
        for name, dtype, shape, producer in entries:
            expected, written = _nbytes(dtype, shape), 0
            for chunk in producer():
                out.write(chunk)
                written += len(chunk)
            if written != expected:
                raise ValueError(f"{name}: wrote {written} bytes, expected {expected}")


def _copier(source: Path, data_start: int, entry: dict):
    def produce():
        begin, end = entry["data_offsets"]
        with source.open("rb") as handle:
            handle.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                data = handle.read(min(remaining, 16 * 1024**2))
                if not data:
                    raise ValueError("Truncated source tensor")
                remaining -= len(data)
                yield data

    return produce


def _source_matrix(source: Path, data_start: int, entry: dict, name: str) -> np.ndarray:
    raw = _read_tensor(source, data_start, entry)
    if entry["dtype"] == "BF16":
        return bf16_to_f32(np.frombuffer(raw, "<u2")).reshape(entry["shape"])
    raise ValueError(f"{name}: no affine conversion from {entry['dtype']}")


def _decoded_fp8_experts(source: Path, data_start: int, header: dict, name: str) -> np.ndarray:
    """The trained MXFP8 expert weights in float32, decoded by the runtime's own MLX kernel."""
    import mlx.core as mx

    codes = np.frombuffer(_read_tensor(source, data_start, header[name]), np.uint8).reshape(header[name]["shape"])
    scales = np.frombuffer(_read_tensor(source, data_start, header[name + "_scale"]), np.uint8).reshape(
        header[name + "_scale"]["shape"]
    )
    out = np.empty(codes.shape, np.float32)
    with mx.stream(mx.cpu):
        for expert in range(codes.shape[0]):
            weight = mx.dequantize(
                mx.array(codes[expert]).view(mx.uint32),
                mx.array(scales[expert]),
                group_size=32,
                bits=8,
                mode="mxfp8",
                dtype=mx.float32,
            )
            out[expert] = np.array(weight)
    return out


def _sha256_range(path: Path, data_start: int, entry: dict) -> str:
    digest = hashlib.sha256()
    for chunk in _copier(path, data_start, entry)():
        digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path):
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return 8 + size, header


# --------------------------------------------------------------------------- audit
def audit_pack(pack: Path, source: Path) -> dict:
    """Law 1 and drafter audit against the pinned upstream. Raises on the first failure."""
    target_name, draft_name = WEIGHTS
    src_start, src_header = _read_header(source / target_name)
    src_vision = {k: e for k, e in src_header.items() if k.startswith("model.visual.")}
    index = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {name: _read_header(pack / name) for name in sorted(set(index.values()))}
    pack_vision = {k: v for k, v in index.items() if k.startswith("model.visual.")}
    problems = []
    if set(pack_vision) != set(src_vision):
        problems.append(
            f"vision keys differ: missing {sorted(set(src_vision) - set(pack_vision))[:5]}, "
            f"extra {sorted(set(pack_vision) - set(src_vision))[:5]}"
        )
    matched = 0
    for key, entry in sorted(src_vision.items()):
        shard = pack_vision.get(key)
        if shard is None:
            continue
        start, header = shards[shard]
        got = header.get(key)
        if got is None or got["dtype"] != entry["dtype"] or got["shape"] != entry["shape"]:
            problems.append(f"{key}: header {got} against upstream {entry}")
            continue
        if entry["dtype"] != "BF16":
            problems.append(f"{key}: upstream dtype {entry['dtype']}")
        if _sha256_range(pack / shard, start, got) != _sha256_range(source / target_name, src_start, entry):
            problems.append(f"{key}: bytes differ from upstream")
            continue
        matched += 1
    src_config = json.loads((source / "config.json").read_text())
    pack_config = json.loads((pack / "config.json").read_text())
    vision_config_equal = src_config.get("vision_config") == pack_config.get("vision_config")
    if not vision_config_equal:
        problems.append("vision_config differs from upstream")
    with (pack / draft_name).open("rb") as handle:
        draft_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if draft_sha != WEIGHTS[draft_name]:
        problems.append("drafter file differs from upstream")
    _, draft_header = _read_header(pack / draft_name)
    receipt = {
        "upstream": {"revision": json.loads((pack / "mtplx-pack.json").read_text()).get("source_revision")
                     if (pack / "mtplx-pack.json").exists() else None,
                     "vision_tensors": len(src_vision),
                     "vision_bf16": sum(e["dtype"] == "BF16" for e in src_vision.values())},
        "pack": {"vision_tensors": len(pack_vision), "vision_tensors_byte_identical": matched,
                 "vision_config_equal": vision_config_equal, "draft_sha256": draft_sha,
                 "draft_tensors": len(draft_header)},
        "problems": problems,
        "pass": not problems and matched == len(src_vision) == 422 and len(draft_header) == 54,
    }
    if not receipt["pass"]:
        raise ValueError(f"Inkling pack audit failed: {problems[:5] or receipt}")
    return receipt


# --------------------------------------------------------------------------- build
def _order(key: str):
    layer = re.match(r"model\.llm\.layers\.(\d+)\.", key)
    if key.startswith("model.visual."):
        return (2, 0, key)
    if layer:
        return (1, int(layer.group(1)), key)
    return (0, 0, key)


def build_pack(source: Path, destination: Path, recipe: str, *, rounding: str = "mse", receipt: Path | None = None) -> dict:
    if recipe == "quality":
        return stage_native_pack(source, destination, recipe=recipe)
    if recipe not in RECIPES:
        raise ValueError(f"Unknown Inkling recipe {recipe!r}")
    source, destination = source.resolve(), destination.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    config, header_size, header = verify_source(source)
    target_name, draft_name = WEIGHTS
    source_target = source / target_name
    data_start = 8 + header_size
    layouts = {key: matrix_layout(key, recipe) for key in header}
    layouts = {key: layout for key, layout in layouts.items() if layout is not None}
    # Removed with the expert codes they scale when a recipe stores those experts as affine codes.
    dropped = {
        key + suffix
        for key in layouts
        if _EXPERTS.fullmatch(key)
        for suffix in ("_scale", "_amax", "_outer_scale_denominator")
        if key + suffix in header
    }
    stats = {}

    def quantized_producers(key):
        bits, group = layouts[key]
        state = {}

        def compute():
            if "codes" not in state:
                entry = header[key]
                if entry["dtype"] == "F8_E4M3":
                    matrix = _decoded_fp8_experts(source_target, data_start, header, key)
                else:
                    matrix = _source_matrix(source_target, data_start, entry, key)
                codes, scales, biases, info = affine_quantize(matrix, bits, group, rounding)
                if rounding == "mse":
                    sample = matrix.reshape(-1, matrix.shape[-1])[:: max(1, matrix.size // matrix.shape[-1] // 2048)]
                    info["mlx_relative_rms_error_sample"] = affine_quantize(sample, bits, group, "mlx")[3]["relative_rms_error"]
                stats[key] = {"bits": bits, "group_size": group, "source_dtype": entry["dtype"],
                              "shape": entry["shape"], **info}
                state.update(codes=codes, scales=scales, biases=biases)
            return state

        def part(field):
            def produce():
                data = compute()[field]
                yield data.tobytes()
                if field == "biases":
                    state.clear()
            return produce

        lead = header[key]["shape"][:-1]
        columns = header[key]["shape"][-1]
        return [
            (key, "U32", [*lead, columns * bits // 32], part("codes")),
            (key + ".mtplx_scales", "BF16", [*lead, columns // group], part("scales")),
            (key + ".mtplx_biases", "BF16", [*lead, columns // group], part("biases")),
        ]

    entries = []
    for key in sorted(header, key=_order):
        if key in dropped:
            continue
        if key in layouts:
            entries.extend(quantized_producers(key))
        else:
            entry = header[key]
            entries.append((key, entry["dtype"], entry["shape"], _copier(source_target, data_start, entry)))
    shards, current, size = [], [], 0
    for item in entries:
        item_size = _nbytes(item[1], item[2])
        if current and size + item_size > SHARD_BYTES and not item[0].endswith((".mtplx_scales", ".mtplx_biases")):
            shards.append(current)
            current, size = [], 0
        current.append(item)
        size += item_size
    shards.append(current)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".inkling-stage-") as scratch:
        staged = Path(scratch) / "pack"
        staged.mkdir()
        index = {}
        for number, shard in enumerate(shards, 1):
            name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
            _write_shard(staged / name, shard, {"format": "mlx"})
            index.update(dict.fromkeys((item[0] for item in shard), name))
        config["mtplx_inkling_precision"] = {
            key: {"bits": bits, "group_size": group} for key, (bits, group) in sorted(layouts.items())
        }
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staged / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": sum(_nbytes(i[1], i[2]) for i in entries)}, "weight_map": index},
                       indent=2) + "\n"
        )
        _copy_weight(source / draft_name, staged / draft_name)
        for name in SIDECARS:
            if name in {"config.json", "model.safetensors.index.json"}:
                continue
            dst = staged / ("UPSTREAM.md" if name == "README.md" else name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
        manifest = write_pack_metadata(staged, recipe)
        audit = audit_pack(staged, source)
        (staged / "vision-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        manifest["vision_audit"] = audit
        if receipt is not None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({"recipe": recipe, "rounding": rounding, "manifest": manifest,
                                           "matrices": stats, "dropped": sorted(dropped)}, indent=2) + "\n")
        os.rename(staged, destination)
    return manifest
