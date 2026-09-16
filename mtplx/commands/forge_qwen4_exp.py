"""Forge convert lane for the Qwen3.8-Flash-Next family (``qwen4_exp``).

The pinned mlx-lm has no ``qwen4_exp`` architecture, so the flat
``mlx_lm convert`` lane cannot see a Flash-Next source at all (issue #390:
``forge build`` on any Flash-Next fine-tune died with ``Model type qwen4_exp
not supported`` while the official packs were converted out of band and
``forge verify --stamp``ed). This module is that out-of-band lane, in tree:

* **n-gram sidecar** -- the 51B PLE table (128 bf16 shards, ~95 GiB) is never
  materialized. Each shard is quantized row-wise and streamed into
  ``ngram-table.safetensors`` in the layout ``NGramTable.attach_sidecar``
  reads (``ngram.weight``/``scales``/``biases`` + ``ngram_bits``/
  ``ngram_group_size`` metadata). Two minutes, ~800 MB peak.
* **body** -- MTPLX's own ``mtplx.models.qwen4_exp`` backend with
  ``ngram_sidecar=True`` (the table registers no parameter), the model's
  convert-time ``quant_predicate`` (the Optimized Speed recipe) layered under
  any ``module_overrides``, and a **per-tensor** evaluation loop: mlx-lm's
  ``save_model`` evaluates a whole ~5 GiB shard's lazy sanitize+quantize graph
  in one Metal command buffer, which trips the GPU watchdog
  (``kIOGPUCommandBufferCallbackErrorTimeout``) on this 180B graph.
* **MTP sidecar** -- the head in the layout ``Model.attach_mtp`` loads: packed
  ``experts.gate_up_proj`` split into ``switch_mlp.{gate,up}_proj``, the
  zero-centered ``(1+w)`` norms shifted (``pre_fc_norm_*`` excluded: attach_mtp
  shifts those itself), per-module quant mirroring the trunk recipe.

Vision graft, MTP sidecar validation, verify and stamping are the ordinary
``_cmd_build`` steps that follow this lane.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Any, Callable

NGRAM_FILE = "ngram-table.safetensors"
MTP_FILE = "mtp.safetensors"
NGRAM_KEY = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{index}.weight"

#: The runtime's fixed-M4 verify lane (qwen4_fixed_verify) pins the production
#: sidecar layout to 4-bit / 32-group; other widths load but are refused at the
#: first request. Kept as data so the guard and this lane cannot disagree.
NGRAM_PRODUCTION_LAYOUT = (4, 32)

#: ``Model.quant_predicate`` ("Optimized Speed"), mirrored for the MTP head,
#: whose modules never pass through mlx-lm's quantize_model.
MTP_EIGHT_BIT_SUFFIXES = (
    "mlp.gate", "shared_expert_gate", "shared_expert.gate_proj",
    "shared_expert.up_proj", "shared_expert.down_proj", "indexer.index_qk_proj",
)
QSA_PROJ_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
#: Model._HF_NORM_SHIFT_SUFFIXES minus the ones attach_mtp shifts itself.
MTP_NORM_SHIFT_SUFFIXES = (
    ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight",
    ".hc_norm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight",
)

Progress = Callable[[str, float, str, bool], None]


class Qwen4ForgeError(RuntimeError):
    pass


def _noop_progress(name: str, progress: float, label: str, finished: bool) -> None:
    del name, progress, label, finished


def _log(message: str) -> None:
    import sys

    print(f"[forge] {message}", file=sys.stderr, flush=True)


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def is_qwen4_exp_source(config: dict[str, Any]) -> bool:
    types = {str(config.get("model_type") or "").lower(), str(_text_config(config).get("model_type") or "").lower()}
    return "qwen4_exp" in types


# --------------------------------------------------------------------------- recipe
def recipe_params(recipe: dict[str, Any]) -> dict[str, int]:
    """Resolved widths for the three quantized payloads of a Flash-Next pack."""
    body_bits = int(4 if recipe.get("body_bits") is None else recipe.get("body_bits"))
    body_group = int(recipe.get("body_group_size") or 32)
    if str(recipe.get("body_mode") or "affine") != "affine":
        raise Qwen4ForgeError("qwen4_exp lane supports body_mode 'affine' only")
    if body_bits <= 0:
        raise Qwen4ForgeError("qwen4_exp lane requires body_bits > 0 (a bf16 trunk does not fit any Mac)")
    ngram = recipe.get("ngram") if isinstance(recipe.get("ngram"), dict) else {}
    ngram_bits = int(NGRAM_PRODUCTION_LAYOUT[0] if ngram.get("bits") is None else ngram.get("bits"))
    ngram_group = int(ngram.get("group_size") or NGRAM_PRODUCTION_LAYOUT[1])
    mtp_bits = recipe.get("qwen4_mtp_bits")
    mtp_bits = body_bits if mtp_bits is None else int(mtp_bits)
    return {
        "body_bits": body_bits, "body_group": body_group,
        "ngram_bits": ngram_bits, "ngram_group": ngram_group,
        "mtp_bits": mtp_bits, "mtp_group": body_group,
    }


# --------------------------------------------------------------------------- n-gram sidecar
def ngram_sidecar_layout(rows: int, dim: int, *, bits: int, group: int) -> dict[str, Any]:
    """Safetensors header for the streamed table (pure arithmetic; unit-tested)."""
    if bits == 0:
        tensors = {"ngram.weight": ("BF16", [rows, dim], 2)}
    else:
        if dim % group:
            raise Qwen4ForgeError(f"n-gram dim {dim} is not divisible by group {group}")
        if (dim * bits) % 32:
            raise Qwen4ForgeError(f"n-gram dim {dim} x {bits} bits does not pack into uint32 rows")
        tensors = {
            "ngram.weight": ("U32", [rows, dim * bits // 32], 4),
            "ngram.scales": ("BF16", [rows, dim // group], 2),
            "ngram.biases": ("BF16", [rows, dim // group], 2),
        }
    header: dict[str, Any] = {"__metadata__": {
        "ngram_bits": str(bits), "ngram_group_size": str(group), "rows": str(rows), "dim": str(dim),
    }}
    offset = 0
    for name, (dtype, shape, itemsize) in tensors.items():
        nbytes = shape[0] * shape[1] * itemsize
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    return header


def ngram_shards(source: Path, config: dict[str, Any]) -> list[tuple[str, Path]]:
    text = _text_config(config)
    layer = int((text.get("ple_layer_ids") or [2])[0]) - 1  # one-indexed in the HF config
    weight_map = _load_json(source / "model.safetensors.index.json")["weight_map"]
    shards: list[tuple[str, Path]] = []
    index = 0
    while (key := NGRAM_KEY.format(layer=layer, index=index)) in weight_map:
        shards.append((key, source / weight_map[key]))
        index += 1
    return shards


def write_ngram_sidecar(
    source: Path,
    destination: Path,
    *,
    bits: int,
    group: int,
    progress: Progress = _noop_progress,
) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    config = _load_json(source / "config.json")
    shards = ngram_shards(source, config)
    if not shards:
        raise Qwen4ForgeError("source carries no n-gram embedding shards")
    rows_per: list[int] = []
    dim = 0
    for key, path in shards:
        header, _ = read_safetensors_header(path)
        rows_per.append(int(header[key]["shape"][0]))
        dim = int(header[key]["shape"][1])
    rows = sum(rows_per)
    if (bits, group) != NGRAM_PRODUCTION_LAYOUT:
        _log(
            f"n-gram sidecar {bits}-bit/g{group} is outside the production layout "
            f"{NGRAM_PRODUCTION_LAYOUT}; the fixed-M4 verify lane will refuse it"
        )
    header = ngram_sidecar_layout(rows, dim, bits=bits, group=group)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    data_start = 8 + len(encoded)
    total = data_start + max(info["data_offsets"][1] for k, info in header.items() if k != "__metadata__")
    out = destination / NGRAM_FILE
    partial = out.with_suffix(".partial")
    destination.mkdir(parents=True, exist_ok=True)
    _log(f"n-gram table: {len(shards)} shards, {rows} rows x {dim} -> {bits}-bit/g{group}, {total / 2**30:.1f} GiB")
    with partial.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.truncate(total)
    maps = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        np_dtype = np.uint32 if info["dtype"] == "U32" else np.uint16
        maps[name] = np.memmap(partial, mode="r+", dtype=np_dtype,
                               offset=data_start + info["data_offsets"][0], shape=tuple(info["shape"]))
    row = 0
    started = time.monotonic()
    for n, ((key, path), count) in enumerate(zip(shards, rows_per)):
        # Read the shard into RAM first. mx.load hands back a file-backed
        # (mmap) array, so the quantize kernel would page the shard in from
        # the source disk *inside* its command buffer; a cold or throttled
        # external SSD then trips the Metal GPU watchdog (seen at shard 27/128
        # on a Thunderbolt NVMe after the page cache was churned).
        header, data_start = read_safetensors_header(path)
        info = header[key]
        raw = np.fromfile(
            path, dtype=np.uint16, count=count * dim,
            offset=data_start + int(info["data_offsets"][0]),
        ).reshape(count, dim)
        table = mx.array(raw).view(mx.bfloat16)
        if bits == 0:
            maps["ngram.weight"][row:row + count] = np.asarray(table.view(mx.uint16))
        else:
            q, s, b = mx.quantize(table, group_size=group, bits=bits)
            mx.eval(q, s, b)
            maps["ngram.weight"][row:row + count] = np.asarray(q)
            maps["ngram.scales"][row:row + count] = np.asarray(s.astype(mx.bfloat16).view(mx.uint16))
            maps["ngram.biases"][row:row + count] = np.asarray(b.astype(mx.bfloat16).view(mx.uint16))
            del q, s, b
        del table
        mx.clear_cache()
        row += count
        progress("convert", 0.30 * (n + 1) / len(shards), "ngram_sidecar", False)
    for m in maps.values():
        m.flush()
    del maps
    os.replace(partial, out)
    _log(f"n-gram table written in {time.monotonic() - started:.0f}s")
    return {"rows": rows, "dim": dim, "bits": bits, "group_size": group, "bytes": total}


# --------------------------------------------------------------------------- body
def convert_body(
    source: Path,
    destination: Path,
    *,
    recipe: dict[str, Any],
    body_bits: int,
    body_group: int,
    shard_bytes: int = 4 * 2**30,
    progress: Progress = _noop_progress,
) -> dict[str, Any]:
    """Construct the MTPLX backend, sanitize, quantize, and stream shards out."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map, tree_map_with_path
    from mlx_lm.utils import quantize_model

    from mtplx.models.qwen4_exp import Model, ModelArgs

    config = _load_json(source / "config.json")
    config.setdefault("text_config", {})["ngram_sidecar"] = True
    config.pop("model_file", None)
    model = Model(ModelArgs.from_dict(config))

    weight_map = _load_json(source / "model.safetensors.index.json")["weight_map"]
    weights: dict[str, Any] = {}
    for filename in sorted(set(weight_map.values())):
        weights.update(mx.load(str(source / filename)))
    weights = model.sanitize(weights)
    dtype_name = config.get("torch_dtype") or _text_config(config).get("dtype")
    if dtype_name in ("float16", "bfloat16", "float32"):
        dtype = getattr(mx, dtype_name)
        cast_ok = getattr(model, "cast_predicate", lambda _: True)
        weights = {
            k: (v.astype(dtype) if cast_ok(k) and mx.issubdtype(v.dtype, mx.floating) else v)
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    del weights

    model_predicate = getattr(model, "quant_predicate", None)
    overrides = None
    if recipe.get("module_overrides"):
        from mtplx.commands.forge_mixed_convert import build_predicate

        overrides = build_predicate(recipe)

    def predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        if overrides is not None:
            decision = overrides(path, module)
            if decision is not True:
                return decision
        return model_predicate(path, module) if model_predicate is not None else True

    progress("convert", 0.35, "quantize_body", False)
    model, config = quantize_model(model, config, body_group, body_bits, mode="affine", quant_predicate=predicate)
    params = tree_flatten(model.parameters())
    model.update(tree_map(lambda _: mx.array([]), model.parameters()))
    del model
    total_parameters = sum(int(v.size) for _, v in params)

    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0
    for key, value in params:
        if current and current_bytes + value.nbytes > shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = value
        current_bytes += value.nbytes
    if current:
        shards.append(current)
    del params

    destination.mkdir(parents=True, exist_ok=True)
    count = len(shards)
    weight_index: dict[str, str] = {}
    total_size = 0
    started = time.monotonic()
    for i in range(count):
        shard = shards[i]
        shards[i] = None
        for value in shard.values():
            mx.eval(value)  # one tensor per command buffer: bounded memory, no watchdog
        name = f"model-{i + 1:05d}-of-{count:05d}.safetensors" if count > 1 else "model.safetensors"
        mx.save_safetensors(str(destination / name), shard, metadata={"format": "mlx"})
        total_size += sum(int(v.nbytes) for v in shard.values())
        for key in shard:
            weight_index[key] = name
        del shard
        mx.clear_cache()
        progress("convert", 0.35 + 0.60 * (i + 1) / count, "to_mlx", False)
        _log(f"body shard {i + 1}/{count} ({total_size / 2**30:.1f} GiB, {time.monotonic() - started:.0f}s)")
    index = {
        "metadata": {"total_size": total_size, "total_parameters": total_parameters},
        "weight_map": {k: weight_index[k] for k in sorted(weight_index)},
    }
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=4), encoding="utf-8")
    config = dict(sorted(config.items()))
    (destination / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    for item in source.iterdir():
        if item.name.startswith(".") or item.name.startswith("model") or item.name == "config.json":
            continue
        if item.suffix in {".json", ".jinja", ".txt", ".md"} and item.is_file():
            shutil.copy2(item, destination / item.name)
    return {"shards": count, "bytes": total_size, "parameters": total_parameters}


# --------------------------------------------------------------------------- MTP sidecar
def mtp_module_rule(name: str, *, mtp_bits: int, mtp_group: int, qsa_8bit: bool = False):
    """``(bits, group)`` for a stripped (``mtp.``-less) weight key, or None for bf16."""
    if "switch_mlp." in name:
        return (mtp_bits, mtp_group) if mtp_bits else None
    if any(name.endswith(s + ".weight") for s in MTP_EIGHT_BIT_SUFFIXES):
        return (8, 64)
    if any(name.endswith(s + ".weight") for s in QSA_PROJ_SUFFIXES):
        if qsa_8bit:
            return (8, 64)
        return (mtp_bits, mtp_group) if mtp_bits else None
    return None


def mtp_layout_keys(raw_keys: list[str]) -> list[str]:
    """Output key set for a raw HF ``mtp.`` key list (pure; unit-tested)."""
    out: list[str] = []
    for key in raw_keys:
        stripped = key[len("mtp."):] if key.startswith("mtp.") else key
        if stripped.endswith(".mlp.experts.gate_up_proj"):
            prefix = stripped.rsplit(".mlp.experts.", 1)[0]
            out += [f"{prefix}.mlp.switch_mlp.gate_proj.weight", f"{prefix}.mlp.switch_mlp.up_proj.weight"]
        elif stripped.endswith(".mlp.experts.down_proj"):
            prefix = stripped.rsplit(".mlp.experts.", 1)[0]
            out.append(f"{prefix}.mlp.switch_mlp.down_proj.weight")
        else:
            out.append(stripped)
    return out


def write_mtp_sidecar(
    source: Path,
    destination: Path,
    *,
    mtp_bits: int,
    mtp_group: int,
    qsa_8bit: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx

    config = _load_json(source / "config.json")
    hidden = int(_text_config(config)["hidden_size"])
    weight_map = _load_json(source / "model.safetensors.index.json")["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key.startswith("mtp."):
            by_file.setdefault(filename, []).append(key)
    if not by_file:
        return {"written": False}
    raw: dict[str, Any] = {}
    for filename, keys in by_file.items():
        loaded = mx.load(str(source / filename))
        for key in keys:
            raw[key[len("mtp."):]] = loaded[key]
    tensors: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key.rsplit(".mlp.experts.", 1)[0]
            if value.shape[1] == hidden:  # transformers bmm layout [E, hidden, 2*inter]
                gate, up = mx.split(value, 2, axis=-1)
                gate, up = gate.swapaxes(1, 2), up.swapaxes(1, 2)
            else:  # hub Linear layout [E, 2*inter, hidden]
                gate, up = mx.split(value, 2, axis=1)
            tensors[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
            tensors[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
            continue
        if key.endswith(".mlp.experts.down_proj"):
            prefix = key.rsplit(".mlp.experts.", 1)[0]
            if value.shape[2] == hidden:
                value = value.swapaxes(1, 2)
            tensors[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = value
            continue
        if value.ndim == 1 and any(key.endswith(s) for s in MTP_NORM_SHIFT_SUFFIXES):
            value = (value.astype(mx.float32) + 1.0).astype(value.dtype)
        tensors[key] = value
    packed: dict[str, Any] = {}
    modules: dict[str, str] = {}
    for key, value in tensors.items():
        rule = mtp_module_rule(key, mtp_bits=mtp_bits, mtp_group=mtp_group, qsa_8bit=qsa_8bit)
        if rule is None or value.ndim < 2:
            packed["mtp." + key] = value.astype(mx.bfloat16) if value.dtype == mx.float32 else value
            modules[key] = "bf16"
            continue
        bits, group = rule
        q, s, b = mx.quantize(value.astype(mx.bfloat16), group_size=group, bits=bits)
        base = "mtp." + key[: -len(".weight")]
        packed[f"{base}.weight"], packed[f"{base}.scales"], packed[f"{base}.biases"] = q, s, b
        modules[key] = f"{bits}b/g{group}"
    mx.eval(list(packed.values()))
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / MTP_FILE), packed, metadata={"format": "mlx"})
    size = (destination / MTP_FILE).stat().st_size
    _log(f"MTP sidecar written ({size / 2**30:.2f} GiB, {len(packed)} tensors)")
    return {"written": True, "bytes": size, "modules": modules}


# --------------------------------------------------------------------------- lane
def run_lane(
    source: Path,
    destination: Path,
    *,
    recipe: dict[str, Any],
    progress: Progress = _noop_progress,
) -> dict[str, Any]:
    """Body + n-gram sidecar + MTP sidecar. Vision, validation, verify and
    stamping are the caller's ordinary steps."""
    params = recipe_params(recipe)
    qsa_8bit = bool(recipe.get("qwen4_qsa_8bit", False))
    report: dict[str, Any] = {"recipe": params}
    progress("convert", 0.0, "ngram_sidecar", False)
    report["ngram"] = write_ngram_sidecar(
        source, destination, bits=params["ngram_bits"], group=params["ngram_group"], progress=progress
    )
    body_recipe = dict(recipe)
    if qsa_8bit:
        body_recipe.setdefault("module_overrides", [])
        body_recipe["module_overrides"] = [
            *[o for o in body_recipe["module_overrides"] if isinstance(o, dict)],
            *[{"suffix": s, "bits": 8, "group_size": 64} for s in QSA_PROJ_SUFFIXES],
        ]
    report["body"] = convert_body(
        source, destination, recipe=body_recipe,
        body_bits=params["body_bits"], body_group=params["body_group"], progress=progress,
    )
    progress("convert", 0.97, "extract_mtp", False)
    report["mtp"] = write_mtp_sidecar(
        source, destination, mtp_bits=params["mtp_bits"], mtp_group=params["mtp_group"], qsa_8bit=qsa_8bit
    )
    config_path = destination / "config.json"
    config = _load_json(config_path)
    config.setdefault("text_config", {})["ngram_sidecar"] = True
    extra = config.get("mlx_lm_extra_tensors") if isinstance(config.get("mlx_lm_extra_tensors"), dict) else {}
    extra["ngram_file"] = NGRAM_FILE
    if report["mtp"].get("written"):
        extra["mtp_file"] = MTP_FILE
    config["mlx_lm_extra_tensors"] = extra
    config["mtplx_recipe"] = {
        "base": {"bits": params["body_bits"], "group_size": params["body_group"]},
        "ngram": {"bits": params["ngram_bits"], "group_size": params["ngram_group"]},
        "mtp": {"bits": params["mtp_bits"], "group_size": params["mtp_group"]},
        "qsa_8bit": qsa_8bit,
        "lane": "qwen4_exp",
    }
    config_path.write_text(json.dumps(dict(sorted(config.items())), indent=4), encoding="utf-8")
    progress("convert", 1.0, "to_mlx", True)
    return report
