"""Convert the output vocabulary while preserving the native expert, loop, vision and draft tensors."""

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .forge_inkling import (
    SIDECARS,
    WEIGHTS,
    _copy_weight,
    stage_native_pack,
    verify_source,
    write_pack_metadata,
)

HEAD = "model.llm.unembed.weight"


def matrix_layout(name: str, recipe: str) -> tuple[int, int] | None:
    return (8, 32) if recipe in {"speed", "native-quality"} and name == HEAD else None


def _copy_header_tensors(source, dest, header, names, data_offset):
    packed, size = {}, 0
    for key in names:
        entry = header[key]
        start, end = entry["data_offsets"]
        packed[key] = {**entry, "data_offsets": [size, size + end - start]}
        size += end - start
    raw = json.dumps(packed, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with source.open("rb") as inp, dest.open("wb") as out:
        out.write(len(raw).to_bytes(8, "little"))
        out.write(raw)
        for key in names:
            start, end = header[key]["data_offsets"]
            inp.seek(data_offset + start)
            remaining = end - start
            while remaining:
                data = inp.read(min(remaining, 8 * 1024**2))
                if not data:
                    raise ValueError(f"Truncated tensor {key}")
                out.write(data)
                remaining -= len(data)


def build_pack(source: Path, destination: Path, recipe: str) -> dict:
    if recipe == "quality":
        return stage_native_pack(source, destination, recipe=recipe)
    if recipe not in {"speed", "native-quality"}:
        raise ValueError(f"Unknown Inkling recipe {recipe!r}")
    import mlx.core as mx

    source, destination = source.resolve(), destination.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    config, header_size, header = verify_source(source)
    target_name, draft_name = WEIGHTS
    source_target = source / target_name
    bits, group_size = matrix_layout(HEAD, recipe)
    if header[HEAD]["dtype"] != "BF16":
        raise ValueError("Expected the published BF16 output vocabulary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=destination.parent, prefix=".inkling-stage-"
    ) as scratch:
        staged = Path(scratch) / "pack"
        staged.mkdir()
        native_keys = [key for key in header if key != HEAD]
        _copy_header_tensors(
            source_target,
            staged / "model-native.safetensors",
            header,
            native_keys,
            8 + header_size,
        )
        source_head = mx.load(str(source_target))[HEAD]
        codes, scales, biases = mx.quantize(
            source_head, group_size=group_size, bits=bits
        )
        quantized = {
            HEAD: codes,
            HEAD + ".mtplx_scales": scales,
            HEAD + ".mtplx_biases": biases,
        }
        mx.save_safetensors(str(staged / "model-head.safetensors"), quantized)
        index = {
            **dict.fromkeys(native_keys, "model-native.safetensors"),
            **dict.fromkeys(quantized, "model-head.safetensors"),
        }
        config["mtplx_inkling_precision"] = {
            HEAD: {"bits": bits, "group_size": group_size}
        }
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staged / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index}, indent=2) + "\n"
        )
        _copy_weight(source / draft_name, staged / draft_name)
        for name in SIDECARS:
            if name in {"config.json", "model.safetensors.index.json"}:
                continue
            dst = staged / ("UPSTREAM.md" if name == "README.md" else name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
        manifest = write_pack_metadata(staged, "speed")
        os.rename(staged, destination)
    return manifest
