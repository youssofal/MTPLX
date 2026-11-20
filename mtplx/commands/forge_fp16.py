"""The FP16 sibling of a finished pack: the same model for M1 and M2 Macs, which have no native bfloat16.

The released siblings' policy (2.7.0 release notes): every bfloat16 tensor is cast to float16, every other
tensor keeps its bytes. So every packed integer weight is the parent's byte for byte, the float32 state is
the parent's, and a bfloat16 value differs only where float16 cannot hold it: below float16's normal range,
where it rounds on the subnormal grid (to nearest, ties to even, the cast MLX makes) or to zero. A value
beyond float16's range would become infinite; the sibling is then refused, naming the tensor, because it
would be another model.

Bfloat16 and float16 are both two bytes, so every tensor keeps its offset: each file is first a clone of the
parent's (APFS shares the blocks, so a packed table of tens of GB costs no disk), then its bfloat16 regions
are rewritten in place, streamed in chunks, and its header names the new type. The vision tower is cast like
every other bfloat16 tensor (Law 1: BF16 in a pack, F16 in its FP16 sibling).

    python -m mtplx.commands.forge_fp16 <parent pack> <destination> [--public-model-id ID]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

MANIFEST = "MTPLX_FP16_CONVERSION_MANIFEST.json"
POLICY = "convert bf16 floating tensors to fp16; preserve packed integer/quantized tensors"
#: Elements converted per chunk (32 MiB of float32 working memory at the default).
CHUNK_ELEMENTS = 1 << 23
_FLOAT_TYPE_KEYS = ("torch_dtype", "dtype")


class Fp16SiblingError(RuntimeError):
    """The sibling would not be the parent's model; the message names why."""


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    """The safetensors header: its byte length and its JSON object."""

    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        return length, json.loads(stream.read(length))


def bf16_to_f16(bits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Bfloat16 bit patterns to float16 bit patterns, with what the cast changed.

    Bfloat16 to float32 is exact (the high half of a float32); float32 to float16 rounds to nearest, ties to
    even, through the subnormals: the cast MLX makes (``tests/test_forge_fp16.py`` checks every one of the
    65,536 bfloat16 patterns against it)."""

    wide = (bits.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        half = wide.astype(np.float16)
        back = half.astype(np.float32)
    finite = np.isfinite(wide)
    overflow = finite & ~np.isfinite(half)
    exact = (back == wide) | (np.isnan(wide) & np.isnan(back))
    changed = finite & ~overflow & ~exact
    error = np.abs(back[changed] - wide[changed]) if changed.any() else np.zeros(0, np.float32)
    stats = {
        "elements": int(bits.size),
        "exact": int(exact.sum()),
        "rounded": int(changed.sum()),
        "to_zero": int((changed & (half == 0)).sum()),
        "overflow": int(overflow.sum()),
        "not_finite_in_parent": int((~finite).sum()),
        "largest_error": float(error.max()) if error.size else 0.0,
        "largest_magnitude": float(np.abs(wide[finite]).max()) if finite.any() else 0.0,
    }
    return half.view(np.uint16), stats


def _merge(total: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if key.startswith("largest"):
            total[key] = max(float(total.get(key, 0.0)), float(value))
        else:
            total[key] = int(total.get(key, 0)) + int(value)


def _clone(source: Path, destination: Path) -> None:
    """A copy-on-write clone where the volume allows it, else a plain copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        done = subprocess.run(["cp", "-c", str(source), str(destination)], capture_output=True)
        if done.returncode == 0:
            return
    shutil.copy2(source, destination)


def convert_file(source: Path, destination: Path, *, chunk: int = CHUNK_ELEMENTS) -> dict[str, Any]:
    """One safetensors file: its bfloat16 tensors cast, every other byte the parent's."""

    length, header = read_header(source)
    names = sorted(
        (name for name, info in header.items() if name != "__metadata__" and info.get("dtype") == "BF16"),
        key=lambda name: header[name]["data_offsets"][0],
    )
    _clone(source, destination)
    report: dict[str, Any] = {"name": destination.name, "tensors": len(header) - ("__metadata__" in header)}
    report["converted"], report["preserved"] = len(names), report["tensors"] - len(names)
    totals: dict[str, Any] = {}
    if names:
        start = 8 + length
        with destination.open("r+b") as out:
            for name in names:
                first, last = (int(offset) for offset in header[name]["data_offsets"])
                stats: dict[str, Any] = {}
                if last > first:
                    values = np.memmap(source, dtype=np.uint16, mode="r", offset=start + first, shape=((last - first) // 2,))
                    for at in range(0, values.size, chunk):
                        half, part = bf16_to_f16(np.asarray(values[at : at + chunk]))
                        _merge(stats, part)
                        out.seek(start + first + 2 * at)
                        out.write(half.tobytes())
                    del values
                if stats.get("overflow"):
                    raise Fp16SiblingError(
                        f"{source.name}: {stats['overflow']} value(s) of {name} are beyond float16's range "
                        f"(largest magnitude {stats['largest_magnitude']:.4g}); an FP16 sibling would change them"
                    )
                _merge(totals, stats)
                header[name]["dtype"] = "F16"
            encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > length:  # never seen: a shorter type name cannot lengthen a compact header
                raise Fp16SiblingError(f"{source.name}: the new header ({len(encoded)} bytes) outgrows the old ({length})")
            out.seek(8)
            out.write(encoded + b" " * (length - len(encoded)))
    _, written = read_header(destination)
    left = sorted(name for name, info in written.items() if name != "__metadata__" and info.get("dtype") == "BF16")
    if left:
        raise Fp16SiblingError(f"{destination.name}: {len(left)} bfloat16 tensor(s) left, first {left[0]}")
    report["values"] = totals
    return report


def _patch_config(path: Path) -> bool:
    """The float type a config names, in the model and in each of its parts, is the sibling's."""

    config = json.loads(path.read_text(encoding="utf-8"))
    touched = False
    for section in (config, config.get("text_config"), config.get("vision_config")):
        if not isinstance(section, dict):
            continue
        for key in _FLOAT_TYPE_KEYS:
            if str(section.get(key, "")) == "bfloat16":
                section[key] = "float16"
                touched = True
    if touched:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return touched


def _patch_stamp(path: Path, *, parent: Path, public_model_id: str | None) -> None:
    stamp = json.loads(path.read_text(encoding="utf-8"))
    stamp["precision_variant"] = "fp16"
    stamp["precision_policy"] = {
        "variant": "fp16",
        "intended_default_for": ["m1", "m2"],
        "routing": "mtplx start auto-selects this artifact on M1/M2 Apple Silicon",
        "note": (
            f"FP16 precision sibling of {parent.name}: byte-identical INT packs, bf16 floats cast to fp16 for "
            "M1/M2 Macs. Same tuned depth and draft-sampler stamps as the parent; not a separate speed claim."
        ),
    }
    if public_model_id:
        stamp["public_model_id"] = public_model_id
    provenance = stamp.setdefault("forge_provenance", {})
    if isinstance(provenance, dict):
        recipe = provenance.get("forge_recipe")
        if isinstance(recipe, dict):
            recipe["body_dtype"] = "fp16"
        provenance["fp16_sibling_of"] = parent.name
    path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive(
    parent: Path | str,
    destination: Path | str,
    *,
    public_model_id: str | None = None,
    chunk: int = CHUNK_ELEMENTS,
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
) -> dict[str, Any]:
    """Write the FP16 sibling of ``parent`` at ``destination``; returns the manifest.

    The sibling is built beside its destination and moved into place only when every file is written and
    checked, so a refused or interrupted derivation leaves no pack behind."""

    parent, destination = Path(parent).resolve(), Path(destination)
    if not (parent / "config.json").is_file():
        raise Fp16SiblingError(f"{parent} is not a pack (no config.json)")
    if destination.exists():
        raise Fp16SiblingError(f"{destination} exists")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    started = time.monotonic()
    files, copied = [], []
    totals: dict[str, Any] = {}
    try:
        for item in sorted(parent.rglob("*")):
            relative = item.relative_to(parent)
            target = partial / relative
            if item.is_dir() or item.name == MANIFEST:
                continue
            if item.suffix == ".safetensors":
                log(f"[fp16] {relative}")
                report = convert_file(item, target, chunk=chunk)
                report["name"] = str(relative)
                _merge(totals, report["values"])
                files.append(report)
            else:
                _clone(item, target)
                copied.append(str(relative))
        config_patched = _patch_config(partial / "config.json")
        if (partial / "mtplx_runtime.json").is_file():
            _patch_stamp(partial / "mtplx_runtime.json", parent=parent, public_model_id=public_model_id)
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "policy": POLICY,
            "source_path": str(parent),
            "output_path": str(destination),
            "files_copied": copied,
            "safetensors": files,
            "config_float_type_patched": config_patched,
            "summary": {
                "files": len(files),
                "tensors_total": sum(report["tensors"] for report in files),
                "converted_bf16_to_fp16": sum(report["converted"] for report in files),
                "preserved": sum(report["preserved"] for report in files),
                "values": totals,
                "seconds": round(time.monotonic() - started, 1),
            },
        }
        (partial / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    values = totals or {"elements": 0, "exact": 0}
    log(
        f"[fp16] {destination.name}: {manifest['summary']['converted_bf16_to_fp16']} tensors cast, "
        f"{manifest['summary']['preserved']} kept byte for byte; "
        f"{values.get('exact', 0)} of {values.get('elements', 0)} values exact, {values.get('rounded', 0)} rounded "
        f"({values.get('to_zero', 0)} to zero), largest error {values.get('largest_error', 0.0):.3g}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parent", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--public-model-id", default=None)
    args = parser.parse_args(argv)
    try:
        manifest = derive(args.parent.expanduser(), args.destination.expanduser(), public_model_id=args.public_model_id)
    except Fp16SiblingError as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
