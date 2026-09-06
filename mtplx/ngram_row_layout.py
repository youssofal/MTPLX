"""Optional bit-exact interleaved n-gram cache (David Tai's proposal, #449).

Create with ``python -m mtplx.ngram_row_layout SOURCE --out CACHE``. The original
pack stays intact. Set MTPLX_NGRAM_ROW_FILE to CACHE to use it; unset to revert.
This local derived cache is bound to the exact source file identity, not a
portable replacement for a published model pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import uuid

import numpy as np

DTYPES = {"U32": np.dtype("<u4"), "BF16": np.dtype("<u2"), "F16": np.dtype("<u2")}
FORMAT = "mtplx_ngram_rows_v1"


def header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("N-gram header exceeds 16 MiB")
        return json.loads(f.read(size)), 8 + size


def identity(path: Path) -> dict:
    stat = path.stat()
    h, _ = header(path)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "header_sha256": hashlib.sha256(
            json.dumps(h, sort_keys=True).encode()
        ).hexdigest(),
    }


def _parts(h: dict) -> list[tuple[str, dict]]:
    return [
        (name, h["ngram." + name])
        for name in ("weight", "scales", "biases")
        if "ngram." + name in h
    ]


def open_rows(cache: Path, source: Path, entries: dict):
    """Return the same raw matrices with a shared record stride and one IO span."""
    h, start = header(cache)
    meta = h.get("__metadata__", {})
    if meta.get("format") != FORMAT or json.loads(meta["source"]) != identity(source):
        raise ValueError(
            "Interleaved n-gram cache is stale or incompatible; rebuild it from this pack"
        )
    parts = json.loads(meta["parts"])
    expected = {name: info for name, (info, _) in entries.items()}
    if dict(parts) != expected:
        raise ValueError("Interleaved n-gram cache geometry differs from the source")
    rows, stride = h["ngram.records"]["shape"]
    if cache.stat().st_size != start + rows * stride:
        raise ValueError("Interleaved n-gram cache is incomplete")
    mm = np.memmap(cache, mode="r", dtype=np.uint8, offset=start, shape=(rows, stride))
    from mtplx.ple_row_gather import madvise_choice

    try:
        mm._mmap.madvise(madvise_choice()[1])
    except (AttributeError, OSError, ValueError):
        pass
    maps = {}
    offset = 0
    for name, info in parts:
        dtype = DTYPES[info["dtype"]]
        shape = tuple(info["shape"])
        if shape[0] != rows or len(shape) != 2:
            raise ValueError("Invalid n-gram matrix shape")
        maps[name] = (
            np.ndarray(
                shape,
                dtype=dtype,
                buffer=mm,
                offset=offset,
                strides=(stride, dtype.itemsize),
            ),
            info["dtype"],
        )
        offset += shape[1] * dtype.itemsize
    if offset != stride:
        raise ValueError("Invalid interleaved row stride")
    return maps, (start, stride)


def convert(source: Path, output: Path, *, chunk_rows: int = 262144) -> dict:
    """Bounded-memory conversion; compare every row bit-for-bit before activation."""
    source = source.resolve()
    output = output.absolute()
    if output.exists() or output == source:
        raise FileExistsError(
            "Choose a new output path; existing files are never overwritten"
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    original = identity(source)
    h, start = header(source)
    parts = _parts(h)
    if not parts:
        raise ValueError("Source contains no n-gram table")
    maps = {}
    rows = parts[0][1]["shape"][0]
    stride = 0
    for name, info in parts:
        dtype = DTYPES[info["dtype"]]
        shape = tuple(info["shape"])
        if len(shape) != 2 or shape[0] != rows:
            raise ValueError("N-gram matrices must have identical row counts")
        maps[name] = np.memmap(
            source,
            mode="r",
            dtype=dtype,
            offset=start + info["data_offsets"][0],
            shape=shape,
        )
        stride += shape[1] * dtype.itemsize
    meta = {
        "format": FORMAT,
        "source": json.dumps(original),
        "parts": json.dumps(parts),
    }
    encoded = json.dumps(
        {
            "__metadata__": meta,
            "ngram.records": {
                "dtype": "U8",
                "shape": [rows, stride],
                "data_offsets": [0, rows * stride],
            },
        }
    ).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    began = time.monotonic()
    with stage.open("xb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.truncate(8 + len(encoded) + rows * stride)
    dest = np.memmap(
        stage, mode="r+", dtype=np.uint8, offset=8 + len(encoded), shape=(rows, stride)
    )
    for lo in range(0, rows, chunk_rows):
        hi = min(rows, lo + chunk_rows)
        offset = 0
        for name, info in parts:
            source_rows = np.ascontiguousarray(maps[name][lo:hi]).view(np.uint8)
            width = source_rows.shape[1]
            dest[lo:hi, offset : offset + width] = source_rows
            if not np.array_equal(dest[lo:hi, offset : offset + width], source_rows):
                raise ValueError(f"Bit parity failed in {name} at row {lo}")
            offset += width
    dest.flush()
    with stage.open("rb") as f:
        os.fsync(f.fileno())
    if identity(source) != original:
        raise ValueError(
            "Source changed during conversion; incomplete output retained for inspection"
        )
    if output.exists():
        raise FileExistsError(output)
    stage.rename(output)
    return {
        "rows": rows,
        "row_bytes": stride,
        "bytes": rows * stride,
        "elapsed_s": time.monotonic() - began,
        "all_rows_bit_exact": True,
        "output": str(output),
        "source": original,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
