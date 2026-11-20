#!/usr/bin/env python3
"""Read throughput of SSD cold-tier blobs: the old path against the new one.

    old:  one Path.read_bytes() per blob, one after another (the restore
          path before blob_io), through the page cache
    new:  blob_io.read_into_many: a thread pool, straight into one buffer,
          F_NOCACHE on macOS (MTPLX_SSD_NOCACHE, MTPLX_SSD_READ_THREADS)

By default it reads blobs from the real store (~/.mtplx/session-cache or
--store); --synthetic writes a set of 256-token-block-sized files instead.
Cold numbers need an empty page cache: pass --purge (runs `sudo purge`
before every pass) or the old path reads from RAM after the first pass.

    python scripts/bench_ssd_blob_read.py --purge --limit-gib 4
    MTPLX_SSD_READ_THREADS=16 python scripts/bench_ssd_blob_read.py --purge
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.cache_bank import blob_io
from mtplx.cache_bank.reconcile import DEFAULT_COLD_TIER_DIR


def _store_blobs(store: Path, limit: int) -> list[Path]:
    paths = list((store / "blobs").glob("*/*.bin"))
    random.Random(0).shuffle(paths)
    picked, total = [], 0
    for path in paths:
        if total >= limit:
            break
        picked.append(path)
        total += path.stat().st_size
    return picked


def _synthetic_blobs(root: Path, limit: int, blob_bytes: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    chunk = os.urandom(blob_bytes)
    paths = []
    for i in range(max(1, limit // blob_bytes)):
        path = root / f"{i:06d}.bin"
        if not path.exists() or path.stat().st_size != blob_bytes:
            path.write_bytes(chunk)
        paths.append(path)
    return paths


def _purge(enabled: bool) -> None:
    if enabled:
        subprocess.run(["sudo", "purge"], check=True)


def _old(paths: list[Path]) -> int:
    return sum(len(path.read_bytes()) for path in paths)


def _new(paths: list[Path], blobs_per_tensor: int) -> int:
    # As a restore does it: one tensor's blobs per call, one staging buffer
    # reused across the tensors of the pass.
    staging = blob_io.StagingBuffer()
    total = 0
    try:
        for start in range(0, len(paths), blobs_per_tensor):
            group = paths[start : start + blobs_per_tensor]
            sizes = [path.stat().st_size for path in group]
            buffer = staging.allocate(sum(sizes))
            items, offset = [], 0
            for path, size in zip(group, sizes):
                items.append((path, buffer[offset : offset + size]))
                offset += size
            blob_io.read_into_many(items)
            total += offset
    finally:
        staging.release()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=Path(DEFAULT_COLD_TIER_DIR).expanduser())
    parser.add_argument("--synthetic", type=Path, help="write and read test blobs in this directory")
    parser.add_argument("--blob-kib", type=int, default=512, help="synthetic blob size")
    parser.add_argument("--limit-gib", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--blobs-per-tensor", type=int, default=128,
        help="blobs read per call (one KV tensor's blocks in a restore)",
    )
    parser.add_argument("--purge", action="store_true", help="sudo purge before every pass")
    args = parser.parse_args()

    limit = int(args.limit_gib * 1024**3)
    if args.synthetic:
        paths = _synthetic_blobs(args.synthetic, limit, args.blob_kib * 1024)
    else:
        paths = _store_blobs(args.store, limit)
    if not paths:
        print(f"no blobs under {args.store}/blobs; use --synthetic DIR", file=sys.stderr)
        return 1
    print(
        f"{len(paths)} blobs, threads={blob_io.read_threads()}, "
        f"nocache={blob_io.nocache_mode()}, purge={args.purge}"
    )
    for round_index in range(args.rounds):
        for label, run in (
            ("old", _old),
            ("new", lambda paths: _new(paths, max(1, args.blobs_per_tensor))),
        ):
            _purge(args.purge)
            started = time.perf_counter()
            nbytes = run(paths)
            seconds = time.perf_counter() - started
            print(
                f"round {round_index + 1} {label}: {nbytes / 1024**3:.2f} GiB "
                f"in {seconds:.3f} s = {nbytes / 1024**3 / seconds:.2f} GiB/s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
