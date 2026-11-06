"""Blob file I/O for the SSD cold tier: parallel reads, no page-cache copy.

MLX-free so it runs (and is tested) anywhere.

Reads. A restore used to read its blobs one ``Path.read_bytes()`` at a time
on the model-owner thread: one request in flight against an NVMe that needs
many to reach its bandwidth (pooled ``pread`` reached 12.9 GiB/s against
1.40 GiB/s for single-stream demand faulting, ``ple_row_gather``, measured
2026-07-11). :func:`read_into_many` reads a batch of blobs on a thread pool
(file reads release the GIL), straight into caller-owned buffers, and splits
a large single blob into ranges so it too is read with several requests in
flight.

Page cache. On a Mac the unified buffer cache lives in the same memory as
the model. A 4 GB restore through buffered reads leaves 4 GB of cached file
pages next to the 4 GB of decoded arrays, and the pressure lands on the
model's wired weights; a cold-tier write does the same with data that is
read back, if ever, long after it would have been evicted.

``F_NOCACHE`` (macOS) sends I/O past that cache. By default only writes use
it: on an M5 Pro, cold reads of a real store ran at 3.9 GiB/s through the
cache against 2.6 GiB/s with ``F_NOCACHE`` (8 threads), so restores keep the
cache's read-ahead. ``MTPLX_SSD_NOCACHE`` picks the side: ``write`` (default),
``read``, ``all`` (``1``) or ``off`` (``0``). On other platforms it is a
no-op.
"""

from __future__ import annotations

import mmap
import os
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# fcntl.F_NOCACHE is only exported on macOS builds; 48 is its value in
# <sys/fcntl.h>.
_F_NOCACHE = 48
DEFAULT_READ_THREADS = 8
# A blob larger than twice this is read as several ranges in parallel.
READ_RANGE_BYTES = 4 * 1024 * 1024

_pool: ThreadPoolExecutor | None = None
_pool_workers = 0
_pool_lock = threading.Lock()


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


_NOCACHE_SIDES = {
    "write": frozenset({"write"}),
    "read": frozenset({"read"}),
    "all": frozenset({"read", "write"}),
    "off": frozenset(),
}
_NOCACHE_ALIASES = {
    "1": "all", "true": "all", "yes": "all", "on": "all", "both": "all",
    "0": "off", "false": "off", "no": "off", "none": "off",
}
DEFAULT_NOCACHE = "write"


def nocache_mode() -> str:
    """``MTPLX_SSD_NOCACHE`` as one of write / read / all / off."""
    raw = (os.environ.get("MTPLX_SSD_NOCACHE") or "").strip().lower()
    mode = _NOCACHE_ALIASES.get(raw, raw)
    return mode if mode in _NOCACHE_SIDES else DEFAULT_NOCACHE


def nocache_enabled(side: str) -> bool:
    """Whether ``side`` ("read" or "write") goes past the page cache."""
    return sys.platform == "darwin" and side in _NOCACHE_SIDES[nocache_mode()]


def read_threads() -> int:
    raw = os.environ.get("MTPLX_SSD_READ_THREADS")
    try:
        value = int(str(raw).strip()) if raw is not None and raw.strip() else DEFAULT_READ_THREADS
    except ValueError:
        value = DEFAULT_READ_THREADS
    return max(1, min(64, value))


def bypass_page_cache(fd: int, side: str) -> None:
    """Best effort: a refused hint only costs the page-cache copy again."""
    if not nocache_enabled(side):
        return
    try:
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", _F_NOCACHE), 1)
    except (ImportError, OSError):
        pass


def allocate(nbytes: int) -> memoryview:
    """A zeroed, page-aligned, writable host buffer of ``nbytes``.

    Anonymous mmap memory is page aligned, which lets an ``F_NOCACHE`` read
    land in it without a bounce through the cache.
    """
    if nbytes <= 0:
        return memoryview(bytearray(0))
    return memoryview(mmap.mmap(-1, int(nbytes)))


class StagingBuffer:
    """One host buffer reused for every tensor of a restore.

    The codec copies each tensor out of its buffer (``mx.array``) before it
    asks for the next one, so one buffer, grown to the largest tensor, serves
    them all. A fresh buffer per tensor pays a page fault for every page the
    read touches, which costs as much as the read itself on a warm cache.
    :meth:`release` drops it when the restore is done; later calls (the lazy
    recurrent-boundary loader) get a fresh buffer each and keep none.
    """

    def __init__(self) -> None:
        self._buffer: memoryview | None = None
        self._released = False
        self._lock = threading.Lock()

    def allocate(self, nbytes: int) -> memoryview:
        with self._lock:
            if self._released:
                return allocate(nbytes)
            if self._buffer is None or len(self._buffer) < nbytes:
                self._buffer = None
                self._buffer = allocate(nbytes)
            return self._buffer[:nbytes]

    def release(self) -> None:
        with self._lock:
            self._released = True
            self._buffer = None


def write_file(path: Path, raw: bytes | bytearray | memoryview) -> None:
    """``path.write_bytes(raw)``, past the page cache where enabled.

    Raises what ``write_bytes`` raises (FileNotFoundError for a missing
    parent, OSError for a full disk); a partial file is the caller's to
    remove, as before.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        bypass_page_cache(fd, "write")
        view = memoryview(raw).cast("B")
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)


def _read_range(path: Path, view: memoryview, offset: int, file_nbytes: int) -> None:
    with open(path, "rb", buffering=0) as handle:
        size = os.fstat(handle.fileno()).st_size
        if size != file_nbytes:
            raise ValueError(
                f"SSD blob size mismatch for {path}: {size} bytes on disk, "
                f"{file_nbytes} expected"
            )
        bypass_page_cache(handle.fileno(), "read")
        if offset:
            handle.seek(offset)
        while view:
            count = handle.readinto(view)
            if not count:
                raise ValueError(f"SSD blob {path} ended early at offset {offset}")
            view = view[count:]
            offset += count


def _executor(workers: int) -> ThreadPoolExecutor:
    global _pool, _pool_workers
    with _pool_lock:
        if _pool is None or _pool_workers != workers:
            # The old pool's threads finish what they hold and exit.
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="mtplx-ssd-read"
            )
            _pool_workers = workers
        return _pool


def read_into_many(items: Iterable[tuple[Path, memoryview]]) -> None:
    """Fill every ``view`` with the whole file at its ``path``.

    Each file must be exactly ``len(view)`` bytes. Raises FileNotFoundError
    for a missing file and ValueError for a size mismatch, after every read
    already started has finished, so no thread is still writing into a
    caller's buffer once this returns or raises.
    """
    tasks: list[tuple[Path, memoryview, int, int]] = []
    for path, view in items:
        view = view.cast("B") if view.format != "B" or view.ndim != 1 else view
        total = len(view)
        if total > 2 * READ_RANGE_BYTES:
            for start in range(0, total, READ_RANGE_BYTES):
                end = min(total, start + READ_RANGE_BYTES)
                tasks.append((path, view[start:end], start, total))
        else:
            tasks.append((path, view, 0, total))
    if not tasks:
        return
    workers = read_threads()
    if workers <= 1 or len(tasks) == 1:
        for task in tasks:
            _read_range(*task)
        return
    futures = [_executor(workers).submit(_read_range, *task) for task in tasks]
    error: BaseException | None = None
    for future in futures:
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if error is None:
                error = exc
    if error is not None:
        raise error
