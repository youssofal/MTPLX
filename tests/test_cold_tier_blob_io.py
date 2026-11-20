"""SSD cold-tier blob I/O: parallel reads into caller buffers, cache bypass.

MLX-free. The codec side (one buffer per tensor, decoded identically to the
blob-by-blob path) is covered in test_cold_tier_parallel_restore.py.
"""

from __future__ import annotations

import os
import threading

import pytest

from mtplx.cache_bank import blob_io


def _blob(tmp_path, name: str, nbytes: int, seed: int):
    data = bytes((seed + i * 31) % 251 for i in range(nbytes))
    path = tmp_path / name
    path.write_bytes(data)
    return path, data


def test_reads_every_file_into_its_slice_of_one_buffer(tmp_path):
    blobs = [_blob(tmp_path, f"b{i}.bin", 1000 + i * 17, i) for i in range(12)]
    buffer = blob_io.allocate(sum(len(data) for _, data in blobs))
    items, offset = [], 0
    for path, data in blobs:
        items.append((path, buffer[offset : offset + len(data)]))
        offset += len(data)
    blob_io.read_into_many(items)
    assert bytes(buffer) == b"".join(data for _, data in blobs)


def test_large_blob_is_read_as_parallel_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(blob_io, "READ_RANGE_BYTES", 4096)
    path, data = _blob(tmp_path, "big.bin", 4096 * 5 + 123, 7)
    seen: list[tuple[int, int]] = []
    lock = threading.Lock()
    real = blob_io._read_range

    def spy(path, view, offset, total):
        with lock:
            seen.append((offset, len(view)))
        real(path, view, offset, total)

    monkeypatch.setattr(blob_io, "_read_range", spy)
    buffer = blob_io.allocate(len(data))
    blob_io.read_into_many([(path, buffer)])
    assert bytes(buffer) == data
    assert sorted(seen) == [(0, 4096), (4096, 4096), (8192, 4096), (12288, 4096), (16384, 4096), (20480, 123)]


def test_single_thread_setting_reads_inline(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_READ_THREADS", "1")
    blobs = [_blob(tmp_path, f"b{i}.bin", 300, i) for i in range(3)]
    buffer = blob_io.allocate(900)
    blob_io.read_into_many(
        (path, buffer[i * 300 : (i + 1) * 300]) for i, (path, _) in enumerate(blobs)
    )
    assert bytes(buffer) == b"".join(data for _, data in blobs)


def test_missing_blob_raises_file_not_found(tmp_path):
    present, _ = _blob(tmp_path, "a.bin", 64, 1)
    buffer = blob_io.allocate(128)
    with pytest.raises(FileNotFoundError):
        blob_io.read_into_many(
            [(present, buffer[:64]), (tmp_path / "gone.bin", buffer[64:])]
        )


@pytest.mark.parametrize("on_disk", [63, 65])
def test_size_mismatch_raises(tmp_path, on_disk):
    path, _ = _blob(tmp_path, "a.bin", on_disk, 1)
    with pytest.raises(ValueError, match="size mismatch"):
        blob_io.read_into_many([(path, blob_io.allocate(64))])


def test_read_threads_parsing(monkeypatch):
    monkeypatch.delenv("MTPLX_SSD_READ_THREADS", raising=False)
    assert blob_io.read_threads() == blob_io.DEFAULT_READ_THREADS
    for raw, expected in (("4", 4), ("0", 1), ("999", 64), ("x", blob_io.DEFAULT_READ_THREADS)):
        monkeypatch.setenv("MTPLX_SSD_READ_THREADS", raw)
        assert blob_io.read_threads() == expected


def test_write_file_matches_write_bytes(tmp_path):
    data = os.urandom(70_000)
    path = tmp_path / "out.bin"
    path.write_bytes(b"stale content that is longer than nothing" * 4000)
    blob_io.write_file(path, data)
    assert path.read_bytes() == data
    with pytest.raises(FileNotFoundError):
        blob_io.write_file(tmp_path / "no-such-dir" / "x.bin", data)


def test_nocache_defaults_to_writes_only_on_darwin(monkeypatch):
    monkeypatch.setattr(blob_io.sys, "platform", "linux")
    monkeypatch.setenv("MTPLX_SSD_NOCACHE", "all")
    assert not blob_io.nocache_enabled("read")
    assert not blob_io.nocache_enabled("write")
    monkeypatch.setattr(blob_io.sys, "platform", "darwin")
    monkeypatch.delenv("MTPLX_SSD_NOCACHE", raising=False)
    assert blob_io.nocache_mode() == "write"
    assert blob_io.nocache_enabled("write")
    assert not blob_io.nocache_enabled("read")


@pytest.mark.parametrize(
    ("value", "read", "write"),
    [
        ("write", False, True),
        ("read", True, False),
        ("all", True, True),
        ("1", True, True),
        ("0", False, False),
        ("off", False, False),
        ("bogus", False, True),
    ],
)
def test_nocache_side_is_switchable(monkeypatch, value, read, write):
    monkeypatch.setattr(blob_io.sys, "platform", "darwin")
    monkeypatch.setenv("MTPLX_SSD_NOCACHE", value)
    assert blob_io.nocache_enabled("read") is read
    assert blob_io.nocache_enabled("write") is write


def test_nocache_hint_sets_f_nocache_on_writes_only_by_default(monkeypatch, tmp_path):
    fcntl = pytest.importorskip("fcntl")
    calls = []
    monkeypatch.setattr(blob_io.sys, "platform", "darwin")
    monkeypatch.delenv("MTPLX_SSD_NOCACHE", raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, arg: calls.append((cmd, arg)))
    blob_io.write_file(tmp_path / "x.bin", b"abc")
    assert calls == [(getattr(fcntl, "F_NOCACHE", 48), 1)]
    calls.clear()
    out = memoryview(bytearray(3))
    blob_io.read_into_many([(tmp_path / "x.bin", out)])
    assert bytes(out) == b"abc"
    assert calls == []


def test_staging_buffer_is_reused_until_released():
    staging = blob_io.StagingBuffer()
    first = staging.allocate(100)
    first[:3] = b"abc"
    again = staging.allocate(50)
    assert len(again) == 50 and bytes(again[:3]) == b"abc"
    grown = staging.allocate(10_000)
    assert len(grown) == 10_000
    staging.release()
    after = staging.allocate(10)
    assert len(after) == 10 and staging._buffer is None
