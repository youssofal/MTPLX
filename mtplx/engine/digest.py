"""Digests of a large array's bytes, chunk by chunk, on the GPU.

A first-use check that runs a program on the conversation's own state banks
instead of private copies of them (``ReplaySite``, in-place check) has to
show that the program changed nothing outside the rows it was meant to
write. Copying a bank to compare it is what that check exists to avoid (at
262K tokens the banks are 7.4 GB), so each bank is read once before and once
after, as one 64-bit digest per 32 KiB chunk, and only the chunks that hold
the round's own rows are copied and compared byte for byte.

The digest of a chunk is two 32-bit sums of a mixing function of every
32-bit word and its position (order does not matter, so the threads of a
group can add in any order and the result is the same bits every time). A
changed word changes both sums except with probability about 2^-64.
"""

from __future__ import annotations

import mlx.core as mx

CHUNK_BYTES = 32768
_WORDS = CHUNK_BYTES // 4
_THREADS = 256

_SOURCE = """
    threadgroup uint part_a[32];
    threadgroup uint part_b[32];
    uint chunk = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_threadgroup;
    uint width = threads_per_threadgroup.x;
    uint total = sizes[0];
    uint per_chunk = sizes[1];
    const device uint* words = reinterpret_cast<const device uint*>(x);
    uint start = chunk * per_chunk;
    uint stop = min(start + per_chunk, total);
    uint a = 0u;
    uint b = 0u;
    for (uint i = start + lane; i < stop; i += width) {
        uint v = words[i];
        uint h = v ^ (i * 0x9E3779B1u);
        h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
        uint g = v ^ (i * 0x7FEB352Du + 0x846CA68Bu);
        g ^= g >> 15; g *= 0x2C1B3C6Du; g ^= g >> 12; g *= 0x297A2D39u; g ^= g >> 15;
        a += h;
        b += g;
    }
    a = simd_sum(a);
    b = simd_sum(b);
    if (thread_index_in_simdgroup == 0) {
        part_a[simdgroup_index_in_threadgroup] = a;
        part_b[simdgroup_index_in_threadgroup] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint ta = 0u;
        uint tb = 0u;
        uint groups = (width + 31u) / 32u;
        for (uint k = 0; k < groups; ++k) {
            ta += part_a[k];
            tb += part_b[k];
        }
        out[2 * chunk] = ta;
        out[2 * chunk + 1] = tb;
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mtplx_chunk_digest",
            input_names=["x", "sizes"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def chunks(nbytes: int) -> int:
    """How many chunks an array of ``nbytes`` bytes has."""

    return (int(nbytes) + CHUNK_BYTES - 1) // CHUNK_BYTES


def supported(array: mx.array) -> bool:
    """A digest reads the array's bytes as 32-bit words: it must be laid out
    row by row and a whole number of words long."""

    return (
        array.nbytes > 0
        and array.nbytes % 4 == 0
        and array.nbytes // 4 < 2**32
        and array.dtype.size in (1, 2, 4, 8)
    )


def digest(array: mx.array) -> mx.array:
    """Lazy ``[chunks, 2]`` uint32: the digest of every 32 KiB chunk of ``array``'s bytes.

    The array is read where it lives (a row-contiguous array is not copied).
    """

    words = int(array.nbytes) // 4
    count = chunks(array.nbytes)
    sizes = mx.array([words, _WORDS], dtype=mx.uint32)
    (out,) = _kernel()(
        inputs=[array, sizes],
        grid=(count * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(count * 2,)],
        output_dtypes=[mx.uint32],
    )
    return out.reshape(count, 2)


def chunk_copy(array: mx.array, chunk_indices: list[int]) -> mx.array:
    """Lazy copy of the given 32 KiB chunks of ``array``'s bytes, one after another."""

    flat = array.reshape(-1).view(mx.uint32)
    total = int(flat.size)
    parts = [
        flat[index * _WORDS : min((index + 1) * _WORDS, total)] for index in chunk_indices
    ]
    joined = mx.concatenate(parts) if len(parts) > 1 else parts[0]
    # A new buffer, never a view of the bank: the bank changes afterwards.
    return mx.bitwise_or(joined, mx.array(0, dtype=mx.uint32))


def window_chunks(shape: tuple[int, ...], itemsize: int, axis: int, start: int, rows: int) -> list[int]:
    """The chunks holding rows ``[start, start + rows)`` of ``axis`` of a
    row-contiguous array of ``shape``."""

    axis = axis % len(shape)
    lead = 1
    for extent in shape[:axis]:
        lead *= int(extent)
    row_bytes = int(itemsize)
    for extent in shape[axis + 1:]:
        row_bytes *= int(extent)
    length = int(shape[axis])
    stop = min(length, int(start) + int(rows))
    start = max(0, int(start))
    found: set[int] = set()
    if stop <= start:
        return []
    for block in range(lead):
        first = (block * length + start) * row_bytes
        last = (block * length + stop) * row_bytes - 1
        found.update(range(first // CHUNK_BYTES, last // CHUNK_BYTES + 1))
    return sorted(found)


__all__ = ["CHUNK_BYTES", "chunk_copy", "chunks", "digest", "supported", "window_chunks"]
