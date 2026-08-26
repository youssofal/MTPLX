"""Mia ``stock432`` NVFP4 rows for DeepSeek V4 target and DSpark K/V."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math

import mlx.core as mx

from mtplx.paged_cache import PagedCachePlan, PagedCachePool


MIA_NVFP4_HEAD_DIM = 512
MIA_NVFP4_NOPE_DIM = 448
MIA_NVFP4_ROPE_DIM = 64
MIA_NVFP4_GROUP_SIZE = 16
MIA_NVFP4_PACKED_BYTES = 256
MIA_NVFP4_SCALE_BYTES = 32
MIA_NVFP4_PADDING_OFFSET = 288
MIA_NVFP4_ROPE_OFFSET = 304
MIA_NVFP4_RECORD_BYTES = 432
MIA_DSPARK_CONTEXT_ROWS = 128


_NVFP4_HEADER = r"""
    using namespace metal;

    inline uchar mtplx_e4m3_encode_positive(float value) {
        if (!(value > 0.0f)) {
            return uchar(0);
        }
        value = min(value, 448.0f);
        constexpr float MIN_NORMAL = 0.015625f;
        constexpr float SUB_STEP = 0.001953125f;
        if (value < MIN_NORMAL) {
            uint mantissa = uint(rint(value / SUB_STEP));
            if (mantissa >= 8u) {
                return uchar(0x08);
            }
            return uchar(mantissa);
        }

        int exponent = int(floor(log2(value)));
        float step = exp2(float(exponent - 3));
        uint significand = uint(rint(value / step));
        if (significand >= 16u) {
            exponent += 1;
            significand = 8u;
        }
        uint stored_exponent = uint(exponent + 7);
        if (stored_exponent >= 15u) {
            stored_exponent = 15u;
            significand = min(significand, 14u);
        }
        uint mantissa = significand - 8u;
        return uchar((stored_exponent << 3) | mantissa);
    }

    inline float mtplx_e4m3_decode(uchar raw) {
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        if (exponent == 0u) {
            return float(mantissa) * 0.001953125f;
        }
        return (1.0f + float(mantissa) * 0.125f)
            * exp2(float(int(exponent) - 7));
    }

    inline uchar mtplx_e2m1_encode(float value) {
        float magnitude = abs(value);
        uchar code;
        if (magnitude <= 0.25f) {
            code = uchar(0);
        } else if (magnitude < 0.75f) {
            code = uchar(1);
        } else if (magnitude <= 1.25f) {
            code = uchar(2);
        } else if (magnitude < 1.75f) {
            code = uchar(3);
        } else if (magnitude <= 2.5f) {
            code = uchar(4);
        } else if (magnitude < 3.5f) {
            code = uchar(5);
        } else if (magnitude <= 5.0f) {
            code = uchar(6);
        } else {
            code = uchar(7);
        }
        uint sign = as_type<uint>(value) >> 31;
        return uchar(code | uchar(sign << 3));
    }
"""


@lru_cache(maxsize=1)
def _stock432_pack_kernel():
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 NVFP4 K/V requires Metal")
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        uint group = thread_index_in_simdgroup;
        const device T* latent_row = latent + size_t(row) * 512u;
        const device T* rope_row = rope + size_t(row) * 64u;
        device uchar* record = records + size_t(row) * 432u;

        uint group_dim = group * 16u;
        float amax = 0.0f;
        for (uint i = 0u; i < 16u; ++i) {
            uint dim = group_dim + i;
            float value = dim < 448u
                ? float(latent_row[dim])
                : float(rope_row[dim - 448u]);
            amax = max(amax, abs(value));
        }
        uchar scale_byte = mtplx_e4m3_encode_positive(amax / 6.0f);
        float scale = mtplx_e4m3_decode(scale_byte);
        float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (uint packed = 0u; packed < 8u; ++packed) {
            uint dim0 = group_dim + packed * 2u;
            float low_value = dim0 < 448u
                ? float(latent_row[dim0])
                : float(rope_row[dim0 - 448u]);
            float high_value = dim0 + 1u < 448u
                ? float(latent_row[dim0 + 1u])
                : float(rope_row[dim0 + 1u - 448u]);
            uchar low = mtplx_e2m1_encode(low_value * inv_scale);
            uchar high = mtplx_e2m1_encode(high_value * inv_scale);
            record[group * 8u + packed] = uchar(low | uchar(high << 4));
        }
        record[256u + group] = scale_byte;
        if (group < 16u) {
            record[288u + group] = uchar(0);
        }
        const device uchar* rope_bytes = reinterpret_cast<const device uchar*>(rope_row);
        for (uint byte = group; byte < 128u; byte += 32u) {
            record[304u + byte] = rope_bytes[byte];
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_stock432_pack",
        input_names=["latent", "rope"],
        output_names=["records"],
        header=_NVFP4_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _pack_stock432(latent: mx.array, rope: mx.array) -> mx.array:
    if latent.dtype != mx.bfloat16 or rope.dtype != mx.bfloat16:
        raise ValueError("Mia stock432 insertion requires BF16 latent and RoPE rows")
    if latent.ndim < 2 or int(latent.shape[-1]) != MIA_NVFP4_HEAD_DIM:
        raise ValueError("Mia stock432 latent rows must end in width 512")
    if tuple(latent.shape[:-1]) != (*rope.shape[:-1],):
        raise ValueError("Mia stock432 latent and RoPE row prefixes differ")
    if int(rope.shape[-1]) != MIA_NVFP4_ROPE_DIM:
        raise ValueError("Mia stock432 RoPE rows must end in width 64")
    return _run_pack_stock432(
        latent,
        rope,
        kernel=_stock432_pack_kernel(),
    )


def _run_pack_stock432(
    latent: mx.array,
    rope: mx.array,
    *,
    kernel,
) -> mx.array:
    """Direct packer for a construction-qualified stock432 route."""
    row_count = 1
    for size in latent.shape[:-1]:
        row_count *= int(size)
    records = kernel(
        inputs=[mx.contiguous(latent), mx.contiguous(rope)],
        template=[("T", mx.bfloat16)],
        grid=(row_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*latent.shape[:-1], MIA_NVFP4_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]
    return records


def install_stock432_record_packer(*, head_dim: int, rope_dim: int):
    """Validate fixed Mia geometry once and bind its direct record finalizer."""
    observed = (int(head_dim), int(rope_dim))
    expected = (MIA_NVFP4_HEAD_DIM, MIA_NVFP4_ROPE_DIM)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 geometry: {observed} != {expected}"
        )
    kernel = _stock432_pack_kernel()
    return partial(_run_pack_stock432, kernel=kernel)


_E2M1_TABLE = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)


def _decode_e4m3(raw_bytes: mx.array) -> mx.array:
    raw = raw_bytes.astype(mx.uint32)
    negative = (raw & 0x80) != 0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    subnormal = mantissa.astype(mx.float32) * (2.0**-9)
    normal = (1.0 + mantissa.astype(mx.float32) / 8.0) * mx.power(
        mx.array(2.0, dtype=mx.float32), exponent.astype(mx.float32) - 7.0
    )
    magnitude = mx.where(exponent == 0, subnormal, normal)
    return mx.where(negative, -magnitude, magnitude)


def decode_stock432(records: mx.array) -> tuple[mx.array, mx.array]:
    """Return source-semantics ``(key, value)`` reconstructed from records."""
    if records.dtype != mx.uint8 or records.ndim < 2:
        raise ValueError("Mia stock432 records must be rank-2-or-higher uint8")
    if int(records.shape[-1]) != MIA_NVFP4_RECORD_BYTES:
        raise ValueError("Mia stock432 records must end in 432 bytes")
    packed = records[..., :MIA_NVFP4_PACKED_BYTES]
    low = mx.take(_E2M1_TABLE, packed & 0xF)
    high = mx.take(_E2M1_TABLE, (packed >> 4) & 0xF)
    values = mx.stack([low, high], axis=-1).reshape(
        *records.shape[:-1], MIA_NVFP4_HEAD_DIM
    )
    scales = _decode_e4m3(records[..., 256:288])
    latent = (values * mx.repeat(scales, MIA_NVFP4_GROUP_SIZE, axis=-1)).astype(
        mx.bfloat16
    )
    rope_bytes = mx.contiguous(records[..., MIA_NVFP4_ROPE_OFFSET:])
    rope = rope_bytes.view(mx.bfloat16)
    key = mx.concatenate([latent[..., :MIA_NVFP4_NOPE_DIM], rope], axis=-1)
    return key, latent


class MiaNVFP4Rows:
    """Appendable DeepSeek V4 K/V rows in Mia's native 432-byte layout."""

    head_dim = MIA_NVFP4_HEAD_DIM
    nope_dim = MIA_NVFP4_NOPE_DIM
    rope_dim = MIA_NVFP4_ROPE_DIM
    group_size = MIA_NVFP4_GROUP_SIZE
    record_bytes = MIA_NVFP4_RECORD_BYTES
    mode = "nvfp4_stock432"

    def __init__(self) -> None:
        self.records: mx.array | None = None
        self._prefix_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return 0 if self.records is None else int(self.records.shape[-2])

    @property
    def shape(self) -> tuple[int, ...]:
        if self.records is None:
            return (0, self.record_bytes)
        return tuple(int(value) for value in self.records.shape)

    @property
    def state(self) -> mx.array | None:
        return self.records

    @property
    def nbytes(self) -> int:
        return 0 if self.records is None else int(self.records.nbytes)

    def _validate_rows(self, latent: mx.array, rope: mx.array) -> tuple[int, ...]:
        if latent.ndim < 2 or int(latent.shape[-1]) != self.head_dim:
            raise ValueError("Mia stock432 latent rows must end in width 512")
        if rope.ndim != latent.ndim or int(rope.shape[-1]) != self.rope_dim:
            raise ValueError("Mia stock432 RoPE rows must end in width 64")
        if tuple(latent.shape[:-1]) != tuple(rope.shape[:-1]):
            raise ValueError("Mia stock432 latent and RoPE row shapes differ")
        prefix = tuple(int(value) for value in latent.shape[:-2])
        if self._prefix_shape is not None and prefix != self._prefix_shape:
            raise ValueError(
                f"Mia stock432 prefix changed from {self._prefix_shape} to {prefix}"
            )
        return prefix

    def append(self, latent: mx.array, rope: mx.array) -> None:
        prefix = self._validate_rows(latent, rope)
        new_records = _pack_stock432(latent, rope)
        self._append_installed_records(new_records, prefix=prefix)

    def _append_installed_records(
        self,
        new_records: mx.array,
        *,
        prefix: tuple[int, ...],
    ) -> None:
        """Append records supplied by an installed, geometry-qualified packer."""
        if self.records is None:
            self.records = new_records
            self._prefix_shape = prefix
        else:
            self.records = mx.concatenate([self.records, new_records], axis=-2)

    def _replace_installed_records(
        self,
        start: int,
        replacement: mx.array,
    ) -> None:
        """Replace a qualified range without repeating record metadata checks."""
        count = int(replacement.shape[-2])
        self.records = mx.concatenate(
            [
                self.records[..., :start, :],
                replacement,
                self.records[..., start + count :, :],
            ],
            axis=-2,
        )

    def decode(self, start: int = 0, stop: int | None = None) -> tuple[mx.array, mx.array]:
        if self.records is None:
            raise ValueError("cannot decode an empty Mia stock432 K/V store")
        begin = int(start)
        end = len(self) if stop is None else int(stop)
        if begin < 0 or end < begin or end > len(self):
            raise ValueError("Mia stock432 decode range is outside the store")
        return decode_stock432(self.records[..., begin:end, :])

    def replace(self, start: int, latent: mx.array, rope: mx.array) -> None:
        if self.records is None:
            raise ValueError("cannot replace rows in an empty Mia stock432 store")
        self._validate_rows(latent, rope)
        start = int(start)
        count = int(latent.shape[-2])
        if start < 0 or count <= 0 or start + count > len(self):
            raise ValueError("replacement Mia stock432 range is outside the store")
        replacement = _pack_stock432(latent, rope)
        self._replace_installed_records(start, replacement)

    def drop_first(self, count: int) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if count >= len(self):
            self.clear()
            return
        self.records = self.records[..., count:, :]

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length >= len(self):
            return
        if length == 0:
            self.clear()
            return
        self.records = self.records[..., :length, :]

    def clear(self) -> None:
        self.records = None
        self._prefix_shape = None

    def replace_state(self, state: mx.array | None) -> None:
        if state is None:
            self.clear()
            return
        if (
            state.dtype != mx.uint8
            or state.ndim < 2
            or int(state.shape[-1]) != self.record_bytes
        ):
            raise ValueError("invalid Mia stock432 K/V state")
        self.records = state
        self._prefix_shape = tuple(int(value) for value in state.shape[:-2])


class FixedMiaNVFP4Ring:
    """One contiguous fixed page with direct slot writes for DSpark SWA."""

    head_dim = MIA_NVFP4_HEAD_DIM
    rope_dim = MIA_NVFP4_ROPE_DIM
    record_bytes = MIA_NVFP4_RECORD_BYTES
    mode = "nvfp4_stock432_fixed_ring"

    def __init__(self, *, capacity_rows: int) -> None:
        capacity_rows = int(capacity_rows)
        if capacity_rows <= 0:
            raise ValueError("Mia fixed ring capacity must be positive")
        plan = PagedCachePlan.contiguous(
            block_size=capacity_rows,
            num_blocks=1,
            array_names=("records",),
        )
        self._capacity_rows = capacity_rows
        self._pool = PagedCachePool(plan)
        self._pages = self._pool.bind(
            "records",
            row_shape=(self.record_bytes,),
            dtype=mx.uint8,
        )
        logical = (
            mx.arange(capacity_rows * 2, dtype=mx.int32) % capacity_rows
        )
        logical_blocks = logical // self._pool.block_size
        self._slot_blocks = self._pool.block_table[logical_blocks]
        self._slot_offsets = logical - logical_blocks * self._pool.block_size
        mx.eval(self._slot_blocks, self._slot_offsets)

    def __len__(self) -> int:
        return int(self._pool.offset)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, len(self), self.record_bytes)

    @property
    def records(self) -> mx.array:
        # num_blocks=1 makes the physical page exactly [1, capacity, 432].
        return self._pages

    @property
    def nbytes(self) -> int:
        return int(self._pages.nbytes)

    def decode(
        self,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        begin = int(start)
        end = self._capacity_rows if stop is None else int(stop)
        return decode_stock432(self.records[:, begin:end])

    def _append_installed_records(
        self,
        records: mx.array,
        *,
        prefix: tuple[int, ...],
    ) -> None:
        del prefix
        count = int(records.shape[1])
        self._pool._write_installed_tail(
            {"records": records[0]},
            count=count,
        )

    def _replace_installed_records(
        self,
        start: int,
        replacement: mx.array,
    ) -> None:
        count = int(replacement.shape[1])
        positions = mx.arange(int(start), int(start) + count, dtype=mx.int32)
        self._pool.write_slots(
            {"records": replacement[0]},
            logical_positions=positions,
        )

    def clear(self) -> None:
        self._pool.clear()


@dataclass(slots=True)
class FixedMiaNVFP4WindowRecords:
    """Construction-owned physical-page view of the target SWA ring.

    ``capacity`` is the logical circular address space.  It is deliberately
    distinct from the padded physical row count in ``pages``.
    """

    pages: mx.array
    block_table: mx.array
    length: int
    block_size: int
    capacity: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, int(self.length), MIA_NVFP4_RECORD_BYTES)

    @property
    def dtype(self):
        return self.pages.dtype

    @property
    def physical_rows(self) -> int:
        return int(self.pages.shape[0]) * int(self.pages.shape[1])


class FixedMiaNVFP4Window:
    """Persistent circular pages for a target SWA batch plus rollback tail."""

    head_dim = MIA_NVFP4_HEAD_DIM
    rope_dim = MIA_NVFP4_ROPE_DIM
    record_bytes = MIA_NVFP4_RECORD_BYTES
    mode = "nvfp4_stock432_fixed_window"

    def __init__(self, *, capacity_rows: int, block_size: int = 64) -> None:
        capacity_rows = int(capacity_rows)
        block_size = int(block_size)
        if capacity_rows <= 0 or block_size <= 0:
            raise ValueError("Mia fixed window geometry must be positive")
        plan = PagedCachePlan.contiguous(
            block_size=block_size,
            num_blocks=math.ceil(capacity_rows / block_size),
            array_names=("records",),
        )
        self._capacity_rows = capacity_rows
        self._pool = PagedCachePool(plan)
        self._pages = self._pool.bind(
            "records",
            row_shape=(self.record_bytes,),
            dtype=mx.uint8,
        )
        self._paged_records = FixedMiaNVFP4WindowRecords(
            pages=self._pages,
            block_table=self._pool.block_table,
            length=0,
            block_size=self._pool.block_size,
            capacity=self._capacity_rows,
        )
        self._rebuild_slot_map()
        self._start = 0
        self._end = 0

    def _rebuild_slot_map(self) -> None:
        logical = (
            mx.arange(self._capacity_rows * 2, dtype=mx.int32)
            % self._capacity_rows
        )
        logical_blocks = logical // self._pool.block_size
        self._slot_blocks = self._pool.block_table[logical_blocks]
        self._slot_offsets = logical - logical_blocks * self._pool.block_size
        mx.eval(self._slot_blocks, self._slot_offsets)

    def __len__(self) -> int:
        return self._end - self._start

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def capacity(self) -> int:
        return self._capacity_rows

    @property
    def nbytes(self) -> int:
        return int(self._pages.nbytes)

    @property
    def records(self) -> mx.array:
        return self.slice(self._start, self._end)

    def paged_records(self, start: int, stop: int) -> FixedMiaNVFP4WindowRecords:
        """Return the persistent physical-page descriptor without gathering."""

        self._paged_records.length = int(stop) - int(start)
        return self._paged_records

    @property
    def state(self):
        return self._pages, self._pool.block_table, self._start, self._end

    def _append_installed_records(
        self,
        records: mx.array,
        *,
        absolute_start: int,
    ) -> None:
        absolute_start = int(absolute_start)
        count = int(records.shape[1])
        physical = (
            mx.arange(absolute_start, absolute_start + count, dtype=mx.int32)
            % self._capacity_rows
        )
        self._pool.write_slots(
            {"records": records[0]},
            logical_positions=physical,
        )
        self._end = absolute_start + count

    def slice(self, start: int, stop: int) -> mx.array:
        start = int(start)
        stop = int(stop)
        slot = start % self._capacity_rows
        count = stop - start
        end = slot + count
        return self._pages[
            self._slot_blocks[slot:end],
            self._slot_offsets[slot:end],
        ][None]

    def drop_before(self, start: int) -> None:
        start = int(start)
        self._start = start

    def truncate(self, length: int) -> None:
        length = int(length)
        self._end = self._start + length

    def rewind(self, end: int, *, keep: int) -> int:
        """Restore the retained frontier around an authoritative cache end."""

        end = int(end)
        retained_start = max(0, end - int(keep))
        self._start = retained_start
        self._end = end
        self._paged_records.length = end - retained_start
        return retained_start

    def clear(self) -> None:
        self._start = 0
        self._end = 0
        self._paged_records.length = 0

    def replace_state(self, state) -> None:
        if state is None:
            self.clear()
            return
        if not isinstance(state, (tuple, list)) or len(state) != 4:
            raise ValueError("invalid Mia fixed target window state")
        pages, block_table, start, end = state
        start = int(start)
        end = int(end)
        if tuple(int(value) for value in block_table.shape) != (
            self._pool.num_blocks,
        ):
            raise ValueError("Mia fixed target window block table shape changed")
        if start < 0 or end < start or end - start > self._capacity_rows:
            raise ValueError("Mia fixed target window frontier changed")
        self._pool.block_table = block_table
        self._pool.replace_state({"records": pages}, 0)
        self._pages = self._pool.buffer("records")
        self._paged_records.pages = self._pages
        self._paged_records.block_table = self._pool.block_table
        self._paged_records.length = end - start
        self._rebuild_slot_map()
        self._start = start
        self._end = end


@dataclass(frozen=True)
class PagedMiaNVFP4Records:
    """Physical ``stock432`` pages and their logical row mapping."""

    records: mx.array
    block_table: mx.array
    length: int
    block_size: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, int(self.length), MIA_NVFP4_RECORD_BYTES)

    @property
    def dtype(self):
        return self.records.dtype


class PagedMiaNVFP4Rows:
    """Fixed-capacity owner for Mia rows, backed by shared MTPLX pages.

    The exact model is single-request/batch-one.  Removing that invariant would
    require one block table per request, so it is rejected at insertion rather
    than hidden behind an execution-time fallback.
    """

    head_dim = MIA_NVFP4_HEAD_DIM
    nope_dim = MIA_NVFP4_NOPE_DIM
    rope_dim = MIA_NVFP4_ROPE_DIM
    group_size = MIA_NVFP4_GROUP_SIZE
    record_bytes = MIA_NVFP4_RECORD_BYTES
    mode = "nvfp4_stock432_paged"

    def __init__(self, *, capacity_rows: int, block_size: int = 64) -> None:
        capacity_rows = int(capacity_rows)
        block_size = int(block_size)
        if capacity_rows <= 0:
            raise ValueError("paged Mia stock432 capacity_rows must be positive")
        plan = PagedCachePlan.contiguous(
            block_size=block_size,
            num_blocks=math.ceil(capacity_rows / block_size),
            array_names=("records",),
        )
        self._capacity_rows = capacity_rows
        self._pool = PagedCachePool(plan)
        self._pages = self._pool.bind(
            "records",
            row_shape=(self.record_bytes,),
            dtype=mx.uint8,
        )

    def __len__(self) -> int:
        return int(self._pool.offset)

    @property
    def capacity(self) -> int:
        return self._capacity_rows

    @property
    def pages(self) -> mx.array:
        return self._pages

    @property
    def block_table(self) -> mx.array:
        return self._pool.block_table

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, len(self), self.record_bytes)

    @property
    def paged_records(self) -> PagedMiaNVFP4Records:
        return PagedMiaNVFP4Records(
            records=self.pages,
            block_table=self.block_table,
            length=len(self),
            block_size=self.block_size,
        )

    @property
    def records(self) -> mx.array:
        """Materialize logical rows only at explicit state/oracle boundaries."""
        return self._pool.active("records")[None]

    @property
    def state(self):
        return self.pages, self.block_table, len(self)

    @property
    def nbytes(self) -> int:
        return int(self.pages.nbytes)

    @staticmethod
    def _validate_rows(latent: mx.array, rope: mx.array) -> None:
        if tuple(int(value) for value in latent.shape[:1]) != (1,):
            raise ValueError("paged Mia stock432 rows require batch size one")
        if latent.ndim != 3 or int(latent.shape[-1]) != MIA_NVFP4_HEAD_DIM:
            raise ValueError("paged Mia stock432 latent rows must be [1, rows, 512]")
        if tuple(int(value) for value in rope.shape) != (
            1,
            int(latent.shape[1]),
            MIA_NVFP4_ROPE_DIM,
        ):
            raise ValueError("paged Mia stock432 RoPE rows must be [1, rows, 64]")

    def append(self, latent: mx.array, rope: mx.array) -> None:
        self._validate_rows(latent, rope)
        self.append_records(_pack_stock432(latent, rope))

    def append_records(self, records: mx.array) -> None:
        """Insert records already finalized by the fused Mia compressor."""
        if (
            records.dtype != mx.uint8
            or records.ndim != 3
            or tuple(int(value) for value in records.shape[:1]) != (1,)
            or int(records.shape[-1]) != self.record_bytes
        ):
            raise ValueError(
                "paged Mia stock432 records must be [1, rows, 432] uint8"
            )
        count = int(records.shape[1])
        if len(self) + count > self.capacity:
            raise ValueError(
                f"paged Mia stock432 capacity exceeded: {len(self) + count} "
                f"> {self.capacity}"
            )
        self._append_installed_records(records)

    def _append_installed_records(self, records: mx.array) -> None:
        """Insert records emitted by the construction-bound fused compressor."""
        self._pool._write_installed_tail(
            {"records": records[0]},
            count=int(records.shape[1]),
        )

    def _append_m6_records(self, records: mx.array, schedule) -> None:
        """Insert physical-M6 records through the request-owned page schedule."""
        self._pool._write_installed_mapping(
            {"records": records[0]},
            physical_blocks=schedule.compressed_blocks,
            block_offsets=schedule.compressed_offsets,
            new_offset=schedule.first_window + schedule.emitted_rows,
        )

    def decode(self, start: int = 0, stop: int | None = None) -> tuple[mx.array, mx.array]:
        begin = int(start)
        end = len(self) if stop is None else int(stop)
        if begin < 0 or end < begin or end > len(self):
            raise ValueError("paged Mia stock432 decode range is outside the store")
        return decode_stock432(self.records[:, begin:end])

    def replace(self, start: int, latent: mx.array, rope: mx.array) -> None:
        self._validate_rows(latent, rope)
        start = int(start)
        count = int(latent.shape[1])
        if start < 0 or count <= 0 or start + count > len(self):
            raise ValueError("replacement paged Mia stock432 range is outside the store")
        positions = mx.arange(start, start + count, dtype=mx.int32)
        self._pool.write_slots(
            {"records": _pack_stock432(latent, rope)[0]},
            logical_positions=positions,
        )

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length >= len(self):
            return
        self._pool.truncate(length)

    def clear(self) -> None:
        self._pool.clear()

    def replace_state(self, state) -> None:
        if state is None:
            self.clear()
            return
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("invalid paged Mia stock432 state")
        pages, block_table, length = state
        if tuple(int(value) for value in block_table.shape) != (
            self._pool.num_blocks,
        ):
            raise ValueError("paged Mia stock432 block table shape changed")
        self._pool.block_table = block_table
        self._pool.replace_state({"records": pages}, int(length))
        self._pages = self._pool.buffer("records")


def _write_fixed_window_records(
    records: mx.array,
    *,
    owner: FixedMiaNVFP4Window,
    absolute_start: int,
) -> None:
    """Scatter one qualified target batch and advance its frontier once."""
    absolute_start = int(absolute_start)
    count = int(records.shape[1])
    slot = absolute_start % owner._capacity_rows
    stop = slot + count
    owner._pages[
        owner._slot_blocks[slot:stop],
        owner._slot_offsets[slot:stop],
    ] = records[0]
    owner._end = absolute_start + count


def install_fixed_window_record_writer(owner: FixedMiaNVFP4Window):
    """Bind the final page owner for fused target prologue records."""
    if not isinstance(owner, FixedMiaNVFP4Window):
        raise TypeError("target record writer requires FixedMiaNVFP4Window")
    return partial(_write_fixed_window_records, owner=owner)


def _write_fixed_ring_context_records(
    records: mx.array,
    *,
    owner: FixedMiaNVFP4Ring,
    absolute_start: int,
) -> None:
    """Install one prompt tail, zero-pad its ring, and mark it ready."""
    count = int(records.shape[1])
    owner._pages[:] = mx.array(0, dtype=mx.uint8)
    slot = int(absolute_start) % owner._capacity_rows
    stop = slot + count
    owner._pages[
        owner._slot_blocks[slot:stop],
        owner._slot_offsets[slot:stop],
    ] = records[0]
    owner._pool.offset = owner._capacity_rows


def install_fixed_ring_context_writer(owner: FixedMiaNVFP4Ring):
    """Bind the persistent DSpark context owner after geometry validation."""
    if not isinstance(owner, FixedMiaNVFP4Ring):
        raise TypeError("context record writer requires FixedMiaNVFP4Ring")
    if int(owner._capacity_rows) != MIA_DSPARK_CONTEXT_ROWS:
        raise ValueError(
            "DSpark context writer requires the fixed 128-row ring"
        )
    return partial(_write_fixed_ring_context_records, owner=owner)


def _write_fixed_ring_commit_records(
    records: mx.array,
    *,
    owner: FixedMiaNVFP4Ring,
    absolute_start: int,
) -> None:
    """Scatter one incremental commit without moving the filled frontier."""
    count = int(records.shape[1])
    slot = int(absolute_start) % owner._capacity_rows
    stop = slot + count
    owner._pages[
        owner._slot_blocks[slot:stop],
        owner._slot_offsets[slot:stop],
    ] = records[0]


def install_fixed_ring_commit_writer(owner: FixedMiaNVFP4Ring):
    """Bind the fixed ring owner for variable-width authoritative commits."""
    if not isinstance(owner, FixedMiaNVFP4Ring):
        raise TypeError("commit record writer requires FixedMiaNVFP4Ring")
    if int(owner._capacity_rows) != MIA_DSPARK_CONTEXT_ROWS:
        raise ValueError("DSpark commit writer requires the fixed 128-row ring")
    return partial(_write_fixed_ring_commit_records, owner=owner)
