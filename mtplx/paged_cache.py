"""Reusable fixed-capacity paged storage for MTPLX caches.

The owner in this module is deliberately unaware of attention, quantization,
or model families.  A construction-time plan fixes physical capacity and named
arrays; execution receives logical positions, maps them through the installed
block table, and writes directly into the owned pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PagedCachePlan:
    """Immutable geometry for one set of lockstep paged arrays."""

    block_size: int
    num_blocks: int
    array_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError("paged cache block_size must be positive")
        if int(self.num_blocks) <= 0:
            raise ValueError("paged cache num_blocks must be positive")
        if not self.array_names:
            raise ValueError("paged cache plan must contain at least one array")
        if any(not str(name) for name in self.array_names):
            raise ValueError("paged cache array names must be non-empty")
        if len(set(self.array_names)) != len(self.array_names):
            raise ValueError("paged cache array names must be unique")

    @classmethod
    def contiguous(
        cls,
        *,
        block_size: int,
        num_blocks: int,
        array_names: Sequence[str],
    ) -> "PagedCachePlan":
        return cls(
            block_size=int(block_size),
            num_blocks=int(num_blocks),
            array_names=tuple(str(name) for name in array_names),
        )

    @property
    def capacity(self) -> int:
        return int(self.block_size) * int(self.num_blocks)


@dataclass(frozen=True)
class PagedArrayBinding:
    """Installed row geometry for one array in a page pool."""

    row_shape: tuple[int, ...]
    dtype: Any


class PagedCachePool:
    """Fixed physical pages plus a logical-to-physical block table.

    The initial table is the contiguous single-request mapping used by current
    MTPLX serving.  Keeping the table explicit makes slot ownership independent
    of that mapping and lets the same owner accept leased/non-contiguous block
    tables without changing model cache code.
    """

    def __init__(
        self,
        plan: PagedCachePlan,
        *,
        block_table: Any | None = None,
        offset: int = 0,
    ) -> None:
        import mlx.core as mx

        self.plan = plan
        self.block_table = (
            mx.arange(plan.num_blocks, dtype=mx.int32)
            if block_table is None
            else block_table
        )
        self.offset = int(offset)
        if self.offset < 0 or self.offset > self.capacity:
            raise ValueError(
                f"paged cache offset {self.offset} is outside capacity {self.capacity}"
            )
        self._bindings: dict[str, PagedArrayBinding] = {}
        self._buffers: dict[str, Any] = {}

    @property
    def block_size(self) -> int:
        return int(self.plan.block_size)

    @property
    def num_blocks(self) -> int:
        return int(self.plan.num_blocks)

    @property
    def capacity(self) -> int:
        return int(self.plan.capacity)

    def bind(
        self,
        name: str,
        *,
        row_shape: Sequence[int],
        dtype: Any,
        buffer: Any | None = None,
    ) -> Any:
        """Install one array's invariant geometry and allocate it once."""

        import mlx.core as mx

        name = str(name)
        if name not in self.plan.array_names:
            raise ValueError(f"{name!r} is not part of this paged cache plan")
        if name in self._bindings:
            raise ValueError(f"paged cache array {name!r} is already bound")
        shape = tuple(int(dim) for dim in row_shape)
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"paged cache row shape must be positive, got {shape}")
        expected_shape = (self.num_blocks, self.block_size, *shape)
        if buffer is None:
            buffer = mx.zeros(expected_shape, dtype=dtype)
            mx.eval(buffer)
        elif tuple(int(dim) for dim in buffer.shape) != expected_shape:
            raise ValueError(
                f"paged cache buffer {name!r} shape changed: "
                f"expected {expected_shape}, got {tuple(buffer.shape)}"
            )
        elif buffer.dtype != dtype:
            raise ValueError(
                f"paged cache buffer {name!r} dtype changed: "
                f"expected {dtype}, got {buffer.dtype}"
            )
        self._bindings[name] = PagedArrayBinding(shape, dtype)
        self._buffers[name] = buffer
        return buffer

    def buffer(self, name: str) -> Any:
        try:
            return self._buffers[str(name)]
        except KeyError as exc:
            raise ValueError(f"paged cache array {name!r} is not bound") from exc

    def binding(self, name: str) -> PagedArrayBinding:
        try:
            return self._bindings[str(name)]
        except KeyError as exc:
            raise ValueError(f"paged cache array {name!r} is not bound") from exc

    def slot_mapping(self, start: int, count: int) -> tuple[Any, Any]:
        """Return physical block ids and offsets for a logical token range."""

        start = int(start)
        count = int(count)
        stop = start + count
        if start < 0 or count < 0 or stop > self.capacity:
            raise ValueError(
                f"paged cache range {start}:{stop} is outside capacity {self.capacity}"
            )
        return self._installed_slot_mapping(start, count)

    def _installed_slot_mapping(self, start: int, count: int) -> tuple[Any, Any]:
        """Map a range already bounded by an installed capacity contract."""
        import mlx.core as mx

        stop = int(start) + int(count)
        positions = mx.arange(start, stop, dtype=mx.int32)
        logical_blocks = positions // self.block_size
        offsets = positions - logical_blocks * self.block_size
        return self.block_table[logical_blocks], offsets

    def write_tail(self, updates: Mapping[str, Any]) -> None:
        """Write lockstep rows at the current logical offset and advance it."""

        if not updates:
            return
        first = next(iter(updates.values()))
        count = int(first.shape[0])
        stop = self.offset + count
        if stop > self.capacity:
            raise ValueError(
                f"paged cache capacity exceeded: {stop} > {self.capacity}"
            )
        self._write_installed_tail(updates, count=count)

    def _write_installed_tail(
        self,
        updates: Mapping[str, Any],
        *,
        count: int,
    ) -> None:
        """Write qualified lockstep rows without repeating installed invariants."""
        stop = self.offset + int(count)
        physical_blocks, block_offsets = self._installed_slot_mapping(
            self.offset,
            int(count),
        )
        for name in self.plan.array_names:
            rows = updates[name]
            self._buffers[name][physical_blocks, block_offsets] = rows
        self.offset = stop

    def _write_installed_mapping(
        self,
        updates: Mapping[str, Any],
        *,
        physical_blocks: Any,
        block_offsets: Any,
        new_offset: int,
    ) -> None:
        """Write through a construction-owned logical-to-physical mapping."""
        for name in self.plan.array_names:
            self._buffers[name][physical_blocks, block_offsets] = updates[name]
        self.offset = int(new_offset)

    def write_slots(
        self,
        updates: Mapping[str, Any],
        *,
        logical_positions: Any,
    ) -> None:
        """Write rows to explicit logical positions without changing offset."""

        logical_blocks = logical_positions // self.block_size
        block_offsets = logical_positions - logical_blocks * self.block_size
        physical_blocks = self.block_table[logical_blocks]
        for name in self.plan.array_names:
            self._buffers[name][physical_blocks, block_offsets] = updates[name]

    def active(self, name: str) -> Any:
        """Gather active logical rows in order."""

        physical_blocks, block_offsets = self.slot_mapping(0, self.offset)
        return self.buffer(name)[physical_blocks, block_offsets]

    def truncate(self, length: int) -> None:
        length = int(length)
        if length < 0 or length > self.offset:
            raise ValueError(
                f"cannot truncate paged cache at {self.offset} to {length}"
            )
        self.offset = length

    def trim(self, count: int) -> int:
        count = min(self.offset, max(0, int(count)))
        self.offset -= count
        return count

    def clear(self) -> None:
        self.offset = 0

    @property
    def nbytes(self) -> int:
        return sum(int(buffer.nbytes) for buffer in self._buffers.values())

    @property
    def state(self) -> tuple[dict[str, Any], int]:
        return dict(self._buffers), int(self.offset)

    def replace_state(self, buffers: Mapping[str, Any], offset: int) -> None:
        """Install already-owned physical pages after construction validation."""

        for name in self.plan.array_names:
            binding = self.binding(name)
            buffer = buffers[name]
            expected = (
                self.num_blocks,
                self.block_size,
                *binding.row_shape,
            )
            if tuple(int(dim) for dim in buffer.shape) != expected:
                raise ValueError(
                    f"paged cache state {name!r} shape changed: "
                    f"expected {expected}, got {tuple(buffer.shape)}"
                )
            if buffer.dtype != binding.dtype:
                raise ValueError(
                    f"paged cache state {name!r} dtype changed: "
                    f"expected {binding.dtype}, got {buffer.dtype}"
                )
        offset = int(offset)
        if offset < 0 or offset > self.capacity:
            raise ValueError(
                f"paged cache state offset {offset} is outside capacity {self.capacity}"
            )
        self._buffers = {name: buffers[name] for name in self.plan.array_names}
        self.offset = offset
