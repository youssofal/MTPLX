"""Owner-thread handoff for the stock Qwen4 PLE row cache.

The native producer owns hashing and cold-row reads, but it never reads or
writes the Python LRU.  This module is the narrow owner-side seam: it turns a
fixed 64-row native request into compact hit/miss storage and publishes a
trusted completed miss batch back into the existing stock cache.

There is intentionally no MLX import here.  The payload is the exact packed
sidecar row used by the stock cache: 20 uint32 weight values followed by five
uint16 scale values and five uint16 bias values (100 bytes total).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MAX_ROW_SLOTS = 64
PACKED_ROW_BYTES = 100
_HIT_FLAG = 0x80
_INDEX_MASK = 0x3F
_PLANE_SPECS = (
    ("weight", np.dtype(np.uint32), (20,), 80),
    ("scales", np.dtype(np.uint16), (5,), 10),
    ("biases", np.dtype(np.uint16), (5,), 10),
)


@dataclass(frozen=True, slots=True)
class PreparedRows:
    """Immutable compact request handed to the native row producer.

    ``source`` has one byte per fixed slot.  Bit 7 selects ``hit_packed``;
    otherwise it selects ``miss_ids``.  The low six bits are the compact index
    in the selected array.  This keeps the native-side storage fixed at 64
    slots without retaining duplicate Python ID arrays.
    """

    source: np.ndarray
    hit_packed: np.ndarray
    miss_ids: np.ndarray
    # Host-only replay order.  The native ABI consumes only the first three
    # arrays; the owner uses this immutable sorted-unique order to reproduce
    # stock's insert-then-touch LRU sequence at publication.
    touch_order: np.ndarray
    # Host-only compact-index tags for ``touch_order``.  A hit tag points into
    # ``hit_packed`` so publication can restore a hit evicted while native I/O
    # was in flight; a miss tag points into ``miss_ids``.
    touch_source: np.ndarray

    @property
    def hit_count(self) -> int:
        return int(self.hit_packed.shape[0])

    @property
    def miss_count(self) -> int:
        return int(self.miss_ids.shape[0])


@dataclass(frozen=True, slots=True)
class _CompletedMisses:
    """A completion ticket minted by one :class:`CachedRowHandoff`.

    The public checked constructor copies and validates an untrusted batch.
    The native path can use ``trusted_completion`` after its extension has
    established the immutable packed-buffer contract; ``publish`` then has no
    repeated ID/shape validation in the enabled path.
    """

    prepared: PreparedRows
    packed: np.ndarray
    _owner_token: object


def _pack_into_trusted(destination: np.ndarray, payload: Any) -> None:
    """Pack a construction-validated stock payload without hot checks."""

    for value, (_name, _dtype, _shape, row_bytes), start in zip(
        payload,
        _PLANE_SPECS,
        (0, 80, 90),
    ):
        destination[start : start + row_bytes] = (
            np.asarray(value).view(np.uint8).reshape(-1)
        )


def _pack_into_checked(destination: np.ndarray, payload: Any) -> None:
    """Checked external boundary for one stock payload."""

    for value, (_name, dtype, shape, row_bytes), start in zip(
        payload,
        _PLANE_SPECS,
        (0, 80, 90),
    ):
        array = np.asarray(value)
        # These are construction-bound stock cache values.  Let a malformed
        # external fake fail at the boundary rather than adding checks to the
        # installed per-cycle handoff path.
        if array.dtype != dtype or tuple(array.shape) != shape:
            raise ValueError(
                "stock PLE row payload geometry mismatch: "
                f"observed={array.dtype}/{tuple(array.shape)} "
                f"expected={dtype}/{shape}"
            )
        destination[start : start + row_bytes] = array.view(np.uint8).reshape(-1)


def pack_row_payload(payload: Any) -> np.ndarray:
    """Checked external helper producing one immutable packed stock row."""

    packed = np.empty((PACKED_ROW_BYTES,), dtype=np.uint8)
    _pack_into_checked(packed, payload)
    packed.flags.writeable = False
    return packed


def _snapshot_row_ids_checked(row_ids: Any) -> np.ndarray:
    """Copy and validate the fixed native row-id boundary once."""

    values = np.asarray(row_ids)
    if values.shape != (MAX_ROW_SLOTS,):
        raise ValueError(
            f"native PLE row handoff requires exactly {MAX_ROW_SLOTS} IDs; "
            f"got shape {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("native PLE row IDs must have an integer dtype")
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise ValueError("native PLE row IDs must fit uint32")
    return np.ascontiguousarray(values, dtype=np.uint32)


def _snapshot_row_ids_trusted(row_ids: Any) -> np.ndarray:
    """Copy the fixed native ABI without revalidating it per cycle."""

    return np.array(row_ids, dtype=np.uint32, order="C", copy=True)


def _readonly_contiguous(array: Any, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.flags.writeable = False
    return result


class CachedRowHandoff:
    """Bind owner-side operations to one existing stock sidecar cache."""

    __slots__ = (
        "_hot",
        "_hot_cap_rows",
        "_hot_row_bytes",
        "_row_specs",
        "_owner_token",
    )

    def __init__(self, sidecar: Any) -> None:
        self._hot = sidecar._hot
        self._hot_row_bytes = int(sidecar._hot_row_bytes)
        self._hot_cap_rows = int(sidecar._hot_cap_rows)
        if self._hot_row_bytes != PACKED_ROW_BYTES:
            raise ValueError(
                "stock PLE cache row-byte policy changed: "
                f"observed={self._hot_row_bytes} expected={PACKED_ROW_BYTES}"
            )
        if self._hot_cap_rows < 0:
            raise ValueError("stock PLE cache row limit cannot be negative")
        self._row_specs = self._capture_row_specs(sidecar)
        self._owner_token = object()

    @staticmethod
    def _capture_row_specs(sidecar: Any) -> tuple[tuple[int, Any, tuple[int, ...], int], ...]:
        specs = []
        for name, expected_dtype, expected_shape, row_bytes in _PLANE_SPECS:
            try:
                matrix, _dtype_name = sidecar._maps[name]
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError(f"stock PLE sidecar lacks {name} map") from exc
            observed_shape = tuple(int(value) for value in matrix.shape[1:])
            observed_dtype = np.dtype(matrix.dtype)
            if observed_dtype != expected_dtype or observed_shape != expected_shape:
                raise ValueError(
                    f"stock PLE {name} geometry mismatch: "
                    f"observed={observed_dtype}/{observed_shape} "
                    f"expected={expected_dtype}/{expected_shape}"
                )
            if int(np.prod(observed_shape)) * observed_dtype.itemsize != row_bytes:
                raise ValueError(f"stock PLE {name} row-byte calculation mismatch")
            start = sum(item[3] for item in specs)
            specs.append((start, observed_dtype, observed_shape, row_bytes))
        return tuple(specs)

    @property
    def row_bytes(self) -> int:
        return self._hot_row_bytes

    @property
    def limit_rows(self) -> int:
        return self._hot_cap_rows

    def prepare(self, row_ids: Any) -> PreparedRows:
        """Resolve a trusted fixed-64 native request against the stock LRU.

        The native installer guarantees shape, dtype, range, and row geometry
        once.  ``checked_prepare`` is the diagnostic/untrusted boundary; this
        enabled method intentionally does not call it or repeat those checks.
        """

        slots = _snapshot_row_ids_trusted(row_ids)
        unique, inverse = np.unique(slots, return_inverse=True)
        unique_count = int(unique.size)
        hit_flags = np.empty((unique_count,), dtype=np.bool_)
        for index, row in enumerate(unique):
            hit_flags[index] = int(row) in self._hot

        hit_unique_indices = np.flatnonzero(hit_flags)
        miss_unique_indices = np.flatnonzero(~hit_flags)
        hit_count = int(hit_unique_indices.size)
        miss_count = int(miss_unique_indices.size)
        hit_packed = np.empty((hit_count, PACKED_ROW_BYTES), dtype=np.uint8)
        compact_index = np.empty((unique_count,), dtype=np.uint8)

        # np.unique returns sorted IDs, exactly as the stock _rows_matrices
        # path does.  Publication inserts misses first, then replays this
        # complete order before eviction, matching stock.  A hit is packed
        # here but deliberately not touched until publication so an owner
        # gather can interleave safely while native misses are outstanding.
        for compact, unique_index in enumerate(hit_unique_indices):
            row = int(unique[unique_index])
            _pack_into_trusted(hit_packed[compact], self._hot[row])
            compact_index[unique_index] = _HIT_FLAG | compact
        for compact, unique_index in enumerate(miss_unique_indices):
            compact_index[unique_index] = compact

        source = _readonly_contiguous(compact_index[inverse], dtype=np.uint8)
        miss_ids = _readonly_contiguous(unique[miss_unique_indices], dtype=np.uint32)
        touch_order = _readonly_contiguous(unique, dtype=np.uint32)
        touch_source = _readonly_contiguous(compact_index, dtype=np.uint8)
        hit_packed.flags.writeable = False
        return PreparedRows(
            source=source,
            hit_packed=hit_packed,
            miss_ids=miss_ids,
            touch_order=touch_order,
            touch_source=touch_source,
        )

    def checked_prepare(self, row_ids: Any) -> PreparedRows:
        """Checked diagnostic/test boundary for a native row request."""

        snapshot = _snapshot_row_ids_checked(row_ids)
        return self.prepare(snapshot)

    def checked_completion(
        self,
        prepared: PreparedRows,
        completed_miss_ids: Any,
        packed: Any,
    ) -> _CompletedMisses:
        """Validate and freeze an untrusted native/test completion boundary."""

        if not isinstance(prepared, PreparedRows):
            raise TypeError("completed PLE rows require PreparedRows")
        observed_ids = np.asarray(completed_miss_ids)
        if observed_ids.shape != prepared.miss_ids.shape:
            raise ValueError("completed PLE miss IDs shape does not match miss IDs")
        if observed_ids.dtype != np.uint32 or not np.array_equal(
            observed_ids, prepared.miss_ids
        ):
            raise ValueError("completed PLE miss IDs do not match prepared miss IDs")
        observed_packed = np.asarray(packed)
        expected_shape = (prepared.miss_count, PACKED_ROW_BYTES)
        if observed_packed.shape != expected_shape:
            raise ValueError(
                "completed PLE packed rows shape does not match prepared misses: "
                f"observed={observed_packed.shape} expected={expected_shape}"
            )
        if observed_packed.dtype != np.uint8:
            raise ValueError("completed PLE packed rows must be uint8")
        frozen = np.array(observed_packed, dtype=np.uint8, order="C", copy=True)
        frozen.flags.writeable = False
        return _CompletedMisses(prepared, frozen, self._owner_token)

    def trusted_completion(
        self, prepared: PreparedRows, packed: np.ndarray
    ) -> _CompletedMisses:
        """Mint a ticket from an already-validated native immutable batch.

        The native producer is expected to return a C-contiguous read-only
        uint8 ``(miss_count, 100)`` array whose rows correspond to the
        immutable ``prepared.miss_ids``.  The checked constructor above is for
        untrusted test/extension boundaries; this method deliberately performs
        no repeated shape or ID work in the installed route.
        """

        return _CompletedMisses(prepared, packed, self._owner_token)

    def publish(self, completion: _CompletedMisses) -> None:
        """Install a trusted completion with stock ownership/eviction rules."""

        prepared = completion.prepared
        packed = completion.packed
        # The ticket is trusted here.  Each row is copied into its own 100-byte
        # owner buffer before typed views are retained by _hot; retaining one
        # cache row therefore cannot retain the producer's entire 6.4 KiB batch.
        for index, row in enumerate(prepared.miss_ids):
            self._hot[int(row)] = self._payload_from_packed_row(packed[index])
        # Stock _rows_matrices inserts all misses, touches every sorted unique
        # ID, and only then evicts.  Replaying the host-only order preserves
        # that exact mixed hit/miss LRU result.
        for index, row in enumerate(prepared.touch_order):
            row_id = int(row)
            source = int(prepared.touch_source[index])
            if source & _HIT_FLAG and row_id not in self._hot:
                hit_index = source & _INDEX_MASK
                self._hot[row_id] = self._payload_from_packed_row(
                    prepared.hit_packed[hit_index]
                )
            self._hot.move_to_end(row_id)
        while len(self._hot) > self._hot_cap_rows:
            self._hot.popitem(last=False)

    def checked_publish(self, completion: _CompletedMisses) -> None:
        """Checked diagnostic/test boundary around the trusted publisher."""

        if not isinstance(completion, _CompletedMisses):
            raise TypeError("publish requires a completed PLE row ticket")
        if completion._owner_token is not self._owner_token:
            raise ValueError("completed PLE row ticket belongs to another handoff")
        self.publish(completion)

    def _payload_from_packed_row(self, packed_row: Any) -> tuple[np.ndarray, ...]:
        owner = np.array(
            packed_row,
            dtype=np.uint8,
            order="C",
            copy=True,
        ).reshape(PACKED_ROW_BYTES)
        owner.flags.writeable = False
        payload = []
        for start, dtype, shape, row_bytes in self._row_specs:
            view = np.frombuffer(
                owner,
                dtype=dtype,
                count=row_bytes // dtype.itemsize,
                offset=start,
            ).reshape(shape)
            view.flags.writeable = False
            payload.append(view)
        return tuple(payload)


def bind_stock_cache(sidecar: Any) -> CachedRowHandoff:
    """Capture one existing sidecar cache without creating a second cache."""

    return CachedRowHandoff(sidecar)


__all__ = [
    "CachedRowHandoff",
    "MAX_ROW_SLOTS",
    "PACKED_ROW_BYTES",
    "PreparedRows",
    "bind_stock_cache",
    "pack_row_payload",
]
