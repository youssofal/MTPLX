"""Cheap counters for the SSD cold tier: where a restore spends its time and
how much the writer puts on disk.

MLX-free so it runs (and is tested) anywhere.

Restore split. A restore reads blobs from disk, copies each into an MLX
array (``mx.array`` over the bytes: a host copy into unified memory) and
then evaluates the decoded tree on the GPU (the concatenate of a tensor's
blocks runs there). :func:`restore_timing` makes a :class:`RestoreTimings`
current for the decode; the codec adds to it at those three points and does
nothing when none is current, so the lazy recurrent-boundary loader and every
other decode stay untimed. Three ``perf_counter`` calls per tensor.

Write volume. :class:`WriteVolume` counts the bytes the writer puts on disk
and the bytes it did not write because the blob was already there, split
into recurrent-checkpoint (GDN boundary) blobs and the rest (attention KV
plus the final recurrent state, logits and hidden), in one-minute buckets
over a rolling hour. One lock and a few additions per entry written.

Write split. A deduplicated blob was found one of two ways: referenced, when
a memo already named its digest and its bytes were never evaluated, copied
or hashed; or hashed, when its bytes were captured and hashed and only then
found on disk. :class:`WriteSplit` counts both per write, with the capture
and hash seconds the hashed duplicates cost, next to the blobs that were new.
Two ``perf_counter`` calls per captured blob.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Iterator
from typing import Any

_ACTIVE: contextvars.ContextVar[RestoreTimings | None] = contextvars.ContextVar(
    "mtplx_ssd_restore_timings", default=None
)


class RestoreTimings:
    """Seconds and bytes of one SSD restore, filled by the codec."""

    __slots__ = ("bytes_read", "copy_s", "eval_s", "read_s")

    def __init__(self) -> None:
        self.read_s = 0.0
        self.copy_s = 0.0
        self.eval_s = 0.0
        self.bytes_read = 0

    def as_dict(self, *, restore_s: float | None = None) -> dict[str, Any]:
        """The split as the restore record carries it. ``other_s`` is what
        ``restore_s`` holds beyond the three parts: the manifest query, the
        payload file and rows refused before their tensors were read."""
        out: dict[str, Any] = {
            "read_s": round(self.read_s, 6),
            "copy_s": round(self.copy_s, 6),
            "eval_s": round(self.eval_s, 6),
            "bytes": int(self.bytes_read),
        }
        if restore_s is not None:
            out["other_s"] = round(
                max(0.0, float(restore_s) - self.read_s - self.copy_s - self.eval_s), 6
            )
        return out


def active_restore_timings() -> RestoreTimings | None:
    return _ACTIVE.get()


@contextlib.contextmanager
def restore_timing() -> Iterator[RestoreTimings]:
    timings = RestoreTimings()
    token = _ACTIVE.set(timings)
    try:
        yield timings
    finally:
        _ACTIVE.reset(token)


def gdn_boundary_tensor_names(payload_spec: dict[str, Any] | None) -> set[str]:
    """Names of the tensors that hold the payload's recurrent checkpoints."""
    names: set[str] = set()
    stack: list[Any] = list((payload_spec or {}).get("gdn_boundaries") or [])
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            kind = node.get("kind")
            if kind == "tensor":
                names.add(str(node["name"]))
            elif kind == "tensor_blocks":
                names.update(str(block["name"]) for block in node.get("blocks") or [])
            else:
                stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return names


class WriteSplit:
    """One write's blobs by how they reached the payload: ``referenced``
    (a memo hit, never captured or hashed), ``hashed_duplicate`` (captured
    and hashed, then found on disk) and ``hashed_new`` (captured, hashed and
    written)."""

    __slots__ = (
        "duplicate_blobs",
        "duplicate_bytes",
        "duplicate_capture_s",
        "duplicate_hash_s",
        "new_blobs",
        "new_bytes",
        "new_capture_s",
        "new_hash_s",
        "referenced_blobs",
        "referenced_bytes",
    )

    def __init__(self) -> None:
        self.referenced_blobs = 0
        self.referenced_bytes = 0
        self.duplicate_blobs = 0
        self.duplicate_bytes = 0
        self.duplicate_capture_s = 0.0
        self.duplicate_hash_s = 0.0
        self.new_blobs = 0
        self.new_bytes = 0
        self.new_capture_s = 0.0
        self.new_hash_s = 0.0

    def referenced(self, nbytes: int) -> None:
        self.referenced_blobs += 1
        self.referenced_bytes += int(nbytes)

    def hashed(
        self, nbytes: int, *, capture_s: float, hash_s: float, on_disk: bool
    ) -> None:
        if on_disk:
            self.duplicate_blobs += 1
            self.duplicate_bytes += int(nbytes)
            self.duplicate_capture_s += float(capture_s)
            self.duplicate_hash_s += float(hash_s)
        else:
            self.new_blobs += 1
            self.new_bytes += int(nbytes)
            self.new_capture_s += float(capture_s)
            self.new_hash_s += float(hash_s)

    def counters(self) -> dict[str, int | float]:
        """The tier's cumulative counter names and this write's values."""
        return {
            "write_referenced_blobs": self.referenced_blobs,
            "write_referenced_bytes": self.referenced_bytes,
            "write_hashed_duplicate_blobs": self.duplicate_blobs,
            "write_hashed_duplicate_bytes": self.duplicate_bytes,
            "write_hashed_duplicate_capture_s": self.duplicate_capture_s,
            "write_hashed_duplicate_hash_s": self.duplicate_hash_s,
            "write_hashed_new_blobs": self.new_blobs,
            "write_hashed_new_bytes": self.new_bytes,
            "write_hashed_new_capture_s": self.new_capture_s,
            "write_hashed_new_hash_s": self.new_hash_s,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "referenced": {
                "blobs": self.referenced_blobs,
                "bytes": self.referenced_bytes,
            },
            "hashed_duplicate": {
                "blobs": self.duplicate_blobs,
                "bytes": self.duplicate_bytes,
                "capture_s": round(self.duplicate_capture_s, 6),
                "hash_s": round(self.duplicate_hash_s, 6),
            },
            "hashed_new": {
                "blobs": self.new_blobs,
                "bytes": self.new_bytes,
                "capture_s": round(self.new_capture_s, 6),
                "hash_s": round(self.new_hash_s, 6),
            },
        }


WINDOW_S = 3600
BUCKET_S = 60


class WriteVolume:
    """Bytes written and deduplicated, per kind, over a rolling hour."""

    def __init__(self, *, clock: Any = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # minute index -> [written, deduped] per kind, flattened, then the
        # deduped bytes again by how they were found: [referenced, hashed]
        self._buckets: dict[int, list[int]] = {}
        self._totals = [0, 0, 0, 0, 0, 0]

    def record(
        self,
        tensor_blobs: dict[str, dict[str, Any]],
        deduped_names: set[str],
        gdn_names: set[str],
        referenced_names: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """One entry's blobs: ``deduped_names`` were already on disk, and of
        those ``referenced_names`` were never captured or hashed."""
        add = [0, 0, 0, 0, 0, 0]
        for name, blob in tensor_blobs.items():
            nbytes = int(blob.get("nbytes") or 0)
            slot = 0 if name in gdn_names else 2
            if name in deduped_names:
                slot += 1
                add[4 if name in referenced_names else 5] += nbytes
            add[slot] += nbytes
        minute = int(self._clock() // BUCKET_S)
        with self._lock:
            bucket = self._buckets.setdefault(minute, [0, 0, 0, 0, 0, 0])
            for index, value in enumerate(add):
                bucket[index] += value
                self._totals[index] += value
            oldest = minute - WINDOW_S // BUCKET_S
            for key in [key for key in self._buckets if key <= oldest]:
                del self._buckets[key]

    def snapshot(self, *, budget_per_hour_bytes: int) -> dict[str, Any]:
        oldest = int(self._clock() // BUCKET_S) - WINDOW_S // BUCKET_S
        hour = [0, 0, 0, 0, 0, 0]
        with self._lock:
            for minute, bucket in self._buckets.items():
                if minute > oldest:
                    for index, value in enumerate(bucket):
                        hour[index] += value
            totals = list(self._totals)
        written = hour[0] + hour[2]
        budget = max(1, int(budget_per_hour_bytes))
        return {
            "written_bytes_last_hour": written,
            "deduped_bytes_last_hour": hour[1] + hour[3],
            "gdn_boundary_written_bytes_last_hour": hour[0],
            "gdn_boundary_deduped_bytes_last_hour": hour[1],
            "kv_written_bytes_last_hour": hour[2],
            "kv_deduped_bytes_last_hour": hour[3],
            "deduped_referenced_bytes_last_hour": hour[4],
            "deduped_hashed_bytes_last_hour": hour[5],
            "write_budget_per_hour_bytes": int(budget_per_hour_bytes),
            "budget_used_fraction": round(written / budget, 4),
            "written_bytes_total": totals[0] + totals[2],
            "deduped_bytes_total": totals[1] + totals[3],
            "gdn_boundary_written_bytes_total": totals[0],
            "kv_written_bytes_total": totals[2],
            "deduped_referenced_bytes_total": totals[4],
            "deduped_hashed_bytes_total": totals[5],
        }


ORIGIN_SAME = "same_session"
ORIGIN_OTHER = "other_session"
ORIGIN_UNKNOWN = "unknown"


def entry_origin(entry_session_id: str | None, session_id: str | None) -> str:
    """Whether a restored entry was saved by the requesting conversation or
    by another one sharing its prompt head (system prompt, tool list)."""
    if not entry_session_id or not session_id:
        return ORIGIN_UNKNOWN
    return ORIGIN_SAME if str(entry_session_id) == str(session_id) else ORIGIN_OTHER


_LANDING_SUMS = (
    "matched_tokens",
    "restore_point_tokens",
    "checkpoint_gap_tokens",
    "suffix_tokens",
)


class RestoreLandings:
    """Where session-bank restores land, per source (RAM or SSD).

    A landing is one served restore: the tokens the prompt matched in the
    entry, the token the state resumed at (on a hybrid model the newest
    recurrent checkpoint at or below the match), the gap between the two
    (matched tokens computed again because no checkpoint sat closer), the
    suffix prefilled after it, and the entry's origin.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_source: dict[str, dict[str, int]] = {}
        self._last: dict[str, Any] | None = None

    def record(self, landing: dict[str, Any]) -> None:
        source = str(landing.get("source") or "ram")
        with self._lock:
            row = self._by_source.setdefault(
                source,
                {
                    "restores": 0,
                    "exact": 0,
                    "partial": 0,
                    "hybrid": 0,
                    ORIGIN_SAME: 0,
                    ORIGIN_OTHER: 0,
                    ORIGIN_UNKNOWN: 0,
                    **{key: 0 for key in _LANDING_SUMS},
                },
            )
            row["restores"] += 1
            row["partial" if landing.get("partial") else "exact"] += 1
            if landing.get("hybrid"):
                row["hybrid"] += 1
            origin = str(landing.get("origin") or ORIGIN_UNKNOWN)
            row[origin if origin in row else ORIGIN_UNKNOWN] += 1
            for key in _LANDING_SUMS:
                row[key] += max(0, int(landing.get(key) or 0))
            self._last = dict(landing)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {
                source: dict(row) for source, row in self._by_source.items()
            }
            out["last"] = dict(self._last) if self._last is not None else None
        return out
