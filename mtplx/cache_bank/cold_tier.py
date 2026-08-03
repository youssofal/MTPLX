"""Persistent SSD cold tier for exact SessionBank boundary snapshots."""

from __future__ import annotations

import hashlib
from collections import deque
import json
import logging
import os
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtplx.cache_state import CacheSnapshot

from .codec import decode_gdn_boundaries, decode_payload, encode_payload


logger = logging.getLogger(__name__)


def _eval_payload_trees(*trees: Any) -> None:
    """Force-evaluate every MLX array reachable from the given trees."""
    import mlx.core as mx

    arrays: list[Any] = []
    seen: set[int] = set()

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, mx.array):
            if id(value) not in seen:
                seen.add(id(value))
                arrays.append(value)
            return
        if isinstance(value, CacheSnapshot):
            collect(value.states)
            collect(value.meta_states)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)

    for tree in trees:
        collect(tree)
    if arrays:
        mx.eval(*arrays)

# v3 (kvcache-v2, 2026-07-03): payload carries interior recurrent boundaries
# (token_count, recurrent-only snapshot, hidden_last) plus a has_recurrent
# identity flag, and encode moved off the caller thread (deferred writer-side
# encode). v2 stores go through the existing legacy-archive migration.
COLD_TIER_FORMAT_VERSION = 3
DEFAULT_COLD_TIER_DIR = Path("~/.mtplx/session-bank").expanduser()
DEFAULT_COLD_TIER_MAX_BYTES = 100 * 1024**3
DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS = 512
DEFAULT_BLOCK_SIZE = 256
DISK_USAGE_CACHE_TTL_S = 30.0
_COMMITTED_CACHE_POLICIES = frozenset({"committed", "last_window"})


def _deferred_encode_enabled() -> bool:
    """RETIRED (#169, 2026-07-17) — writer-side encode is gone; always False.

    The kvcache-v2 writer-thread encode block-sliced tensors at write time
    (TreeCodec builds lazy slice arrays and mx.eval()s them), which crashed
    on restore-derived arrays ("There is no Stream(gpu, 1) in current
    thread") and could serialize donation-mutated KV pages from the writer
    backlog. put_entry now always encodes at enqueue on the owner thread.
    MTPLX_SSD_DEFERRED_ENCODE is parsed nowhere and ignored."""
    return False


@dataclass(frozen=True)
class DeferredPayload:
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    mtp_history_snapshot: CacheSnapshot | None
    gdn_boundaries: tuple[tuple[int, CacheSnapshot, Any], ...]
    has_recurrent: bool
    block_size: int


@dataclass(frozen=True)
class PendingWrite:
    entry_id: str
    token_ids: tuple[int, ...]
    metadata: dict[str, Any]
    payload_spec: dict[str, Any] | None
    tensors: dict[str, bytes]
    deferred: DeferredPayload | None = None
    created_at_s: float = field(default_factory=time.time)
    # Estimated bytes this write pins in memory until the writer drains it
    # (deferred payloads hold live KV arrays; encoded ones hold the buffers).
    pinned_nbytes: int = 0


@dataclass(frozen=True)
class ColdRestoreRecord:
    entry_id: str
    token_ids: tuple[int, ...]
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    mtp_history_snapshot: CacheSnapshot | None
    metadata: dict[str, Any]
    nbytes: int
    restore_s: float
    gdn_boundaries: tuple[tuple[int, CacheSnapshot, Any], ...] = ()
    has_recurrent: bool = False
    # Lazy loader for boundaries skipped at exact-restore time (callable -> tuple).
    gdn_boundary_loader: Any = None


@dataclass(frozen=True)
class ColdPrefixRestoreRecord:
    record: ColdRestoreRecord
    matched_tokens: int
    restore_kind: str


GIB = 1024**3
LOW_DISK_FLOOR_BYTES = 10 * GIB


def detect_total_ram_bytes() -> int | None:
    try:
        import subprocess

        out = subprocess.run(
            # Absolute path: app-owned daemons run with a sanitized PATH
            # that lacks /usr/sbin (see engine_session RAM detection).
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
        )
        value = int(out.stdout.strip())
        return value if value > 0 else None
    except Exception:
        try:
            import os as _os

            return int(_os.sysconf("SC_PAGE_SIZE")) * int(_os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            return None


def default_cold_tier_max_bytes() -> int:
    """RAM-tier-scaled default SSD cap (kvcache-v2 P2.3).

    A 16 GB Mac should not default to a 100 GB session store. Bands: <=16 GB
    RAM -> 16 GB cap, <=32 -> 24 GB, <=64 -> 32 GB, else 100 GB (the legacy
    flat default for big-RAM machines that also tend to have big disks).
    """
    ram = detect_total_ram_bytes()
    if ram is None:
        return DEFAULT_COLD_TIER_MAX_BYTES
    if ram <= 16 * GIB:
        return 16 * GIB
    if ram <= 32 * GIB:
        return 24 * GIB
    if ram <= 64 * GIB:
        return 32 * GIB
    return DEFAULT_COLD_TIER_MAX_BYTES


def _env_size_bytes(name: str, default: int) -> int:
    return parse_size_bytes(os.environ.get(name), default)


def parse_size_bytes(value: str | int | None, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return max(1, int(value))
    raw = str(value).strip()
    if not raw:
        return int(default)
    normalized = raw.upper().replace("IB", "B")
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024**2,
        "M": 1024**2,
        "GB": 1024**3,
        "G": 1024**3,
        "TB": 1024**4,
        "T": 1024**4,
    }
    try:
        for suffix, multiplier in sorted(
            suffixes.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if normalized.endswith(suffix):
                number = normalized[: -len(suffix)].strip()
                return max(1, int(float(number) * multiplier))
        return max(1, int(float(normalized)))
    except (TypeError, ValueError):
        # "auto"/garbage falls back to the caller's default (which is already
        # RAM-tiered for the cold tier) instead of failing daemon startup.
        return int(default)


def token_hash(token_ids: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def block_aligned_prefix_len(matched_tokens: int, *, block_size: int) -> int:
    block = max(1, int(block_size))
    matched = max(0, int(matched_tokens))
    return (matched // block) * block


def chain_block_hashes(
    token_ids: tuple[int, ...],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    identity: dict[str, Any] | None = None,
) -> list[str]:
    parent = ""
    identity = identity or {}
    hashes: list[str] = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        h = hashlib.sha256()
        h.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(parent.encode("utf-8"))
        for token in block:
            h.update(int(token).to_bytes(8, byteorder="little", signed=True))
        parent = h.hexdigest()
        hashes.append(parent)
    return hashes


class SessionBankColdTier:
    """Async SSD persistence for committed SessionBank snapshots.

    The foreground/model-owner thread calls :meth:`put_entry`, which evaluates
    arrays and copies them into immutable bytes before enqueueing. The writer
    thread only writes files and updates SQLite; it never sees live MLX arrays.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path = DEFAULT_COLD_TIER_DIR,
        mode: str = "off",
        max_bytes: int = DEFAULT_COLD_TIER_MAX_BYTES,
        min_prefix_tokens: int = DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
        writer_queue_depth: int = 32,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.mode = _normalize_mode(mode)
        self.max_bytes = int(max(1, max_bytes))
        self.min_prefix_tokens = int(max(1, min_prefix_tokens))
        self.block_size = int(max(1, block_size))
        self._queue: queue.Queue[PendingWrite | None] = queue.Queue(
            maxsize=max(1, int(writer_queue_depth))
        )
        # Backlog byte cap (issue #145): every queued write pins its payload
        # (deferred ones pin LIVE KV arrays) until the writer drains it. A
        # count-bounded queue of 32 multi-GB snapshots can pin ~50 GB under
        # distinct-prefix churn — measured live 2026-07-09 (active memory
        # climbed 35 -> 66 GB while the bank ledger stayed flat). Cap the
        # pinned bytes, drop new writes beyond it.
        self._pending_bytes = 0
        self._backlog_budget_bytes = _env_size_bytes(
            "MTPLX_SSD_WRITER_BACKLOG_BYTES", 4 * 1024**3
        )
        # Hourly write budget (issue #144: 7 TB written / SSD wear): the
        # different-repos pattern writes GBs per task and never restores
        # them (measured 58 GB in 45 min with restore_hits=0). Rolling
        # one-hour byte budget; beyond it new writes are skipped.
        self._write_budget_per_hour_bytes = _env_size_bytes(
            "MTPLX_SSD_WRITE_BUDGET_PER_HOUR", 64 * 1024**3
        )
        self._written_window: deque[tuple[float, int]] = deque()
        self._stop = threading.Event()
        self._base_lock = threading.RLock()
        self._disk_usage_lock = threading.Lock()
        self._disk_usage_cache: dict[str, int | float] | None = None
        self._disk_usage_scan_running = False
        self._orphan_cleanup_running = False
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int | float | str | bool | None] = {
            "format_version": COLD_TIER_FORMAT_VERSION,
            "mode": self.mode,
            "dir": str(self.base_dir),
            "max_bytes": self.max_bytes,
            "min_prefix_tokens": self.min_prefix_tokens,
            "block_size": self.block_size,
            "writes_enqueued": 0,
            "writes_completed": 0,
            "write_failures": 0,
            "write_only_skips": 0,
            "skipped_too_short": 0,
            "skipped_queue_full": 0,
            "skipped_size_cap": 0,
            "skipped_serialize_error": 0,
            "deduped_blob_hits": 0,
            "entries_evicted": 0,
            "restore_hits": 0,
            "restore_misses": 0,
            "restore_failures": 0,
            "corrupt_entries": 0,
            "orphan_cleanup_runs": 0,
            "orphan_cleanup_files_deleted": 0,
            "orphan_cleanup_dirs_deleted": 0,
            "orphan_cleanup_disk_bytes_deleted": 0,
            "orphan_cleanup_last_s": 0.0,
            "last_write_s": None,
            "last_restore_s": None,
            "last_miss_reason": None,
            "last_archive_path": None,
        }
        self._ensure_store()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="mtplx-sessionbank-ssd-writer",
            daemon=True,
        )
        self._writer.start()

    @property
    def enabled(self) -> bool:
        return self.mode in {"on", "write-only"}

    @property
    def restorable(self) -> bool:
        return self.mode == "on"

    def put_entry(
        self,
        entry: Any,
        *,
        capabilities: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        if self.mode == "off":
            return False
        token_ids = tuple(int(token) for token in getattr(entry, "token_ids"))
        if len(token_ids) < self.min_prefix_tokens:
            self._inc("skipped_too_short")
            return False
        estimated_nbytes = int(getattr(entry, "nbytes", 0) or 0)
        if not self._admit_write(estimated_nbytes):
            return False
        boundaries = tuple(
            (int(r[0]), r[1], r[2] if len(r) > 2 else None)
            for r in (getattr(entry, "gdn_boundaries", None) or [])
        )
        # Encode ALWAYS happens here, on the enqueueing (owner) thread, never
        # on the writer thread (#169, 2026-07-17). The retired writer-side
        # "deferred encode" (kvcache-v2) block-sliced tensors at write time:
        # TreeCodec._encode_tensor_blocks builds new lazy slice arrays and
        # mx.eval()s them, which (a) crashed the process on restore-derived
        # arrays whose graphs referenced the restore stream ("There is no
        # Stream(gpu, 1) in current thread"), and (b) held live KV references
        # for seconds in the writer backlog, so under buffer donation the
        # eventual serialization could capture mutated pages — silently
        # corrupt persisted sessions that degrade on every restore. Bytes are
        # captured at snapshot time; the writer thread is pure file IO.
        try:
            encoded = encode_payload(
                cache_snapshot=getattr(entry, "cache_snapshot"),
                logits=getattr(entry, "logits"),
                hidden=getattr(entry, "hidden"),
                mtp_history_snapshot=getattr(entry, "mtp_history_snapshot", None),
                gdn_boundaries=boundaries,
                has_recurrent=bool(getattr(entry, "has_recurrent", False)),
                block_size=self.block_size,
            )
        except Exception as exc:
            self._inc("skipped_serialize_error")
            logger.warning("SessionBank SSD serialize skipped: %s: %s", type(exc).__name__, exc)
            return False
        metadata = self._metadata_for_entry(
            entry,
            capabilities=capabilities or (),
            payload_nbytes=encoded.nbytes,
        )
        pending = PendingWrite(
            entry_id=str(metadata["entry_id"]),
            token_ids=token_ids,
            metadata=metadata,
            payload_spec=encoded.spec,
            tensors=encoded.tensors,
            pinned_nbytes=max(estimated_nbytes, int(encoded.nbytes)),
        )
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            self._release_pending(pending.pinned_nbytes)
            self._inc("skipped_queue_full")
            logger.warning(
                "SessionBank SSD writer queue full; skipping prefix_len=%d token_hash=%s",
                len(token_ids),
                metadata["token_hash"],
            )
            return False
        self._inc("writes_enqueued")
        return True

    def lookup(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        model_path: str,
        mtp_enabled: bool,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
    ) -> ColdRestoreRecord | None:
        if self.mode == "off":
            self._set_last_miss("ssd_cache_off")
            return None
        if self.mode == "write-only":
            self._inc("write_only_skips")
            self._set_last_miss("ssd_cache_write_only")
            return None
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            self._set_last_miss("ssd_empty_lookup")
            return None
        started = time.perf_counter()
        try:
            rows = self._candidate_rows(
                tokens,
                model_path=model_path,
                mtp_enabled=mtp_enabled,
                hidden_variant=hidden_variant,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
            )
            for row in rows:
                record = self._restore_row(row, tokens, started_s=started)
                if record is not None:
                    return record
        except Exception as exc:
            self._inc("restore_failures")
            self._set_last_miss(f"ssd_restore_error:{type(exc).__name__}")
            logger.warning("SessionBank SSD restore failed: %s: %s", type(exc).__name__, exc)
            return None
        self._inc("restore_misses")
        self._set_last_miss("ssd_prefix_miss")
        return None

    def lookup_prefix_boundary(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        model_path: str,
        mtp_enabled: bool,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        max_token_gap: int = 8,
        min_matched_tokens: int = 64,
        block_size: int = DEFAULT_BLOCK_SIZE,
        block_min_matched_tokens: int = DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
        allow_block_prefix: bool = True,
    ) -> ColdPrefixRestoreRecord | None:
        if self.mode == "off":
            self._set_last_miss("ssd_cache_off")
            return None
        if self.mode == "write-only":
            self._inc("write_only_skips")
            self._set_last_miss("ssd_cache_write_only")
            return None
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            self._set_last_miss("ssd_empty_lookup")
            return None
        gap_limit = max(0, int(max_token_gap))
        min_match = max(1, int(min_matched_tokens))
        block = max(1, int(block_size))
        block_min_match = max(block, int(block_min_matched_tokens))
        started = time.perf_counter()
        try:
            rows = self._candidate_rows_for_prefix_boundary(
                model_path=model_path,
                mtp_enabled=mtp_enabled,
                hidden_variant=hidden_variant,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
            )
            best: tuple[sqlite3.Row, int, str] | None = None
            best_key: tuple[int, int, int] | None = None
            for row in rows:
                prefix = tuple(int(token) for token in json.loads(str(row["token_ids_json"])))
                if not prefix:
                    continue
                matched = common_prefix_len(tokens, prefix)
                gap = len(prefix) - matched
                safe_block = min(
                    block_aligned_prefix_len(matched, block_size=block),
                    len(prefix),
                    len(tokens),
                )
                required_match = min(min_match, max(1, len(prefix) - gap_limit))
                near_match = gap >= 0 and gap <= gap_limit and matched >= required_match
                block_match = (
                    bool(allow_block_prefix)
                    and safe_block >= block_min_match
                    and safe_block >= 2
                    and safe_block <= matched
                )
                if near_match:
                    candidate_matched = int(matched)
                    restore_kind = "near_prefix"
                elif block_match:
                    candidate_matched = int(safe_block)
                    restore_kind = "block_prefix"
                else:
                    continue
                candidate_key = (candidate_matched, int(matched), len(prefix))
                if best_key is None or candidate_key > best_key:
                    best = (row, candidate_matched, restore_kind)
                    best_key = candidate_key
            if best is None:
                self._inc("restore_misses")
                self._set_last_miss("ssd_prefix_miss")
                return None
            record = self._restore_row(
                best[0],
                tokens,
                started_s=started,
                require_exact_prefix=False,
                include_gdn_boundaries=True,
            )
            if record is None:
                return None
            return ColdPrefixRestoreRecord(
                record=record,
                matched_tokens=int(best[1]),
                restore_kind=str(best[2]),
            )
        except Exception as exc:
            self._inc("restore_failures")
            self._set_last_miss(f"ssd_restore_error:{type(exc).__name__}")
            logger.warning("SessionBank SSD prefix-boundary restore failed: %s: %s", type(exc).__name__, exc)
            return None

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._stats_lock:
            pending_bytes = int(self._pending_bytes)
            written_last_hour = sum(nbytes for _, nbytes in self._written_window)
        stats.update(
            {
                "enabled": self.enabled,
                "restorable": self.restorable,
                "writer_queue_depth": int(self._queue.qsize()),
                "writer_backlog_bytes": pending_bytes,
                "writer_backlog_budget_bytes": int(self._backlog_budget_bytes),
                "written_bytes_last_hour": int(written_last_hour),
                "write_budget_per_hour_bytes": int(self._write_budget_per_hour_bytes),
                "dir": str(self.base_dir),
                "manifest_path": str(self._manifest_path),
            }
        )
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), "
                    "COALESCE(SUM(CASE WHEN logical_nbytes > 0 "
                    "THEN logical_nbytes ELSE nbytes END), 0), "
                    "COALESCE(SUM(CASE WHEN physical_nbytes > 0 "
                    "THEN physical_nbytes ELSE nbytes END), 0), "
                    "COALESCE(SUM(deduped_nbytes), 0) FROM entries"
                ).fetchone()
            stats["entries"] = int(row[0])
            stats["logical_bytes"] = int(row[1])
            stats["bytes"] = int(row[2])
            stats["physical_bytes"] = int(row[2])
            stats["live_physical_bytes"] = int(row[2])
            stats["deduped_bytes"] = int(row[3])
            logical = max(1, int(row[1]))
            stats["dedupe_ratio"] = max(0.0, float(row[3]) / float(logical))
            usage = self._managed_disk_usage()
            managed_file_bytes = int(usage.get("managed_file_bytes", 0))
            managed_disk_bytes = int(usage.get("managed_disk_bytes", 0))
            database_file_bytes = int(usage.get("database_file_bytes", 0))
            database_disk_bytes = int(usage.get("database_disk_bytes", 0))
            if managed_file_bytes <= 0 and bool(usage.get("disk_usage_scan_pending")):
                managed_file_bytes = int(row[2])
                managed_disk_bytes = int(row[2])
                usage["managed_file_bytes"] = managed_file_bytes
                usage["managed_disk_bytes"] = managed_disk_bytes
            stats.update(usage)
            # Pair a cached filesystem scan with the manifest total captured
            # in that same snapshot. Mixing stale filesystem bytes with the
            # live manifest row creates phantom orphan bytes while the next
            # asynchronous scan is pending.
            manifest_bytes_at_scan = int(
                usage.get("manifest_physical_bytes_at_scan", row[2])
            )
            stats["untracked_file_bytes"] = max(
                0,
                managed_file_bytes - database_file_bytes - manifest_bytes_at_scan,
            )
            stats["untracked_disk_bytes"] = max(
                0,
                managed_disk_bytes - database_disk_bytes - manifest_bytes_at_scan,
            )
            stats["orphan_cleanup_running"] = self._orphan_cleanup_is_running()
            if (
                self.enabled
                and managed_disk_bytes > self.max_bytes
                and int(stats["untracked_disk_bytes"]) > 0
                and not bool(usage.get("disk_usage_scan_pending"))
                and not bool(usage.get("disk_usage_stale"))
            ):
                self._start_orphan_cleanup()
        except sqlite3.Error as exc:
            stats["entries_error"] = str(exc)
        return stats

    def flush(self, *, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        # ``Queue.empty()`` becomes true as soon as the writer dequeues an
        # item, before that item has reached disk or updated the manifest.
        # Wait on Queue's task accounting instead so a successful flush means
        # every accepted write has actually finished.
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
        return True

    def cancel_pending(self) -> int:
        """Drop queued writes without encoding them.

        Used by the admin cache-clear quiesce: once the RAM bank has been
        cleared, queued PendingWrite items describe state the operator just
        asked to discard, and deferred-encode payloads pin multi-GB cache
        snapshots until the writer thread gets to them (a 128k-token entry
        encodes for minutes and starves foreground decode bandwidth — the
        post-long-row slowdown measured 2026-07-05: 20 tok/s with the
        backlog live vs 75 tok/s after dropping it). An in-flight encode, if
        any, finishes on its own; everything behind it is discarded.
        """

        dropped = 0
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                break
            if pending is None:
                # Preserve shutdown sentinels for the writer thread.
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass
                break
            dropped += 1
            self._queue.task_done()
        if dropped:
            with self._stats_lock:
                self._stats["writes_cancelled"] = (
                    int(self._stats.get("writes_cancelled") or 0) + dropped
                )
        return dropped

    def archive(self) -> dict[str, Any]:
        self.flush(timeout_s=10.0)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with self._base_lock:
            source = self.base_dir
            archive_path = source.with_name(f"{source.name}-archive-{timestamp}")
            if source.exists():
                suffix = 0
                candidate = archive_path
                while candidate.exists():
                    suffix += 1
                    candidate = source.with_name(f"{source.name}-archive-{timestamp}-{suffix}")
                source.rename(candidate)
                archive_path = candidate
            self._ensure_store()
        with self._stats_lock:
            self._stats["last_archive_path"] = str(archive_path)
        logger.info("SessionBank SSD cache archived to %s", archive_path)
        return {
            "archived": True,
            "archive_path": str(archive_path),
            "active_dir": str(self.base_dir),
        }

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._writer.join(timeout=5.0)

    def _metadata_for_entry(
        self,
        entry: Any,
        *,
        capabilities: list[str] | tuple[str, ...],
        payload_nbytes: int,
    ) -> dict[str, Any]:
        token_ids = tuple(int(token) for token in getattr(entry, "token_ids"))
        identity = {
            "model_path": str(getattr(entry, "model_path", "")),
            "mtp_enabled": bool(getattr(entry, "mtp_enabled", False)),
            "hidden_variant": getattr(entry, "hidden_variant", None),
            "template_hash": getattr(entry, "template_hash", None),
            "mtp_history_policy": getattr(entry, "mtp_history_policy", None),
            "draft_head_identity": getattr(entry, "draft_head_identity", None),
            "policy_fingerprint": getattr(entry, "policy_fingerprint", None),
            "session_id": getattr(entry, "session_id", None),
        }
        digest = hashlib.sha256()
        digest.update(token_hash(token_ids).encode("utf-8"))
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(str(int(getattr(entry, "snapshot_epoch", 0) or 0)).encode("ascii"))
        digest.update(str(int(getattr(entry, "mtp_snapshot_epoch", 0) or 0)).encode("ascii"))
        entry_id = digest.hexdigest()[:32]
        block_identity = {
            key: value
            for key, value in identity.items()
            if key != "session_id"
        }
        block_hashes = chain_block_hashes(
            token_ids,
            block_size=self.block_size,
            identity=block_identity,
        )
        nbytes = int(max(int(getattr(entry, "nbytes", 0) or 0), int(payload_nbytes)))
        return {
            "entry_id": entry_id,
            "format_version": COLD_TIER_FORMAT_VERSION,
            "token_hash": token_hash(token_ids),
            "prefix_len": len(token_ids),
            "token_ids": list(token_ids),
            "model_path": identity["model_path"],
            "mtp_enabled": identity["mtp_enabled"],
            "hidden_variant": identity["hidden_variant"],
            "template_hash": identity["template_hash"],
            "mtp_history_policy": identity["mtp_history_policy"],
            "draft_head_identity": identity["draft_head_identity"],
            "policy_fingerprint": identity["policy_fingerprint"],
            "session_id": identity["session_id"],
            "snapshot_epoch": int(getattr(entry, "snapshot_epoch", 0) or 0),
            "mtp_snapshot_epoch": (
                int(getattr(entry, "mtp_snapshot_epoch"))
                if getattr(entry, "mtp_snapshot_epoch", None) is not None
                else None
            ),
            "capabilities": sorted({str(item) for item in capabilities}),
            "has_recurrent": bool(getattr(entry, "has_recurrent", False)),
            "gdn_boundary_count": len(getattr(entry, "gdn_boundaries", None) or []),
            "nbytes": nbytes,
            "block_size": self.block_size,
            "block_hashes": block_hashes,
            "created_at_s": time.time(),
            "logical_nbytes": int(payload_nbytes),
            "physical_nbytes": int(payload_nbytes),
            "deduped_nbytes": 0,
        }

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            pending = self._queue.get()
            if pending is None:
                self._queue.task_done()
                break
            try:
                wrote = self._write_pending(pending)
                if wrote:
                    self._inc("writes_completed")
                    with self._stats_lock:
                        self._stats["last_write_s"] = time.time()
                        self._written_window.append(
                            (time.time(), int(pending.pinned_nbytes))
                        )
                    logger.info(
                        "SessionBank SSD wrote entry_id=%s prefix_len=%d nbytes=%d",
                        pending.entry_id,
                        int(pending.metadata["prefix_len"]),
                        int(pending.metadata["nbytes"]),
                    )
            except Exception as exc:
                self._inc("write_failures")
                logger.warning(
                    "SessionBank SSD write failed entry_id=%s: %s: %s",
                    pending.entry_id,
                    type(exc).__name__,
                    exc,
                )
            finally:
                self._release_pending(pending.pinned_nbytes)
                self._queue.task_done()

    def _admit_write(self, estimated_nbytes: int) -> bool:
        """Backlog + hourly-budget admission for a new SSD write."""

        now = time.time()
        with self._stats_lock:
            if (
                self._pending_bytes + max(0, estimated_nbytes)
                > self._backlog_budget_bytes
            ):
                self._stats["skipped_backlog_bytes"] = (
                    int(self._stats.get("skipped_backlog_bytes", 0) or 0) + 1
                )
                return False
            while self._written_window and self._written_window[0][0] < now - 3600:
                self._written_window.popleft()
            written_last_hour = sum(nbytes for _, nbytes in self._written_window)
            if (
                written_last_hour + max(0, estimated_nbytes)
                > self._write_budget_per_hour_bytes
            ):
                self._stats["skipped_write_budget"] = (
                    int(self._stats.get("skipped_write_budget", 0) or 0) + 1
                )
                return False
            self._pending_bytes += max(0, estimated_nbytes)
        return True

    def _release_pending(self, nbytes: int) -> None:
        with self._stats_lock:
            self._pending_bytes = max(0, self._pending_bytes - max(0, int(nbytes)))

    def _write_pending(self, pending: PendingWrite) -> bool:
        if pending.deferred is not None:
            # Retired path (#169, 2026-07-17): writer-side encode ran MLX
            # slice/eval graph work on the writer thread (crash on
            # restore-stream arrays, donation-corruption window). put_entry
            # now always encodes at enqueue; a deferred payload reaching the
            # writer is a programming error, never silently encoded here.
            self._inc("skipped_deferred_retired")
            logger.error(
                "SessionBank SSD writer received a deferred payload "
                "entry_id=%s; writer-side encode is retired (#169), skipping",
                pending.entry_id,
            )
            return False
        with self._base_lock:
            self._ensure_store()
            entry_hash_prefix = pending.entry_id[:2]
            final_dir = self.base_dir / "entries" / entry_hash_prefix / pending.entry_id
            if final_dir.exists():
                if self._entry_in_manifest(pending.entry_id):
                    self._touch_entry(pending.entry_id)
                    return True
                self._archive_orphan_entry_dir(final_dir, pending.entry_id)
            effective_cap, budget_block = self._effective_write_budget()
            if budget_block is not None:
                self._inc("skipped_low_disk")
                with self._stats_lock:
                    self._stats["low_disk_writes_disabled"] = True
                logger.warning(
                    "SessionBank SSD writes disabled (%s): free disk below %d GiB",
                    budget_block,
                    LOW_DISK_FLOOR_BYTES // GIB,
                )
                return False
            with self._stats_lock:
                self._stats["low_disk_writes_disabled"] = False
                self._stats["effective_max_bytes"] = int(effective_cap)
            tensor_blobs, missing_blob_bytes = self._plan_tensor_blobs(pending.tensors)
            payload = {
                "format_version": COLD_TIER_FORMAT_VERSION,
                "metadata": pending.metadata,
                "payload_spec": pending.payload_spec,
                "tensor_names": sorted(pending.tensors),
                "tensor_blobs": tensor_blobs,
            }
            payload_bytes = len(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            logical_bytes = sum(int(item["nbytes"]) for item in tensor_blobs.values())
            pending_bytes = int(missing_blob_bytes + payload_bytes)
            if pending_bytes > effective_cap:
                self._inc("skipped_size_cap")
                logger.warning(
                    "SessionBank SSD size cap skipped entry_id=%s prefix_len=%d pending=%d max=%d",
                    pending.entry_id,
                    int(pending.metadata["prefix_len"]),
                    pending_bytes,
                    self.max_bytes,
                )
                return False
            if not self._evict_until_room(pending_bytes, cap_bytes=effective_cap):
                self._inc("skipped_size_cap")
                return False
            temp_parent = self.base_dir / "entries" / entry_hash_prefix
            temp_parent.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(prefix=f".{pending.entry_id}.tmp-", dir=temp_parent))
            (temp_dir / "payload.json").write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            for name, raw in pending.tensors.items():
                blob = tensor_blobs[name]
                if self._write_blob(blob["sha256"], raw):
                    continue
                self._inc("deduped_blob_hits")
            temp_dir.rename(final_dir)
            metadata = dict(pending.metadata)
            metadata["entry_dir"] = str(final_dir.relative_to(self.base_dir))
            metadata["logical_nbytes"] = int(logical_bytes)
            metadata["physical_nbytes"] = int(pending_bytes)
            metadata["deduped_nbytes"] = max(0, int(logical_bytes) - int(pending_bytes))
            self._insert_manifest(metadata)
            self._invalidate_disk_usage_cache()
            return True

    def _plan_tensor_blobs(
        self,
        tensors: dict[str, bytes],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        blobs: dict[str, dict[str, Any]] = {}
        missing_bytes = 0
        planned_missing: set[str] = set()
        for name, raw in tensors.items():
            digest = hashlib.sha256(raw).hexdigest()
            blobs[name] = {"sha256": digest, "nbytes": len(raw)}
            if digest in planned_missing:
                continue
            if not self._blob_path(digest).exists():
                planned_missing.add(digest)
                missing_bytes += len(raw)
        return blobs, missing_bytes

    def _blob_path(self, digest: str) -> Path:
        return self.base_dir / "blobs" / digest[:2] / f"{digest}.bin"

    def _write_blob(self, digest: str, raw: bytes) -> bool:
        path = self._blob_path(digest)
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
        temp_path.write_bytes(raw)
        try:
            temp_path.rename(path)
        except FileExistsError:
            return False
        return True

    def _evict_until_room(self, required_bytes: int, *, cap_bytes: int | None = None) -> bool:
        required = max(0, int(required_bytes))
        cap = int(self.max_bytes if cap_bytes is None else cap_bytes)
        current = self._current_bytes_for_cap(required)
        if current + required <= cap:
            return True
        with self._connect() as conn:
            rows = list(
                conn.execute(
                    "SELECT entry_id, entry_dir, physical_nbytes, nbytes, last_access_s "
                    "FROM entries ORDER BY last_access_s ASC"
                ).fetchall()
            )
        for row in rows:
            self._delete_entry_row(row)
            current -= int(row["physical_nbytes"] or row["nbytes"] or 0)
            self._inc("entries_evicted")
            if current + required <= cap:
                return True
        return current + required <= cap

    def _effective_write_budget(self) -> tuple[int, str | None]:
        """min(configured cap, free_disk/4), writes disabled under 10 GiB free.

        Re-checked on every write (cheap statvfs); guards strangers' Macs where
        a flat configured cap could fill the disk (kvcache-v2 P2.3)."""
        try:
            free = shutil.disk_usage(self.base_dir).free
        except Exception:
            return self.max_bytes, None
        if free < LOW_DISK_FLOOR_BYTES:
            return 0, "low_disk"
        return min(int(self.max_bytes), int(free // 4)), None

    def _current_bytes_for_cap(self, required_bytes: int = 0) -> int:
        required = max(0, int(required_bytes))
        manifest_bytes = self._current_bytes()
        try:
            usage = self._managed_disk_usage(force=True)
        except Exception as exc:
            logger.warning(
                "SessionBank SSD disk usage scan failed during cap check: %s: %s",
                type(exc).__name__,
                exc,
            )
            return manifest_bytes
        managed_file_bytes = int(usage.get("managed_file_bytes", 0) or 0)
        database_file_bytes = int(usage.get("database_file_bytes", 0) or 0)
        managed_cache_bytes = max(0, managed_file_bytes - database_file_bytes)
        untracked_bytes = max(
            0,
            managed_cache_bytes - manifest_bytes,
        )
        if (
            self.enabled
            and managed_cache_bytes + required > self.max_bytes
            and untracked_bytes > 0
        ):
            try:
                cleanup = self._cleanup_untracked_cache_once()
                self._record_orphan_cleanup_result(cleanup)
                usage = self._managed_disk_usage(force=True)
                managed_file_bytes = int(usage.get("managed_file_bytes", 0) or 0)
                database_file_bytes = int(
                    usage.get("database_file_bytes", 0) or 0
                )
                managed_cache_bytes = max(
                    0,
                    managed_file_bytes - database_file_bytes,
                )
            except Exception as exc:
                logger.warning(
                    "SessionBank SSD orphan cleanup failed during cap check: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        return max(manifest_bytes, managed_cache_bytes)

    def _delete_entry_row(self, row: sqlite3.Row) -> None:
        entry_id = str(row["entry_id"])
        entry_dir = self.base_dir / str(row["entry_dir"])
        blob_hashes = self._entry_blob_hashes(entry_dir)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        self._delete_unreferenced_blobs(blob_hashes)
        self._invalidate_disk_usage_cache()

    def _archive_entry_row(self, row: sqlite3.Row) -> None:
        entry_id = str(row["entry_id"])
        entry_dir = self.base_dir / str(row["entry_dir"])
        if entry_dir.exists():
            self._archive_orphan_entry_dir(entry_dir, entry_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        self._invalidate_disk_usage_cache()

    def _entry_in_manifest(self, entry_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        return row is not None

    def _archive_orphan_entry_dir(self, entry_dir: Path, entry_id: str) -> None:
        archive_parent = self.base_dir / "evicted_entries" / entry_id[:2]
        archive_parent.mkdir(parents=True, exist_ok=True)
        target = archive_parent / f"{int(time.time())}-{entry_id}"
        suffix = 0
        candidate = target
        while candidate.exists():
            suffix += 1
            candidate = archive_parent / f"{int(time.time())}-{entry_id}-{suffix}"
        entry_dir.rename(candidate)
        self._invalidate_disk_usage_cache()

    def _entry_blob_hashes(self, entry_dir: Path) -> set[str]:
        payload_path = entry_dir / "payload.json"
        if not payload_path.exists():
            return set()
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        tensor_blobs = payload.get("tensor_blobs") or {}
        blob_hashes: set[str] = set()
        if isinstance(tensor_blobs, dict):
            for blob in tensor_blobs.values():
                if not isinstance(blob, dict):
                    continue
                digest = blob.get("sha256")
                if digest:
                    blob_hashes.add(str(digest))
        return blob_hashes

    def _manifest_blob_hashes(self) -> set[str]:
        with self._connect() as conn:
            rows = list(conn.execute("SELECT entry_dir FROM entries").fetchall())
        blob_hashes: set[str] = set()
        for row in rows:
            entry_dir = self.base_dir / str(row["entry_dir"])
            blob_hashes.update(self._entry_blob_hashes(entry_dir))
        return blob_hashes

    def _delete_unreferenced_blobs(self, candidate_hashes: set[str]) -> None:
        if not candidate_hashes:
            return
        still_referenced = self._manifest_blob_hashes()
        for digest in sorted(candidate_hashes - still_referenced):
            path = self._blob_path(digest)
            if not path.exists():
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "SessionBank SSD blob cleanup failed digest=%s: %s: %s",
                    digest[:12],
                    type(exc).__name__,
                    exc,
                )
                continue
            self._prune_empty_parents(path.parent, stop_at=self.base_dir / "blobs")

    def _orphan_cleanup_is_running(self) -> bool:
        with self._disk_usage_lock:
            return bool(self._orphan_cleanup_running)

    def _start_orphan_cleanup(self) -> None:
        with self._disk_usage_lock:
            if self._orphan_cleanup_running:
                return
            self._orphan_cleanup_running = True
        threading.Thread(
            target=self._orphan_cleanup_worker,
            name="mtplx-sessionbank-orphan-cleanup",
            daemon=True,
        ).start()

    def _orphan_cleanup_worker(self) -> None:
        try:
            result = self._cleanup_untracked_cache_once()
            self._record_orphan_cleanup_result(result)
        except Exception as exc:  # pragma: no cover - defensive background task
            logger.warning(
                "SessionBank SSD orphan cleanup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
        finally:
            self._invalidate_disk_usage_cache()
            with self._disk_usage_lock:
                self._orphan_cleanup_running = False

    def _record_orphan_cleanup_result(self, result: dict[str, int | float]) -> None:
        with self._stats_lock:
            self._stats["orphan_cleanup_runs"] = int(
                self._stats.get("orphan_cleanup_runs", 0) or 0
            ) + 1
            self._stats["orphan_cleanup_files_deleted"] = int(
                self._stats.get("orphan_cleanup_files_deleted", 0) or 0
            ) + int(result["files_deleted"])
            self._stats["orphan_cleanup_dirs_deleted"] = int(
                self._stats.get("orphan_cleanup_dirs_deleted", 0) or 0
            ) + int(result["dirs_deleted"])
            self._stats["orphan_cleanup_disk_bytes_deleted"] = int(
                self._stats.get("orphan_cleanup_disk_bytes_deleted", 0) or 0
            ) + int(result["disk_bytes_deleted"])
            self._stats["orphan_cleanup_last_s"] = float(result["elapsed_s"])

    def _cleanup_untracked_cache_once(self) -> dict[str, int | float]:
        started = time.perf_counter()
        files_deleted = 0
        dirs_deleted = 0
        disk_bytes_deleted = 0
        with self._base_lock:
            manifest_entry_dirs = self._manifest_entry_dirs()
            manifest_blob_hashes = self._manifest_blob_hashes()

            for root in (self.base_dir / "evicted_entries",):
                removed = self._remove_tree_counting(root)
                files_deleted += int(removed["files_deleted"])
                dirs_deleted += int(removed["dirs_deleted"])
                disk_bytes_deleted += int(removed["disk_bytes_deleted"])

            entries_root = self.base_dir / "entries"
            if entries_root.exists():
                for prefix_dir in list(entries_root.iterdir()):
                    if not prefix_dir.is_dir():
                        continue
                    for entry_dir in list(prefix_dir.iterdir()):
                        if not entry_dir.is_dir():
                            continue
                        try:
                            rel = str(entry_dir.relative_to(self.base_dir))
                        except ValueError:
                            continue
                        if rel in manifest_entry_dirs:
                            continue
                        removed = self._remove_tree_counting(entry_dir)
                        files_deleted += int(removed["files_deleted"])
                        dirs_deleted += int(removed["dirs_deleted"])
                        disk_bytes_deleted += int(removed["disk_bytes_deleted"])
                    self._prune_empty_parents(prefix_dir, stop_at=entries_root)

            blobs_root = self.base_dir / "blobs"
            if blobs_root.exists():
                for prefix_dir in list(blobs_root.iterdir()):
                    if not prefix_dir.is_dir():
                        continue
                    for path in list(prefix_dir.glob("*.bin")):
                        digest = path.stem
                        if digest in manifest_blob_hashes:
                            continue
                        try:
                            disk_bytes_deleted += self._allocated_bytes(path)
                            path.unlink()
                            files_deleted += 1
                        except FileNotFoundError:
                            continue
                        except OSError as exc:
                            logger.warning(
                                "SessionBank SSD orphan blob cleanup failed path=%s: %s: %s",
                                path,
                                type(exc).__name__,
                                exc,
                            )
                    self._prune_empty_parents(prefix_dir, stop_at=blobs_root)
        self._invalidate_disk_usage_cache()
        return {
            "files_deleted": int(files_deleted),
            "dirs_deleted": int(dirs_deleted),
            "disk_bytes_deleted": int(disk_bytes_deleted),
            "elapsed_s": float(time.perf_counter() - started),
        }

    def _manifest_entry_dirs(self) -> set[str]:
        with self._connect() as conn:
            rows = list(conn.execute("SELECT entry_dir FROM entries").fetchall())
        return {str(row["entry_dir"]) for row in rows}

    @staticmethod
    def _remove_tree_counting(path: Path) -> dict[str, int]:
        if not path.exists():
            return {"files_deleted": 0, "dirs_deleted": 0, "disk_bytes_deleted": 0}
        files_deleted = 0
        dirs_deleted = 0
        disk_bytes_deleted = 0
        for root, dirs, files in os.walk(path):
            dirs_deleted += len(dirs)
            for filename in files:
                file_path = Path(root) / filename
                try:
                    disk_bytes_deleted += SessionBankColdTier._allocated_bytes(file_path)
                    files_deleted += 1
                except FileNotFoundError:
                    continue
        try:
            shutil.rmtree(path)
            dirs_deleted += 1
        except FileNotFoundError:
            pass
        return {
            "files_deleted": int(files_deleted),
            "dirs_deleted": int(dirs_deleted),
            "disk_bytes_deleted": int(disk_bytes_deleted),
        }

    @staticmethod
    def _allocated_bytes(path: Path) -> int:
        stat = path.stat()
        blocks = int(getattr(stat, "st_blocks", 0) or 0)
        return blocks * 512 if blocks > 0 else int(stat.st_size)

    @staticmethod
    def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
        current = path
        stop = stop_at.resolve()
        while True:
            try:
                if current.resolve() == stop:
                    return
                current.rmdir()
            except (FileNotFoundError, OSError):
                return
            current = current.parent

    def _restore_row(
        self,
        row: sqlite3.Row,
        lookup_tokens: tuple[int, ...],
        *,
        started_s: float,
        require_exact_prefix: bool = True,
        include_gdn_boundaries: bool = False,
    ) -> ColdRestoreRecord | None:
        metadata = dict(row)
        token_ids = tuple(int(token) for token in json.loads(str(metadata["token_ids_json"])))
        if require_exact_prefix and lookup_tokens[: len(token_ids)] != token_ids:
            return None
        if int(metadata["format_version"]) != COLD_TIER_FORMAT_VERSION:
            self._set_last_miss("ssd_format_mismatch")
            return None
        mtp_snapshot_epoch = metadata.get("mtp_snapshot_epoch")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(metadata["snapshot_epoch"]):
            self._set_last_miss("ssd_mtp_epoch_mismatch")
            return None
        entry_dir = self.base_dir / str(metadata["entry_dir"])
        payload_path = entry_dir / "payload.json"
        if not payload_path.exists():
            self._inc("corrupt_entries")
            self._set_last_miss("ssd_payload_missing")
            return None
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        tensor_blobs = dict(payload.get("tensor_blobs") or {})

        def read_tensor(name: str) -> bytes:
            blob = tensor_blobs.get(name)
            if blob:
                path = self._blob_path(str(blob["sha256"]))
                if not path.exists():
                    raise FileNotFoundError(str(path))
                return path.read_bytes()
            path = entry_dir / "tensors" / f"{name}.bin"
            if not path.exists():
                raise FileNotFoundError(str(path))
            return path.read_bytes()

        # Exact-prefix restores never rewind below the stored boundary, so the
        # (MB-scale x boundary-count) interior snapshots are decoded only for
        # partial restores — keeps exact restart-warm on the fast path.
        decoded = decode_payload(
            payload["payload_spec"],
            read_tensor,
            include_gdn_boundaries=include_gdn_boundaries,
        )
        restore_s = time.perf_counter() - started_s
        self._inc("restore_hits")
        with self._stats_lock:
            self._stats["last_restore_s"] = time.time()
            self._stats["last_miss_reason"] = None
        self._touch_entry(str(metadata["entry_id"]))
        logger.info(
            "SessionBank SSD restored entry_id=%s prefix_len=%d restore_s=%.4f",
            metadata["entry_id"],
            len(token_ids),
            restore_s,
        )
        metadata["capabilities"] = json.loads(str(metadata.get("capabilities_json") or "[]"))
        metadata["block_hashes"] = json.loads(str(metadata.get("block_hashes_json") or "[]"))
        payload_spec = payload["payload_spec"]
        boundary_loader = (
            None
            if include_gdn_boundaries
            else (lambda: decode_gdn_boundaries(payload_spec, read_tensor))
        )
        return ColdRestoreRecord(
            entry_id=str(metadata["entry_id"]),
            token_ids=token_ids,
            cache_snapshot=decoded.cache_snapshot,
            logits=decoded.logits,
            hidden=decoded.hidden,
            mtp_history_snapshot=decoded.mtp_history_snapshot,
            metadata=metadata,
            nbytes=int(metadata["nbytes"]),
            restore_s=restore_s,
            gdn_boundaries=tuple(decoded.gdn_boundaries or ()),
            has_recurrent=bool(decoded.has_recurrent),
            gdn_boundary_loader=boundary_loader,
        )

    def _candidate_rows(
        self,
        tokens: tuple[int, ...],
        *,
        model_path: str,
        mtp_enabled: bool,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
    ) -> list[sqlite3.Row]:
        query = [
            "SELECT * FROM entries WHERE model_path = ?",
            "AND mtp_enabled = ?",
            "AND prefix_len <= ?",
        ]
        params: list[Any] = [str(model_path), 1 if mtp_enabled else 0, len(tokens)]
        if hidden_variant is not None:
            query.append("AND hidden_variant = ?")
            params.append(str(hidden_variant))
        if template_hash is not None:
            query.append("AND template_hash = ?")
            params.append(str(template_hash))
        if draft_head_identity is not None:
            query.append("AND draft_head_identity = ?")
            params.append(str(draft_head_identity))
        if policy_fingerprint is not None:
            query.append("AND policy_fingerprint = ?")
            params.append(str(policy_fingerprint))
        query.append("ORDER BY prefix_len DESC, last_access_s DESC")
        with self._connect() as conn:
            rows = list(conn.execute(" ".join(query), params).fetchall())
        if mtp_history_policy is None:
            return rows
        compatible: list[sqlite3.Row] = []
        for row in rows:
            if _policy_compatible(row["mtp_history_policy"], mtp_history_policy):
                compatible.append(row)
        return compatible

    def _candidate_rows_for_prefix_boundary(
        self,
        *,
        model_path: str,
        mtp_enabled: bool,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
    ) -> list[sqlite3.Row]:
        query = [
            "SELECT * FROM entries WHERE model_path = ?",
            "AND mtp_enabled = ?",
        ]
        params: list[Any] = [str(model_path), 1 if mtp_enabled else 0]
        if hidden_variant is not None:
            query.append("AND hidden_variant = ?")
            params.append(str(hidden_variant))
        if template_hash is not None:
            query.append("AND template_hash = ?")
            params.append(str(template_hash))
        if draft_head_identity is not None:
            query.append("AND draft_head_identity = ?")
            params.append(str(draft_head_identity))
        if policy_fingerprint is not None:
            query.append("AND policy_fingerprint = ?")
            params.append(str(policy_fingerprint))
        query.append("ORDER BY prefix_len DESC, last_access_s DESC")
        with self._connect() as conn:
            rows = list(conn.execute(" ".join(query), params).fetchall())
        if mtp_history_policy is None:
            return rows
        compatible: list[sqlite3.Row] = []
        for row in rows:
            if _policy_compatible(row["mtp_history_policy"], mtp_history_policy):
                compatible.append(row)
        return compatible

    def _ensure_store(self) -> None:
        self._archive_legacy_store_if_needed()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "entries").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "blobs").mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    entry_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    prefix_len INTEGER NOT NULL,
                    token_ids_json TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    mtp_enabled INTEGER NOT NULL,
                    hidden_variant TEXT,
                    template_hash TEXT,
                    mtp_history_policy TEXT,
                    draft_head_identity TEXT,
                    policy_fingerprint TEXT,
                    session_id TEXT,
                    snapshot_epoch INTEGER NOT NULL,
                    mtp_snapshot_epoch INTEGER,
                    capabilities_json TEXT NOT NULL,
                    block_size INTEGER NOT NULL,
                    block_hashes_json TEXT NOT NULL,
                    entry_dir TEXT NOT NULL,
                    nbytes INTEGER NOT NULL,
                    logical_nbytes INTEGER NOT NULL DEFAULT 0,
                    physical_nbytes INTEGER NOT NULL DEFAULT 0,
                    deduped_nbytes INTEGER NOT NULL DEFAULT 0,
                    created_at_s REAL NOT NULL,
                    last_access_s REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    format_version INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_lookup "
                "ON entries(model_path, mtp_enabled, prefix_len DESC, last_access_s DESC)"
            )
            self._ensure_column(conn, "entries", "session_id", "TEXT")
            self._ensure_column(conn, "entries", "logical_nbytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "entries", "physical_nbytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "entries", "deduped_nbytes", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_token_hash "
                "ON entries(token_hash)"
            )

    def _archive_legacy_store_if_needed(self) -> None:
        manifest = self._manifest_path
        if not manifest.exists():
            return
        try:
            with sqlite3.connect(str(manifest), timeout=5.0) as conn:
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
                ).fetchone()
                if table is None:
                    return
                row = conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE format_version != ?",
                    (COLD_TIER_FORMAT_VERSION,),
                ).fetchone()
                legacy_count = int(row[0] or 0)
        except sqlite3.Error:
            legacy_count = 1
        if legacy_count <= 0:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        source = self.base_dir
        archive_path = source.with_name(f"{source.name}-legacy-v1-archive-{timestamp}")
        suffix = 0
        candidate = archive_path
        while candidate.exists():
            suffix += 1
            candidate = source.with_name(
                f"{source.name}-legacy-v1-archive-{timestamp}-{suffix}"
            )
        source.rename(candidate)
        with self._stats_lock:
            self._stats["last_archive_path"] = str(candidate)
            self._stats["last_miss_reason"] = "legacy_ssd_cache_archived"
        logger.info("Archived legacy SessionBank SSD cache to %s", candidate)

    @property
    def _manifest_path(self) -> Path:
        return self.base_dir / "manifest.sqlite"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._manifest_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _current_bytes(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN physical_nbytes > 0 "
                "THEN physical_nbytes ELSE nbytes END), 0) FROM entries"
            ).fetchone()
        return int(row[0] or 0)

    def _managed_disk_usage(self, *, force: bool = False) -> dict[str, int | float]:
        now = time.time()
        with self._disk_usage_lock:
            cached = self._disk_usage_cache
            if (
                not force
                and cached is not None
                and now - float(cached["disk_usage_last_scan_s"]) < DISK_USAGE_CACHE_TTL_S
            ):
                fresh = dict(cached)
                fresh["disk_usage_scan_pending"] = False
                fresh["disk_usage_stale"] = False
                return fresh
            if not force:
                self._start_disk_usage_scan_locked()
                if cached is not None:
                    stale = dict(cached)
                    stale["disk_usage_scan_pending"] = True
                    stale["disk_usage_stale"] = True
                    return stale
                return self._empty_disk_usage(scan_pending=True)
        return self._refresh_disk_usage_now()

    def _start_disk_usage_scan_locked(self) -> None:
        if self._disk_usage_scan_running:
            return
        self._disk_usage_scan_running = True
        threading.Thread(
            target=self._disk_usage_scan_worker,
            name="mtplx-sessionbank-disk-usage-scan",
            daemon=True,
        ).start()

    def _disk_usage_scan_worker(self) -> None:
        try:
            self._refresh_disk_usage_now()
        finally:
            with self._disk_usage_lock:
                self._disk_usage_scan_running = False

    def _refresh_disk_usage_now(self) -> dict[str, int | float]:
        # The writer mutates entry directories, blobs, and the manifest as one
        # logical transaction under ``_base_lock``. Scanning without the same
        # lock could cache a half-written view as fresh, making disk telemetry
        # briefly report phantom untracked bytes and potentially trigger an
        # unnecessary orphan cleanup. The scan already runs off the hot path;
        # waiting for the current write keeps the snapshot coherent.
        with self._base_lock:
            manifest_bytes_at_scan = self._current_bytes()
            usage = self._scan_managed_disk_usage()
            usage["manifest_physical_bytes_at_scan"] = manifest_bytes_at_scan
            # Keep the writer excluded until the coherent snapshot has been
            # installed. Otherwise a write can invalidate the old cache in
            # the gap and this older scan can overwrite it as fresh.
            with self._disk_usage_lock:
                self._disk_usage_cache = dict(usage)
        return dict(usage)

    @staticmethod
    def _empty_disk_usage(*, scan_pending: bool) -> dict[str, int | float]:
        return {
            "managed_file_bytes": 0,
            "managed_disk_bytes": 0,
            "database_file_bytes": 0,
            "database_disk_bytes": 0,
            "manifest_physical_bytes_at_scan": 0,
            "managed_file_count": 0,
            "managed_dir_count": 0,
            "disk_usage_scan_s": 0.0,
            "disk_usage_last_scan_s": 0.0,
            "disk_usage_scan_pending": bool(scan_pending),
            "disk_usage_stale": bool(scan_pending),
        }

    def _scan_managed_disk_usage(self) -> dict[str, int | float]:
        now = time.time()
        started = time.perf_counter()
        file_bytes = 0
        disk_bytes = 0
        database_file_bytes = 0
        database_disk_bytes = 0
        file_count = 0
        dir_count = 0
        for root, dirs, files in os.walk(self.base_dir):
            dir_count += len(dirs)
            for filename in files:
                path = Path(root) / filename
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                file_count += 1
                file_bytes += int(stat.st_size)
                blocks = int(getattr(stat, "st_blocks", 0) or 0)
                allocated = blocks * 512 if blocks > 0 else int(stat.st_size)
                disk_bytes += allocated
                if filename.startswith("manifest.sqlite"):
                    database_file_bytes += int(stat.st_size)
                    database_disk_bytes += int(allocated)
        usage: dict[str, int | float] = {
            "managed_file_bytes": int(file_bytes),
            "managed_disk_bytes": int(disk_bytes),
            "database_file_bytes": int(database_file_bytes),
            "database_disk_bytes": int(database_disk_bytes),
            "managed_file_count": int(file_count),
            "managed_dir_count": int(dir_count),
            "disk_usage_scan_s": float(time.perf_counter() - started),
            "disk_usage_last_scan_s": float(now),
            "disk_usage_scan_pending": False,
            "disk_usage_stale": False,
        }
        return usage

    def _invalidate_disk_usage_cache(self) -> None:
        with self._disk_usage_lock:
            if self._disk_usage_cache is not None:
                self._disk_usage_cache["disk_usage_last_scan_s"] = 0.0
                self._disk_usage_cache["disk_usage_stale"] = True

    def _insert_manifest(self, metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entries (
                    entry_id, token_hash, prefix_len, token_ids_json,
                    model_path, mtp_enabled, hidden_variant, template_hash,
                    mtp_history_policy, draft_head_identity, policy_fingerprint,
                    session_id, snapshot_epoch, mtp_snapshot_epoch, capabilities_json,
                    block_size, block_hashes_json, entry_dir, nbytes,
                    logical_nbytes, physical_nbytes, deduped_nbytes,
                    created_at_s, last_access_s, hits, format_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata["entry_id"],
                    metadata["token_hash"],
                    int(metadata["prefix_len"]),
                    json.dumps(metadata["token_ids"], separators=(",", ":")),
                    metadata["model_path"],
                    1 if metadata["mtp_enabled"] else 0,
                    metadata.get("hidden_variant"),
                    metadata.get("template_hash"),
                    metadata.get("mtp_history_policy"),
                    metadata.get("draft_head_identity"),
                    metadata.get("policy_fingerprint"),
                    metadata.get("session_id"),
                    int(metadata["snapshot_epoch"]),
                    metadata.get("mtp_snapshot_epoch"),
                    json.dumps(metadata["capabilities"], separators=(",", ":")),
                    int(metadata["block_size"]),
                    json.dumps(metadata["block_hashes"], separators=(",", ":")),
                    metadata["entry_dir"],
                    int(metadata["nbytes"]),
                    int(metadata.get("logical_nbytes") or metadata["nbytes"]),
                    int(metadata.get("physical_nbytes") or metadata["nbytes"]),
                    int(metadata.get("deduped_nbytes") or 0),
                    float(metadata["created_at_s"]),
                    time.time(),
                    0,
                    COLD_TIER_FORMAT_VERSION,
                ),
            )

    def _touch_entry(self, entry_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE entries SET last_access_s = ?, hits = hits + 1 WHERE entry_id = ?",
                (time.time(), entry_id),
            )

    def _inc(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0) or 0) + int(amount)

    def _set_last_miss(self, reason: str) -> None:
        with self._stats_lock:
            self._stats["last_miss_reason"] = reason


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "off").strip().lower().replace("_", "-")
    if normalized not in {"off", "on", "write-only"}:
        raise ValueError("ssd session cache mode must be off, on, or write-only")
    return normalized


def _policy_compatible(entry_policy: str | None, lookup_policy: str | None) -> bool:
    if entry_policy == lookup_policy:
        return True
    if entry_policy is None or lookup_policy is None:
        return False
    return entry_policy in _COMMITTED_CACHE_POLICIES and lookup_policy in _COMMITTED_CACHE_POLICIES
