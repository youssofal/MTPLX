"""Persistent SSD cold tier for exact SessionBank boundary snapshots."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, deque
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
from typing import Any, Callable

from mtplx.cache_state import CacheSnapshot

from .codec import (
    ColdEncodeInterrupted,
    TreeCodec,
    build_payload_spec,
    decode_gdn_boundaries,
    decode_payload,
    decode_payload_prefix,
    encode_payload,
    payload_supports_prefix_decode,
    snapshot_supports_prefix_decode,
)
from .disk_budget import DISK_FLOOR_BYTES, DISK_FULL, DiskBudget, disk_budget
from .reconcile import (
    DEFAULT_COLD_TIER_DIR,
    ReconcileReport,
    entry_blob_hashes as _entry_blob_hashes_of,
    manifest_blob_hashes as _manifest_blob_hashes_of,
    manifest_entry_dirs as _manifest_entry_dirs_of,
    prune_empty_parents as _prune_empty_parents_of,
    reconcile_store,
)


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
# DEFAULT_COLD_TIER_DIR is defined once, in .reconcile (MLX-free), and
# re-exported here for the server and the package namespace.
DEFAULT_COLD_TIER_MAX_BYTES = 100 * 1024**3
DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS = 512
DEFAULT_BLOCK_SIZE = 256
DISK_USAGE_CACHE_TTL_S = 30.0
# stats() manifest-aggregate snapshot TTL. /health and the dashboard poll
# stats() continuously; the aggregate is exact while _store_generation is
# unchanged, so the TTL only bounds staleness against out-of-process writers
# sharing the directory.
MANIFEST_STATS_TTL_S = 5.0
# A rescan is due only when the store changed since the last scan AND at
# least DUTY_DIVISOR x the last scan's own duration has passed, so the
# reconciliation walk can never occupy more than 1/DUTY_DIVISOR of a core
# regardless of how many blobs a long-lived bank has accumulated
# (measured 2026-08-15: 816k files, 41.7 s per walk on an M5 Max — under the
# previous 30 s TTL that walk ran back to back whenever /health was polled).
DISK_USAGE_SCAN_DUTY_DIVISOR = 20
# The walk yields to foreground traffic every this many files.
DISK_USAGE_SCAN_YIELD_EVERY_FILES = 4096
# An encode unit (one eval plus one host copy of at most 32 MiB) that takes
# this long is reported: it is the bound on how long a request waits.
ENCODE_SLOW_UNIT_S = 1.0
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
    # Incremental encode (MTPLX_SSD_INCREMENTAL_ENCODE): names referenced to
    # blobs the store already holds ({"sha256", "nbytes"}, no bytes here),
    # the content key of every fingerprinted name, and the memo to update.
    reused_blobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    fingerprints: dict[str, tuple] = field(default_factory=dict)
    memo_key: str | None = None


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
    # Prefix lengths represented by the hydrated snapshots.  A cold
    # sub-prefix restore retains the entry's full token identity, while its
    # decoded KV only covers the boundary it will actually serve.
    cache_snapshot_prefix_len: int | None = None
    mtp_history_snapshot_prefix_len: int | None = None


@dataclass(frozen=True)
class ColdPrefixRestoreRecord:
    record: ColdRestoreRecord
    matched_tokens: int
    restore_kind: str


GIB = 1024**3
# Free disk the tier never writes into (the policy lives in .disk_budget).
LOW_DISK_FLOOR_BYTES = DISK_FLOOR_BYTES

# _install_entry outcomes.
_INSTALL_INSTALLED = "installed"
_INSTALL_PRESENT = "present"  # the same entry was already in the manifest
_INSTALL_FAILED = "failed"


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
        try:
            free = shutil.disk_usage(DEFAULT_COLD_TIER_DIR).free
            if free >= 150 * GIB:
                return DEFAULT_COLD_TIER_MAX_BYTES
        except Exception:
            pass
        return 32 * GIB
    return DEFAULT_COLD_TIER_MAX_BYTES


def _env_size_bytes(name: str, default: int) -> int:
    return parse_size_bytes(os.environ.get(name), default)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw.strip())
    except ValueError:
        return float(default)
    if value != value or value in (float("inf"), float("-inf")) or value < 0.0:
        return float(default)
    return value


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


# Ranked SSD rows a prefix lookup opens before it gives up: each refusal
# costs only its payload.json read.
_PREFIX_BOUNDARY_ROWS_TRIED = 4


def _payload_boundary_at_or_below(
    payload_spec: dict[str, Any], prefix_len: int
) -> int | None:
    """Return the newest persisted recurrent boundary at or below ``prefix_len``.

    This deliberately inspects JSON metadata only.  The matching boundary's
    tensors are decoded later by ``decode_payload_prefix``; no large blob is
    read merely to choose a safe restore point.
    """

    best: int | None = None
    for record in payload_spec.get("gdn_boundaries") or []:
        try:
            tokens = int(record.get("tokens", 0))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 < tokens <= int(prefix_len) and (best is None or tokens > best):
            best = tokens
    return best


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

    # Capability marker for SessionBank: lookup_prefix_boundary accepts the
    # resident_duplicates shadow kwarg. Explicit attribute so callers never
    # need per-request signature inspection; duck-typed tiers without it get
    # the pre-shadow call shape.
    SUPPORTS_RESIDENT_DUPLICATE_SHADOW = True
    SUPPORTS_MIN_USEFUL_MATCHED_TOKENS = True
    SUPPORTS_MIN_RESTORE_POINT = True

    def __init__(
        self,
        *,
        base_dir: str | Path = DEFAULT_COLD_TIER_DIR,
        mode: str = "off",
        max_bytes: int = DEFAULT_COLD_TIER_MAX_BYTES,
        min_prefix_tokens: int = DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
        writer_queue_depth: int = 32,
        block_size: int = DEFAULT_BLOCK_SIZE,
        reconcile_listener: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser()
        # Called once with the summary of the reconciliation pass that runs
        # when the store opens (#493). The server wires its structured
        # stdout event here; it is a constructor argument because the pass
        # starts inside __init__ and a small store finishes it in
        # milliseconds, before any attribute set afterwards could be seen.
        self.reconcile_listener = reconcile_listener
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
            "MTPLX_SSD_WRITE_BUDGET_PER_HOUR", 128 * 1024**3
        )
        self._written_window: deque[tuple[float, int]] = deque()
        # Foreground-yield contract (2026-08-07): the server wires this to
        # ModelWorkScheduler.foreground_busy so both halves of the SSD path
        # stand down for latency-critical traffic — the encode aborts between
        # tensor evals (it runs on the model-owner thread; a 16k entry is
        # ~2.5 GB of eval+copy) and the writer thread pauses between entry
        # writes (a 2.5 GB file write steals unified-memory bandwidth from
        # decode: measured -30% decode with 0.66-0.75 s unattributed
        # prompt-state wall, gate254-c4s receipts 2026-08-07). None (e.g.
        # standalone/test construction) keeps legacy behavior.
        self.foreground_busy: Callable[[], bool] | None = None
        self._writer_pause_enabled = _env_flag(
            "MTPLX_SSD_WRITER_FOREGROUND_PAUSE", default=True
        )
        # 600 s, not 60: coding-agent turns run 60-620 s, so a 60 s bound
        # expired mid-decode and each blob write fired under the live turn —
        # a ~1 GB/min unified-memory drumbeat that tripped macOS pressure
        # (banner) and stole decode (gate254-c4s -30%; manifest receipts
        # 2026-08-28 21:38-22:17, 9+ GB written across live OpenCode turns).
        # The cold tier is a cache: waiting out even a 10-minute turn costs
        # only delayed durability, never correctness. Inter-turn gaps drain
        # the queue (the pause loop samples every 50 ms).
        self._writer_pause_max_s = _env_float(
            "MTPLX_SSD_WRITER_FOREGROUND_PAUSE_MAX_S", 600.0
        )
        self._encode_yield_enabled = _env_flag(
            "MTPLX_SSD_ENCODE_FOREGROUND_YIELD", default=True
        )
        # Incremental encode (opt-in, MTPLX_SSD_INCREMENTAL_ENCODE=1): each
        # agent turn re-encodes the whole session on the model-owner thread
        # (eval + host copy of every KV block) and the writer re-hashes every
        # blob, although only the new tail is new on disk. With this on,
        # put_entry fingerprints every KV block on the GPU (one reduction per
        # tensor) and references a block whose content key matches the
        # session's last completed write instead of capturing it. Off by
        # default: the fingerprint is not a content identity. Flipping the
        # signs of any two float32 elements leaves both of its sums unchanged,
        # so a changed tensor can borrow its predecessor's blob and restore
        # the wrong state (tests/test_cold_tier_incremental_encode.py).
        self._incremental_encode = _env_flag(
            "MTPLX_SSD_INCREMENTAL_ENCODE", default=False
        )
        # session_id -> {content key: {"sha256", "nbytes"}} of the newest
        # completed write; written by the writer thread, read at encode.
        self._block_memos: OrderedDict[str, dict[tuple, dict[str, Any]]] = OrderedDict()
        self._block_memos_lock = threading.Lock()
        self._block_memos_max = 16
        self._stop = threading.Event()
        self._base_lock = threading.RLock()
        self._disk_usage_lock = threading.Lock()
        self._disk_usage_cache: dict[str, int | float] | None = None
        self._disk_usage_scan_running = False
        # Bumped by every store mutation (write commit, eviction, archive,
        # orphan cleanup). A snapshot records the generation it was taken
        # at; a snapshot whose generation still matches is exact no matter
        # how old it is, and a rescan is only ever due for a changed store.
        self._store_generation = 0
        # stats() aggregate snapshot: exact while _store_generation is
        # unchanged; TTL bounds staleness against out-of-process writers.
        self._manifest_stats_lock = threading.Lock()
        self._manifest_stats_cache: dict[str, Any] | None = None
        self._orphan_cleanup_running = False
        self._cleanup_holds_scan_slot = False
        # Writes in flight, guarded by _base_lock: the entry directory and
        # blob digests a write will reference once its manifest row lands.
        # Blobs reach disk in phase 2 (no lock) before the row does, and a
        # dedupe hit re-references a blob the manifest may not name yet, so
        # the orphan cleanup consults these before deleting anything.
        self._inflight_entry_dirs: Counter[str] = Counter()
        self._inflight_blob_hashes: Counter[str] = Counter()
        # Disk bytes that admitted writes may still add, guarded by
        # _base_lock. A write reserves before its first blob and releases
        # once its manifest row is installed or it gives up, so two writers
        # (the owner-thread spill and the writer thread) cannot both spend
        # the same free space above the floor.
        self._reserved_bytes = 0
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
            # The disk above the floor had no room for this write; the copy
            # already saved was kept.
            "skipped_no_disk_room": 0,
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
            # The newest write that did not land: {"reason", "message",
            # "prefix_len", "at_s"}, the message in plain words.
            "last_write_skip": None,
            # Post-install evictions put off because a request arrived; the
            # next write settles the store back under its cap.
            "settle_deferred": 0,
            "last_restore_s": None,
            "last_miss_reason": None,
            "last_archive_path": None,
            "encode_yields_foreground": 0,
            # Issue #505: the owner-thread encode in units. The longest unit
            # is the longest a request can have waited behind this tier, so
            # a stall names itself here instead of in a watchdog line.
            "encode_units": 0,
            "encode_slow_units": 0,
            "encode_longest_unit_s": 0.0,
            "encode_longest_unit": None,
            "writer_foreground_pauses": 0,
            "writer_foreground_pause_s": 0.0,
            "writer_pause_expired_busy": 0,
            "incremental_encode": self._incremental_encode,
            "incremental_encodes": 0,
            "incremental_reused_blobs": 0,
            "incremental_reused_bytes": 0,
            "incremental_missing_blobs": 0,
        }
        self._ensure_store()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="mtplx-sessionbank-ssd-writer",
            daemon=True,
        )
        self._writer.start()
        # #493: reconcile the store against its manifest once at open, in the
        # background. The walk yields to foreground traffic, takes the store
        # lock only per directory batch, and never blocks the boot or the
        # first request; it also installs the first disk-usage snapshot, so
        # the first write does not walk the store itself. A fresh store has
        # nothing to reconcile and skips the pass.
        if self.enabled and self._store_has_content():
            self._start_orphan_cleanup(at_start=True)

    def _store_has_content(self) -> bool:
        for name in ("entries", "blobs", "evicted_entries"):
            root = self.base_dir / name
            try:
                with os.scandir(root) as it:
                    if next(it, None) is not None:
                        return True
            except OSError:
                continue
        return False

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
        raise_on_yield: bool = False,
    ) -> bool:
        """Encode an entry and enqueue it for the writer thread.

        raise_on_yield: when True (the idle-lane cold_enqueue job), a
        foreground arrival mid-encode raises ColdEncodeInterrupted so the
        caller can re-dispatch for the next quiet window. Default False keeps
        every legacy caller's contract: the interrupt is swallowed and the
        write is simply skipped for this attempt.
        """
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
        should_abort = self.encode_should_abort()
        memo_key: str | None = None
        reuse: dict[tuple, dict[str, Any]] | None = None
        if self._incremental_encode:
            session_id = getattr(entry, "session_id", None)
            memo_key = str(session_id) if session_id else None
            # An empty memo still fingerprints, so the next turn can reuse.
            reuse = dict(self._block_memo(memo_key))
        try:
            encoded = encode_payload(
                cache_snapshot=getattr(entry, "cache_snapshot"),
                logits=getattr(entry, "logits"),
                hidden=getattr(entry, "hidden"),
                mtp_history_snapshot=getattr(entry, "mtp_history_snapshot", None),
                gdn_boundaries=boundaries,
                has_recurrent=bool(getattr(entry, "has_recurrent", False)),
                block_size=self.block_size,
                should_abort=should_abort,
                on_unit=self._observe_encode_unit,
                reuse=reuse,
            )
        except ColdEncodeInterrupted:
            self._release_pending(estimated_nbytes)
            self._inc("encode_yields_foreground")
            if raise_on_yield:
                raise
            return False
        except Exception as exc:
            self._release_pending(estimated_nbytes)
            self._inc("skipped_serialize_error")
            logger.warning("SessionBank SSD serialize skipped: %s: %s", type(exc).__name__, exc)
            return False
        if should_abort is not None:
            # Fence the encode's GPU work inside this idle item. The byte
            # capture above schedules evals whose command buffers otherwise
            # drain into whatever runs next — measured as 0.66-0.75 s of
            # unattributed prompt-state wall plus a decode dip on the
            # following request when an arrival landed on the tail
            # (gate254-y1 vs gate254-y2, 2026-08-07). Synchronizing here
            # keeps the tail in the idle window where the scheduler already
            # accounts for it, and the per-tensor abort check above bounds
            # how much work can pile up before an arrival is noticed.
            try:
                import mlx.core as _mx

                _mx.synchronize()
            except Exception:
                pass
        metadata = self._metadata_for_entry(
            entry,
            capabilities=capabilities or (),
            payload_nbytes=encoded.nbytes + encoded.reused_nbytes,
        )
        if encoded.reused:
            self._inc("incremental_encodes")
            self._inc("incremental_reused_blobs", len(encoded.reused))
            self._inc("incremental_reused_bytes", encoded.reused_nbytes)
        pending = PendingWrite(
            entry_id=str(metadata["entry_id"]),
            token_ids=token_ids,
            metadata=metadata,
            payload_spec=encoded.spec,
            tensors=encoded.tensors,
            pinned_nbytes=max(estimated_nbytes, int(encoded.nbytes)),
            reused_blobs=encoded.reused,
            fingerprints=encoded.fingerprints,
            memo_key=memo_key,
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

    @property
    def spill_threshold_bytes(self) -> int:
        """Entries at or above this cannot ride the staged writer queue
        (put_entry holds the fully encoded payload in RAM behind a backlog
        budget) and must use :meth:`spill_entry`'s streaming path."""
        return int(self._backlog_budget_bytes)

    def _admit_hourly(self, estimated_nbytes: int) -> bool:
        """The hourly-write-budget half of _admit_write, for writes that
        never stage bytes in the queue (spill_entry)."""
        now = time.time()
        with self._stats_lock:
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
        return True

    def _warn_spill_size_capped(
        self, entry: Any, estimated: int, effective_cap: int
    ) -> None:
        """One console line per session when a spill is refused for size.

        skipped_size_cap was a silent in-memory counter — a session bigger
        than the effective cap lost durability with no trace, and the
        user's next restart paid a full re-prefill with nothing to explain
        it (the 2026-08-27 48G-sim finding: a 4 TB disk at 28 GiB free
        capped the lane at ~7 GiB and every 100k+ session skipped mutely,
        the exact #278 silence class in a new lane). Same dedup shape as
        warn_oversized_snapshot_skip: first refusal per session speaks,
        repeats stay counters.
        """
        session_id = getattr(entry, "session_id", None)
        with self._stats_lock:
            warned = getattr(self, "_size_cap_warned_sessions", None)
            if warned is None:
                warned = set()
                self._size_cap_warned_sessions = warned
            if session_id in warned:
                return
            warned.add(session_id)
        try:
            free_gib = shutil.disk_usage(self.base_dir).free / GIB
        except Exception:
            free_gib = -1.0
        logger.warning(
            "SessionBank SSD spill skipped for session %s: entry ~%.1f GiB "
            "exceeds the effective cap %.1f GiB (the configured cap, or what "
            "free disk above the %d GiB floor allows; free disk %.1f GiB). "
            "The session stays warm in RAM but will re-prefill after a "
            "restart. Free disk space or raise the cap to restore durability "
            "for sessions this large.",
            session_id or "<anon>",
            estimated / GIB,
            effective_cap / GIB,
            LOW_DISK_FLOOR_BYTES // GIB,
            free_gib,
        )

    def spill_entry(
        self,
        entry: Any,
        *,
        capabilities: list[str] | tuple[str, ...] | None = None,
        raise_on_yield: bool = False,
    ) -> bool:
        """Stream one entry to disk tensor-by-tensor, no RAM staging.

        put_entry encodes the WHOLE payload into queue-staged bytes behind
        a 4 GiB backlog budget — so a >4 GiB session could never persist,
        and the RAM per-session cap (8 GiB on the 48 GB tier) excluded the
        very coding-agent sessions whose re-prefill costs minutes (#305's
        514 s TTFT loop; #323's live-ref sessions never reached SSD at
        all). Here each tensor is encoded (one bounded eval — lazy COW
        snapshot views materialize per tensor/block), hashed, written or
        deduped, and dropped before the next starts: the high-water mark
        is one tensor, not the payload. Runs on the caller's thread (the
        idle lane) with the same per-tensor foreground yield as put_entry;
        the writer thread is not involved.

        The on-disk result is byte-compatible with put_entry — same
        content-addressed blobs, same payload.json shape, same manifest
        row — so lookups and restores cannot tell the paths apart.

        Replacement is a transaction, in this order: reserve the disk the
        new copy may need, write its blobs, install its manifest row, and
        only then retire older entries for the cap. The copy a restore
        would use today stays whole until its successor is installed, so a
        failed or interrupted write, a full disk or a killed process leaves
        one restorable copy (2026-09-29: evicting first left none between
        20:55 and 20:56, and a 138K request had nothing to restore).
        """
        if self.mode == "off":
            return False
        token_ids = tuple(int(token) for token in getattr(entry, "token_ids"))
        if len(token_ids) < self.min_prefix_tokens:
            self._inc("skipped_too_short")
            return False
        estimated = int(getattr(entry, "nbytes", 0) or 0)
        if estimated <= 0:
            estimated = int(getattr(entry, "oversized_nbytes", 0) or 0)
        if not self._admit_hourly(estimated):
            return False
        metadata = self._metadata_for_entry(
            entry, capabilities=capabilities or (), payload_nbytes=estimated
        )
        entry_id = str(metadata["entry_id"])
        entry_hash_prefix = entry_id[:2]
        final_dir = self.base_dir / "entries" / entry_hash_prefix / entry_id
        entry_dir_rel = f"entries/{entry_hash_prefix}/{entry_id}"
        self._ensure_disk_usage_snapshot()
        # This runs on the model-owner thread: it must never walk the store.
        # Orphans that price it over the cap are reclaimed in the background
        # and are not charged to this spill meanwhile.
        self._reclaim_orphans_if_over_cap(estimated, inline=False)

        # The owner thread yields back to the scheduler rather than waiting
        # behind foreground work. In particular, never pause with _base_lock
        # held: a foreground restore may need that same lock.
        def yield_to_foreground() -> None:
            if self._stop.is_set() or (
                self._encode_yield_enabled
                and self.foreground_busy is not None
                and self.foreground_busy()
            ):
                raise ColdEncodeInterrupted()

        try:
            # Before anything is reserved or claimed: a spill that would
            # yield at its first tensor should not hold disk it never uses.
            yield_to_foreground()
        except ColdEncodeInterrupted:
            self._inc("encode_yields_foreground")
            if raise_on_yield:
                raise
            return False
        with self._base_lock:
            self._ensure_store()
            if final_dir.exists():
                if self._entry_in_manifest(entry_id):
                    self._touch_entry(entry_id)
                    return True
                self._archive_orphan_entry_dir(final_dir, entry_id)
            # A streamed payload's physical size is known only once it is
            # encoded, so the whole logical size is reserved: an upper
            # bound, because blobs the store already holds (the entry this
            # one replaces, kept until it is installed) are deduplicated.
            budget = self._admit_write_room(
                estimated,
                entry_nbytes=estimated,
                prefix_len=len(token_ids),
                warn_size_cap=lambda cap: self._warn_spill_size_capped(
                    entry, estimated, cap
                ),
            )
            if budget is None:
                return False
            self._claim_inflight(entry_dirs=(entry_dir_rel,))
        claimed_digests: set[str] = set()
        try:
            stored = self._spill_entry_admitted(
                entry,
                metadata=metadata,
                entry_id=entry_id,
                entry_hash_prefix=entry_hash_prefix,
                final_dir=final_dir,
                token_ids=token_ids,
                raise_on_yield=raise_on_yield,
                claimed_digests=claimed_digests,
            )
        finally:
            self._release_inflight(entry_dirs=(entry_dir_rel,), digests=claimed_digests)
            self._release_reserved(estimated)
        if stored:
            # Only now, with the new entry installed, may anything older
            # (the entry it replaces included) be retired for the cap.
            self._settle_to_cap(keep=entry_id, on_yield=yield_to_foreground)
        return stored

    def _claim_inflight(
        self, *, entry_dirs: tuple[str, ...] = (), digests: set[str] | None = None
    ) -> None:
        """Mark an entry directory and blob digests as owned by a write in
        flight; the orphan cleanup will not delete them until released.
        Counted, not flagged: two writes may reference one blob."""
        with self._base_lock:
            for rel in entry_dirs:
                self._inflight_entry_dirs[rel] += 1
            for digest in digests or ():
                self._inflight_blob_hashes[digest] += 1

    def _release_inflight(
        self, *, entry_dirs: tuple[str, ...] = (), digests: set[str] | None = None
    ) -> None:
        with self._base_lock:
            for rel in entry_dirs:
                if self._inflight_entry_dirs[rel] <= 1:
                    del self._inflight_entry_dirs[rel]
                else:
                    self._inflight_entry_dirs[rel] -= 1
            for digest in digests or ():
                if self._inflight_blob_hashes[digest] <= 1:
                    del self._inflight_blob_hashes[digest]
                else:
                    self._inflight_blob_hashes[digest] -= 1

    def _spill_entry_admitted(
        self,
        entry: Any,
        *,
        metadata: dict[str, Any],
        entry_id: str,
        entry_hash_prefix: str,
        final_dir: Path,
        token_ids: tuple[int, ...],
        raise_on_yield: bool,
        claimed_digests: set[str],
    ) -> bool:
        should_abort = self.encode_should_abort()
        tensor_blobs: dict[str, dict[str, Any]] = {}
        written_state = {"logical": 0, "physical": 0, "deduped_hits": 0}
        tier = self

        class _WriteThroughTensors(dict):
            """TreeCodec sink: each tensor's bytes go straight to a blob
            file (or dedupe against one) and are dropped, never retained —
            this dict stays empty on purpose."""

            def __setitem__(self, name: str, raw: bytes) -> None:  # noqa: N804
                # Abort, never wait (2026-08-29). This sink runs on the
                # single model-owner thread (spill_entry runs on the idle
                # lane), and foreground_busy counts QUEUED work — a request
                # queued behind this very spill cannot start until the
                # spill yields the thread, so a _pause_for_foreground here
                # could never see busy clear: it ran out its full deadline
                # (600 s of frozen TTFT) before writing anyway. Busy on
                # the encode path therefore means raise, same contract as
                # the codec's per-tensor check: nothing restorable is left
                # behind (payload.json + manifest row land only at the
                # end) and session_bank re-dispatches the coalesce-keyed
                # job for the next idle window. The WRITER-thread pause in
                # _writer_loop keeps waiting on purpose — nothing queues
                # behind that thread (ac0386d0 decode-theft fix).
                if should_abort is not None:
                    try:
                        busy = bool(should_abort())
                    except Exception:
                        busy = False
                    if busy:
                        raise ColdEncodeInterrupted()
                if tier._stop.is_set():
                    raise ColdEncodeInterrupted()
                digest = hashlib.sha256(raw).hexdigest()
                tensor_blobs[name] = {"sha256": digest, "nbytes": len(raw)}
                written_state["logical"] += len(raw)
                # Claim the digest before the blob exists on disk so the
                # orphan cleanup can never take it between write and row.
                if digest not in claimed_digests:
                    claimed_digests.add(digest)
                    tier._claim_inflight(digests={digest})
                if tier._write_blob(digest, raw):
                    written_state["physical"] += len(raw)
                else:
                    written_state["deduped_hits"] += 1

        codec = TreeCodec(
            block_size=self.block_size,
            should_abort=should_abort,
            on_unit=self._observe_encode_unit,
        )
        codec.tensors = _WriteThroughTensors()
        boundaries = tuple(
            (int(r[0]), r[1], r[2] if len(r) > 2 else None)
            for r in (getattr(entry, "gdn_boundaries", None) or [])
        )
        try:
            payload_spec = build_payload_spec(
                codec,
                cache_snapshot=getattr(entry, "cache_snapshot"),
                logits=getattr(entry, "logits"),
                hidden=getattr(entry, "hidden"),
                mtp_history_snapshot=getattr(entry, "mtp_history_snapshot", None),
                gdn_boundaries=boundaries,
                has_recurrent=bool(getattr(entry, "has_recurrent", False)),
            )
        except ColdEncodeInterrupted:
            self._inc("encode_yields_foreground")
            if raise_on_yield:
                raise
            return False
        except OSError as exc:
            # A blob write the disk refused (ENOSPC, EIO). Nothing of this
            # entry is restorable yet; its blobs are orphans for the
            # cleanup, and the copy it was replacing is untouched.
            self._record_write_failure(entry_id, len(token_ids), exc)
            return False
        except Exception as exc:
            self._inc("skipped_serialize_error")
            logger.warning(
                "SessionBank SSD spill skipped: %s: %s", type(exc).__name__, exc
            )
            return False
        if should_abort is not None:
            # Same fencing rationale as put_entry: the per-tensor evals
            # above schedule GPU work whose command buffers must drain in
            # this idle window, not into the next request's decode.
            try:
                import mlx.core as _mx

                _mx.synchronize()
            except Exception:
                pass
        metadata["logical_nbytes"] = int(written_state["logical"])
        metadata["physical_nbytes"] = int(written_state["physical"])
        metadata["deduped_nbytes"] = max(
            0, int(written_state["logical"]) - int(written_state["physical"])
        )
        metadata["nbytes"] = max(
            int(metadata.get("nbytes", 0) or 0), int(written_state["logical"])
        )
        payload = {
            "format_version": COLD_TIER_FORMAT_VERSION,
            "metadata": metadata,
            "payload_spec": payload_spec,
            "tensor_names": sorted(tensor_blobs),
            "tensor_blobs": tensor_blobs,
        }
        outcome = self._install_entry(
            entry_id=entry_id,
            entry_hash_prefix=entry_hash_prefix,
            final_dir=final_dir,
            payload_text=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            metadata=metadata,
            prefix_len=len(token_ids),
        )
        if outcome == _INSTALL_FAILED:
            return False
        if outcome == _INSTALL_PRESENT:
            return True
        self._inc("writes_completed")
        self._inc("spill_writes_completed")
        with self._stats_lock:
            self._stats["last_write_s"] = time.time()
            self._written_window.append(
                (
                    time.time(),
                    int(written_state["physical"] or written_state["logical"]),
                )
            )
        logger.info(
            "SessionBank SSD spilled entry_id=%s prefix_len=%d nbytes=%d",
            entry_id,
            len(token_ids),
            int(written_state["logical"]),
        )
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

    def exact_prefix_len(
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
    ) -> int:
        """Length of the longest stored entry that is a whole prefix of
        ``token_ids``, read from the manifest alone (no tensor is read).

        The rows are the ones ``lookup`` would try, under the row gates
        ``_restore_row`` applies before it reads a tensor. A probe, not a
        lookup: no counter and no miss reason change.
        """
        if self.mode in {"off", "write-only"}:
            return 0
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return 0
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
                if int(row["format_version"]) != COLD_TIER_FORMAT_VERSION:
                    continue
                epoch = row["mtp_snapshot_epoch"]
                if epoch is not None and int(epoch) != int(row["snapshot_epoch"]):
                    continue
                prefix = tuple(
                    int(token) for token in json.loads(str(row["token_ids_json"]))
                )
                if prefix and tokens[: len(prefix)] == prefix:
                    return len(prefix)
        except Exception as exc:
            logger.warning(
                "SessionBank SSD exact-prefix probe failed: %s: %s",
                type(exc).__name__,
                exc,
            )
        return 0

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
        resident_duplicates: dict[str, dict[str, Any]] | None = None,
        min_useful_matched_tokens: int = 0,
        min_restore_point: int = 0,
    ) -> ColdPrefixRestoreRecord | None:
        # min_restore_point: the lowest point the caller can restore at. A
        # hybrid row restores at its newest recurrent boundary at or below
        # the match, which the row's payload.json gives before any tensor is
        # read; a row whose boundary lies below this point is refused
        # undecoded and the next one is tried (see _restore_row).
        #
        # resident_duplicates: token_hash -> {prefix_len, has_mtp_history}
        # for RAM entries the caller has ALREADY proven identity-compatible
        # with this request, snapshot-capable (never live-ref-only), and
        # recurrent-boundary-covered. The metadata scan below runs exactly
        # as before, but when the best cold row IS one of those resident
        # entries (same token hash and stored length, and the resident copy
        # matches the row's committed-MTP coverage), the lookup returns
        # no-candidate BEFORE _restore_row: fully hydrating a candidate the
        # caller's stable sort would resolve to its RAM twin anyway is pure
        # request-path waste (measured 0.66-1.17s per warm turn, probe pair
        # 2026-08-06). A cold row with NO serve-equivalent resident twin —
        # different tokens, longer prefix, missing coverage in RAM — always
        # hydrates as before, so cold-only recovery and strictly-better-cold
        # behavior are unchanged, and an ineligible RAM match can never
        # shadow a valid cold candidate.
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
            ranked: list[tuple[tuple[int, int, int], sqlite3.Row, int, str, int]] = []
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
                ranked.append(
                    (
                        (candidate_matched, int(matched), len(prefix)),
                        row,
                        candidate_matched,
                        restore_kind,
                        len(prefix),
                    )
                )
            if not ranked:
                self._inc("restore_misses")
                self._set_last_miss("ssd_prefix_miss")
                return None
            ranked.sort(key=lambda item: item[0], reverse=True)
            if int(min_useful_matched_tokens) > 0 and ranked[0][2] < int(
                min_useful_matched_tokens
            ):
                # The caller's best RAM candidate already matches more
                # tokens than this row possibly can — the stable sort would
                # discard the hydrated result unread. Skip the multi-GB
                # request-path hydration entirely. Equal-matched rows fall
                # through to the resident-duplicate shadow (and may still
                # legitimately hydrate when no twin covers them), so
                # strictly-better-cold and cold-only recovery semantics are
                # untouched.
                self._inc("prefix_lookups_not_better_than_ram")
                self._set_last_miss("ssd_prefix_not_better_than_ram")
                return None
            # Best first; a row that cannot serve (a hybrid entry with no
            # recurrent boundary at or below its match, a format or epoch
            # mismatch) is refused by _restore_row before any tensor is read,
            # and the next one is tried: a new conversation shares its system
            # prompt with every older one, and the longest of them is often a
            # conversation without a boundary there.
            for _key, row, candidate_matched, restore_kind, prefix_len in ranked[
                :_PREFIX_BOUNDARY_ROWS_TRIED
            ]:
                if candidate_matched < int(min_useful_matched_tokens):
                    # Ranked best first: this row and every later one are
                    # under the caller's RAM match.
                    self._inc("prefix_lookups_not_better_than_ram")
                    self._set_last_miss("ssd_prefix_not_better_than_ram")
                    return None
                if resident_duplicates and self._shadowed_by_resident_duplicate(
                    row, prefix_len, resident_duplicates
                ):
                    self._inc("prefix_lookups_shadowed_by_ram")
                    self._set_last_miss("ssd_prefix_shadowed_by_resident_duplicate")
                    return None
                record = self._restore_row(
                    row,
                    tokens,
                    started_s=started,
                    require_exact_prefix=False,
                    include_gdn_boundaries=True,
                    prefix_restore_tokens=int(candidate_matched),
                    min_restore_point=int(min_restore_point),
                )
                if record is not None:
                    return ColdPrefixRestoreRecord(
                        record=record,
                        matched_tokens=int(candidate_matched),
                        restore_kind=str(restore_kind),
                    )
            return None
        except Exception as exc:
            self._inc("restore_failures")
            self._set_last_miss(f"ssd_restore_error:{type(exc).__name__}")
            logger.warning("SessionBank SSD prefix-boundary restore failed: %s: %s", type(exc).__name__, exc)
            return None

    @staticmethod
    def _shadowed_by_resident_duplicate(
        row: sqlite3.Row,
        prefix_len: int,
        resident_duplicates: dict[str, dict[str, Any]],
    ) -> bool:
        """Whether a RAM entry the caller can serve holds this row's tokens
        and at least its committed-MTP coverage (``lookup_prefix_boundary``)."""

        dup = resident_duplicates.get(str(row["token_hash"]))
        if dup is None or int(dup.get("prefix_len") or -1) != int(prefix_len):
            return False
        try:
            row_caps = {str(c) for c in json.loads(str(row["capabilities_json"] or "[]"))}
        except Exception:
            # Unknown capabilities: assume the row is maximal so only a
            # fully-covered resident twin may shadow it.
            row_caps = {"mtp_full"}
        row_has_mtp = "mtp_full" in row_caps or row["mtp_snapshot_epoch"] is not None
        return (not row_has_mtp) or bool(dup.get("has_mtp_history"))

    def _manifest_stats_row(self) -> tuple[int, int, int, int, int]:
        """The stats() aggregate, opening the manifest only when it changed.

        Before 2.8.2 every stats() call — i.e. every /health and dashboard
        poll — opened a fresh sqlite connection and ran this full-table
        aggregate, which showed up as continuous manifest.sqlite access on
        idle daemons (issue #280). The row is exact while _store_generation
        matches (every store mutation bumps it via
        _invalidate_disk_usage_cache); MANIFEST_STATS_TTL_S bounds staleness
        against out-of-process writers.
        """
        now = time.monotonic()
        with self._disk_usage_lock:
            generation = self._store_generation
        with self._manifest_stats_lock:
            cached = self._manifest_stats_cache
            if (
                cached is not None
                and int(cached.get("generation", -1)) == generation
                and now - float(cached.get("at", 0.0)) < MANIFEST_STATS_TTL_S
            ):
                return cached["row"]
        with self._connect() as conn:
            fetched = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN logical_nbytes > 0 "
                "THEN logical_nbytes ELSE nbytes END), 0), "
                "COALESCE(SUM(CASE WHEN physical_nbytes > 0 "
                "THEN physical_nbytes ELSE nbytes END), 0), "
                "COALESCE(SUM(deduped_nbytes), 0), "
                "COALESCE(MAX(max(nbytes, logical_nbytes)), 0) FROM entries"
            ).fetchone()
        row = (
            int(fetched[0]),
            int(fetched[1]),
            int(fetched[2]),
            int(fetched[3]),
            int(fetched[4]),
        )
        with self._manifest_stats_lock:
            self._manifest_stats_cache = {
                "generation": generation,
                "at": now,
                "row": row,
            }
        return row

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
            row = self._manifest_stats_row()
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
            # The disk as the next write will see it, live on every read (a
            # statvfs; the manifest figures are the cached aggregate above):
            # whether the cache can still hold two copies of the largest
            # conversation above the free-disk floor, in plain words for the
            # app. Field names added here are new; existing ones keep their
            # meaning (low_disk_writes_disabled is the last write's verdict).
            # One int read, no lock: /health must never wait behind a writer.
            reserved = int(self._reserved_bytes)
            budget = disk_budget(
                configured_max_bytes=self.max_bytes,
                free_bytes=self._free_disk_bytes(),
                store_bytes=int(row[2]) + self._untracked_bytes_estimate(),
                largest_session_bytes=int(row[4]),
                reserved_bytes=reserved,
                floor_bytes=LOW_DISK_FLOOR_BYTES,
            )
            stats["effective_max_bytes"] = int(budget.cap_bytes)
            stats["disk_state"] = budget.state
            stats["low_disk"] = bool(budget.low)
            stats["disk_message"] = budget.message()
            stats["disk_free_bytes"] = int(budget.free_bytes)
            stats["disk_floor_bytes"] = int(budget.floor_bytes)
            stats["disk_room_bytes"] = int(budget.room_bytes)
            stats["largest_session_bytes"] = int(budget.largest_session_bytes)
            stats["reserved_write_bytes"] = reserved
            # Reclaimable garbage is the part of untracked_* a cleanup can
            # free; the rest is live blobs booked under evicted entries.
            if (
                self.enabled
                and managed_disk_bytes > self.max_bytes
                and int(usage.get("orphan_disk_bytes", 0) or 0) > 0
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

    @staticmethod
    def _entry_identity(entry: Any) -> dict[str, Any]:
        return {
            "model_path": str(getattr(entry, "model_path", "")),
            "mtp_enabled": bool(getattr(entry, "mtp_enabled", False)),
            "hidden_variant": getattr(entry, "hidden_variant", None),
            "template_hash": getattr(entry, "template_hash", None),
            "mtp_history_policy": getattr(entry, "mtp_history_policy", None),
            "draft_head_identity": getattr(entry, "draft_head_identity", None),
            "policy_fingerprint": getattr(entry, "policy_fingerprint", None),
            "session_id": getattr(entry, "session_id", None),
        }

    def entry_id_for(self, entry: Any) -> str:
        """The id a bank entry is written under: its tokens, identity and
        commit epochs, so the same entry always maps to the same record."""

        token_ids = tuple(int(token) for token in getattr(entry, "token_ids"))
        digest = hashlib.sha256()
        digest.update(token_hash(token_ids).encode("utf-8"))
        digest.update(
            json.dumps(
                self._entry_identity(entry), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(str(int(getattr(entry, "snapshot_epoch", 0) or 0)).encode("ascii"))
        digest.update(str(int(getattr(entry, "mtp_snapshot_epoch", 0) or 0)).encode("ascii"))
        return digest.hexdigest()[:32]

    def is_published(self, entry: Any) -> bool:
        """Whether a restore can find this entry on disk: its manifest row
        has landed. A write still in the writer's queue, or one that failed,
        is not published; only the manifest row makes a record restorable."""

        if self.mode == "off":
            return False
        return self._entry_in_manifest(self.entry_id_for(entry))

    def _metadata_for_entry(
        self,
        entry: Any,
        *,
        capabilities: list[str] | tuple[str, ...],
        payload_nbytes: int,
    ) -> dict[str, Any]:
        token_ids = tuple(int(token) for token in getattr(entry, "token_ids"))
        identity = self._entry_identity(entry)
        entry_id = self.entry_id_for(entry)
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

    def _pause_for_foreground(self) -> None:
        """Hold the writer while latency-critical traffic is in flight.

        A 2.5 GB entry write is CPU memcpy + page-cache churn on unified
        memory — direct bandwidth competition with decode (measured -30%
        decode with 0.66-0.75 s unattributed prompt-state wall when the write
        overlapped the next turn, gate254-c4s 2026-08-07). Durability is
        deferrable by seconds; the pause is bounded so a saturated server
        still persists eventually.
        """
        if not self._writer_pause_enabled:
            return
        check = self.foreground_busy
        if check is None:
            return
        waited = 0.0
        deadline = time.monotonic() + max(0.0, self._writer_pause_max_s)
        paused = False
        expired_busy = False
        while not self._stop.is_set():
            try:
                busy = bool(check())
            except Exception:
                break
            if not busy:
                break
            if time.monotonic() >= deadline:
                # Liveness bound hit with traffic still in flight: the write
                # proceeds under the live turn. Counted so a saturated server
                # shows exactly how often durability had to fight decode.
                expired_busy = True
                break
            paused = True
            time.sleep(0.05)
            waited += 0.05
        if expired_busy:
            with self._stats_lock:
                self._stats["writer_pause_expired_busy"] = (
                    int(self._stats.get("writer_pause_expired_busy", 0) or 0) + 1
                )
        if paused:
            with self._stats_lock:
                self._stats["writer_foreground_pauses"] = (
                    int(self._stats.get("writer_foreground_pauses", 0) or 0) + 1
                )
                self._stats["writer_foreground_pause_s"] = float(
                    self._stats.get("writer_foreground_pause_s", 0.0) or 0.0
                ) + waited

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            pending = self._queue.get()
            if pending is None:
                self._queue.task_done()
                break
            try:
                self._pause_for_foreground()
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
            needed = max(0, estimated_nbytes)
            if self._pending_bytes + needed > self._backlog_budget_bytes:
                if self._pending_bytes <= 0:
                    # #384: the backlog budget bounds QUEUED bytes, but a
                    # single entry larger than the whole budget was rejected
                    # on every attempt even with an empty queue — at ~84
                    # KB/token of 27B KV the 4 GiB default made the SSD tier
                    # silently off past ~50k tokens, exactly the sessions it
                    # exists to serve, with only a skipped_backlog_bytes
                    # counter as the trace. An empty queue is not backlog
                    # pressure: admit the lone entry and say so by name.
                    self._stats["admitted_oversized_alone"] = (
                        int(self._stats.get("admitted_oversized_alone", 0) or 0)
                        + 1
                    )
                    logger.warning(
                        "SessionBank SSD write of %.2f GiB exceeds the "
                        "writer backlog budget of %.2f GiB "
                        "(MTPLX_SSD_WRITER_BACKLOG_BYTES); admitting it "
                        "alone because the queue is empty",
                        needed / (1024**3),
                        self._backlog_budget_bytes / (1024**3),
                    )
                else:
                    self._stats["skipped_backlog_bytes"] = (
                        int(self._stats.get("skipped_backlog_bytes", 0) or 0)
                        + 1
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
        # Phase 0 (no lock): make sure one reconciliation snapshot exists so
        # the admission gate below can price orphan bytes without walking the
        # store itself. Cold start only: a bank that already has a snapshot
        # pays nothing here. The walk yields to foreground traffic.
        self._ensure_disk_usage_snapshot()
        # Phase 0 (no lock): pause-aware digest planning. This is where the
        # real per-entry cost lives once blob dedupe kicks in — hashing a
        # ~2.5 GB payload is ~0.8 s of CPU/memory traffic even when every
        # blob already exists on disk and nothing gets written. Running it
        # under _base_lock blocked concurrent foreground SSD lookups for the
        # whole hash (measured 0.66-0.87 s unattributed prompt-state wall,
        # gate254-y1/y3/y4 — unchanged by write-side pauses because the
        # writes were all dedupe-skipped), and the per-blob GIL churn
        # degraded the live SSE decode stream ~30%. Per-blob pause checks
        # bound the collision to one blob's hash.
        entry_hash_prefix = pending.entry_id[:2]
        final_dir = self.base_dir / "entries" / entry_hash_prefix / pending.entry_id
        entry_dir_rel = f"entries/{entry_hash_prefix}/{pending.entry_id}"
        with self._base_lock:
            self._ensure_store()
            if final_dir.exists():
                if self._entry_in_manifest(pending.entry_id):
                    self._touch_entry(pending.entry_id)
                    return True
                self._archive_orphan_entry_dir(final_dir, pending.entry_id)
        tensor_blobs, missing_blob_bytes = self._plan_tensor_blobs(
            pending.tensors, pause_for_foreground=True
        )
        # Incremental encode: referenced blobs carry their digest already.
        # They are claimed with the rest below and must exist on disk then.
        tensor_blobs.update(
            {name: dict(blob) for name, blob in pending.reused_blobs.items()}
        )
        payload = {
            "format_version": COLD_TIER_FORMAT_VERSION,
            "metadata": pending.metadata,
            "payload_spec": pending.payload_spec,
            "tensor_names": sorted(set(pending.tensors) | set(pending.reused_blobs)),
            "tensor_blobs": tensor_blobs,
        }
        payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        pending_bytes = int(missing_blob_bytes + len(payload_text.encode("utf-8")))
        digests = {str(blob["sha256"]) for blob in tensor_blobs.values()}
        # Claimed before the reclaim below so a blob this write dedupes
        # against (an orphan until its row lands) is kept, not rewritten.
        self._claim_inflight(entry_dirs=(entry_dir_rel,), digests=digests)
        try:
            # The claim above keeps eviction and orphan cleanup away from
            # these digests, so a blob seen here stays until the row lands.
            # One gone already (its entries were evicted) means this write
            # lacks bytes it needs: skip it and let the next turn encode in
            # full.
            if any(
                not self._blob_path(str(blob["sha256"])).exists()
                for blob in pending.reused_blobs.values()
            ):
                self._inc("incremental_missing_blobs")
                self._drop_block_memo(pending.memo_key)
                return False
            # Phase 0.5 (no lock): if the snapshot prices the store over the
            # cap and part of that is reclaimable garbage, reclaim it now, on
            # this thread. A walk here delays durability, never a request,
            # and it frees room before the gate below would evict live
            # entries for it.
            self._reclaim_orphans_if_over_cap(pending_bytes, inline=True)
            wrote = self._write_pending_admitted(
                pending,
                tensor_blobs=tensor_blobs,
                payload_text=payload_text,
                pending_bytes=pending_bytes,
                final_dir=final_dir,
                entry_hash_prefix=entry_hash_prefix,
            )
            if wrote and pending.fingerprints:
                self._store_block_memo(pending, tensor_blobs)
            return wrote
        finally:
            self._release_inflight(entry_dirs=(entry_dir_rel,), digests=digests)

    def _write_pending_admitted(
        self,
        pending: PendingWrite,
        *,
        tensor_blobs: dict[str, dict[str, Any]],
        payload_text: str,
        pending_bytes: int,
        final_dir: Path,
        entry_hash_prefix: str,
    ) -> bool:
        logical_bytes = sum(int(item["nbytes"]) for item in tensor_blobs.values())
        entry_nbytes = max(int(pending.metadata.get("nbytes") or 0), int(logical_bytes))
        prefix_len = int(pending.metadata["prefix_len"])
        # Phase 1 (under lock): admission and reservation — no bulk IO, no
        # hashing, and nothing evicted. pending_bytes is exact here: the
        # blobs this payload does not share with the store (the entry it
        # replaces stays, so its blobs stay deduplicated) plus payload.json.
        with self._base_lock:
            budget = self._admit_write_room(
                pending_bytes, entry_nbytes=entry_nbytes, prefix_len=prefix_len
            )
            if budget is None:
                return False
        try:
            # Phase 2 (no lock): pause-aware bulk blob writes. Blobs are
            # content-addressed, atomic (tmp+rename), idempotent, and
            # invisible to restores until the manifest row lands in phase 3
            # — a crash or a skip here leaves only orphan blobs, which the
            # existing orphan cleanup already handles. Pausing per blob
            # bounds the bandwidth-contention window to one blob write
            # (gate254-c4s: an entry-granular pause left the 2.5 GB write
            # straddling the arrival).
            for name, raw in pending.tensors.items():
                self._pause_for_foreground()
                if self._stop.is_set():
                    return False
                blob = tensor_blobs[name]
                try:
                    wrote = self._write_blob(blob["sha256"], raw)
                except OSError as exc:
                    self._record_write_failure(pending.entry_id, prefix_len, exc)
                    return False
                if not wrote:
                    self._inc("deduped_blob_hits")
            metadata = dict(pending.metadata)
            metadata["logical_nbytes"] = int(logical_bytes)
            metadata["physical_nbytes"] = int(pending_bytes)
            metadata["deduped_nbytes"] = max(0, int(logical_bytes) - int(pending_bytes))
            # Phase 3 (under lock): payload directory, then the manifest
            # row, the commit point.
            outcome = self._install_entry(
                entry_id=pending.entry_id,
                entry_hash_prefix=entry_hash_prefix,
                final_dir=final_dir,
                payload_text=payload_text,
                metadata=metadata,
                prefix_len=prefix_len,
            )
        finally:
            self._release_reserved(pending_bytes)
        if outcome == _INSTALL_FAILED:
            return False
        # Phase 4 (no lock): only with this entry installed may older ones,
        # the entry it replaces included, be retired for the cap.
        self._settle_to_cap(keep=pending.entry_id)
        return True

    def _block_memo(self, memo_key: str | None) -> dict[tuple, dict[str, Any]]:
        if memo_key is None:
            return {}
        with self._block_memos_lock:
            memo = self._block_memos.get(memo_key)
            if memo is None:
                return {}
            self._block_memos.move_to_end(memo_key)
            return memo

    def _drop_block_memo(self, memo_key: str | None) -> None:
        if memo_key is None:
            return
        with self._block_memos_lock:
            self._block_memos.pop(memo_key, None)

    def _store_block_memo(
        self,
        pending: PendingWrite,
        tensor_blobs: dict[str, dict[str, Any]],
    ) -> None:
        """Newest completed write per session becomes the reuse source: its
        content keys map to the digests the manifest row now references."""
        if pending.memo_key is None:
            return
        memo = {
            key: {
                "sha256": str(tensor_blobs[name]["sha256"]),
                "nbytes": int(tensor_blobs[name]["nbytes"]),
            }
            for name, key in pending.fingerprints.items()
            if name in tensor_blobs
        }
        with self._block_memos_lock:
            self._block_memos[pending.memo_key] = memo
            self._block_memos.move_to_end(pending.memo_key)
            while len(self._block_memos) > self._block_memos_max:
                self._block_memos.popitem(last=False)

    def _plan_tensor_blobs(
        self,
        tensors: dict[str, bytes],
        *,
        pause_for_foreground: bool = False,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        blobs: dict[str, dict[str, Any]] = {}
        missing_bytes = 0
        planned_missing: set[str] = set()
        for name, raw in tensors.items():
            if pause_for_foreground:
                self._pause_for_foreground()
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
        try:
            try:
                temp_path.write_bytes(raw)
            except FileNotFoundError:
                # An orphan cleanup pruned the now-empty prefix directory
                # between the mkdir above and this write; the directory is
                # ours to remake.
                path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(raw)
            try:
                temp_path.rename(path)
            except FileExistsError:
                temp_path.unlink(missing_ok=True)
                return False
        except OSError:
            # A write the disk refused (ENOSPC) leaves a partial temp file
            # that would hold its bytes for the reconcile's hour-long temp
            # grace, on exactly the disk that just ran out. It is ours and
            # unreferenced: remove it, then report the failure.
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return True

    def _evict_until_room(
        self,
        required_bytes: int,
        *,
        cap_bytes: int | None = None,
        on_yield: Callable[[], None] | None = None,
        protect: frozenset[str] = frozenset(),
    ) -> bool:
        """Retire LRU entries, then reclaim their blobs in one paced pass.

        Called WITHOUT _base_lock. The writer can wait out a foreground
        turn; owner-thread spills instead raise ColdEncodeInterrupted.
        Re-reading every surviving payload for every victim made a 66-entry
        eviction run for 89 seconds alongside the next two app replies.
        ``protect`` names entries that must survive this pass (the entry a
        write has just installed).
        """
        required = max(0, int(required_bytes))
        cap = int(self.max_bytes if cap_bytes is None else cap_bytes)
        yield_work = on_yield or self._pause_for_foreground
        with self._base_lock:
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
        garbage: set[str] = set()
        try:
            for row in rows:
                yield_work()
                if self._stop.is_set():
                    return False
                with self._base_lock:
                    if str(row["entry_id"]) in protect:
                        continue
                    # A concurrent spill may be reusing this entry. Its
                    # manifest and blobs remain protected until it commits.
                    if str(row["entry_dir"]) in self._inflight_entry_dirs:
                        continue
                    if not self._entry_in_manifest(str(row["entry_id"])):
                        continue
                    garbage.update(self._retire_entry_row(row))
                    current -= int(row["physical_nbytes"] or row["nbytes"] or 0)
                    self._inc("entries_evicted")
                if current + required <= cap:
                    break
            self._delete_unreferenced_blobs(garbage, on_yield=yield_work)
            garbage.clear()
            with self._base_lock:
                # A spill can commit while the writer yields. Admission must
                # include that new entry, not the pre-pause byte count.
                return (
                    not self._stop.is_set()
                    and self._current_bytes_for_cap(required) + required <= cap
                )
        finally:
            if garbage:
                # An interrupted owner-thread spill is retried by its
                # scheduler. Retired blobs are also recoverable by the
                # existing foreground-aware orphan worker after a stop.
                self._start_orphan_cleanup()

    def _free_disk_bytes(self) -> int | None:
        try:
            return int(shutil.disk_usage(self.base_dir).free)
        except Exception:
            return None

    def _largest_entry_bytes(self) -> int:
        """Logical size of the largest entry the manifest holds."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(max(nbytes, logical_nbytes)), 0) FROM entries"
            ).fetchone()
        return int(row[0] or 0)

    def _disk_budget(self, *, incoming_bytes: int = 0) -> DiskBudget:
        """What the disk allows now (the policy is in .disk_budget).

        Re-read on every write (one statvfs and two manifest queries), so a
        disk that fills or empties between turns is seen at once; it guards
        strangers' Macs where a flat configured cap could fill the disk
        (kvcache-v2 P2.3). ``incoming_bytes`` is the logical size of the
        entry being written, which may be the largest conversation yet.
        Writers call it under _base_lock, so the reservation read is exact.
        """
        return disk_budget(
            configured_max_bytes=self.max_bytes,
            free_bytes=self._free_disk_bytes(),
            store_bytes=self._current_bytes_for_cap(),
            largest_session_bytes=max(
                int(incoming_bytes), self._largest_entry_bytes()
            ),
            reserved_bytes=self._reserved_bytes,
            floor_bytes=LOW_DISK_FLOOR_BYTES,
        )

    def _admit_write_room(
        self,
        need_bytes: int,
        *,
        entry_nbytes: int,
        prefix_len: int,
        warn_size_cap: Callable[[int], None] | None = None,
    ) -> DiskBudget | None:
        """Admit one write and reserve the disk it may add; None skips it.

        Called under _base_lock before the write's first blob. Nothing is
        evicted here: the entries on disk, the one this write replaces
        included, stay whole until the new entry is installed. A write that
        does not fit is skipped with its reason recorded, keeping the copy
        that exists instead of destroying it for one that may never land.
        """
        budget = self._disk_budget(incoming_bytes=entry_nbytes)
        if budget.state == DISK_FULL:
            self._inc("skipped_low_disk")
            with self._stats_lock:
                self._stats["low_disk_writes_disabled"] = True
            self._record_write_skip("low_disk", budget.message() or "", prefix_len)
            logger.warning(
                "SessionBank SSD writes disabled: free disk %.1f GiB is at or "
                "below the %d GiB floor",
                budget.free_bytes / GIB,
                LOW_DISK_FLOOR_BYTES // GIB,
            )
            return None
        with self._stats_lock:
            self._stats["low_disk_writes_disabled"] = False
            self._stats["effective_max_bytes"] = int(budget.cap_bytes)
        need = max(0, int(need_bytes))
        entry_nbytes = max(0, int(entry_nbytes))
        if entry_nbytes > budget.cap_bytes and not budget.cap_limited_by_disk:
            # The configured cap cannot hold this entry even alone.
            self._inc("skipped_size_cap")
            self._record_write_skip(
                "size_cap",
                f"Not saved to the SSD cache: this conversation needs "
                f"{entry_nbytes / GIB:.1f} GiB and the cache limit is "
                f"{budget.cap_bytes / GIB:.1f} GiB.",
                prefix_len,
            )
            if warn_size_cap is not None:
                warn_size_cap(int(budget.cap_bytes))
            else:
                logger.warning(
                    "SessionBank SSD size cap skipped prefix_len=%d nbytes=%d max=%d",
                    int(prefix_len),
                    entry_nbytes,
                    int(budget.cap_bytes),
                )
            return None
        if need > budget.room_bytes or entry_nbytes > budget.cap_bytes:
            self._inc("skipped_no_disk_room")
            self._record_write_skip(
                "no_disk_room",
                f"Not saved to the SSD cache: the disk has "
                f"{budget.room_bytes / GIB:.1f} GiB it can use above the "
                f"{budget.floor_bytes / GIB:.0f} GiB it always leaves free, "
                f"and this save needs {max(need, entry_nbytes) / GIB:.1f} GiB. "
                "The copy saved before it was kept.",
                prefix_len,
            )
            logger.warning(
                "SessionBank SSD write skipped, no disk room: prefix_len=%d "
                "need=%d nbytes=%d room=%d cap=%d free=%d",
                int(prefix_len),
                need,
                entry_nbytes,
                int(budget.room_bytes),
                int(budget.cap_bytes),
                int(budget.free_bytes),
            )
            return None
        self._reserved_bytes += need
        return budget

    def _release_reserved(self, nbytes: int) -> None:
        with self._base_lock:
            self._reserved_bytes = max(0, self._reserved_bytes - max(0, int(nbytes)))

    def _install_entry(
        self,
        *,
        entry_id: str,
        entry_hash_prefix: str,
        final_dir: Path,
        payload_text: str,
        metadata: dict[str, Any],
        prefix_len: int,
    ) -> str:
        """Install a written entry: its payload directory, then its manifest row.

        The row is the commit point. A restore sees an entry only once its
        row exists, and a row always names a directory whose payload and
        blobs are complete. When the disk refuses the payload, the rename or
        the SQLite write, what this call created is set aside and the store
        is as it was before the entry was written.
        """
        with self._base_lock:
            if final_dir.exists():
                if self._entry_in_manifest(entry_id):
                    self._touch_entry(entry_id)
                    return _INSTALL_PRESENT
                self._archive_orphan_entry_dir(final_dir, entry_id)
            temp_parent = self.base_dir / "entries" / entry_hash_prefix
            temp_dir: Path | None = None
            renamed = False
            try:
                temp_parent.mkdir(parents=True, exist_ok=True)
                temp_dir = Path(
                    tempfile.mkdtemp(prefix=f".{entry_id}.tmp-", dir=temp_parent)
                )
                (temp_dir / "payload.json").write_text(payload_text, encoding="utf-8")
                temp_dir.rename(final_dir)
                renamed = True
                metadata["entry_dir"] = str(final_dir.relative_to(self.base_dir))
                self._insert_manifest(metadata)
            except (OSError, sqlite3.Error) as exc:
                try:
                    if renamed:
                        # The directory landed without its row: set it aside
                        # so the next write of this entry starts clean.
                        self._archive_orphan_entry_dir(final_dir, entry_id)
                    elif temp_dir is not None:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except OSError as cleanup_exc:
                    # Left for the reconcile, which removes directories no
                    # manifest row names.
                    logger.warning(
                        "SessionBank SSD could not set aside the partial entry "
                        "entry_id=%s: %s: %s",
                        entry_id,
                        type(cleanup_exc).__name__,
                        cleanup_exc,
                    )
                self._record_write_failure(entry_id, prefix_len, exc)
                return _INSTALL_FAILED
            finally:
                self._invalidate_disk_usage_cache()
        return _INSTALL_INSTALLED

    def _settle_to_cap(
        self,
        *,
        keep: str,
        on_yield: Callable[[], None] | None = None,
    ) -> None:
        """Retire LRU entries until the store is back under its cap.

        Runs only after a write's manifest row is installed and never takes
        that entry. The cap is re-read, since the write changed both the
        store and the free space. A request arriving mid-pass (owner-thread
        spills) or a failing removal leaves the store over its cap until the
        next write settles it; the new entry is installed and restorable
        either way.
        """
        with self._base_lock:
            cap = self._disk_budget().cap_bytes
        try:
            self._evict_until_room(
                0, cap_bytes=cap, on_yield=on_yield, protect=frozenset({keep})
            )
        except ColdEncodeInterrupted:
            self._inc("settle_deferred")
        except (OSError, sqlite3.Error) as exc:
            self._inc("settle_deferred")
            logger.warning(
                "SessionBank SSD could not retire older entries after writing "
                "entry_id=%s: %s: %s",
                keep,
                type(exc).__name__,
                exc,
            )

    def _record_write_skip(self, reason: str, message: str, prefix_len: int) -> None:
        with self._stats_lock:
            self._stats["last_write_skip"] = {
                "reason": str(reason),
                "message": str(message),
                "prefix_len": int(prefix_len),
                "at_s": time.time(),
            }

    def _record_write_failure(
        self, entry_id: str, prefix_len: int, exc: BaseException
    ) -> None:
        self._inc("write_failures")
        self._record_write_skip(
            "write_failed",
            f"Not saved to the SSD cache: the disk refused the write "
            f"({type(exc).__name__}: {exc}). The copy saved before it was kept.",
            prefix_len,
        )
        logger.warning(
            "SessionBank SSD write failed entry_id=%s: %s: %s",
            entry_id,
            type(exc).__name__,
            exc,
        )

    def _current_bytes_for_cap(self, required_bytes: int = 0) -> int:
        """Bytes the cap gate prices: the whole managed directory.

        The manifest SUM is exact for every tracked entry and costs one SQLite
        query. Everything else on disk (orphans, blobs that outlived the entry
        whose physical bytes they were booked under, crash leftovers) comes
        from the last reconciliation snapshot as a delta over the manifest
        bytes that snapshot saw; the delta survives evictions unchanged, so a
        snapshot of any age is a sound estimate here. The gate never walks the
        store (41.7 s per write on an 816k-file bank, the pathology the
        snapshot exists to avoid): reclaimable orphans are taken by
        :meth:`_reclaim_orphans_if_over_cap` before the gate runs, and the
        bytes a pass still running will free are not charged to this write.
        """
        manifest_bytes = self._current_bytes()
        untracked_bytes = self._untracked_bytes_estimate()
        if untracked_bytes > 0 and self._orphan_cleanup_is_running():
            untracked_bytes = max(0, untracked_bytes - self._reclaimable_bytes_estimate())
        return manifest_bytes + untracked_bytes

    def _untracked_bytes_estimate(self) -> int:
        """Managed bytes the manifest does not account for (orphans included)."""
        with self._disk_usage_lock:
            cached = self._disk_usage_cache
            if cached is None:
                return 0
            managed_file_bytes = int(cached.get("managed_file_bytes", 0) or 0)
            database_file_bytes = int(cached.get("database_file_bytes", 0) or 0)
            manifest_at_scan = int(cached.get("manifest_physical_bytes_at_scan", 0) or 0)
        return max(0, managed_file_bytes - database_file_bytes - manifest_at_scan)

    def _reclaimable_bytes_estimate(self) -> int:
        """Bytes the last walk classified as garbage a cleanup would delete."""
        with self._disk_usage_lock:
            cached = self._disk_usage_cache
            if cached is None:
                return 0
            return int(cached.get("orphan_file_bytes", 0) or 0)

    def _reclaim_orphans_if_over_cap(self, required_bytes: int, *, inline: bool) -> None:
        """Reclaim garbage before a write is priced against the cap (#493).

        ``inline`` runs the pass on the calling thread (the writer thread:
        the walk yields to foreground traffic and only delays durability);
        ``inline=False`` schedules it in the background (the owner thread
        must never walk). The cap here is the effective write budget, so a
        filling disk reclaims earlier than the configured cap alone would.
        """
        if not self.enabled:
            return
        reclaimable = self._reclaimable_bytes_estimate()
        if reclaimable <= 0 or self._orphan_cleanup_is_running():
            return
        budget = self._disk_budget()
        cap = 0 if budget.state == DISK_FULL else budget.cap_bytes
        current = self._current_bytes() + self._untracked_bytes_estimate()
        if current + max(0, int(required_bytes)) <= cap:
            return
        if not inline:
            self._start_orphan_cleanup()
            return
        result = self._run_orphan_cleanup()
        if result is not None:
            logger.info(
                "SessionBank SSD store over its cap: reclaimed %d orphan files "
                "(%.2f GB) in %.1f s before admitting a write",
                int(result["files_deleted"]),
                float(result["disk_bytes_deleted"]) / 1e9,
                float(result["elapsed_s"]),
            )

    def _delete_entry_row(self, row: sqlite3.Row) -> None:
        with self._base_lock:
            blob_hashes = self._retire_entry_row(row)
        self._delete_unreferenced_blobs(blob_hashes)

    def _retire_entry_row(self, row: sqlite3.Row) -> set[str]:
        """Remove the small entry record under _base_lock, retaining blobs.

        The manifest row goes first. A row must never name a directory that
        is gone (a restore would find the row and fail on the missing
        payload), while a directory no row names is garbage the reconcile
        already removes; a process killed between the two leaves only that.
        """
        entry_id = str(row["entry_id"])
        entry_dir = self.base_dir / str(row["entry_dir"])
        blob_hashes = self._entry_blob_hashes(entry_dir)
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        try:
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
        finally:
            self._invalidate_disk_usage_cache()
        return blob_hashes

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
        return _entry_blob_hashes_of(entry_dir)

    def _manifest_blob_hashes(self) -> set[str]:
        return _manifest_blob_hashes_of(self.base_dir)

    def _delete_unreferenced_blobs(
        self,
        candidate_hashes: set[str],
        *,
        on_yield: Callable[[], None] | None = None,
    ) -> None:
        if not candidate_hashes:
            return
        yield_work = on_yield or self._pause_for_foreground
        candidates = sorted(candidate_hashes)
        generation = -1
        still_referenced: set[str] = set()
        index = 0
        while index < len(candidates):
            yield_work()
            if self._stop.is_set():
                return
            observed = self._generation_now()
            if generation != observed:
                # Scan without the store lock, yielding between payloads.
                # Validate its generation under the lock before deleting.
                still_referenced = _manifest_blob_hashes_of(
                    self.base_dir, on_yield=yield_work
                )
                generation = observed
            with self._base_lock:
                if generation != self._generation_now():
                    continue
                for digest in candidates[index : index + 64]:
                    if digest in still_referenced or digest in self._inflight_blob_hashes:
                        continue
                    path = self._blob_path(digest)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning(
                            "SessionBank SSD blob cleanup failed digest=%s: %s: %s",
                            digest[:12], type(exc).__name__, exc,
                        )
                        continue
                    self._prune_empty_parents(path.parent, stop_at=self.base_dir / "blobs")
                index += 64
        self._invalidate_disk_usage_cache()

    def _orphan_cleanup_is_running(self) -> bool:
        with self._disk_usage_lock:
            return bool(self._orphan_cleanup_running)

    def _claim_orphan_cleanup(self) -> bool:
        """Take the single cleanup slot; False when a pass already runs.

        The slot also holds the disk-usage scan slot while it can: the pass
        installs the snapshot it produces, so no measurement walk needs to
        start beside it (one already running keeps its own flag).
        """
        with self._disk_usage_lock:
            if self._orphan_cleanup_running:
                return False
            self._orphan_cleanup_running = True
            self._cleanup_holds_scan_slot = not self._disk_usage_scan_running
            self._disk_usage_scan_running = True
            return True

    def _release_orphan_cleanup(self) -> None:
        with self._disk_usage_lock:
            self._orphan_cleanup_running = False
            if self._cleanup_holds_scan_slot:
                self._disk_usage_scan_running = False
            self._cleanup_holds_scan_slot = False

    def _start_orphan_cleanup(self, *, at_start: bool = False) -> None:
        if not self._claim_orphan_cleanup():
            return
        threading.Thread(
            target=self._orphan_cleanup_worker,
            kwargs={"at_start": at_start},
            name="mtplx-sessionbank-orphan-cleanup",
            daemon=True,
        ).start()

    def _run_orphan_cleanup(self) -> dict[str, int | float] | None:
        """One reconciliation pass on the calling thread, or None if one runs."""
        if not self._claim_orphan_cleanup():
            return None
        try:
            result = self._cleanup_untracked_cache_once()
            self._record_orphan_cleanup_result(result)
            return result
        finally:
            self._release_orphan_cleanup()

    def _orphan_cleanup_worker(self, *, at_start: bool = False) -> None:
        try:
            result = self._cleanup_untracked_cache_once()
            self._record_orphan_cleanup_result(result)
            if at_start:
                self._log_startup_reconcile(result)
        except Exception as exc:  # pragma: no cover - defensive background task
            logger.warning(
                "SessionBank SSD orphan cleanup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
        finally:
            self._release_orphan_cleanup()

    def _log_startup_reconcile(self, result: dict[str, int | float]) -> None:
        with self._disk_usage_lock:
            cached = dict(self._disk_usage_cache or {})
        summary: dict[str, Any] = {
            "dir": str(self.base_dir),
            "files_deleted": int(result["files_deleted"]),
            "dirs_deleted": int(result["dirs_deleted"]),
            "disk_bytes_deleted": int(result["disk_bytes_deleted"]),
            "elapsed_s": round(float(result["elapsed_s"]), 3),
            "entries": int(cached.get("manifest_entries_at_scan", 0) or 0),
            "disk_bytes": int(cached.get("managed_disk_bytes", 0) or 0),
            "max_bytes": int(self.max_bytes),
            "interrupted": bool(cached.get("disk_usage_stale", False)) and self._stop.is_set(),
        }
        if summary["files_deleted"] or summary["dirs_deleted"]:
            logger.info(
                "SessionBank SSD store reconciled at start: reclaimed %d orphan "
                "files (%.2f GB) in %.1f s; %d entries, %.2f GB on disk, cap %.2f GB",
                summary["files_deleted"],
                summary["disk_bytes_deleted"] / 1e9,
                summary["elapsed_s"],
                summary["entries"],
                summary["disk_bytes"] / 1e9,
                summary["max_bytes"] / 1e9,
            )
        else:
            logger.info(
                "SessionBank SSD store reconciled at start: nothing to reclaim "
                "(%d entries, %.2f GB on disk, %.1f s)",
                summary["entries"],
                summary["disk_bytes"] / 1e9,
                summary["elapsed_s"],
            )
        with self._stats_lock:
            self._stats["startup_reconcile"] = dict(summary)
        listener = self.reconcile_listener
        if listener is not None:
            try:
                listener(summary)
            except Exception as exc:  # pragma: no cover - listener is observability
                logger.warning(
                    "SessionBank SSD reconcile listener failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

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
        """One delete-mode reconciliation pass; installs the census it took.

        The walk is :func:`reconcile_store` with this tier's coordination:
        deletions per directory batch under ``_base_lock``, writes in flight
        protected, the manifest re-read whenever the store generation moved,
        foreground yields at the blob-write cadence, early stop on close().
        The pass's own deletions do not bump the store generation (nothing
        the manifest names changed), so its post-pass totals install as the
        exact snapshot unless a concurrent commit tore the walk.
        """
        usage = self._walk_and_install(
            lambda: self._usage_from_report(self._reconcile_walk(dry_run=False))
        )
        return {
            "files_deleted": int(usage.get("files_deleted", 0) or 0),
            "dirs_deleted": int(usage.get("dirs_deleted", 0) or 0),
            "disk_bytes_deleted": int(usage.get("disk_bytes_deleted", 0) or 0),
            "elapsed_s": float(usage.get("elapsed_s", 0.0) or 0.0),
        }

    def _reconcile_walk(self, *, dry_run: bool) -> ReconcileReport:
        return reconcile_store(
            self.base_dir,
            dry_run=dry_run,
            lock=None if dry_run else self._base_lock,
            protected=self._inflight_snapshot,
            manifest_generation=self._generation_now,
            on_yield=self._pause_for_foreground,
            should_stop=self._stop.is_set,
            yield_every=DISK_USAGE_SCAN_YIELD_EVERY_FILES,
        )

    def _inflight_snapshot(self) -> tuple[set[str], set[str]]:
        # Called under _base_lock by the reconciliation pass.
        return set(self._inflight_entry_dirs), set(self._inflight_blob_hashes)

    def _generation_now(self) -> int:
        with self._disk_usage_lock:
            return int(self._store_generation)

    def _manifest_entry_dirs(self) -> set[str]:
        return _manifest_entry_dirs_of(self.base_dir)

    @staticmethod
    def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
        _prune_empty_parents_of(path, stop_at=stop_at)

    def _restore_row(
        self,
        row: sqlite3.Row,
        lookup_tokens: tuple[int, ...],
        *,
        started_s: float,
        require_exact_prefix: bool = True,
        include_gdn_boundaries: bool = False,
        prefix_restore_tokens: int | None = None,
        min_restore_point: int = 0,
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

        payload_spec = payload["payload_spec"]
        partial_restore = (
            prefix_restore_tokens is not None
            and not require_exact_prefix
            and int(prefix_restore_tokens) > 0
        )
        if partial_restore:
            requested_prefix = int(prefix_restore_tokens)
            has_recurrent = bool(payload_spec.get("has_recurrent", False))
            boundary_prefix = _payload_boundary_at_or_below(
                payload_spec, requested_prefix
            )
            if has_recurrent and boundary_prefix is None:
                from mtplx.session_bank import _boundary_true_restore_enabled

                if _boundary_true_restore_enabled():
                    # A hybrid entry restores only at a stored recurrent
                    # boundary at or below the match, tiny gaps included
                    # (SessionBank.restore_entry_prefix_cache, since
                    # 2026-09-08). With none, decoding the whole entry only
                    # fed a candidate the restore refuses: on 2026-10-01 a
                    # Pi turn sharing its tool definitions with a 131,735-
                    # token conversation decoded that entry's 4.17 GB on
                    # every attempt, and it stayed resident through the cold
                    # prefill that followed, which the engine then refused.
                    self._inc("prefix_restores_without_boundary")
                    self._set_last_miss("ssd_prefix_no_recurrent_boundary")
                    return None
            if (
                has_recurrent
                and boundary_prefix is not None
                and boundary_prefix < int(min_restore_point)
            ):
                # The restore would land below the lowest point the caller
                # can use (the near lane's floor: its exact prefix in RAM or
                # on SSD). The lane refused such a candidate only after this
                # decode, as boundary_not_better.
                self._inc("prefix_restores_below_caller_floor")
                self._set_last_miss("ssd_prefix_boundary_below_floor")
                return None
            # Cache types whose metadata couples rolling state to their
            # tensors must use the exact decoder plus trim; so does a hybrid
            # entry without a boundary when boundary-true restores are off.
            if (
                (has_recurrent and boundary_prefix is None)
                or not payload_supports_prefix_decode(payload_spec)
            ):
                decoded = decode_payload(
                    payload_spec,
                    read_tensor,
                    include_gdn_boundaries=include_gdn_boundaries,
                )
            else:
                restore_point = (
                    int(boundary_prefix)
                    if boundary_prefix is not None
                    else requested_prefix
                )
                # Non-boundary restores re-forward token N-1 for fresh
                # logits; boundary restores resume directly at N.
                cache_prefix_len = (
                    restore_point
                    if boundary_prefix is not None
                    else restore_point - 1
                )
                mtp_history_prefix_len = None
                mtp_spec = payload_spec.get("mtp_history_snapshot")
                # A committed MTP cache holds prompt_ids[1:], so its useful
                # prefix is exactly one token shorter than the target-cache
                # boundary.  This is the 0.28 GiB tail of the user's observed
                # restore: leave windowed or coupled layouts on the legacy
                # full-decode-and-trim path, but never hydrate a long
                # committed-history snapshot merely to trim it immediately.
                if (
                    mtp_spec is not None
                    and str(metadata.get("mtp_history_policy") or "").lower()
                    == "committed"
                    and snapshot_supports_prefix_decode(mtp_spec)
                ):
                    mtp_history_prefix_len = max(0, restore_point - 1)
                decoded = decode_payload_prefix(
                    payload_spec,
                    read_tensor,
                    cache_prefix_len=max(0, cache_prefix_len),
                    mtp_history_prefix_len=mtp_history_prefix_len,
                    boundary_prefix_len=boundary_prefix,
                )
        else:
            # Exact-prefix restores never rewind below the stored boundary, so
            # the interior snapshots are decoded lazily as before.
            decoded = decode_payload(
                payload_spec,
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
        boundary_loader = (
            None
            if include_gdn_boundaries or partial_restore
            else (
                lambda should_abort=None: decode_gdn_boundaries(
                    payload_spec, read_tensor, should_abort=should_abort
                )
            )
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
            cache_snapshot_prefix_len=decoded.cache_snapshot_prefix_len,
            mtp_history_snapshot_prefix_len=(
                decoded.mtp_history_snapshot_prefix_len
            ),
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
        """Return the reconciliation snapshot; schedule a rescan only when due.

        A snapshot is exact while the store generation it recorded still
        matches. A rescan is due when the store has changed AND the adaptive
        interval has elapsed: max(DISK_USAGE_CACHE_TTL_S, DUTY_DIVISOR x the
        last walk's duration). Reads never block; ``force`` walks now.
        """
        if force:
            return self._refresh_disk_usage_now()
        now = time.time()
        with self._disk_usage_lock:
            cached = self._disk_usage_cache
            if cached is None:
                self._start_disk_usage_scan_locked()
                return self._empty_disk_usage(scan_pending=True)
            view = dict(cached)
            changed = int(cached.get("disk_usage_generation", -1)) != self._store_generation
            if changed and self._rescan_interval_elapsed_locked(cached, now):
                self._start_disk_usage_scan_locked()
            view["disk_usage_scan_pending"] = bool(self._disk_usage_scan_running)
            view["disk_usage_stale"] = bool(cached.get("disk_usage_stale")) or changed
            return view

    def _rescan_interval_elapsed_locked(self, cached: dict[str, int | float], now: float) -> bool:
        interval = max(
            DISK_USAGE_CACHE_TTL_S,
            DISK_USAGE_SCAN_DUTY_DIVISOR * float(cached.get("disk_usage_scan_s", 0.0) or 0.0),
        )
        return now - float(cached.get("disk_usage_last_scan_s", 0.0) or 0.0) >= interval

    def _ensure_disk_usage_snapshot(self) -> None:
        """Cold start only: take the first snapshot synchronously (off-lock).

        Later writes reuse the snapshot; if a background scan is already
        running the caller proceeds on manifest bytes alone rather than
        waiting for it.
        """
        with self._disk_usage_lock:
            if self._disk_usage_cache is not None or self._disk_usage_scan_running:
                return
            self._disk_usage_scan_running = True
        try:
            self._refresh_disk_usage_now()
        except Exception as exc:
            logger.warning(
                "SessionBank SSD disk usage scan failed at cold start: %s: %s",
                type(exc).__name__,
                exc,
            )
        finally:
            with self._disk_usage_lock:
                self._disk_usage_scan_running = False

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
        except Exception as exc:  # pragma: no cover - defensive background task
            logger.warning(
                "SessionBank SSD disk usage scan failed: %s: %s",
                type(exc).__name__,
                exc,
            )
        finally:
            with self._disk_usage_lock:
                self._disk_usage_scan_running = False

    def _refresh_disk_usage_now(self) -> dict[str, int | float]:
        return self._walk_and_install(self._scan_managed_disk_usage)

    # Keys a walk contributes to the installed snapshot. A delete-mode pass
    # also reports what it removed; those figures are returned to the caller
    # but never installed (stats() would otherwise show them as store state).
    _SNAPSHOT_KEYS = frozenset(
        {
            "managed_file_bytes",
            "managed_disk_bytes",
            "database_file_bytes",
            "database_disk_bytes",
            "managed_file_count",
            "managed_dir_count",
            "manifest_entries_at_scan",
            "orphan_file_bytes",
            "orphan_disk_bytes",
            "orphan_file_count",
            "disk_usage_scan_s",
            "disk_usage_last_scan_s",
            "disk_usage_scan_pending",
            "disk_usage_stale",
        }
    )

    def _walk_and_install(
        self, walk: Callable[[], dict[str, int | float]]
    ) -> dict[str, int | float]:
        # The walk runs without _base_lock so a 40 s reconciliation of a large
        # bank never stalls the writer or blocks archive/cleanup. Coherence
        # comes from the store generation instead: the walk records the
        # generation it started at, and a snapshot taken across a mutation is
        # installed as an estimate (stale=True) rather than as truth — stats
        # will not trigger orphan cleanup from it, and the next due rescan
        # replaces it. The manifest total is captured before and after; the
        # larger of the two is paired with the physical bytes so an entry
        # committed mid-walk can never masquerade as orphan bytes.
        with self._disk_usage_lock:
            generation = self._store_generation
        manifest_before = self._current_bytes()
        usage = walk()
        if self._stop.is_set() or bool(usage.get("disk_usage_interrupted")):
            # Interrupted by close(): a partial walk is not a snapshot.
            usage["disk_usage_stale"] = True
            return usage
        manifest_after = self._current_bytes()
        with self._disk_usage_lock:
            torn = self._store_generation != generation
            usage["manifest_physical_bytes_at_scan"] = max(manifest_before, manifest_after)
            usage["disk_usage_generation"] = int(generation)
            usage["disk_usage_stale"] = bool(torn)
            self._disk_usage_cache = {
                key: value
                for key, value in usage.items()
                if key in self._SNAPSHOT_KEYS
                or key in ("manifest_physical_bytes_at_scan", "disk_usage_generation")
            }
        return dict(usage)

    @staticmethod
    def _empty_disk_usage(*, scan_pending: bool) -> dict[str, int | float]:
        return {
            "managed_file_bytes": 0,
            "managed_disk_bytes": 0,
            "database_file_bytes": 0,
            "database_disk_bytes": 0,
            "manifest_physical_bytes_at_scan": 0,
            "manifest_entries_at_scan": 0,
            "managed_file_count": 0,
            "managed_dir_count": 0,
            "orphan_file_bytes": 0,
            "orphan_disk_bytes": 0,
            "orphan_file_count": 0,
            "disk_usage_scan_s": 0.0,
            "disk_usage_last_scan_s": 0.0,
            "disk_usage_generation": -1,
            "disk_usage_scan_pending": bool(scan_pending),
            "disk_usage_stale": bool(scan_pending),
        }

    def _scan_managed_disk_usage(self) -> dict[str, int | float]:
        """The measurement walk: the reconciliation in dry-run mode."""
        return self._usage_from_report(self._reconcile_walk(dry_run=True))

    @staticmethod
    def _usage_from_report(report: ReconcileReport) -> dict[str, int | float]:
        usage: dict[str, int | float] = {
            "managed_file_bytes": int(report.file_bytes),
            "managed_disk_bytes": int(report.disk_bytes),
            "database_file_bytes": int(report.database_file_bytes),
            "database_disk_bytes": int(report.database_disk_bytes),
            "managed_file_count": int(report.file_count),
            "managed_dir_count": int(report.dir_count),
            "manifest_entries_at_scan": int(report.entries),
            "orphan_file_bytes": int(report.orphan_file_bytes),
            "orphan_disk_bytes": int(report.orphan_disk_bytes),
            "orphan_file_count": int(report.orphan_file_count),
            "disk_usage_scan_s": float(report.elapsed_s),
            "disk_usage_last_scan_s": float(time.time()),
            "disk_usage_scan_pending": False,
            "disk_usage_stale": bool(report.interrupted),
            "disk_usage_interrupted": bool(report.interrupted),
        }
        if not report.dry_run:
            # A delete-mode pass leaves no reclaimable garbage behind (what it
            # kept was claimed by the manifest or a write in flight).
            usage["orphan_file_bytes"] = 0
            usage["orphan_disk_bytes"] = 0
            usage["orphan_file_count"] = 0
            usage["files_deleted"] = int(report.files_deleted)
            usage["dirs_deleted"] = int(report.dirs_deleted)
            usage["disk_bytes_deleted"] = int(report.disk_bytes_deleted)
            usage["elapsed_s"] = float(report.elapsed_s)
        return usage

    def _invalidate_disk_usage_cache(self) -> None:
        # Every store mutation lands here. The snapshot is kept (its orphan
        # delta stays a sound estimate for the cap gate); it is only marked
        # as belonging to an older generation so a rescan becomes due once
        # the adaptive interval has passed.
        with self._disk_usage_lock:
            self._store_generation += 1
            if self._disk_usage_cache is not None:
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

    def encode_should_abort(self) -> Callable[[], bool] | None:
        """The foreground signal the owner-thread persistence work polls, or
        None when the yield contract is off or unwired. One accessor so the
        encode, the streaming spill and the boundary hydration that runs
        before them cannot drift apart."""
        if self._encode_yield_enabled and self.foreground_busy is not None:
            return self.foreground_busy
        return None

    def note_encode_yield(self) -> None:
        """Count a yield taken outside the tier (boundary hydration)."""
        self._inc("encode_yields_foreground")

    def _observe_encode_unit(self, kind: str, seconds: float, nbytes: int) -> None:
        """TreeCodec.on_unit sink. A unit is one eval plus one host copy of
        at most 32 MiB; anything over ENCODE_SLOW_UNIT_S is logged with what
        it was, because that is the wait a queued request paid."""
        slow = float(seconds) >= ENCODE_SLOW_UNIT_S
        with self._stats_lock:
            self._stats["encode_units"] = int(self._stats.get("encode_units", 0) or 0) + 1
            if slow:
                self._stats["encode_slow_units"] = (
                    int(self._stats.get("encode_slow_units", 0) or 0) + 1
                )
            if float(seconds) > float(self._stats.get("encode_longest_unit_s", 0.0) or 0.0):
                self._stats["encode_longest_unit_s"] = round(float(seconds), 4)
                self._stats["encode_longest_unit"] = f"{kind}:{int(nbytes)}B"
        if slow:
            logger.warning(
                "SessionBank SSD encode unit took %.2f s (%s, %d bytes); a unit "
                "is bounded in size, so this is the machine (memory pressure, "
                "swap, disk) and not the amount of work",
                float(seconds),
                kind,
                int(nbytes),
            )

    def _set_last_miss(self, reason: str) -> None:
        with self._stats_lock:
            self._stats["last_miss_reason"] = reason

    @property
    def last_miss_reason(self) -> str | None:
        """Cheap request-path accessor. ``stats()`` is observability and may
        schedule a reconciliation walk; a lookup miss must never do that."""
        with self._stats_lock:
            value = self._stats.get("last_miss_reason")
        return str(value) if value else None


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
