"""Serving-session scaffolding for long-context MTPLX chat.

This module keeps HTTP/OpenAI behavior out of the prefix cache. It is small on
purpose: the first production step is to make lifecycle, metrics, and admin
state explicit before the generation loop accepts warm prompt state directly.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any, Iterator, Mapping

from .session_bank import (
    CacheMissReason,
    DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
    DEFAULT_IDLE_TTL_S,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_MAX_BYTES,
    DEFAULT_PER_SESSION_MAX_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    SessionBank,
    block_aligned_prefix_len,
)
from .runtime_options import block_prefix_restore_enabled


logger = logging.getLogger(__name__)

_HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES = 96 * 1024**3
_HIGH_MEMORY_PER_SESSION_MAX_BYTES = 24 * 1024**3
_HIGH_MEMORY_MAX_ENTRIES = 48
# Model-aware auto budget (v2, founder ruling 2026-07-05): the RAM cache
# defaults to half of the RAM that remains after the model weights, so a
# 128 GB Mac gets a big warm cache while a 32 GB Mac is not handed the old
# flat 24 GiB cap that could push the whole process past physical RAM.
_AUTO_BUDGET_SURPLUS_FRACTION = 0.5
_AUTO_BUDGET_FLOOR_BYTES = 1 * 1024**3
_AUTO_BUDGET_CAP_BYTES = 48 * 1024**3


def _bank_bytes_from_env(name: str, default: int) -> int:
    """Read a SessionBank byte-cap override from the environment.

    Supports plain integers (bytes) and the suffixes K, M, G, T — bare
    ("8G"), with B ("8GB"), or IEC ("8GiB"), case-insensitive; all are
    powers of 1024. Unparseable or nonpositive values fall back to the
    default WITH a warning: the silent fallback shipped before 2.5.4 made a
    typo'd "8GB" behave exactly like success while the bank ran at the
    default size (#229).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = raw.strip().upper()
    if not s:
        return default
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    body = s
    if body.endswith("IB") and len(body) > 2:
        body = body[:-2]
    elif body.endswith("B") and len(body) > 1 and body[-2] in suffixes:
        body = body[:-1]
    try:
        if body and body[-1] in suffixes:
            value = int(float(body[:-1]) * suffixes[body[-1]])
        else:
            value = int(body)
    except (OverflowError, ValueError, IndexError):
        logger.warning(
            "Invalid %s=%r (expected bytes or K/M/G/T size, e.g. 8G or 8GiB); "
            "falling back to default %d bytes",
            name,
            raw,
            default,
        )
        return default
    if value < 1:
        logger.warning(
            "Invalid %s=%r (must be positive); falling back to default %d bytes",
            name,
            raw,
            default,
        )
        return default
    return value


def _bank_entries_from_env(name: str, default: int) -> int:
    """Read a SessionBank entry-count override from the environment.

    Parses a plain integer (no K/M/G/T suffixes - entry count is a count, not
    a byte size). Validates that the value is >= 1. On parse error or invalid
    value, logs a warning and returns the default.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid %s=%r (expected integer >= 1); falling back to default %d",
            name,
            raw,
            default,
        )
        return default
    if value < 1:
        logger.warning(
            "Invalid %s=%r (must be >= 1); falling back to default %d",
            name,
            raw,
            default,
        )
        return default
    return value


def _detect_total_ram_bytes_for_session_bank() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        # Absolute path: the app-owned daemon runs with a sanitized PATH that
        # does not include /usr/sbin, so a bare "sysctl" raises
        # FileNotFoundError and RAM-aware budgets silently fell back to the
        # legacy flat defaults (caught by v2 app QA, 2026-07-05).
        output = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        total = int(str(output).strip())
    except Exception:
        return None
    return total if total > 0 else None


def _default_max_entries() -> int:
    total_ram = _detect_total_ram_bytes_for_session_bank()
    if (
        total_ram is not None
        and total_ram >= _HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES
    ):
        return _HIGH_MEMORY_MAX_ENTRIES
    return DEFAULT_MAX_ENTRIES


def _session_bank_max_entries() -> int:
    raw = os.environ.get("MTPLX_SESSION_BANK_MAX_ENTRIES")
    default = _default_max_entries()
    if raw is None:
        return default
    return _bank_entries_from_env("MTPLX_SESSION_BANK_MAX_ENTRIES", default)


def _default_per_session_max_bytes() -> int:
    total_ram = _detect_total_ram_bytes_for_session_bank()
    if (
        total_ram is not None
        and total_ram >= _HIGH_MEMORY_SESSION_BANK_THRESHOLD_BYTES
    ):
        return _HIGH_MEMORY_PER_SESSION_MAX_BYTES
    return DEFAULT_PER_SESSION_MAX_BYTES


def model_weights_bytes(model_path: Any) -> int | None:
    """Total bytes of the model's safetensors shards (weights actually wired
    into memory), following symlink wrappers. None when unknown.

    Recursive on purpose: shipped layouts nest shards below the root — the
    MTP sidecar lives at ``mtp/weights.safetensors`` (artifacts.py) and
    wrapper dirs keep shards under a subdirectory. The old top-level-only
    scan undercounted those (or returned None outright), silently skewing
    the RAM-aware session-bank budget this number feeds.

    The Flash-Next n-gram sidecar (~30 GiB) is excluded by name: in its
    default streamed mode only touched pages become resident and they are
    reclaimable file-backed pages, not wired weight. Counting it here was
    the 2026-08-28 defect chain (false MODEL DOES NOT FIT, 30G-pessimistic
    window and bank on 128G Macs). When the resident policy arms, the
    caller adds ``ngram_table_bytes`` back explicitly — see
    ``memory_plan.ngram_table_resident_policy``."""
    from mtplx.memory_plan import NGRAM_TABLE_FILENAME

    try:
        root = Path(str(model_path))
        if not root.is_dir():
            return None
        total = 0
        for shard in root.rglob("*.safetensors"):
            if shard.name == NGRAM_TABLE_FILENAME:
                continue
            try:
                total += shard.stat().st_size
            except OSError:
                continue
        return total if total > 0 else None
    except Exception:
        return None


def ngram_table_bytes(model_path: Any) -> int:
    """Size of the Flash-Next n-gram sidecar next to the weights (0 when
    absent). Kept separate from ``model_weights_bytes`` so callers count it
    as a commitment exactly when the resident policy says it will be one."""
    from mtplx.memory_plan import NGRAM_TABLE_FILENAME

    try:
        table = Path(str(model_path)) / NGRAM_TABLE_FILENAME
        return table.stat().st_size if table.is_file() else 0
    except Exception:
        return 0


def _memory_budget_bytes_env() -> int | None:
    """MTPLX_MEMORY_BUDGET: total RAM envelope the server was asked to fit.

    Set by the server from ``--memory-budget`` (normalized to plain bytes)
    but accepted with K/M/G/T suffixes for direct env users.
    """
    raw = os.environ.get("MTPLX_MEMORY_BUDGET")
    if raw is None or not raw.strip():
        return None
    value = _bank_bytes_from_env("MTPLX_MEMORY_BUDGET", 0)
    return value if value > 0 else None


# Loud once per process when the auto budget lands on its floor: a silently
# tiny warm cache reads as "the cache broke" (same lesson as #229/#230).
_auto_floor_announced = False


def _announce_auto_budget_floor(
    total_ram: int, model_bytes: int, surplus: int
) -> None:
    global _auto_floor_announced
    if _auto_floor_announced:
        return
    _auto_floor_announced = True
    print(
        "[mtplx] session-bank auto budget floored at "
        f"{_AUTO_BUDGET_FLOOR_BYTES / 1024**3:.1f}G: "
        f"total_ram={total_ram / 1024**3:.1f}G "
        f"model_weights={model_bytes / 1024**3:.1f}G "
        f"post-model surplus={surplus / 1024**3:.1f}G. Warm-cache capacity "
        "is minimal on this machine; longer contexts will re-prefill. "
        "Override with MTPLX_SESSION_BANK_MAX_BYTES (sizes like 4G).",
        flush=True,
    )


def _auto_session_bank_max_bytes(model_bytes: int | None) -> int | None:
    """Half of the RAM surplus left after the model weights, clamped.

    Founder ruling 2026-07-05 after the in-flight memory climb to 55 GB on a
    128 GB Mac: fine there, lethal on a 32 GB M1. The RAM cache budget scales
    with what the machine actually has left once the model is resident:
    ``0.5 * (total_ram - model_weights)``, floored at 1 GiB (below that the
    bank is pure churn) and capped at 48 GiB. Returns None when either input
    is unknown so callers fall back to the legacy tiered defaults.

    ``--memory-budget`` (MTPLX_MEMORY_BUDGET) substitutes for machine RAM in
    the surplus formula when it is tighter, so a declared envelope scales the
    whole cache stack down with one knob.
    """
    if model_bytes is None or model_bytes <= 0:
        return None
    total_ram = _detect_total_ram_bytes_for_session_bank()
    budget = _memory_budget_bytes_env()
    if budget is not None:
        total_ram = budget if total_ram is None else min(total_ram, budget)
    if total_ram is None:
        return None
    surplus = total_ram - int(model_bytes)
    if surplus <= 0:
        _announce_auto_budget_floor(total_ram, int(model_bytes), surplus)
        return _AUTO_BUDGET_FLOOR_BYTES
    budget = int(surplus * _AUTO_BUDGET_SURPLUS_FRACTION)
    if budget < _AUTO_BUDGET_FLOOR_BYTES:
        _announce_auto_budget_floor(total_ram, int(model_bytes), surplus)
    return max(_AUTO_BUDGET_FLOOR_BYTES, min(_AUTO_BUDGET_CAP_BYTES, budget))


def _is_auto_bytes_setting(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() in {"auto", "default"}


def _explicit_max_bytes_env_set() -> bool:
    raw = os.environ.get("MTPLX_SESSION_BANK_MAX_BYTES")
    return bool(raw and raw.strip()) and not _is_auto_bytes_setting(raw)


def resolve_session_bank_max_bytes(
    model_bytes: int | None = None,
    *,
    memory_plan: Any | None = None,
) -> tuple[int, bool]:
    """MTPLX_SESSION_BANK_MAX_BYTES resolution with model-aware auto sizing.

    Returns ``(max_bytes, auto_active)``. Explicit byte values keep today's
    semantics (auto_active False). In auto mode a machine memory plan
    (mtplx.memory_plan) wins when available: the bank takes everything the
    engine envelope leaves after weights and transients (its dynamic
    ceiling yields to live KV at runtime, so idle-aggressive is safe).
    Without a plan the legacy half-surplus formula applies; when nothing
    can be computed the flat default applies (auto_active False) so every
    legacy behavior stays byte-identical.
    """
    raw = os.environ.get("MTPLX_SESSION_BANK_MAX_BYTES")
    if raw is not None and raw.strip() and not _is_auto_bytes_setting(raw):
        return (
            _bank_bytes_from_env(
                "MTPLX_SESSION_BANK_MAX_BYTES", DEFAULT_MAX_BYTES
            ),
            False,
        )
    if memory_plan is not None and getattr(memory_plan, "available", False):
        return int(memory_plan.bank_idle_max_bytes), True
    auto = _auto_session_bank_max_bytes(model_bytes)
    if auto is not None:
        return auto, True
    return DEFAULT_MAX_BYTES, False


def resolve_session_bank_per_session_bytes(
    max_bytes: int,
    *,
    auto_active: bool = True,
) -> int:
    """Per-session cap resolution.

    Explicit env wins (clamped to the bank budget when the budget was
    auto-computed). In auto mode the default is 2/3 of the budget so one
    conversation cannot monopolize the whole cache; in legacy mode the
    RAM-tiered defaults are preserved exactly.
    """
    raw = os.environ.get("MTPLX_SESSION_BANK_PER_SESSION_BYTES")
    if raw is not None and raw.strip() and not _is_auto_bytes_setting(raw):
        parsed = _bank_bytes_from_env(
            "MTPLX_SESSION_BANK_PER_SESSION_BYTES",
            _default_per_session_max_bytes(),
        )
        return min(parsed, int(max_bytes)) if auto_active else parsed
    if auto_active:
        # 2/3 of the bank budget, additionally clamped to the RAM-tier
        # ceiling (8 GiB below 96 GiB RAM, 24 GiB above). The auto rule on
        # its own RAISED the admission gate on small boxes relative to the
        # v1.0.4 flat gate (64 GB Mac: 15 GiB vs 8 GiB), admitting snapshots
        # whose restore-time transient copies blow past physical RAM (#150,
        # ArthoPacini). Oversized snapshots still get the live-ref lease
        # fallback, so warm reuse survives the clamp.
        auto_cap = max(_AUTO_BUDGET_FLOOR_BYTES, int(max_bytes) * 2 // 3)
        return min(auto_cap, _default_per_session_max_bytes())
    return _default_per_session_max_bytes()


def _session_bank_per_session_max_bytes() -> int:
    raw = os.environ.get("MTPLX_SESSION_BANK_PER_SESSION_BYTES")
    default = _default_per_session_max_bytes()
    if raw is None:
        return default
    return _bank_bytes_from_env("MTPLX_SESSION_BANK_PER_SESSION_BYTES", default)


@dataclass
class BoundarySnapshot:
    kind: str
    token_len: int
    token_hash: str
    created_at_s: float = field(default_factory=time.time)
    bank_token_hash: str | None = None
    nbytes: int = 0
    snapshot_epoch: int = 0


@dataclass
class EngineSessionCommit:
    committed: bool
    reason: str
    prefix_len: int


class EngineSessionBusy(RuntimeError):
    """Raised when a foreground request tries to mutate an in-flight session."""


def token_hash_short(token_ids: list[int] | tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()[:16]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


IMPLICIT_SESSION_SOURCES = frozenset(
    {"longest_prefix", "pending_postcommit_near_prefix", "common_prefix_reuse"}
)

# common_prefix_reuse thresholds: adopt an existing session's identity when
# no exact prefix matches but a session shares at least this much committed
# prefix with the incoming prompt. Two unrelated conversations essentially
# never share 4K+ identical leading tokens (system prompt + tool contract +
# early history), while one mutated conversation (compaction flip, edit
# rewrite — the 2026-07-20 chess session rotated through SIX anon ids and
# quadrupled the RAM bank to 25.9 GB) always does.
_COMMON_PREFIX_REUSE_MIN_TOKENS = 4096
_COMMON_PREFIX_REUSE_MIN_FRACTION = 0.25
_COMMON_PREFIX_PROBE_TOKENS = 64


def _new_anon_session_id() -> str:
    return f"anon-{secrets.token_hex(8)}"


def common_prefix_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, Mapping):
        value = message.get("content", "")
    else:
        value = getattr(message, "content", "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def system_prompt_hash(messages: list[Any]) -> str | None:
    for message in messages:
        if _message_role(message) in {"system", "developer"}:
            return hash_text(_message_content(message))
    return None


def is_no_history_shape(messages: list[Any]) -> bool:
    roles = [_message_role(message) for message in messages if _message_role(message)]
    return roles in (["system", "user"], ["developer", "user"])


def is_background_request(
    *,
    messages: list[Any],
    max_tokens: int | None,
    headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    main_system_hash: str | None = None,
) -> bool:
    if max_tokens is None or int(max_tokens) > 48:
        return False
    headers = headers or {}
    metadata = metadata or {}
    header_task = ""
    for key, value in headers.items():
        if key.lower() == "x-openwebui-task":
            header_task = str(value)
            break
    metadata_task = str(metadata.get("task") or metadata.get("openwebui_task") or "")
    current_system_hash = system_prompt_hash(messages)
    # A system-prompt mismatch only infers a background task for the task
    # shape itself: one system prompt plus one user turn, which is what an
    # unmarked title/tags job looks like. A short-answer turn that continues
    # a conversation (assistant history present) is foreground work from a
    # second client and must keep its session — issue #454: every
    # conversation after the first ran sessionless, re-prefilling every
    # turn, because its 30-token answers matched this heuristic.
    system_mismatch = (
        main_system_hash is not None
        and current_system_hash is not None
        and current_system_hash != main_system_hash
        and is_no_history_shape(messages)
    )
    return bool(header_task or metadata_task or system_mismatch)


_DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S = 8.0
_DEFAULT_NEAR_PREFIX_MAX_TOKEN_GAP = 8
_DEFAULT_NEAR_PREFIX_MIN_MATCH_TOKENS = 64
_DEFAULT_PREFIX_BLOCK_SIZE = DEFAULT_PREFIX_BLOCK_SIZE
_DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS = DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS


_DEFAULT_POSTCOMMIT_ARRIVAL_WAIT_S = 0.6


def _postcommit_arrival_wait_s() -> float:
    """Read MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S from the environment.

    Bounded window a new same-session request grants an already-RUNNING
    canonical postcommit before aborting it (B', 2026-08-06). Defaults to
    0.6s. Values <= 0 restore the exact 2026-07-17 immediate-abort
    behavior. Bad values fall back to the default.
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S")
    if raw is None or not str(raw).strip():
        return _DEFAULT_POSTCOMMIT_ARRIVAL_WAIT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_POSTCOMMIT_ARRIVAL_WAIT_S
    if not math.isfinite(value):
        # NaN would read as "disabled" and +inf would defeat the bounded
        # policy entirely; both are configuration mistakes, not intents.
        return _DEFAULT_POSTCOMMIT_ARRIVAL_WAIT_S
    return value if value > 0.0 else 0.0


def _marathon_postcommit_protect_tokens() -> int:
    """MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS: 0 (default) disables.

    A marathon turn's postcommit (50k+ tokens of think interior to render
    and commit) can never finish inside the standard 0.6s arrival window
    under continuous agent pressure — each next turn aborts it, the retry
    queue starves, and the eventual cost is a multi-10k-token warm
    re-prefill (measured: an 18,011-token, 79s TTFT wall after a 54k-think
    turn, 2026-08-22 chess gauntlet). When set (>0), a pending postcommit
    whose token_count meets the threshold is granted the marathon wait
    below instead of the standard window. Off by default: the tradeoff
    (next-turn TTFB vs the re-prefill wall) is a product decision.

    Since #432 the same threshold also guards the CROSS-session admission
    sweep (EngineSessionManager.abort_cross_session_postcommits), where the
    reported cost of ignoring it was worse: a 130-token vision request from
    a second session killed a 200k commit and bought a 316s re-prefill.
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS")
    if raw is None or not str(raw).strip():
        return 0
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _marathon_postcommit_wait_s() -> float:
    """MTPLX_POSTCOMMIT_MARATHON_WAIT_S: escalated wait cap (default 30s).

    Bounded on purpose — a wedged commit must still lose to the foreground
    eventually; 30s covers the measured marathon commit times with margin
    while staying far below the re-prefill wall it prevents.

    #432 reuses this as the cross-session grace as well, so operators tune
    one bound rather than two. The cross-session window is spent once per
    landed commit, not once per arrival or per retry record; see
    EngineSession.cross_session_postcommit_protection.
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_MARATHON_WAIT_S")
    if raw is None or not str(raw).strip():
        return 30.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 30.0
    if not math.isfinite(value) or value <= 0.0:
        return 30.0
    return value


_DEFAULT_POSTCOMMIT_WAIT_STALL_S = 15.0
_DEFAULT_POSTCOMMIT_WAIT_CEILING_S = 600.0


def _postcommit_wait_stall_s() -> float:
    """MTPLX_POSTCOMMIT_WAIT_STALL_S: heartbeat gap that means "wedged".

    A postcommit prefill stamps the record at every chunk boundary; a 2048
    token chunk is a few seconds even under memory pressure, so a job silent
    for this long is not making progress and the waiter aborts it. Default
    15 s. Values <= 0 disable the progress extension (the bounded wait then
    aborts at its timeout exactly as before 2026-09-03).
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_WAIT_STALL_S")
    if raw is None or not str(raw).strip():
        return _DEFAULT_POSTCOMMIT_WAIT_STALL_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_POSTCOMMIT_WAIT_STALL_S
    if not math.isfinite(value):
        return _DEFAULT_POSTCOMMIT_WAIT_STALL_S
    return value if value > 0.0 else 0.0


def _postcommit_wait_ceiling_s() -> float:
    """MTPLX_POSTCOMMIT_WAIT_CEILING_S: absolute cap on a progress-extended wait."""
    raw = os.environ.get("MTPLX_POSTCOMMIT_WAIT_CEILING_S")
    if raw is None or not str(raw).strip():
        return _DEFAULT_POSTCOMMIT_WAIT_CEILING_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_POSTCOMMIT_WAIT_CEILING_S
    if not math.isfinite(value) or value <= 0.0:
        return _DEFAULT_POSTCOMMIT_WAIT_CEILING_S
    return value


def _wait_while_progressing(future: Any, record: Any, *, timeout_s: float) -> None:
    """``future.result`` with a progress-gated extension past ``timeout_s``.

    The bounded wait exists so a same-session request can pick up the
    previous turn's committed state instead of re-prefilling it. When the
    bound expires while the job is still prefilling, aborting is the worst
    of both worlds: the partial prefill is thrown away and the request then
    redoes the same tokens from scratch (2026-09-03 founder session: 30 s
    waited, job aborted at ~75%, 38 s re-prefill on top -- 68 s of dead air
    for a state the engine held in memory 120 ms earlier). The postcommit's
    remaining work IS the request's own alternative, so a job that keeps
    reaching chunk boundaries is waited for; only a job silent for
    ``stall_s`` (or one that never started) is abandoned at the bound. The
    absolute ceiling keeps a heartbeating-but-endless job from pinning a
    request forever.

    Raises TimeoutError when the wait is abandoned (same contract as
    ``future.result(timeout=...)``), so the caller's abort path is unchanged.
    """

    stall_s = _postcommit_wait_stall_s()
    ceiling_s = max(float(timeout_s), _postcommit_wait_ceiling_s())
    deadline = time.monotonic() + float(timeout_s)
    hard_deadline = time.monotonic() + ceiling_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("postcommit wait bound reached")
        try:
            future.result(timeout=remaining)
            return
        except TimeoutError:
            pass
        except BaseException:
            raise
        if stall_s <= 0.0:
            raise TimeoutError("postcommit wait bound reached")
        started = getattr(record, "started_at_s", None) is not None
        progressing = getattr(record, "progressing", None)
        if not started or not callable(progressing) or not progressing(stall_s=stall_s):
            raise TimeoutError("postcommit wait bound reached; job not progressing")
        if time.monotonic() >= hard_deadline:
            raise TimeoutError("postcommit wait ceiling reached")
        # Still prefilling: grant another slice, never longer than the stall
        # window, so a job that stops mid-slice is caught within stall_s.
        deadline = min(hard_deadline, time.monotonic() + stall_s)


def _postcommit_wait_timeout_s() -> float:
    """Read MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S from the environment.

    Defaults to 8s. Values <= 0 disable the wait (returns 0.0). Bad values
    fall back to the default so a typo does not leave the server hanging.
    """
    raw = os.environ.get("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_POSTCOMMIT_WAIT_TIMEOUT_S
    if value < 0:
        return 0.0
    return value


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(int(minimum), value)


def _near_prefix_max_token_gap() -> int:
    return _env_int(
        "MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP",
        _DEFAULT_NEAR_PREFIX_MAX_TOKEN_GAP,
        minimum=0,
    )


def _near_prefix_min_match_tokens() -> int:
    return _env_int(
        "MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS",
        _DEFAULT_NEAR_PREFIX_MIN_MATCH_TOKENS,
        minimum=1,
    )


def _block_prefix_restore_enabled() -> bool:
    """The single parse of ``MTPLX_SESSION_BLOCK_PREFIX_RESTORE`` (default ON).

    Shared by :mod:`mtplx.generation`, :mod:`mtplx.session_bank` and the
    server so one spelling cannot mean ON in the decode loop and OFF in the
    cold tier.
    """

    return block_prefix_restore_enabled()


def _prefix_block_size() -> int:
    return _env_int(
        "MTPLX_SESSION_PREFIX_BLOCK_SIZE",
        _DEFAULT_PREFIX_BLOCK_SIZE,
        minimum=1,
    )


def _block_prefix_min_match_tokens() -> int:
    return _env_int(
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS",
        _DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
        minimum=1,
    )


@dataclass
class PendingPostcommit:
    """Best-effort SessionBank maintenance currently tied to a session.

    The future remains the compatibility surface; this record adds the
    information needed to make idle maintenance cancellable and observable.
    """

    future: Any
    abort_event: Event = field(default_factory=Event)
    reason: str = "postcommit"
    token_count: int = 0
    created_at_s: float = field(default_factory=time.time)
    started_at_s: float | None = None
    finished_at_s: float | None = None
    last_outcome: dict[str, Any] | None = None
    last_abort_reason: str | None = None
    #: Monotonic stamp of the job's last chunk boundary (its abort check
    #: doubles as the heartbeat). A waiter reads it to tell a job that is
    #: still prefilling from one that is wedged.
    last_progress_mono_s: float | None = None
    progress_ticks: int = 0

    def mark_started(self) -> None:
        if self.started_at_s is None:
            self.started_at_s = time.time()
            self.last_progress_mono_s = time.monotonic()

    def note_progress(self) -> None:
        self.last_progress_mono_s = time.monotonic()
        self.progress_ticks += 1

    def progressing(self, *, stall_s: float) -> bool:
        """Whether the job reached a chunk boundary within ``stall_s``.

        The start stamp alone is not progress: a job that never reaches its
        first chunk boundary inside the caller's bound is abandoned there
        exactly as before, so a wedged render/tokenize phase cannot borrow
        the stall window.
        """
        stamp = self.last_progress_mono_s
        if stamp is None or self.progress_ticks <= 0:
            return False
        return (time.monotonic() - float(stamp)) <= float(stall_s)

    def mark_finished(self, outcome: dict[str, Any] | None = None) -> None:
        self.finished_at_s = time.time()
        if outcome is not None:
            self.last_outcome = outcome

    def abort(self, reason: str) -> bool:
        self.last_abort_reason = str(reason)
        self.abort_event.set()
        cancel = getattr(self.future, "cancel", None)
        if callable(cancel):
            try:
                return bool(cancel())
            except BaseException:
                return False
        return False

    def update_token_count(self, token_count: int) -> None:
        try:
            self.token_count = max(0, int(token_count))
        except (TypeError, ValueError):
            self.token_count = 0

    def to_admin_dict(self) -> dict[str, Any]:
        now = time.time()
        future = self.future
        return {
            "active": bool(future is not None and not getattr(future, "done", lambda: False)()),
            "reason": self.reason,
            "token_count": int(self.token_count),
            "age_s": max(0.0, now - float(self.created_at_s)),
            "created_at_s": self.created_at_s,
            "started_at_s": self.started_at_s,
            "finished_at_s": self.finished_at_s,
            "abort_requested": self.abort_event.is_set(),
            "last_abort_reason": self.last_abort_reason,
            "last_outcome": self.last_outcome,
        }


class EngineSession:
    def __init__(self, session_id: str, *, idle_ttl_s: float = DEFAULT_IDLE_TTL_S) -> None:
        self.session_id = str(session_id)
        self.idle_ttl_s = float(idle_ttl_s)
        self.created_at_s = time.time()
        self.last_access_s = self.created_at_s
        self.committed_token_ids: tuple[int, ...] = ()
        self.boundaries: list[BoundarySnapshot] = []
        self.in_flight = False
        self.in_flight_started_s: float | None = None
        self.last_commit_s: float | None = None
        self.last_finish_reason: str | None = None
        self.last_cache_miss_reason: str | None = CacheMissReason.NEW_SESSION.value
        self.last_restore_mode: str = "cold"
        self.bytes_estimate = 0
        self.revision = 0
        self._lock = Lock()
        # Reference to the most recent postcommit work scheduled for this
        # session. The next request in this session waits briefly on this
        # before acquiring the session lock so the SessionBank entry is
        # available when its prefix lookup runs - avoiding the cold-prefill
        # cascade documented in PR #34. Always written/read while NOT holding
        # the session lock to preserve the no-deadlock ordering. Type kept as
        # Any to avoid pulling concurrent.futures into hot import paths.
        self._pending_postcommit: PendingPostcommit | None = None
        # Per-session lock guarding `pending_postcommit` reads/writes. This is
        # SEPARATE from `_lock` (which guards `in_flight_generation`) so that
        # `wait_for_pending_postcommit` can serialize access to the future
        # field without ever contending with the foreground/in-flight lock,
        # preserving the no-deadlock ordering callers rely on. The lock is
        # only held while reading or mutating `pending_postcommit` itself,
        # never while awaiting on the future.
        self._postcommit_lock = Lock()
        # Last wait outcome, exposed via to_admin_dict for the metrics endpoint.
        self.last_postcommit_wait: dict[str, Any] | None = None
        self.last_postcommit_outcome: dict[str, Any] | None = None
        # Monotonic deadline for cross-session marathon protection (#432).
        # Armed on the first protected admission sweep and cleared only when
        # a postcommit actually lands, so the whole abort/re-arm chain shares
        # one bounded grace instead of one per record. See
        # `cross_session_postcommit_protection`.
        self._cross_session_protect_deadline_s: float | None = None

    @property
    def pending_postcommit(self) -> Any:
        record = self._pending_postcommit
        return None if record is None else record.future

    @pending_postcommit.setter
    def pending_postcommit(self, future: Any) -> None:
        if future is None:
            self._pending_postcommit = None
        elif isinstance(future, PendingPostcommit):
            self._pending_postcommit = future
        else:
            self._pending_postcommit = PendingPostcommit(future=future)

    @property
    def prefix_len(self) -> int:
        return len(self.committed_token_ids)

    def touch(self) -> None:
        self.last_access_s = time.time()

    def is_stale(self, *, now_s: float | None = None) -> bool:
        now = time.time() if now_s is None else float(now_s)
        return now - self.last_access_s > self.idle_ttl_s

    def set_pending_postcommit(
        self,
        future: Any,
        *,
        abort_event: Event | None = None,
        reason: str = "postcommit",
        token_count: int = 0,
    ) -> PendingPostcommit:
        """Record a reference to in-flight postcommit work for this session.

        The next request in this session calls wait_for_pending_postcommit()
        before acquiring the session lock so the prior turn's SessionBank
        entry is visible at lookup time. Older references are dropped on each
        new commit; only the most recent matters for the next turn's lookup.

        The write is guarded by `_postcommit_lock` so it cannot race with a
        concurrent `wait_for_pending_postcommit` reading the field.
        """
        record = (
            future
            if isinstance(future, PendingPostcommit)
            else PendingPostcommit(
                future=future,
                abort_event=abort_event or Event(),
                reason=str(reason or "postcommit"),
                token_count=max(0, int(token_count or 0)),
            )
        )
        with self._postcommit_lock:
            self._pending_postcommit = record
        return record

    def has_pending_postcommit(self) -> bool:
        with self._postcommit_lock:
            return self._pending_postcommit is not None

    def pending_postcommit_admin(self) -> dict[str, Any] | None:
        with self._postcommit_lock:
            record = self._pending_postcommit
            if record is None:
                return None
            return record.to_admin_dict()

    def cross_session_postcommit_protection(
        self,
        *,
        protect_tokens: int,
        grace_s: float,
    ) -> dict[str, Any] | None:
        """Should this session's pending postcommit survive a foreign request?

        Returns None when the caller must abort exactly as before (no
        pending record, protection disabled, commit below the threshold, or
        the grace already spent). Returns a grant dict when the commit is
        marathon-sized and still inside its grace window.

        Issue #432: `MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS` guarded only
        the same-session arrival path, so a 130-token vision request from
        another session killed a 200k-token commit at admission and the deep
        session then paid a 316s full re-prefill. The docstring rationale for
        the unconditional abort prices the arriving request's TTFT but not
        the destroyed checkpoint.

        THE BOUND: the deadline is armed once, on the first protected sweep,
        and is NOT refreshed by later arrivals. It is cleared only by a
        postcommit that actually lands (`finish_pending_postcommit` with a
        stored outcome). An aborted commit re-arms with a fresh record up to
        16 times, so a per-record deadline would multiply the grace by the
        retry chain; keying it to the session caps total cross-session
        protection at one `grace_s` window (default 30s, the same
        `MTPLX_POSTCOMMIT_MARATHON_WAIT_S` bound the same-session path uses)
        per landed commit. Past the deadline the foreground wins every time,
        so a wedged commit cannot starve other sessions.
        """
        if protect_tokens <= 0 or grace_s <= 0.0:
            return None
        with self._postcommit_lock:
            record = self._pending_postcommit
            if record is None:
                return None
            try:
                token_count = int(getattr(record, "token_count", 0) or 0)
            except (TypeError, ValueError):
                token_count = 0
            if token_count < protect_tokens:
                return None
            now = time.monotonic()
            deadline = self._cross_session_protect_deadline_s
            if deadline is None:
                deadline = now + float(grace_s)
                self._cross_session_protect_deadline_s = deadline
            if now >= deadline:
                return None
            remaining_s = deadline - now
        return {
            "session_id": self.session_id,
            "token_count": token_count,
            "protect_tokens": int(protect_tokens),
            "grace_s": float(grace_s),
            "grace_remaining_s": round(remaining_s, 3),
        }

    def abort_pending_postcommit(self, reason: str) -> dict[str, Any]:
        with self._postcommit_lock:
            record = self._pending_postcommit
        if record is None:
            return {"aborted": False, "reason": "no_pending"}
        cancelled = record.abort(reason)
        return {
            "aborted": True,
            "reason": str(reason),
            "future_cancelled": bool(cancelled),
        }

    def finish_pending_postcommit(
        self,
        record: PendingPostcommit,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        if record is None:
            return
        record.mark_finished(outcome)
        if outcome is not None:
            self.last_postcommit_outcome = outcome
        landed = bool((outcome or {}).get("stored"))
        with self._postcommit_lock:
            if self._pending_postcommit is record:
                self._pending_postcommit = None
            if landed:
                # A landed commit re-arms the cross-session marathon grace
                # (#432). Aborted/abandoned outcomes deliberately do NOT:
                # the re-armed retry inherits the spent deadline so the
                # whole chain shares one bounded window.
                self._cross_session_protect_deadline_s = None

    def wait_for_pending_postcommit(
        self,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Bounded wait for the prior postcommit job to land.

        Returns a small telemetry dict the caller can attach to request stats:
            {"waited": bool, "elapsed_s": float, "outcome": str,
             "timeout_s": float}

        `outcome` is one of:
            - "no_pending"     : nothing was scheduled, no wait performed
            - "completed"      : the future resolved within the timeout
            - "timeout"        : the future did not resolve in time; the
                                 request should fall through to a cold
                                 prefill rather than hang
            - "error:<Type>"   : the future raised; we swallow it because the
                                 postcommit's job is best-effort caching, not
                                 correctness
            - "disabled"       : timeout_s <= 0; wait short-circuits

        CRITICAL: This must be called WITHOUT the session lock (`_lock`)
        held. The postcommit work runs on the model scheduler's owner thread;
        a foreground request that holds the session lock and then waits on
        scheduler-bound work risks priority inversion against other
        same-session commits queued behind it. Callers should invoke this
        before entering `in_flight_generation()`.

        Concurrency contract (the bug fix from PR #37 review):

        Two concurrent same-session waiters MUST observe the SAME active
        future and either both report "completed" (on resolve) or both
        report "timeout" (on timeout). Neither must ever observe
        "no_pending" while a real postcommit is still in flight.

        We achieve that by capturing the current future under
        `_postcommit_lock`, releasing the lock BEFORE awaiting on the
        future (so other same-session waiters can observe the same future),
        then re-acquiring the lock after the wait and clearing
        `pending_postcommit` ONLY if it is still the same future identity
        (`is` comparison). If a newer commit superseded the future while we
        were waiting, the newer reference belongs to the next caller and we
        leave it alone.
        """
        if timeout_s is None:
            timeout_s = _postcommit_wait_timeout_s()
        timeout_s = float(timeout_s)
        # Capture the current future under the lock so a concurrent
        # `set_pending_postcommit` cannot tear our read. The lock is released
        # immediately - we MUST NOT hold it while awaiting on the future, and
        # leaving it visible on the session is the whole point: a second
        # same-session waiter must observe the same future, not "no_pending".
        with self._postcommit_lock:
            record = self._pending_postcommit
        if record is None:
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "no_pending",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            return outcome
        future = record.future
        if not hasattr(future, "result"):
            # Defensive: anything stashed on the session that does not look
            # like a future is treated as "nothing to wait on". We do NOT
            # clear it - the field will be overwritten on the next legitimate
            # set_pending_postcommit call.
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "no_pending",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            return outcome
        if timeout_s <= 0.0:
            # Disabled mode: do NOT touch `pending_postcommit`. Operators
            # toggle this dynamically via MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S; if
            # they re-enable later we want the existing future still visible
            # so it is not silently dropped.
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "disabled",
                "timeout_s": timeout_s,
            }
            self.last_postcommit_wait = outcome
            record.last_outcome = outcome
            return outcome
        t0 = time.monotonic()
        # We catch BaseException because any failure in the postcommit must
        # not propagate into the foreground request: the wait is a best-effort
        # cache warmup, not a correctness dependency. Timeout is the most
        # common non-success outcome and is reported distinctly so operators
        # can spot a stuck postcommit lane.
        progress_extended_s = 0.0
        try:
            _wait_while_progressing(future, record, timeout_s=timeout_s)
            outcome = {
                "waited": True,
                "elapsed_s": time.monotonic() - t0,
                "outcome": "completed",
                "timeout_s": timeout_s,
            }
            progress_extended_s = max(0.0, outcome["elapsed_s"] - timeout_s)
            if progress_extended_s > 0.0:
                outcome["progress_extended_s"] = progress_extended_s
            # "completed" only means the future resolved — abandoned jobs
            # also complete (they return normally after logging their own
            # outcome). Surface the job's result so receipts distinguish a
            # stored commit from ran-and-gave-up (2026-08-21: every
            # 3.7-10.2s "completed" wait could hide an abandoned job while
            # the committed stream froze).
            job_outcome = getattr(record, "last_outcome", None)
            if isinstance(job_outcome, dict) and "stored" in job_outcome:
                outcome["job_stored"] = bool(job_outcome.get("stored"))
                outcome["job_mode"] = job_outcome.get("mode")
                outcome["job_reason"] = job_outcome.get("reason")
        except BaseException as exc:
            exc_name = type(exc).__name__
            preempted_cancel = (
                exc_name == "CancelledError"
                and record.abort_event.is_set()
                and record.last_abort_reason == "foreground_preempted_postcommit"
            )
            label = (
                "timeout"
                if exc_name == "TimeoutError" or preempted_cancel
                else f"error:{exc_name}"
            )
            future_cancelled = False
            if label == "timeout":
                future_cancelled = record.abort("foreground_preempted_postcommit")
            outcome = {
                "waited": True,
                "elapsed_s": time.monotonic() - t0,
                "outcome": label,
                "timeout_s": timeout_s,
                "abort_requested": label == "timeout",
                "future_cancelled": bool(future_cancelled),
                "abort_reason": (
                    "foreground_preempted_postcommit" if label == "timeout" else None
                ),
            }
        # Clear the reference ONLY if it is still the same future we
        # observed. A concurrent same-session commit may have superseded it
        # while we were waiting; in that case the newer future belongs to
        # the next caller and we must not stomp it. Identity check (`is`)
        # is required: equality could collapse two distinct futures with
        # the same result.
        with self._postcommit_lock:
            if self._pending_postcommit is record:
                self._pending_postcommit = None
        self.last_postcommit_wait = outcome
        self.last_postcommit_outcome = outcome
        record.mark_finished(outcome)
        return outcome

    def resolve_pending_postcommit_for_request(self) -> dict[str, Any]:
        """Foreground-request policy for a prior turn's pending postcommit.

        Default (POSTCOMMIT_STALL_DESIGN step 2, 2026-07-17): a live user
        request never queues behind cache maintenance. If a postcommit is
        still in flight, abort it immediately and admit the request: the
        canonical snapshot is superseded by the turn about to run anyway,
        and the bank still holds the prompt-prefix boundary plus any
        live-frontier reference for warm restore. This replaces the
        unconditional bounded wait (default 8s) that agent clients paid on
        every tool continuation: the wait usually timed out (a 20k-history
        re-encode needs longer than the bound), aborted the job anyway, and
        the user watched dead air before prefill even began.

        B' amendment (2026-08-06, arrival-wait design note): the immediate
        abort's "superseded anyway" premise is disproven by the causal
        probes — the pending snapshot is the arriving request's own
        exact-prefix restore anchor (SSD-off receipts: 8-16ms warm
        residual vs 0.66-1.17s on degraded anchors). A pending job that
        has NOT started still aborts immediately (it would run after this
        request and commit a stale revision — zero value). A RUNNING job
        is granted a bounded finish window,
        MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S (default 0.6s; <= 0 restores the
        exact 2026-07-17 immediate abort), then aborts on timeout exactly
        as before. This composes with — and does not replace — the
        worker's own 2.0s foreground-pressure self-yield.

        Operators restore the old blocking behavior by setting
        MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S explicitly.
        """
        raw = os.environ.get("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S")
        if raw is not None and raw.strip():
            return self.wait_for_pending_postcommit()
        with self._postcommit_lock:
            record = self._pending_postcommit
        if record is None:
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "no_pending",
                "timeout_s": 0.0,
            }
            self.last_postcommit_wait = outcome
            return outcome
        future = record.future
        if hasattr(future, "done") and future.done():
            outcome = {
                "waited": False,
                "elapsed_s": 0.0,
                "outcome": "completed",
                "timeout_s": 0.0,
            }
        else:
            arrival_wait_s = _postcommit_arrival_wait_s()
            marathon_protected = False
            protect_tokens = _marathon_postcommit_protect_tokens()
            if (
                protect_tokens > 0
                and int(getattr(record, "token_count", 0) or 0) >= protect_tokens
            ):
                # Marathon protection: give a big commit the room to land
                # instead of aborting it into the starvation/re-prefill
                # cycle. See _marathon_postcommit_protect_tokens.
                arrival_wait_s = max(arrival_wait_s, _marathon_postcommit_wait_s())
                marathon_protected = True
            waited_s = 0.0
            finished_within_window = False
            if (
                arrival_wait_s > 0.0
                and record.started_at_s is not None
                and hasattr(future, "result")
            ):
                wait_started = time.monotonic()
                # BaseException guard mirrors wait_for_pending_postcommit:
                # the postcommit is best-effort caching, never a
                # correctness dependency of the arriving request.
                try:
                    future.result(timeout=arrival_wait_s)
                    finished_within_window = True
                except BaseException:
                    finished_within_window = bool(
                        getattr(future, "done", lambda: False)()
                        and not getattr(future, "cancelled", lambda: False)()
                    )
                waited_s = time.monotonic() - wait_started
            if finished_within_window:
                outcome = {
                    "waited": True,
                    "elapsed_s": waited_s,
                    "outcome": "completed",
                    "timeout_s": arrival_wait_s,
                    "arrival_wait_s": arrival_wait_s,
                }
                if marathon_protected:
                    outcome["marathon_protected"] = True
            else:
                future_cancelled = record.abort("foreground_preempted_postcommit")
                outcome = {
                    "waited": waited_s > 0.0,
                    "elapsed_s": waited_s,
                    "outcome": "aborted_for_foreground",
                    "timeout_s": arrival_wait_s,
                    "arrival_wait_s": arrival_wait_s,
                    "abort_requested": True,
                    "future_cancelled": bool(future_cancelled),
                    "abort_reason": "foreground_preempted_postcommit",
                }
                if marathon_protected:
                    outcome["marathon_protected"] = True
        with self._postcommit_lock:
            if self._pending_postcommit is record:
                self._pending_postcommit = None
        self.last_postcommit_wait = outcome
        self.last_postcommit_outcome = outcome
        record.mark_finished(outcome)
        return outcome

    def try_begin_generation(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self.in_flight = True
        self.in_flight_started_s = time.time()
        self.touch()
        return True

    def end_generation(self) -> None:
        self.in_flight = False
        self.in_flight_started_s = None
        self.touch()
        self._lock.release()

    @contextmanager
    def in_flight_generation(self) -> Iterator["EngineSession"]:
        if not self.try_begin_generation():
            raise EngineSessionBusy(f"session {self.session_id} is already in flight")
        try:
            yield self
        finally:
            self.end_generation()

    def commit(
        self,
        *,
        prompt_ids: list[int] | tuple[int, ...],
        generated_ids: list[int] | tuple[int, ...],
        finish_reason: str,
        boundary_kind: str = "assistant_end",
        nbytes: int = 0,
    ) -> EngineSessionCommit:
        # "tool_calls" is the OpenAI label the bridge stamps on a turn that
        # stopped naturally after complete tool calls; the KV boundary is as
        # safe as "stop". Refusing it (as before 2026-09-03) left every agent
        # tool round without a committed stream, so the committed-think and
        # committed-body canonicalization never applied to the turns that
        # need it most, and each tool result re-prefilled the whole previous
        # assistant turn. Aborted/cancelled/error finishes stay refused.
        if finish_reason not in {"stop", "length", "tool_calls"}:
            return EngineSessionCommit(False, f"unsafe_finish:{finish_reason}", self.prefix_len)
        tokens = tuple(int(token) for token in prompt_ids) + tuple(int(token) for token in generated_ids)
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = finish_reason
        self.bytes_estimate = int(nbytes)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens, nbytes=nbytes)
        return EngineSessionCommit(True, "committed", self.prefix_len)

    def commit_prompt_prefix(
        self,
        *,
        prompt_ids: list[int] | tuple[int, ...],
        finish_reason: str,
        boundary_kind: str = "prompt_prefix",
    ) -> EngineSessionCommit:
        """Publish a safe foreground prompt prefix for async postcommit waiters.

        Streaming tool-call responses cannot commit the structured assistant
        history directly; that history is canonicalized by a low-priority
        retokenized postcommit. The next tool-result request still needs to
        resolve to this EngineSession so it can wait for that postcommit
        instead of cold-prefilling. Publishing the already-prefilled prompt
        prefix gives the resolver a stable session anchor without claiming the
        assistant/tool history has landed yet.
        """
        tokens = tuple(int(token) for token in prompt_ids)
        if not tokens:
            return EngineSessionCommit(False, "empty_prompt_prefix", self.prefix_len)
        current = self.committed_token_ids
        if current:
            if len(tokens) < len(current):
                return EngineSessionCommit(
                    False,
                    "prompt_prefix_older_than_session",
                    self.prefix_len,
                )
            if tokens[: len(current)] != current:
                return EngineSessionCommit(
                    False,
                    "prompt_prefix_not_extending_session",
                    self.prefix_len,
                )
            if len(tokens) == len(current):
                return EngineSessionCommit(False, "prompt_prefix_unchanged", self.prefix_len)
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = str(finish_reason)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens)
        return EngineSessionCommit(True, "committed_prompt_prefix", self.prefix_len)

    def commit_retokenized_prefix(
        self,
        *,
        token_ids: list[int] | tuple[int, ...],
        expected_revision: int | None = None,
        boundary_kind: str = "retokenized_history",
        nbytes: int = 0,
    ) -> EngineSessionCommit:
        """Publish canonical retokenized history after async postcommit.

        Streaming tool-call turns first publish the foreground prompt as a
        temporary same-session anchor. The idle postcommit later renders the
        canonical assistant/tool history. Publishing that canonical prefix
        prevents the next OpenCode turn from resolving by a stale prompt
        boundary that can differ by one chat-template token.
        """
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return EngineSessionCommit(False, "empty_retokenized_prefix", self.prefix_len)
        if expected_revision is not None and self.revision != int(expected_revision):
            return EngineSessionCommit(False, "stale_session_revision", self.prefix_len)
        current = self.committed_token_ids
        commit_reason = "committed_retokenized_prefix"
        if current:
            if len(tokens) < len(current):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_older_than_session",
                    self.prefix_len,
                )
            matched = common_prefix_len(current, tokens)
            block_boundary = block_aligned_prefix_len(
                matched,
                block_size=_prefix_block_size(),
            )
            block_min_match = max(_prefix_block_size(), _block_prefix_min_match_tokens())
            if (
                matched < len(current)
                and (len(current) - matched) > _near_prefix_max_token_gap()
                and (
                    not _block_prefix_restore_enabled()
                    or block_boundary < block_min_match
                )
            ):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_not_extending_session",
                    self.prefix_len,
                )
            if (
                matched < len(current)
                and (len(current) - matched) > _near_prefix_max_token_gap()
            ):
                commit_reason = "committed_retokenized_prefix_after_block_overlap"
            if len(tokens) == len(current) and matched == len(current):
                return EngineSessionCommit(
                    False,
                    "retokenized_prefix_unchanged",
                    self.prefix_len,
                )
        self.committed_token_ids = tokens
        self.last_commit_s = time.time()
        self.last_finish_reason = "postcommit"
        self.bytes_estimate = int(nbytes)
        self.revision += 1
        self._record_interval_boundaries(tokens)
        self.add_boundary(boundary_kind, tokens, nbytes=nbytes)
        return EngineSessionCommit(True, commit_reason, self.prefix_len)

    def add_boundary(
        self,
        kind: str,
        token_ids: list[int] | tuple[int, ...],
        *,
        bank_token_hash: str | None = None,
        nbytes: int = 0,
        snapshot_epoch: int | None = None,
    ) -> BoundarySnapshot:
        epoch = len(self.boundaries) if snapshot_epoch is None else int(snapshot_epoch)
        boundary = BoundarySnapshot(
            kind=str(kind),
            token_len=len(token_ids),
            token_hash=token_hash_short(token_ids),
            bank_token_hash=bank_token_hash,
            nbytes=int(nbytes),
            snapshot_epoch=epoch,
        )
        self.boundaries.append(boundary)
        self.touch()
        return boundary

    def nearest_boundary_at_or_before(self, token_len: int) -> BoundarySnapshot | None:
        candidates = [boundary for boundary in self.boundaries if boundary.token_len <= token_len]
        if not candidates:
            self.last_cache_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return None
        return max(candidates, key=lambda boundary: boundary.token_len)

    def _record_interval_boundaries(self, token_ids: tuple[int, ...], *, every: int = 512) -> None:
        existing = {boundary.token_len for boundary in self.boundaries}
        for token_len in range(every, len(token_ids), every):
            if token_len not in existing:
                self.add_boundary("interval_512", token_ids[:token_len])

    def to_admin_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prefix_len": self.prefix_len,
            "bytes": self.bytes_estimate,
            "created_at_s": self.created_at_s,
            "last_access_s": self.last_access_s,
            "last_commit_s": self.last_commit_s,
            "last_finish_reason": self.last_finish_reason,
            "revision": self.revision,
            "in_flight": self.in_flight,
            "in_flight_started_s": self.in_flight_started_s,
            "last_cache_miss_reason": self.last_cache_miss_reason,
            "last_restore_mode": self.last_restore_mode,
            "last_postcommit_wait": self.last_postcommit_wait,
            "last_postcommit_outcome": self.last_postcommit_outcome,
            "pending_postcommit": bool(self.pending_postcommit is not None),
            "pending_postcommit_detail": self.pending_postcommit_admin(),
            "boundaries": [
                {
                    "kind": boundary.kind,
                    "token_len": boundary.token_len,
                    "token_hash": boundary.token_hash,
                    "bank_token_hash": boundary.bank_token_hash,
                    "nbytes": boundary.nbytes,
                    "snapshot_epoch": boundary.snapshot_epoch,
                    "created_at_s": boundary.created_at_s,
                }
                for boundary in self.boundaries[-32:]
            ],
        }


class EngineSessionManager:
    def __init__(
        self,
        *,
        bank: SessionBank | None = None,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        cold_tier: Any | None = None,
        model_weights_bytes: int | None = None,
        memory_plan: Any | None = None,
    ) -> None:
        # Byte caps resolve model-aware by default (v2): unset or "auto" env
        # gives the bank half of the RAM surplus left after the model weights
        # (floored 1 GiB, capped 48 GiB), so a 32 GB Mac never inherits the
        # old flat 24 GiB budget while a 128 GB Mac keeps a big warm cache.
        # Explicit byte values (MTPLX_SESSION_BANK_MAX_BYTES /
        # MTPLX_SESSION_BANK_PER_SESSION_BYTES, e.g. "16G") keep their exact
        # semantics; per-session is additionally clamped to the bank budget.
        # The entry-count cap stays overridable via
        # MTPLX_SESSION_BANK_MAX_ENTRIES (plain integer).
        if bank is None:
            resolved_max_bytes, auto_active = resolve_session_bank_max_bytes(
                model_weights_bytes,
                memory_plan=memory_plan,
            )
            bank = SessionBank(
                max_entries=_session_bank_max_entries(),
                max_bytes=resolved_max_bytes,
                per_session_max_bytes=resolve_session_bank_per_session_bytes(
                    resolved_max_bytes,
                    auto_active=auto_active,
                ),
                idle_ttl_s=idle_ttl_s,
                cold_tier=cold_tier,
            )
            if (
                auto_active
                and memory_plan is not None
                and getattr(memory_plan, "available", False)
            ):
                # The "guard that turns on" (#305): while live requests hold
                # little KV the bank keeps its full idle budget; as a
                # long-context request materializes KV, the ceiling walks
                # down and evictions demote entries ahead of any swap.
                # Working set = allocator active minus weights minus the
                # bank's own bytes — the same attribution the dashboard
                # reports. mlx is imported lazily so plan-carrying tests
                # without a GPU fall back to the static budget (the bank
                # counts, not swallows, ceiling failures).
                from mtplx.memory_plan import (
                    bank_dynamic_ceiling,
                    transient_reserve_bytes,
                )

                def _dynamic_ceiling(
                    _bank: SessionBank = bank, _plan: Any = memory_plan
                ) -> int:
                    import mlx.core as mx

                    active = int(mx.get_active_memory())
                    working = (
                        active
                        - int(_plan.model_weights_bytes)
                        - int(_bank.total_nbytes)
                    )
                    # Observed spike (peak high-water over current active)
                    # replaces the static 3 GiB reserve: a deep chunked
                    # prefill measured 12.4 GiB over active, and with only
                    # the static term the bank held entries while the
                    # allocator peak kissed 0.99+ of the Metal limit —
                    # tripping the warning banner on every long coding turn
                    # (2026-08-29 receipts).
                    reserve = transient_reserve_bytes(
                        int(mx.get_peak_memory()),
                        active,
                        play_bytes=int(_plan.usable_bytes)
                        - int(_plan.model_weights_bytes),
                    )
                    return bank_dynamic_ceiling(
                        _plan, max(0, working), transient_bytes=reserve
                    )

                bank.dynamic_ceiling_fn = _dynamic_ceiling
            # Visible on the daemon console on purpose (#229/#230): the
            # 2.4.2 notes promised this line but it shipped as logger.info,
            # which default logging swallows — users debugging "the cache
            # stopped working" had no way to see the resolved budgets.
            if (
                auto_active
                and memory_plan is not None
                and getattr(memory_plan, "available", False)
            ):
                budget_mode = (
                    "auto: machine memory plan, yields to live KV"
                )
            elif auto_active:
                budget_mode = "auto: half of post-model RAM surplus"
            elif _explicit_max_bytes_env_set():
                budget_mode = "explicit"
            else:
                # Auto sizing could not engage (model size or RAM unknown)
                # and nothing was configured: say so instead of implying the
                # user chose this budget.
                budget_mode = "legacy default; auto sizing unavailable"
            print(
                "[mtplx] session-bank budget: "
                f"{bank.max_bytes / 1024**3:.1f}G total "
                f"({budget_mode}), "
                f"{bank.per_session_max_bytes / 1024**3:.1f}G per-session cap, "
                f"{bank.max_entries} entries max, model weights "
                + (
                    f"{model_weights_bytes / 1024**3:.1f}G"
                    if model_weights_bytes
                    else "unknown"
                )
                + ". Override: MTPLX_SESSION_BANK_MAX_BYTES / "
                "MTPLX_SESSION_BANK_PER_SESSION_BYTES (sizes like 12G or 12GiB).",
                flush=True,
            )
        self.bank = bank
        self.idle_ttl_s = float(idle_ttl_s)
        self._sessions: dict[str, EngineSession] = {}
        self._lock = Lock()
        self.last_prefix_diagnostic: dict[str, Any] | None = None

    def resolve_session_id(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        user: str | None = None,
        chat_id: str | None = None,
        conversation_id: str | None = None,
        prompt_ids: list[int] | tuple[int, ...] | None = None,
        diagnostic_out: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        # The caller owns its evidence even while another request resolves
        # during a postcommit wait. The last value is only for health.
        self.last_prefix_diagnostic = None
        if diagnostic_out is not None:
            diagnostic_out.clear()

        def record(diagnostic: dict[str, Any]) -> None:
            self.last_prefix_diagnostic = diagnostic
            if diagnostic_out is not None:
                diagnostic_out.update(diagnostic)

        headers = headers or {}
        metadata = metadata or {}
        lowered_headers = {
            str(key).lower(): value
            for key, value in headers.items()
        }
        for key in (
            "x-mtplx-session-id",
            # OpenCode's V1 request path stamps both of these with its own
            # session id on every request (session/llm/request.ts). Trusting
            # the client's stable id beats prompt-prefix inference when the
            # client rewrites history mid-loop (2026-08-01 live session:
            # on-wire prompt shrank at r4/r8/r9 and prefix identity churned).
            "x-session-affinity",
            "x-session-id",
            "x-openwebui-chat-id",
            "x-openwebui-user-id",
        ):
            value = lowered_headers.get(key)
            if str(value or "").strip():
                return str(value).strip(), f"header.{key}"
        for key in ("session_id", "mtplx_session_id", "chat_id", "conversation_id"):
            value = metadata.get(key)
            if value:
                return str(value), f"metadata.{key}"
        if user:
            return str(user), "user"
        if chat_id:
            return str(chat_id), "chat_id"
        if conversation_id:
            return str(conversation_id), "conversation_id"
        if prompt_ids:
            best = self.longest_prefix_session(prompt_ids)
            if best is not None:
                record(self._prefix_diagnostic(
                    prompt_ids,
                    selected=best,
                    exact=True,
                ))
                return best.session_id, "longest_prefix"
            pending, matched = self.pending_near_prefix_session(prompt_ids)
            if pending is not None:
                diagnostic = self._prefix_diagnostic(prompt_ids)
                diagnostic.update(
                    {
                        "best_session_id": pending.session_id,
                        "best_prefix_len": len(pending.committed_token_ids),
                        "matched_prefix_len": int(matched),
                        "divergence_at_token": int(matched),
                        "best_token_hash": token_hash_short(
                            pending.committed_token_ids
                        ),
                        "reason": "pending_postcommit_near_prefix_match",
                        "near_prefix_gap": len(pending.committed_token_ids)
                        - int(matched),
                    }
                )
                record(diagnostic)
                return pending.session_id, "pending_postcommit_near_prefix"
            best, matched = self.best_common_prefix_session(prompt_ids)
            if best is not None:
                diagnostic = self._prefix_diagnostic(prompt_ids)
                diagnostic.update(
                    {
                        "best_session_id": best.session_id,
                        "best_prefix_len": len(best.committed_token_ids),
                        "matched_prefix_len": int(matched),
                        "divergence_at_token": int(matched),
                        "best_token_hash": token_hash_short(
                            best.committed_token_ids
                        ),
                        "reason": "common_prefix_reuse",
                    }
                )
                record(diagnostic)
                return best.session_id, "common_prefix_reuse"
            record(self._prefix_diagnostic(prompt_ids))
        else:
            self.last_prefix_diagnostic = None
        return _new_anon_session_id(), "new"

    def get_or_create(self, session_id: str) -> EngineSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = EngineSession(session_id, idle_ttl_s=self.idle_ttl_s)
                self._sessions[session_id] = session
            session.touch()
            return session

    def peek(self, session_id: str) -> EngineSession | None:
        """Read-only lookup: no creation, no touch. Pre-encode consumers
        (committed-reasoning canonicalization) must not mint sessions or
        refresh TTLs for requests that may never adopt the id."""
        with self._lock:
            return self._sessions.get(session_id)

    def _sessions_snapshot(self) -> list[EngineSession]:
        with self._lock:
            return list(self._sessions.values())

    def abort_cross_session_postcommits(
        self,
        *,
        except_session_id: str | None,
        reason: str = "cross_session_foreground_preempted",
    ) -> dict[str, Any] | None:
        """Abort every OTHER session's pending idle postcommit.

        The idle postcommit's foreground grace exists so a SAME-session
        follow-up request can profit from the commit it is waiting on
        (agent tool loops: losing that commit costs a 2-4k block-salvage
        re-prefill). A request from a DIFFERENT session gains nothing from
        someone else's commit — it just pays the commit's runtime and its
        memory-bandwidth residue (2026-08-05 showdown receipts: 0.5-3.5GB
        retokenized_history jobs finishing inside the 2s grace taxed the
        next request's TTFT by the job's remaining runtime and degraded
        its decode ~30-50% at <2s cadence). Called at request admission,
        off the scheduler-owner thread. Best-effort: a job past its last
        abort check still completes; that window is a few hundred ms.

        Marathon exception (#432, reporter nomishbhardwaj): that trade
        inverts once the foreign commit is huge. A 130-token vision request
        interleaved into a 200k-token agent session killed the session's
        pending commit at admission every time, and the deep session then
        paid a 316s full re-prefill (measured: cached=0 at 207k, cached=18432
        of 114655 at 115k) to save the vision request a few seconds. When
        MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS is set, a commit at or above
        that size keeps a bounded grace instead of dying here; see
        EngineSession.cross_session_postcommit_protection for the bound.
        """
        aborted: list[str] = []
        protected: list[dict[str, Any]] = []
        protect_tokens = _marathon_postcommit_protect_tokens()
        grace_s = _marathon_postcommit_wait_s()
        for other in self._sessions_snapshot():
            session_id = getattr(other, "session_id", None)
            if except_session_id is not None and session_id == except_session_id:
                continue
            try:
                if not other.has_pending_postcommit():
                    continue
                if protect_tokens > 0:
                    grant = other.cross_session_postcommit_protection(
                        protect_tokens=protect_tokens,
                        grace_s=grace_s,
                    )
                    if grant is not None:
                        protected.append(grant)
                        continue
                outcome = other.abort_pending_postcommit(reason)
                if outcome.get("aborted"):
                    aborted.append(str(session_id))
            except BaseException:
                continue
        if not aborted and not protected:
            return None
        result: dict[str, Any] = {
            "count": len(aborted),
            "sessions": aborted[:8],
            "reason": reason,
        }
        if protected:
            result["marathon_protected"] = protected[:8]
            result["marathon_protected_count"] = len(protected)
        return result

    @contextmanager
    def generation_slot(
        self,
        session: EngineSession,
        *,
        source: str | None = None,
    ) -> Iterator[EngineSession]:
        acquired = session
        if not session.try_begin_generation():
            if str(source or "") not in IMPLICIT_SESSION_SOURCES:
                raise EngineSessionBusy(
                    f"session {session.session_id} is already in flight"
                )
            while True:
                acquired = self.get_or_create(_new_anon_session_id())
                if acquired.try_begin_generation():
                    break
        try:
            yield acquired
        finally:
            acquired.end_generation()

    def longest_prefix_session(self, token_ids: list[int] | tuple[int, ...]) -> EngineSession | None:
        tokens = tuple(int(token) for token in token_ids)
        best: EngineSession | None = None
        for session in self._sessions_snapshot():
            prefix = session.committed_token_ids
            if not prefix:
                continue
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.committed_token_ids):
                best = session
        return best

    def best_common_prefix_session(
        self, token_ids: list[int] | tuple[int, ...]
    ) -> tuple["EngineSession | None", int]:
        """Deepest shared-prefix session past the reuse thresholds, or None.

        Identity continuity for mutated histories: when a mid-prefix byte
        changed (transcript compaction flip, client edit-rewrite), the exact
        longest_prefix_session misses and a fresh anon id would fork the
        session bank. Restores already handle mid-prefix divergence via
        boundary clones, so adopting the mutated conversation's existing id
        is strictly better than forking. A 64-token probe rejects unrelated
        sessions before the O(prefix) compare.
        """
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            return None, 0
        probe = tokens[:_COMMON_PREFIX_PROBE_TOKENS]
        best: EngineSession | None = None
        best_common = 0
        for session in self._sessions_snapshot():
            prefix = session.committed_token_ids
            if not prefix:
                continue
            if prefix[: len(probe)] != probe[: len(prefix)]:
                continue
            common = common_prefix_len(prefix, tokens)
            if common > best_common:
                best_common = common
                best = session
        threshold = max(
            _COMMON_PREFIX_REUSE_MIN_TOKENS,
            int(len(tokens) * _COMMON_PREFIX_REUSE_MIN_FRACTION),
        )
        if best is not None and best_common >= threshold:
            return best, best_common
        return None, 0

    def pending_near_prefix_session(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_token_gap: int | None = None,
        min_matched_tokens: int | None = None,
    ) -> tuple[EngineSession | None, int]:
        """Return a pending-postcommit session with a near-exact prompt prefix."""
        tokens = tuple(int(token) for token in token_ids)
        gap_limit = (
            _near_prefix_max_token_gap()
            if max_token_gap is None
            else max(0, int(max_token_gap))
        )
        min_match = (
            _near_prefix_min_match_tokens()
            if min_matched_tokens is None
            else max(1, int(min_matched_tokens))
        )
        best: EngineSession | None = None
        best_matched = 0
        block_size = _prefix_block_size()
        block_min_match = max(block_size, _block_prefix_min_match_tokens())
        for session in self._sessions_snapshot():
            if not session.has_pending_postcommit():
                continue
            prefix = session.committed_token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            gap = len(prefix) - matched
            required_match = min(min_match, max(1, len(prefix) - gap_limit))
            safe_block = min(
                block_aligned_prefix_len(matched, block_size=block_size),
                len(prefix),
                len(tokens),
            )
            near_match = gap >= 0 and gap <= gap_limit and matched >= required_match
            block_match = (
                _block_prefix_restore_enabled()
                and safe_block >= block_min_match
                and safe_block >= 2
            )
            if not near_match and not block_match:
                continue
            candidate_matched = int(matched if near_match else safe_block)
            if best is None or candidate_matched > best_matched or (
                candidate_matched == best_matched and len(prefix) > len(best.committed_token_ids)
            ):
                best = session
                best_matched = candidate_matched
        return best, best_matched

    def nearest_prefix_session(
        self,
        token_ids: list[int] | tuple[int, ...],
    ) -> tuple[EngineSession | None, int]:
        tokens = tuple(int(token) for token in token_ids)
        best: EngineSession | None = None
        best_len = 0
        for session in self._sessions_snapshot():
            prefix = session.committed_token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            if matched > best_len:
                best = session
                best_len = matched
        return best, best_len

    def _prefix_diagnostic(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        selected: EngineSession | None = None,
        exact: bool = False,
    ) -> dict[str, Any]:
        tokens = tuple(int(token) for token in token_ids)
        session, matched = (selected, len(selected.committed_token_ids)) if selected is not None else self.nearest_prefix_session(tokens)
        if session is None:
            return {
                "prompt_len": len(tokens),
                "exact_prefix_match": False,
                "best_session_id": None,
                "best_prefix_len": 0,
                "matched_prefix_len": 0,
                "divergence_at_token": None,
                "reason": "no_existing_session_prefix",
            }
        divergence_at = None if exact else matched
        boundary = block_aligned_prefix_len(
            matched,
            block_size=_prefix_block_size(),
        )
        return {
            "prompt_len": len(tokens),
            "exact_prefix_match": bool(exact),
            "best_session_id": session.session_id,
            "best_prefix_len": len(session.committed_token_ids),
            "matched_prefix_len": int(matched),
            "nearest_boundary_tokens": int(boundary),
            "new_prefill_tokens": max(0, len(tokens) - int(boundary)),
            "divergence_at_token": divergence_at,
            "best_token_hash": token_hash_short(session.committed_token_ids),
            "prompt_token_hash": token_hash_short(tokens),
            "reason": "exact_prefix_match" if exact else "prefix_divergence_at_token",
        }

    def evict_stale(self) -> int:
        now = time.time()
        with self._lock:
            stale_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_stale(now_s=now) and not session.in_flight
            ]
            for session_id in stale_ids:
                self._sessions.pop(session_id, None)
                self.bank.clear(session_id=session_id)
        return len(stale_ids)

    def clear_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        bank_entries = self.bank.clear(session_id=session_id)
        return {"session_id": session_id, "existed": existed, "bank_entries_cleared": bank_entries}

    def clear_all(self) -> dict[str, Any]:
        with self._lock:
            sessions = len(self._sessions)
            self._sessions.clear()
        bank_entries = self.bank.clear()
        return {"sessions_cleared": sessions, "bank_entries_cleared": bank_entries}

    def quiesce(self, *, reason: str = "admin_cache_clear") -> dict[str, Any]:
        """Abort pending idle maintenance so a clear leaves nothing running.

        Dropping sessions/entries alone leaves two kinds of background work
        alive: per-session idle postcommits (retokenize + snapshot on the
        model thread) and the SSD cold-tier writer's deferred-encode queue,
        whose PendingWrite items pin full cache snapshots in memory until
        encoded. A benchmark-boundary or operator "clear cache" call means
        "clean slate": without this, work queued by earlier traffic keeps
        stealing GPU/memory bandwidth from the rows measured after the clear
        (observed 2026-07-05: 8k cold prefill 790 tok/s on a fresh daemon vs
        ~650 mid-sweep, and 90-107 GB active-memory peaks during
        post-128k-row batch phases from queued 128k snapshot encodes).
        """

        aborted = 0
        for session in self._sessions_snapshot():
            record = getattr(session, "_pending_postcommit", None)
            if record is None:
                continue
            future = getattr(record, "future", None)
            done = getattr(future, "done", None)
            if callable(done):
                try:
                    if done():
                        continue
                except BaseException:
                    pass
            abort = getattr(record, "abort", None)
            if callable(abort):
                try:
                    abort(reason)
                    aborted += 1
                except BaseException:
                    pass
        # Cancel (not flush) queued SSD writes: after a clear they describe
        # discarded state, and deferred-encode backlogs from long-context
        # rows pin snapshots for minutes while starving foreground decode.
        cancelled = 0
        cold_tier = getattr(self.bank, "cold_tier", None)
        cancel_pending = getattr(cold_tier, "cancel_pending", None)
        if callable(cancel_pending):
            try:
                cancelled = int(cancel_pending())
            except BaseException:
                cancelled = 0
        # Bounded wait for the (at most one) in-flight encode to finish so
        # the response reflects a genuinely idle writer.
        flushed = self.flush_cold_tier(timeout_s=10.0)
        return {
            "postcommits_aborted": aborted,
            "ssd_writes_cancelled": cancelled,
            "cold_tier_flushed": bool(flushed),
        }

    def archive_cold_tier(self) -> dict[str, Any]:
        archive = getattr(self.bank, "archive_cold_tier", None)
        if not callable(archive):
            return {"archived": False, "reason": "ssd_cache_archive_unavailable"}
        return archive()

    def flush_cold_tier(self, *, timeout_s: float = 30.0) -> bool:
        flush = getattr(self.bank, "flush_cold_tier", None)
        if not callable(flush):
            return True
        return bool(flush(timeout_s=timeout_s))

    def list_sessions(self) -> dict[str, Any]:
        self.evict_stale()
        sessions = sorted(
            (session.to_admin_dict() for session in self._sessions_snapshot()),
            key=lambda row: row["last_access_s"],
            reverse=True,
        )
        return {
            "sessions": sessions,
            "count": len(sessions),
            "session_bank": self.bank.to_dict(),
            "last_prefix_diagnostic": self.last_prefix_diagnostic,
        }
