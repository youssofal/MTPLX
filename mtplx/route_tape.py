"""Route Tape — per-round route census for MTPLX decode.

Answers, for every speculative round: which verify/attention/GDN/cache/sampler
path actually ran, what fell back and why, and what the round cost. The record
is emitted as ``{"ev": "rt", ...}`` through the Flight Recorder (batched,
rotated, non-blocking) or to a standalone JSONL path for non-server runs.

Design rules (Pulse X-Ray contract, frozen 2026-08-25):

- Off means off: when disabled, the only cost is one ``tape.enabled`` check
  per round. No allocation, clock read, file open, or branch inside kernel
  loops. ``MTPLX_DROP_EVENTS=1`` must NOT silence the tape — it never routes
  through ``append_event``.
- Clocks: ``mach_continuous_time`` converted with the recorded timebase,
  paired with wall time and boot identity on every record.
- The tape records what ran; it never changes what runs.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

# -- frozen route vocabulary (grounded in source, 2026-08-25) ---------------
# generation.py:87-96 (VerifyStrategy), verify dispatch arms 9728-9820,
# copy-block pair 8694/8703.
VERIFY_ROUTES = frozenset({
    "compiled_bank", "graphbank", "eager_capture",
    "a3b_m3_rebased", "a3b_m3", "a3b_m2_rebased", "a3b_m2",
    "compiled_bank_tp", "graphbank_plain", "eager_plain",
    "ccopy_bank", "ccopy_block", "inkling_block_verify", "ar", "not_run", "unknown",
})


def is_verify_route(value: object) -> bool:
    """Whether *value* is a valid concrete verify dispatch receipt.

    Bank fallbacks carry their reason in the route itself because the
    aggregate fallback delta fires only when a state changes.  This keeps
    every later eager-tail round independently truthful.
    """
    return isinstance(value, str) and (
        value in VERIFY_ROUTES
        or (value.startswith("bank_eager:") and len(value) > len("bank_eager:"))
    )

# gdn_capture.resolve_gdn_capture_backend (gdn_capture.py:1455).
GDN_CAPTURE_ROUTES = frozenset({
    "stock", "linear_gdn", "linear_gdn_from_conv", "linear_gdn_from_conv_stream",
    "linear_gdn_from_conv_stream_skip0", "linear_gdn_from_conv_tape",
    "linear_gdn_from_conv_inline_g", "linear_gdn_final",
})

# Commit lanes (generation.py:10668-10844).
COMMIT_ROUTES = frozenset({
    "capture_commit", "trim_commit", "rollback_reforward", "copy_block",
    "verify_retained", "rebase_deferred", "primary_only", "none",
})

# graphbank._fallback_reason (graphbank.py:1909-1989) + legacy bank (193-211).
BANK_FALLBACK_REASONS = frozenset({
    "permanent_eager", "hidden_not_requested", "invalid_input_shape",
    "batch_size", "invalid_length", "length_outside_bank",
    "unsupported_capture_backend", "owned_attn_kv_env",
    "owned_recurrent_state_env", "no_cache", "growth_budget_exhausted",
    "post_restore_warmup", "quantized_paged_kv", "python_cache_offsets",
    "unsupported_container", "gdn_meta_unavailable", "capacity_overflow",
    "context_above_threshold", "paged_kernel_ineligible", "empty_state_leaf",
    "length_outside_graphbank",
})

# -- clock (p1c) --------------------------------------------------------------

_libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


class _TimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_TB = _TimebaseInfo()
_libsystem.mach_timebase_info(ctypes.byref(_TB))
_libsystem.mach_continuous_time.restype = ctypes.c_uint64
TIMEBASE_NUMER = _TB.numer
TIMEBASE_DENOM = _TB.denom


def mono_ns() -> int:
    return _libsystem.mach_continuous_time() * TIMEBASE_NUMER // TIMEBASE_DENOM


def wall_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


_BOOT_ID: str | None = None


def boot_id() -> str:
    global _BOOT_ID
    if _BOOT_ID is None:
        buf = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(len(buf))
        _libsystem.sysctlbyname(b"kern.bootsessionuuid", buf, ctypes.byref(size), None, 0)
        _BOOT_ID = buf.value.decode()
    return _BOOT_ID


# -- sink registry (mirrors set_live_decode_sink, generation.py:1126) ---------

_ROUTE_TAPE_SINK: Callable[[dict[str, Any]], None] | None = None


def set_route_tape_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    global _ROUTE_TAPE_SINK
    _ROUTE_TAPE_SINK = sink


def route_tape_requested() -> bool:
    return os.environ.get("MTPLX_ROUTE_TAPE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class RouteTape:
    """Per-request per-round emitter. ``enabled=False`` is the zero-cost path."""

    __slots__ = ("enabled", "request_id", "_seq", "_path", "_sink", "_dropped")

    def __init__(self, request_id: str | None = None) -> None:
        self._seq = 0
        self._dropped = 0
        requested = route_tape_requested()
        self._sink = _ROUTE_TAPE_SINK if requested else None
        path = (
            os.environ.get("MTPLX_ROUTE_TAPE_JSONL")
            if requested and self._sink is None
            else None
        )
        self._path = Path(path).expanduser() if path else None
        self.enabled = requested and (self._sink is not None or self._path is not None)
        # Standalone callers can omit an ID. The server sink supplies its canonical ID.
        self.request_id = request_id if request_id is not None else (
            f"rt-{os.urandom(8).hex()}" if self.enabled else ""
        )

    def emit(self, kind: str, name: str, round_id: int | None, attrs: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._seq += 1
        record = {
            "ev": "rt",
            "schema_version": 1,
            "seq": self._seq,
            "boot_id": boot_id(),
            "wall_ns": wall_ns(),
            "mono_ns": mono_ns(),
            "rid": self.request_id,
            "round": round_id,
            "kind": kind,
            "name": name,
            "attrs": attrs,
            # Cumulative loss is carried by the next successful row. A sink
            # failure therefore cannot disappear behind a contiguous-looking
            # stream even though telemetry must never take down decode.
            "dropped_before": self._dropped,
        }
        try:
            if self._sink is not None:
                self._sink(record)
            else:
                with open(self._path, "a", encoding="utf-8") as f:  # type: ignore[arg-type]
                    f.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        except Exception:
            # Telemetry never takes down decode; loss is accounted, not hidden.
            self._dropped += 1

    @property
    def dropped(self) -> int:
        return self._dropped


def counter_deltas(prev: dict[str, int], cur: dict[str, int]) -> dict[str, int]:
    """Per-round delta of a monotonic counter dict (nax/gdn/attention bails)."""
    out: dict[str, int] = {}
    for key, value in cur.items():
        delta = value - prev.get(key, 0)
        if delta:
            out[key] = delta
    return out
