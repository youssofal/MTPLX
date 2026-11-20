"""macOS thermal pressure, sampled in the background and read from a cache.

Thermal pressure is the kernel's own verdict on how hard the machine may run:
nominal, moderate, heavy, trapping or sleeping (the levels behind
``ProcessInfo.thermalState``). It is not the fan state. Fans can run at their
maximum under nominal pressure, and a Mac throttles under heavy pressure
whatever its fans are doing. On 2026-09-29 heavy pressure covered 74 percent
of the decode seconds of a Pi session and no receipt recorded it.

A daemon thread reads the level every ``interval_s`` (one notify(3) state read,
about ten microseconds) and records each change. Everything else only reads
the cached value: flight samples (at most once a second, on the event loop),
receipts (once per request) and ``/health``. Nothing on the decode path calls
the operating system, and a change is seen at most ``interval_s`` late.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
import time
from collections import deque
from typing import Any, Callable

# kOSThermalPressureLevel* on macOS (libkern/OSThermalNotification.h),
# in increasing severity.
THERMAL_PRESSURE_LEVELS = ("nominal", "moderate", "heavy", "trapping", "sleeping")
UNAVAILABLE = "unavailable"
DEFAULT_INTERVAL_S = 2.0
# Heavy pressure and worse are where macOS cuts the GPU clock (on an M5 Max
# from about 1,450 to 700 MHz after a few minutes of decode), so a speed read
# there is a throttled one. The app names the same three levels.
THROTTLING_LEVELS = frozenset(THERMAL_PRESSURE_LEVELS[2:])


def is_throttling(level: str | None) -> bool:
    return level in THROTTLING_LEVELS


def level_name(level: int) -> str:
    level = int(level)
    if 0 <= level < len(THERMAL_PRESSURE_LEVELS):
        return THERMAL_PRESSURE_LEVELS[level]
    return f"level_{level}"


def notify_thermal_pressure_reader() -> Callable[[], int] | None:
    """A function that reads the current pressure level, or None where the
    notification is unavailable (not macOS, or the lookup failed)."""

    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib")
        name = ctypes.c_char_p.in_dll(lib, "kOSThermalNotificationPressureLevelName").value
        register = lib.notify_register_check
        register.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        register.restype = ctypes.c_uint32
        get_state = lib.notify_get_state
        get_state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        get_state.restype = ctypes.c_uint32
    except (OSError, AttributeError, ValueError):
        return None
    if not name:
        return None
    token = ctypes.c_int(0)
    if register(name, ctypes.byref(token)) != 0:
        return None

    def read() -> int:
        state = ctypes.c_uint64(0)
        status = get_state(token, ctypes.byref(state))
        if status != 0:
            raise OSError(f"notify_get_state returned {status}")
        return int(state.value)

    return read


class ThermalPressureSampler:
    """Caches the thermal pressure level and the changes it went through.

    ``reader`` returns the current level (0 nominal ... 4 sleeping) or raises;
    None means the source is unavailable. ``current()`` and ``worst_since()``
    never call the reader.
    """

    def __init__(
        self,
        reader: Callable[[], int] | None,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        clock: Callable[[], float] = time.perf_counter,
        history: int = 512,
    ) -> None:
        self._reader = reader
        self.interval_s = float(interval_s)
        self._clock = clock
        self._lock = threading.Lock()
        # (clock time the level was first seen, level)
        self._changes: deque[tuple[float, int]] = deque(maxlen=int(history))
        self._level: int | None = None
        self._sampled_at: float | None = None
        self.read_errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._reader is not None

    def sample_once(self) -> str:
        if self._reader is None:
            return UNAVAILABLE
        try:
            level = int(self._reader())
        except Exception as exc:  # the source failed this read; keep the last level
            with self._lock:
                self.read_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            return self.current()
        now = self._clock()
        with self._lock:
            if level != self._level:
                self._changes.append((now, level))
                self._level = level
            self._sampled_at = now
        return level_name(level)

    def current(self) -> str:
        # One attribute read, no lock: the sampler thread replaces the int
        # whole, so a reader sees the old level or the new one.
        level = self._level
        return UNAVAILABLE if level is None else level_name(level)

    def worst_since(self, since_s: float | None, now_s: float | None = None) -> str:
        """The most severe level in effect between ``since_s`` and now.

        The level in effect at ``since_s`` is the last one seen at or before
        it; later changes up to ``now_s`` are included. Changes are stamped
        when the sampler saw them, up to ``interval_s`` after they happened.
        """

        with self._lock:
            changes = list(self._changes)
        if not changes:
            return UNAVAILABLE
        if since_s is None:
            return level_name(changes[-1][1])
        since_s = float(since_s)
        end = float(now_s) if now_s is not None else float("inf")
        worst: int | None = None
        for when, level in changes:
            if when <= since_s:
                worst = level  # the level in effect when the window opened
                continue
            if when > end:
                break
            worst = level if worst is None else max(worst, level)
        return UNAVAILABLE if worst is None else level_name(worst)

    def start(self) -> bool:
        if self._reader is None:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="mtplx-thermal-pressure", daemon=True
            )
            self._thread.start()
        return True

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            sampled_at = self._sampled_at
            errors = self.read_errors
            last_error = self.last_error
        return {
            "level": self.current(),
            "available": self.available,
            "sampled_age_s": (
                None if sampled_at is None else max(0.0, self._clock() - sampled_at)
            ),
            "interval_s": self.interval_s,
            "read_errors": errors,
            "last_error": last_error,
        }


_process_sampler: ThermalPressureSampler | None = None
_process_lock = threading.Lock()


def process_sampler() -> ThermalPressureSampler:
    """The server process's sampler (created on first use, not started)."""

    global _process_sampler
    with _process_lock:
        if _process_sampler is None:
            _process_sampler = ThermalPressureSampler(notify_thermal_pressure_reader())
        return _process_sampler


def start_process_sampler() -> ThermalPressureSampler:
    sampler = process_sampler()
    sampler.sample_once()
    sampler.start()
    return sampler


def current_level() -> str:
    """The cached level, or "unavailable" when nothing sampled it."""

    sampler = _process_sampler
    return UNAVAILABLE if sampler is None else sampler.current()


def _seconds(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def prefill_level(request_started_s: Any, ttft_s: Any) -> str:
    """The worst level while a request's prompt was read: from its arrival
    (``request_received_monotonic_s``, perf_counter) to its first token."""

    sampler = _process_sampler
    since, ttft = _seconds(request_started_s), _seconds(ttft_s)
    if sampler is None or since is None or ttft is None or ttft < 0:
        return UNAVAILABLE
    return sampler.worst_since(since, since + ttft)


def receipt_fields(request_started_s: Any, ttft_s: Any = None) -> dict[str, str]:
    """``thermal_pressure`` at completion, ``thermal_pressure_max`` since the
    request arrived (``request_received_monotonic_s``, perf_counter), and
    ``thermal_pressure_prefill_max`` while its prompt was read, when the
    request reached a first token."""

    sampler = _process_sampler
    if sampler is None:
        return {"thermal_pressure": UNAVAILABLE, "thermal_pressure_max": UNAVAILABLE}
    fields = {
        "thermal_pressure": sampler.current(),
        "thermal_pressure_max": sampler.worst_since(_seconds(request_started_s)),
    }
    if _seconds(ttft_s) is not None:
        fields["thermal_pressure_prefill_max"] = prefill_level(request_started_s, ttft_s)
    return fields
