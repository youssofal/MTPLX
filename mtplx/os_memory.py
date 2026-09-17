"""Real OS-reported process memory, independent of MLX's own bookkeeping.

Every existing memory guard in ``server/openai.py`` (the prefill admission
shed, the background pressure loop, the qwen4 fixed-M4 lane gate) compares
against ``mx.get_active_memory() + mx.get_cache_memory()`` -- MLX's own
account of what it allocated through Metal. That account can drift below
what the kernel actually holds resident for this process: touched-then-kept
weight pages, non-Metal Python/NumPy heap, thread stacks, the session bank's
host-side buffers. Two independently reported machines hit exactly this gap:
a request (or an already-running generation) was judged safe by the
allocator's numbers while the OS was already out of free pages, and both
ended in a kernel panic (watchdog timeout) rather than the intended
structured 507 or sustained-pressure abort.

``phys_footprint_bytes()`` reads the same counter macOS's own jetsam /
memory-pressure subsystem uses to make that exact call: ``RUSAGE_INFO_V4``'s
``ri_phys_footprint`` via ``proc_pid_rusage`` (stable since macOS 10.9 --
no new dependency, no version gating needed for MTPLX's Apple-Silicon-only,
macOS 14+ target). It is meant to be combined with the allocator's own
numbers as a floor, ``max(allocator_bytes, phys_footprint_bytes())``, never
as a replacement -- the allocator numbers still drive every token-count
projection this module cannot make on its own.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

_RUSAGE_INFO_V4 = 4


class _RUsageInfoV4(ctypes.Structure):
    """Byte-for-byte ``struct rusage_info_v4`` (``<sys/resource.h>``).

    RUSAGE_INFO_V4 is a frozen ABI -- Apple adds new ``rusage_info_vN``
    structs rather than changing an existing one, specifically so old
    callers of a given flavor keep working forever -- so this layout does
    not need to track future SDKs. It does need to be complete: an earlier
    draft of this struct was missing the eight trailing fields below (232
    bytes instead of the real 296), and ``proc_pid_rusage`` writes a full
    296-byte reply regardless of how large the caller's buffer actually is.
    That silently overran the ctypes-allocated buffer by 64 bytes on every
    call -- no exception, just corrupted heap metadata that crashed minutes
    later at an unrelated allocation. ``_load_libproc`` below re-checks
    ``ctypes.sizeof(_RUsageInfoV4) == 296`` before this struct is ever used,
    so any future transcription slip disables the probe instead of
    repeating that failure mode.
    """

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4_SIZE = 296


def _load_libproc():
    try:
        if ctypes.sizeof(_RUsageInfoV4) != _RUSAGE_INFO_V4_SIZE:
            # See the struct docstring: a size mismatch means this layout
            # no longer matches the kernel's RUSAGE_INFO_V4 reply, and using
            # it would overrun the buffer on every call. Disable the probe
            # instead of risking that again.
            return None
        path = ctypes.util.find_library("proc") or "libproc.dylib"
        lib = ctypes.CDLL(path)
        lib.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.proc_pid_rusage.restype = ctypes.c_int
        return lib
    except Exception:  # noqa: BLE001 - a missing probe must degrade, never raise
        return None


_LIBPROC = _load_libproc()


def phys_footprint_bytes(pid: int | None = None) -> int | None:
    """This process's (or ``pid``'s) real physical footprint, in bytes.

    Returns ``None`` on any failure -- non-Darwin platform, missing
    ``libproc``, or a non-zero ``proc_pid_rusage`` return -- so every caller
    can fall back to allocator-only accounting exactly as before. Never
    raises: a memory guard must not cost a request over its own probe.
    """
    if _LIBPROC is None:
        return None
    try:
        info = _RUsageInfoV4()
        ptr = ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_void_p))
        ret = _LIBPROC.proc_pid_rusage(
            int(pid if pid is not None else os.getpid()), _RUSAGE_INFO_V4, ptr
        )
        if ret != 0:
            return None
        return int(info.ri_phys_footprint)
    except Exception:  # noqa: BLE001 - a memory guard must never cost a request
        return None
