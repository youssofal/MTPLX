"""The desktop's share of memory: the guard that looks past the engine's own limit.

Receipt (2026-09-19): a 129,050-token Pi compaction prompt on a fresh daemon,
128 GB Mac, a game editor and two browsers open. The engine stayed under its
Metal limit the whole time, macOS reported "normal" pressure, and the Mac froze
about a minute into the prefill. These tests pin the three places the kernel's
available-memory figure now acts: the reading itself, prefill admission, and the
guard loop's level.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from ctypes import sizeof as ctypes_sizeof

GIB = 1024**3
RAM = 128 * GIB
LIMIT = 96 * GIB
KV_PER_TOKEN = 24576
AUX_PER_TOKEN = 7872
PER_TOKEN = KV_PER_TOKEN + AUX_PER_TOKEN


@pytest.fixture(autouse=True)
def _served_prefill_is_chunked(monkeypatch):
    # The admission prices the rows each prefill forward runs. The served
    # profiles (sustained, turbo) prefill in chunks; without the flag a
    # prompt is one forward and is priced as one.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")


def _reading(available_gib: float, total: int = RAM) -> sm.SystemMemory:
    available = int(available_gib * GIB)
    return sm.SystemMemory(
        available_bytes=available,
        total_bytes=total,
        level_percent=int(available * 100 // total),
    )


def _install(monkeypatch, available_gib: float | None, total: int = RAM) -> None:
    monkeypatch.setattr(
        sm,
        "_reader",
        (lambda: None)
        if available_gib is None
        else (lambda: _reading(available_gib, total)),
    )


class TestReading:
    def test_the_real_kernel_answers_on_this_mac(self):
        reading = sm._read_kernel()
        assert reading is not None
        assert reading.total_bytes >= 8 * GIB
        assert 0 <= reading.available_bytes <= reading.total_bytes
        # The supply is what the kernel hands out without compressing:
        # free (speculative included) and purgeable pages plus file cache.
        assert reading.free_bytes is not None
        assert reading.file_backed_bytes is not None
        assert reading.available_bytes == min(
            reading.total_bytes, reading.free_bytes + reading.file_backed_bytes
        )
        assert reading.wired_bytes > 0
        assert reading.compressor_bytes is not None
        assert reading.swap_used_bytes is not None

    def test_the_level_is_not_free_memory(self):
        """kern.memorystatus_level is (total - wired - compressor) / total.

        It counts every app's anonymous pages as available, which is why the
        guard no longer reads its supply from it (the old test asserted that
        available == total x level / 100, the defect this replaces).
        """

        reading = sm._read_kernel()
        assert reading is not None
        unwired = reading.total_bytes - reading.wired_bytes - int(reading.compressor_bytes)
        assert abs(reading.level_percent - unwired * 100 / reading.total_bytes) <= 3

    def test_kill_switch_reads_unknown(self, monkeypatch):
        _install(monkeypatch, 40)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_GUARD", "0")
        assert sm.read_system_memory() is None

    def test_the_struct_matches_apples_rev2_layout(self):
        """``struct vm_statistics64`` (mach/vm_statistics.h, rev2): 160 bytes,
        aligned to 8, and the offsets the supply reads."""

        layout = sm._VMStatistics64
        assert ctypes_sizeof(layout) == 160
        offsets = {
            "free_count": 0,
            "active_count": 4,
            "inactive_count": 8,
            "wire_count": 12,
            "zero_fill_count": 16,
            "purges": 80,
            "purgeable_count": 88,
            "speculative_count": 92,
            "decompressions": 96,
            "swapouts": 120,
            "compressor_page_count": 128,
            "throttled_count": 132,
            "external_page_count": 136,
            "internal_page_count": 140,
            "total_uncompressed_pages_in_compressor": 144,
            "swapped_count": 152,
        }
        for name, offset in offsets.items():
            assert getattr(layout, name).offset == offset, name
        # host_statistics64 is told the buffer size in 32-bit words.
        assert ctypes_sizeof(layout) // 4 == 40

    def test_speculative_pages_count_once(self):
        """The 2026-09-27 review: free_count and external_page_count both
        include the speculative pages, so adding them counted those twice.
        4 GiB free (3 GiB of it speculative) and 8 GiB of file pages is
        9 GiB the kernel can hand out, not 12."""

        page = 16384
        stats = sm._VMStatistics64()
        stats.free_count = 4 * GIB // page
        stats.speculative_count = 3 * GIB // page
        stats.external_page_count = 8 * GIB // page
        stats.purgeable_count = 0
        free, file_backed = sm._supply_from_statistics(stats, page)
        assert free == 4 * GIB
        assert file_backed == 5 * GIB
        # At 88 GiB wired the 11 GiB shed floor is short by 2 GiB: WARNING,
        # where the double count read normal.
        reading = sm.SystemMemory(
            available_bytes=free + file_backed,
            total_bytes=RAM,
            level_percent=30,
            free_bytes=free,
            file_backed_bytes=file_backed,
            wired_bytes=88 * GIB,
        )
        assert sm.system_pressure_level(reading) == 2
        # An 8 GiB request is held to the 12 GiB shed floor the Mac will
        # have once those 8 GiB are wired.
        assert sm.admission_shortfall_bytes(
            reading, growth_bytes=8 * GIB, floor="shed"
        ) == 8 * GIB + 12 * GIB - 9 * GIB

    def test_purgeable_pages_are_free(self):
        page = 16384
        stats = sm._VMStatistics64()
        stats.free_count = GIB // page
        stats.purgeable_count = 2 * GIB // page
        free, file_backed = sm._supply_from_statistics(stats, page)
        assert (free, file_backed) == (3 * GIB, 0)

    def test_a_reader_that_raises_reads_unknown(self, monkeypatch):
        def boom():
            raise OSError("sysctl refused")

        monkeypatch.setattr(sm, "_reader", boom)
        assert sm.read_system_memory() is None

    def test_rehearsal_replaces_the_available_figure_only(self, monkeypatch):
        _install(monkeypatch, 40)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", "2G")
        reading = sm.read_system_memory()
        assert reading is not None
        assert reading.available_bytes == 2 * GIB
        assert reading.total_bytes == RAM

    @pytest.mark.parametrize(
        "raw, expected",
        [("1073741824", GIB), ("2G", 2 * GIB), ("2GiB", 2 * GIB), ("512M", 512 * 1024**2),
         ("1.5g", int(1.5 * GIB)), ("", None), ("lots", None)],
    )
    def test_byte_sizes_parse(self, raw, expected):
        assert sm._parse_bytes(raw) == expected


class TestFloorsAndLevel:
    def test_floors_scale_with_the_machine_and_never_vanish(self):
        shed, abort = sm.system_memory_floors(128 * GIB)
        assert abort == int(128 * GIB * 0.025)
        assert shed == 2 * abort
        shed_small, abort_small = sm.system_memory_floors(16 * GIB)
        assert abort_small == 1 * GIB
        assert shed_small == 2 * GIB

    def test_floor_overrides(self, monkeypatch):
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES", "4G")
        assert sm.system_memory_floors(RAM) == (8 * GIB, 4 * GIB)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES", "5G")
        assert sm.system_memory_floors(RAM) == (5 * GIB, 4 * GIB)

    def test_level_ladder(self):
        assert sm.system_pressure_level(None) == 1
        assert sm.system_pressure_level(_reading(30)) == 1
        assert sm.system_pressure_level(_reading(6.3)) == 2
        assert sm.system_pressure_level(_reading(3.1)) == 4

    def test_unknown_never_reports_a_shortfall(self):
        for floor in ("shed", "abort"):
            assert sm.admission_shortfall_bytes(None, growth_bytes=50 * GIB, floor=floor) == 0

    def test_the_request_is_charged_once_against_each_floor(self):
        """The shed line says when the engine gives back its own memory
        first, the abort line when the request is refused. The request is
        counted once: the old rule also held a margin as large as the
        request, and refused an 18K turn that would have left 9.4 GB
        free over a 6.0 GB abort floor (2026-09-27 validation)."""

        shed, abort = sm.system_memory_floors(RAM)
        reading = _reading(10)
        assert sm.admission_shortfall_bytes(reading, growth_bytes=8 * GIB, floor="shed") == (
            8 * GIB + shed - 10 * GIB
        )
        assert sm.admission_shortfall_bytes(reading, growth_bytes=8 * GIB, floor="abort") == (
            8 * GIB + abort - 10 * GIB
        )
        assert sm.admission_shortfall_bytes(reading, growth_bytes=5 * GIB, floor="abort") == 0
        with pytest.raises(ValueError):
            sm.admission_shortfall_bytes(reading, growth_bytes=GIB, floor="margin")


def _machine(
    *,
    free_gib: float,
    file_gib: float = 0.0,
    wired_gib: float = 0.0,
    compressor_gib: float = 0.0,
    swap_gib: float = 0.0,
    at_s: float = 0.0,
    total: int = RAM,
) -> sm.SystemMemory:
    free = int(free_gib * GIB)
    file_backed = int(file_gib * GIB)
    wired = int(wired_gib * GIB)
    compressor = int(compressor_gib * GIB)
    return sm.SystemMemory(
        available_bytes=free + file_backed,
        total_bytes=total,
        level_percent=int((total - wired - compressor) * 100 // total),
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=wired,
        compressor_bytes=compressor,
        swap_used_bytes=int(swap_gib * GIB),
        monotonic_s=at_s,
    )


class TestWiredFloorsAndDeathSignature:
    """The field report (M5 Max 128 GB, Flash-Next, 2026-09-26): compactions
    that survived sat at 12-17 percent memorystatus_level with free pages at
    0.1-0.5 GiB and the compressor growing 10 GiB; the ones that froze had
    the same shape. 83 GiB wired did not recover from 3 GiB free."""

    def test_the_floor_scales_with_what_is_wired(self):
        shed, abort = sm.system_memory_floors(128 * GIB, wired_bytes=88 * GIB)
        assert abort == 88 * GIB // 16  # 5.5 GiB
        assert shed == 2 * abort  # 11 GiB
        # 83 GiB wired: the abort floor is above the 3 GiB that did not recover.
        assert sm.system_memory_floors(128 * GIB, wired_bytes=83 * GIB)[1] > 3 * GIB
        # 48 GB Mac with the 27B wired: barely above the old 2.5 percent line.
        _shed48, abort48 = sm.system_memory_floors(48 * GIB, wired_bytes=21 * GIB)
        assert abort48 == 21 * GIB // 16
        # 16 GB Mac: the 1 GiB minimum still rules, unchanged.
        assert sm.system_memory_floors(16 * GIB, wired_bytes=10 * GIB) == (2 * GIB, GIB)
        # No wired reading: the shipped 2.5 percent rule, unchanged.
        assert sm.system_memory_floors(128 * GIB) == (
            2 * int(128 * GIB * 0.025),
            int(128 * GIB * 0.025),
        )

    def test_seventeen_percent_level_with_no_free_pages_is_critical(self):
        # The surviving compaction's trough: 92.2 GiB wired, 14.7 GiB
        # compressed (level 16.5 percent, reported 17), free pages 0.3 GiB.
        reading = _machine(
            free_gib=0.3, file_gib=2.0, wired_gib=92.2, compressor_gib=14.7
        )
        assert 15 <= reading.level_percent <= 17
        assert sm.system_pressure_level(reading) == 4

    def test_a_healthy_flash_next_desktop_stays_normal(self):
        # Julian's ordinary moment plus the n-gram table's file cache:
        # 88 GiB wired, 3.5 GiB free, 12 GiB of file-backed pages.
        reading = _machine(free_gib=3.5, file_gib=12.0, wired_gib=88.0, compressor_gib=3.7)
        assert sm.system_pressure_level(reading) == 1

    # Free pages under the kernel's own free-page target (4,000 x 16 KiB =
    # 62.5 MiB, system_memory.starved_free_bytes), where the recorded deaths
    # sat (09-03 at 0.0 GB free, the 09-23 panic at 878 pages). With 20 GiB
    # of file cache, 0.2 GiB free is healthy on macOS 27: the 2026-09-29
    # agent replay ran 0.1 to 0.6 GB free with the compressor stepping 0.3 to
    # 0.5 GB/s and swap flat (b15ddcfb), so the mechanism tests below read a
    # starved Mac.
    STARVED_FREE_GIB = 16 / 1024

    def test_death_signature_between_two_readings_is_critical(self):
        before = _machine(
            free_gib=self.STARVED_FREE_GIB, file_gib=20, wired_gib=88,
            compressor_gib=10, at_s=100.0,
        )
        after = _machine(
            free_gib=self.STARVED_FREE_GIB, file_gib=20, wired_gib=88,
            compressor_gib=11.2, at_s=102.0,
        )
        # Plenty of file cache: the supply alone reads normal ...
        assert sm.system_pressure_level(after) == 1
        # ... but free pages under the floor while the compressor grew
        # 1.2 GiB in 2 s is the crash receipts' death signature.
        assert sm.memory_thrashing(after, before)
        assert sm.system_pressure_level(after, previous=before) == 4

    def test_swap_growth_under_the_floor_is_the_same_signature(self):
        before = _machine(free_gib=0.1, file_gib=20, wired_gib=88, swap_gib=1.0, at_s=0.0)
        after = _machine(free_gib=0.1, file_gib=20, wired_gib=88, swap_gib=1.5, at_s=2.0)
        assert sm.memory_thrashing(after, before)

    def test_swap_growth_under_the_minimum_is_not_the_signature(self):
        # 128 MiB in 2 s is 64 MiB/s, the rate line, but under the 256 MiB
        # a reading pair must show (the review found the swap branch
        # skipped that minimum).
        before = _machine(free_gib=0.1, file_gib=20, wired_gib=88, swap_gib=1.0, at_s=0.0)
        after = _machine(
            free_gib=0.1, file_gib=20, wired_gib=88, swap_gib=1.0 + 0.125, at_s=2.0
        )
        assert not sm.memory_thrashing(after, before)

    def test_compression_with_free_pages_to_spare_is_not_the_signature(self):
        before = _machine(free_gib=9.0, file_gib=5, wired_gib=88, compressor_gib=4, at_s=0.0)
        after = _machine(free_gib=9.0, file_gib=5, wired_gib=88, compressor_gib=6, at_s=2.0)
        assert not sm.memory_thrashing(after, before)

    def test_growth_is_measured_over_the_window_not_one_step(self):
        """320 MiB/s arriving in 80 MiB steps a quarter second apart: no two
        adjacent readings grew 256 MiB, and the reading a second back did."""

        window = sm.ReadingWindow()
        tripped_at = None
        for i in range(8):
            reading = _machine(
                free_gib=self.STARVED_FREE_GIB,
                file_gib=20,
                wired_gib=88,
                compressor_gib=10 + i * 80 / 1024,
                at_s=i * 0.25,
            )
            earlier = window.readings()
            if earlier:
                assert not sm.memory_thrashing(reading, earlier[-1])
            if tripped_at is None and sm.memory_thrashing(reading, earlier):
                tripped_at = i
            window.add(reading)
        assert tripped_at == 4
        assert sm.system_pressure_level(reading, previous=window.readings()[:-1]) == 4

    def test_the_window_keeps_the_newest_reading_before_it(self):
        """The guard loop reads every 10 s: the reading one tick back is the
        only earlier one, and it stays the base."""

        window = sm.ReadingWindow(window_s=10.0)
        first = _machine(free_gib=0.2, file_gib=20, wired_gib=88, at_s=0.0)
        second = _machine(free_gib=0.2, file_gib=20, wired_gib=88, at_s=10.0)
        third = _machine(free_gib=0.2, file_gib=20, wired_gib=88, at_s=20.0)
        window.add(first)
        window.add(second)
        assert window.readings() == [first, second]
        window.add(third)
        assert window.readings() == [second, third]
        # Readings every quarter second, a one-second window: the last
        # second's readings plus the one exactly a second back.
        many = sm.ReadingWindow(window_s=1.0)
        for i in range(20):
            many.add(_machine(free_gib=0.2, file_gib=20, wired_gib=88, at_s=i * 0.25))
        assert [r.monotonic_s for r in many.readings()] == [3.75, 4.0, 4.25, 4.5, 4.75]

    def test_small_or_slow_growth_is_not_the_signature(self):
        before = _machine(free_gib=0.2, file_gib=20, wired_gib=88, compressor_gib=10, at_s=0.0)
        small = _machine(free_gib=0.2, file_gib=20, wired_gib=88, compressor_gib=10.2, at_s=0.5)
        slow = _machine(free_gib=0.2, file_gib=20, wired_gib=88, compressor_gib=11, at_s=30.0)
        assert not sm.memory_thrashing(small, before)
        assert not sm.memory_thrashing(slow, before)
        assert not sm.memory_thrashing(small, None)

    def test_a_512gb_seat_keeps_floors_proportional_to_what_it_wires(self):
        # Intended, not yet measured: the GPU's transient wiring during a
        # command buffer grew 88 -> 92.2 GiB (4.8 percent) on the report's
        # Mac, and it scales with what is wired. 400 GiB wired: 25 / 50 GiB.
        assert sm.system_memory_floors(512 * GIB, wired_bytes=400 * GIB) == (
            50 * GIB,
            25 * GIB,
        )

    def test_a_request_between_the_floors_is_short_of_the_shed_line_only(self):
        reading = _machine(free_gib=4, file_gib=2, wired_gib=88)  # 6 GiB supply
        shed, abort = sm.admission_floors(reading, GIB // 4)
        assert (shed, abort) == (2 * (88.25 * GIB // 16), 88.25 * GIB // 16)
        # 0.25 GiB of growth leaves 5.75 GiB: over the 5.52 GiB abort floor,
        # under the 11.03 GiB shed floor.
        assert sm.admission_shortfall_bytes(reading, growth_bytes=GIB // 4, floor="abort") == 0
        assert sm.admission_shortfall_bytes(
            reading, growth_bytes=GIB // 4, floor="shed"
        ) == GIB // 4 + shed - 6 * GIB
        # 1 GiB leaves 5 GiB: 0.56 GiB under the 5.56 GiB abort floor (the
        # old margin rule said 1.5).
        _shed, abort = sm.admission_floors(reading, GIB)
        assert sm.admission_shortfall_bytes(
            reading, growth_bytes=GIB, floor="abort"
        ) == GIB + abort - 6 * GIB == int(0.5625 * GIB)

    def test_the_admission_holds_the_floor_the_prefill_will_read(self):
        """What the engine allocates is wired, and the floors grow with what
        is wired: after an 8 GiB prefill at 88 GiB wired the per-chunk check
        reads a 6 GiB abort floor, not 5.5. A request that would leave 5.75
        GiB was admitted against 5.5 and tripped late in its own prefill; it
        is now short by 0.25 GiB before it starts."""

        reading = _machine(free_gib=4, file_gib=9.75, wired_gib=88)  # 13.75 GiB
        assert sm.reading_floors(reading)[1] == int(5.5 * GIB)
        assert sm.admission_floors(reading, 8 * GIB) == (12 * GIB, 6 * GIB)
        assert sm.admission_shortfall_bytes(
            reading, growth_bytes=8 * GIB, floor="abort"
        ) == GIB // 4
        # A reading with no wired figure keeps its RAM-share floors.
        unwired = _machine(free_gib=4, file_gib=9.75)
        assert sm.admission_floors(unwired, 8 * GIB) == sm.reading_floors(unwired)

    def test_admission_counts_the_wired_floor(self):
        reading = _machine(free_gib=4, file_gib=10, wired_gib=88)
        shed, _abort = sm.admission_floors(reading, 8 * GIB)
        assert shed == 2 * (96 * GIB // 16)
        assert sm.admission_shortfall_bytes(
            reading, growth_bytes=8 * GIB, floor="shed"
        ) == 8 * GIB + shed - 14 * GIB

    def test_rehearsal_caps_the_free_pages_too(self, monkeypatch):
        monkeypatch.setattr(sm, "_reader", lambda: _machine(free_gib=40, file_gib=20, wired_gib=10))
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", "2G")
        reading = sm.read_system_memory()
        assert reading.available_bytes == 2 * GIB
        assert reading.free_bytes == 2 * GIB
        assert reading.wired_bytes == 10 * GIB


class _Bank:
    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        self.shrink_calls: list[tuple[int, str, bool]] = []
        self.cleared_sessions: list[str | None] = []

    @property
    def total_nbytes(self):
        return self.nbytes

    def longest_prefix(self, token_ids):
        return None

    def longest_shared_prefix_tokens(self, token_ids, *, session_id=None):
        return 0

    def clear(self, *, session_id=None):
        self.cleared_sessions.append(session_id)
        return 0

    def touch_sessions(self, session_ids):
        pass

    def shrink_to_bytes(
        self,
        target_bytes,
        *,
        reason="",
        protect_active=False,
        protect_keys=None,
        protect_session_ids=None,
        cancel_queued_persistence=True,
        protect_unwritten=False,
        receipt_out=None,
    ):
        self.shrink_calls.append((int(target_bytes), reason, protect_active))
        evicted = 1 if self.nbytes > int(target_bytes) else 0
        self.nbytes = min(self.nbytes, int(target_bytes))
        return evicted


def _state(allow_swap: bool = False):
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": LIMIT},
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=KV_PER_TOKEN,
            aux_bytes_per_token=AUX_PER_TOKEN,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
        allow_swap=allow_swap,
    )


def _pin_engine(monkeypatch, *, active_gib: float, cache_gib: float) -> None:
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(active_gib * GIB),
            "cache_memory_bytes": int(cache_gib * GIB),
        },
    )
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


PROMPT = list(range(129_050))


class TestAdmission:
    """The engine is healthy against its own limit in every case here."""

    def test_a_quiet_desktop_admits_the_saturday_prompt_untouched(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 40)
        bank = _Bank(0)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=bank, session_id="pi"
        ) is None
        assert bank.shrink_calls == []

    def test_an_unreadable_machine_changes_nothing(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, None)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        ) is None

    def test_a_full_desktop_refuses_before_prefill(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt is not None
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "system_memory_short_after_reclamation"
        # The request's growth is its largest moment: prefill end (the rows
        # and one chunk's scratch) or decode start (the rows and the copy of
        # the banked prompt snapshot). This test used to add both copies and
        # the scratch, as if all three were resident at once.
        rows = len(PROMPT) * PER_TOKEN
        growth = max(rows + srv_transients(), 2 * rows)
        shed_floor, abort_floor = sm.system_memory_floors(RAM)
        # The shed line starts reclamation; the abort line refuses. The
        # allocator pool is not credited: it is cleared first, and what it
        # gives back is in the next reading.
        assert receipt["system_shortfall_bytes"] == growth + shed_floor - 9 * GIB
        assert receipt["cache_cleared"] is True
        # The banked copy of the prompt gave way (it was what made decode
        # start the largest moment); the prefill end is still short of the
        # abort floor.
        assert receipt["prompt_publish_skipped"] is True
        after = rows + srv_transients()
        assert receipt["growth_bytes_after"] == after
        assert receipt["system_shortfall_bytes_after"] == after + abort_floor - 9 * GIB
        detail = srv._prefill_admission_refusal(_state(), receipt).detail
        message = detail["message"]
        assert "other apps" in message and "9.0 GiB free" in message
        assert "--allow-swap" in message
        assert detail["code"] == "insufficient_memory"
        assert detail["memory"]["retry_can_succeed"] is True
        assert detail["memory"]["retry_when"] == "after_other_apps_free_memory"

    def test_the_bank_is_shed_for_the_desktop_too(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        bank = _Bank(7 * GIB)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=bank, session_id="pi"
        )
        assert receipt is not None
        # No superseded clear since 2026-09-30: the LRU pass is the bank step.
        assert bank.cleared_sessions == []
        assert bank.shrink_calls and bank.shrink_calls[0][1] == "prefill_admission"

    def test_a_desktop_that_recovers_after_the_shed_is_admitted(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        readings = iter([_reading(9), _reading(30)])
        last = [_reading(30)]

        def reader():
            try:
                last[0] = next(readings)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(7 * GIB), session_id="pi"
        )
        assert receipt is not None
        assert receipt.get("refused") is not True
        assert receipt["system_shortfall_bytes_after"] == 0

    def test_allow_swap_keeps_the_operators_choice(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        receipt = srv._prefill_admission_shed(
            _state(allow_swap=True), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt is not None
        assert receipt.get("refused") is not True

    def test_the_receipt_reports_the_floors_it_enforced(self, monkeypatch):
        """The review's defect 7: the receipt printed the 2.5 percent floor
        (6.4 GiB on 128 GB) while the wired-aware one (11 GiB at 88 GiB
        wired) decided the refusal."""

        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        monkeypatch.setattr(
            sm, "_reader", lambda: _machine(free_gib=2, file_gib=4, wired_gib=88)
        )
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        )
        # The Mac's floors once the request's growth is wired: at the
        # requested chunk first, then at the chunk it runs.
        first = receipt["growth_by_chunk"][str(receipt["prefill_chunk_requested"])]
        assert receipt["system_abort_floor_bytes"] == (88 * GIB + first) // 16
        assert receipt["system_shed_floor_bytes"] == 2 * ((88 * GIB + first) // 16)
        after = receipt["growth_bytes_after"]
        assert receipt["system_abort_floor_bytes_after"] == (88 * GIB + after) // 16
        assert receipt["system_shed_floor_bytes_after"] == 2 * ((88 * GIB + after) // 16)
        assert receipt["system_memory"]["wired_bytes"] == 88 * GIB

    def test_a_short_prompt_between_the_floors_runs_after_reclamation(self, monkeypatch):
        """This test used to pin that a prompt under 4,096 tokens was never
        refused, however little the Mac had left. Every request is projected
        now. One that would leave the Mac between its floors runs after the
        engine gives back what it can; one that would take it under the
        abort floor is refused."""

        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        prompt = list(range(512))
        # 5 GiB free: between the 3.2 GiB abort floor and the 6.4 GiB shed
        # floor of a 128 GB Mac with nothing wired in this reading.
        _install(monkeypatch, 5)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=prompt, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt["reclamation_steps"][0] == "allocator_pool"
        assert receipt.get("refused") is not True
        _install(monkeypatch, 1)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=prompt, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "system_memory_short_after_reclamation"

    def test_the_pool_is_cleared_only_when_the_macs_line_needs_it(self, monkeypatch):
        """The engine line leaves the allocator pool out (MLX releases pooled
        buffers before it passes its own limit), so a request the Mac has
        room for is admitted without clearing a warm allocator."""

        import mlx.core as mx

        cleared = []
        monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
        # 85 GiB in use and 4 GiB pooled: with the pool, the 40K prompt's
        # 4.2 GiB of growth crosses the 93.1 GiB line; without it, it does
        # not, and the Mac has 40 GiB to give.
        _pin_engine(monkeypatch, active_gib=85, cache_gib=4)
        _install(monkeypatch, 40)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=list(range(40_000)), session_bank=_Bank(0), session_id="pi"
        ) is None
        assert cleared == []


def srv_transients() -> int:
    from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

    return int(RUNTIME_TRANSIENTS_BYTES)


class _LoopBank:
    def __init__(self, total, max_bytes):
        self.total_nbytes = total
        self.max_bytes = max_bytes
        self.calls = []

    def shrink_to_bytes(
        self,
        target,
        *,
        reason,
        protect_active=False,
        protect_session_ids=None,
        protect_unwritten=False,
        receipt_out=None,
    ):
        # The real bank's keywords: the trim names the sessions generating.
        self.calls.append((target, reason))
        self.total_nbytes = min(self.total_nbytes, target)
        return 1


def _loop_state(bank):
    return SimpleNamespace(
        sessions=SimpleNamespace(bank=bank),
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


def _run_loop(state, monkeypatch, *, seconds: float):
    import asyncio

    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: 1)

    async def run():
        task = asyncio.ensure_future(srv._memory_pressure_loop(state, interval_s=3600))
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


class TestGuardLoop:
    """macOS says normal and the allocator is far from its limit in every case."""

    def test_a_short_desktop_raises_the_level_and_names_itself(self, monkeypatch):
        _install(monkeypatch, 5)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 2
        assert state.dashboard.last_memory_pressure_source == "system_available"
        assert state.dashboard.last_system_available_bytes == 5 * GIB
        assert bank.calls == [(4 * GIB, "memory_pressure_warning")]
        event = list(state.dashboard.memory_guard_events)[-1]
        assert event["level_source"] == "system_available"
        assert event["system_available_bytes"] == 5 * GIB

    def test_under_the_abort_floor_is_critical_and_empties_the_bank(self, monkeypatch):
        _install(monkeypatch, 2)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 4
        assert bank.calls == [(0, "memory_pressure_critical")]

    def test_a_busy_engine_under_the_abort_floor_is_aborted_within_seconds(self, monkeypatch):
        _install(monkeypatch, 2)
        monkeypatch.setattr(srv, "_SYSTEM_SHORT_INTERVAL_S", 0.01)
        monkeypatch.setattr(srv, "_engine_busy_signal", lambda state: True)
        state = _loop_state(_LoopBank(total=0, max_bytes=8 * GIB))
        _run_loop(state, monkeypatch, seconds=0.3)
        assert srv._pressure_abort_requested(state)
        actions = [e["action"] for e in state.dashboard.memory_guard_events]
        assert "pressure_abort_armed" in actions

    def test_a_roomy_desktop_keeps_the_slow_tick_and_does_nothing(self, monkeypatch):
        _install(monkeypatch, 40)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 1
        assert state.dashboard.last_system_available_bytes == 40 * GIB
        assert bank.calls == []
