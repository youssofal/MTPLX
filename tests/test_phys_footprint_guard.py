"""Real OS footprint as an independent floor on the memory guards.

Two independently reported machines (M5 Max 128 GB) kernel-panicked
(watchdog timeout, near-zero free pages) with a single MTPLX worker process
resident *above* its own configured Metal memory limit, even though every
guard along the way -- the prefill admission shed (#450's own refusal
check), the background pressure loop -- reported the allocator's
active+cache comfortably under the line. #456 shows the same family from a
different angle: active_bytes itself crept ~17 GB past a fresh floor with
the bank and cache both at zero, "nothing sheddable, only restart
recovers".

Both shapes share one property: the number every guard trusts,
``mx.get_active_memory() + mx.get_cache_memory()``, is MLX's own account of
what it allocated through Metal, never a read of what the kernel actually
holds resident for this process. These tests pin that a *real* OS-reported
footprint (``mtplx.os_memory.phys_footprint_bytes``, imported into this
module as ``phys_footprint_bytes``) now floors both guards -- and that a
healthy process, or a platform where the probe cannot run at all, sees
byte-identical behavior to before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv

GIB = 1024**3
LIMIT = 96 * GIB
KV_PER_TOKEN = 24576
AUX_PER_TOKEN = 7872


def _state():
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": LIMIT},
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=KV_PER_TOKEN,
            aux_bytes_per_token=AUX_PER_TOKEN,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
    )


class _EmptyBank:
    total_nbytes = 0

    def longest_prefix(self, token_ids):
        return None


def _pin_live_stats(monkeypatch, *, active, cache):
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(active),
            "cache_memory_bytes": int(cache),
        },
    )


def _pin_footprint(monkeypatch, value):
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: value)


def _shed(state, prompt_ids, bank=None, session_id=None):
    return srv._prefill_admission_shed(
        state,
        prompt_ids=prompt_ids,
        session_bank=bank,
        session_id=session_id,
    )


class TestAdmissionShedFootprintFloor:
    def test_healthy_process_unaffected_when_probe_fails(self, monkeypatch):
        """A missing/failed probe (non-Darwin, no libproc) must reproduce
        exactly today's behavior: inert when the allocator reports health."""
        _pin_live_stats(monkeypatch, active=60 * GIB, cache=2 * GIB)
        _pin_footprint(monkeypatch, None)
        assert _shed(_state(), list(range(40_000)), _EmptyBank(), "pi") is None

    def test_low_real_footprint_does_not_mask_allocator_pressure(self, monkeypatch):
        """The floor is a max(), never a replacement: a real footprint lower
        than what the allocator already reports must not hide a real
        allocator-reported deficit."""
        miss = 60_000
        per_token = KV_PER_TOKEN + AUX_PER_TOKEN
        from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

        transients = int(RUNTIME_TRANSIENTS_BYTES)
        active = LIMIT + GIB - miss * per_token - transients
        _pin_live_stats(monkeypatch, active=active, cache=0)
        _pin_footprint(monkeypatch, 1 * GIB)  # far below active+cache
        receipt = _shed(_state(), list(range(miss)), _EmptyBank(), "pi")
        assert receipt is not None
        assert receipt["refused"] is True

    def test_footprint_alone_trips_the_cheap_gate(self, monkeypatch):
        """The allocator says everything fits (a fully-cold prefill would
        stay under the line); a real footprint at the #456/#450 shape must
        still make the guard act instead of returning early."""
        _pin_live_stats(monkeypatch, active=10 * GIB, cache=0)
        _pin_footprint(monkeypatch, int(LIMIT * 0.99))
        receipt = _shed(_state(), list(range(40_000)), _EmptyBank(), "pi")
        assert receipt is not None
        assert receipt["action"] == "prefill_admission_shed"
        assert receipt["phys_footprint_bytes"] == int(LIMIT * 0.99)

    def test_footprint_alone_forces_refusal_after_reclamation(self, monkeypatch):
        """The #450 shape exactly: allocator numbers read under the limit
        after every reclamation step, but the real OS footprint does not.
        Without the floor this receipt would admit the request; with it,
        the request must be refused before prefill -- a structured 507
        beats a kernel panic."""
        miss = 40_000
        _pin_live_stats(monkeypatch, active=50 * GIB, cache=0)
        _pin_footprint(monkeypatch, LIMIT + 2 * GIB)
        receipt = _shed(_state(), list(range(miss)), _EmptyBank(), "pi")
        assert receipt is not None
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["phys_footprint_bytes_after"] == LIMIT + 2 * GIB
        assert receipt["projected_bytes_after"] > LIMIT

    def test_admits_when_both_signals_agree_it_fits(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=50 * GIB, cache=0)
        _pin_footprint(monkeypatch, 55 * GIB)
        assert _shed(_state(), list(range(1000)), _EmptyBank(), "pi") is None


class TestAllocatorPressureLevelFootprintFloor:
    def test_none_footprint_preserves_prior_behavior(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=50 * GIB, cache=0)
        _pin_footprint(monkeypatch, None)
        level, fraction = srv._allocator_pressure_level(_state())
        assert level == 1
        assert fraction == (50 * GIB) / LIMIT

    def test_low_real_footprint_does_not_lower_the_allocator_reading(
        self, monkeypatch
    ):
        _pin_live_stats(monkeypatch, active=int(LIMIT * 1.05), cache=0)
        _pin_footprint(monkeypatch, GIB)
        level, fraction = srv._allocator_pressure_level(_state())
        assert level == 4
        assert fraction == pytest.approx(1.05)

    def test_critical_from_real_footprint_alone(self, monkeypatch):
        """The exact shape both panics shared: the allocator's own numbers
        read healthy while the OS already sees the process past the limit.
        The background pressure loop (and the sustained-abort it arms) must
        see CRITICAL here, not WARNING-or-below."""
        _pin_live_stats(monkeypatch, active=10 * GIB, cache=0)
        _pin_footprint(monkeypatch, int(LIMIT * 1.10))
        level, fraction = srv._allocator_pressure_level(_state())
        assert level == 4
        assert fraction == pytest.approx(1.10)

    def test_warning_from_real_footprint_alone(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=10 * GIB, cache=0)
        _pin_footprint(monkeypatch, int(LIMIT * 0.98))
        level, _fraction = srv._allocator_pressure_level(_state())
        assert level == 2

    def test_healthy_on_both_signals(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=10 * GIB, cache=0)
        _pin_footprint(monkeypatch, int(LIMIT * 0.5))
        level, _fraction = srv._allocator_pressure_level(_state())
        assert level == 1


class TestPhysFootprintBytes:
    """The probe itself: stdlib-only, never raises, returns a plausible
    number for the current process on macOS."""

    def test_returns_a_plausible_value_for_self(self):
        from mtplx.os_memory import phys_footprint_bytes

        value = phys_footprint_bytes()
        assert value is None or value > 0

    def test_bad_pid_returns_none_not_raise(self):
        from mtplx.os_memory import phys_footprint_bytes

        # PID 2**30 will not exist; proc_pid_rusage must fail closed.
        assert phys_footprint_bytes(pid=2**30) is None
