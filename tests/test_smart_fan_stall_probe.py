"""Smart-fan activity probe must distinguish *progress* from *presence* (#201).

The stale-lease reconciler in ``SmartFanController`` only drops leaked fan
leases while the activity probe reports the engine idle. Before this fix the
server probe returned a bare ``True`` for "a foreground request is
registered", so a WEDGED generation — request registered, owner thread making
no forward progress — read as legitimately busy forever and pinned the fans at
maximum indefinitely (observed 2026-08-18: 20 leases held for ~15 h, fans
commanded to 5349/5777 RPM, ``foreground_active: 1`` with zero completions
while the client had already been killed).

The engine already ticks ``progress_heartbeat`` on every decode microbatch and
every completed owner work item, and ``_OwnerStallProbe`` (#86) already turns
that into a "frozen for N seconds" reading for the streaming watchdog. These
tests pin that the fan probe consults the same signal, so "busy" means
*advancing*, not merely *registered*.
"""

from __future__ import annotations

import mtplx.server.openai as openai_mod


class _FakeStallProbe:
    """Stands in for ``_OwnerStallProbe``: None = progressing, float = frozen."""

    def __init__(self, frozen_for_s: float | None):
        self._frozen_for_s = frozen_for_s
        self.observed = 0

    def observe(self, now_s: float | None = None) -> float | None:
        self.observed += 1
        return self._frozen_for_s


class _StubState:
    """Minimal stand-in exposing exactly what the probe reads."""

    def __init__(
        self,
        *,
        foreground: bool,
        stall_probe: _FakeStallProbe | None = None,
        last_request_started_at: float = 0.0,
    ):
        self._foreground = foreground
        self._fan_stall_probe = stall_probe
        self.model_scheduler = None
        self.last_request_started_at = last_request_started_at
        self.last_request_at = last_request_started_at

    def has_foreground(self) -> bool:
        return self._foreground


def _probe(state) -> bool:
    return openai_mod.ServerState._smart_fan_activity_probe(state)


def test_probe_reports_busy_while_a_generation_makes_progress():
    """A long legitimate generation must keep its fan boost (#201 contract)."""
    stall = _FakeStallProbe(None)
    assert _probe(_StubState(foreground=True, stall_probe=stall)) is True
    assert stall.observed == 1


def test_probe_reports_idle_when_a_foreground_request_is_wedged():
    """The regression: registered request, owner heartbeat frozen past the
    deadline. Without the fix this returns True forever and the fans never
    come back to auto."""
    stall = _FakeStallProbe(900.0)
    assert _probe(_StubState(foreground=True, stall_probe=stall)) is False


def test_probe_stays_busy_when_no_stall_probe_is_wired():
    """Fail safe: a state without the probe keeps today's behaviour rather
    than restoring fans under a live workload."""
    assert _probe(_StubState(foreground=True, stall_probe=None)) is True


def test_probe_reports_idle_with_no_foreground_and_no_recent_request():
    """Unchanged behaviour: nothing registered, nothing recent -> idle."""
    state = _StubState(foreground=False, last_request_started_at=0.0)
    assert _probe(state) is False


def test_server_wires_a_real_stall_probe_with_a_positive_deadline():
    """The production deadline must be finite and > 0, or the fan probe can
    never report a wedge (``_OwnerStallProbe`` disables itself at <= 0)."""
    deadline = openai_mod.FOREGROUND_STALL_DEADLINE_S
    assert deadline > 0
    probe = openai_mod._OwnerStallProbe(deadline_s=deadline)
    assert probe.observe(now_s=0.0) is None


# --- Rearming across an idle window ---------------------------------------
#
# ``_fan_stall_probe`` is a single long-lived instance on ``ServerState``. It
# is only READ while foreground work exists, so its frozen-since baseline is
# whatever the last poll of the previous request left behind. After a quiet
# night the heartbeat has been frozen for hours; without a rearm on the
# idle-to-active edge the very next request is classified as wedged on its
# FIRST poll, and the documented budget (FOREGROUND_STALL_DEADLINE_S plus
# MTPLX_SMART_FAN_STALE_LEASE_S) collapses to the stale-lease window alone.
# The rearm must fire ONLY on the 0 -> 1 transition: rearming for each extra
# concurrent request would let a busy server hide a genuinely wedged owner.


class _ForegroundStub:
    """Exactly what ``begin_foreground`` and the fan probe touch."""

    def __init__(self, probe):
        import threading

        self.foreground_lock = threading.Lock()
        self.foreground_active = 0
        self.last_request_started_at = 0.0
        self.last_request_at = 0.0
        self.model_scheduler = None
        self._fan_stall_probe = probe

    def has_foreground(self) -> bool:
        return self.foreground_active > 0


def _begin(state) -> None:
    openai_mod.ServerState.begin_foreground(state)


def test_a_fresh_request_is_not_stalled_by_a_stale_idle_baseline():
    now = [0.0]
    probe = openai_mod._OwnerStallProbe(deadline_s=100.0, clock=lambda: now[0])
    state = _ForegroundStub(probe)

    # A long idle window: no owner work, so no heartbeat ticks at all.
    now[0] = 500.0
    _begin(state)

    assert _probe(state) is True, "a just-admitted request must not read wedged"


def test_a_second_concurrent_request_does_not_rearm_the_stall_probe():
    now = [0.0]
    probe = openai_mod._OwnerStallProbe(deadline_s=100.0, clock=lambda: now[0])
    state = _ForegroundStub(probe)

    _begin(state)
    now[0] = 500.0
    assert _probe(state) is False, "the first request is genuinely wedged"

    _begin(state)  # 1 -> 2: another request arrives behind the wedge
    assert _probe(state) is False, "a queued arrival must not forge liveness"


def test_the_probe_rearms_again_after_the_server_goes_fully_idle():
    now = [0.0]
    probe = openai_mod._OwnerStallProbe(deadline_s=100.0, clock=lambda: now[0])
    state = _ForegroundStub(probe)

    _begin(state)
    now[0] = 500.0
    assert _probe(state) is False
    openai_mod.ServerState.end_foreground(state)

    now[0] = 900.0
    _begin(state)
    assert _probe(state) is True
