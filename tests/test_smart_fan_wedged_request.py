"""A parked request gives the fans back; a healthy long one keeps them (#295).

End to end through the two real halves: the server's activity probe reading
the real owner progress heartbeat, and the real ``SmartFanController``
stale-lease reconciler. Only the fan hardware is stubbed, with the same stub
the thermal tests use, so nothing here can command a real fan.

Also pins the parsing law of ``MTPLX_FOREGROUND_STALL_DEADLINE_S``. The stream
deadline once read ``0`` as "use the default" (#448) and an unreadable value
used to stop the server at import; this one must do neither.
"""

from __future__ import annotations

import threading
import time

import pytest

import mtplx.server.openai as openai_mod
from mtplx import progress_heartbeat, thermal
from test_thermal import _patch_smart_fan_hardware, _wait_until


class _RegisteredRequest:
    """What the fan activity probe reads, with one request registered."""

    def __init__(self, *, deadline_s: float) -> None:
        self._fan_stall_probe = openai_mod._OwnerStallProbe(deadline_s=deadline_s)
        self.model_scheduler = None
        self.last_request_started_at = 0.0
        self.last_request_at = 0.0

    def has_foreground(self) -> bool:
        return True


def _controller(monkeypatch, state, calls):
    _patch_smart_fan_hardware(monkeypatch, calls)
    monkeypatch.setattr(thermal.SmartFanController, "_ACTIVITY_POLL_INTERVAL_S", 0.02)
    monkeypatch.setenv("MTPLX_SMART_FAN_STALE_LEASE_S", "0.2")
    return thermal.SmartFanController(
        restore_delay_s=0,
        sensorless_delay_s=0,
        activity_probe=lambda: openai_mod.ServerState._smart_fan_activity_probe(state),
    )


def test_a_parked_request_gives_the_fans_back(monkeypatch):
    calls: list[str] = []
    state = _RegisteredRequest(deadline_s=0.2)
    controller = _controller(monkeypatch, state, calls)
    try:
        controller.begin_request("parked")
        assert controller.wait_for_ramp(5.0) is True

        # The request never ends and the owner never settles another eval.
        assert _wait_until(lambda: controller.status()["active_count"] == 0)
        assert _wait_until(lambda: "auto" in calls)
        status = controller.status()
        assert status["stale_leases_reconciled"] == 1
        assert status["commanded_max"] is False
    finally:
        controller._shutdown = True


def test_a_long_healthy_request_keeps_the_fans(monkeypatch):
    calls: list[str] = []
    state = _RegisteredRequest(deadline_s=0.2)
    controller = _controller(monkeypatch, state, calls)
    stop = threading.Event()

    def owner() -> None:
        while not stop.is_set():
            progress_heartbeat.tick()  # one settled eval
            time.sleep(0.01)

    worker = threading.Thread(target=owner, daemon=True)
    worker.start()
    try:
        controller.begin_request("long-generation")
        assert controller.wait_for_ramp(5.0) is True

        # Five times the deadline plus the stale window, with work settling.
        time.sleep(2.0)
        status = controller.status()
        assert status["active_count"] == 1
        assert status["stale_leases_reconciled"] == 0
        assert status["commanded_max"] is True
        assert "auto" not in calls
        controller.end_request("long-generation", wait_for_restore=True)
    finally:
        stop.set()
        worker.join(timeout=2.0)
        controller._shutdown = True


def test_the_check_switched_off_keeps_the_old_presence_only_reading():
    state = _RegisteredRequest(deadline_s=0.0)
    probe = openai_mod.ServerState._smart_fan_activity_probe
    assert probe(state) is True
    time.sleep(0.05)
    assert probe(state) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 180.0),
        ("", 180.0),
        ("   ", 180.0),
        ("0", 0.0),
        ("0.0", 0.0),
        ("45", 45.0),
        (" 600 ", 600.0),
        ("-5", 0.0),
        ("three minutes", 180.0),
        (90, 90.0),
    ],
)
def test_the_deadline_parses_like_the_stream_deadline(raw, expected):
    assert openai_mod._resolve_foreground_stall_deadline_s(raw) == expected


def test_the_fan_deadline_is_more_eager_than_the_stream_deadline_by_default():
    """It only gives up the fans, so it may fire first; the sum with the
    120 s stale-lease window is the 300 s at which a stream is failed."""
    assert (
        openai_mod.FOREGROUND_STALL_DEADLINE_DEFAULT_S
        < openai_mod.STREAM_STALL_DEADLINE_DEFAULT_S
    )
