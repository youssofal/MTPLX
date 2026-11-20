"""A metrics stream opened with rows=changed sends the request rows (about
80% of a snapshot's bytes) only when they changed, and at least every
DASHBOARD_ROWS_REFRESH_S; every path that adds or updates a row says so."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from mtplx.server import openai as server
from mtplx.server.stream_recovery import _metric_for_request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_openai import _fake_state  # noqa: E402


def test_the_gate_sends_rows_when_they_change_and_on_a_refresh():
    gate = server._DashboardRowsGate(only_when_changed=True)
    assert gate.rows_due(3, 10.0) is True  # the stream's first snapshot
    assert gate.rows_due(3, 10.5) is False
    assert gate.rows_due(4, 11.0) is True  # a row was added or updated
    assert gate.rows_due(4, 11.5) is False
    assert gate.rows_due(4, 11.0 + server.DASHBOARD_ROWS_REFRESH_S) is True
    # A server state without the counter never leaves the rows out.
    assert gate.rows_due(None, 20.0) is True and gate.rows_due(None, 20.1) is True


def test_a_stream_that_did_not_ask_always_gets_the_rows():
    gate = server._DashboardRowsGate(only_when_changed=False)
    assert [gate.rows_due(1, t) for t in (0.0, 0.1, 0.2)] == [True, True, True]


def test_a_snapshot_without_rows_says_so_and_a_full_one_is_unchanged():
    state = _fake_state()
    state.last_metrics = [{"request_id": "a", "decode_tok_s": 70.0}]
    full = server._mtplx_dashboard_snapshot(state)
    assert full["latest"]["request_id"] == "a" and len(full["recent"]) == 1
    assert "rows_unchanged" not in full
    lean = server._mtplx_dashboard_snapshot(state, rows=False)
    assert lean["latest"] is None and lean["recent"] == [] and lean["rows_unchanged"] is True
    assert set(full) - set(lean) == set() and set(lean) - set(full) == {"rows_unchanged"}


def test_finding_a_row_to_update_counts_as_a_change():
    state = SimpleNamespace(last_metrics=[{"request_id": "a"}, {"request_id": "b"}])
    assert _metric_for_request(state, "missing") is None
    assert getattr(state, "metrics_revision", 0) == 0
    row = _metric_for_request(state, "a")
    row["stream_attempts"] = 2
    assert state.metrics_revision == 1


def test_a_response_tail_landing_in_its_row_counts_as_a_change():
    state = SimpleNamespace(last_metrics=[{"request_id": "a"}], metrics_revision=5)
    server._merge_response_tail_into_metrics(state, "a", {"postcommit_ok": True})
    assert state.last_metrics[0]["postcommit_ok"] is True and state.metrics_revision == 6
    server._merge_response_tail_into_metrics(state, "other", {"postcommit_ok": False})
    assert state.metrics_revision == 6


def test_a_row_joining_the_ring_is_appended_trimmed_and_noted():
    # _append_metrics_row is how every finished request's row joins the
    # ring (the record sink and both completion paths call it).
    rows = [{"request_id": str(index)} for index in range(100)]
    state = SimpleNamespace(last_metrics=rows, metrics_revision=7)

    server._append_metrics_row(state, {"request_id": "new"})

    assert len(state.last_metrics) == 100
    assert state.last_metrics[-1] == {"request_id": "new"}
    assert state.last_metrics[0] == {"request_id": "1"}
    assert state.metrics_revision == 8


def test_a_row_joining_a_short_ring_keeps_every_row():
    state = SimpleNamespace(last_metrics=[{"request_id": "a"}])

    server._append_metrics_row(state, {"request_id": "b"})

    assert [row["request_id"] for row in state.last_metrics] == ["a", "b"]
    assert state.metrics_revision == 1
