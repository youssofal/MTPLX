"""MTPLX_CLIENT_CONTROLS_DEFAULT=honor: anonymous body params get applied.

Default stays 'hints' (founder-owned generation policy). Managed surfaces
stay server-owned in BOTH modes.
"""

from __future__ import annotations

from mtplx.server.openai import _client_controls_allowed


def test_default_hints_ignores_anonymous_controls(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({}, {}) is False


def test_header_opt_in_still_works(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({"x-mtplx-allow-client-controls": "1"}, {}) is True


def test_honor_mode_applies_anonymous_controls(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "honor")
    assert _client_controls_allowed({}, {}) is True


def test_honor_mode_keeps_managed_surfaces_server_owned(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "honor")
    assert _client_controls_allowed({"x-mtplx-client": "opencode"}, {}) is False
    assert _client_controls_allowed({"x-mtplx-client": "chat"}, {}) is False


def test_unknown_value_falls_back_to_hints(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "yolo")
    assert _client_controls_allowed({}, {}) is False
