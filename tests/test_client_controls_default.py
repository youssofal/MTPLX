"""MTPLX_CLIENT_CONTROLS_DEFAULT: anonymous body params get applied by default.

Default is 'honor' since 2.5.3 (OpenAI-API semantics for anonymous clients;
issue #241 receipts). Managed surfaces stay server-owned in BOTH modes, and
MTPLX_CLIENT_CONTROLS_DEFAULT=hints restores the pre-2.5.3 policy.
"""

from __future__ import annotations

from mtplx.server.openai import _client_controls_allowed


def test_default_honors_anonymous_controls(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({}, {}) is True


def test_header_opt_in_works_under_hints_mode(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    assert _client_controls_allowed({"x-mtplx-allow-client-controls": "1"}, {}) is True


def test_hints_mode_ignores_anonymous_controls(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    assert _client_controls_allowed({}, {}) is False


def test_default_keeps_managed_surfaces_server_owned(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({"x-mtplx-client": "opencode"}, {}) is False
    assert _client_controls_allowed({"x-mtplx-client": "chat"}, {}) is False


def test_honor_mode_keeps_managed_surfaces_server_owned(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "honor")
    assert _client_controls_allowed({"x-mtplx-client": "opencode"}, {}) is False
    assert _client_controls_allowed({"x-mtplx-client": "chat"}, {}) is False


def test_unknown_value_falls_back_to_honor(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "yolo")
    assert _client_controls_allowed({}, {}) is True
