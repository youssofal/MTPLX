"""Stats footer scoping: MTPLX-owned surfaces only, never the anonymous API.

The visible TPS footer is product UI on MTPLX-owned chat surfaces. On the
OpenAI-compat API it is server-injected prose inside model content:
it broke temp-0 byte equality, created a wire-vs-usage token mismatch, and
deflated externally measured tok/s (2026-08-05 showdown receipts).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import mtplx.server.openai as openai_mod
from mtplx.server.openai import STATS_FOOTER_MARKER, create_app

from test_server_openai import _fake_generation, _fake_state


def _footer_state():
    state = _fake_state()
    state.args.stats_footer = True
    # managed-client lanes touch tokenizer.encode; the shared stub only decodes
    state.runtime.tokenizer.encode = lambda text, **_kw: [ord(c) % 251 for c in str(text)]
    return state


def _post_chat(client, headers=None):
    return client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", **(headers or {})},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )


def _content(response) -> str:
    return response.json()["choices"][0]["message"]["content"] or ""


def test_anonymous_api_client_gets_no_footer(monkeypatch):
    state = _footer_state()
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    r = _post_chat(client)
    assert r.status_code == 200
    assert STATS_FOOTER_MARKER not in _content(r)


def test_managed_client_hint_keeps_footer(monkeypatch):
    state = _footer_state()
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    r = _post_chat(client, headers={"x-mtplx-client": "chat"})
    assert r.status_code == 200
    assert STATS_FOOTER_MARKER in _content(r)


def test_managed_agent_clients_get_no_footer(monkeypatch):
    """opencode/pi/hermes/openwebui are MANAGED but parse assistant content
    programmatically — the exact consumer class footer scoping protects.
    Regression for the 2.5.3 pre-ship review F3: the first scoping pass
    admitted every managed hint, so OpenCode still received the footer."""
    state = _footer_state()
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    for hint in ("opencode", "pi", "hermes", "openwebui"):
        r = _post_chat(client, headers={"x-mtplx-client": hint})
        assert r.status_code == 200
        assert STATS_FOOTER_MARKER not in _content(r), hint
    # UA-sniffed OpenCode (no explicit header) must also stay footer-free.
    r = _post_chat(client, headers={"user-agent": "opencode/1.14.48"})
    assert r.status_code == 200
    assert STATS_FOOTER_MARKER not in _content(r)


def test_app_ui_hints_keep_footer(monkeypatch):
    state = _footer_state()
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    for hint in ("mtplx-app", "mtplxapp", "mtplx"):
        r = _post_chat(client, headers={"x-mtplx-client": hint})
        assert r.status_code == 200
        assert STATS_FOOTER_MARKER in _content(r), hint


def test_scope_all_env_restores_legacy_behavior(monkeypatch):
    state = _footer_state()
    monkeypatch.setenv("MTPLX_STATS_FOOTER_SCOPE", "all")
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    r = _post_chat(client)
    assert r.status_code == 200
    assert STATS_FOOTER_MARKER in _content(r)


def test_no_stats_footer_flag_still_wins_everywhere(monkeypatch):
    state = _footer_state()
    state.args.stats_footer = False
    monkeypatch.setattr(openai_mod, "_run_generation", lambda *a, **k: _fake_generation("ok"))
    client = TestClient(create_app(state))
    r = _post_chat(client, headers={"x-mtplx-client": "chat"})
    assert r.status_code == 200
    assert STATS_FOOTER_MARKER not in _content(r)
