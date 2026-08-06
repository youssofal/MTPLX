"""Endpoint-discovery hygiene: probes get clean 4xx, never Python internals.

External conformance tools POST empty/malformed bodies to every OpenAI-shaped
path and quote our error bodies verbatim in their reports. Contract:
- empty/invalid request bodies → 4xx with a human message;
- unimplemented endpoints → 404;
- unhandled server errors → 500 whose body carries a request_id but NO
  exception class names or reprs (those go to the server log).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import mtplx.server.openai as openai_mod
from mtplx.server.openai import create_app

from test_server_openai import _fake_state


def _client(state=None):
    if state is None:
        state = _fake_state()
    return TestClient(create_app(state), raise_server_exceptions=False)


def _generation_ready_state():
    state = _fake_state()
    # reach _run_generation: the shared stub tokenizer lacks encode
    state.runtime.tokenizer.encode = lambda text, **_kw: [ord(c) % 251 for c in str(text)]
    return state


def test_empty_bodies_get_clean_400s():
    client = _client(_generation_ready_state())
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
        r = client.post(path, json={})
        assert r.status_code == 400, (path, r.status_code, r.text)
        assert "must not be empty" in r.text


def test_malformed_types_get_422_not_500():
    client = _client(_generation_ready_state())
    r = client.post(
        "/v1/chat/completions", json={"messages": "not-an-array", "input": 42}
    )
    assert r.status_code == 422


def test_unknown_endpoints_are_404():
    client = _client(_generation_ready_state())
    for path in ("/v1/embeddings", "/v1/responses", "/v1/images/generations"):
        assert client.post(path, json={}).status_code == 404


def test_unhandled_errors_hide_python_internals(monkeypatch):
    client = _client(_generation_ready_state())

    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(openai_mod, "_run_generation", boom)
    monkeypatch.delenv("MTPLX_DEBUG_ERRORS", raising=False)
    r = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert r.status_code == 500
    assert "RuntimeError" not in r.text
    assert "secret internal detail" not in r.text
    assert "request_id=" in r.text


def test_debug_env_restores_detail(monkeypatch):
    client = _client(_generation_ready_state())

    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(openai_mod, "_run_generation", boom)
    monkeypatch.setenv("MTPLX_DEBUG_ERRORS", "1")
    r = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert r.status_code == 500
    assert "RuntimeError" in r.text


def test_non_finite_sampler_controls_get_400():
    """Honored-by-default controls (2.5.3): a JSON NaN temperature must be
    a clean 400 at the boundary, not a 500 from the softmax mid-request."""
    client = _client(_generation_ready_state())
    for field in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        r = client.post(
            "/v1/chat/completions",
            headers={
                "x-mtplx-cache-mode": "bypass",
                "content-type": "application/json",
            },
            content=(
                '{"messages":[{"role":"user","content":"hi"}],'
                f'"max_tokens":4,"{field}":NaN}}'
            ),
        )
        assert r.status_code == 400, (field, r.status_code, r.text[:200])
        assert "finite" in r.text
