"""Per-request presence/frequency penalty wiring and live-settings tests.

PR #120 added the engine + server-default flags; this suite pins the product
wiring on top of it:

* typed request fields on ``/v1/chat/completions`` and ``/v1/completions``
  (previously accepted via ``extra="allow"`` and silently dropped — the #102
  failure mode),
* precedence inside ``_generation_params``: request value > server default
  (``--default-*-penalty`` / live settings) > 0.0,
* the control-ownership policy: client penalties are ignored (fall back to the
  server default) unless client controls are allowed, mirroring temperature,
* the ``/v1/mtplx/settings`` live-update path mapping the public
  ``presence_penalty``/``frequency_penalty`` keys onto the server's
  ``default_*_penalty`` args without a restart.
"""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import (
    ChatCompletionRequest,
    CompletionRequest,
    MTPLXSettingsUpdate,
    _generation_params,
    create_app,
)
from tests.test_server_openai import _fake_state


def _params_state(**arg_overrides):
    args = SimpleNamespace(
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)
    return SimpleNamespace(context_window=1000, args=args)


def test_generation_params_default_zero_when_nothing_set():
    _max, sampler, _limits = _generation_params(
        _params_state(),
        prompt_token_count=10,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    assert sampler.presence_penalty == 0.0
    assert sampler.frequency_penalty == 0.0


def test_generation_params_server_default_applies_when_request_unset():
    _max, sampler, _limits = _generation_params(
        _params_state(default_presence_penalty=0.7, default_frequency_penalty=0.3),
        prompt_token_count=10,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        presence_penalty=None,
        frequency_penalty=None,
    )
    assert sampler.presence_penalty == 0.7
    assert sampler.frequency_penalty == 0.3


def test_generation_params_request_value_wins_over_server_default():
    _max, sampler, _limits = _generation_params(
        _params_state(default_presence_penalty=0.7, default_frequency_penalty=0.3),
        prompt_token_count=10,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        presence_penalty=1.5,
        frequency_penalty=0.0,
    )
    assert sampler.presence_penalty == 1.5
    # An explicit request 0.0 must override a non-zero server default, not
    # fall through to it — 0.0 is a real value, not "unset".
    assert sampler.frequency_penalty == 0.0


def test_request_models_parse_penalties_as_typed_fields():
    chat = ChatCompletionRequest.model_validate(
        {"messages": [], "presence_penalty": 1.2, "frequency_penalty": -0.5}
    )
    assert chat.presence_penalty == 1.2
    assert chat.frequency_penalty == -0.5
    completion = CompletionRequest.model_validate(
        {"prompt": "x", "presence_penalty": 0.4}
    )
    assert completion.presence_penalty == 0.4
    assert completion.frequency_penalty is None
    settings = MTPLXSettingsUpdate.model_validate({"presence_penalty": 0.9})
    assert settings.presence_penalty == 0.9


def _fake_run_generation_capture(captured):
    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["presence_penalty"] = kwargs.get("presence_penalty")
        captured["frequency_penalty"] = kwargs.get("frequency_penalty")
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {"completion_tokens": 1},
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    return fake_run_generation


def test_chat_request_penalties_reach_generation_when_controls_allowed(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai, "_run_generation", _fake_run_generation_capture(captured)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "presence_penalty": 1.1,
            "frequency_penalty": 0.2,
        },
    )

    assert response.status_code == 200
    assert captured["presence_penalty"] == 1.1
    assert captured["frequency_penalty"] == 0.2


def test_chat_request_penalties_ignored_without_client_controls(monkeypatch):
    # Pre-2.5.3 'hints' policy, kept selectable via env.
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai, "_run_generation", _fake_run_generation_capture(captured)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "presence_penalty": 1.1,
        },
    )

    assert response.status_code == 200
    # Server-owned controls: the request field is observability, not policy.
    # None here means _generation_params falls to the server default.
    assert captured["presence_penalty"] is None
    assert captured["frequency_penalty"] is None


def test_completions_request_penalties_reach_generation(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_prompt", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai, "_run_generation", _fake_run_generation_capture(captured)
    )

    response = client.post(
        "/v1/completions",
        headers={"x-mtplx-allow-client-controls": "1"},
        json={
            "prompt": "hello",
            "max_tokens": 4,
            "presence_penalty": 0.6,
            "frequency_penalty": 1.4,
        },
    )

    assert response.status_code == 200
    assert captured["presence_penalty"] == 0.6
    assert captured["frequency_penalty"] == 1.4


def test_settings_endpoint_updates_penalties_live():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))
    headers = {"Authorization": "Bearer mtplx-local"}

    initial = client.get("/v1/mtplx/settings", headers=headers)
    assert initial.status_code == 200
    assert initial.json()["presence_penalty"] == 0.0
    assert initial.json()["frequency_penalty"] == 0.0

    updated = client.post(
        "/v1/mtplx/settings",
        json={"presence_penalty": 0.8, "frequency_penalty": 0.25},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["applied"] == {
        "presence_penalty": 0.8,
        "frequency_penalty": 0.25,
    }
    assert updated.json()["presence_penalty"] == 0.8
    assert updated.json()["frequency_penalty"] == 0.25
    # The live args attribute _generation_params reads (the CLI's
    # --default-*-penalty destination) must be updated in place.
    assert state.args.default_presence_penalty == 0.8
    assert state.args.default_frequency_penalty == 0.25


def test_settings_endpoint_rejects_out_of_range_penalty():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))
    headers = {"Authorization": "Bearer mtplx-local"}

    response = client.post(
        "/v1/mtplx/settings",
        json={"presence_penalty": 3.5},
        headers=headers,
    )
    assert response.status_code == 400
    assert "between -2 and 2" in str(response.json())
    assert float(getattr(state.args, "default_presence_penalty", 0.0) or 0.0) == 0.0


def test_settings_penalty_update_does_not_touch_draft_sampler():
    # temperature/top_p/top_k settings implicitly mirror onto the draft
    # sampler; penalties must not (the draft proposes, the target enforces
    # the penalized distribution through speculative verification).
    state = _fake_state(api_key="mtplx-local")
    state.lock = Lock()
    state.draft_sampler = openai.SamplerConfig(
        temperature=0.1, top_p=0.95, top_k=20
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={"presence_penalty": 1.0},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert response.status_code == 200
    assert state.draft_sampler.presence_penalty == 0.0
    assert state.draft_sampler.temperature == 0.1
