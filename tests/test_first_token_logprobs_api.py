"""First-token logprobs on the regular generation routes.

/v1/completions (logprobs=K, no echo) and /v1/chat/completions
(logprobs=true + top_logprobs=K) return the raw next-token distribution of
the first generated token, and only for max_tokens=1 non-stream requests:
every other shape is a clear 400, never a silent response without logprobs.
The real ``_run_generation`` runs over faked generators (the golden-matrix
harness), so the result-dict plumbing is exercised end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_api_benchmark_contracts import (
    BYPASS,
    _envelope_client,
    _fake_generation_output,
    _scoring_state,
)
from test_server_openai import _mtp_batch_dispatch_state

from mtplx.server import openai
from mtplx.server.openai import create_app

# Engine-shaped first-token result (mtplx.generation.FirstTokenLogprobs);
# strings are whatever the fake state's tokenizer decodes these ids to.
FIRST = SimpleNamespace(
    token_id=79,
    logprob=-0.25,
    top=((79, -0.25), (75, -1.5), (80, -3.0)),
)


def _logprobs_generator(calls: list[dict]):
    base = _fake_generation_output()

    def generate(*args, **kwargs):
        calls.append(kwargs)
        out = base(*args, **kwargs)
        out.tokens = [FIRST.token_id]
        out.first_token_logprobs = FIRST
        return out

    return generate


def _decode(state, token_id: int) -> str:
    return state.runtime.tokenizer.decode([token_id])


# --- /v1/completions ---------------------------------------------------------


def test_completions_first_token_logprobs_openai_shape(monkeypatch):
    calls: list[dict] = []
    client, state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 2, "max_tokens": 1},
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    sampled = _decode(state, 79)
    assert logprobs["tokens"] == [sampled]
    assert logprobs["token_logprobs"] == [pytest.approx(-0.25)]
    assert logprobs["text_offset"] == [0]
    assert logprobs["token_ids"] == [79]
    top = logprobs["top_logprobs"][0]
    assert top[sampled] == pytest.approx(-0.25)
    assert top[_decode(state, 75)] == pytest.approx(-1.5)
    assert calls[0]["first_token_logprobs_top_k"] == 2


def test_completions_logprobs_zero_reports_sampled_token(monkeypatch):
    calls: list[dict] = []
    client, _state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 0, "max_tokens": 1},
    )

    assert response.status_code == 200
    assert calls[0]["first_token_logprobs_top_k"] == 0
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["token_logprobs"] == [pytest.approx(-0.25)]


def test_logprobs_request_skips_blank_retries(monkeypatch):
    """A whitespace/stop first token is a valid classifier answer: it must
    not trigger the unseeded blank-retry loop (up to 4 full generations)."""

    calls: list[dict] = []
    generate = _logprobs_generator(calls)

    def blank(*args, **kwargs):
        out = generate(*args, **kwargs)
        out.text = ""
        return out

    client, _state = _envelope_client(monkeypatch, generator=blank)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 1, "max_tokens": 1},
    )

    assert response.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("body_extra", "message_part"),
    [
        ({"max_tokens": 8}, "max_tokens to 1"),
        ({}, "max_tokens to 1"),
        ({"max_tokens": 1, "stream": True}, "stream"),
        ({"max_tokens": 1, "stop": ["\n"]}, "stop"),
        ({"max_tokens": 1, "logprobs": 10_000}, "MTPLX_PROMPT_LOGPROBS_MAX"),
    ],
)
def test_completions_unservable_logprobs_shapes_are_400(
    monkeypatch, body_extra, message_part
):
    def _explode(*_args, **_kwargs):
        raise AssertionError("an unservable logprobs request must not generate")

    monkeypatch.setattr(openai, "_run_generation_dispatched", _explode)
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "logprobs": 2, **body_extra},
    )

    assert response.status_code == 400
    assert message_part in response.json()["error"]["message"]


def test_completions_missing_engine_logprobs_is_loud(monkeypatch):
    """An engine lane that ran without producing logprobs is a server
    error, never a 200 that silently drops the requested field."""

    client, _state = _envelope_client(monkeypatch)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 2, "max_tokens": 1},
    )

    assert response.status_code == 500
    assert "first-token logprobs" in response.json()["error"]["message"]


# --- /v1/chat/completions ----------------------------------------------------


def test_chat_first_token_logprobs_openai_shape(monkeypatch):
    calls: list[dict] = []
    client, state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "label?"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["refusal"] is None
    [entry] = logprobs["content"]
    sampled = _decode(state, 79)
    assert entry["token"] == sampled
    assert entry["logprob"] == pytest.approx(-0.25)
    assert entry["bytes"] == list(sampled.encode("utf-8"))
    assert [alt["token"] for alt in entry["top_logprobs"]] == [
        _decode(state, token) for token, _value in FIRST.top
    ]
    assert [alt["logprob"] for alt in entry["top_logprobs"]] == [
        pytest.approx(value) for _token, value in FIRST.top
    ]
    assert calls[0]["first_token_logprobs_top_k"] == 3


def test_chat_without_logprobs_has_no_logprobs_field(monkeypatch):
    calls: list[dict] = []
    client, _state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
    )

    assert response.status_code == 200
    assert "logprobs" not in response.json()["choices"][0]
    assert calls[0]["first_token_logprobs_top_k"] is None


@pytest.mark.parametrize(
    ("body_extra", "message_part"),
    [
        ({"top_logprobs": 3}, "requires logprobs=true"),
        ({"logprobs": True}, "max_tokens to 1"),
        ({"logprobs": True, "max_completion_tokens": 4}, "max_tokens to 1"),
        ({"logprobs": True, "max_tokens": 1, "stream": True}, "stream"),
        ({"logprobs": True, "max_tokens": 1, "stop": "x"}, "stop"),
        ({"logprobs": True, "max_tokens": 1, "top_logprobs": -1}, ">= 0"),
    ],
)
def test_chat_unservable_logprobs_shapes_are_400(
    monkeypatch, body_extra, message_part
):
    def _explode(*_args, **_kwargs):
        raise AssertionError("an unservable logprobs request must not generate")

    monkeypatch.setattr(openai, "_run_generation_dispatched", _explode)
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body_extra},
    )

    assert response.status_code == 400
    assert message_part in response.json()["error"]["message"]


# --- lane routing ------------------------------------------------------------


def test_logprobs_requests_bypass_live_ar_batch(monkeypatch):
    state = _mtp_batch_dispatch_state()
    state.args.scheduler_mode = "ar_batch"
    state.ar_batch_service = SimpleNamespace(
        submit=lambda _job: pytest.fail("logprobs must not ride the AR batch")
    )
    seen: dict = {}

    def solo(*_args, **kwargs):
        seen.update(kwargs)
        return {"route": "solo"}

    monkeypatch.setattr(openai, "_run_generation", solo)

    generated = openai._run_generation_dispatched(
        state,
        [1],
        batch_key="test.logprobs",
        generation_mode="ar",
        first_token_logprobs_top_k=5,
    )

    assert generated == {"route": "solo"}
    assert seen["first_token_logprobs_top_k"] == 5
    lane = seen["request_observability"]
    assert lane["scheduler_lane"] == "solo_logprobs"
    assert lane["ar_batch_bypass_reason"] == "first_token_logprobs"


def test_mtp_batch_rejects_logprobs_without_solo_fallback(monkeypatch):
    state = _mtp_batch_dispatch_state()
    state.mtp_batch_service = SimpleNamespace(
        submit=lambda _job: pytest.fail("no submit")
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: pytest.fail("mtp_batch logprobs cannot go solo"),
    )

    with pytest.raises(openai.MTPBatchRequestError, match="does not support logprobs"):
        openai._run_generation_dispatched(
            state,
            [1],
            batch_key="test.logprobs",
            generation_mode="mtp",
            first_token_logprobs_top_k=5,
        )
