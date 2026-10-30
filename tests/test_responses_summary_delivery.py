"""Codex sends reasoning_summary_delivery in reasoning and stream_options (#658).

MTPLX writes no reasoning summary item, so the hint is met as sent; refusing
it answered every turn of Codex desktop build 26.1016 with a 400. The request
body is the one in the #658 report.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from tests.test_server_openai import (
    CaptureTokenizer,
    _fake_state,
    _fake_streaming_generation,
)


def test_the_codex_desktop_request_with_summary_delivery_streams(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("hi"))

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "input": "Say hi",
            "stream": True,
            "max_output_tokens": 16,
            "stream_options": {"reasoning_summary_delivery": "auto"},
            "reasoning": {
                "effort": "low",
                "summary": "auto",
                "reasoning_summary_delivery": "auto",
            },
        },
    )

    assert response.status_code == 200, response.text
    assert "event: response.completed" in response.text


@pytest.mark.parametrize("field", ["stream_options", "reasoning"])
def test_a_summary_delivery_that_is_not_a_string_fails_precisely(field):
    body = {"input": "hi", field: {"reasoning_summary_delivery": {"mode": "auto"}}}
    response = TestClient(create_app(_fake_state())).post("/v1/responses", json=body)

    assert response.status_code == 400
    assert (
        f"{field}.reasoning_summary_delivery must be a string"
        in response.json()["error"]["message"]
    )
