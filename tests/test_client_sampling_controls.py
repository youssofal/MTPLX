"""logit_bias, min_p and repetition_penalty reach the sampler and are applied.

Until 2.13 the request models accepted these fields (``extra="allow"``) and the server then
built a sampler that could not carry them, so a client suppressing a token could still
receive it. They now travel in ``SamplerConfig``; a malformed value is a 400 naming the
field, a neutral value changes nothing, and the draws that serve them apply them
(``ClientControlledDraw``: the shared AR loop and Inkling's host decision).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.generation import ClientControlledDraw
from mtplx.sampling import SamplerConfig, SparseDistribution, client_controls, min_p_filter, parse_client_controls
from mtplx.server import openai
from mtplx.server.openai import _generation_params, create_app
from tests.test_penalty_request_wiring import _params_state
from tests.test_server_openai import _fake_state

ALLOW = {"x-mtplx-cache-mode": "bypass", "x-mtplx-allow-client-controls": "1"}


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _capture(captured):
    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["client_controls"] = kwargs.get("client_controls")
        return {"text": "ok", "tokens": [4], "stats": {"completion_tokens": 1},
                "prompt_tokens": len(prompt_ids), "completion_tokens": 1, "finish_reason": "stop"}

    return fake_run_generation


def _post(monkeypatch, path, body, headers=ALLOW):
    captured: dict = {}
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(openai, "_encode_prompt", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", _capture(captured))
    return client.post(path, headers=headers, json=body), captured


def test_chat_controls_reach_the_generation_call(monkeypatch):
    response, captured = _post(monkeypatch, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
        "logit_bias": {"5": -100, "07": 3.5}, "min_p": 0.1, "repetition_penalty": 1.2,
    })
    assert response.status_code == 200, response.text
    assert captured["client_controls"] == {
        "logit_bias": ((5, -100.0), (7, 3.5)), "min_p": 0.1, "repetition_penalty": 1.2}


def test_completion_controls_reach_the_generation_call(monkeypatch):
    response, captured = _post(monkeypatch, "/v1/completions",
                               {"prompt": "x", "max_tokens": 4, "repetition_penalty": 1.3})
    assert response.status_code == 200, response.text
    assert captured["client_controls"] == {"repetition_penalty": 1.3}


@pytest.mark.parametrize("field,value", [
    ("logit_bias", {"abc": 1}), ("logit_bias", {"5": 101}), ("logit_bias", [5]),
    ("min_p", 1.5), ("min_p", "x"), ("repetition_penalty", 0), ("repetition_penalty", -1.0),
])
def test_malformed_control_is_a_400_naming_the_field(monkeypatch, field, value):
    response, captured = _post(monkeypatch, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, field: value})
    assert response.status_code == 400 and field in response.text
    assert "client_controls" not in captured


def test_generation_params_build_the_sampler_with_the_controls():
    controls = {"logit_bias": ((5, -100.0),), "min_p": 0.1, "repetition_penalty": 1.2}
    _max, sampler, _limits = _generation_params(_params_state(), prompt_token_count=10, max_tokens=None,
                                                temperature=None, top_p=None, top_k=None,
                                                client_controls=controls)
    assert (sampler.logit_bias, sampler.min_p, sampler.repetition_penalty) == ((((5, -100.0),)), 0.1, 1.2)
    assert client_controls(sampler) == ("logit_bias", "min_p", "repetition_penalty")


def test_neutral_values_change_nothing():
    fields = parse_client_controls(logit_bias={"3": 0}, min_p=0, repetition_penalty=1.0)
    assert client_controls(SamplerConfig(**fields)) == ()


def _row(values):
    return mx.array(np.asarray(values, dtype=np.float32))


def test_logit_bias_forces_and_suppresses_tokens():
    greedy = SamplerConfig(temperature=0.0, logit_bias=((3, 100.0),))
    assert ClientControlledDraw(greedy, []).draw(_row([5.0, 1.0, 0.0, 0.0]), np.random.default_rng(0)) == 3
    sampled = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, logit_bias=((0, -100.0),))
    draw = ClientControlledDraw(sampled, [])
    rng = np.random.default_rng(1)
    assert all(draw.draw(_row([5.0, 4.0, 3.0, 0.0]), rng) != 0 for _ in range(200))


def test_repetition_penalty_covers_the_prompt_and_the_completion():
    config = SamplerConfig(temperature=0.0, repetition_penalty=2.0)
    draw = ClientControlledDraw(config, [0])
    row = _row([2.0, 1.9, -1.0, 1.5])
    # token 0 (in the prompt): 2.0 / 2 = 1.0 < 1.9, so the argmax moves to token 1
    assert draw.draw(row, np.random.default_rng(0)) == 1
    # token 1 is now in the completion: 1.9 / 2 = 0.95 < 1.5 (token 3)
    assert draw.draw(row, np.random.default_rng(0)) == 3
    logits = np.asarray(draw.logits(row))
    np.testing.assert_allclose(logits, [1.0, 0.95, -1.0, 0.75])


def test_min_p_drops_tokens_below_the_relative_floor():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    out = min_p_filter(probs, 0.2)  # floor 0.1: the 0.05 token goes
    np.testing.assert_allclose(out, np.array([0.5, 0.3, 0.15, 0.0]) / 0.95)
    sparse = min_p_filter(SparseDistribution(np.array([4, 9, 2]), np.array([0.6, 0.35, 0.05]), 10), 0.1)
    assert sparse.token_ids.tolist() == [4, 9] and abs(sparse.probs.sum() - 1.0) < 1e-12
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.2)
    draw = ClientControlledDraw(config, [])
    rng = np.random.default_rng(2)
    row = _row(np.log([0.5, 0.3, 0.15, 0.05]))
    assert {draw.draw(row, rng) for _ in range(400)} == {0, 1, 2}
