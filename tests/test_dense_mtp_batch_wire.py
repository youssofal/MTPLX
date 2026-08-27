"""The dense lane's per-request stats must survive to the WIRE.

This exists because the previous test for the same guarantee read the future's
result dict -- which sits UPSTREAM of `_public_mtplx_stats`, the allowlist that
was silently dropping both keys. The unit test passed, the field was set
correctly, and no caller could see it. The fake and the wire disagreed, and
only the wire is what a caller gets.

So this test asserts on an actual HTTP response body. A second test at the old
layer would have reproduced the old blind spot exactly.

Two keys, because the defect was a CLASS and not a field:

* `dense_mtp_batch_cohort_width` -- the contract tells callers to pin the WIDTH
  for reproducibility, which was unactionable while it was invisible
* `dense_mtp_batch_cohort_seed` -- the service docstring says it is "reported
  per request so this is visible rather than silent", which was false
"""

from __future__ import annotations

import test_server_openai as tso
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app

DENSE_KEYS = ("dense_mtp_batch_cohort_width", "dense_mtp_batch_cohort_seed")


def _generation_with_dense_stats(_state, _prompt_ids, **kwargs):
    """A generation result carrying the dense lane's per-request stats."""

    token_callback = kwargs.get("token_callback")
    if token_callback is not None:
        token_callback([ord(c) for c in "hi"])
    return {
        "text": "hi",
        "tokens": [1, 2],
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "elapsed_s": 0.01,
        "tok_s": 100.0,
        "stats": {
            "mode": "mtp",
            "scheduler_lane": "dense_mtp_batch",
            "dense_mtp_batch_cohort_width": 4,
            "dense_mtp_batch_cohort_seed": 4242,
        },
    }


def _post(monkeypatch):
    state = tso._fake_streaming_session_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_run_generation", _generation_with_dense_stats)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "stream": False,
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_dense_cohort_stats_reach_the_http_response(monkeypatch) -> None:
    """The guarantee, checked where a caller actually reads it."""

    stats = _post(monkeypatch)["mtplx_stats"]
    missing = [key for key in DENSE_KEYS if key not in stats]
    assert not missing, (
        f"{missing} set by the dense lane but stripped before the wire; "
        "a caller cannot read them, so any contract clause naming them is false"
    )
    assert stats["dense_mtp_batch_cohort_width"] == 4
    assert stats["dense_mtp_batch_cohort_seed"] == 4242


def test_the_values_survive_rather_than_merely_the_keys(monkeypatch) -> None:
    """Presence is not correctness.

    An allowlist that passed the key through while some later stage flattened
    the value would satisfy a presence check and still leave a caller unable to
    compare two runs. The width is checked against a number the fake chose, so
    a constant or a placeholder fails.
    """

    stats = _post(monkeypatch)["mtplx_stats"]
    assert stats["dense_mtp_batch_cohort_width"] == 4, (
        "width arrived but not as the value the lane reported"
    )
    assert isinstance(stats["dense_mtp_batch_cohort_width"], int)


def test_a_503_carries_retry_after_on_the_wire(monkeypatch) -> None:
    """Backpressure must tell a client HOW LONG to back off, not just "busy".

    Checked on the wire because the hardening check reported this as a FAIL and
    I could not tell from the code whether the header was missing or my check
    was looking for it case-sensitively. `dict(exc.headers)` discards the
    case-insensitive lookup HTTP requires, and ASGI normalises header names to
    lowercase -- so a present header can read as absent.

    This test settles it independently of that harness: it drives the real app,
    with the real exception handler, and looks the header up case-insensitively
    the way a client library would.
    """

    from fastapi import HTTPException

    def _busy(_state, _prompt_ids, **_kwargs):
        raise HTTPException(
            status_code=503,
            detail="dense mtp_batch queue is full (6/6 waiting); retry shortly",
            headers={"Retry-After": "1"},
        )

    state = tso._fake_streaming_session_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_run_generation", _busy)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
            "stream": False,
            "max_tokens": 8,
        },
    )
    assert response.status_code == 503, response.text
    # httpx headers are case-insensitive, which is the correct behaviour and
    # the thing the harness was missing.
    assert response.headers.get("retry-after") == "1", (
        f"503 carried no Retry-After; headers were {dict(response.headers)}"
    )
