"""Backpressure must reach the requests that are actually queued.

The dense lane bounds its own queue and answers `DenseMTPBatchQueueFull` as a
503 with `Retry-After`. Both halves work. They are simply DOWNSTREAM of where
the backlog forms: a connection uvicorn has accepted and not yet dispatched is
invisible to an in-process depth check, because the handler has not run.

Measured 2026-08-25 on the 27B at 128 concurrent against 8 slots: the guarded
queue peaked at 31 of 64 while 128 requests were outstanding. 464 issued, 288
TimeoutError, ZERO 503s. The contract promises "503 means back off and
Retry-After says how long" and under overload it delivered neither.

These tests pin the gap closed at the ASGI boundary, which is the first point
MTPLX can see a request at all.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mtplx.server.openai import _AdmissionControlMiddleware


class _Recorder:
    """Collects what the middleware sent, without a server."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return int(m["status"])
        return None

    @property
    def headers(self) -> dict[bytes, bytes]:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return dict(m.get("headers", []))
        return {}


def _scope(path: str = "/v1/chat/completions") -> dict[str, Any]:
    return {"type": "http", "path": path}


async def _noop_receive() -> dict[str, Any]:  # pragma: no cover - never awaited
    return {"type": "http.request"}


def test_over_the_limit_is_refused_with_retry_after() -> None:
    """The whole point: a refusal a client can ACT on, not a hang.

    Uvicorn's own `limit_concurrency` would also shed load, but its 503 carries
    no `Retry-After`, so it would satisfy the bound while breaking the contract.
    """
    held = asyncio.Event()

    async def slow_app(scope: Any, receive: Any, send: Any) -> None:
        await held.wait()

    mw = _AdmissionControlMiddleware(slow_app, state=None, limit=2)

    async def run() -> _Recorder:
        # Two requests occupy the limit and stay in flight.
        a = asyncio.create_task(mw(_scope(), _noop_receive, _Recorder()))
        b = asyncio.create_task(mw(_scope(), _noop_receive, _Recorder()))
        await asyncio.sleep(0)  # let both enter the middleware
        # The third arrives with no capacity left.
        rec = _Recorder()
        await mw(_scope(), _noop_receive, rec)
        held.set()
        await asyncio.gather(a, b)
        return rec

    rec = asyncio.run(run())
    assert rec.status == 503, "an over-capacity request must be refused, not queued"
    assert rec.headers.get(b"retry-after") == b"1", (
        "a 503 without Retry-After is exactly the uvicorn behaviour this "
        "middleware exists to avoid: the client cannot tell busy from broken"
    )


def test_under_the_limit_passes_through_untouched() -> None:
    seen: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["path"])

    mw = _AdmissionControlMiddleware(app, state=None, limit=4)
    asyncio.run(mw(_scope("/v1/chat/completions"), _noop_receive, _Recorder()))
    assert seen == ["/v1/chat/completions"]


def test_the_counter_releases_so_capacity_is_not_leaked() -> None:
    """A finished request must give its slot back.

    If the decrement were missed, the server would refuse everything forever
    after `limit` requests -- a far worse failure than the one being fixed.
    """
    async def app(scope: Any, receive: Any, send: Any) -> None:
        return

    mw = _AdmissionControlMiddleware(app, state=None, limit=1)
    for _ in range(5):
        rec = _Recorder()
        asyncio.run(mw(_scope(), _noop_receive, rec))
        assert rec.status is None, "sequential requests must not exhaust the limit"
    assert mw._in_flight == 0


def test_the_slot_is_released_even_when_the_app_raises() -> None:
    """An erroring handler must not permanently consume capacity."""

    async def boom(scope: Any, receive: Any, send: Any) -> None:
        raise RuntimeError("handler exploded")

    mw = _AdmissionControlMiddleware(boom, state=None, limit=1)
    for _ in range(3):
        try:
            asyncio.run(mw(_scope(), _noop_receive, _Recorder()))
        except RuntimeError:
            pass
    assert mw._in_flight == 0, "a raising handler leaked its admission slot"


def test_observability_paths_answer_while_shedding_load() -> None:
    """Refusing /metrics under load would blind the operator exactly when
    they most need to see what is happening."""
    reached: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["path"])

    mw = _AdmissionControlMiddleware(app, state=None, limit=1)
    mw._in_flight = 99  # saturated well past the limit

    for path in ("/health", "/metrics", "/v1/models"):
        rec = _Recorder()
        asyncio.run(mw(_scope(path), _noop_receive, rec))
        assert rec.status is None, f"{path} must answer while the server sheds load"
    assert reached == ["/health", "/metrics", "/v1/models"]


def test_a_zero_limit_disables_the_gate() -> None:
    """Operators must be able to turn admission control off outright."""
    reached: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["path"])

    mw = _AdmissionControlMiddleware(app, state=None, limit=0)
    mw._in_flight = 10_000
    asyncio.run(mw(_scope(), _noop_receive, _Recorder()))
    assert reached, "limit=0 must pass everything through"


def test_non_http_scopes_are_never_gated() -> None:
    """Lifespan and websocket scopes have no status line to send."""
    reached: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope["type"])

    mw = _AdmissionControlMiddleware(app, state=None, limit=1)
    mw._in_flight = 50
    asyncio.run(mw({"type": "lifespan"}, _noop_receive, _Recorder()))
    assert reached == ["lifespan"]


def test_a_streaming_response_holds_its_slot_until_the_stream_ends() -> None:
    """Chat is streamed, and a stream occupies the server for its whole life.

    The middleware wraps the entire ASGI call, so a slot is held until the last
    chunk is sent rather than until the response STARTS. That is the correct
    accounting -- an SSE stream generating 512 tokens is using a decode slot the
    whole time -- but it is worth pinning, because releasing at response-start
    would let an unbounded number of live streams accumulate behind a limit that
    believed they had finished.
    """
    started = asyncio.Event()
    finish = asyncio.Event()

    async def streamer(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"tok", "more_body": True})
        started.set()
        await finish.wait()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = _AdmissionControlMiddleware(streamer, state=None, limit=1)

    async def run() -> tuple[int | None, int]:
        task = asyncio.create_task(mw(_scope(), _noop_receive, _Recorder()))
        await started.wait()
        held = mw._in_flight  # mid-stream: the slot must still be taken
        rec = _Recorder()
        await mw(_scope(), _noop_receive, rec)  # second request, no capacity
        finish.set()
        await task
        return rec.status, held

    status, held = asyncio.run(run())
    assert held == 1, "a mid-flight stream must still occupy its slot"
    assert status == 503, (
        "capacity was released at response-start, so live streams would "
        "accumulate behind a limit that thought they had finished"
    )
    assert mw._in_flight == 0, "the slot must be released once the stream ends"
