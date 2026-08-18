"""Bounded admission for generation requests (single-model serving).

MTPLX's default scheduler mode is ``serial``: one owner thread runs all model
work, deliberately, because MLX stream state is thread-affine. Concurrency is
therefore already 1 — extra requests do not run in parallel, they queue.

The queue was unbounded (``ModelScheduler`` uses a bare ``deque()`` and
``--max-active-requests`` has no default). On 2026-08-18 a memory-graph client
opened 10 concurrent chat completions against one 27B model: one executed and
nine sat in line, each holding an HTTP connection and a max-fan lease, with no
cap and no backpressure signalling anything was wrong.

This middleware bounds the line and tells over-eager clients to back off with
HTTP 429 + Retry-After, rather than accepting work the server cannot start.
"""

from __future__ import annotations

import asyncio
import json

import mtplx.server.openai as openai_mod


def _scope(path: str = "/v1/chat/completions", method: str = "POST", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }


class _Sink:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for message in self.messages:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def body(self):
        raw = b"".join(
            message.get("body", b"")
            for message in self.messages
            if message["type"] == "http.response.body"
        )
        return json.loads(raw) if raw else None

    @property
    def headers(self):
        for message in self.messages:
            if message["type"] == "http.response.start":
                return {
                    key.decode().lower(): value.decode()
                    for key, value in message.get("headers", [])
                }
        return {}


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class _BlockingApp:
    """Downstream app that parks until released, simulating a busy engine."""

    def __init__(self):
        self.release = asyncio.Event()
        self.entered = 0

    async def __call__(self, scope, receive, send):
        self.entered += 1
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _middleware(app, *, limit: int):
    return openai_mod._GenerationAdmissionMiddleware(app, max_inflight=limit)


def test_admits_up_to_the_limit_and_rejects_beyond_it():
    async def scenario():
        app = _BlockingApp()
        mw = _middleware(app, limit=2)
        parked = [
            asyncio.create_task(mw(_scope(), _receive, _Sink())) for _ in range(2)
        ]
        await asyncio.sleep(0)  # let both enter the downstream app
        assert app.entered == 2

        overflow = _Sink()
        await mw(_scope(), _receive, overflow)
        assert overflow.status == 429
        assert app.entered == 2, "rejected request must not reach the engine"
        assert "retry-after" in overflow.headers
        assert overflow.body["error"]["type"] == "server_busy"

        app.release.set()
        await asyncio.gather(*parked)

    asyncio.run(scenario())


def test_slot_is_returned_after_a_request_completes():
    async def scenario():
        app = _BlockingApp()
        app.release.set()
        mw = _middleware(app, limit=1)
        for _ in range(3):
            sink = _Sink()
            await mw(_scope(), _receive, sink)
            assert sink.status == 200
        assert app.entered == 3

    asyncio.run(scenario())


def test_non_generation_paths_are_never_gated():
    async def scenario():
        app = _BlockingApp()
        app.release.set()
        mw = _middleware(app, limit=0)
        sink = _Sink()
        await mw(_scope(path="/health", method="GET"), _receive, sink)
        assert sink.status == 200

    asyncio.run(scenario())


def test_background_task_probes_are_never_gated():
    """Open WebUI title/tag probes ride the same exemption as the fan lease."""

    async def scenario():
        app = _BlockingApp()
        app.release.set()
        mw = _middleware(app, limit=0)
        sink = _Sink()
        await mw(
            _scope(headers=[(b"x-openwebui-task", b"title")]),
            _receive,
            sink,
        )
        assert sink.status == 200

    asyncio.run(scenario())


def test_limit_of_zero_disables_admission_control():
    async def scenario():
        app = _BlockingApp()
        mw = _middleware(app, limit=0)
        parked = [
            asyncio.create_task(mw(_scope(), _receive, _Sink())) for _ in range(5)
        ]
        await asyncio.sleep(0)
        assert app.entered == 5
        app.release.set()
        await asyncio.gather(*parked)

    asyncio.run(scenario())


def test_default_limit_is_a_small_positive_number():
    assert 0 < openai_mod.MAX_INFLIGHT_GENERATION_REQUESTS <= 8
