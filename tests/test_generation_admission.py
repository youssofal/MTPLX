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


# --- Capacity-aware resolution -------------------------------------------
#
# A flat cap of 4 bounds the WAITING LINE in serial mode (1 executing + 3
# queued), which is exactly R4. Applied unconditionally it instead caps ACTIVE
# capacity: the documented mtp_batch lane seals cohorts of width 8, so four
# requests would be admitted, four would get 429, and a real width-8 cohort
# could never form. The default therefore floors at the scheduler's own
# concurrency; an explicitly configured value always wins verbatim.


def _args(**overrides):
    from types import SimpleNamespace

    base = {
        "scheduler_mode": "serial",
        "batching_preset": "latency",
        "max_active_requests": None,
        "decode_batch_max": None,
        "batch_wait_ms": None,
        "prefill_chunk_tokens": None,
        "experimental_mtp_cohorts": False,
        "max_inflight_generation_requests": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _state(**overrides):
    from types import SimpleNamespace

    lanes = overrides.pop("mtp_batch_lanes", {})
    return SimpleNamespace(args=_args(**overrides), mtp_batch_lanes=lanes)


def test_default_generation_admission_limit_is_exactly_four():
    """R4's user decision: reject fast, small depth, default 4."""
    assert openai_mod.MAX_INFLIGHT_GENERATION_REQUESTS == 4


def test_serial_mode_resolves_to_the_documented_default_of_four():
    assert openai_mod._resolve_generation_admission_limit(_state()) == 4


def test_mtp_batch_mode_admits_a_full_width_eight_cohort():
    """Four admitted + four 429 can never seal a width-8 cohort."""
    limit = openai_mod._resolve_generation_admission_limit(
        _state(scheduler_mode="mtp_batch", mtp_batch_lanes={3: object(), 8: object()})
    )
    assert limit >= 8


def test_mtp_batch_mode_without_installed_lanes_still_admits_the_documented_width():
    limit = openai_mod._resolve_generation_admission_limit(
        _state(scheduler_mode="mtp_batch")
    )
    assert limit >= 8


def test_ar_batch_throughput_mode_admits_its_full_decode_batch():
    limit = openai_mod._resolve_generation_admission_limit(
        _state(scheduler_mode="ar_batch", batching_preset="throughput")
    )
    assert limit >= 8


def test_an_explicit_limit_is_honoured_verbatim_even_below_batch_width():
    """An operator who asks for 2 gets 2; the floor only lifts the default."""
    limit = openai_mod._resolve_generation_admission_limit(
        _state(scheduler_mode="mtp_batch", max_inflight_generation_requests=2)
    )
    assert limit == 2


def test_an_explicit_zero_disables_admission_control():
    limit = openai_mod._resolve_generation_admission_limit(
        _state(scheduler_mode="mtp_batch", max_inflight_generation_requests=0)
    )
    assert limit == 0


# --- Public CLI plumbing --------------------------------------------------


def test_public_serve_parser_exposes_the_admission_flag():
    """The override is worthless if `mtplx serve` cannot pass it."""
    import mtplx.cli as cli_mod

    parser = cli_mod.build_parser()
    args = parser.parse_args(
        ["serve", "--max-inflight-generation-requests", "12"]
    )
    assert args.max_inflight_generation_requests == 12


def test_public_serve_parser_leaves_the_limit_unset_by_default():
    """Unset must stay None so the capacity-aware default can apply."""
    import mtplx.cli as cli_mod

    args = cli_mod.build_parser().parse_args(["serve"])
    assert getattr(args, "max_inflight_generation_requests", "missing") is None


def test_public_child_argument_forwarding_includes_the_admission_flag():
    """The public command spawns the server as a child process; an option it
    does not forward silently does nothing."""
    import inspect

    import mtplx.commands.public as public_mod

    source = inspect.getsource(public_mod)
    assert source.count(
        '("max_inflight_generation_requests", "--max-inflight-generation-requests")'
    ) >= 2
