"""A Metal buffer-count failure is answered by letting go of arrays (#640).

The field report (2.12.2, Flash-Next, Hermes Agent, 51 hours up): every
request failed with ``[metal::malloc] Resource limit (499000) exceeded.``
1,124 times in a row, each followed by the allocation-failure shed with
``bank_entries_evicted: 0`` and ``bank_bytes_after: 7130505904``, until a
restart. MLX refuses a new buffer once 499,000 are alive, whatever memory is
free (the Mac had 98% free), and it has already released its cache of free
buffers before it raises. The shed halved the bank's byte budget, which a
bank under that half (7.1 GB) answers by evicting nothing, then cleared the
allocator cache that was already empty: the next request met the same count.

A count failure now releases the saved state of every conversation that is
not generating (``release_idle_sessions``: the ones already on the SSD first),
never a session in flight, and the 507 says what happened. A byte failure
keeps its byte target. The allocator here is a double that counts the
buffers the bank's entries hold on top of a fixed baseline, the way MLX
counts every live buffer object.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
from mtplx.engine_session import EngineSessionManager
from mtplx.errors import is_allocation_failure, is_resource_count_failure
from mtplx.session_bank import SessionBank
from tests.test_memguard_admission import RUNTIME

GIB = 1024**3
COUNT_ERROR = "[metal::malloc] Resource limit (499000) exceeded."
LIMIT = 499_000
BUFFERS_PER_ENTRY = 750  # a Flash-Next entry: 48 layers' states and 8 checkpoints


class _MetalBuffers:
    """MLX's live-buffer count: weights, the request in flight and anything
    else the engine holds (``baseline``), plus the bank's entries."""

    def __init__(self, *, baseline: int) -> None:
        self.baseline = int(baseline)
        self.entries: list[weakref.ref] = []

    def hold(self, entry) -> None:
        self.entries.append(weakref.ref(entry))

    def live(self) -> int:
        gc.collect()
        alive = sum(1 for ref in self.entries if ref() is not None)
        return self.baseline + BUFFERS_PER_ENTRY * alive

    def allocate(self) -> None:
        """One new buffer, refused at the limit as MLX refuses it."""

        if self.live() >= LIMIT:
            raise RuntimeError(COUNT_ERROR)


def _put(bank, metal, tokens, *, session_id, nbytes) -> tuple[int, ...]:
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=int(nbytes),
    )
    assert entry is not None
    metal.hold(entry)
    return entry.token_ids


def _engine(*, idle_sessions: int = 3):
    """The #640 shape: a 16 GiB bank holding 7 GiB, under the byte shed's
    half-budget target, with one conversation generating and the rest idle,
    and the buffer count at the limit."""

    manager = EngineSessionManager(
        bank=SessionBank(max_entries=48, max_bytes=16 * GIB, per_session_max_bytes=16 * GIB),
        idle_ttl_s=3600,
    )
    metal = _MetalBuffers(baseline=0)
    live_key = _put(manager.bank, metal, range(1, 1001), session_id="hermes", nbytes=1 * GIB)
    idle_keys = [
        _put(
            manager.bank,
            metal,
            range(10_000 * (i + 1), 10_000 * (i + 1) + 1000),
            session_id=f"anon-{i}",
            nbytes=2 * GIB,
        )
        for i in range(idle_sessions)
    ]
    metal.baseline = LIMIT - BUFFERS_PER_ENTRY * (1 + idle_sessions)
    assert metal.live() == LIMIT
    generating = manager.get_or_create("hermes")
    assert generating.try_begin_generation()
    state = SimpleNamespace(
        sessions=manager,
        dashboard=SimpleNamespace(),
        memory_plan=None,
        allow_swap=False,
    )
    return state, manager, metal, generating, live_key, idle_keys


def _request(state, metal):
    """What a request does at the wall: its allocation fails and the server
    answers with the allocation-failure 507 (the stream and non-stream
    paths both call ``_allocation_failure_http_exception``)."""

    try:
        metal.allocate()
    except RuntimeError as exc:
        assert is_allocation_failure(exc)
        return srv._allocation_failure_http_exception(state, exc)
    return None


@pytest.fixture(autouse=True)
def _no_mlx_cache(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(mx, "clear_cache", lambda: None)


def test_the_count_message_is_told_apart_from_a_memory_one():
    assert is_resource_count_failure(RuntimeError(COUNT_ERROR))
    assert not is_resource_count_failure(
        RuntimeError("[metal::malloc] Attempting to allocate 9 bytes which is "
                     "greater than the maximum allowed buffer size")
    )
    assert not is_resource_count_failure(RuntimeError("Insufficient Memory"))
    assert not is_resource_count_failure(ValueError(COUNT_ERROR))


def test_a_count_failure_releases_the_idle_conversations_and_the_next_request_runs():
    state, manager, metal, generating, live_key, idle_keys = _engine()
    try:
        refused = _request(state, metal)
        assert refused is not None and refused.status_code == 507
        # The next request finds room: the count fell under the limit.
        assert _request(state, metal) is None
        assert metal.live() < LIMIT
    finally:
        generating.end_generation()
    # The conversation generating keeps its state; the idle ones gave theirs.
    assert live_key in manager.bank._entries
    assert not any(key in manager.bank._entries for key in idle_keys)
    shed = state.dashboard.memory_guard_events[-1]
    assert shed["action"] == "resource_count_shed"
    assert shed["released_entries"] == len(idle_keys)
    assert shed["released_sessions"] == ["anon-0", "anon-1", "anon-2"]
    assert shed["bank_entries_after"] == 1


def test_the_507_says_it_is_a_buffer_count_and_not_memory():
    state, _manager, metal, generating, _live, _idle = _engine()
    try:
        refused = _request(state, metal)
    finally:
        generating.end_generation()
    detail = refused.detail
    assert "Resource limit (499000)" in detail
    assert "not a shortage of memory" in detail
    assert "3 idle conversations" in detail
    # The advice for a memory shortfall would send the user the wrong way.
    assert "context-window" not in detail


def test_with_nothing_idle_left_the_507_says_a_restart_recovers():
    state, _manager, metal, generating, _live, _idle = _engine(idle_sessions=0)
    try:
        refused = _request(state, metal)
        # Nothing the engine may release holds the buffers: the next request
        # meets the same count, and the answer says so.
        assert _request(state, metal) is not None
    finally:
        generating.end_generation()
    assert "restart" in refused.detail
    assert state.dashboard.memory_guard_events[-1]["released_entries"] == 0


def test_a_memory_failure_keeps_its_byte_target():
    """A byte shortfall is not a count one: the idle conversations under the
    half-budget target stay, as before."""

    state, manager, _metal, generating, live_key, idle_keys = _engine()
    try:
        refused = srv._allocation_failure_http_exception(
            state, RuntimeError("[metal::malloc] Insufficient Memory")
        )
    finally:
        generating.end_generation()
    assert refused.status_code == 507
    assert all(key in manager.bank._entries for key in idle_keys + [live_key])
    assert state.dashboard.memory_guard_events[-1]["action"] == "allocation_failure_shed"
