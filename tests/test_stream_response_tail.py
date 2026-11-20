"""A named session's streamed turn ends at its last token.

The terminal frame and ``[DONE]`` used to wait for the session commit (the
history re-render and encode, the bank write, the prompt-prefix commit and
the idle postcommit's scheduling): 18 ms at the median and 160 ms at p90 in
the flight logs, up to 30 s when another client's prefill held the model
thread (#425), all of it between the client's last token and its next tool
call. The frame now goes out first and the stream worker commits after it,
holding the session's generation slot until the commit landed.

Adapted from PR #557 (@jvmenen) without its admission barrier: only the next
request of the same session waits for the commit, before it reads the
session, so it reads exactly the state it read when the frame waited. Other
sessions never wait. A session found by prompt inference keeps the frame
after the commit, because its next turn finds it through the commit.

Everything here runs on the fake streaming session state: no model, CPU only.
"""

from __future__ import annotations

import threading
import time
from threading import Event

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.engine_session import EngineSession
from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    _fake_final_state,
    _fake_streaming_session_state,
    _stream_payloads,
)

TURN1 = [{"role": "user", "content": "Say OK"}]
TURN2 = [
    {"role": "user", "content": "Say OK"},
    {"role": "assistant", "content": "OK"},
    {"role": "user", "content": "Again"},
]


@pytest.fixture(autouse=True)
def _frame_before_commit(monkeypatch):
    monkeypatch.delenv("MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT", raising=False)


class _Engine:
    """Fake generation plus the two commit halves. With ``stored`` False the
    generation-final store refuses (the tool-call shape) and the turn takes
    the released path: prompt-prefix commit, then the idle postcommit's
    scheduling. ``hold`` names the commit step that waits on ``gate`` for
    the session ``held_session``: "schedule" (on the stream worker, after
    the prompt-prefix commit) or "store" (the generation-final store on the
    model thread, before ``session.commit``)."""

    def __init__(
        self,
        *,
        stored: bool = False,
        hold: str | None = "schedule",
        held_session: str | None = None,
        store_error: BaseException | None = None,
    ) -> None:
        self.stored = stored
        self.hold = hold
        self.held_session = held_session
        self.store_error = store_error
        self.gate = Event()
        self.held = Event()
        self.scheduled: list[dict] = []
        self.generations: list[dict] = []

    def _maybe_hold(self, step: str, session_id: str | None) -> None:
        if self.hold != step:
            return
        if self.held_session is not None and session_id != self.held_session:
            return
        self.held.set()
        assert self.gate.wait(5.0), "the test never released the commit"

    def run_generation(self, _state, prompt_ids, **kwargs):
        observability = dict(kwargs["request_observability"])
        self.generations.append(
            {"prompt_ids": list(prompt_ids), "session_id": kwargs.get("session_id")}
        )
        # The real generation writes the request's metrics row.
        _state.last_metrics.append({"request_id": observability["request_id"]})
        tokens = [ord("O"), ord("K")]
        callback = kwargs.get("token_callback")
        if callback is not None:
            callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                **observability,
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    def store_generation_final(self, *_args, **kwargs):
        self._maybe_hold("store", kwargs.get("session_id"))
        if self.store_error is not None:
            raise self.store_error
        if self.stored:
            return {
                "stored": True,
                "mode": "generation_final_exact",
                "reason": "compatible",
                "prefix_len": 3,
                "nbytes": 123,
            }
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "retokenized_history_mismatch",
        }

    def schedule(self, _state, **kwargs):
        self._maybe_hold("schedule", kwargs.get("session_id"))
        self.scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(openai, "_run_generation", self.run_generation)
        monkeypatch.setattr(
            openai,
            "_store_generation_final_history_snapshot",
            self.store_generation_final,
        )
        monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", self.schedule)


def _post(client, session_id, messages, *, stream):
    headers = {"x-mtplx-session-id": session_id} if session_id else {}
    return client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "messages": messages,
            "enable_thinking": False,
            "stream": stream,
            "max_tokens": 4,
        },
    )


def _final_stats(response_text: str) -> dict:
    final = [
        payload
        for payload in _stream_payloads(response_text)
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    return final[-1]["mtplx_stats"]


def _session_view(session: EngineSession, engine: _Engine) -> dict:
    return {
        "committed_token_ids": tuple(session.committed_token_ids),
        "revision": int(session.revision),
        "scheduled": len(engine.scheduled),
    }


def test_the_terminal_frame_goes_out_before_a_slow_commit(monkeypatch):
    engine = _Engine()
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        try:
            response = _post(client, "tail-frame", TURN1, stream=True)
            # The client has the whole turn while the commit is still held.
            assert engine.held.wait(5.0)
            assert not engine.scheduled
        finally:
            engine.gate.set()
        session = state.sessions.peek("tail-frame")
        session.wait_for_response_tail(5.0)
        outcome = session.last_response_tail

    assert response.status_code == 200
    assert response.text.rstrip().endswith("data: [DONE]")
    assert _final_stats(response.text)["session_postcommit_snapshot"] == {
        "stored": None,
        "mode": "after_response",
        "reason": "terminal_frame_before_commit",
    }
    # The commit ran in full after the frame.
    assert outcome["session_postcommit_snapshot"]["mode"] == "async_pending"
    assert (
        outcome["session_prompt_prefix_commit"]["boundary_kind"]
        == "postcommit_prompt_prefix"
    )
    assert len(engine.scheduled) == 1


@pytest.mark.parametrize(
    "stored, hold",
    [
        # Released path, held after the prompt-prefix commit: what is
        # missing is the idle postcommit's scheduling.
        (False, "schedule"),
        # Stored path, held in the generation-final store: what is missing
        # is the committed stream and its revision.
        (True, "store"),
    ],
)
def test_the_next_turn_reads_the_session_the_commit_left(monkeypatch, stored, hold):
    """An immediate next turn of the same session waits for the commit
    before it reads the session. What it reads (committed stream, revision,
    idle postcommit scheduled) is the state the commit left, the state it
    read when the frame waited for the commit, never the half-committed
    state the early frame left behind."""

    engine = _Engine(stored=stored, hold=hold, held_session="tail-next")
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()
    tail_waiting = Event()
    prologue_reads: list[dict] = []
    landed: list[dict] = []

    original_wait = EngineSession.wait_for_response_tail
    original_resolve = EngineSession.resolve_pending_postcommit_for_request
    original_end = EngineSession.end_response_tail

    def observed_wait(self, timeout_s):
        with self._postcommit_lock:
            running = self._response_tail is not None
        if running:
            tail_waiting.set()
        return original_wait(self, timeout_s)

    def observed_resolve(self):
        prologue_reads.append(_session_view(self, engine))
        return original_resolve(self)

    def observed_end(self, tail, outcome):
        landed.append(_session_view(self, engine))
        return original_end(self, tail, outcome)

    monkeypatch.setattr(EngineSession, "wait_for_response_tail", observed_wait)
    monkeypatch.setattr(
        EngineSession, "resolve_pending_postcommit_for_request", observed_resolve
    )
    monkeypatch.setattr(EngineSession, "end_response_tail", observed_end)

    def release_once_waiting() -> None:
        if tail_waiting.wait(5.0):
            time.sleep(0.2)
        engine.gate.set()

    with TestClient(create_app(state)) as client:
        try:
            first = _post(client, "tail-next", TURN1, stream=True)
            assert engine.held.wait(5.0)
            before = _session_view(state.sessions.peek("tail-next"), engine)
            threading.Thread(target=release_once_waiting, daemon=True).start()
            second = _post(client, "tail-next", TURN2, stream=False)
        finally:
            engine.gate.set()

    assert first.status_code == 200
    assert '"after_response"' in first.text
    assert second.status_code == 200, second.text
    assert tail_waiting.is_set()
    turn2_read = prologue_reads[-1]
    assert turn2_read == landed[0]
    assert turn2_read != before
    first_prompt = tuple(engine.generations[0]["prompt_ids"])
    if stored:
        assert before["revision"] == 0 and before["committed_token_ids"] == ()
        assert turn2_read["revision"] == 1
        assert turn2_read["committed_token_ids"] == first_prompt + (
            ord("O"),
            ord("K"),
        )
    else:
        assert before["scheduled"] == 0
        assert turn2_read["scheduled"] == 1
        committed = turn2_read["committed_token_ids"]
        assert committed and first_prompt[: len(committed)] == committed
    wait = second.json()["mtplx_stats"]["response_tail_wait"]
    assert wait["finished"] is True
    assert wait["waited_s"] >= 0.15
    snapshot = wait["tail"]["session_postcommit_snapshot"]
    assert snapshot["mode"] == ("generation_final_exact" if stored else "async_pending")


@pytest.mark.parametrize("stored, hold", [(False, "schedule"), (True, "store")])
def test_a_commit_longer_than_one_wait_round_is_still_waited_for(
    monkeypatch, stored, hold
):
    """The review of 4c9da1ba: with the commit held past the wait's bound
    (another client's long prefill ahead of it on the model thread), the
    next turn stopped waiting and read the older frontier. It now waits
    until the commit landed, in as many rounds as that takes."""

    monkeypatch.setattr(openai, "STREAM_COMMIT_WAIT_MAX_S", 0.05)
    engine = _Engine(stored=stored, hold=hold, held_session="tail-long")
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()
    tail_waiting = Event()
    prologue_reads: list[dict] = []
    landed: list[dict] = []

    original_wait = EngineSession.wait_for_response_tail
    original_resolve = EngineSession.resolve_pending_postcommit_for_request
    original_end = EngineSession.end_response_tail

    def observed_wait(self, timeout_s):
        with self._postcommit_lock:
            running = self._response_tail is not None
        if running:
            tail_waiting.set()
        return original_wait(self, timeout_s)

    def observed_resolve(self):
        prologue_reads.append(_session_view(self, engine))
        return original_resolve(self)

    def observed_end(self, tail, outcome):
        landed.append(_session_view(self, engine))
        return original_end(self, tail, outcome)

    monkeypatch.setattr(EngineSession, "wait_for_response_tail", observed_wait)
    monkeypatch.setattr(
        EngineSession, "resolve_pending_postcommit_for_request", observed_resolve
    )
    monkeypatch.setattr(EngineSession, "end_response_tail", observed_end)

    def release_after_several_rounds() -> None:
        if tail_waiting.wait(5.0):
            time.sleep(0.5)
        engine.gate.set()

    with TestClient(create_app(state)) as client:
        try:
            first = _post(client, "tail-long", TURN1, stream=True)
            assert engine.held.wait(5.0)
            before = _session_view(state.sessions.peek("tail-long"), engine)
            threading.Thread(target=release_after_several_rounds, daemon=True).start()
            second = _post(client, "tail-long", TURN2, stream=False)
        finally:
            engine.gate.set()

    assert first.status_code == 200
    assert second.status_code == 200, second.text
    # The next turn read the session once, after the commit landed.
    assert prologue_reads[-1] == landed[0]
    assert prologue_reads[-1] != before
    wait = second.json()["mtplx_stats"]["response_tail_wait"]
    assert wait["finished"] is True
    assert wait["rounds"] >= 3
    assert wait["waited_s"] >= 0.4
    snapshot = wait["tail"]["session_postcommit_snapshot"]
    assert snapshot["mode"] == ("generation_final_exact" if stored else "async_pending")


def test_other_sessions_never_wait_for_the_commit(monkeypatch):
    engine = _Engine(held_session="tail-owner")
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        try:
            first = _post(client, "tail-owner", TURN1, stream=True)
            assert engine.held.wait(5.0)
            other = _post(client, "tail-other", TURN1, stream=False)
            # Served while the first session's commit is still held.
            still_held = not engine.gate.is_set()
            held_tail = state.sessions.peek("tail-owner").wait_for_response_tail(0.0)
        finally:
            engine.gate.set()
        state.sessions.peek("tail-owner").wait_for_response_tail(5.0)

    assert first.status_code == 200
    assert other.status_code == 200, other.text
    assert still_held
    assert held_tail is not None and held_tail["finished"] is False
    assert "response_tail_wait" not in other.json()["mtplx_stats"]


def test_a_session_found_by_prompt_inference_keeps_the_frame_after_the_commit(
    monkeypatch,
):
    # Its next turn finds it through the committed stream, which must have
    # landed before the client can send that turn.
    engine = _Engine(stored=True, hold=None)
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        response = _post(client, None, TURN1, stream=True)

    assert response.status_code == 200
    snapshot = _final_stats(response.text)["session_postcommit_snapshot"]
    assert snapshot["mode"] == "generation_final_exact"
    assert '"after_response"' not in response.text


@pytest.mark.parametrize(
    "source, named",
    [
        ("header.x-mtplx-session-id", True),
        ("header.x-session-affinity", True),
        ("metadata.session_id", True),
        ("user", True),
        ("chat_id", True),
        ("conversation_id", True),
        ("new", False),
        ("longest_prefix", False),
        ("pending_postcommit_near_prefix", False),
        ("common_prefix_reuse", False),
        ("vision_bank_prefix", False),
        (None, False),
    ],
)
def test_only_a_session_the_client_named_takes_the_early_frame(source, named):
    assert openai._session_named_by_client(source) is named


def test_the_commit_outcome_lands_in_the_requests_own_metrics_row(monkeypatch):
    engine = _Engine(stored=True, hold=None)
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    def row_for(request_id: str) -> dict:
        for metric in state.last_metrics:
            if metric.get("request_id") == request_id:
                return metric
        return {}

    with TestClient(create_app(state)) as client:
        response = _post(client, "tail-row", TURN1, stream=True)
        state.sessions.peek("tail-row").wait_for_response_tail(5.0)
        request_id = _stream_payloads(response.text)[0]["id"]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            snapshot = row_for(request_id).get("session_postcommit_snapshot") or {}
            if snapshot.get("mode") == "generation_final_exact":
                break
            time.sleep(0.02)
        row = dict(row_for(request_id))

    assert response.status_code == 200
    assert '"after_response"' in response.text
    # The frame's placeholder never overwrites the real outcome.
    assert row["session_postcommit_snapshot"]["mode"] == "generation_final_exact"
    assert row["session_postcommit_snapshot"]["stored"] is True


def test_the_switch_restores_the_frame_after_the_commit(monkeypatch):
    monkeypatch.setenv("MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT", "0")
    engine = _Engine(stored=True, hold=None)
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        response = _post(client, "tail-switch", TURN1, stream=True)

    assert response.status_code == 200
    assert _final_stats(response.text)["session_postcommit_snapshot"]["mode"] == (
        "generation_final_exact"
    )
    assert state.sessions.peek("tail-switch").last_response_tail is None


# --- A commit that fails ----------------------------------------------------

FAILED_COMMIT = {
    "stored": False,
    "mode": "failed",
    "reason": "commit_error",
    "error": "UnboundLocalError: inherited_loader",
}


def _failing_engine() -> _Engine:
    return _Engine(hold=None, store_error=UnboundLocalError("inherited_loader"))


def _error_payloads(response_text: str) -> list[dict]:
    return [p for p in _stream_payloads(response_text) if "error" in p]


@pytest.mark.parametrize(
    "session_id, frame_before_commit",
    [
        # A session found by prompt inference: its frame waits for the commit.
        (None, "1"),
        # A named session with the frame held back by the switch.
        ("tail-failed", "0"),
    ],
)
def test_a_failed_commit_does_not_fail_the_finished_answer(
    monkeypatch, session_id, frame_before_commit
):
    """#696: the bank write raised after the last token had been streamed, and
    the client was sent a stream error for an answer it already held. The
    stream ends as the finished answer it is; the request's stats say the
    commit failed."""

    monkeypatch.setenv("MTPLX_STREAM_TERMINAL_FRAME_BEFORE_COMMIT", frame_before_commit)
    engine = _failing_engine()
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        response = _post(client, session_id, TURN1, stream=True)

    assert response.status_code == 200
    assert not _error_payloads(response.text)
    assert response.text.rstrip().endswith("data: [DONE]")
    assert _final_stats(response.text)["session_postcommit_snapshot"] == FAILED_COMMIT


def test_a_commit_that_fails_after_the_frame_lands_in_the_row(monkeypatch):
    """With the frame already out, the failure used to stay invisible: the
    row kept the frame's placeholder. It lands like any other outcome."""

    engine = _failing_engine()
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        response = _post(client, "tail-failed-late", TURN1, stream=True)
        session = state.sessions.peek("tail-failed-late")
        session.wait_for_response_tail(5.0)
        request_id = _stream_payloads(response.text)[0]["id"]
        deadline = time.monotonic() + 5.0
        row: dict = {}
        while time.monotonic() < deadline and row != FAILED_COMMIT:
            row = next(
                (
                    metric.get("session_postcommit_snapshot") or {}
                    for metric in state.last_metrics
                    if metric.get("request_id") == request_id
                ),
                {},
            )
            time.sleep(0.02)

    assert not _error_payloads(response.text)
    assert session.last_response_tail == {"session_postcommit_snapshot": FAILED_COMMIT}
    assert row == FAILED_COMMIT


def test_a_failed_commit_leaves_the_session_at_its_last_good_commit(monkeypatch):
    engine = _failing_engine()
    engine.install(monkeypatch)
    state = _fake_streaming_session_state()

    with TestClient(create_app(state)) as client:
        _post(client, "tail-failed-state", TURN1, stream=True)
        session = state.sessions.peek("tail-failed-state")
        session.wait_for_response_tail(5.0)
        # The session is free again and holds nothing of the failed turn.
        engine.store_error = None
        engine.stored = True
        retry = _post(client, "tail-failed-state", TURN1, stream=True)
        session.wait_for_response_tail(5.0)

    assert retry.status_code == 200
    assert not _error_payloads(retry.text)
    assert session.last_response_tail["session_postcommit_snapshot"]["stored"] is True


# --- The session's tail primitives -----------------------------------------


def test_no_tail_means_no_wait():
    session = EngineSession("s")
    assert session.wait_for_response_tail(5.0) is None


def test_a_waiter_wakes_when_the_tail_ends():
    session = EngineSession("s")
    tail = session.begin_response_tail()
    results: list[dict | None] = []
    waiter = threading.Thread(
        target=lambda: results.append(session.wait_for_response_tail(5.0))
    )
    waiter.start()
    time.sleep(0.1)
    session.end_response_tail(tail, {"session_postcommit_snapshot": {"mode": "x"}})
    waiter.join(5.0)

    assert results[0]["finished"] is True
    assert results[0]["waited_s"] >= 0.05
    assert results[0]["tail"] == {"session_postcommit_snapshot": {"mode": "x"}}
    assert session.wait_for_response_tail(5.0) is None


def test_a_waiter_that_gives_up_leaves_the_tail_to_finish():
    # A cancelled or timed-out arrival changes nothing: the commit still
    # lands and the session records it.
    session = EngineSession("s")
    tail = session.begin_response_tail()

    gave_up = session.wait_for_response_tail(0.05)
    session.end_response_tail(tail, {"done": True})

    assert gave_up["finished"] is False and gave_up["tail"] is None
    assert session.last_response_tail == {"done": True}
    assert session.wait_for_response_tail(5.0) is None


def test_an_old_tail_ending_late_does_not_release_a_newer_one():
    session = EngineSession("s")
    old = session.begin_response_tail()
    new = session.begin_response_tail()

    session.end_response_tail(old, {"turn": 1})
    still_running = session.wait_for_response_tail(0.05)
    session.end_response_tail(new, {"turn": 2})

    assert still_running["finished"] is False
    assert session.last_response_tail == {"turn": 2}
    assert session.wait_for_response_tail(5.0) is None
