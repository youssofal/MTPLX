"""Serving-path tests for the dense batched-MTP cohort service (T-204 item 1).

These exercise the SERVER's behaviour — admission, sealing, per-caller result
routing, cancellation — with a fake driver standing in for
``generate_dense_mtp_batch``. No model and no GPU: the driver's own correctness
is covered by ``tests/test_dense_mtp_batch.py`` and its GPU parity gate, and
what is unproven until this file exists is that the service hands the right
tokens to the right caller.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import time
from threading import Event
from typing import Any

import pytest

from mtplx.dense_mtp_batch import DenseBatchResult, DenseBatchStreamResult
from mtplx.server.dense_mtp_batch import (
    DenseMTPBatchGenerationService,
    DenseMTPBatchJob,
    DenseMTPBatchQueueFull,
    dense_mtp_batch_compatibility_key,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class _FakeGeometry:
    cohort_slots: int = 8
    max_context_tokens: int = 4096
    depth: int = 3


@dataclass
class _FakeLane:
    runtime: Any = "runtime-sentinel"
    geometry: _FakeGeometry = None
    route_id: str = "dense_mtp_batch/test"
    capture_backend: str = "stock"
    head_history: str = "committed"
    loop_mode: str = "pipelined"
    draft_core: str = "eager"
    history_window: int = 8192
    prefill_chunk: int = 2048
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.geometry is None:
            self.geometry = _FakeGeometry()


class _OwnerScheduler:
    """Runs the pump inline, standing in for the model-owner thread."""

    def __init__(self) -> None:
        self.foreground_calls: list[str] = []

    def submit_foreground(self, fn, *args, batch_key=None, **kwargs):
        self.foreground_calls.append(str(batch_key))
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def is_owner_thread(self) -> bool:
        return False


class _FakeState:
    def __init__(self) -> None:
        self.model_scheduler = _OwnerScheduler()


class _RecordingDriver:
    """Stands in for generate_dense_mtp_batch, recording exactly what it got.

    Emits a deterministic, ROW-DISTINCT token stream through ``on_commit`` so a
    test can prove the service routed each row's tokens to that row's caller
    rather than, say, giving everyone row 0's output.
    """

    def __init__(self, *, tokens_per_row: int = 4, stop_row: int | None = None):
        self.tokens_per_row = tokens_per_row
        self.stop_row = stop_row
        self.calls: list[dict[str, Any]] = []

    def __call__(self, runtime, prompts, **kwargs):
        self.calls.append({"runtime": runtime, "prompts": prompts, **kwargs})
        caps = kwargs.get("max_new_tokens_per_row") or [
            kwargs["max_new_tokens"]
        ] * len(prompts)
        on_commit = kwargs.get("on_commit")
        stop_ids = set(kwargs.get("stop_token_ids") or set())
        stop_id = next(iter(stop_ids), 999)
        streams = []
        for row, cap in enumerate(caps):
            emitted: list[int] = []
            for step in range(min(cap, self.tokens_per_row)):
                token = 1000 * (row + 1) + step
                emitted.append(token)
                if on_commit is not None:
                    on_commit(row, token)
            if self.stop_row == row:
                emitted.append(stop_id)
                if on_commit is not None:
                    on_commit(row, stop_id)
                reason = "stop"
            else:
                reason = "length" if len(emitted) >= cap else "cycle_cap"
            streams.append(
                DenseBatchStreamResult(
                    index=row,
                    prompt_len=len(prompts[row]),
                    tokens=emitted,
                    finish_reason=reason,
                    sha=f"sha-{row}",
                )
            )
        return DenseBatchResult(
            batch_size=len(prompts),
            depth=int(kwargs.get("depth", 3)),
            streams=streams,
            cycles=3,
            generated_tokens=sum(len(s.tokens) for s in streams),
            accepted_draft_tokens=6,
            accepted_by_depth=[2, 2, 2],
            drafted_by_depth=[3, 3, 3],
            prefill_s=0.01,
            decode_s=0.05,
        )


@dataclass
class _FakeSampler:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


def _make_job(
    *,
    request_id: str,
    prompt_len: int = 8,
    max_tokens: int = 4,
    sampler: _FakeSampler | None = None,
    lane: _FakeLane,
    stop_token_ids: set[int] | None = None,
    solo_result: dict[str, Any] | None = None,
    token_sink: list[list[int]] | None = None,
) -> DenseMTPBatchJob:
    sampler = sampler or _FakeSampler()
    stop_token_ids = stop_token_ids if stop_token_ids is not None else {999}
    return DenseMTPBatchJob(
        request_id=request_id,
        prompt_ids=list(range(1, prompt_len + 1)),
        max_tokens=max_tokens,
        sampler=sampler,
        seed=1234,
        stop_token_ids=set(stop_token_ids),
        compatibility_key=dense_mtp_batch_compatibility_key(
            lane, sampler, set(stop_token_ids)
        ),
        generation_limits={},
        solo_runner=(lambda _job: dict(solo_result or {"solo": True})),
        cancel_error=lambda item: RuntimeError(f"cancelled {item.request_id}"),
        token_callback=(token_sink.append if token_sink is not None else None),
    )


def _service(lane: _FakeLane, driver: Any, **kwargs) -> DenseMTPBatchGenerationService:
    return DenseMTPBatchGenerationService(
        _FakeState(),
        lane=lane,
        driver=driver,
        batch_wait_s=0.0,
        auto_schedule=False,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# The milestone: concurrent requests are batched, and each caller gets its own
# --------------------------------------------------------------------------- #
def test_cohort_batches_compatible_requests_and_routes_tokens_per_caller() -> None:
    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(4)]
    for job in jobs:
        service.submit(job)

    assert service.pump_once() is True
    assert len(driver.calls) == 1, "four compatible requests must seal ONE cohort"
    assert len(driver.calls[0]["prompts"]) == 4

    for row, job in enumerate(jobs):
        result = job.future.result(timeout=1)
        expected = [1000 * (row + 1) + step for step in range(4)]
        assert result["tokens"] == expected, f"row {row} got another row's tokens"
        assert result["stats"]["dense_mtp_batch_real_width"] == 4
        assert result["stats"]["active_batch_size"] == 4
        assert result["stats"]["scheduler_lane"] == "dense_mtp_batch"

    assert service.snapshot()["batch_histogram"] == {"4": 1}


def test_tokens_reach_the_client_callback_as_they_commit() -> None:
    """Each caller's callback must see its own tokens, and only its own."""

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    sinks: list[list[list[int]]] = [[] for _ in range(3)]
    jobs = [
        _make_job(request_id=f"r{i}", lane=lane, token_sink=sinks[i])
        for i in range(3)
    ]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    for row, sink in enumerate(sinks):
        flat = [token for chunk in sink for token in chunk]
        assert flat == [1000 * (row + 1) + step for step in range(4)]


def test_stop_token_is_kept_in_the_result_but_not_forwarded_to_the_client() -> None:
    lane = _FakeLane()
    service = _service(lane, _RecordingDriver(stop_row=1))
    sinks: list[list[list[int]]] = [[] for _ in range(2)]
    jobs = [
        _make_job(request_id=f"r{i}", lane=lane, token_sink=sinks[i])
        for i in range(2)
    ]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    stopped = jobs[1].future.result(timeout=1)
    assert stopped["tokens"][-1] == 999
    assert stopped["finish_reason"] == "stop"
    assert 999 not in [t for chunk in sinks[1] for t in chunk]


# --------------------------------------------------------------------------- #
# Item 3 removed the uniform-sampling constraint
# --------------------------------------------------------------------------- #
def test_requests_with_different_sampling_now_share_one_cohort() -> None:
    """Item 1 split these into separate cohorts; item 3 lets them share one.

    What makes it safe is that the sampling parameters reach the driver as
    per-row vectors, so each caller's tokens are drawn from that caller's own
    distribution. The vectors must line up with the prompt order or a caller
    silently gets a neighbour's sampling settings.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    greedy = _make_job(request_id="greedy", lane=lane, sampler=_FakeSampler(0.0))
    hot = _make_job(
        request_id="hot", lane=lane, sampler=_FakeSampler(0.8, top_p=0.95, top_k=40)
    )
    service.submit(greedy)
    service.submit(hot)
    service.pump_once()

    assert len(driver.calls) == 1, "the pair must now be co-admitted"
    call = driver.calls[0]
    assert call["temperature"] == [0.0, 0.8]
    assert call["top_p"] == [1.0, 0.95]
    assert call["top_k"] == [0, 40]
    assert greedy.future.result(timeout=1)["tokens"]
    assert hot.future.result(timeout=1)["tokens"]


def test_three_different_sampling_settings_still_form_one_cohort() -> None:
    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    samplers = [_FakeSampler(0.0), _FakeSampler(0.7), _FakeSampler(1.3, top_k=10)]
    for i, sampler in enumerate(samplers):
        service.submit(_make_job(request_id=f"r{i}", lane=lane, sampler=sampler))
    service.pump_once()

    assert len(driver.calls) == 1
    assert driver.calls[0]["temperature"] == [0.0, 0.7, 1.3]


def test_the_cohort_seed_actually_used_is_reported() -> None:
    """A per-request seed is not honoured in a cohort, so say which one was."""

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(2)]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    for job in jobs:
        stats = job.future.result(timeout=1)["stats"]
        assert stats["dense_mtp_batch_cohort_seed"] == 1234


# --------------------------------------------------------------------------- #
# Item 2 removed the uniform-length constraint
# --------------------------------------------------------------------------- #
def test_mixed_lengths_reach_the_driver_unpadded() -> None:
    """No padding at all: true lengths, and ragged_prompts asked for."""

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    short = _make_job(request_id="short", lane=lane, prompt_len=10)
    long = _make_job(request_id="long", lane=lane, prompt_len=40)
    service.submit(short)
    service.submit(long)
    service.pump_once()

    call = driver.calls[0]
    assert call["ragged_prompts"] is True
    assert [len(p) for p in call["prompts"]] == [10, 40]
    assert call["prompts"][0] == list(range(1, 11)), "no pad prefix"

    for job in (short, long):
        stats = job.future.result(timeout=1)["stats"]
        assert stats["dense_mtp_batch_left_pad_tokens"] == 0
        assert stats["dense_mtp_batch_ragged_prompts"] is True


def test_a_wildly_longer_prompt_is_admitted_by_default() -> None:
    """Item 1 refused this pair to bound padding. With no padding, it is fine."""

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    tiny = _make_job(request_id="tiny", lane=lane, prompt_len=8)
    huge = _make_job(request_id="huge", lane=lane, prompt_len=2000)
    service.submit(tiny)
    service.submit(huge)
    service.pump_once()

    assert len(driver.calls) == 1
    assert [len(p) for p in driver.calls[0]["prompts"]] == [8, 2000]


def test_an_operator_can_still_bound_the_length_spread_for_latency() -> None:
    """The knob survives as a fairness policy, off unless asked for.

    A cohort finishes together, so an operator who cares more about short
    request latency than aggregate throughput can keep a 24k prompt from
    dragging a tiny one along for its prefill.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(
        lane, driver, length_spread_tokens=16, length_spread_ratio=0.25
    )
    tiny = _make_job(request_id="tiny", lane=lane, prompt_len=8)
    huge = _make_job(request_id="huge", lane=lane, prompt_len=2000)
    service.submit(tiny)
    service.submit(huge)

    service.pump_once()
    service.pump_once()
    # Each runs on its own. Under continuous batching a lone request goes
    # through the driver rather than the solo path (see
    # test_a_lone_request_no_longer_takes_the_solo_path), so the assertion is
    # that they were never in the SAME run -- which is what the bound promises.
    # Asserting "the driver was never called" would now be asserting the solo
    # path, which is a different guarantee.
    assert len(driver.calls) == 2, "each ran on its own"
    for call in driver.calls:
        assert len(call["prompts"]) == 1, "the bound must keep the pair apart"
    assert tiny.future.done() and huge.future.done()


def test_similar_lengths_are_batched_under_a_bound() -> None:
    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver, length_spread_tokens=512)
    for i, length in enumerate((500, 520, 505)):
        service.submit(
            _make_job(request_id=f"r{i}", lane=lane, prompt_len=length)
        )
    service.pump_once()
    assert len(driver.calls) == 1
    assert len(driver.calls[0]["prompts"]) == 3


# --------------------------------------------------------------------------- #
# Per-request max_tokens is NOT a constraint
# --------------------------------------------------------------------------- #
def test_cohort_mixes_different_max_tokens_and_passes_per_row_caps() -> None:
    lane = _FakeLane()
    driver = _RecordingDriver(tokens_per_row=99)
    service = _service(lane, driver)
    jobs = [
        _make_job(request_id=f"r{i}", lane=lane, max_tokens=cap)
        for i, cap in enumerate((2, 6, 3))
    ]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    call = driver.calls[0]
    assert call["max_new_tokens_per_row"] == [2, 6, 3]
    assert call["max_new_tokens"] == 6
    for job, cap in zip(jobs, (2, 6, 3)):
        assert len(job.future.result(timeout=1)["tokens"]) == cap


# --------------------------------------------------------------------------- #
# A lone request must not pay for batching it did not get
# --------------------------------------------------------------------------- #
def test_single_request_runs_the_solo_path_when_nothing_can_join() -> None:
    """The solo fast path survives, and is correct exactly when it is safe.

    A one-row cohort through the batch driver is slower than the tuned solo
    loop, so a request that really is alone must not pay for batching. With
    continuous batching OFF, nothing can join a running cohort, so "alone at
    seal time" and "alone for the whole run" are the same statement.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver, continuous=False)
    job = _make_job(
        request_id="alone", lane=lane, solo_result={"tokens": [7], "stats": {}}
    )
    service.submit(job)
    service.pump_once()

    assert driver.calls == [], "a cohort of one must not enter the batch driver"
    result = job.future.result(timeout=1)
    assert result["_dense_mtp_batch_solo"] is True
    assert result["tokens"] == [7]


def test_a_lone_request_no_longer_takes_the_solo_path() -> None:
    """A deliberate regression in lone-request latency, bought for a large win.

    "Alone at seal time" stopped meaning "alone for the run" the moment the
    driver could pull. Measured on the 4B at 7f37ec4, eight simultaneous
    requests sealed as one SOLO run plus a cohort of seven, and those seven
    waited 7.9 seconds -- an entire solo generation -- because the solo path
    cannot admit anyone. The service cannot tell at seal time which case it is
    in, so it assumes company. Operators who know their traffic is one caller
    at a time get the old behaviour back with continuous=False.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    job = _make_job(
        request_id="alone", lane=lane, solo_result={"tokens": [7], "stats": {}}
    )
    service.submit(job)
    service.pump_once()

    assert len(driver.calls) == 1, "the lone request went through the driver"
    assert len(driver.calls[0]["prompts"]) == 1, "as a cohort of exactly one"
    assert callable(driver.calls[0]["pull_queued"]), "and it can be joined"
    assert "_dense_mtp_batch_solo" not in job.future.result(timeout=1)


# --------------------------------------------------------------------------- #
# Lane settings reach the driver
# --------------------------------------------------------------------------- #
def test_lane_settings_are_passed_through_to_the_driver() -> None:
    lane = _FakeLane(
        geometry=_FakeGeometry(cohort_slots=8, max_context_tokens=4096, depth=2),
        capture_backend="stock",
        head_history="committed",
        loop_mode="pipelined",
        draft_core="eager",
    )
    driver = _RecordingDriver()
    service = _service(lane, driver)
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    call = driver.calls[0]
    assert call["runtime"] == "runtime-sentinel"
    assert call["depth"] == 2
    assert call["capture_backend"] == "stock"
    assert call["head_history"] == "committed"
    assert call["loop_mode"] == "pipelined"
    assert call["draft_core"] == "eager"
    assert call["stop_token_ids"] == {999}
    assert call["ragged_prompts"] is True


def test_cohort_width_is_capped_by_the_lane_geometry() -> None:
    """The DECODED width is capped; the surplus rides along as refill.

    Before item 4 the leftover jobs waited for a second cohort, and this test
    asserted that second pump. They now join the same run and take slots as
    they free, which is the point of continuous batching. The cap still holds:
    only three rows are ever decoded at once.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=3))
    driver = _RefillAwareDriver()
    service = _service(lane, driver)
    for i in range(5):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))

    service.pump_once()
    assert len(driver.calls[0]["prompts"]) == 3, "sealed width still capped"
    assert driver.calls[0]["max_cohort_rows"] == 3, (
        "the cap now reaches the driver as a GROWTH ceiling: the driver decides "
        "the width cycle by cycle, so a cap the service enforced only at seal "
        "time would not bound anything once rows start joining"
    )
    assert len(driver.calls[0]["pulled"]) == 2, "surplus joins the same run"


# --------------------------------------------------------------------------- #
# Cancellation and failure
# --------------------------------------------------------------------------- #
def test_cancelled_request_is_dropped_before_the_cohort_seals() -> None:
    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    keep = [_make_job(request_id=f"keep{i}", lane=lane) for i in range(2)]
    doomed = _make_job(request_id="doomed", lane=lane)
    for job in (*keep, doomed):
        service.submit(job)
    doomed.cancel_event.set()

    service.pump_once()
    assert len(driver.calls[0]["prompts"]) == 2
    with pytest.raises(RuntimeError, match="cancelled doomed"):
        doomed.future.result(timeout=1)
    for job in keep:
        assert job.future.result(timeout=1)["tokens"]


def test_a_driver_failure_fails_every_row_of_that_cohort() -> None:
    """One cohort, one fate. A partial success would leave callers hanging."""

    def _boom(runtime, prompts, **kwargs):
        raise RuntimeError("verify forward exploded")

    lane = _FakeLane()
    service = _service(lane, _boom)
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(3)]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    for job in jobs:
        with pytest.raises(RuntimeError, match="verify forward exploded"):
            job.future.result(timeout=1)
    assert "verify forward exploded" in (service.snapshot()["last_error"] or "")


def test_shutdown_rejects_pending_requests_rather_than_dropping_them() -> None:
    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    job = _make_job(request_id="late", lane=lane)
    service.submit(job)
    service.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        job.future.result(timeout=1)


def test_submit_after_shutdown_is_refused_immediately() -> None:
    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    service.shutdown()
    job = _make_job(request_id="too-late", lane=lane)
    service.submit(job)
    with pytest.raises(RuntimeError, match="shut down"):
        job.future.result(timeout=1)


# --------------------------------------------------------------------------- #
# The counters a concurrency measurement depends on
# --------------------------------------------------------------------------- #
def test_snapshot_reports_the_widths_actually_sealed() -> None:
    """`batch_histogram` is how a run proves it batched at all.

    Throughput alone cannot distinguish "never batched" from "batched and did
    not help", and only the first is a bug.
    """

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    for i in range(3):
        service.submit(_make_job(request_id=f"a{i}", lane=lane))
    service.pump_once()
    for i in range(2):
        service.submit(_make_job(request_id=f"b{i}", lane=lane))
    service.pump_once()

    snapshot = service.snapshot()
    assert snapshot["batch_histogram"] == {"2": 1, "3": 1}
    assert snapshot["last_real_width"] == 2
    assert snapshot["left_pad_tokens_total"] == 0
    assert snapshot["target_verify_cycles"] == 6


# --------------------------------------------------------------------------- #
# The cohort must run on a cache layout that can express a batch
# --------------------------------------------------------------------------- #
def test_cohort_runs_on_the_batch_generic_cache_lane() -> None:
    """Regression: the served paged KV cache raises at batch > 1.

    A real server returned HTTP 500 for every cohort with
    "VllmMetalPagedKVCache currently supports batch size 1". The driver builds
    its caches through the runtime, which reads these environment switches at
    make_cache time, so the cohort has to run with them off.
    """

    import os

    seen: dict[str, str | None] = {}

    def _capture_env(runtime, prompts, **kwargs):
        for key in (
            "MTPLX_VLLM_METAL_PAGED_ATTN",
            "MTPLX_OWNED_ATTN_KV",
            "MTPLX_BLOCK_OWNED_ATTN_KV",
        ):
            seen[key] = os.environ.get(key)
        return _RecordingDriver()(runtime, prompts, **kwargs)

    lane = _FakeLane()
    service = _service(lane, _capture_env)
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))

    os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] = "1"
    try:
        service.pump_once()
    finally:
        os.environ.pop("MTPLX_VLLM_METAL_PAGED_ATTN", None)

    assert seen["MTPLX_VLLM_METAL_PAGED_ATTN"] == "0"
    assert seen["MTPLX_OWNED_ATTN_KV"] == "0"
    assert seen["MTPLX_BLOCK_OWNED_ATTN_KV"] == "0"


def test_the_servers_cache_layout_is_restored_after_a_cohort() -> None:
    """Solo requests must keep the paged cache, so the scope has to restore."""

    import os

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))

    os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] = "1"
    os.environ.pop("MTPLX_OWNED_ATTN_KV", None)
    try:
        service.pump_once()
        assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
        assert "MTPLX_OWNED_ATTN_KV" not in os.environ
    finally:
        os.environ.pop("MTPLX_VLLM_METAL_PAGED_ATTN", None)


def test_the_cache_layout_is_restored_even_when_the_cohort_fails() -> None:
    import os

    def _boom(runtime, prompts, **kwargs):
        raise RuntimeError("cohort exploded")

    lane = _FakeLane()
    service = _service(lane, _boom)
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))

    os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] = "1"
    try:
        service.pump_once()
        assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    finally:
        os.environ.pop("MTPLX_VLLM_METAL_PAGED_ATTN", None)


# --------------------------------------------------------------------------- #
# Streaming semantics (T-204 item 5): tokens must reach the caller DURING the
# run, not at cohort drain
# --------------------------------------------------------------------------- #
def test_tokens_reach_the_caller_before_the_request_completes() -> None:
    """The property that makes streaming real, stated as a timing assertion.

    ``test_tokens_reach_the_client_callback_as_they_commit`` proves each caller
    gets its OWN tokens. It does not prove they arrive early: a lane that
    buffered everything and replayed the callbacks at the end would pass it.
    That is exactly the difference between item 1 and item 5.

    Here the fake driver asks, at the moment each callback fires, whether that
    caller's future is already resolved. For streaming to be real the answer
    must be no.
    """

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(3)]
    future_done_at_callback: list[bool] = []

    for job in jobs:
        job.token_callback = (
            lambda toks, j=job: future_done_at_callback.append(j.future.done())
        )
        service.submit(job)
    service.pump_once()

    assert future_done_at_callback, "no tokens were delivered at all"
    assert not any(future_done_at_callback), (
        "a token was delivered only after the request had already completed, "
        "which is buffering rather than streaming"
    )
    for job in jobs:
        assert job.future.done()


def test_a_slow_client_callback_does_not_reorder_another_callers_tokens() -> None:
    """Per-caller ordering must hold even when one client is slow.

    The cohort commits on one thread, so a callback that blocks delays the
    whole cohort. That is a throughput property and acceptable. What is NOT
    acceptable is a caller receiving its own tokens out of order.
    """

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    received: dict[str, list[int]] = {}

    def make_cb(request_id: str, slow: bool):
        def cb(tokens: list[int]) -> None:
            if slow:
                time.sleep(0.002)
            received.setdefault(request_id, []).extend(int(t) for t in tokens)
        return cb

    jobs = []
    for i in range(3):
        job = _make_job(request_id=f"r{i}", lane=lane)
        job.token_callback = make_cb(job.request_id, slow=(i == 1))
        jobs.append(job)
        service.submit(job)
    service.pump_once()

    for row, job in enumerate(jobs):
        expected = [1000 * (row + 1) + step for step in range(4)]
        assert received[job.request_id] == expected, job.request_id


# --------------------------------------------------------------------------- #
# Continuous batching at the SERVICE level (T-204 item 4)
# --------------------------------------------------------------------------- #
def _drain_pull(kwargs) -> list[dict]:
    """Call ``pull_queued`` until it stops giving, as the real driver does.

    The real driver pulls at every cycle boundary; a fake that pulls once at
    the start reaches the same end state and keeps the tests synchronous. A
    fake that does NOT pull is also a valid fake -- it models a driver that
    never gets a free row -- which is why pulling is opt-in per driver class
    rather than done for every test.
    """

    pull = kwargs.get("pull_queued")
    if pull is None:
        return []
    room = int(kwargs.get("max_cohort_rows") or len(kwargs.get("prompts") or [])) or 1
    taken: list[dict] = []
    while True:
        more = list(pull(max(1, room)) or [])
        if not more:
            return taken
        taken.extend(more)


class _RefillAwareDriver(_RecordingDriver):
    """Fake driver that takes joiners the way the real one does.

    Returns one stream per REQUEST, initial cohort first then everything it
    pulled, and routes tokens through on_commit by REQUEST index.
    """

    def __call__(self, runtime, prompts, **kwargs):
        pulled = _drain_pull({**kwargs, "prompts": prompts})
        self.calls.append(
            {"runtime": runtime, "prompts": prompts, "pulled": pulled, **kwargs}
        )
        queue = list(kwargs.get("refill_queue") or []) + list(pulled)
        caps = list(kwargs.get("max_new_tokens_per_row") or [])
        caps += [int(item.get("max_new_tokens", 4)) for item in queue]
        on_commit = kwargs.get("on_commit")
        streams = []
        for request, cap in enumerate(caps):
            emitted: list[int] = []
            for step in range(min(cap, self.tokens_per_row)):
                token = 1000 * (request + 1) + step
                emitted.append(token)
                if on_commit is not None:
                    on_commit(request, token)
            streams.append(
                DenseBatchStreamResult(
                    index=request,
                    prompt_len=len(prompts[0]),
                    tokens=emitted,
                    finish_reason="length",
                    sha=f"sha-{request}",
                    slot=request if request < len(prompts) else 0,
                )
            )
        return DenseBatchResult(
            batch_size=len(prompts),
            depth=int(kwargs.get("depth", 3)),
            streams=streams,
            cycles=3,
            generated_tokens=sum(len(s.tokens) for s in streams),
            accepted_draft_tokens=6,
            accepted_by_depth=[2, 2, 2],
            drafted_by_depth=[3, 3, 3],
            prefill_s=0.01,
            decode_s=0.05,
        )


def test_service_hands_the_driver_a_live_queue_not_a_frozen_list() -> None:
    """The capability must be reachable from the server, not just the library.

    A driver feature nothing calls is the trap this whole task opened with.
    Rewritten for the live queue: the service no longer decides at seal time
    which waiting jobs may join, it hands the driver a handle and the driver
    pulls. The observable difference is that `refill_queue` is empty and
    everything arrives through `pull_queued`.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    driver = _RefillAwareDriver()
    service = _service(lane, driver)
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(4)]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    call = driver.calls[0]
    assert len(call["prompts"]) == 2, "cohort is capped by cohort_slots"
    assert callable(call["pull_queued"]), "the driver must get a LIVE handle"
    assert call["refill_queue"] in (None, []), (
        "the frozen pre-move is what item 4 removes; nothing may ride along"
    )
    assert len(call["pulled"]) == 2, "the rest join by being pulled"
    assert call["max_cohort_rows"] == 2, "width ceiling reaches the driver"
    for job in jobs:
        assert job.future.result(timeout=1)["tokens"], job.request_id


def test_joiner_tokens_reach_the_joiners_own_caller() -> None:
    """Routing by REQUEST index, so a joiner streams to the right client."""

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _RefillAwareDriver())
    sinks: list[list[list[int]]] = [[] for _ in range(4)]
    jobs = []
    for i in range(4):
        job = _make_job(request_id=f"r{i}", lane=lane, token_sink=sinks[i])
        jobs.append(job)
        service.submit(job)
    service.pump_once()

    for request, sink in enumerate(sinks):
        flat = [token for chunk in sink for token in chunk]
        assert flat == [1000 * (request + 1) + step for step in range(4)], request


def test_a_sampling_job_joins_an_all_greedy_cohort() -> None:
    """The filter that kept it out was measured costing 27-second queue waits.

    The rule was real: an all-greedy cohort ran a path with no randomness, so a
    sampling joiner would have been served with none. The driver now ACQUIRES
    the sampling path for a joiner that needs it, so holding the request back
    protects nothing and costs a great deal. Found by hammering under an
    ordinary mixed load, where 65% of requests being greedy meant most cohorts
    sealed all-greedy and every sampling request waited one out.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    driver = _RefillAwareDriver()
    service = _service(lane, driver)
    for i in range(2):
        service.submit(
            _make_job(request_id=f"greedy{i}", lane=lane, sampler=_FakeSampler(0.0))
        )
    hot = _make_job(request_id="hot", lane=lane, sampler=_FakeSampler(0.9))
    service.submit(hot)
    service.pump_once()

    pulled = driver.calls[0].get("pulled") or []
    assert len(pulled) == 1, "the sampling job must be offered to the cohort"
    assert pulled[0]["temperature"] == pytest.approx(0.9), (
        "and it must carry its OWN temperature, not the cohort's"
    )
    assert hot.future.result(timeout=1)["tokens"]


def test_a_sampling_job_is_still_held_back_from_a_compiled_draft_chain() -> None:
    """The one case the driver genuinely cannot acquire.

    A compiled draft chain has no sampling path, so this is not a policy the
    scheduler is free to relax. The lane installs `draft_core='eager'` only, so
    this guards a future revision rather than a live configuration -- which is
    exactly why it needs a test: nothing else would notice it rotting.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    lane.draft_core = "compiled"
    driver = _RefillAwareDriver()
    service = _service(lane, driver)
    for i in range(2):
        service.submit(
            _make_job(request_id=f"greedy{i}", lane=lane, sampler=_FakeSampler(0.0))
        )
    hot = _make_job(request_id="hot", lane=lane, sampler=_FakeSampler(0.9))
    service.submit(hot)
    service.pump_once()

    assert not driver.calls[0].get("pulled"), "hot job must not join"
    assert not hot.future.done(), "it stays pending for its own cohort"


def test_a_joiner_carries_its_own_sampling_into_the_payload() -> None:
    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    driver = _RefillAwareDriver()
    service = _service(lane, driver)
    for i in range(2):
        service.submit(
            _make_job(request_id=f"warm{i}", lane=lane, sampler=_FakeSampler(0.7))
        )
    service.submit(
        _make_job(
            request_id="joiner",
            lane=lane,
            sampler=_FakeSampler(0.2, top_p=0.8, top_k=15),
        )
    )
    service.pump_once()

    payload = driver.calls[0]["pulled"]
    assert payload[0]["temperature"] == pytest.approx(0.2)
    assert payload[0]["top_p"] == pytest.approx(0.8)
    assert payload[0]["top_k"] == 15


def test_service_counts_continuous_batching_rather_than_implying_it() -> None:
    """The batch histogram records cohort WIDTH; item 4's claim is different.

    "One cohort served more requests than its width" is the single fact that
    distinguishes continuous batching working from continuous batching merely
    existing, and nothing recorded it. Without these counters a GPU run could
    only infer it, which is exactly what the observed-width discipline exists
    to prevent.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _RefillAwareDriver())
    for i in range(5):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    snap = service.snapshot()
    assert snap["last_real_width"] == 2, "only two rows SEALED the cohort"
    # All five are served by the one run now. The old bound (refill_depth,
    # defaulting to cohort_slots) existed because a job accepted into the
    # frozen refill list LEFT the pending queue and could no longer be picked
    # up by a different cohort, so an unbounded list let one cohort claim
    # everything waiting. A live queue has no such hazard: a job is taken at
    # the moment there is a row for it. What bounds a cohort now is
    # max_requests_per_cohort, and it is eight times the width by default.
    assert snap["max_requests_in_one_cohort"] == 5, "two rows served five requests"
    assert snap["refill_admitted_total"] == 3
    assert snap["requests_served_total"] == 5
    assert snap["continuous_batching_observed"] is True


def test_snapshot_active_counts_joiners_not_just_the_sealed_cohort() -> None:
    """The soak aborts on `pending > 0 and 0 < active < slots`.

    `_active` is written once, at seal. Under continuous batching the joiners
    arrive afterwards and land in `_refill`, so reporting only `_active` made a
    healthy wide cohort read as a narrow one -- and that is exactly the
    zombie-slot signature the soak treats as a defect. An overnight run would
    have aborted on its own instrumentation.

    Found while hammering, BEFORE the soak was launched rather than after it
    had wasted a night, which is the whole argument for verifying every abort
    condition against the behaviour it now describes.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    seen: list[dict] = []
    service = None

    class _Peek(_RefillAwareDriver):
        def __call__(self, runtime, prompts, **kwargs):
            pulled = _drain_pull({**kwargs, "prompts": prompts})
            seen.append(service.snapshot())
            return super().__call__(
                runtime, prompts,
                **{**kwargs, "refill_queue": pulled, "pull_queued": None},
            )

    service = _service(lane, _Peek())
    for i in range(5):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    snap = seen[0]
    assert snap["active_sealed"] == 2, "the cohort sealed at two rows"
    assert snap["active_joined"] == 3, "and three more joined it"
    assert snap["active"] == 5, (
        "`active` must count every request the cohort is serving; reporting "
        "only the sealed set is the zombie-slot false positive"
    )


def test_one_oversized_request_does_not_take_its_cohort_mates_down() -> None:
    """Found by the soak's fault-injection arm, which exists to ask exactly this.

    A caller asked for `max_tokens: 260943`. That is inside the model's context
    window, so nothing clamped it -- but the KV reservation is sized from
    max_new_tokens, so a two-row cohort reserved for a quarter of a million
    tokens per row and the machine died with a Metal out-of-memory. BOTH
    requests failed and only one had asked for anything unreasonable.

    The `memory_headroom` guard does not catch this: it bounds GROWTH, and a
    cohort that is already too wide never grows.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=4))
    driver = _RecordingDriver()
    service = _service(lane, driver, memory_headroom=0.5)
    # Stand in for a measured reservation from a previous cohort. Without one
    # the check is deliberately inert -- see the "first cohort" test below.
    service._last_kv = {"kv_bytes_per_token_per_slot": 65536.0}

    ok = _make_job(request_id="ok", lane=lane, max_tokens=64)
    huge = _make_job(request_id="huge", lane=lane, max_tokens=260943)
    service.submit(ok)
    service.submit(huge)
    service.pump_once()

    assert ok.future.result(timeout=1), "the innocent request must still be served"
    assert huge.future.done(), "the impossible request must be told so"
    with pytest.raises(DenseMTPBatchQueueFull, match="cannot be served"):
        huge.future.result(timeout=1)
    snap = service.snapshot()
    assert snap["unservable_total"] == 1
    assert snap["rejected_total"] == 0, (
        "too big for the machine is a different answer from too busy right now, "
        "and a caller acts on them differently"
    )


def test_a_cohort_too_large_in_aggregate_is_narrowed_not_failed() -> None:
    """Rows that fit alone but not together stay pending, they are not refused.

    Being in the wrong queue at the wrong moment is not the caller's fault and
    must not be their problem. Only a request that cannot fit on an empty
    machine gets an error.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=4))
    driver = _RecordingDriver()
    service = _service(lane, driver, memory_headroom=0.5)
    service._last_kv = {"kv_bytes_per_token_per_slot": 65536.0}

    budget = service._kv_budget_bytes()
    # Each request asks for a third of the budget, so two fit and the rest do
    # not -- and none of them is individually unservable.
    per_request = int((budget / 65536.0) / 3)
    jobs = [
        _make_job(request_id=f"r{i}", lane=lane, max_tokens=per_request)
        for i in range(4)
    ]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    assert service.snapshot()["unservable_total"] == 0, "nobody may be refused"
    assert len(driver.calls[0]["prompts"]) < 4, "the cohort must have narrowed"
    assert service.snapshot()["pending"] > 0, "the surplus waits for the next one"


def test_the_first_cohort_is_not_refused_for_want_of_a_measurement() -> None:
    """Deliberately permissive with no data, rather than guessing high.

    Bytes-per-token depends on layer count, KV heads, head dimension and dtype.
    A constant baked in here would be wrong for the next model somebody points
    this lane at, and wrong in the direction of refusing work the machine could
    have served. So the check is measured from the PREVIOUS cohort and is inert
    until there has been one.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=4))
    driver = _RecordingDriver()
    service = _service(lane, driver, memory_headroom=0.5)
    assert service._last_kv == {}, "fixture must start with no measurement"

    job = _make_job(request_id="huge", lane=lane, max_tokens=10_000_000)
    service.submit(job)
    service.pump_once()

    assert service.snapshot()["unservable_total"] == 0
    assert job.future.done(), "it ran; the check had no data and did not guess"


def test_the_lane_shares_the_servers_bank_rather_than_building_a_second() -> None:
    """One cache for both lanes, which was the whole point of using SessionBank.

    Until 2026-08-24 this lane constructed its OWN bank while the solo path used
    `state.sessions.bank` -- two banks, two budgets, twice the memory and no
    cross-lane reuse, which is the exact duplication the migration was meant to
    remove. The justification was in the docs and the commit messages for hours
    while the code did the opposite, and no test would have noticed, because the
    fixture had no bank to share.
    """

    from mtplx.session_bank import SessionBank

    shared = SessionBank(max_bytes=64 * 1024**2)

    class _Sessions:
        bank = shared

    class _StateWithBank(_FakeState):  # type: ignore[misc, valid-type]
        sessions = _Sessions()

    service = DenseMTPBatchGenerationService(
        _StateWithBank(),
        lane=_FakeLane(),
        driver=_RecordingDriver(),
        batch_wait_s=0.0,
        auto_schedule=False,
        prefix_cache_bytes=32 * 1024**2,
    )
    assert service.prefix_bank is shared, (
        "the lane built its own bank instead of sharing the server's"
    )


def test_a_lane_with_no_server_bank_still_gets_prefix_reuse() -> None:
    """Sharing is preferred, not required -- a bare lane must still work."""

    service = _service(_FakeLane(), _RecordingDriver(),
                       prefix_cache_bytes=64 * 1024**2)
    assert service.prefix_bank is not None


def test_the_prefix_cache_is_off_by_default_and_reachable_when_asked() -> None:
    """A driver feature nothing calls is the trap this whole task opened with.

    Off by default deliberately: it is new, it holds gigabytes, and the soak that
    would catch a leak in it has not run. An operator turns it on; nobody gets it
    by surprise.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()

    off = _service(lane, driver)
    assert off.prefix_bank is None
    assert off.snapshot()["prefix_cache"] is None

    on = _service(lane, driver, prefix_cache_bytes=64 * 1024**2,
                  prefix_cache_min_tokens=32)
    assert on.prefix_bank is not None
    stats = on.snapshot()["prefix_cache"]
    assert stats["max_bytes"] == 64 * 1024**2
    assert stats["entries"] == 0
    # hit_rate is the number that says whether it earns its memory.
    assert stats["hit_rate"] == 0.0

    on.submit(_make_job(request_id="r0", lane=lane))
    on.submit(_make_job(request_id="r1", lane=lane))
    on.pump_once()
    assert driver.calls[0]["session_bank"] is on.prefix_bank, (
        "the store must actually reach the driver"
    )


def test_max_requests_per_cohort_is_configurable() -> None:
    """Replaces `refill_depth`, which the live queue made inert.

    `refill_depth` bounded how many jobs could be COMMITTED to a cohort at seal
    time, and existed because a committed job left the pending queue and could
    not be picked up by any other cohort. Nothing is committed in advance now,
    so the bound that matters is how many requests one cohort will serve before
    it winds down -- which is about bounding per-cohort bookkeeping, not about
    protecting jobs from being stranded.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(
        lane, _RefillAwareDriver(), max_requests_per_cohort=6
    )
    for i in range(7):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    snap = service.snapshot()
    assert snap["last_real_width"] == 2
    assert snap["max_requests_in_one_cohort"] == 6, "2 sealed + 4 pulled"
    assert snap["refill_admitted_total"] == 4
    assert snap["pending"] == 1, "the seventh waits for the next cohort"


def test_counters_stay_flat_when_nothing_is_refilled() -> None:
    """A cohort that never refills must not look like one that did."""

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=4))
    service = _service(lane, _RecordingDriver())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    snap = service.snapshot()
    assert snap["refill_admitted_total"] == 0
    assert snap["max_requests_in_one_cohort"] == 2
    assert snap["last_real_width"] == 2
    assert snap["continuous_batching_observed"] is False


# --------------------------------------------------------------------------- #
# P0-1: a driver failure must not leave any caller hanging
# --------------------------------------------------------------------------- #
def test_driver_failure_resolves_joiners_not_just_the_initial_cohort() -> None:
    """The silent-hang bug, as a test.

    The exception handler iterated the initial cohort only, so refill joiners
    got no exception, no result and no ownership finalisation: their futures
    never resolved and their callers hung until timeout. A cohort-mate that
    receives an error can retry; a caller that receives silence cannot even
    tell something went wrong, which is why this ranked above every other
    hardening item.
    """

    def _boom(runtime, prompts, **kwargs):
        # Pull FIRST. The bug this test protects is in the failure path's
        # handling of joiners, so the fixture has to produce joiners before it
        # explodes; a driver that raises before pulling has none and the test
        # would pass without exercising anything.
        _drain_pull({**kwargs, "prompts": prompts})
        raise RuntimeError("verify forward exploded")

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _boom, max_requests_per_cohort=4)
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(5)]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    # Two sealed the cohort and two more were pulled into it before the raise.
    # Every one of them must learn something -- the two in the cohort AND the
    # two joiners, which is the half that used to hang.
    in_run, still_queued = jobs[:4], jobs[4]
    for job in in_run:
        assert job.future.done(), f"{job.request_id} was left hanging"
        with pytest.raises(RuntimeError, match="verify forward exploded"):
            job.future.result(timeout=1)

    # r4 never entered the run. PENDING is not HANGING, and the difference
    # matters: a pending job is served by the next seal, a hanging one never
    # resolves. Asserting it is still queued rather than resolved is the
    # correct expectation, and it is served once the service pumps again.
    assert not still_queued.future.done(), "a queued job must not be failed"
    assert service.snapshot()["pending"] == 1


def test_driver_failure_finalises_ownership_for_every_job() -> None:
    """Cleanup must run even though the raise happens before the result loop.

    The driver call sat outside the try whose finally finalises ownership, so
    an exception propagated past the cleanup entirely: no MLX finalisation and
    jobs left holding finalize ownership.
    """

    finalized_for: list[str] = []

    def _boom(runtime, prompts, **kwargs):
        _drain_pull({**kwargs, "prompts": prompts})
        raise RuntimeError("kaboom")

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(
        lane, _boom, owner_finalize=lambda js: finalized_for.extend(
            j.request_id for j in js
        ) or {}
    )
    jobs = [_make_job(request_id=f"r{i}", lane=lane) for i in range(4)]
    for job in jobs:
        service.submit(job)
    service.pump_once()

    assert sorted(finalized_for) == ["r0", "r1", "r2", "r3"], (
        "owner finalize must see every job, joiners included"
    )
    for job in jobs:
        assert job.finalize_owner_finished, job.request_id


def test_cohort_failures_are_counted_for_the_3am_reader() -> None:
    """`last_error` is one overwritten string; a count is what shows a pattern."""

    def _boom(runtime, prompts, **kwargs):
        raise RuntimeError("nope")

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _boom)
    for round_index in range(2):
        for i in range(2):
            service.submit(
                _make_job(request_id=f"round{round_index}-r{i}", lane=lane)
            )
        service.pump_once()

    assert service.snapshot()["cohort_failures"] == 2


# --------------------------------------------------------------------------- #
# P0-3: backpressure — the queue must be bounded
# --------------------------------------------------------------------------- #
def test_queue_is_bounded_and_rejects_rather_than_growing() -> None:
    """Unbounded acceptance is the absence of a policy, not a policy.

    `submit()` appended with no cap, so under overload the queue grew until
    memory did. A caller told "busy, retry" can back off; a caller silently
    queued behind work it will never reach the front of cannot.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _RecordingDriver(), max_queue_depth=3)
    accepted = []
    for i in range(3):
        job = _make_job(request_id=f"ok{i}", lane=lane)
        service.submit(job)
        accepted.append(job)

    overflow = _make_job(request_id="overflow", lane=lane)
    with pytest.raises(DenseMTPBatchQueueFull, match="queue is full"):
        service.submit(overflow)

    # Rejected synchronously: it must NOT be left waiting on a cohort.
    assert not overflow.future.done() or overflow.future.exception() is not None
    assert service.snapshot()["rejected_total"] == 1
    assert service.snapshot()["max_queue_depth"] == 3
    # The accepted work is unaffected by the rejection. `_RecordingDriver`
    # never pulls, so the surplus needs a second seal -- which is the correct
    # model of a driver that never frees a row, and is why the loop is here
    # rather than a single pump.
    while service.pump_once():
        pass
    for job in accepted:
        assert job.future.done(), job.request_id


def test_queue_depth_defaults_to_a_bound_not_infinity() -> None:
    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=4))
    service = _service(lane, _RecordingDriver())
    assert service.max_queue_depth == 32, "eight cohorts' worth, but finite"


def test_rejection_frees_up_once_the_queue_drains() -> None:
    """Backpressure must be transient, not a latch."""

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    service = _service(lane, _RecordingDriver(), max_queue_depth=2)
    for i in range(2):
        service.submit(_make_job(request_id=f"first{i}", lane=lane))
    with pytest.raises(DenseMTPBatchQueueFull):
        service.submit(_make_job(request_id="rejected", lane=lane))

    service.pump_once()  # drains the queue
    later = _make_job(request_id="later", lane=lane)
    service.submit(later)  # must be accepted again
    service.pump_once()
    assert later.future.done()


# --------------------------------------------------------------------------- #
# P2: the 3am test — stalled vs slow
# --------------------------------------------------------------------------- #
def test_live_view_distinguishes_a_stalled_cohort_from_a_slow_one() -> None:
    """The question actually asked when a server stops responding.

    Duration cannot answer it: a five-minute cohort is healthy if its requests
    are long. Whether tokens are still arriving can. This drives the verdict
    from inside a running cohort, which is the only place it can be observed.
    """

    lane = _FakeLane()
    seen: list[dict] = []
    service = None

    class _SlowDriver(_RecordingDriver):
        def __call__(self, runtime, prompts, **kwargs):
            on_commit = kwargs.get("on_commit")
            # One token, then look at the live view from inside the cohort.
            on_commit(0, 111)
            seen.append(service.snapshot()["live_cohort"])
            return super().__call__(runtime, prompts, **kwargs)

    service = _service(lane, _SlowDriver())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    live = seen[0]
    assert live["state"] == "running"
    assert live["health"] == "healthy", "tokens are arriving, so not stalled"
    assert live["tokens_committed"] >= 1
    assert live["seconds_since_progress"] < 1.0
    assert live["width"] == 2

    # Between cohorts the view must say idle rather than report a stale cohort.
    assert service.snapshot()["live_cohort"]["state"] == "idle"


def test_a_cohort_committing_nothing_reports_STALLED() -> None:
    """A cohort past the stall threshold with no tokens must say so."""

    lane = _FakeLane()
    seen: list[dict] = []
    service = None

    class _StalledDriver(_RecordingDriver):
        def __call__(self, runtime, prompts, **kwargs):
            # Backdate the cohort's progress marker beyond the threshold, which
            # is what a genuinely wedged cohort looks like from outside.
            service._live["last_progress_s"] -= 999.0
            service._live["tokens_committed"] = 5
            seen.append(service.snapshot()["live_cohort"])
            return super().__call__(runtime, prompts, **kwargs)

    service = _service(lane, _StalledDriver())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    assert seen[0]["health"] == "STALLED"
    assert seen[0]["seconds_since_progress"] > 900


def test_snapshot_reports_duration_and_queue_wait_distributions() -> None:
    """A distribution shows drift where a running mean hides it."""

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    for round_index in range(3):
        for i in range(2):
            service.submit(
                _make_job(request_id=f"r{round_index}-{i}", lane=lane)
            )
        service.pump_once()

    snap = service.snapshot()
    assert snap["cohort_duration_s"]["n"] == 3
    assert set(snap["cohort_duration_s"]) >= {"n", "p50", "p90", "p99", "max"}
    assert snap["queue_wait_s"]["n"] == 6
    assert snap["queue_wait_s"]["p50"] >= 0.0


def test_percentiles_on_an_empty_window_are_empty_not_zero() -> None:
    """Absence of data must not read as a measurement of zero."""

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    snap = service.snapshot()
    assert snap["cohort_duration_s"] == {}
    assert snap["queue_wait_s"] == {}
    assert snap["live_cohort"] == {"state": "idle"}




def test_snapshot_counts_committed_refill_as_backlog() -> None:
    """Found by mutation: reverting the queue-depth fix kept the suite green.

    Jobs moved from _pending into a running cohort's refill queue are still
    backlog -- they have not been served. Reporting only _pending understates
    the queue, and the backpressure bound is measured against that number.
    """

    # Two slots against six jobs, so four must land in refill rather than the
    # cohort. Without the narrow cohort the fixture cannot produce the state it
    # claims to measure -- which is the guard below.
    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    seen: list[dict] = []

    class _Peeking(_RefillAwareDriver):
        def __call__(self, runtime, prompts, **kwargs):
            # Pull first, THEN snapshot: committed-but-unserved work only
            # exists once the driver has taken it, and that is exactly the
            # state this test measures.
            pulled = _drain_pull({**kwargs, "prompts": prompts})
            seen.append(service.snapshot())
            return super().__call__(
                runtime,
                prompts,
                **{**kwargs, "refill_queue": pulled, "pull_queued": None},
            )

    service = _service(lane, _Peeking())
    for i in range(6):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    snap = seen[0]
    assert snap["pending"] == snap["pending_new"] + snap["pending_refill"], (
        "total backlog must be the sum of its parts"
    )
    assert snap["pending_refill"] > 0, "fixture must actually produce refill"
    assert snap["pending"] > snap["pending_new"], (
        "committed refill vanished from the reported queue depth"
    )


# --------------------------------------------------------------------------- #
# Guarantees I reasoned were safe, which nothing was checking
# --------------------------------------------------------------------------- #
def test_different_stop_sets_never_share_a_cohort() -> None:
    """The driver takes jobs[0].stop_token_ids for the WHOLE cohort.

    That is only safe because the compatibility key binds the stop set, so
    cohort-mates have identical stop sets by construction. I checked that
    mechanism by reading it and concluded it was safe -- and a mutation audit
    showed nothing was verifying it. Remove the stop set from the key and every
    test still passed, while callers would silently inherit jobs[0]'s stop
    tokens and stop in the wrong place.

    Reading a mechanism is not the same as checking it.
    """

    lane = _FakeLane()
    driver = _RecordingDriver()
    service = _service(lane, driver)
    # Two of each, because a cohort of one takes the solo path and never
    # reaches the driver -- so a single job per stop set would prove nothing.
    for name, stop in (("a", 5), ("b", 5), ("c", 6), ("d", 6)):
        service.submit(
            _make_job(request_id=name, lane=lane, stop_token_ids={stop})
        )
    service.pump_once()
    service.pump_once()

    assert len(driver.calls) == 2, (
        "expected one cohort per stop set; got "
        f"{len(driver.calls)} driver calls"
    )
    for call in driver.calls:
        assert len(call["prompts"]) == 2, (
            "a cohort mixed the two stop sets, so one caller's stop tokens "
            "would be applied to the other"
        )
        assert len(set(call["stop_token_ids"])) == 1


def test_an_incompatible_job_is_not_pulled_into_a_running_cohort() -> None:
    """Refill must respect the compatibility key, not merely the queue order.

    A job admitted into a cohort it is incompatible with would be decoded under
    that cohort's route and stop set -- the same contamination as above, just
    arriving through the refill path instead of the initial seal.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    driver = _RecordingDriver()
    service = _service(lane, driver)
    for i in range(2):
        service.submit(
            _make_job(request_id=f"same{i}", lane=lane, stop_token_ids={7})
        )
    # Queued behind them, and incompatible: it must wait for its own cohort.
    service.submit(
        _make_job(request_id="other", lane=lane, stop_token_ids={8})
    )
    service.pump_once()

    first = driver.calls[0]
    assert len(first["prompts"]) == 2, "the two compatible jobs form the cohort"
    # Ask the live queue directly rather than reading an empty refill list.
    # After the pre-move was removed, `refill_queue` is ALWAYS empty, so
    # asserting on it here would have passed no matter what the queue did --
    # a guarantee protected by a check that cannot fail is not protected.
    assert first["pull_queued"](4) == [], (
        "an incompatible job was offered to the driver as a joiner"
    )


def test_each_response_reports_the_width_it_decoded_at() -> None:
    """The contract tells callers to pin the WIDTH, not just the seed.

    Advice a caller cannot act on is not advice. Without this field there is no
    way to learn what width a request got, so "pin the width" was unactionable
    and two runs that differed only by geometry were indistinguishable from two
    that differed for a real reason.
    """

    # TWO different widths, because a field hardcoded to any single value --
    # or to 1 -- would pass a one-width test and then make every geometry
    # comparison silently agree with itself. A stats field is an instrument
    # too, and it needs the same positive demonstration as any other.
    for expected in (2, 4):
        lane = _FakeLane()
        service = _service(lane, _RecordingDriver())
        futures = [
            service.submit(_make_job(request_id=f"r{i}", lane=lane))
            for i in range(expected)
        ]
        service.pump_once()
        widths = {
            f.result()["stats"]["dense_mtp_batch_cohort_width"] for f in futures
        }
        assert widths == {expected}, (
            f"a cohort of {expected} reported {widths}"
        )


def test_a_solo_request_reports_width_one() -> None:
    """A caller must learn its width whether or not it got batched.

    Without this the field appears only when batched, so a single-request
    baseline has no width to compare a batched run against -- which is exactly
    the comparison the contract tells callers to make.
    """

    lane = _FakeLane()
    service = _service(lane, _RecordingDriver())
    future = service.submit(_make_job(request_id="lonely", lane=lane))
    service.pump_once()
    stats = future.result()["stats"]
    assert stats["dense_mtp_batch_cohort_width"] == 1


def test_stall_verdict_flips_at_its_threshold_not_somewhere_else() -> None:
    """A second, structurally different check on the stall discriminator.

    The existing test backdates progress by 999s. This one checks the verdict
    flips at the CONFIGURED threshold rather than at some other constant the
    comparison might be using.

    Note what this test canNOT do, since I first claimed otherwise and the
    mutation audit disagreed: it reads `service.stall_threshold_s` and scales
    from it, so it is invariant to that value and will never catch a threshold
    set to the wrong number. Moving 120 to 1200 leaves it green; the far-past
    test is what catches that. The absolute value is pinned separately below.

    The soak's entire reading rests on this verdict, which is why it is worth
    two tests that fail for different reasons.
    """

    lane = _FakeLane()
    seen: list[dict] = []
    service = None

    class _Probing(_RecordingDriver):
        def __call__(self, runtime, prompts, **kwargs):
            kwargs.get("on_commit")(0, 1)
            threshold = service.stall_threshold_s
            for offset in (threshold * 0.9, threshold * 1.1):
                service._live["last_progress_s"] = (
                    time.perf_counter() - offset
                )
                seen.append(service.snapshot()["live_cohort"])
            return super().__call__(runtime, prompts, **kwargs)

    service = _service(lane, _Probing())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    inside, outside = seen
    assert inside["health"] == "healthy", (
        f"{inside['seconds_since_progress']:.1f}s is inside the "
        f"{service.stall_threshold_s}s threshold and must not read STALLED"
    )
    assert outside["health"] == "STALLED", (
        f"{outside['seconds_since_progress']:.1f}s is past the "
        f"{service.stall_threshold_s}s threshold and must read STALLED"
    )


def test_the_stall_threshold_is_the_documented_two_minutes() -> None:
    """Pin the VALUE, which the boundary test structurally cannot.

    The boundary test scales from whatever the threshold happens to be, so it
    stays green if the number is wrong. An operator reading `health: STALLED`
    is being told "no tokens for two minutes"; if that silently became two
    hours, every stall would look healthy until long past useful.
    """

    service = _service(_FakeLane(), _RecordingDriver())
    assert service.stall_threshold_s == 120.0
    assert service.snapshot()["live_cohort"] == {"state": "idle"}

def test_backpressure_counts_refill_toward_the_bound_not_just_pending() -> None:
    """Second guard on the refill-accounting fix, from the bound's side.

    The existing test checks what `snapshot()` REPORTS. This checks what the
    BOUND does, which is the half that protects the server: jobs committed to a
    running cohort's refill queue are unserved work, and a bound that ignores
    them admits more than it was configured to hold.

    Arranging the discriminating case is fiddlier than it looks, and the first
    version of this test got it wrong. Total backlog DROPS at seal -- the cohort
    rows leave the queue -- so during a cohort the total is always below what it
    was before. The bound therefore only bites when new work arrives mid-cohort.
    So: submit from inside the running cohort until one is refused, then check
    that at the moment of refusal `pending_new` was still UNDER the bound. If it
    was, the only thing that can have triggered the refusal is the refill count.
    """

    lane = _FakeLane(geometry=_FakeGeometry(cohort_slots=2))
    outcome: dict[str, Any] = {}
    service = None

    class _SubmitDuringCohort(_RefillAwareDriver):
        def __call__(self, runtime, prompts, **kwargs):
            # Take the waiting work into the cohort first, so `pending_refill`
            # is non-zero when the late submits start arriving. That is the
            # state the bound is being tested against.
            pulled = _drain_pull({**kwargs, "prompts": prompts})
            kwargs = {**kwargs, "refill_queue": pulled, "pull_queued": None}
            for n in range(10):
                snap = service.snapshot()
                try:
                    service.submit(
                        _make_job(request_id=f"late{n}", lane=lane)
                    )
                except DenseMTPBatchQueueFull as exc:
                    outcome["at_refusal"] = snap
                    outcome["message"] = str(exc)
                    break
            return super().__call__(runtime, prompts, **kwargs)

    service = _service(lane, _SubmitDuringCohort(), max_queue_depth=5)
    for i in range(5):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    assert "at_refusal" in outcome, "no submit was ever refused"
    snap = outcome["at_refusal"]
    assert snap["pending_refill"] > 0, "fixture must actually populate refill"
    assert snap["pending_new"] < service.max_queue_depth, (
        f"pending_new was {snap['pending_new']} against a bound of "
        f"{service.max_queue_depth}, so this refusal does not prove refill was "
        "counted -- pending alone would have triggered it"
    )
    assert snap["pending"] >= service.max_queue_depth, (
        "the refusal fired before the TOTAL backlog reached the bound"
    )


def test_a_cohort_still_prefilling_is_starting_not_stalled() -> None:
    """Prefill produces no tokens, and that is not a stall.

    Eight rows of several thousand prompt tokens each take well over the 120s
    progress threshold to read, committing nothing the whole time. The first
    version checked STALLED first, which made the `starting` branch unreachable
    past that threshold -- so a healthy cohort reading its prompts reported
    STALLED, and the soak's STOP condition reads exactly this field. It would
    have aborted a good run and recorded it as a lane failure.

    Observed in a soak smoke run, which is the only reason it was found: no
    unit fixture prefills for two minutes.
    """

    lane = _FakeLane()
    seen: list[dict] = []
    service = None

    class _Prefilling(_RecordingDriver):
        def __call__(self, runtime, prompts, **kwargs):
            # No token has been committed, and the cohort is well past the
            # progress threshold but inside the prefill budget.
            service._live["started_s"] -= 300.0
            service._live["last_progress_s"] -= 300.0
            seen.append(service.snapshot()["live_cohort"])
            # And past the prefill budget it must finally say STALLED, or a
            # genuinely wedged prefill would never be reported at all.
            service._live["started_s"] -= 500.0
            service._live["last_progress_s"] -= 500.0
            seen.append(service.snapshot()["live_cohort"])
            return super().__call__(runtime, prompts, **kwargs)

    service = _service(lane, _Prefilling())
    for i in range(2):
        service.submit(_make_job(request_id=f"r{i}", lane=lane))
    service.pump_once()

    inside, beyond = seen
    assert inside["tokens_committed"] == 0
    assert inside["health"] == "starting", (
        f"a cohort {inside['age_s']:.0f}s into prefill with no tokens must be "
        "'starting'; reporting STALLED aborts healthy soaks"
    )
    assert beyond["health"] == "STALLED", (
        f"{beyond['age_s']:.0f}s with no tokens is past the "
        f"{service.start_threshold_s}s prefill budget and must be STALLED"
    )
