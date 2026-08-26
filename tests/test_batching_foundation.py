from __future__ import annotations

from mtplx.batching import (
    ARBatchKey,
    AdmissionPolicy,
    AdmissionStatus,
    BatchSchedulerConfig,
    MTPBatchKey,
    MTPContinuousScheduler,
    MemoryPressure,
    RequestPhase,
    RequestPriority,
    RequestState,
    SchedulerMode,
    SchedulerPreset,
    StepResult,
)


class FakeHooks:
    def __init__(self) -> None:
        self.prefill_order: list[str] = []
        self.decode_batches: list[list[str]] = []
        self.postcommit_order: list[str] = []

    def prefill_step(self, request: RequestState, *, token_budget: int) -> StepResult:
        self.prefill_order.append(request.request_id)
        remaining = len(request.prompt_ids) - request.prompt_tokens_done
        consumed = min(token_budget, remaining)
        next_phase = (
            RequestPhase.DECODE_READY
            if request.prompt_tokens_done + consumed >= len(request.prompt_ids)
            else RequestPhase.PREFILLING
        )
        return StepResult(next_phase=next_phase, prompt_tokens_done=consumed)

    def decode_step(self, requests: list[RequestState]) -> list[StepResult]:
        self.decode_batches.append([request.request_id for request in requests])
        return [
            StepResult(
                next_phase=(
                    RequestPhase.FINISHED
                    if request.tokens_generated + 1 >= request.max_tokens
                    else RequestPhase.DECODE_READY
                ),
                generated_tokens=1,
                finished=request.tokens_generated + 1 >= request.max_tokens,
            )
            for request in requests
        ]

    def postcommit_step(self, request: RequestState) -> StepResult:
        self.postcommit_order.append(request.request_id)
        return StepResult(next_phase=RequestPhase.FINISHED, finished=True)


def test_admission_policy_shrinks_under_memory_pressure():
    policy = AdmissionPolicy.from_preset("agent")
    assert (
        policy.classify_memory(active_memory_bytes=80, total_memory_bytes=100)
        == MemoryPressure.NORMAL
    )
    assert (
        policy.classify_memory(active_memory_bytes=86, total_memory_bytes=100)
        == MemoryPressure.SOFT
    )
    assert (
        policy.classify_memory(active_memory_bytes=93, total_memory_bytes=100)
        == MemoryPressure.HARD
    )

    request = RequestState("background", priority=RequestPriority.BACKGROUND)
    decision = policy.decide(
        request,
        active_count=0,
        active_memory_bytes=93,
        total_memory_bytes=100,
    )

    assert decision.status == AdmissionStatus.WAIT
    assert decision.reason == "hard_memory_pressure"
    assert decision.effective_decode_batch_max == 1


def test_agent_preset_is_opencode_fair_by_default():
    config = BatchSchedulerConfig.from_values(mode="ar_batch", preset="agent")

    assert config.to_dict()["max_active_requests"] == 4
    assert config.to_dict()["decode_batch_max"] == 4
    assert config.to_dict()["batch_wait_ms"] == 50.0
    assert config.to_dict()["prefill_chunk_tokens"] == 2048


def test_latency_preset_is_true_solo_mtp():
    config = BatchSchedulerConfig.from_values(mode="serial", preset="latency")

    assert config.to_dict()["max_active_requests"] == 1
    assert config.to_dict()["decode_batch_max"] == 1
    assert config.to_dict()["batch_wait_ms"] == 0.0
    assert config.to_dict()["prefill_chunk_tokens"] == 1024


def test_mtp_batch_config_is_fixed_width_eight():
    config = BatchSchedulerConfig.from_values(
        mode="mtp_batch",
        preset="throughput",
        max_active_requests=8,
        decode_batch_max=8,
    )

    assert config.mode is SchedulerMode.MTP_BATCH
    assert config.to_dict()["max_active_requests"] == 8
    assert config.to_dict()["decode_batch_max"] == 8


def test_batch_keys_are_stable_and_separate_ar_from_mtp():
    request = RequestState(
        "r1",
        sampler={"temperature": 0.6, "top_p": 0.95},
        stop_token_ids={1, 2},
    )
    ar_key = ARBatchKey.from_request(
        request,
        model_id="model-a",
        tokenizer_template_hash="template",
    )
    mtp_key = MTPBatchKey(
        model_id="model-a",
        quant_policy="q4",
        speculative_depth=3,
        verify_width=4,
        mtp_hidden_variant="post_norm",
        mtp_history_policy="committed",
        cache_kind="dynamic_paged_kv",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert ar_key.as_batch_key().startswith("ar|model-a|template")
    assert mtp_key.as_batch_key().startswith("mtp|model-a|q4|3|4")
    assert ar_key.as_batch_key() != mtp_key.as_batch_key()


def test_cooperative_scheduler_batches_ar_decode_ready_requests():
    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.AR_BATCH,
        preset=SchedulerPreset.AGENT,
        max_active_requests=4,
        decode_batch_max=2,
        prefill_chunk_tokens=8,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    scheduler.submit(RequestState("r1", prompt_ids=[1, 2], max_tokens=1))
    scheduler.submit(RequestState("r2", prompt_ids=[3, 4], max_tokens=1))

    scheduler.run_until_idle()

    assert hooks.prefill_order == ["r1", "r2"]
    assert hooks.decode_batches == [["r1", "r2"]]
    assert hooks.postcommit_order == ["r1", "r2"]
    snapshot = scheduler.snapshot()
    assert snapshot["finished"] == 2
    assert snapshot["stats"]["batch_histogram"] == {"2": 1}
    assert snapshot["stats"]["last_mtp_disabled_reason"] == "batch_size_gt_1"


def test_mtp_batch_scheduler_does_not_mark_parallel_decode_as_ar_fallback():
    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.MTP_BATCH,
        preset=SchedulerPreset.THROUGHPUT,
        max_active_requests=8,
        decode_batch_max=8,
        prefill_chunk_tokens=8,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    scheduler.submit(RequestState("r1", prompt_ids=[1, 2], max_tokens=1))
    scheduler.submit(RequestState("r2", prompt_ids=[3, 4], max_tokens=1))

    scheduler.run_until_idle()

    snapshot = scheduler.snapshot()
    assert hooks.decode_batches == [["r1", "r2"]]
    assert snapshot["stats"]["batch_histogram"] == {"2": 1}
    assert snapshot["stats"]["last_mtp_disabled_reason"] is None


def test_experimental_cohort_mode_is_not_reported_as_mtp_disabled():
    """`mtp_cohort_experimental` is an opt-IN cohort mode, not an absence of one.

    The gate tested only `!= MTP_BATCH`, so the experimental mode -- whose whole
    purpose is batched MTP decode -- reported speculation as disabled every time
    it decoded a cohort wider than one.
    """

    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.MTP_COHORT_EXPERIMENTAL,
        preset=SchedulerPreset.THROUGHPUT,
        max_active_requests=8,
        decode_batch_max=8,
        prefill_chunk_tokens=8,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    scheduler.submit(RequestState("r1", prompt_ids=[1, 2], max_tokens=1))
    scheduler.submit(RequestState("r2", prompt_ids=[3, 4], max_tokens=1))

    scheduler.run_until_idle()

    snapshot = scheduler.snapshot()
    assert hooks.decode_batches == [["r1", "r2"]]
    assert snapshot["stats"]["last_mtp_disabled_reason"] is None


def test_the_mtp_disabled_reason_clears_when_it_stops_applying():
    """It describes the LAST decode batch, so it has to be able to go away.

    The reason was only ever set, never cleared, so one wide cohort in a
    non-MTP mode made the scheduler report MTP as disabled for the rest of its
    life -- including after the condition stopped holding. A stale diagnostic
    that never recovers is worse than none, because it is read as current.
    """

    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.AR_BATCH,
        preset=SchedulerPreset.AGENT,
        max_active_requests=4,
        decode_batch_max=2,
        prefill_chunk_tokens=8,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    scheduler.submit(RequestState("r1", prompt_ids=[1, 2], max_tokens=1))
    scheduler.submit(RequestState("r2", prompt_ids=[3, 4], max_tokens=1))
    scheduler.run_until_idle()
    assert scheduler.stats.last_mtp_disabled_reason == "batch_size_gt_1"

    # A subsequent SOLO decode batch is not an MTP-disabled situation.
    scheduler.submit(RequestState("r3", prompt_ids=[5, 6], max_tokens=1))
    scheduler.run_until_idle()
    assert scheduler.stats.last_mtp_disabled_reason is None, (
        "the reason survived the condition that produced it"
    )


def test_cooperative_scheduler_cancellation_finishes_once():
    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.COOPERATIVE,
        preset=SchedulerPreset.LATENCY,
        max_active_requests=2,
        prefill_chunk_tokens=1,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    request = RequestState("r1", prompt_ids=list(range(300)), max_tokens=1)
    scheduler.submit(request)

    assert scheduler.step() is True
    assert request.phase == RequestPhase.PREFILLING
    assert scheduler.cancel("r1") is True
    scheduler.run_until_idle()

    assert request.phase == RequestPhase.CANCELLED
    assert len(scheduler.finished) == 1
    assert scheduler.snapshot()["stats"]["cancelled"] == 1


def test_finished_tracking_is_bounded_but_total_is_exact():
    hooks = FakeHooks()
    config = BatchSchedulerConfig(
        mode=SchedulerMode.AR_BATCH,
        preset=SchedulerPreset.AGENT,
        max_active_requests=4,
        decode_batch_max=4,
        prefill_chunk_tokens=8,
    )
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    scheduler.max_finished_retained = 8

    total = 50
    for i in range(total):
        scheduler.submit(RequestState(f"r{i}", prompt_ids=[i, i + 1], max_tokens=1))
    scheduler.run_until_idle()

    # Retained finished state stays bounded instead of growing with every
    # request, but the reported total remains exact.
    assert len(scheduler.finished) <= 8
    assert scheduler.finished_total == total
    snapshot = scheduler.snapshot()
    assert snapshot["finished"] == total
    assert snapshot["finished_retained"] <= 8


def test_record_finished_is_idempotent_per_request():
    hooks = FakeHooks()
    config = BatchSchedulerConfig(mode=SchedulerMode.SERIAL, preset=SchedulerPreset.LATENCY)
    scheduler = MTPContinuousScheduler(config=config, hooks=hooks)
    request = RequestState("r1", prompt_ids=[1, 2], max_tokens=1)

    scheduler._record_finished(request)
    scheduler._record_finished(request)

    assert scheduler.finished_total == 1
    assert len(scheduler.finished) == 1
    assert scheduler.snapshot()["finished"] == 1
