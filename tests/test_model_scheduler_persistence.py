"""Persistence-band scheduling: foreground > postcommit > persistence.

The 2026-08-06 causal probe showed FIFO idle ordering let 1-2s SSD cold
encodes displace the canonical postcommit whose entry anchors the NEXT
turn's restore. The persistence band fixes the contract deterministically:
queued postcommit outranks earlier-queued persistence; persistence waits a
QUIET GRACE anchored to the most recent foreground/postcommit COMPLETION
(a submission-time grace expires during a long generation and would
release cold work in the few-ms gap before the server tail submits its
postcommit); running work is never preempted; persistence drains after a
genuine quiet window — there is deliberately no age-based valve, and
continuous latency-critical work may defer background durability.
"""

from __future__ import annotations

import time
from threading import Event

from mtplx.model_scheduler import ModelWorkScheduler


def _scheduler(**kwargs) -> ModelWorkScheduler:
    defaults = dict(
        name="test-persistence-scheduler",
        idle_grace_s=0.0,
        persistence_quiet_grace_s=0.05,
    )
    defaults.update(kwargs)
    return ModelWorkScheduler(**defaults)


def test_priority_foreground_over_postcommit_over_persistence():
    scheduler = _scheduler()
    order: list[str] = []
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)
        order.append("foreground-1")

    try:
        first = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        # Queue all three bands while the owner thread is busy.
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence")
        )
        postcommit = scheduler.submit_idle_postcommit(
            lambda: order.append("postcommit")
        )
        second = scheduler.submit_foreground(lambda: order.append("foreground-2"))
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        postcommit.result(timeout=2)
        persistence.result(timeout=2)
        assert order == [
            "foreground-1",
            "foreground-2",
            "postcommit",
            "persistence",
        ]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_tail_postcommit_beats_persistence_queued_during_long_foreground():
    """THE race: persistence is queued during a long generation, so any
    submission-time grace has long expired when the generation ends. The
    quiet grace is anchored to the foreground COMPLETION, so the cold job
    must still yield to the postcommit the server tail submits a few ms
    after the foreground future resolves."""
    scheduler = _scheduler(persistence_quiet_grace_s=0.15)
    order: list[str] = []
    started = Event()
    release = Event()

    def generation() -> None:
        started.set()
        assert release.wait(timeout=2)
        order.append("foreground")

    try:
        foreground = scheduler.submit_foreground(generation)
        assert started.wait(timeout=2)
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence")
        )
        # Let far more than the grace elapse while the generation runs: a
        # submission-anchored grace would now read "ready".
        time.sleep(0.2)
        release.set()
        foreground.result(timeout=2)
        # Server tail submits the canonical postcommit a few ms later.
        time.sleep(0.005)
        postcommit = scheduler.submit_idle_postcommit(
            lambda: order.append("postcommit")
        )
        postcommit.result(timeout=2)
        persistence.result(timeout=2)
        assert order == ["foreground", "postcommit", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_later_postcommit_beats_earlier_queued_persistence():
    scheduler = _scheduler()
    order: list[str] = []
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)

    try:
        foreground = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence")
        )
        postcommit = scheduler.submit_idle_postcommit(
            lambda: order.append("postcommit")
        )
        release.set()
        foreground.result(timeout=2)
        postcommit.result(timeout=2)
        persistence.result(timeout=2)
        assert order == ["postcommit", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_persistence_drains_after_quiet_without_other_activity():
    scheduler = _scheduler(persistence_quiet_grace_s=0.05)
    try:
        done: list[str] = []
        persistence = scheduler.submit_idle_persistence(lambda: done.append("ran"))
        persistence.result(timeout=2)
        assert done == ["ran"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_persistence_never_bypasses_quiet_grace_after_long_foreground():
    """Regression guard for the removed max-defer valve: an item queued
    during a foreground many multiples longer than any age clock must STILL
    wait out the completion-anchored quiet grace — a >valve-length
    generation would otherwise dequeue cold work in the tail gap. (0.5s
    generation stands in for the >30s case; the item's age at completion is
    10x the grace, exactly the stale-clock shape.)"""
    scheduler = _scheduler(persistence_quiet_grace_s=0.05)
    order: list[str] = []
    started = Event()
    release = Event()

    def long_generation() -> None:
        started.set()
        assert release.wait(timeout=5)
        order.append("foreground")

    try:
        foreground = scheduler.submit_foreground(long_generation)
        assert started.wait(timeout=2)
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence")
        )
        time.sleep(0.5)  # item age >> grace by completion time
        release.set()
        foreground.result(timeout=2)
        time.sleep(0.005)  # server tail submits a few ms after resolution
        postcommit = scheduler.submit_idle_postcommit(
            lambda: order.append("postcommit")
        )
        postcommit.result(timeout=2)
        persistence.result(timeout=2)
        assert order == ["foreground", "postcommit", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_persistence_defers_under_recurring_activity_by_design():
    """With no valve, recurring latency-critical completions keep deferring
    durability — the documented trade: never race the tail gap."""
    scheduler = _scheduler(persistence_quiet_grace_s=60.0)
    try:
        persistence = scheduler.submit_idle_persistence(lambda: None)
        for _ in range(3):
            scheduler.submit_idle_postcommit(lambda: None).result(timeout=2)
        time.sleep(0.05)
        assert not persistence.done()
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
        assert persistence.cancelled()


def test_foreground_immediacy_with_eligible_persistence():
    scheduler = _scheduler(persistence_quiet_grace_s=0.0)
    order: list[str] = []
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)
        order.append("foreground-1")

    try:
        first = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence")
        )
        second = scheduler.submit_foreground(lambda: order.append("foreground-2"))
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        persistence.result(timeout=2)
        assert order == ["foreground-1", "foreground-2", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_cancel_queued_persistence_before_start():
    scheduler = _scheduler(persistence_quiet_grace_s=5.0)
    try:
        persistence = scheduler.submit_idle_persistence(lambda: None)
        assert persistence.cancel() is True
        stats_deadline = time.monotonic() + 2.0
        while time.monotonic() < stats_deadline:
            if scheduler.stats()["persistence_pending"] == 1:
                break
            time.sleep(0.01)
        # The queued item stays cancelled; the run loop skips it exactly like
        # other queued cancellations.
        assert persistence.cancelled()
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_shutdown_cancel_futures_covers_persistence_queue():
    scheduler = _scheduler(persistence_quiet_grace_s=5.0)
    persistence = scheduler.submit_idle_persistence(lambda: None)
    scheduler.shutdown(wait=True, cancel_futures=True)
    assert persistence.cancelled()


def test_coalesce_key_keeps_at_most_one_pending_and_newest_wins():
    scheduler = _scheduler(persistence_quiet_grace_s=0.05)
    started = Event()
    release = Event()
    ran: list[str] = []

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)

    try:
        foreground = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        futures = [
            scheduler.submit_idle_persistence(
                lambda i=i: ran.append(f"job-{i}"),
                coalesce_key="ssd_cold:session-1",
            )
            for i in range(4)
        ]
        stats = scheduler.stats()
        assert stats["persistence_pending"] == 1
        assert stats["persistence_coalesced"] == 3
        assert all(f.cancelled() for f in futures[:3])
        release.set()
        foreground.result(timeout=2)
        futures[-1].result(timeout=2)
        assert ran == ["job-3"], "only the NEWEST submission may run"
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_args_and_kwargs_survive_coalescing_and_reach_only_newest_job():
    """Compatibility parity with submit_idle_postcommit: positional AND
    ordinary keyword arguments forward to the target callable; batch_key
    and coalesce_key stay scheduler controls. Under newest-wins only the
    newest submission's payload runs."""
    scheduler = _scheduler(persistence_quiet_grace_s=0.05)
    started = Event()
    release = Event()
    ran: list[tuple] = []

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)

    def job(a, b, *, value):
        ran.append((a, b, value))

    try:
        foreground = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        stale = scheduler.submit_idle_persistence(
            job, "stale", 1, value=10, coalesce_key="k"
        )
        newest = scheduler.submit_idle_persistence(
            job, "newest", 2, value=3, coalesce_key="k"
        )
        release.set()
        foreground.result(timeout=2)
        newest.result(timeout=2)
        assert stale.cancelled()
        assert ran == [("newest", 2, 3)]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_superseded_persistence_closure_is_released():
    import gc
    import weakref

    scheduler = _scheduler(persistence_quiet_grace_s=5.0)

    class Payload:
        pass

    try:
        payload = Payload()
        ref = weakref.ref(payload)

        def job(p=payload) -> None:
            _ = p

        scheduler.submit_idle_persistence(job, coalesce_key="k")
        del job, payload
        gc.collect()
        assert ref() is not None, "queued closure must pin its payload"
        replacement = scheduler.submit_idle_persistence(
            lambda: None, coalesce_key="k"
        )
        gc.collect()
        assert ref() is None, "superseded closure must release its payload"
        assert not replacement.done()
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_different_coalesce_keys_both_drain():
    scheduler = _scheduler(persistence_quiet_grace_s=0.02)
    ran: list[str] = []
    try:
        a = scheduler.submit_idle_persistence(
            lambda: ran.append("a"), coalesce_key="session-a"
        )
        b = scheduler.submit_idle_persistence(
            lambda: ran.append("b"), coalesce_key="session-b"
        )
        a.result(timeout=2)
        b.result(timeout=2)
        assert sorted(ran) == ["a", "b"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_running_persistence_item_is_never_cancelled_by_coalescing():
    scheduler = _scheduler(persistence_quiet_grace_s=0.0)
    running = Event()
    release = Event()
    ran: list[str] = []

    def slow_job() -> None:
        running.set()
        assert release.wait(timeout=2)
        ran.append("first")

    try:
        first = scheduler.submit_idle_persistence(slow_job, coalesce_key="k")
        assert running.wait(timeout=2)
        second = scheduler.submit_idle_persistence(
            lambda: ran.append("second"), coalesce_key="k"
        )
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        assert ran == ["first", "second"]
        assert not first.cancelled()
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_uncoalesced_submissions_never_coalesce():
    scheduler = _scheduler(persistence_quiet_grace_s=5.0)
    try:
        futures = [
            scheduler.submit_idle_persistence(lambda: None) for _ in range(3)
        ]
        stats = scheduler.stats()
        assert stats["persistence_pending"] == 3
        assert stats["persistence_coalesced"] == 0
        assert not any(f.cancelled() for f in futures)
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_postcommit_still_beats_coalesced_persistence():
    scheduler = _scheduler(persistence_quiet_grace_s=0.05)
    order: list[str] = []
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)

    try:
        foreground = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        scheduler.submit_idle_persistence(
            lambda: order.append("stale"), coalesce_key="k"
        )
        persistence = scheduler.submit_idle_persistence(
            lambda: order.append("persistence"), coalesce_key="k"
        )
        postcommit = scheduler.submit_idle_postcommit(
            lambda: order.append("postcommit")
        )
        release.set()
        foreground.result(timeout=2)
        postcommit.result(timeout=2)
        persistence.result(timeout=2)
        assert order == ["postcommit", "persistence"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_shutdown_cancels_coalesced_queue_correctly():
    scheduler = _scheduler(persistence_quiet_grace_s=5.0)
    stale = scheduler.submit_idle_persistence(lambda: None, coalesce_key="k")
    newest = scheduler.submit_idle_persistence(lambda: None, coalesce_key="k")
    scheduler.shutdown(wait=True, cancel_futures=True)
    assert stale.cancelled()
    assert newest.cancelled()


def test_bank_cold_jobs_carry_session_coalesce_key():
    from pathlib import Path
    from types import SimpleNamespace

    from mtplx.cache_state import CacheSnapshot
    from mtplx.session_bank import SessionBank

    dispatched: list = []
    bank = SessionBank(
        max_entries=4,
        max_bytes=4096,
        per_session_max_bytes=4096,
        cold_tier=SimpleNamespace(put_entry=lambda entry, capabilities=None: True),
    )
    bank.cold_enqueue_dispatch = dispatched.append
    bank.put_snapshot(
        runtime=SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True),
        token_ids=[1, 2, 3],
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        session_id="session-42",
        snapshot_epoch=3,
        nbytes_override=64,
    )
    assert len(dispatched) == 1
    assert getattr(dispatched[0], "coalesce_key", None) == "ssd_cold:session-42"
    assert getattr(dispatched[0], "pinned_bytes", None) == 64


def test_capability_marker_and_legacy_fallback_shape():
    """Server wiring gates on the explicit capability attribute; a legacy
    scheduler exposing only submit_idle_postcommit keeps the old lane."""
    assert ModelWorkScheduler.SUPPORTS_IDLE_PERSISTENCE is True

    class Legacy:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def submit_idle_postcommit(self, job, *, batch_key=None):
            self.calls.append(str(batch_key))
            job()

    legacy = Legacy()
    # Mirror the wiring's capability choice exactly.
    if getattr(legacy, "SUPPORTS_IDLE_PERSISTENCE", False):
        raise AssertionError("legacy scheduler must not advertise the band")
    legacy.submit_idle_postcommit(lambda: None, batch_key="ssd.cold_enqueue")
    assert legacy.calls == ["ssd.cold_enqueue"]


def test_completed_persistence_closure_released_while_worker_parks_idle():
    import gc
    import weakref

    scheduler = _scheduler(persistence_quiet_grace_s=0.02)

    class Payload:
        pass

    try:
        payload = Payload()
        ref = weakref.ref(payload)

        def job(p=payload) -> None:
            _ = p

        future = scheduler.submit_idle_persistence(job, coalesce_key="k")
        del job, payload
        future.result(timeout=2)
        # After completion the worker loops back into _take_next and parks
        # idle; the run frame must not keep the finished item (and its
        # snapshot closure) alive across that park. Bounded poll: pre-fix
        # the pin is indefinite (frame local survives the park), post-fix
        # release is refcount-prompt.
        deadline = time.monotonic() + 2.0
        while ref() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert ref() is None, (
            "completed closure must be released once the worker parks idle"
        )
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_cancelled_persistence_closure_released_while_worker_parks_idle():
    import gc
    import weakref

    scheduler = _scheduler(persistence_quiet_grace_s=0.02)

    class Payload:
        pass

    started = Event()
    release = Event()

    def blocker():
        started.set()
        assert release.wait(timeout=2)

    try:
        payload = Payload()
        ref = weakref.ref(payload)

        def job(p=payload) -> None:
            _ = p

        foreground = scheduler.submit_foreground(blocker)
        assert started.wait(timeout=2)
        future = scheduler.submit_idle_persistence(job, coalesce_key="k")
        assert future.cancel()
        del job, payload
        release.set()
        foreground.result(timeout=2)
        # After the quiet grace the worker dequeues the canceled item,
        # set_running_or_notify_cancel() returns False, and the loop
        # continues — it must release the item before parking idle, same
        # contract as the completed-item path.
        deadline = time.monotonic() + 2.0
        while ref() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert ref() is None, (
            "canceled closure must be released once the worker parks idle"
        )
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def _hold_owner(scheduler: ModelWorkScheduler):
    started = Event()
    release = Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=2)

    future = scheduler.submit_foreground(blocker)
    assert started.wait(timeout=2)
    return future, release


def test_persistence_budget_drops_oldest_pinned_and_keeps_newest():
    scheduler = _scheduler(persistence_max_pending_bytes=100)
    ran: list[str] = []
    held, release = _hold_owner(scheduler)
    try:
        futures = [
            scheduler.submit_idle_persistence(
                lambda name=name: ran.append(name),
                coalesce_key=f"ssd_cold:{name}",
                pinned_bytes=40,
            )
            for name in ("a", "b", "c")
        ]
        stats = scheduler.stats()
        assert futures[0].cancelled()
        assert stats["persistence_pending"] == 2
        assert stats["persistence_pending_bytes"] == 80
        assert stats["persistence_budget_dropped"] == 1
        release.set()
        held.result(timeout=2)
        futures[1].result(timeout=2)
        futures[2].result(timeout=2)
        assert ran == ["b", "c"]
        assert scheduler.stats()["persistence_pending_bytes"] == 0
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_persistence_budget_never_drops_newest_or_unpinned_items():
    scheduler = _scheduler(persistence_max_pending_bytes=10)
    _held, release = _hold_owner(scheduler)
    try:
        unpinned = scheduler.submit_idle_persistence(lambda: None)
        pinned = scheduler.submit_idle_persistence(lambda: None, pinned_bytes=50)
        stats = scheduler.stats()
        assert not unpinned.cancelled()
        assert not pinned.cancelled()
        assert stats["persistence_pending_bytes"] == 50
        assert stats["persistence_budget_dropped"] == 0
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_zero_persistence_budget_disables_the_cap():
    scheduler = _scheduler(persistence_max_pending_bytes=0)
    _held, release = _hold_owner(scheduler)
    try:
        futures = [
            scheduler.submit_idle_persistence(lambda: None, pinned_bytes=10**12)
            for _ in range(3)
        ]
        assert not any(future.cancelled() for future in futures)
        assert scheduler.stats()["persistence_budget_dropped"] == 0
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_coalescing_releases_pinned_bytes():
    scheduler = _scheduler(persistence_max_pending_bytes=100)
    _held, release = _hold_owner(scheduler)
    try:
        scheduler.submit_idle_persistence(
            lambda: None, coalesce_key="ssd_cold:s", pinned_bytes=60
        )
        scheduler.submit_idle_persistence(
            lambda: None, coalesce_key="ssd_cold:s", pinned_bytes=70
        )
        stats = scheduler.stats()
        assert stats["persistence_pending_bytes"] == 70
        assert stats["persistence_budget_dropped"] == 0
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_persistence_budget_reads_env(monkeypatch):
    def budget() -> int:
        scheduler = _scheduler()
        try:
            return scheduler.persistence_max_pending_bytes
        finally:
            scheduler.shutdown(wait=True)

    monkeypatch.setenv("MTPLX_PERSISTENCE_MAX_PENDING_BYTES", "2G")
    assert budget() == 2 * 1024**3
    monkeypatch.setenv("MTPLX_PERSISTENCE_MAX_PENDING_BYTES", "off")
    assert budget() == 0
    monkeypatch.delenv("MTPLX_PERSISTENCE_MAX_PENDING_BYTES")
    assert budget() == 4 * 1024**3
