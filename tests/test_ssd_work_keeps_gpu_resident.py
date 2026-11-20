"""SSD work on the model-owner thread keeps the GPU working set resident (12 Nov).

The residency keepalive beats once per interval between owner-thread items,
and macOS drops a process's Metal residency about 2.5 s after its last command
buffer. An SSD spill or restore is one item of seconds of CPU work with no
command buffer: the spill reads a lease's KV as block views and copies them
with the CPU, the restore reads blob files (M5 Max, a 1.5 GiB wired set: after
4 s of the spill's work the next full read took 24.7 ms, as after 4 s idle,
against 4.8 ms with one beat a second; n4-ttft receipts
spill-residency-small-1.json). A 131K answer's spill runs for 11 s, so a
request arriving mid-write (a tool round) paid to bring the 77 GiB of weights
back, Stop's flush ran with them non-resident, and under memory pressure they
could be compressed (the 06:43 kill). These tests drive the real bank, cold
tier and scheduler, wired by the server's own function.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mlx.core as mx

from mtplx.cache_bank import blob_io
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai
from mtplx.session_bank import SessionBank
from test_ssd_spill import LOOKUP_IDENTITY, FakeRuntime, TrimmableKV, make_cold

TOKENS = list(range(1, 2049))


class Beats:
    def __init__(self) -> None:
        self.at: list[float] = []
        self.threads: set[int] = set()

    def __call__(self) -> None:
        self.at.append(time.monotonic())
        self.threads.add(threading.get_ident())

    def between(self, start: float, end: float) -> int:
        return sum(1 for t in self.at if start < t < end)


def _scheduler(beats: Beats) -> ModelWorkScheduler:
    scheduler = ModelWorkScheduler(
        name="test-ssd-residency", idle_grace_s=0.0, persistence_quiet_grace_s=0.0
    )
    scheduler.arm_idle_keepalive(beats, interval_s=0.05, attentive_s=60.0)
    return scheduler


def _timed(fn, span: dict[str, float]):
    def run():
        span["start"] = time.monotonic()
        try:
            return fn()
        finally:
            span["end"] = time.monotonic()

    return run


def _lease_spill_job(tmp_path, scheduler: ModelWorkScheduler, *, server_wiring: bool):
    cold = make_cold(tmp_path, block_size=16)
    if server_wiring:
        openai._wire_cold_tier_to_scheduler(cold, scheduler)
    else:
        # The wiring before 12 Nov: the foreground yield only.
        cold.foreground_busy = scheduler.foreground_busy
    jobs: list = []
    bank = SessionBank(per_session_max_bytes=1, cold_tier=cold)
    bank.cold_enqueue_dispatch = jobs.append

    def answer():
        # On the owner thread, materialized, as a lease's KV is in the
        # server (MLX streams belong to the thread that made them).
        kv = TrimmableKV(rows=2048, width=8)
        logits = mx.array([[0.5, 1.5]], dtype=mx.float16)
        hidden = mx.array([[[2.0, 3.0]]], dtype=mx.float16)
        mx.eval(kv.state, logits, hidden)
        return bank.put(
            runtime=FakeRuntime(),
            token_ids=TOKENS,
            cache=[kv],
            logits=logits,
            hidden=hidden,
            keep_live_ref=True,
            session_id="s1",
            template_hash="template-a",
            policy_fingerprint="policy-a",
            snapshot_epoch=7,
        )

    entry = scheduler.submit_foreground(answer).result(timeout=30)
    assert entry is not None and entry.live_ref_only and len(jobs) == 1
    # A disk at test scale: each blob write takes 5 ms, so the spill of the
    # lease's 128 blocks runs for well over ten keepalive intervals.
    write_blob = cold._write_blob

    def slow_write(digest, raw):
        time.sleep(0.005)
        return write_blob(digest, raw)

    cold._write_blob = slow_write
    return cold, jobs[0]


def _run_spill(tmp_path, *, server_wiring: bool):
    beats = Beats()
    scheduler = _scheduler(beats)
    cold, job = _lease_spill_job(tmp_path, scheduler, server_wiring=server_wiring)
    span: dict[str, float] = {}
    try:
        assert scheduler.submit_idle_persistence(_timed(job, span)).result(timeout=30) is True
        assert cold.lookup(TOKENS, **LOOKUP_IDENTITY) is not None
        return beats, span, scheduler.owner_thread_id
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_a_long_spill_beats_the_keepalive_once_per_interval(tmp_path):
    beats, span, owner = _run_spill(tmp_path, server_wiring=True)
    duration = span["end"] - span["start"]
    assert duration > 0.5, duration
    # About one beat per 50 ms interval while the spill held the owner thread
    # (measured: 16 in 0.84 s).
    assert beats.between(span["start"], span["end"]) >= int(duration / 0.05) // 2
    assert beats.threads == {owner}


def test_without_the_beat_wiring_a_spill_starves_the_keepalive(tmp_path):
    # The behaviour the server had before 12 Nov: no beat for the whole spill.
    beats, span, _owner = _run_spill(tmp_path, server_wiring=False)
    assert span["end"] - span["start"] > 0.5
    assert beats.between(span["start"], span["end"]) == 0


def test_an_ssd_restore_beats_the_keepalive_while_it_reads(tmp_path, monkeypatch):
    beats = Beats()
    scheduler = _scheduler(beats)
    cold, job = _lease_spill_job(tmp_path, scheduler, server_wiring=True)
    try:
        assert scheduler.submit_idle_persistence(job).result(timeout=30) is True
        # A disk at test scale on the way back: 5 ms per blob read on the
        # serial path, and a quarter second per read on the read pool's (its
        # reads run side by side: the owner thread waits for a tensor's batch
        # and must beat while it waits).
        read_bytes = Path.read_bytes

        def slow_read(self):
            time.sleep(0.005)
            return read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", slow_read)
        read_range = blob_io._read_range

        def slow_range(*args):
            time.sleep(0.25)
            return read_range(*args)

        monkeypatch.setattr(blob_io, "_read_range", slow_range)
        span: dict[str, float] = {}
        restore = _timed(lambda: cold.lookup(TOKENS, **LOOKUP_IDENTITY), span)
        # A request's restore runs inside its foreground item.
        assert scheduler.submit_foreground(restore).result(timeout=30) is not None
        duration = span["end"] - span["start"]
        assert duration > 0.5, duration
        assert beats.between(span["start"], span["end"]) >= int(duration / 0.05) // 2
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
        cold.close()


def test_the_inline_beat_is_owner_thread_only_and_once_per_interval():
    beats = Beats()
    scheduler = _scheduler(beats)
    try:
        # Off the owner thread it never beats.
        assert scheduler.beat_within_item() is False

        def item() -> list[bool]:
            results = []
            for _ in range(40):
                results.append(scheduler.beat_within_item())
                time.sleep(0.01)
            return results

        results = scheduler.submit_idle_persistence(item).result(timeout=10)
        # 0.4 s inside one item at a 50 ms interval: several beats, never one
        # per call.
        assert 4 <= sum(results) <= 10, sum(results)
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_an_unarmed_keepalive_never_beats_inline():
    scheduler = ModelWorkScheduler(name="test-ssd-residency-unarmed", idle_grace_s=0.0)
    try:
        assert scheduler.submit_idle_persistence(
            lambda: [scheduler.beat_within_item() for _ in range(5)]
        ).result(timeout=10) == [False] * 5
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
