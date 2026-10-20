"""The prefill event says why and how much is read again, before the wait.

On 2026-09-29 every screenshot turn of a Pi session re-read 123,000 to 138,000
tokens cold while the app showed only a token counter. The event published
right before a replay (or a cold read) now carries the matched history, the
resume point and its source, the tokens to recompute, the cause, and an
estimate from the measured prefill rate (mtplx/prefill_plan.py).

Covered here: the pure classifier on a real SessionBank, the event each
restore lane publishes (RAM hit, near-prefix checkpoint, SSD restore, cold
read, and the screenshot turn on the toy vision runtime), and the server's
enrichment, carry-forward and receipt.

CPU-sized: toy models and real SessionBanks, no model pack.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace
import time

import mlx.core as mx
import pytest

import mtplx.generation as generation
from mtplx.generation import restore_or_prefill_prompt_state
from mtplx.mtp_patch import MTPContract
from mtplx.prefill_plan import (
    estimate_seconds,
    explain_reread,
    publishable_reread,
    reread_facts,
)
from mtplx.runtime import MTPLXRuntime
from mtplx.server import openai
from mtplx.server.dashboard_state import InFlightHandle
from mtplx.session_bank import SessionBank
from test_server_openai import _fake_state
from test_vision_session_restore import (
    IMAGE_A,
    IMAGE_A_OTHER_PIXELS,
    IM_END,
    PAD,
    VISION_END,
    VISION_START,
    SeamTokenizer,
    _toy_runtime,
    _toy_splice,
)

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
IDS = list(range(1, 201))


def _bank(**kwargs) -> SessionBank:
    kwargs.setdefault("max_entries", 16)
    kwargs.setdefault("max_bytes", 1 << 20)
    kwargs.setdefault("per_session_max_bytes", 1 << 20)
    return SessionBank(**kwargs)


def _put(bank: SessionBank, token_ids: list[int], session_id: str, nbytes: int = 64):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=token_ids,
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=nbytes,
    )
    assert entry is not None
    return entry


def _explain(bank, prompt, restore_point, source, *, session_id="s", served=False, spans=None):
    facts = reread_facts(
        session_bank=bank,
        session_id=session_id,
        bank_ids=prompt,
        restore_point=restore_point,
        source=source,
        image_spans=spans,
    )
    return explain_reread(facts, session_served_before=served)


# ---- The classifier on a real SessionBank --------------------------------


def test_a_prompt_that_extends_the_saved_history_reads_only_its_new_tokens():
    bank = _bank()
    _put(bank, IDS[:100], "s")
    explanation = _explain(bank, IDS[:130], 100, "ram", served=True)
    assert explanation["history_matched_tokens"] == 100
    assert explanation["restore_point_tokens"] == 100
    assert explanation["recompute_tokens"] == 30
    assert explanation["source"] == "ram"
    assert explanation["cause"] is None
    assert explanation["resume_limit"] is None
    assert explanation["text"] == "Reading 30 new tokens."


def test_a_checkpoint_replay_names_the_saved_state_it_resumes_from():
    # The conversation's state is saved at 120, but the running state resumes
    # from a checkpoint at 96: 24 matched tokens are computed again.
    bank = _bank()
    _put(bank, IDS[:120], "s")
    explanation = _explain(bank, IDS[:130], 96, "ram", served=True)
    assert explanation["history_matched_tokens"] == 120
    assert explanation["restore_point_tokens"] == 96
    assert explanation["recompute_tokens"] == 34
    assert explanation["cause"] is None
    assert explanation["resume_limit"] == "saved_state"
    assert explanation["resume_limit_at_token"] == 96
    assert explanation["text"] == (
        "Resuming from the saved state at token 96. Re-reading 34 tokens."
    )


def test_a_changed_history_names_the_token_where_it_changed():
    bank = _bank()
    _put(bank, IDS[:150], "s")
    edited = IDS[:110] + [999] * 40
    explanation = _explain(bank, edited, 96, "ram", served=True)
    assert explanation["cause"] == "history_changed"
    assert explanation["cause_at_token"] == 110
    assert explanation["resume_limit_at_token"] == 96
    assert explanation["text"].startswith("History changed at token 110. ")


def test_a_prompt_that_shares_only_its_opening_says_so_without_naming_why():
    # 2026-10-01, Pi under one session id: the turn after a compaction shares
    # 41 tokens with the compaction's summary prompt, and the compaction's
    # second summary shares 110 with its first. An early edit of a long chat
    # looks the same, so the text states the overlap, never the intent.
    def explain(matched, cached, prompt):
        return explain_reread(
            {
                "prompt_tokens": prompt,
                "history_matched_tokens": matched,
                "restore_point_tokens": 0,
                "session_cached_tokens": cached,
                "source": "none",
            },
            session_served_before=True,
        )

    turn = explain(41, 47_765, 32_409)
    assert turn["cause"] == "short_shared_prefix"
    assert turn["cause_at_token"] == 41
    assert turn["text"] == (
        "Only the first 41 tokens match this conversation's saved state. "
        "Re-reading 32,409 tokens."
    )
    assert explain(110, 64_276, 46_267)["cause"] == "short_shared_prefix"
    # Past the opening it is the same conversation, edited.
    assert explain(5_000, 40_000, 41_000)["cause"] == "history_changed"
    assert explain(1_500, 4_000, 4_100)["cause"] == "history_changed"


def test_an_ssd_restore_names_the_ssd():
    bank = _bank()
    _put(bank, IDS[:100], "s")
    explanation = _explain(bank, IDS[:130], 100, "ssd", served=True)
    assert explanation["source"] == "ssd"
    assert explanation["cause"] is None
    assert explanation["text"] == (
        "Restored 100 tokens from the SSD cache. Reading 30 new tokens."
    )


def test_a_changed_screenshot_is_named_at_its_first_token():
    # Keyed ids: the image's pads become digest surrogates, so other pixels in
    # the same slot part from the saved history at the image's first token.
    bank = _bank()
    old = IDS[:60] + [5001] * 8 + IDS[68:100]
    _put(bank, old, "s")
    new = IDS[:60] + [7001] * 8 + IDS[68:120]
    explanation = _explain(bank, new, 60, "ram", served=True, spans=[(60, 68)])
    assert explanation["cause"] == "screenshot_changed"
    assert explanation["cause_at_token"] == 60
    assert explanation["resume_limit"] is None
    assert explanation["text"] == (
        "The screenshot at token 60 changed. Re-reading 60 tokens."
    )


def test_a_restore_that_stops_before_an_unchanged_screenshot_says_so():
    bank = _bank()
    history = IDS[:60] + [5001] * 8 + IDS[68:100]
    _put(bank, history, "s")
    # Same image, the text after it changed at 90; the restore stopped
    # before the image at 60.
    prompt = history[:90] + [999] * 30
    explanation = _explain(bank, prompt, 60, "ram", served=True, spans=[(60, 68)])
    assert explanation["cause"] == "history_changed"
    assert explanation["cause_at_token"] == 90
    assert explanation["resume_limit"] == "screenshot"
    assert explanation["resume_limit_at_token"] == 60
    assert explanation["text"] == (
        "History changed at token 90. Resuming before the screenshot at token 60. "
        "Re-reading 60 tokens."
    )


def test_a_cold_start_is_a_new_conversation_and_a_later_miss_is_not():
    bank = _bank()
    fresh = _explain(bank, IDS[:50], 0, "none", served=False)
    assert fresh["cause"] == "new_conversation"
    assert fresh["source"] == "none"
    assert fresh["recompute_tokens"] == 50
    served = _explain(bank, IDS[:50], 0, "none", served=True)
    assert served["cause"] == "not_cached"


def test_a_conversation_whose_state_was_replaced_by_another_one():
    bank = _bank(max_entries=1)
    _put(bank, IDS[:50], "mine")
    _put(bank, [7] * 60, "other")
    assert bank.eviction_log[-1]["reason"] == "evicted"
    explanation = _explain(bank, IDS[:80], 0, "none", session_id="mine", served=True)
    assert explanation["cause"] == "switched_conversation"
    assert explanation["session_eviction_reason"] == "evicted"


def test_a_state_freed_under_memory_pressure_says_so():
    bank = _bank()
    _put(bank, IDS[:50], "mine")
    bank.shrink_to_bytes(0, reason="memory_pressure_warning")
    explanation = _explain(bank, IDS[:80], 0, "none", session_id="mine", served=True)
    assert explanation["cause"] == "freed_for_memory"
    assert explanation["text"].startswith("Saved state was freed to relieve memory pressure.")


def test_a_conversation_too_large_to_keep_says_so():
    bank = _bank(max_bytes=1024, per_session_max_bytes=512)
    assert bank.put(
        runtime=RUNTIME, token_ids=IDS[:50], cache=[], logits=None, hidden=None,
        session_id="big", nbytes_override=4096,
    ) is None
    explanation = _explain(bank, IDS[:80], 0, "none", session_id="big", served=True)
    assert explanation["cause"] == "too_large"


def test_changed_settings_and_a_disabled_cache_are_named():
    bank = _bank()
    bank.last_miss_reason = "template_mismatch"
    assert _explain(bank, IDS[:40], 0, "none")["cause"] == "settings_changed"
    off = explain_reread(
        reread_facts(
            session_bank=None, session_id="s", bank_ids=IDS[:40],
            restore_point=0, source="none",
        )
    )
    assert off["cause"] == "cache_off"
    assert off["text"] == "The prompt cache is off for this request. Re-reading 40 tokens."


def test_the_estimate_comes_from_the_measured_rate_and_is_never_invented():
    assert estimate_seconds(10_000, None) is None
    assert estimate_seconds(10_000, {"tokens": 0, "compute_time_s": 0}) is None
    assert estimate_seconds(0, {"tokens": 4096, "compute_time_s": 4.0}) is None
    assert estimate_seconds(10_240, {"tokens": 4096, "compute_time_s": 4.0}) == 10.0
    bank = _bank()
    facts = reread_facts(
        session_bank=bank, session_id="s", bank_ids=IDS[:200], restore_point=0,
        source="none",
    )
    published = publishable_reread(
        facts, session_served_before=False,
        rates={"tokens": 100, "compute_time_s": 1.0},
    )
    assert published["eta_s"] == 2.0
    assert published["text"] == (
        "New conversation or first turn since MTPLX started. "
        "Re-reading 200 tokens, about 2 s."
    )
    unmeasured = publishable_reread(facts, session_served_before=False, rates=None)
    assert unmeasured["eta_s"] is None
    assert "about" not in unmeasured["text"]


# ---- The event each lane publishes before its replay --------------------


class _TinyModel:
    def __init__(self):
        self.calls = 0

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def __call__(self, input_ids, *, cache=None, return_hidden=False,
                 hidden_variant=None, emit_logits=True, logits_keep=None):
        self.calls += 1
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        return (logits, hidden) if return_hidden else logits


def _tiny_runtime(model: _TinyModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda tokens, **_: ""),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


class _ExactRestoreBank:
    """Serves one exact restore of ``prefix_len`` tokens; answers the facts
    accessors the way SessionBank does for one saved conversation."""

    last_miss_reason = None
    cold_tier = None
    eviction_log = ()

    def __init__(self, prefix_len: int, *, source: str = "ram"):
        self.prefix_len = prefix_len
        self.source = source

    def longest_prefix(self, _ids):
        return SimpleNamespace(prefix_len=self.prefix_len)

    def restore(self, rt, _ids, **kwargs):
        factory = kwargs.get("cache_factory")
        return SimpleNamespace(
            entry=SimpleNamespace(prefix_len=self.prefix_len),
            cache=factory() if callable(factory) else rt.make_cache(),
            logits=mx.zeros((1, 4), dtype=mx.float32),
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            mtp_history_cache=[],
            restore_mode="clone",
            cache_source=self.source,
            ssd_cache_hit=self.source == "ssd",
            ssd_cached_tokens=self.prefix_len if self.source == "ssd" else 0,
            ssd_restore_s=0.25 if self.source == "ssd" else 0.0,
        )

    def held_by_session(self):
        return [{"session_id": "s", "longest_prefix_tokens": self.prefix_len}]

    def longest_shared_prefix_tokens(self, _ids, *, session_id=None):
        return self.prefix_len


def _recorder(model: _TinyModel):
    events: list[tuple[dict, int]] = []

    def record(payload: dict) -> None:
        events.append((dict(payload), model.calls))

    return events, record


def _reread_events(events):
    return [(payload, calls) for payload, calls in events if "reread" in payload]


@pytest.mark.parametrize("source", ["ram", "ssd"])
def test_an_exact_restore_publishes_the_facts_before_the_suffix_replay(source):
    model = _TinyModel()
    events, record = _recorder(model)
    restore_or_prefill_prompt_state(
        _tiny_runtime(model),
        IDS[:5],
        session_bank=_ExactRestoreBank(3, source=source),
        session_id="s",
        prefill_callback=record,
    )
    ((payload, calls_then),) = _reread_events(events)
    # Published before the suffix forward ran; the replay ran afterwards.
    assert calls_then == 0
    assert model.calls > 0
    assert payload["phase"] == "chunk"
    assert payload["cached_tokens"] == 3
    facts = payload["reread"]
    assert facts["restore_point_tokens"] == 3
    assert facts["recompute_tokens"] == 2
    assert facts["history_matched_tokens"] == 3
    assert facts["source"] == source
    explanation = explain_reread(facts, session_served_before=True)
    assert explanation["cause"] is None
    if source == "ssd":
        assert explanation["text"] == (
            "Restored 3 tokens from the SSD cache. Reading 2 new tokens."
        )


def test_a_cold_read_publishes_the_facts_before_the_first_forward():
    model = _TinyModel()
    events, record = _recorder(model)
    restore_or_prefill_prompt_state(
        _tiny_runtime(model),
        IDS[:6],
        session_bank=_bank(),
        session_id="new-session",
        prefill_callback=record,
    )
    ((payload, calls_then),) = _reread_events(events)
    assert calls_then == 0
    assert model.calls > 0
    facts = payload["reread"]
    assert facts["restore_point_tokens"] == 0
    assert facts["recompute_tokens"] == 6
    assert facts["source"] == "none"
    assert explain_reread(facts)["cause"] == "new_conversation"
    # The event comes before every chunk of the read.
    order = [p.get("phase") for p, _ in events]
    assert order.index("chunk") == [i for i, (p, _) in enumerate(events) if "reread" in p][0]


def test_the_near_prefix_lane_publishes_its_checkpoint_before_the_replay():
    """A boundary (checkpoint) restore below the matched prefix: the event
    carries the lane's own restore point, before the suffix forward."""

    model = _TinyModel()
    rt = _tiny_runtime(model)
    bank = _bank()
    entry = bank.put(
        runtime=rt,
        token_ids=IDS[:8],
        cache=[],
        logits=None,
        hidden=None,
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        session_id="s",
        mtp_history_policy="cycle",
        nbytes_override=64,
    )
    assert entry is not None

    class NearBank:
        last_miss_reason = None

        def near_prefix_candidates(self, _ids, **_kwargs):
            yield entry, 6

        def restore_entry_prefix_cache(self, _rt, _entry, _matched, **_kwargs):
            return [], None, "clone", 4  # the checkpoint under the match

    events, record = _recorder(model)
    asked: list[tuple[int, str]] = []

    def facts(restore_point: int, source: str) -> dict:
        asked.append((restore_point, source))
        return {"restore_point_tokens": restore_point, "source": source}

    state = generation._restore_near_prefix_prompt_state(
        rt,
        IDS[:10],
        base_hidden_variant=None,
        mtp_hidden_variant=None,
        mtp_history_policy="cycle",
        session_bank=NearBank(),
        template_hash=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        min_restore_tokens=0,
        allow_block_prefix=True,
        chunk_callback=record,
        chunk_started_s=time.perf_counter(),
        reread=facts,
    )
    assert state is not None and state.cached_tokens == 4
    assert asked == [(4, "ram")]
    ((payload, calls_then),) = _reread_events(events)
    assert calls_then == 0
    assert payload["reread"] == {"restore_point_tokens": 4, "source": "ram"}
    assert payload["cached_tokens"] == 4


class _LaneProbe(Exception):
    pass


def test_both_near_prefix_lanes_get_facts_bound_to_this_conversation(monkeypatch):
    bank = _bank()
    runtime = SimpleNamespace(
        model_path=Path("models/example"), mtp_enabled=False, contract=SimpleNamespace()
    )
    _put(bank, IDS[:60] + [999] * 20, "s")
    captured: list[dict] = []

    def lane(_rt, _prompt_ids, **kwargs):
        captured.append(kwargs)
        raise _LaneProbe()

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", lane)
    with pytest.raises(_LaneProbe):
        restore_or_prefill_prompt_state(
            runtime, IDS[:100], mtp_history_policy="cycle",
            session_bank=bank, session_id="s",
        )
    facts = captured[0]["reread"](48, "ram")
    assert facts["history_matched_tokens"] == 60
    assert facts["session_cached_tokens"] == 80
    assert explain_reread(facts, session_served_before=True)["cause_at_token"] == 60

    # The second lane (after the exact restore declines) gets the same facts.
    monkeypatch.setattr(bank, "longest_prefix", lambda _ids: None)
    monkeypatch.setattr(bank, "restore", lambda *_a, **_k: None)
    calls: list[dict] = []

    def second_lane(_rt, _prompt_ids, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise _LaneProbe()
        return None

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", second_lane)
    with pytest.raises(_LaneProbe):
        restore_or_prefill_prompt_state(
            runtime, IDS[:100], mtp_history_policy="cycle",
            session_bank=bank, session_id="s",
        )
    assert calls[1]["reread"](48, "ram")["history_matched_tokens"] == 60


# ---- The screenshot turn, end to end on the toy vision runtime ----------


def _screenshot_prompt(history: list[int], image: bytes, tail: str):
    splice = _toy_splice([image])
    ids = (
        history
        + [VISION_START]
        + [PAD] * splice.pad_counts[0]
        + [VISION_END]
        + SeamTokenizer().encode(tail)
    )
    return ids, splice


def _store(rt, bank, ids, image):
    restore_or_prefill_prompt_state(
        rt, ids, vision_splice=_toy_splice([image]), session_bank=bank,
        session_id="s", store_prefix_snapshot=True,
    )


def _screenshot_turn(rt, bank, ids, image):
    events: list[dict] = []
    restore_or_prefill_prompt_state(
        rt, ids, vision_splice=_toy_splice([image]), session_bank=bank,
        session_id="s", prefill_callback=events.append,
    )
    (payload,) = [event for event in events if "reread" in event]
    return payload


@pytest.fixture
def vision_bank(monkeypatch):
    # The toy prompts are far below the production store threshold.
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    return SessionBank(max_entries=16, max_bytes=1 << 30, per_session_max_bytes=1 << 30)


HISTORY = SeamTokenizer().encode(
    "<|im_start|>user\n" + "the quick brown fox jumps over the lazy dog. " * 30
)


def test_other_pixels_in_the_screenshot_are_named_before_the_replay(vision_bank):
    rt = _toy_runtime()
    first, _ = _screenshot_prompt(HISTORY, IMAGE_A, "what is on screen.")
    _store(rt, vision_bank, first, IMAGE_A)
    second, _ = _screenshot_prompt(HISTORY, IMAGE_A_OTHER_PIXELS, "what is on screen.")
    payload = _screenshot_turn(rt, vision_bank, second, IMAGE_A_OTHER_PIXELS)
    pad = second.index(PAD)
    facts = payload["reread"]
    assert facts["history_matched_tokens"] == pad
    assert facts["image_changed_at"] == pad
    assert facts["restore_point_tokens"] <= pad
    explanation = explain_reread(facts, session_served_before=True)
    assert explanation["cause"] == "screenshot_changed"
    assert explanation["cause_at_token"] == pad
    assert explanation["recompute_tokens"] == len(second) - facts["restore_point_tokens"]


def test_a_reply_that_changed_after_the_screenshot_resumes_at_the_change(vision_bank):
    """The restore matches an image prompt in its content-keyed view, so a
    reply that parts after an unchanged screenshot resumes where it parts,
    past the screenshot (it used to stop before the screenshot, re-reading
    it and everything after it)."""

    rt = _toy_runtime()
    tok = SeamTokenizer()
    first, _ = _screenshot_prompt(HISTORY, IMAGE_A, "what is on screen.")
    reply = tok.encode("<|im_start|>assistant\nthe window is blue.") + [IM_END]
    _store(rt, vision_bank, first, IMAGE_A)
    _store(rt, vision_bank, first + reply, IMAGE_A)
    # The client resends the answer re-rendered: it parts after the image.
    resent = tok.encode("<|im_start|>assistant\nthe window is navy.") + [IM_END]
    second = first + resent + tok.encode("<|im_start|>user\nand now.")
    payload = _screenshot_turn(rt, vision_bank, second, IMAGE_A)
    pad = second.index(PAD)
    seam = len(first) + next(i for i, (a, b) in enumerate(zip(reply, resent)) if a != b)
    facts = payload["reread"]
    assert facts["history_matched_tokens"] == seam
    assert facts["restore_point_tokens"] == seam > pad
    explanation = explain_reread(facts, session_served_before=True)
    assert explanation["cause"] == "history_changed"
    assert explanation["cause_at_token"] == seam
    assert explanation["resume_limit"] is None
    assert explanation["recompute_tokens"] == len(second) - seam


@pytest.mark.parametrize("source", ["ram", "ssd"])
def test_the_completed_prefill_reports_the_cache_outcome_the_stats_will(source):
    model = _TinyModel()
    events, record = _recorder(model)
    prompt_state = restore_or_prefill_prompt_state(
        _tiny_runtime(model),
        IDS[:5],
        session_bank=_ExactRestoreBank(3, source=source),
        session_id="s",
        prefill_callback=record,
    )
    ((completed, _),) = [(p, c) for p, c in events if p.get("phase") == "completed"]
    assert completed["cache_hit"] is prompt_state.cache_hit is True
    assert completed["cached_tokens"] == prompt_state.cached_tokens == 3
    assert completed["new_prefill_tokens"] == prompt_state.suffix_tokens == 2
    assert completed["cache_source"] == prompt_state.cache_source == source
    assert completed["cache_miss_reason"] is None
    assert completed["ssd_cache_hit"] is (source == "ssd")
    assert completed["ssd_cached_tokens"] == (3 if source == "ssd" else 0)
    assert completed["ssd_suffix_tokens"] == (2 if source == "ssd" else 0)
    assert completed["session_restore_served"] == prompt_state.restore_served


def test_the_completed_prefill_of_a_cold_read_reports_the_miss_reason():
    model = _TinyModel()
    events, record = _recorder(model)
    prompt_state = restore_or_prefill_prompt_state(
        _tiny_runtime(model),
        IDS[:6],
        session_bank=_bank(),
        session_id="new-session",
        prefill_callback=record,
    )
    ((completed, _),) = [(p, c) for p, c in events if p.get("phase") == "completed"]
    assert completed["cache_hit"] is False
    assert completed["cache_source"] == "none"
    assert completed["cache_miss_reason"] == prompt_state.cache_miss_reason
    assert completed["cache_miss_reason"]


# ---- The server: classify, estimate, carry forward, receipt -------------


def _register(state, request_id: str) -> None:
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id=request_id, cancel_event=Event(), started_s=time.time(),
            session_id="s",
        )
    )


def test_the_server_explains_estimates_and_carries_the_reread_forward():
    state = _fake_state()
    state.dashboard.prefill_history.record_chunk(4096, 2.0)  # 2,048 tok/s measured
    _register(state, "req-1")
    facts = reread_facts(
        session_bank=_bank(), session_id="s", bank_ids=IDS[:200],
        restore_point=0, source="none",
    )
    facts["prompt_tokens"] = facts["recompute_tokens"] = 20_480
    openai._dashboard_publish_prefill(
        state, request_id="req-1", session_id="s",
        payload={"phase": "chunk", "tokens_done": 0, "tokens_total": 20_480,
                 "cached_tokens": 0, "reread": facts},
    )
    reread = state.dashboard.in_flight.get("req-1").prefill_state["reread"]
    assert reread["cause"] == "new_conversation"
    assert reread["eta_s"] == 10.0
    assert reread["text"].endswith("Re-reading 20,480 tokens, about 10 s.")
    # A later chunk replaces the prefill state and still carries it.
    openai._dashboard_publish_prefill(
        state, request_id="req-1", session_id="s",
        payload={"phase": "chunk", "tokens_done": 4096, "tokens_total": 20_480,
                 "chunk_size": 4096, "chunk_elapsed_s": 2.0, "elapsed_s": 2.0},
    )
    assert state.dashboard.in_flight.get("req-1").prefill_state["reread"] == reread
    openai._dashboard_publish_prefill(
        state, request_id="req-1", session_id="s",
        payload={"phase": "completed", "tokens_total": 20_480},
    )
    assert state.dashboard.in_flight.get("req-1").prefill_state is None
    # The receipt keeps it; the ledger lets go.
    openai._record_request_metrics(state, {"request_id": "req-1"})
    assert state.last_metrics[-1]["reread"] == reread
    assert state.dashboard.rereads.get("req-1") is None


def test_served_before_is_the_engine_sessions_committed_history():
    state = _fake_state()
    state.sessions = openai.EngineSessionManager(bank=_bank())
    assert openai._session_served_before(state, "s") is False
    session = state.sessions.get_or_create("s")
    assert openai._session_served_before(state, "s") is False
    session.committed_token_ids = (1, 2, 3)
    assert openai._session_served_before(state, "s") is True
    assert openai._session_served_before(state, None) is False


def test_the_ledger_is_bounded():
    from mtplx.server.dashboard_state import RereadLedger

    ledger = RereadLedger(capacity=2)
    for index in range(3):
        ledger.put(f"r{index}", {"cause": None})
    assert ledger.get("r0") is None
    assert ledger.pop("r2") == {"cause": None}
    assert ledger.pop("r2") is None


# ---- The app has a sentence for every miss code the server sends ---------


def _server_miss_codes() -> set[str]:
    import re

    from mtplx.session_bank import CacheMissReason

    root = Path(__file__).resolve().parents[1] / "mtplx"
    codes = {reason.value for reason in CacheMissReason}
    cold = (root / "cache_bank" / "cold_tier.py").read_text()
    codes |= set(re.findall(r'_set_last_miss\(\s*f?"([a-z_]+)', cold))
    codes |= set(re.findall(r'_stats\["last_miss_reason"\] = "([a-z_]+)"', cold))
    bank = (root / "session_bank.py").read_text()
    codes |= set(re.findall(r'last_miss_reason = "([a-z_]+)"', bank))
    codes |= {"block_prefix_disabled", "no_gdn_boundaries", "below_block_min_match"}
    server = (root / "server" / "openai.py").read_text()
    codes |= set(re.findall(r'cache_miss_reason = "([a-z_]+)"', server))
    codes |= set(re.findall(r'cache_miss_reason=None if restore_hit else "([a-z_]+)"', server))
    frontier = server[server.index("def _live_frontier_miss_reason_from_counts"):]
    frontier = frontier[: frontier.index("\ndef ", 1)]
    codes |= set(re.findall(r'"(miss_[a-z_]+)"', frontier))
    return codes


def test_the_app_has_a_sentence_for_every_miss_code_the_server_sends():
    import re

    swift = (
        Path(__file__).resolve().parents[1]
        / "apps/MTPLXApp/Sources/MTPLXAppCore/Services/CacheExplanation.swift"
    ).read_text()
    quoted = set(re.findall(r'"([a-z_]+)"', swift))
    prefixes = set(re.findall(r'hasPrefix\("([a-z_]+)"\)', swift))
    codes = _server_miss_codes()
    # The scan found the families it is meant to find.
    assert {"new_session", "ssd_prefix_miss", "miss_no_tool_result", "legacy_ssd_cache_archived",
            "vision_request_cache_bypass", "mtp_batch_cold_prefill"} <= codes
    missing = sorted(
        code for code in codes
        if code not in quoted and not any(code.startswith(p) for p in prefixes)
    )
    assert not missing, f"no app sentence for server miss codes: {missing}"
