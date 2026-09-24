"""Solo MTP -> batched AR lane handover (server side; the generator's poll is exercised live).

When another request is waiting on the batched lane behind a solo MTP owner, the solo run returns
finish_reason="handover" with its trunk cache holding prompt + tokens[:-1]; the solo runner submits a
continuation job (insert the cache plus the last token, remaining max_tokens, penalty counts seeded)
and the dispatcher waits for it off the owner thread and finalizes both segments as one response.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
from mtplx.sampling import SamplerConfig


class _Tok:
    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


class _Service:
    def __init__(self, pending=False, unavailable=None):
        self.jobs = []
        self._pending = pending
        self.ar_batch_unavailable_reason = unavailable

    def has_pending(self):
        return self._pending

    def submit(self, job):
        self.jobs.append(job)
        job.future = Future()
        return job.future


def _state(service, mode="ar_batch"):
    st = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=_Tok()),
        ar_batch_service=service,
        args=SimpleNamespace(scheduler_mode=mode),
        model_scheduler=SimpleNamespace(foreground_pending=lambda: 0),
        fg=0,
    )
    st.begin_foreground = lambda: setattr(st, "fg", st.fg + 1)
    st.end_foreground = lambda: setattr(st, "fg", st.fg - 1)
    return st


def _out(tokens, cache=("cache",)):
    return SimpleNamespace(
        tokens=list(tokens),
        finish_reason="handover",
        final_state=SimpleNamespace(final_trunk_cache=list(cache)),
        stats=SimpleNamespace(to_dict=lambda: {"mode": "mtpk", "verify_calls": 3}),
    )


def test_handover_check_gates(monkeypatch):
    monkeypatch.delenv("MTPLX_LANE_HANDOVER", raising=False)
    monkeypatch.setenv("MTPLX_LANE_HANDOVER_MIN_SOLO_TOKENS", "1")  # hysteresis is covered separately
    assert srv._make_handover_check(_state(_Service()), seed_is_explicit=False) is None
    monkeypatch.setenv("MTPLX_LANE_HANDOVER", "1")
    assert srv._make_handover_check(_state(_Service()), seed_is_explicit=True) is None, "seeded streams never hand over"
    assert srv._make_handover_check(_state(_Service(), mode="serial"), seed_is_explicit=False) is None
    assert srv._make_handover_check(_state(_Service(unavailable="no merge")), seed_is_explicit=False) is None
    check = srv._make_handover_check(_state(_Service(pending=False)), seed_is_explicit=False)
    assert check is not None and check() is False
    st = _state(_Service(pending=True))
    assert srv._make_handover_check(st, seed_is_explicit=False)() is True
    st2 = _state(_Service(pending=False)); st2.model_scheduler = SimpleNamespace(foreground_pending=lambda: 1)
    assert srv._make_handover_check(st2, seed_is_explicit=False)() is False, "scheduler foreground work (commits) must not trigger a handover"


def test_submit_lane_continuation_builds_the_insertable_job(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    service = _Service(); st = _state(service)
    prompt = list(range(10, 20)); solo = [30, 31, 32, 31]
    calls = []
    marker = srv._submit_lane_continuation(
        st, prompt, _out(solo), request_id="req-1", response_max=100, sampler=SamplerConfig(), generation_seed=7,
        generation_limits={"x": 1}, request_observability={"request_id": "req-1"}, token_callback=calls.append,
        prefill_callback=None, cancel_event=None, session_id="s1", session_bank="bank", session_restore_mode="cold",
        session_template_hash="t", session_draft_head_identity=None, session_policy_fingerprint="p",
        token_times=[1.0, 2.0, 3.0, 4.0], started=0.0,
    )
    job = service.jobs[0]
    assert job.continuation is True and job.insert_cache == ["cache"]
    assert job.insert_all_tokens == prompt + solo[:-1] and job.insert_prompt_ids == [solo[-1]], "cache holds prompt + g[:-1]; the last token is inserted"
    assert job.max_tokens == 96 and job.cached_tokens == len(prompt) + 3 and job.session_cache_hit is True
    assert dict(job.completion_token_counts) == {30: 1, 31: 2, 32: 1}, "penalty counts seeded with the solo tokens"
    assert job.token_callback == calls.append and job.session_bank == "bank" and job.session_id == "s1"
    assert job.request_observability["scheduler_lane"] == "solo_mtp->ar_batch"
    assert marker["_handover_job"] is job and marker["_handover_solo_tokens"] == solo and marker["finish_reason"] == "handover"
    assert marker["_handover_solo_token_times"] == [1.0, 2.0, 3.0, 4.0]


def test_submit_lane_continuation_refuses_without_cache_or_tokens(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: set())
    st = _state(_Service())
    common = dict(request_id=None, response_max=10, sampler=SamplerConfig(), generation_seed=0, generation_limits={},
                  request_observability=None, token_callback=None, prefill_callback=None, cancel_event=None, session_id=None,
                  session_bank=None, session_restore_mode="cold", session_template_hash=None, session_draft_head_identity=None,
                  session_policy_fingerprint=None, token_times=[], started=0.0)
    with pytest.raises(RuntimeError):
        srv._submit_lane_continuation(st, [1, 2], _out([]), **common)
    with pytest.raises(RuntimeError):
        srv._submit_lane_continuation(st, [1, 2], SimpleNamespace(tokens=[3], finish_reason="handover", final_state=None, stats=None), **common)


def test_finish_lane_handover_merges_both_segments(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    monkeypatch.setattr(srv, "_strip_terminal_stop", lambda toks, stops: [t for t in toks if t not in stops])
    captured = {}
    def fake_finalize(state, prompt_ids, generated, **kw):
        captured.update(generated=generated, kw=kw); return {"ok": True}
    monkeypatch.setattr(srv, "_finalize_batched_ar_generation", fake_finalize)
    st = _state(_Service())
    job = SimpleNamespace(future=Future(), request_observability={"scheduler_lane": "solo_mtp->ar_batch"})
    job.future.set_result({"tokens": [5, 6, 99], "text": "fg", "stats": {"mode": "ar", "tok_s": 20.0}, "_token_times": [7.0, 8.0], "elapsed_s": 2.0})
    marker = {"_handover_job": job, "_handover_solo_tokens": [1, 2, 3], "_handover_solo_token_times": [1.0, 2.0, 3.0],
              "_handover_solo_stats": {"mode": "mtpk"}, "_handover_started": 0.0}
    out = srv._finish_lane_handover(st, [10, 11], marker, {"session_id": "s1", "session_cache_hit": True, "cache_miss_reason": None})
    assert out == {"ok": True} and st.fg == 0, "foreground accounting balanced"
    g = captured["generated"]
    assert g["tokens"] == [1, 2, 3, 5, 6, 99] and g["text"] == "bcdfg" and g["completion_tokens"] == 6
    assert g["_token_times"] == [1.0, 2.0, 3.0, 7.0, 8.0]
    assert g["stats"]["scheduler_lane"] == "solo_mtp->ar_batch" and g["stats"]["lane_handover"] == {"solo_tokens": 3, "batched_tokens": 3, "solo_stats": {"mode": "mtpk"}}
    assert captured["kw"]["session_restore_mode"] == "solo_mtp->ar_batch" and captured["kw"]["session_id"] == "s1"


def test_prepare_prompt_inputs_skips_continuations():
    from mtplx.server.openai import _BatchedARGenerationService
    svc = _BatchedARGenerationService.__new__(_BatchedARGenerationService)
    seen = []
    svc._prepare_session_bank_restore = lambda job: seen.append(("restore", job.request_id))
    svc._prepare_shared_prefix = lambda jobs: seen.append(("shared", [j.request_id for j in jobs]))
    svc._prepare_stable_prefix = lambda job: seen.append(("stable", job.request_id))
    fresh = SimpleNamespace(request_id="a", continuation=False); cont = SimpleNamespace(request_id="b", continuation=True)
    svc._prepare_prompt_inputs([fresh, cont])
    assert seen == [("restore", "a"), ("shared", ["a"]), ("stable", "a")]


# ---------------------------------------------------------------- return trip (batch -> solo)
def _svc_with(active_jobs, pending=()):
    from mtplx.server.openai import _BatchedARGenerationService
    import threading
    svc = _BatchedARGenerationService.__new__(_BatchedARGenerationService)
    svc._condition = threading.Condition()
    svc._active = {uid: job for uid, job in active_jobs}
    svc._pending = list(pending)
    return svc


class _Gen:
    def __init__(self): self.removed = []
    def remove(self, uids): self.removed.extend(uids)


def _cont_job(tokens, max_tokens=200, return_to_solo=True, continuation=True):
    job = SimpleNamespace(continuation=continuation, return_to_solo=return_to_solo, future=Future(), tokens=list(tokens),
                          max_tokens=max_tokens, token_times=[0.5] * len(tokens), created_s=0.0, request_id="r", session_id="s",
                          cancel_requested=lambda: False)
    return job


def test_return_to_solo_pulls_the_last_continuation_row(monkeypatch):
    monkeypatch.delenv("MTPLX_LANE_HANDOVER_RETURN_MIN_TOKENS", raising=False)
    job = _cont_job([4, 5, 6]); svc = _svc_with([(7, job)]); gen = _Gen()
    svc._maybe_return_to_solo(gen)
    assert gen.removed == [7] and svc._active == {} and job.future.done()
    res = job.future.result()
    assert res["_resume_solo"] is True and res["tokens"] == [4, 5, 6] and res["finish_reason"] == "handover_return"


def test_return_to_solo_leaves_other_shapes_alone():
    gen = _Gen()
    # two rows active
    a, b = _cont_job([1]), _cont_job([2]); svc = _svc_with([(1, a), (2, b)]); svc._maybe_return_to_solo(gen)
    # one row but something pending
    c = _cont_job([1]); svc = _svc_with([(1, c)], pending=[SimpleNamespace(cancel_requested=lambda: False)]); svc._maybe_return_to_solo(gen)
    # a fresh (non-continuation) row
    d = _cont_job([1], continuation=False); svc = _svc_with([(1, d)]); svc._maybe_return_to_solo(gen)
    # handover state was not banked
    e = _cont_job([1], return_to_solo=False); svc = _svc_with([(1, e)]); svc._maybe_return_to_solo(gen)
    # near the end of its budget
    f = _cont_job(list(range(190)), max_tokens=200); svc = _svc_with([(1, f)]); svc._maybe_return_to_solo(gen)
    assert gen.removed == [] and not any(j.future.done() for j in (a, b, c, d, e, f))


def test_finish_lane_handover_resumes_on_the_solo_lane(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    monkeypatch.setattr(srv, "_strip_terminal_stop", lambda toks, stops: [t for t in toks if t not in stops])
    calls = {}
    class _F:
        def __init__(self, fn): self.fn = fn
        def result(self): return self.fn()
    monkeypatch.setattr(srv, "_submit_foreground_model_work", lambda state, fn, batch_key=None: _F(fn))
    def fake_run_generation(state, prompt_ids, **kw):
        calls["prompt_ids"] = list(prompt_ids); calls["max_tokens"] = kw.get("max_tokens")
        return {"tokens": [7, 8, 99], "text": "hi", "stats": {"mode": "mtpk"}, "finish_reason": "stop"}
    monkeypatch.setattr(srv, "_run_generation", fake_run_generation)
    st = _state(_Service())
    job = SimpleNamespace(future=Future(), max_tokens=100, request_id="r", session_id="s", request_observability={})
    job.future.set_result({"_resume_solo": True, "tokens": [4, 5, 6], "stats": {"mode": "ar"}, "_token_times": [], "elapsed_s": 1.0})
    marker = {"_handover_job": job, "_handover_solo_tokens": [1, 2, 3], "_handover_solo_token_times": [], "_handover_solo_stats": {}, "_handover_started": 0.0}
    out = srv._finish_lane_handover(st, [10, 11], marker, {"max_tokens": 100, "temperature": 0.0})
    assert calls["prompt_ids"] == [10, 11, 1, 2, 3, 4, 5, 6] and calls["max_tokens"] == 97, "resume prompt = prompt + solo + batched; budget minus batched"
    assert out["tokens"] == [1, 2, 3, 4, 5, 6, 7, 8, 99] and out["text"] == "bcdefghi" and out["completion_tokens"] == 9
    assert out["stats"]["scheduler_lane"] == "solo_mtp->ar_batch->solo_mtp"
    assert out["stats"]["lane_handover"] == {"solo_tokens": 3, "batched_tokens": 3, "resumed_solo_tokens": 3, "solo_stats": {}}
    assert st.fg == 0


def test_submit_lane_continuation_banks_the_handover_state_for_the_return(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: set())
    monkeypatch.setattr(srv, "snapshot_cache", lambda cache: ("snap", cache))
    monkeypatch.setattr(srv, "_bank_history_policy", lambda state: "committed")
    monkeypatch.delenv("MTPLX_LANE_HANDOVER_RETURN", raising=False)
    puts = []
    bank = SimpleNamespace(put_snapshot=lambda **kw: puts.append(kw) or object())
    service = _Service(); st = _state(service); st.draft_head_identity = "dh"
    out = _out([30, 31, 32]); out.final_state.final_committed_mtp_cache = ["mtp"]
    common = dict(request_id="r", response_max=50, sampler=SamplerConfig(), generation_seed=0, generation_limits={}, request_observability={},
                  token_callback=None, prefill_callback=None, cancel_event=None, session_id="s1", session_bank=bank, session_restore_mode="cold",
                  session_template_hash="t", session_draft_head_identity=None, session_policy_fingerprint="p", token_times=[], started=0.0)
    srv._submit_lane_continuation(st, [1, 2, 3], out, **common)
    job = service.jobs[-1]
    assert job.return_to_solo is True and len(puts) == 1
    put = puts[0]
    assert put["token_ids"] == [1, 2, 3, 30, 31] and put["snapshot_epoch"] == 5 and put["mtp_snapshot_epoch"] == 5
    assert put["cache_snapshot"] == ("snap", ["cache"]) and put["mtp_history_snapshot"] == ("snap", ["mtp"])
    assert put["session_id"] == "s1" and put["policy_fingerprint"] == "p" and put["hidden_variant"] == "post_norm" and put["draft_head_identity"] == "dh"
    # no MTP history -> banked without it, no return trip
    out2 = _out([30, 31, 32]); out2.final_state.final_committed_mtp_cache = None
    srv._submit_lane_continuation(st, [1, 2, 3], out2, **common)
    assert service.jobs[-1].return_to_solo is False


# ---------------------------------------------------------------- hysteresis + interleave
def test_handover_check_hysteresis(monkeypatch):
    monkeypatch.setenv("MTPLX_LANE_HANDOVER", "1"); monkeypatch.setenv("MTPLX_LANE_HANDOVER_MIN_SOLO_TOKENS", "3"); monkeypatch.setenv("MTPLX_LANE_HANDOVER_COOLDOWN_S", "100")
    st = _state(_Service(pending=True)); check = srv._make_handover_check(st, seed_is_explicit=False)
    assert [check(), check(), check()] == [False, False, True], "at least min_solo_tokens polls before a handover"
    st2 = _state(_Service(pending=True)); st2._lane_handover_last_resume_s = __import__("time").perf_counter()
    check2 = srv._make_handover_check(st2, seed_is_explicit=False)
    assert [check2() for _ in range(5)] == [False] * 5, "no handover within the cooldown after a return trip"



def test_continuation_jumps_the_pending_queue():
    """A handover continuation must be inserted ahead of waiting prompts:
    mlx-lm's prompt batch holds prefill_batch_size (2) sequences, so a
    continuation queued behind two cold prompts sat through their whole
    prefill with no decode step (2026-09-24 live receipt: 7.5 s at 0 tok/s)."""
    import mtplx.server.openai as srv

    svc = srv._BatchedARGenerationService.__new__(srv._BatchedARGenerationService)
    svc._condition = srv.Condition()
    svc._pending = []
    svc._pump_scheduled = True  # never schedule a real pump
    def job(name, continuation=False):
        j = SimpleNamespace(continuation=continuation, name=name, future=srv.Future())
        return j
    svc.submit(job("p1")); svc.submit(job("p2"))
    svc.submit(job("c1", continuation=True))
    svc.submit(job("p3"))
    svc.submit(job("c2", continuation=True))
    assert [j.name for j in svc._pending] == ["c1", "c2", "p1", "p2", "p3"]


# ---------------------------------------------------------------- prefill-phase handover
def test_maybe_prefill_handover_raises_only_when_wanted():
    from mtplx.generation import PrefillHandover, _maybe_prefill_handover
    seen = []
    _maybe_prefill_handover(None, "c", 2048, 100_000)  # no check: nothing
    _maybe_prefill_handover(lambda p, t: seen.append((p, t)) or False, "c", 2048, 100_000)
    assert seen == [(2048, 100_000)]
    _maybe_prefill_handover(lambda p, t: True, "c", 0, 100)  # nothing in the cache yet
    _maybe_prefill_handover(lambda p, t: True, "c", 100, 100)  # prefill complete
    _maybe_prefill_handover(lambda p, t: 1 / 0, "c", 50, 100)  # a broken check never stops prefill
    with pytest.raises(PrefillHandover) as info:
        _maybe_prefill_handover(lambda p, t: True, ["cache"], 4096, 100_000)
    assert info.value.cache == ["cache"] and info.value.prefix_len == 4096


def test_prefill_handover_output_carries_the_prefix():
    from mtplx.generation import PrefillHandover, _prefill_handover_output
    out = _prefill_handover_output(SimpleNamespace(mtp_enabled=True), PrefillHandover(["c"], 6144), started_s=0.0, mtp_history_policy="committed")
    assert out.finish_reason == "handover" and out.tokens == [] and out.text == ""
    assert out.final_state.final_trunk_cache == ["c"]
    assert out.final_state.extra_state == {"prefill_handover_prefix_len": 6144}
    assert out.final_state.safe_to_commit is False and out.stats.generated_tokens == 0
    assert out.stats.to_dict()["mode"] == "mtpk"


def test_prefill_handover_check_gates(monkeypatch):
    monkeypatch.setenv("MTPLX_LANE_HANDOVER", "1")
    monkeypatch.setenv("MTPLX_LANE_HANDOVER_PREFILL_MIN_REMAINING_TOKENS", "4096")
    monkeypatch.delenv("MTPLX_LANE_HANDOVER_PREFILL", raising=False)
    st = _state(_Service(pending=True))
    check = srv._make_prefill_handover_check(st, seed_is_explicit=False)
    assert check is not None
    assert check(2048, 100_000) is True, "pending lane + plenty left -> hand over"
    assert check(98_000, 100_000) is False, "under the min remaining: finish solo"
    st2 = _state(_Service(pending=False))
    assert srv._make_prefill_handover_check(st2, seed_is_explicit=False)(2048, 100_000) is False
    assert srv._make_prefill_handover_check(st, seed_is_explicit=True) is None
    monkeypatch.setenv("MTPLX_LANE_HANDOVER_PREFILL", "0")
    assert srv._make_prefill_handover_check(st, seed_is_explicit=False) is None
    monkeypatch.setenv("MTPLX_LANE_HANDOVER_PREFILL", "1")
    st.  _lane_handover_last_resume_s = __import__("time").perf_counter()
    assert srv._make_prefill_handover_check(st, seed_is_explicit=False)(2048, 100_000) is False, "cooldown after a return"


def test_submit_lane_continuation_prefill_handover(monkeypatch):
    monkeypatch.setattr(srv, "_default_stop_tokens", lambda tok: {99})
    service = _Service(); st = _state(service)
    prompt = list(range(100, 200))
    out = SimpleNamespace(
        tokens=[], finish_reason="handover",
        final_state=SimpleNamespace(final_trunk_cache=["partial"], extra_state={"prefill_handover_prefix_len": 40}),
        stats=SimpleNamespace(to_dict=lambda: {"mode": "mtpk"}),
    )
    banked = []
    bank = SimpleNamespace(put_snapshot=lambda **kw: banked.append(kw) or object())
    marker = srv._submit_lane_continuation(
        st, prompt, out, request_id="req-p", response_max=120, sampler=SamplerConfig(), generation_seed=1,
        generation_limits={}, request_observability={"request_id": "req-p"}, token_callback=None,
        prefill_callback=None, cancel_event=None, session_id="s", session_bank=bank, session_restore_mode="cold",
        session_template_hash="t", session_draft_head_identity=None, session_policy_fingerprint="p",
        token_times=[], started=0.0,
    )
    job = service.jobs[0]
    assert job.continuation is True and job.insert_cache == ["partial"]
    assert job.insert_all_tokens == prompt[:40] and job.insert_prompt_ids == prompt[40:]
    assert job.cached_tokens == 40 and job.max_tokens == 120, "no solo tokens: the whole budget"
    assert job.session_cache_hit is False, "the row's first token banks the prompt boundary"
    assert job.effective_restore_mode == "lane_handover_prefill"
    assert job.return_to_solo is False and banked == [], "no MTP history yet: no return trip, nothing cloned"
    assert marker["_handover_solo_tokens"] == [] and marker["finish_reason"] == "handover"
    assert job.request_observability["lane_handover"]["prefill_prefix_len"] == 40
    with pytest.raises(RuntimeError):
        bad = SimpleNamespace(tokens=[], finish_reason="handover",
                              final_state=SimpleNamespace(final_trunk_cache=["c"], extra_state={"prefill_handover_prefix_len": 100}),
                              stats=out.stats)
        srv._submit_lane_continuation(st, prompt, bad, request_id="x", response_max=10, sampler=SamplerConfig(), generation_seed=1,
            generation_limits={}, request_observability={}, token_callback=None, prefill_callback=None, cancel_event=None,
            session_id=None, session_bank=None, session_restore_mode="cold", session_template_hash=None,
            session_draft_head_identity=None, session_policy_fingerprint=None, token_times=[], started=0.0)
