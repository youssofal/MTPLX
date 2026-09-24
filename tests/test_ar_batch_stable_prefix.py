"""Batched lane session handling (#420 follow-up, 2026-09-24 opencode receipt).

(1) `_prepare_stable_prefix`: bank the STABLE prefix (prompt minus MTPLX's transient trailing hint)
as an exact-prefix entry before the batch generator runs, advancing the restored or fresh cache to
the stable edge with the runtime's own forward. (2) `_wait_pending_postcommit`: a same-session
follow-up turn waits for the previous turn's pending postcommit like the solo lane does.
CPU-only, fakes for runtime / bank / sessions; no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import mtplx.server.openai as srv
from mtplx.sampling import SamplerConfig
from mtplx.server.openai import _BatchedARGenerationService, _BatchedARJob


class _Entry:
    def __init__(self):
        self.state = mx.zeros((1, 1))

    def merge(self, others):
        return self


class _Runtime:
    def __init__(self):
        self.forwarded = []
        self.model_path = "m"

    def make_cache(self):
        return [_Entry(), _Entry()]

    def forward_ar(self, ids, cache=None, return_hidden=False):
        self.forwarded.append(ids.tolist()[0])
        return mx.zeros((1, ids.shape[1], 4))


class _Bank:
    def __init__(self):
        self.puts = []

    def put_snapshot(self, **kw):
        self.puts.append(kw)
        return object()

    def restore(self, *a, **k):
        return None


class _Session:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def wait_for_pending_postcommit(self):
        self.calls += 1
        return self.outcome


def _service(runtime, session=None):
    svc = _BatchedARGenerationService.__new__(_BatchedARGenerationService)
    svc.state = SimpleNamespace(runtime=runtime, sessions=SimpleNamespace(peek=lambda sid: session))
    return svc


def _job(prompt_ids, *, stable=None, bank=None, session_id="s1"):
    obs = {} if stable is None else {"stable_prefix_len": stable}
    job = _BatchedARJob(
        request_id="req", prompt_ids=prompt_ids, max_tokens=8, sampler=SamplerConfig(), seed=0,
        stop_token_ids=set(), token_callback=None, prefill_callback=None, request_observability=obs,
        mtp_disabled_reason=None, generation_limits={}, seed_is_explicit=False,
    )
    job.session_bank = bank
    job.session_id = session_id
    job.session_template_hash = "t"
    job.session_policy_fingerprint = "p"
    return job


def test_stable_prefix_is_prefilled_in_chunks_banked_and_split_from_the_hint(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    monkeypatch.setattr(srv, "_STABLE_PREFIX_PREFILL_CHUNK", 3)
    monkeypatch.setattr(srv, "snapshot_cache", lambda cache: SimpleNamespace(states=[], meta_states=[]))
    rt = _Runtime(); bank = _Bank(); svc = _service(rt)
    prompt = list(range(100, 112))  # 12 tokens: stable 9, hint 3
    job = _job(prompt, stable=9, bank=bank)
    svc._prepare_stable_prefix(job)
    assert rt.forwarded == [prompt[0:3], prompt[3:6], prompt[6:9]], "stable edge reached in 3-token chunks, hint untouched"
    assert len(bank.puts) == 1 and bank.puts[0]["token_ids"] == prompt[:9] and bank.puts[0]["snapshot_epoch"] == 9
    assert bank.puts[0]["session_id"] == "s1" and bank.puts[0]["template_hash"] == "t"
    assert job.insert_all_tokens == prompt[:9] and job.insert_prompt_ids == prompt[9:], "generator prefills only the hint"
    assert job.insert_cache is not None and job.request_observability["ar_batch_stable_prefix_bank_stored"] is True
    assert job.request_observability["ar_batch_stable_prefix_prefilled"] == 9


def test_stable_prefix_extends_a_shorter_restore_and_skips_a_longer_one(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    monkeypatch.setattr(srv, "snapshot_cache", lambda cache: SimpleNamespace(states=[], meta_states=[]))
    rt = _Runtime(); bank = _Bank(); svc = _service(rt)
    prompt = list(range(20))
    job = _job(prompt, stable=15, bank=bank)
    restored = rt.make_cache(); job.insert_cache = restored; job.insert_all_tokens = prompt[:10]; job.insert_prompt_ids = prompt[10:]
    svc._prepare_stable_prefix(job)
    assert rt.forwarded == [prompt[10:15]], "only the gap between the restore point and the stable edge is forwarded"
    assert job.insert_cache is restored and job.insert_all_tokens == prompt[:15] and job.insert_prompt_ids == prompt[15:]
    # a restore that already covers the stable edge is left alone
    rt2 = _Runtime(); bank2 = _Bank(); svc2 = _service(rt2)
    job2 = _job(prompt, stable=15, bank=bank2)
    job2.insert_cache = rt2.make_cache(); job2.insert_all_tokens = prompt[:17]; job2.insert_prompt_ids = prompt[17:]
    svc2._prepare_stable_prefix(job2)
    assert rt2.forwarded == [] and bank2.puts == [] and job2.insert_all_tokens == prompt[:17]


def test_stable_prefix_gates(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    rt = _Runtime(); svc = _service(rt); prompt = list(range(12))
    for job in (
        _job(prompt, stable=None, bank=_Bank()),   # no stable length known
        _job(prompt, stable=9, bank=None),         # no bank
        _job(prompt, stable=12, bank=_Bank()),     # stable == whole prompt: nothing to split
        _job(prompt, stable=2, bank=_Bank()),      # below the minimum
    ):
        svc._prepare_stable_prefix(job)
        assert rt.forwarded == [] and job.insert_cache is None and job.insert_prompt_ids == prompt


def test_stable_prefix_failure_leaves_the_job_untouched(monkeypatch):
    monkeypatch.setattr(srv, "_STABLE_PREFIX_BANK_MIN_TOKENS", 4)
    rt = _Runtime()
    rt.forward_ar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    svc = _service(rt); prompt = list(range(12)); job = _job(prompt, stable=9, bank=_Bank())
    svc._prepare_stable_prefix(job)
    assert job.insert_cache is None and job.insert_prompt_ids == prompt
    assert job.request_observability["ar_batch_stable_prefix_error"].startswith("RuntimeError")


def test_restore_waits_for_the_sessions_pending_postcommit():
    session = _Session({"waited": True, "outcome": "completed", "elapsed_s": 0.2})
    rt = _Runtime(); svc = _service(rt, session=session); bank = _Bank()
    job = _job(list(range(600)), bank=bank)
    svc._prepare_session_bank_restore(job)
    assert session.calls == 1 and job.request_observability["ar_batch_postcommit_wait"]["outcome"] == "completed"
    # no session / no bank: no wait, no error
    job2 = _job(list(range(600)), bank=bank, session_id=None); svc._prepare_session_bank_restore(job2)
    assert session.calls == 1


def test_restore_retries_a_live_reference_entry_as_a_lease():
    class _LeaseBank:
        def __init__(self):
            self.calls = []
            self.last_ram_miss_reason = None
            self.last_miss_reason = None

        def restore(self, runtime, prompt_ids, *, mode, **kw):
            self.calls.append(mode)
            if mode == "clone":
                self.last_ram_miss_reason = "no_snapshot_coverage"
                self.last_miss_reason = "ssd_prefix_miss"
                return None
            entry = SimpleNamespace(prefix_len=len(prompt_ids) - 5, token_ids=list(prompt_ids[:-5]))
            return SimpleNamespace(cache=[_Entry()], entry=entry, restore_mode="reference_lease")

    rt = _Runtime(); svc = _service(rt); bank = _LeaseBank()
    job = _job(list(range(600)), bank=bank)
    assert svc._prepare_session_bank_restore(job) is True
    assert bank.calls == ["clone", "reference"]
    assert job.cached_tokens == 595 and job.insert_cache is not None and job.cache_miss_reason is None
    assert job.request_observability["ar_batch_restore_retried_as_reference"] is True


def test_restore_does_not_retry_other_ram_misses():
    class _MissBank:
        def __init__(self):
            self.calls = []
            self.last_ram_miss_reason = "policy_mismatch"
            self.last_miss_reason = "ssd_prefix_miss"

        def restore(self, runtime, prompt_ids, *, mode, **kw):
            self.calls.append(mode); return None

    rt = _Runtime(); svc = _service(rt); bank = _MissBank()
    job = _job(list(range(600)), bank=bank)
    assert svc._prepare_session_bank_restore(job) is False and bank.calls == ["clone"]


def _near_bank(prefix_len_extra=200, boundary_back=40, live_ref=False, exact=None):
    """Fake bank: exact restore returns `exact`; near-prefix candidate = one entry longer than the
    prompt with a recurrent boundary `boundary_back` tokens below the match."""
    class _Bank:
        last_ram_miss_reason = "prefix_divergence_at_token"
        last_miss_reason = "ssd_prefix_miss"
        def __init__(self): self.calls = []; self.restore_calls = []
        def restore(self, runtime, prompt_ids, *, mode, **kw):
            self.calls.append(mode); return exact
        def longest_prefix(self, prompt_ids): return None
        def near_prefix_candidates(self, head, **kw):
            self.candidate_kw = kw
            entry = SimpleNamespace(prefix_len=len(head) + prefix_len_extra, token_ids=list(head) + [7] * prefix_len_extra,
                                    mtp_history_snapshot="hist", mtp_history_cache_ref=None, live_ref_only=live_ref,
                                    cache_ref=("live" if live_ref else None), cache_source="ram", model_path="m")
            return [(entry, len(head))]
        def restore_entry_prefix_cache(self, rt, entry, matched, mode="clone", **kw):
            self.restore_calls.append((matched, mode))
            return ([_Entry()], None, mode, matched - boundary_back, None)
    return _Bank()


def _near_rt():
    rt = _Runtime(); rt.contract = SimpleNamespace(hidden_variant="post_norm", base_hidden_variant="post_norm"); rt.mtp_enabled = True
    return rt


def test_exact_miss_falls_back_to_the_near_prefix_boundary_restore(monkeypatch):
    import mtplx.generation as gen
    monkeypatch.setattr(gen, "_entry_matches_restore_lookup", lambda *a, **k: True)
    rt = _near_rt(); svc = _service(rt); bank = _near_bank(boundary_back=40); prompt = list(range(700)); job = _job(prompt, bank=bank)
    assert svc._prepare_session_bank_restore(job) is True
    assert bank.restore_calls == [(699, "clone")], "lookup on the prompt minus its last token; snapshot entry -> clone"
    assert job.cached_tokens == 659 and job.insert_all_tokens == prompt[:659] and job.insert_prompt_ids == prompt[659:], "the suffix goes to the batch generator, not prefilled inline"
    assert job.session_cache_hit and job.cache_miss_reason is None and job.effective_restore_mode == "ar_batch_near_prefix:clone"
    assert job.request_observability["ar_batch_near_prefix_restore"]["suffix_to_generator"] == 41
    assert bank.candidate_kw["template_hash"] == "t" and bank.candidate_kw["policy_fingerprint"] == "p"


def test_near_prefix_live_ref_entry_restores_by_reference(monkeypatch):
    import mtplx.generation as gen
    monkeypatch.setattr(gen, "_entry_matches_restore_lookup", lambda *a, **k: True)
    rt = _near_rt(); svc = _service(rt); bank = _near_bank(live_ref=True); job = _job(list(range(700)), bank=bank)
    assert svc._prepare_session_bank_restore(job) is True and bank.restore_calls == [(699, "reference")]


def test_near_prefix_miss_keeps_the_exact_miss_reason(monkeypatch):
    class _EmptyBank(_near_bank().__class__):
        def near_prefix_candidates(self, head, **kw): return []
    rt = _near_rt(); svc = _service(rt); job = _job(list(range(700)), bank=_EmptyBank())
    assert svc._prepare_session_bank_restore(job) is False and job.cache_miss_reason == "ssd_prefix_miss"


def test_longer_exact_entry_falls_through_to_the_boundary_restore(monkeypatch):
    """The exact entry is the previous turn's prompt + generation: longer than the prompt, not
    insertable. The lane must use the boundary restore (as the solo lane does), not refuse."""
    import mtplx.generation as gen
    monkeypatch.setattr(gen, "_entry_matches_restore_lookup", lambda *a, **k: True)
    exact = SimpleNamespace(cache=[_Entry()], entry=SimpleNamespace(prefix_len=860, token_ids=list(range(860))), restore_mode="clone")
    rt = _near_rt(); svc = _service(rt); bank = _near_bank(exact=exact, boundary_back=1); bank.last_ram_miss_reason = None
    job = _job(list(range(700)), bank=bank)
    assert svc._prepare_session_bank_restore(job) is True
    assert bank.calls == ["clone"], "no reference lease is taken for a non-insertable entry"
    assert job.cached_tokens == 698 and job.insert_prompt_ids == [698, 699]


def test_longer_live_ref_entry_is_not_leased(monkeypatch):
    class _LiveBank(_near_bank().__class__):
        last_ram_miss_reason = "no_snapshot_coverage"
        def __init__(self): super().__init__()
        def restore(self, runtime, prompt_ids, *, mode, **kw): self.calls.append(mode); return None
        def longest_prefix(self, prompt_ids): return SimpleNamespace(prefix_len=len(prompt_ids) + 40)
        def near_prefix_candidates(self, head, **kw): return []
    rt = _near_rt(); svc = _service(rt); bank = _LiveBank(); job = _job(list(range(700)), bank=bank)
    assert svc._prepare_session_bank_restore(job) is False
    assert bank.calls == ["clone"], "the lease retry is skipped when the entry could not be inserted anyway"
    assert job.cache_miss_reason == "ar_batch_full_prefix_not_insertable"


def test_interleaved_generator_takes_extra_decode_steps_only_while_prefilling(monkeypatch):
    class _GenBatch:
        def __init__(self, n): self.n = n; self.calls = 0
        def __len__(self): return self.n
        def next(self): self.calls += 1; return [("tok", self.calls)]
    class _Base:
        def __init__(self): self._generation_batch = _GenBatch(2); self._currently_processing = ["p"]; self._unprocessed_sequences = []; self._gen_tokens_counter = 0; self._steps_counter = 0; self.base_calls = 0
        def _next(self): self.base_calls += 1; return (["prompt"], [("tok", "base")])
    cls = srv._interleaved_batch_generator_class(_Base)
    monkeypatch.setenv("MTPLX_AR_BATCH_DECODE_STEPS_PER_CHUNK", "4")
    g = cls(); p, r = g._next()
    assert p == ["prompt"] and r == [("tok", 1), ("tok", 2), ("tok", 3), ("tok", "base")] and g._steps_counter == 3
    g2 = cls(); g2._currently_processing = []; g2._unprocessed_sequences = []
    assert g2._next() == (["prompt"], [("tok", "base")]), "no prefill in progress: stock behaviour"
    monkeypatch.setenv("MTPLX_AR_BATCH_DECODE_STEPS_PER_CHUNK", "1")
    g3 = cls(); assert g3._next() == (["prompt"], [("tok", "base")])
