"""CPU unit tests for Inkling K3 fused round_program timing and verify_forward_time_s accounting."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import route_tape
from mtplx.generation import PromptState
from mtplx.models.inkling import generation
from mtplx.models.inkling.fast import engine_site, round as fast_round, round_program
from mtplx.sampling import SamplerConfig

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class _FakeClock:
    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class _StubRuntime:
    backend_id = "inkling2_loop"
    trunk_entries = 1

    def __init__(self, *, vocab: int = 16, round_warm: dict | None = None, round_cost_detail: dict | None = None):
        self.vocab = vocab
        self.model = None
        self.fast = None
        self.round_warm = round_warm
        self.round_cost_detail = round_cost_detail
        self.drafter = SimpleNamespace(
            width=8,
            propose=self._draft_propose,
        )
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda _s: vocab + 50,
            decode=lambda ids: str(ids),
        )
        self.clock: _FakeClock | None = None
        self.verify_delay_s = 0.0
        self.verify_route: str | None = None
        self.commit_delay_s = 0.0
        self.draft_delay_s = 0.0
        self.committed: list[tuple[int, int]] = []

    def scores(self, token: int) -> mx.array:
        nxt = (int(token) + 1) % self.vocab
        return mx.array([6.0 if i == nxt else -6.0 for i in range(self.vocab)], mx.float32)

    def _draft_propose(self, _model, anchor, _cache, count, **_kw):
        if self.clock is not None and self.draft_delay_s:
            self.clock.advance(self.draft_delay_s)
        cur = int(anchor)
        out = []
        for _ in range(count):
            cur = (cur + 1) % self.vocab
            out.append(cur)
        return mx.array(out, mx.int32)

    def verify(self, block, cache):
        if self.clock is not None and self.verify_delay_s:
            self.clock.advance(self.verify_delay_s)
        if self.verify_route is not None:
            engine_site._note(self.verify_route, None, len(block))
        ids = block if isinstance(block, list) else block.tolist()
        cache[0].extend(ids)
        rows = mx.stack([self.scores(t) for t in ids])
        features = [mx.array([[ids]]).transpose(0, 2, 1)]
        return rows, features, None

    def commit(self, cache, features, tapes, kept, total):
        if self.clock is not None and self.commit_delay_s:
            self.clock.advance(self.commit_delay_s)
        self.committed.append((kept, total))
        if kept < total:
            del cache[0][-(total - kept) :]
        cache[1].extend(cache[0][-kept:])

    def forward_ar(self, ids, cache, **_kw):
        t = int(ids.item())
        cache[0].append(t)
        cache[1].append(t)
        return self.scores(t)[None, None], mx.array([[[t]]])

    def prefill(self, _rt, ids, **_kw):
        return PromptState(
            [list(ids), list(ids)],
            self.scores(ids[-1])[None, None],
            mx.zeros((1, 1, 1)),
            None,
            tuple(ids),
            0.0,
            suffix_tokens=len(ids),
        )


def _run_generate(rt, prompt, *, max_tokens=16, depth=3, sampler=GREEDY):
    return generation.generate(
        rt,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        depth=depth,
        seed=7,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        session_bank=None,
        session_id=None,
        session_restore_mode="cold",
        session_template_hash=None,
        session_draft_head_identity=None,
        session_policy_fingerprint=None,
        capture_final_state=False,
        commit_prompt_state_to_bank=False,
        vision_splice=None,
        constraint=None,
        abort_check=None,
        first_token_logprobs_top_k=None,
        store_prefix_snapshot=False,
    )


def _assert_timing_invariants(stats, *, min_calls: int = 1):
    raw = asdict(stats)
    for key in ("draft_time_s", "verify_time_s", "verify_forward_time_s", "verify_calls"):
        assert key in raw
    assert stats.verify_calls >= min_calls
    assert stats.draft_time_s > 0.0
    assert stats.verify_forward_time_s > 0.0
    assert stats.verify_time_s >= stats.verify_forward_time_s
    assert stats.draft_time_s + stats.verify_time_s <= stats.decode_elapsed_s + 1e-9
    assert stats.decode_elapsed_s <= stats.elapsed_s + 1e-9


def test_stage_shares_lookup_priority_and_clamping():
    # Normal native calibration: 28ms fused round, 21ms verify forward, 1ms decide -> ~0.214 draft, 0.75 vf.
    d_share, vf_share = round_program._compute_stage_shares(28.0, 21.0, decide_ms=1.0)
    assert vf_share == pytest.approx(0.75, abs=1e-3)
    assert d_share == pytest.approx(0.2143, abs=1e-3)
    assert d_share + vf_share <= 0.98

    # Thermal jitter where step_ms exceeds round_ms still clamps to strictly positive bounded shares.
    d_clamped, vf_clamped = round_program._compute_stage_shares(22.0, 24.0, decide_ms=1.0)
    assert 0.05 <= d_clamped <= 0.60
    assert 0.35 <= vf_clamped <= 0.90
    assert d_clamped + vf_clamped <= 0.98

    # With standalone draft_ms present, fallback normalizes piece timings directly.
    d_piece, vf_piece = round_program._compute_stage_shares(20.0, 22.0, decide_ms=1.0, draft_ms=6.0)
    assert d_piece == pytest.approx(6.0 / 29.0, abs=1e-3)
    assert vf_piece == pytest.approx(22.0 / 29.0, abs=1e-3)

    # Lookup order: round_warm["stage_shares"] exact -> nearest -> round_cost_detail -> default.
    rt_warm = SimpleNamespace(
        round_warm={"stage_shares": {3: {"draft": 0.24, "verify_forward": 0.72}}},
        round_cost_detail=None,
    )
    assert round_program.stage_shares(rt_warm, 3) == (0.24, 0.72)
    assert round_program.stage_shares(rt_warm, 5) == (0.24, 0.72)  # nearest calibrated width

    rt_detail = SimpleNamespace(
        round_warm=None,
        round_cost_detail={"timing": {"step_ms": {4: 21.0}, "draft_ms": {3: 6.0}, "decide_ms": {3: 1.0}}},
    )
    assert round_program.stage_shares(rt_detail, 3) == (round(6.0 / 28.0, 4), round(21.0 / 28.0, 4))

    rt_bare = SimpleNamespace()
    assert round_program.stage_shares(rt_bare, 3) == round_program._DEFAULT_STAGE_SHARES
    assert round_program.stage_shares(rt_bare, 3, committed=True) == round_program._DEFAULT_COMMIT_STAGE_SHARES


def test_warm_calibrates_native_stage_shares_on_cpu_mock(monkeypatch):
    clock = _FakeClock(10.0)

    class _FakeNativeRound:
        native = True
        packet = mx.zeros((8,), mx.int32)
        rows = mx.zeros((4, 16), mx.float32)
        features = []
        pending = SimpleNamespace(deltas=[])

    def fake_serve(rt, anchor, cache, k, **kwargs):
        clock.advance(0.030)  # 30ms fused round
        return _FakeNativeRound()

    def fake_step(ids, state):
        clock.advance(0.022)  # 22ms standalone native verify forward
        engine_site._note("native", None, int(ids.shape[0]))
        return mx.zeros((1, int(ids.shape[0]), 16), mx.float32), [], []

    def fake_decide(logits, proposals, **kwargs):
        clock.advance(0.001)  # 1ms GPU decision
        return mx.zeros((2 + int(logits.shape[0]) + int(proposals.shape[0]),), mx.int32)

    monkeypatch.setattr(round_program, "serve", fake_serve)
    monkeypatch.setattr(round_program.engine_runtime, "mode", lambda: "auto")
    monkeypatch.setattr(round_program.engine_runtime, "available", lambda: True)
    monkeypatch.setattr(round_program.engine_site, "_synthetic_cache", lambda rt, length, seed: [object()])
    monkeypatch.setattr(round_program, "_synthetic_draft_cache", lambda rt, length, seed: [object()])
    monkeypatch.setattr(round_program, "DecodeState", lambda model, cache: SimpleNamespace(prepare=lambda: None))
    monkeypatch.setattr(round_program, "site", lambda rt: SimpleNamespace(report=lambda: {}))
    monkeypatch.setattr(round_program.fast_round, "decide", fake_decide)
    monkeypatch.setattr(round_program.fast_round, "read_packet", lambda packet, width: None)
    monkeypatch.setattr(round_program, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(round_program.mx, "synchronize", lambda: None)

    rt = SimpleNamespace(
        drafter=object(),
        trunk_entries=1,
        fast=SimpleNamespace(step=fake_step),
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=mx.zeros((16, 4)))),
    )
    report = round_program.warm(rt, (3,))
    assert report["round_ms"] == {3: 30.0}
    assert 3 in report["stage_shares"]
    entry = report["stage_shares"][3]
    assert entry["verify_forward"] == pytest.approx(22.0 / 30.0, abs=1e-3)
    assert entry["draft"] == pytest.approx(7.0 / 30.0, abs=1e-3)
    assert report["counts"][3]["verify_forward_ms"] == pytest.approx(22.0, abs=0.1)


@pytest.mark.parametrize("exec_mode", ["native_sync", "mlx_lazy", "committed"])
def test_fused_round_program_attributes_draft_verify_and_forward_without_extra_eval(monkeypatch, exec_mode):
    clock = _FakeClock(50.0)
    rt = _StubRuntime(
        round_warm={"stage_shares": {3: {"draft": 0.25, "verify_forward": 0.70}}},
    )
    rt.clock = clock
    rt.commit_delay_s = 0.002 if exec_mode != "committed" else 0.0

    eval_calls = []

    def fake_settle(*values):
        eval_calls.append(len(values))
        if exec_mode == "mlx_lazy":
            # In lazy MLX mode the fused round executes during read_packet's settle.
            clock.advance(0.040)

    def fake_serve(rt_, anchor, cache, k, **kwargs):
        if exec_mode in ("native_sync", "committed"):
            # Native replay executes draft + verify_forward + decision inside serve().
            clock.advance(0.040)
        else:
            clock.advance(0.002)  # lazy graph build only
        block_ids = [(int(anchor) + i) % rt_.vocab for i in range(k + 1)]
        rows = mx.stack([rt_.scores(t) for t in block_ids])
        features = [mx.array([[block_ids]]).transpose(0, 2, 1)]
        d_share, vf_share = round_program.stage_shares(rt_, k, committed=(exec_mode == "committed"))
        pkt = mx.zeros((2 + (k + 1) + k,), mx.int32)
        return round_program.Round(
            mx.array(block_ids, mx.int32),
            rows,
            features,
            None,
            pkt,
            native=(exec_mode != "mlx_lazy"),
            route="native" if exec_mode != "mlx_lazy" else "mlx",
            committed=(exec_mode == "committed"),
            kept=(k + 1) if exec_mode == "committed" else None,
            emitted=(k + 1) if exec_mode == "committed" else None,
            rewind=(lambda _kept: None) if exec_mode == "committed" else None,
            draft_share=d_share,
            verify_forward_share=vf_share,
        )

    def fake_read_packet(packet, width, settle=None):
        if settle is not None:
            settle(packet)
        # Accept all k proposals + bonus token so each round emits k + 1 tokens.
        k = width - 1
        return fast_round.Decision(k, 0, [1] * width, [1.0] * k)

    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_INKLING_ROUND_PROGRAM", "on")
    monkeypatch.setenv("MTPLX_INKLING_ROUND_COMMIT", "1" if exec_mode == "committed" else "0")
    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", rt.prefill)
    monkeypatch.setattr(generation, "perf_counter", clock)
    monkeypatch.setattr(generation, "settle_owner_work", fake_settle)
    monkeypatch.setattr(generation, "decision_route", lambda *a, **k: (True, None))
    monkeypatch.setattr(generation, "checked_device_draw", lambda *a, **k: (1, True))
    monkeypatch.setattr(generation.round_program, "serve", fake_serve)
    monkeypatch.setattr(generation.fast_round, "read_packet", fake_read_packet)

    out = _run_generate(rt, [0], max_tokens=13, depth=3)
    stats = out.stats
    _assert_timing_invariants(stats, min_calls=3)

    # 3 rounds emitting 4 tokens each after first token (1 + 12 = 13).
    assert stats.verify_calls == 3
    # Exactly one settle call per round (from read_packet) — ZERO extra mx.eval sync barriers.
    assert len(eval_calls) == stats.verify_calls

    per_round_fused = 0.042 if exec_mode == "mlx_lazy" else 0.040
    per_round_post = 0.0 if exec_mode == "committed" else 0.002
    assert stats.draft_time_s == pytest.approx(3 * per_round_fused * 0.25, rel=1e-4)
    assert stats.verify_forward_time_s == pytest.approx(3 * per_round_fused * 0.70, rel=1e-4)
    assert stats.verify_time_s == pytest.approx(3 * (per_round_fused * 0.75 + per_round_post), rel=1e-4)
    assert stats.inkling_draft_sampling["draft_clock"] == "calibrated_fused_split"
    assert stats.inkling_draft_sampling["verify_clock"] == "calibrated_fused_split"


def test_mixed_copy_and_fused_rounds_and_host_path_timings(monkeypatch):
    # Case 1: Mixed fused K3 rounds + native synchronous copy rounds in one generation.
    clock = _FakeClock(200.0)
    rt = _StubRuntime(
        vocab=32,
        round_warm={"stage_shares": {3: {"draft": 0.20, "verify_forward": 0.75}}},
    )
    rt.clock = clock
    rt.verify_delay_s = 0.060  # native copy verify forward: 60ms
    rt.verify_route = "native"
    rt.commit_delay_s = 0.003

    served_count = 0

    def fake_serve(rt_, anchor, cache, k, **kwargs):
        nonlocal served_count
        served_count += 1
        clock.advance(0.030)  # 30ms fused K3 round
        block_ids = [(int(anchor) + i) % rt_.vocab for i in range(k + 1)]
        rows = mx.stack([rt_.scores(t) for t in block_ids])
        features = [mx.array([[block_ids]]).transpose(0, 2, 1)]
        pkt = mx.zeros((2 + (k + 1) + k,), mx.int32)
        return round_program.Round(
            mx.array(block_ids, mx.int32), rows, features, None, pkt,
            native=True, route="native", draft_share=0.20, verify_forward_share=0.75,
        )

    def fake_decide(rows, proposed, **kwargs):
        clock.advance(0.001)
        return mx.zeros((2 + int(rows.shape[0]) + int(proposed.shape[0]),), mx.int32)

    def fake_read_packet(packet, width, settle=None):
        if settle is not None:
            settle(packet)
        k = width - 1
        return fast_round.Decision(k, 0, list(range(1, width + 1)), [1.0] * k)

    # Prompt contains [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] so copy matches after initial K3 round.
    prompt = [20, 21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23]
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMIN", "2")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMAX", "6")
    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", rt.prefill)
    monkeypatch.setattr(generation, "perf_counter", clock)
    monkeypatch.setattr(generation, "settle_owner_work", lambda *a: None)
    monkeypatch.setattr(generation, "decision_route", lambda *a, **k: (True, None))
    monkeypatch.setattr(generation, "checked_device_draw", lambda *a, **k: (24, True))
    monkeypatch.setattr(generation.round_program, "serve", fake_serve)
    monkeypatch.setattr(generation.fast_round, "decide", fake_decide)
    monkeypatch.setattr(generation.fast_round, "read_packet", fake_read_packet)

    out_mixed = _run_generate(rt, prompt, max_tokens=15, depth=3)
    assert served_count >= 1
    assert out_mixed.stats.context_copy_rounds >= 1
    _assert_timing_invariants(out_mixed.stats, min_calls=2)

    # Case 2: Non-round_program lazy host path (MTPLX_INKLING_FAST=off, copy off).
    clock2 = _FakeClock(300.0)
    rt_host = _StubRuntime(vocab=16)
    rt_host.clock = clock2
    rt_host.draft_delay_s = 0.005
    rt_host.verify_delay_s = 0.004
    rt_host.commit_delay_s = 0.002

    def lazy_settle(*values):
        clock2.advance(0.015)

    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_INKLING_FAST", "off")
    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", rt_host.prefill)
    monkeypatch.setattr(generation, "perf_counter", clock2)
    monkeypatch.setattr(generation, "settle_owner_work", lazy_settle)
    monkeypatch.setattr(generation, "decision_route", lambda *a, **k: (False, "fast_path_off"))

    out_host = _run_generate(rt_host, [0], max_tokens=9, depth=3)
    _assert_timing_invariants(out_host.stats, min_calls=2)
    assert out_host.stats.verify_forward_time_s == pytest.approx(2 * (0.004 + 0.015), rel=1e-4)
    assert out_host.stats.draft_time_s == pytest.approx(2 * 0.005, rel=1e-4)


def test_route_tape_records_split_timings_on_fused_round(monkeypatch):
    clock = _FakeClock(400.0)
    events = []
    rt = _StubRuntime(round_warm={"stage_shares": {3: {"draft": 0.25, "verify_forward": 0.70}}})
    rt.clock = clock
    rt.commit_delay_s = 0.001

    def fake_serve(rt_, anchor, cache, k, **kwargs):
        clock.advance(0.040)
        block_ids = [(int(anchor) + i) % rt_.vocab for i in range(k + 1)]
        rows = mx.stack([rt_.scores(t) for t in block_ids])
        features = [mx.array([[block_ids]]).transpose(0, 2, 1)]
        pkt = mx.zeros((2 + (k + 1) + k,), mx.int32)
        return round_program.Round(
            mx.array(block_ids, mx.int32), rows, features, None, pkt,
            native=True, route="native", draft_share=0.25, verify_forward_share=0.70,
        )

    monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(route_tape, "_ROUTE_TAPE_SINK", events.append)
    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", rt.prefill)
    monkeypatch.setattr(generation, "perf_counter", clock)
    monkeypatch.setattr(generation, "settle_owner_work", lambda *a: None)
    monkeypatch.setattr(generation, "decision_route", lambda *a, **k: (True, None))
    monkeypatch.setattr(generation, "checked_device_draw", lambda *a, **k: (1, True))
    monkeypatch.setattr(generation.round_program, "serve", fake_serve)
    monkeypatch.setattr(
        generation.fast_round,
        "read_packet",
        lambda pkt, width, settle=None: (settle(pkt) if settle else None) or fast_round.Decision(width - 1, 0, [1] * width, [1.0] * (width - 1)),
    )

    out = _run_generate(rt, [0], max_tokens=5, depth=3)
    round_events = [e for e in events if e["name"] == "round"]
    assert len(round_events) == 1
    timing = round_events[0]["attrs"]["timing_s"]
    assert timing["proposal"] == pytest.approx(0.010, rel=1e-4)
    assert timing["verify_forward"] == pytest.approx(0.028, rel=1e-4)
    assert timing["verify_logits_eval"] == pytest.approx(0.002, rel=1e-4)
    assert timing["commit"] == pytest.approx(0.001, rel=1e-4)
    _assert_timing_invariants(out.stats, min_calls=1)
