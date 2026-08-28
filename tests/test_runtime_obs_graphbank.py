"""Runtime observability: prewarm one-shot truth (F6a) + permanent-eager
visibility (F23a) on the compiled-verify bank.

No model, no GPU kernels: the bank's ladder internals are monkeypatched so
the one-shot / bucket-dedupe / completion logic is exercised with stub
compiles, and the flip paths run on a tiny fake runtime whose forward
returns constant arrays.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.graphbank as graphbank
from mtplx.graphbank import CompiledVerifyBank


class _MiniRuntime:
    """Unquantized fake: passes the bits gate, forward returns constants."""

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        del cache, hidden_variant, capture_backend
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        logits = mx.zeros((B, S, 4), dtype=mx.float32)
        hidden = mx.zeros((B, S, 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden, {}
        return logits, {}


def _quantized_runtime(bits: int) -> SimpleNamespace:
    q_proj = SimpleNamespace(bits=bits)
    layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))
    inner = SimpleNamespace(layers=[layer])
    model = SimpleNamespace(model=inner)
    runtime = SimpleNamespace(model=model)
    runtime.forward_ar_capture = _MiniRuntime().forward_ar_capture
    return runtime


@pytest.fixture()
def _fresh_module_state(monkeypatch):
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", False)
    monkeypatch.setattr(graphbank, "_PREWARMED_BUCKETS", set())
    monkeypatch.setattr(
        graphbank,
        "prewarm_status",
        {"done": False, "buckets": [], "walks": 0, "last_report": None},
    )
    monkeypatch.setattr(
        graphbank,
        "compiled_verify_status",
        {
            "mode": None,
            "permanent_eager": False,
            "reason": None,
            "flipped_at": None,
            "flip_count": 0,
            "transient_exception_count": 0,
        },
    )
    monkeypatch.setattr(graphbank, "_PERMANENT_EAGER_LOGGED", set())
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_PREWARM", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_FORCE", raising=False)


# ---------------------------------------------------------------------------
# F6a: the boot warmup must not spend or clamp the prewarm one-shot.
# ---------------------------------------------------------------------------


def _dispatch(bank: CompiledVerifyBank) -> None:
    # cache=None short-circuits to an eager fallback right after the
    # prewarm trigger — exactly the path a boot-warmup-shaped probe takes.
    bank.forward_ar_capture(mx.array([[1, 2]]), cache=None)


def test_clamped_walk_does_not_spend_oneshot(monkeypatch, _fresh_module_state):
    walks: list[dict] = []
    reports = [
        # Boot warmup: paged ladder clamped by tiny capacity -> incomplete.
        {
            "buckets": [{"bucket": 1024, "s": 0.1}],
            "skipped": [],
            "already": [],
            "elapsed_s": 0.1,
            "complete": False,
        },
        # Warmup ladder rung with real capacity: reaches the ceiling.
        {
            "buckets": [{"bucket": 8192, "s": 0.4}],
            "skipped": [],
            "already": [1024],
            "elapsed_s": 0.4,
            "complete": True,
        },
    ]

    def fake_walk(self, cache, input_ids, hidden_variant=None, max_context=None):
        walks.append({"bank": id(self)})
        return reports[len(walks) - 1]

    monkeypatch.setattr(CompiledVerifyBank, "prewarm_ladder", fake_walk)
    rt = _MiniRuntime()

    bank1 = CompiledVerifyBank(rt)
    _dispatch(bank1)
    assert len(walks) == 1
    assert bank1.stats["prewarm"]["complete"] is False
    assert graphbank._PREWARM_DONE is False  # clamped walk left it unspent
    assert graphbank.prewarm_status["walks"] == 1
    assert graphbank.prewarm_status["done"] is False

    bank2 = CompiledVerifyBank(rt)
    _dispatch(bank2)
    assert len(walks) == 2  # retried and extended
    assert graphbank._PREWARM_DONE is True
    assert graphbank.prewarm_status["done"] is True
    assert graphbank.prewarm_status["last_report"]["complete"] is True

    bank3 = CompiledVerifyBank(rt)
    _dispatch(bank3)
    assert len(walks) == 2  # complete: never walked again
    assert "prewarm" not in bank3.stats


def test_prewarm_env_off_keeps_flag_untouched(monkeypatch, _fresh_module_state):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    called: list[int] = []
    monkeypatch.setattr(
        CompiledVerifyBank,
        "prewarm_ladder",
        lambda self, *a, **k: called.append(1) or {},
    )
    bank = CompiledVerifyBank(_MiniRuntime())
    _dispatch(bank)
    assert called == []
    assert graphbank._PREWARM_DONE is False


def test_walk_error_is_recorded_not_fatal(monkeypatch, _fresh_module_state):
    def broken_walk(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(CompiledVerifyBank, "prewarm_ladder", broken_walk)
    bank = CompiledVerifyBank(_MiniRuntime())
    _dispatch(bank)  # must not raise
    report = bank.stats["prewarm"]
    assert report["skipped"] == ["walk_error:RuntimeError"]
    assert report["complete"] is False
    assert graphbank._PREWARM_DONE is False


class _FakePagedEntry:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)


def _ladder_bank(monkeypatch, capacity: int, natural: int):
    """Bank with ladder internals stubbed: real walk logic, fake compiles."""
    bank = CompiledVerifyBank(_MiniRuntime())
    compiled_buckets: list[int] = []
    bank._spec = [(0, graphbank.VERIFY_SPEC_KIND_FULL_ATTN, 3)]
    monkeypatch.setattr(bank, "_fallback_reason", lambda *a, **k: None)
    monkeypatch.setattr(bank, "_resolve_bucket", lambda cache, length: natural)
    monkeypatch.setattr(bank, "_ensure_shadow", lambda cache: None)
    monkeypatch.setattr(
        bank, "_read_state_leaves", lambda cache: [mx.array(0.0)]
    )
    monkeypatch.setattr(bank, "_paged_ineligibility", lambda *a: None)
    monkeypatch.setattr(bank, "_apply_bucket", lambda cache, bucket: None)

    def fake_shared(key, length, hidden_variant):
        compiled_buckets.append(int(key[2]))
        return lambda *args: (mx.array(0.0),)

    monkeypatch.setattr(bank, "_shared_or_new_verify_step", fake_shared)
    cache = [_FakePagedEntry(capacity)]
    return bank, cache, compiled_buckets


def test_ladder_extends_and_dedupes_buckets(monkeypatch, _fresh_module_state):
    input_ids = mx.array([[1, 2]])  # length 2 -> ceiling pow2(6144+2+512)=8192

    # Boot-warmup-sized cache: min_capacity 2048 clamps the walk.
    bank1, cache1, compiled1 = _ladder_bank(monkeypatch, capacity=2048, natural=512)
    report1 = bank1.prewarm_ladder(cache1, input_ids)
    assert compiled1 == [512, 1024, 2048]
    assert report1["complete"] is False  # clamped below the 8192 ceiling
    assert report1["already"] == []
    assert [b["bucket"] for b in report1["buckets"]] == [512, 1024, 2048]

    # Same runtime id space is irrelevant here: new bank, bigger capacity.
    bank2, cache2, compiled2 = _ladder_bank(monkeypatch, capacity=16384, natural=512)
    # Reuse bank1's runtime identity for the process-global bucket keys.
    bank2.runtime = bank1.runtime
    report2 = bank2.prewarm_ladder(cache2, input_ids)
    assert compiled2 == [4096, 8192]  # 512..2048 skipped as already warmed
    assert report2["already"] == [512, 1024, 2048]
    assert report2["complete"] is True  # reached the router ceiling

    # Third walk: nothing pending, no compiles, still complete.
    bank3, cache3, compiled3 = _ladder_bank(monkeypatch, capacity=16384, natural=512)
    bank3.runtime = bank1.runtime
    report3 = bank3.prewarm_ladder(cache3, input_ids)
    assert compiled3 == []
    assert report3["complete"] is True
    assert report3["already"] == [512, 1024, 2048, 4096, 8192]


def test_ladder_skips_walk_above_router(monkeypatch, _fresh_module_state):
    input_ids = mx.array([[1, 2]])
    bank, cache, compiled = _ladder_bank(
        monkeypatch, capacity=262144, natural=16384
    )
    report = bank.prewarm_ladder(cache, input_ids)
    assert compiled == []  # natural 16384 > ceiling 8192: nothing compiled
    assert report["skipped"] == ["context_above_router"]
    assert report["complete"] is False


def test_dense_cache_walk_is_complete_noop(_fresh_module_state, monkeypatch):
    bank = CompiledVerifyBank(_MiniRuntime())
    monkeypatch.setattr(bank, "_fallback_reason", lambda *a, **k: None)
    monkeypatch.setattr(bank, "_resolve_bucket", lambda cache, length: 0)
    report = bank.prewarm_ladder([], mx.array([[1, 2]]))
    assert report["skipped"] == ["no_paged_entries"]
    assert report["complete"] is True  # dense: designed no-op, spend the shot


# ---------------------------------------------------------------------------
# F23a: permanent-eager flips are recorded and logged once, not silent.
# ---------------------------------------------------------------------------


def test_bits_gate_flip_records_reason_and_logs_once(
    capsys, _fresh_module_state
):
    bank = CompiledVerifyBank(_quantized_runtime(bits=6))
    assert bank.permanent_eager is True
    assert bank.permanent_eager_reason == "quant_bits_gate:bits=6"
    status = graphbank.compiled_verify_status
    assert status["permanent_eager"] is True
    assert status["reason"] == "quant_bits_gate:bits=6"
    assert status["flip_count"] == 1
    assert status["mode"] == "on"
    assert status["flipped_at"] is not None
    assert bank.to_dict()["permanent_eager_reason"] == "quant_bits_gate:bits=6"
    out = capsys.readouterr().out
    assert out.count("compiled-verify permanent-eager") == 1

    # Per-request re-construction must not spam the log or the count.
    CompiledVerifyBank(_quantized_runtime(bits=6))
    assert graphbank.compiled_verify_status["flip_count"] == 1
    assert "permanent-eager" not in capsys.readouterr().out


def test_supported_bits_do_not_flip(_fresh_module_state):
    bank = CompiledVerifyBank(_quantized_runtime(bits=4))
    assert bank.permanent_eager is False
    assert graphbank.compiled_verify_status["permanent_eager"] is False
    assert graphbank.compiled_verify_status["flip_count"] == 0


def test_exception_streak_flips_with_reason_and_counts(
    monkeypatch, capsys, _fresh_module_state
):
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)  # isolate from F6
    bank = CompiledVerifyBank(_MiniRuntime())
    monkeypatch.setattr(bank, "_fallback_reason", lambda *a, **k: None)

    def broken_resolve(cache, length):
        raise RuntimeError("probe")

    monkeypatch.setattr(bank, "_resolve_bucket", broken_resolve)
    cache: list = []
    for _ in range(3):
        bank.forward_ar_capture(mx.array([[1, 2]]), cache=cache)

    assert bank.permanent_eager is True
    assert bank.permanent_eager_reason == "exception_streak:RuntimeError"
    status = graphbank.compiled_verify_status
    assert status["permanent_eager"] is True
    assert status["reason"] == "exception_streak:RuntimeError"
    assert status["flip_count"] == 1
    assert status["transient_exception_count"] == 3
    assert status["last_exception"] == "RuntimeError: probe"
    assert bank.stats["fallback_reasons"]["exception:RuntimeError"] == 3
    out = capsys.readouterr().out
    assert out.count("compiled-verify exception: RuntimeError: probe") == 1
    assert out.count("compiled-verify permanent-eager") == 1


def test_two_transient_exceptions_only_count(
    monkeypatch, capsys, _fresh_module_state
):
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    bank = CompiledVerifyBank(_MiniRuntime())
    monkeypatch.setattr(bank, "_fallback_reason", lambda *a, **k: None)
    calls = {"n": 0}

    def flaky_resolve(cache, length):
        calls["n"] += 1
        raise RuntimeError("probe")

    monkeypatch.setattr(bank, "_resolve_bucket", flaky_resolve)
    for _ in range(2):
        bank.forward_ar_capture(mx.array([[1, 2]]), cache=[])

    assert bank.permanent_eager is False
    status = graphbank.compiled_verify_status
    assert status["permanent_eager"] is False
    assert status["flip_count"] == 0
    assert status["transient_exception_count"] == 2
    assert "permanent-eager" not in capsys.readouterr().out
