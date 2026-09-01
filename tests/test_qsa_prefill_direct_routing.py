"""Producer/consumer routing gates for the direct Steel QSA prefill lane.

The failure these exist to prevent is the quiet one: a native kernel that
imports, probes, reports supported, and is never called, because the producer
that emits its ``("flash_prefill", ids, valid)`` tuple was gated on Metal 4
TensorOps that M3 does not have. That produces a perfect null A/B result at
the cost of a machine-night.
"""

from __future__ import annotations

import pytest

import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.attention_context import attention_phase


@pytest.fixture()
def lanes(monkeypatch):
    """Control both fast consumers' availability independently."""

    state = {"nax": False, "direct": True, "metal": True}

    import mlx.core as mx

    import mtplx.kernels.qsa_indexer_select as select_module
    import mtplx.kernels.qsa_prefill_direct as direct_module

    monkeypatch.setattr(
        select_module, "qsa_indexer_select_nax_available", lambda: state["nax"]
    )
    monkeypatch.setattr(
        direct_module, "qsa_prefill_direct_module_ready", lambda: state["direct"]
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: state["metal"])
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.delenv("MTPLX_QSA_PREFILL", raising=False)
    monkeypatch.delenv("MTPLX_QSA_PREFILL_DIRECT", raising=False)
    return state


def test_m3_arms_the_lane_on_native_readiness_alone(lanes):
    """No NAX, native module ready: the producer must arm."""

    lanes["nax"] = False
    lanes["direct"] = True
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is True
    assert qwen4_exp._qsa_prefill_enabled() is True


def test_m4_m5_still_arm_on_nax_without_the_native_module(lanes):
    lanes["nax"] = True
    lanes["direct"] = False
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is True


def test_machine_with_neither_consumer_stays_off(lanes):
    """Otherwise the eager selector pays for a dense-mask reconstruction."""

    lanes["nax"] = False
    lanes["direct"] = False
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is False
    assert qwen4_exp._qsa_prefill_enabled() is False


def test_cpu_or_no_metal_never_arms(lanes):
    lanes["nax"] = True
    lanes["direct"] = True
    lanes["metal"] = False
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is False


def test_killing_the_direct_consumer_also_disarms_the_m3_producer(
    lanes, monkeypatch
):
    """Selecting blocks for a consumer that has been switched off is pure
    tax; the gate must follow the kill switch."""

    lanes["nax"] = False
    lanes["direct"] = True
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT", "0")
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is False
    assert qwen4_exp._qsa_prefill_enabled() is False


def test_master_kill_switch_outranks_native_readiness(lanes, monkeypatch):
    lanes["nax"] = False
    lanes["direct"] = True
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "0")
    assert qwen4_exp._qsa_prefill_enabled() is False


def test_explicit_master_on_arms_without_either_consumer(lanes, monkeypatch):
    lanes["nax"] = False
    lanes["direct"] = False
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    assert qwen4_exp._qsa_prefill_enabled() is True


def test_direct_consumer_crossover_is_independent_of_the_flash_one(
    lanes, monkeypatch
):
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "2")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT", "65536")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT", "2049")

    rows, total = 64, 4096
    with attention_phase("prefill"):
        assert qwen4_exp._qsa_large_prefill_enabled(rows, total) is True
        assert qwen4_exp._qsa_prefill_flash_attention_enabled(rows, total) is False
        assert qwen4_exp._qsa_prefill_direct_attention_enabled(rows, total) is True


def test_direct_consumer_respects_the_master_row_and_phase_gates(
    lanes, monkeypatch
):
    """MTP verify rows and decode must never reach this lane."""

    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "32")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT", "2049")

    with attention_phase("prefill"):
        assert qwen4_exp._qsa_prefill_direct_attention_enabled(8, 4096) is False
        assert qwen4_exp._qsa_prefill_direct_attention_enabled(64, 4096) is True
        # Chunk gated on its EARLIEST query, not its final T.
        assert qwen4_exp._qsa_prefill_direct_attention_enabled(64, 2100) is False
    with attention_phase("decode"):
        assert qwen4_exp._qsa_prefill_direct_attention_enabled(64, 4096) is False


def test_counters_distinguish_all_four_tiers():
    """The A/B receipt has to be able to say WHICH consumer ran.

    A source check, kept as a cheap extra: the behavioural proof that each
    tier bumps its own counter is the dispatch group at the bottom of this
    file.  The counters live in the tier chooser that Attention.__call__
    hands its four thunks to.
    """

    import ast
    from pathlib import Path

    path = Path(qwen4_exp.__file__)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    chooser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_qsa_prefill_dispatch_tier"
    )
    source = ast.get_source_segment(text, chooser) or ""
    for lane in ("flash_kernel", "direct_kernel", "gather_tier", "dense_fallback"):
        assert f'_qsa_prefill_count("{lane}")' in source
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Attention"
    )
    call = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert "_qsa_prefill_dispatch_tier(" in (ast.get_source_segment(text, call) or "")


def test_engagement_snapshot_reports_the_direct_lane():
    qwen4_exp._qsa_prefill_count("direct_kernel")
    assert qwen4_exp.qsa_prefill_engagement().get("direct_kernel", 0) >= 1


# --------------------------------------------------------------------------
# Tier dispatch: which consumer actually ran (Codex CHANGES_REQUIRED
# follow-up).  The gate tests above prove the producer arms; these prove the
# consumer choice, by driving the extracted chooser with callable fakes
# instead of reading the source.
# --------------------------------------------------------------------------


class _Tiers:
    """Records which tier bodies ran, and in which order."""

    def __init__(self, *, flash=False, direct=False, gather=True):
        import mlx.core as mx

        self.calls: list[str] = []
        self._flash = flash
        self._direct = direct
        self.gather_enabled = gather
        self._out = mx.zeros((1, 24, 4, 256), dtype=mx.bfloat16)

    def _make(self, name):
        def run():
            self.calls.append(name)
            return self._out

        return run

    def _pred(self, name, answer):
        def ask():
            self.calls.append(f"{name}?")
            return answer

        return ask

    def kwargs(self):
        return dict(
            flash_supported=self._pred("flash", self._flash),
            flash_call=self._make("flash"),
            direct_supported=self._pred("direct", self._direct),
            direct_call=self._make("direct"),
            gather_enabled=self.gather_enabled,
            gather_call=self._make("gather"),
        )


def _dispatch(tiers):
    """Run the chooser and return (output, engagement delta)."""

    before = qwen4_exp.qsa_prefill_engagement()
    out = qwen4_exp._qsa_prefill_dispatch_tier(**tiers.kwargs())
    after = qwen4_exp.qsa_prefill_engagement()
    delta = {
        lane: after.get(lane, 0) - before.get(lane, 0)
        for lane in set(before) | set(after)
        if after.get(lane, 0) != before.get(lane, 0)
    }
    return out, delta


def test_m3_dispatch_runs_only_the_direct_kernel():
    """No NAX flash consumer, direct ready: the direct body runs, nothing
    below it does, and the receipt says direct_kernel."""

    tiers = _Tiers(flash=False, direct=True, gather=True)
    out, delta = _dispatch(tiers)

    assert out is not None
    assert [c for c in tiers.calls if not c.endswith("?")] == ["direct"]
    assert delta == {"direct_kernel": 1}


def test_m4_m5_dispatch_prefers_flash_even_when_direct_is_ready():
    """M4/M5 must not have their answer changed by whether someone built the
    extension: the MPP kernel keeps first refusal."""

    tiers = _Tiers(flash=True, direct=True, gather=True)
    out, delta = _dispatch(tiers)

    assert [c for c in tiers.calls if not c.endswith("?")] == ["flash"]
    assert "direct?" not in tiers.calls, "the direct predicate must not even run"
    assert delta == {"flash_kernel": 1}
    assert out is not None


def test_missing_or_killed_direct_module_routes_to_gather():
    tiers = _Tiers(flash=False, direct=False, gather=True)
    out, delta = _dispatch(tiers)

    assert [c for c in tiers.calls if not c.endswith("?")] == ["gather"]
    assert delta == {"gather_tier": 1}
    assert out is not None


def test_gather_off_routes_to_the_dense_mask():
    """The chooser returns None to mean "rebuild the dense mask"; no tier
    body may have run."""

    tiers = _Tiers(flash=False, direct=False, gather=False)
    out, delta = _dispatch(tiers)

    assert out is None
    assert [c for c in tiers.calls if not c.endswith("?")] == []
    assert delta == {"dense_fallback": 1}


def test_no_lower_tier_runs_after_a_successful_direct_dispatch():
    """The failure this port cannot survive: a lane that dispatched and then
    quietly let a lower tier produce the numbers."""

    tiers = _Tiers(flash=False, direct=True, gather=True)
    _dispatch(tiers)

    assert "gather" not in tiers.calls
    assert tiers.calls.count("direct") == 1


def test_a_dispatched_tier_failure_is_not_retried_lower():
    """Fail-closed before dispatch, fail-loud after."""

    import mlx.core as mx

    tiers = _Tiers(flash=False, direct=True, gather=True)
    kwargs = tiers.kwargs()

    def _boom():
        tiers.calls.append("direct")
        raise RuntimeError("failed to load metallib mtplx_qsa_kernels")

    kwargs["direct_call"] = _boom
    with pytest.raises(RuntimeError, match="metallib"):
        qwen4_exp._qsa_prefill_dispatch_tier(**kwargs)
    assert "gather" not in tiers.calls
    assert mx is not None


# --------------------------------------------------------------------------
# The same choice driven through the REAL support predicates, so the tier
# order is proved against the production contracts and not only fakes.
# --------------------------------------------------------------------------


_ROWS = 64
_TOTAL = 4096  # 4096 // 4 == 1024 > 512: past the dense/sparse boundary
_SCALE = 0.0625


class _FakeDirectExt:
    BUILT_AGAINST_NANOBIND = "2.15.0"
    METAL_LIBRARY = "mtplx_qsa_kernels"

    def __init__(self):
        import mlx.core as mx

        self.BUILT_AGAINST_MLX = mx.__version__
        self.calls = 0

    def abi_probe(self, a):
        return a.size

    def qwen4_qsa_sparse_gqa_attention(self, q, k, v, selected, *args, **kwargs):
        import mlx.core as mx

        self.calls += 1
        return mx.zeros(tuple(q.shape), dtype=q.dtype)


@pytest.fixture()
def production_call(monkeypatch):
    """Production geometry plus a stubbed native extension."""

    import mlx.core as mx

    import mtplx.kernels.qsa_prefill_direct as direct_module

    fake = _FakeDirectExt()
    monkeypatch.setattr(direct_module, "_EXT", fake)
    monkeypatch.setattr(direct_module, "_on_metal_device", lambda: True)
    monkeypatch.setattr(
        direct_module, "_PIPELINE_STATE", direct_module._PIPELINE_UNPROVEN
    )
    monkeypatch.setattr(direct_module, "_PIPELINE_PROVEN_DTYPES", frozenset())
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "2")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT", "2049")
    monkeypatch.delenv("MTPLX_QSA_PREFILL_DIRECT", raising=False)

    pos_start = _TOTAL - _ROWS
    ids = mx.broadcast_to(
        mx.arange(512, dtype=mx.int32)[None, :], (_ROWS, 512)
    )
    tensors = dict(
        q=mx.zeros((1, 24, _ROWS, 256), dtype=mx.bfloat16),
        k=mx.zeros((1, 2, _TOTAL, 256), dtype=mx.bfloat16),
        v=mx.zeros((1, 2, _TOTAL, 256), dtype=mx.bfloat16),
        block_ids=mx.contiguous(ids),
        block_valid=mx.ones((_ROWS, 512), dtype=mx.bool_),
        pos_start=pos_start,
    )
    return fake, tensors


def _real_predicates(tensors, *, gather: bool):
    """The exact predicates Attention.__call__ hands the chooser."""

    from mtplx.kernels.qsa_prefill_direct import (
        qsa_prefill_direct,
        qsa_prefill_direct_supported,
    )
    from mtplx.kernels.qsa_prefill_flash import qsa_prefill_flash_supported

    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    ids = tensors["block_ids"]
    valid = tensors["block_valid"]
    pos_start = tensors["pos_start"]
    ran: list[str] = []

    def flash_supported():
        return bool(
            qwen4_exp._qsa_prefill_flash_attention_enabled(_ROWS, _TOTAL)
        ) and bool(
            qsa_prefill_flash_supported(
                q,
                k,
                v,
                ids,
                valid,
                pos_start=pos_start,
                total_tokens=_TOTAL,
                scale=_SCALE,
            )
        )

    def direct_supported():
        if not qwen4_exp._qsa_prefill_direct_attention_enabled(_ROWS, _TOTAL):
            return False
        return bool(
            qsa_prefill_direct_supported(
                q,
                k,
                v,
                ids,
                valid,
                pos_start=pos_start,
                total_tokens=_TOTAL,
                scale=_SCALE,
            )
        )

    def direct_call():
        ran.append("direct")
        return qsa_prefill_direct(
            q,
            k,
            v,
            ids,
            valid,
            pos_start=pos_start,
            total_tokens=_TOTAL,
            scale=_SCALE,
        )

    def flash_call():  # pragma: no cover - only reached if NAX exists here
        ran.append("flash")
        return q

    def gather_call():
        ran.append("gather")
        return q

    return ran, dict(
        flash_supported=flash_supported,
        flash_call=flash_call,
        direct_supported=direct_supported,
        direct_call=direct_call,
        gather_enabled=gather,
        gather_call=gather_call,
    )


def test_real_predicates_route_a_production_call_to_the_direct_kernel(
    production_call, monkeypatch
):
    """End to end through the real support checks: on a machine with no NAX
    flash kernel, the loaded direct module takes the call and nothing
    below it is asked."""

    fake, tensors = production_call
    with attention_phase("prefill"):
        ran, kwargs = _real_predicates(tensors, gather=True)
        assert kwargs["flash_supported"]() is False, "no NAX consumer here"
        before = qwen4_exp.qsa_prefill_engagement().get("direct_kernel", 0)
        out = qwen4_exp._qsa_prefill_dispatch_tier(**kwargs)
        after = qwen4_exp.qsa_prefill_engagement().get("direct_kernel", 0)

    assert out is not None
    assert ran == ["direct"]
    assert fake.calls == 1
    assert after - before == 1


def test_real_predicates_fall_to_gather_when_the_direct_lane_is_killed(
    production_call, monkeypatch
):
    fake, tensors = production_call
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT", "0")
    with attention_phase("prefill"):
        ran, kwargs = _real_predicates(tensors, gather=True)
        assert kwargs["direct_supported"]() is False
        out = qwen4_exp._qsa_prefill_dispatch_tier(**kwargs)

    assert ran == ["gather"]
    assert fake.calls == 0
    assert out is not None


def test_real_predicates_fall_to_dense_when_gather_is_also_off(
    production_call, monkeypatch
):
    fake, tensors = production_call
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT", "0")
    with attention_phase("prefill"):
        ran, kwargs = _real_predicates(tensors, gather=False)
        out = qwen4_exp._qsa_prefill_dispatch_tier(**kwargs)

    assert out is None
    assert ran == []
    assert fake.calls == 0


def test_real_predicates_refuse_the_direct_lane_after_a_failed_proof(
    production_call, monkeypatch
):
    """The blocker: a failed pipeline proof must move the call to gather
    instead of re-dispatching a kernel that cannot run."""

    import mtplx.kernels.qsa_prefill_direct as direct_module

    fake, tensors = production_call
    monkeypatch.setattr(
        direct_module, "_PIPELINE_STATE", direct_module._PIPELINE_FAILED
    )
    monkeypatch.setattr(direct_module, "_PIPELINE_PROVEN_DTYPES", frozenset())
    with attention_phase("prefill"):
        ran, kwargs = _real_predicates(tensors, gather=True)
        assert kwargs["direct_supported"]() is False
        out = qwen4_exp._qsa_prefill_dispatch_tier(**kwargs)

    assert ran == ["gather"]
    assert fake.calls == 0
    assert out is not None
