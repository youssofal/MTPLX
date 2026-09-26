"""Bank-route v2 for context-copy block rounds (extended-window dispatch).

The copy-block verify dispatches at its NATIVE ladder length (T=9..33)
through ``CompiledVerifyBank.forward_ar_capture(extended_window=True)``.
Contract under test:

- extended dispatch is routing only: bit-equal outputs and cache state vs
  the eager reference forward at the same T (the byte-identical-trajectory
  precondition);
- the default (non-extended) path is byte-unchanged: length > max_verify_len
  still falls back with ``length_outside_bank``;
- the extended lane never changes the request's memory contract: a dense
  leaf whose grant cannot hold the window refuses (``block_window_capacity``)
  instead of growing;
- interleaved MTP + block dispatch traces once per length and never rebuilds
  the shadow (the v1 "~240 ms/call dispatch tax" was the first-call trace,
  not a recurring clone — probe receipts 2026-08-26);
- ``prewarm_extended_lengths`` traces without side effects on the live cache;
- ``generate_mtpk`` with MTPLX_CCOPY_BANK_ROUTE=1 emits a byte-identical
  stream vs route-off on the scripted copy lane, bank present or absent.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.gdn_capture import commit_captured_prefix
from mtplx.graphbank import (
    CompiledVerifyBank,
    TensorOffsetKVCache,
    promote_kv_cache_offsets,
)

from test_graphbank_compiled_verify import PRE_M5_BIT_PARITY, ToyHybridRuntime, _prefill


def _window(start: int, length: int) -> list[int]:
    return [(start + i) % ToyHybridRuntime.V for i in range(length)]


def test_extended_window_dispatches_compiled_above_max_verify_len():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, max_verify_len=6)
    cache = _prefill(rt, [0, 1, 2])

    # Default path regression: T=9 without the extended flag stays eager.
    bank.forward_ar_capture(mx.array([_window(0, 9)]), cache=cache)
    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_reasons"] == {"length_outside_bank": 1}

    logits, hidden, captures = bank.forward_ar_capture(
        mx.array([_window(0, 9)]), cache=cache, extended_window=True
    )
    assert bank.stats["compiled_calls"] == 1
    assert bank.stats["extended_calls"] == 1
    assert bank.stats["fallback_calls"] == 1  # only the pre-flag call
    assert int(logits.shape[1]) == 9 and int(hidden.shape[1]) == 9
    assert 0 in captures

    # Extended flag with an in-window length behaves exactly like a normal
    # call (no extended accounting).
    bank.forward_ar_capture(
        mx.array([_window(0, 3)]), cache=cache, extended_window=True
    )
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["extended_calls"] == 1


def test_extended_window_ceiling_env(monkeypatch):
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, max_verify_len=6)
    cache = _prefill(rt, [0, 1, 2])

    monkeypatch.setenv("MTPLX_CCOPY_BANK_MAX_LEN", "10")
    bank.forward_ar_capture(
        mx.array([_window(0, 11)]), cache=cache, extended_window=True
    )
    assert bank.stats["fallback_reasons"] == {"length_outside_bank": 1}
    bank.forward_ar_capture(
        mx.array([_window(0, 10)]), cache=cache, extended_window=True
    )
    assert bank.stats["compiled_calls"] == 1
    assert bank.stats["extended_calls"] == 1

    # Default ceiling covers the whole default ladder (blocks up to 32).
    monkeypatch.delenv("MTPLX_CCOPY_BANK_MAX_LEN", raising=False)
    bank.forward_ar_capture(
        mx.array([_window(0, 34)]), cache=cache, extended_window=True
    )
    assert bank.stats["fallback_reasons"]["length_outside_bank"] == 2


@PRE_M5_BIT_PARITY
def test_extended_block_bit_equal_vs_eager_reference():
    """Interleaved MTP + block windows: compiled session == eager session.

    Mirrors test_compiled_bit_equal_vs_eager_reference_with_accept_path with
    the block lane's shapes: native-length block windows (T=9, T=13), a
    position-1 reject (keep=1), a partial accept, and a full accept.
    """

    plan = [
        (_window(0, 3), 3),  # MTP round, full accept
        (_window(1, 9), 1),  # block round, rejected at position 1
        (_window(2, 3), 2),  # MTP round, partial accept
        (_window(3, 13), 7),  # block round, partial accept
        (_window(2, 9), 9),  # block round, full accept
    ]

    def run_session(compiled: bool):
        rt = ToyHybridRuntime(seed=7)
        cache = _prefill(rt, [0, 1, 2])
        bank = (
            CompiledVerifyBank(rt, max_verify_len=6) if compiled else None
        )
        if not compiled:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=13)
            assert promoted == 1 and failures == {}
        outputs = []
        for window, keep in plan:
            ids = mx.array([window])
            if compiled:
                logits, hidden, captures = bank.forward_ar_capture(
                    ids, cache=cache, extended_window=len(window) > 6
                )
            else:
                logits, hidden, captures = rt.forward_ar_capture(
                    ids, cache=cache, return_hidden=True
                )
            committed = commit_captured_prefix(
                cache, captures, keep_tokens=keep, verified_tokens=len(window)
            )
            assert committed is True
            offset = int(cache[1].size())
            outputs.append(
                {
                    "logits": np.array(logits),
                    "hidden": np.array(hidden),
                    "conv_states": np.array(captures[0]["conv_states"]),
                    "states": np.array(captures[0]["states"]),
                    "gdn_conv": np.array(cache[0].cache[0]),
                    "gdn_state": np.array(cache[0].cache[1]),
                    "offset": offset,
                    "kv_prefix": np.array(cache[1].cache[0][..., :offset, :]),
                    "v_prefix": np.array(cache[1].cache[1][..., :offset, :]),
                }
            )
        if compiled:
            assert bank.stats["compiled_calls"] == len(plan)
            assert bank.stats["fallback_calls"] == 0
            assert bank.stats["extended_calls"] == 3
        return outputs

    compiled_outputs = run_session(compiled=True)
    eager_outputs = run_session(compiled=False)

    for step, (got, want) in enumerate(zip(compiled_outputs, eager_outputs)):
        assert got["offset"] == want["offset"], f"step {step}"
        for name in (
            "logits",
            "hidden",
            "conv_states",
            "states",
            "gdn_conv",
            "gdn_state",
            "kv_prefix",
            "v_prefix",
        ):
            assert got[name].shape == want[name].shape, f"step {step}: {name}"
            assert np.array_equal(got[name], want[name]), f"step {step}: {name}"


def test_extended_dense_capacity_preflight_refuses_growth():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, max_verify_len=6)
    cache = _prefill(rt, [0, 1, 2])

    # Hand-promote with a tight grant: capacity 8, offset 3 -> a T=9 block
    # cannot fit without growing the granted leaf.
    entry = cache[1]
    assert isinstance(entry, KVCache)
    keys = mx.concatenate(
        [entry.keys, mx.zeros((1, 1, 8 - int(entry.keys.shape[2]), 4))], axis=2
    ) if int(entry.keys.shape[2]) < 8 else entry.keys[..., :8, :]
    values = mx.concatenate(
        [entry.values, mx.zeros((1, 1, 8 - int(entry.values.shape[2]), 4))], axis=2
    ) if int(entry.values.shape[2]) < 8 else entry.values[..., :8, :]
    adapter = TensorOffsetKVCache(keys, values, int(entry.offset), step=8)
    adapter._granted = True
    cache[1] = adapter
    capacity_before = int(adapter.keys.shape[2])

    logits, hidden, captures = bank.forward_ar_capture(
        mx.array([_window(0, 9)]), cache=cache, extended_window=True
    )
    # The oversized window triggers the standard growth-demotion transition
    # (the MTP top-up would trip it within max_verify_len tokens anyway):
    # stock containers take over, the eager fallback completes the verify by
    # growing natively, and the adapter's granted buffer never grew.
    assert bank.stats["fallback_reasons"] == {"block_window_capacity": 1}
    assert int(adapter.keys.shape[2]) == capacity_before  # grant never grew
    assert adapter.growth_after_grant is False
    assert bank.stats["growth_demotions"] == 1
    assert bank._growth_demoted is True
    assert isinstance(cache[1], KVCache)  # demoted to stock
    assert int(cache[1].offset) == 12  # 3 prefill + 9 verified rows
    assert int(logits.shape[1]) == 9 and int(hidden.shape[1]) == 9

    # The request stays eager from here, exactly like MTP grant exhaustion.
    bank.forward_ar_capture(mx.array([_window(0, 2)]), cache=cache)
    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_reasons"]["growth_budget_exhausted"] == 1


def test_interleaved_block_rounds_trace_once_and_keep_shadow():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, max_verify_len=6)
    cache = _prefill(rt, [0, 1, 2])

    def mtp():
        bank.forward_ar_capture(mx.array([_window(0, 4)]), cache=cache)

    def block(length: int):
        bank.forward_ar_capture(
            mx.array([_window(1, length)]), cache=cache, extended_window=True
        )

    mtp()
    shadow_id = id(bank._shadow)
    mtp()
    block(9)
    mtp()
    block(9)
    block(13)
    mtp()
    block(13)

    assert bank.stats["compiled_calls"] == 8
    assert bank.stats["fallback_calls"] == 0
    # One trace per distinct length; interleaving re-traces nothing.
    assert bank.stats["traces"] == 3
    assert id(bank._shadow) == shadow_id
    assert bank.stats["promoted"] == 1


def test_prewarm_extended_lengths_traces_without_side_effects():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, max_verify_len=6)
    cache = _prefill(rt, [0, 1, 2])
    # Promote via a first organic call so prewarm sees steady state.
    bank.forward_ar_capture(mx.array([_window(0, 4)]), cache=cache)

    before = [np.array(leaf) for leaf in bank._read_state_leaves(cache)]
    offset_before = int(cache[1].size())

    report = bank.prewarm_extended_lengths(cache, [3, 9, 13, 99])
    assert [item["m"] for item in report["lengths"]] == [9, 13]
    assert any(s.startswith("m3:") for s in report["skipped"])
    assert any(s.startswith("m99:") for s in report["skipped"])
    assert bank.stats["extended_prewarm"] is report

    after = [np.array(leaf) for leaf in bank._read_state_leaves(cache)]
    assert int(cache[1].size()) == offset_before
    assert len(after) == len(before)
    for pre, post in zip(before, after):
        assert np.array_equal(pre, post)

    # The organic extended call replays the prewarmed trace: no new trace.
    traces_before = bank.stats["traces"]
    bank.forward_ar_capture(
        mx.array([_window(0, 9)]), cache=cache, extended_window=True
    )
    assert bank.stats["traces"] == traces_before
    assert bank.stats["extended_calls"] == 1


# --- end-to-end: generate_mtpk byte-identity under the route env ------------

from test_context_copy_stats import _ScriptedModel, _mtpk  # noqa: E402


@pytest.mark.parametrize("compiled_verify", ["off", "on"])
def test_generate_mtpk_route_env_is_byte_identical(monkeypatch, compiled_verify):
    for name in (
        "MTPLX_CONTEXT_COPY",
        "MTPLX_CONTEXT_COPY_K",
        "MTPLX_CONTEXT_COPY_NGMIN",
        "MTPLX_CONTEXT_COPY_NGMAX",
        "MTPLX_CONTEXT_COPY_MINEXT",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
        "MTPLX_CCOPY_BANK_ROUTE",
        "MTPLX_CCOPY_BANK_PREWARM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    if compiled_verify == "on":
        # A real CompiledVerifyBank drives BOTH the MTP verifies and the
        # block rounds of the scripted lane end-to-end (the scripted model's
        # empty cache list compiles with an empty state spec).
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    else:
        monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)

    prompt = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    baseline = _mtpk(_ScriptedModel(8, lambda t: t + 1), prompt, max_tokens=80)
    monkeypatch.setenv("MTPLX_CCOPY_BANK_ROUTE", "1")
    routed = _mtpk(_ScriptedModel(8, lambda t: t + 1), prompt, max_tokens=80)

    assert list(routed.tokens) == list(baseline.tokens)
    for field in (
        "context_copy_rounds",
        "context_copy_probes",
        "context_copy_drafted_tokens",
        "context_copy_accepted_tokens",
        "context_copy_accepted_blocks",
        "context_copy_suspensions",
    ):
        assert getattr(routed.stats, field) == getattr(baseline.stats, field), field
