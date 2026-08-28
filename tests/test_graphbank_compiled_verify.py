"""CompiledVerifyBank tests on a tiny synthetic hybrid model.

The toy runtime has one GDN-like ArraysCache layer and one full-attention
layer with deliberately small dims.  Its forward callable exercises the same
cache-mutation pattern as ``mtplx.gdn_capture.forward_with_gdn_capture``:
python-level assignment of fresh arrays into the GDN slots, KV writes via
``update_and_fetch`` on the attention entry, and an offset-sensitive readout,
returning ``(logits, hidden, captures)`` in the standard capture layout.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import TensorOffsetVllmMetalPagedKVCache, VllmMetalPagedKVCache
from mtplx.gdn_capture import commit_captured_prefix
from mtplx.graphbank import (
    CompiledVerifyBank,
    CompiledVerifyParityError,
    TensorOffsetKVCache,
    _compiled_verify_route_fingerprint,
    build_verify_state_spec,
    compare_verify_outputs,
    compiled_verify_mode,
    promote_kv_cache_offsets,
)


def test_shared_verify_route_key_tracks_qwen38_route_fingerprint() -> None:
    runtime = SimpleNamespace(
        qwen38_route=SimpleNamespace(fingerprint="row8-control")
    )

    assert _compiled_verify_route_fingerprint(runtime) == "row8-control"

    runtime.qwen38_route = SimpleNamespace(fingerprint="row8-gdn-s2")
    assert _compiled_verify_route_fingerprint(runtime) == "row8-gdn-s2"


def _arrays_cache_cls() -> type:
    """Resolve ``ArraysCache`` lazily at use time, exactly like production.

    ``mtplx.arrays_cache_patch.install_arrays_cache_fix`` (triggered by any
    earlier test importing ``mtplx.a3b_mtp_batch``) replaces the class bound
    in ``mlx_lm.models.cache``; ``mtplx.graphbank`` resolves it per call
    (``build_verify_state_spec``). A module-scope ``from ... import
    ArraysCache`` here froze the pre-patch class identity at collection
    time, so instances this file constructed failed graphbank's isinstance
    checks whenever a patch-triggering file ran first (23 false failures).
    Per-use resolution keeps this file green both standalone and after any
    patch-triggering file, without touching the process-global patch state.
    """

    import mlx_lm.models.cache as cache_module

    return cache_module.ArraysCache


class ToyHybridRuntime:
    """One GDN-like layer + one attention layer over tiny f32 tensors."""

    D = 4  # model dim
    K = 3  # conv taps
    V = 5  # vocab

    def __init__(self, seed: int = 7) -> None:
        mx.random.seed(seed)
        scale = 0.3
        self.embed = mx.random.normal((self.V, self.D)).astype(mx.float32)
        self.w_conv = scale * mx.random.normal((self.K * self.D, self.D)).astype(mx.float32)
        self.w_q = scale * mx.random.normal((self.D, self.D)).astype(mx.float32)
        self.w_out = scale * mx.random.normal((self.D, self.V)).astype(mx.float32)
        self.calls: list[str] = []

    def make_cache(self) -> list:
        gdn = _arrays_cache_cls()(2)
        gdn[0] = mx.zeros((1, self.K, self.D), dtype=mx.float32)
        gdn[1] = mx.zeros((1, 1, self.D, self.D), dtype=mx.float32)
        return [gdn, KVCache()]

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        del hidden_variant, capture_backend
        self.calls.append("forward")
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        gdn_entry, attn_entry = cache
        h = self.embed[input_ids]  # (B, S, D)

        # GDN-like layer: sequential shift-in conv + recurrent matrix state.
        conv = gdn_entry.cache[0]
        state = gdn_entry.cache[1]
        conv_steps = []
        state_steps = []
        outs = []
        for t in range(S):
            x_t = h[:, t : t + 1, :]
            conv = mx.concatenate([conv[:, 1:, :], x_t], axis=1)
            mixed = mx.tanh(conv.reshape(B, -1) @ self.w_conv)  # (B, D)
            state = mx.tanh(
                state + mixed[:, None, :, None] * mixed[:, None, None, :]
            )  # (B, 1, D, D)
            conv_steps.append(conv)
            state_steps.append(state)
            outs.append(mx.sum(state, axis=-1))  # (B, 1, D)
        # The poisoning pattern: python-level slot assignment inside forward.
        gdn_entry[0] = conv
        gdn_entry[1] = state
        gdn_entry.advance(S)
        h = h + mx.concatenate(outs, axis=1)

        # Attention-like layer: KV write via update_and_fetch, offset-masked
        # linear readout (offset-sensitive on purpose).
        keys = (0.5 * h)[:, None, :, :]  # (B, 1, S, D)
        values = (-0.25 * h)[:, None, :, :]
        k_buf, v_buf = attn_entry.update_and_fetch(keys, values)
        offset = attn_entry.offset  # int (stock) or mx.array (adapter)
        capacity = int(k_buf.shape[2])
        q = h @ self.w_q  # (B, S, D)
        scores = q @ mx.swapaxes(k_buf[:, 0, :, :], 1, 2)  # (B, S, T)
        pos = mx.arange(capacity)
        limit = offset - S + 1 + mx.arange(S)
        mask = (pos[None, :] < limit[:, None]).astype(mx.float32)  # (S, T)
        attn = (scores * mask[None, :, :]) @ v_buf[:, 0, :, :]  # (B, S, D)
        h = h + attn

        hidden = h
        logits = h @ self.w_out
        captures = {
            0: {
                "conv_states": mx.stack(conv_steps, axis=1),  # (B, S, K, D)
                "states": mx.stack(state_steps, axis=1),  # (B, S, 1, D, D)
            }
        }
        if return_hidden:
            return logits, hidden, captures
        return logits, captures


def _prefill(rt: ToyHybridRuntime, tokens: list[int]) -> list:
    cache = rt.make_cache()
    rt.forward_ar_capture(mx.array([tokens]), cache=cache, return_hidden=True)
    return cache


def _leaf_arrays(cache) -> list[mx.array]:
    leaves: list[mx.array] = []
    for entry in cache:
        if entry is None:
            continue
        if isinstance(entry, (TensorOffsetKVCache, TensorOffsetVllmMetalPagedKVCache)):
            leaves.extend(entry.cache[:3])
        elif isinstance(entry, _arrays_cache_cls()):
            leaves.extend(item for item in entry.cache if item is not None)
        elif isinstance(entry, KVCache):
            leaves.extend(item for item in (entry.keys, entry.values) if item is not None)
    return leaves


VERIFY_WINDOWS = [[3, 4, 0], [1, 2, 3], [4, 4, 1], [0, 2, 4]]


def test_compiled_verify_mode_env(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    assert compiled_verify_mode() == "off"
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "0")
    assert compiled_verify_mode() == "off"
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    assert compiled_verify_mode() == "on"
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    assert compiled_verify_mode() == "parity"
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    assert compiled_verify_mode() == "parity2"
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", " PARITY2 ")
    assert compiled_verify_mode() == "parity2"


def test_prewarm_trigger_fires_once_per_process_and_is_env_gated(monkeypatch):
    import mtplx.graphbank as graphbank_module

    # Env off: the trigger must not consume the one-shot flag, so enabling
    # the env later in the same process still gets a prewarm.
    monkeypatch.setattr(graphbank_module, "_PREWARM_DONE", False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert "prewarm" not in bank.stats
    assert graphbank_module._PREWARM_DONE is False

    # Default (unset) = enabled: first dispatch prewarns, exactly once per
    # process — a second bank in the same process must not re-walk.
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_PREWARM", raising=False)
    bank2 = CompiledVerifyBank(rt)
    cache2 = _prefill(rt, [0, 1, 2])
    bank2.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache2)
    report = bank2.stats["prewarm"]
    # Dense toy adapters have no paged capacity, so the ladder is skipped —
    # the trigger, report shape, and one-shot semantics are what's under test.
    assert report["skipped"] == ["no_paged_entries"]
    assert isinstance(report["elapsed_s"], float)
    assert graphbank_module._PREWARM_DONE is True
    bank3 = CompiledVerifyBank(rt)
    cache3 = _prefill(rt, [0, 1, 2])
    bank3.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache3)
    assert "prewarm" not in bank3.stats


def test_prewarm_ladder_is_harmless_before_organic_calls(monkeypatch):
    # Calling prewarm_ladder directly must not perturb subsequent organic
    # dispatch: same compiled/fallback accounting, state still advances.
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    report = bank.prewarm_ladder(cache, mx.array([VERIFY_WINDOWS[0]]))
    assert report["buckets"] == []  # dense toy: nothing paged to walk
    assert bank.stats["compiled_calls"] == 0
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[1]]), cache=cache)
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["fallback_calls"] == 0
    assert int(cache[1].size()) == 9


def test_build_verify_state_spec_orders_layers():
    rt = ToyHybridRuntime()
    cache = _prefill(rt, [0, 1, 2])
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=4)
    assert promoted == 1 and failures == {}

    spec, reason = build_verify_state_spec(cache)

    assert reason is None
    assert spec == [(0, "gdn", 2), (1, "fa", 3)]

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    paged.update_without_fetch(
        mx.ones((1, 1, 2, 1), dtype=mx.float32),
        mx.ones((1, 1, 2, 1), dtype=mx.float32),
    )
    adapter = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    spec, reason = build_verify_state_spec([None, adapter, cache[0]])
    assert reason is None
    assert spec == [(1, "fa", 3), (2, "gdn", 2)]

    spec, reason = build_verify_state_spec([object()])
    assert spec is None
    assert reason == "unsupported_container:object"


def test_real_entries_unchanged_until_mirror_commit():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    # Warm call promotes entries and compiles once.
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert bank.stats["compiled_calls"] == 1

    before = [np.array(leaf) for leaf in bank._read_state_leaves(cache)]
    seen: dict[str, list[np.ndarray]] = {}
    original = bank._mirror_commit

    def spying_commit(target_cache, state_out):
        seen["at_commit"] = [
            np.array(leaf) for leaf in bank._read_state_leaves(target_cache)
        ]
        original(target_cache, state_out)

    bank._mirror_commit = spying_commit
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[1]]), cache=cache)

    # At mirror-commit time the real leaves were still bit-identical to the
    # pre-call snapshot: the compiled step ran purely on the shadow cache.
    assert len(seen["at_commit"]) == len(before)
    for pre, at_commit in zip(before, seen["at_commit"]):
        assert np.array_equal(pre, at_commit)
    # And the commit itself moved the offset forward.
    assert int(cache[1].size()) == 9


def test_two_consecutive_calls_trace_once():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[1]]), cache=cache)

    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["fallback_calls"] == 0
    assert bank.stats["traces"] == 1  # second call replayed the cached trace
    # A different verify length compiles a separate entry.
    bank.forward_ar_capture(mx.array([[2, 3]]), cache=cache)
    assert bank.stats["traces"] == 2
    assert bank.stats["compiled_calls"] == 3


def test_no_tracer_leaves_in_real_cache_after_call():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])

    for window in VERIFY_WINDOWS[:3]:
        logits, hidden, captures = bank.forward_ar_capture(
            mx.array([window]), cache=cache
        )
    assert bank.stats["compiled_calls"] == 3

    # Poison regression: a leaked tracer raises on any eval — including a
    # zero-cost zero-slice — with "cannot eval an array without a primitive".
    def eval_zero_slice(leaf):
        mx.eval(leaf[:0] if leaf.ndim else leaf)

    for leaf in _leaf_arrays(cache):
        eval_zero_slice(leaf)
    for capture in captures.values():
        for value in capture.values():
            if isinstance(value, mx.array):
                eval_zero_slice(value)
    mx.eval(logits, hidden)
    # The next compiled call (fresh trace via new length) must also survive.
    logits2, _hidden2, _captures2 = bank.forward_ar_capture(
        mx.array([[1]]), cache=cache
    )
    mx.eval(logits2)
    for leaf in _leaf_arrays(cache):
        eval_zero_slice(leaf)


def test_compiled_bit_equal_vs_eager_reference_with_accept_path():
    keep_plan = [3, 2, 1, 3]  # accepted prefix per verify window

    def run_session(compiled: bool):
        rt = ToyHybridRuntime(seed=7)
        cache = _prefill(rt, [0, 1, 2])
        bank = CompiledVerifyBank(rt) if compiled else None
        if not compiled:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=3)
            assert promoted == 1 and failures == {}
        outputs = []
        for window, keep in zip(VERIFY_WINDOWS, keep_plan):
            ids = mx.array([window])
            if compiled:
                logits, hidden, captures = bank.forward_ar_capture(ids, cache=cache)
            else:
                logits, hidden, captures = rt.forward_ar_capture(
                    ids, cache=cache, return_hidden=True
                )
            committed = commit_captured_prefix(
                cache,
                captures,
                keep_tokens=keep,
                verified_tokens=len(window),
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
            assert bank.stats["compiled_calls"] == len(VERIFY_WINDOWS)
            assert bank.stats["fallback_calls"] == 0
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


def test_reject_path_trim_takes_offset_only_branch():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    entry = cache[1]
    assert isinstance(entry, TensorOffsetKVCache)
    # Mirror-commit cleared the rollback window.
    assert entry.rollback_state == [None, None, None]
    assert entry.size() == 6

    entry.trim(2)  # full-window reject of two tokens
    assert entry.size() == 4
    # Next verify writes at the trimmed offset.
    bank.forward_ar_capture(mx.array([[1, 3]]), cache=cache)
    assert entry.size() == 6


class NullRuntime:
    """Eager stub that records calls and returns sentinels untouched."""

    def __init__(self) -> None:
        self.calls = 0

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        self.calls += 1
        return "eager-logits", "eager-hidden", {}


def test_fallback_matrix_reasons(monkeypatch):
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])

    # Length above the bank ceiling.
    bank.forward_ar_capture(mx.array([[0, 1, 2, 3, 4, 0, 1]]), cache=cache)
    assert bank.stats["fallback_reasons"]["length_outside_bank"] == 1

    # Owned-state env wrappers force eager.
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    bank.forward_ar_capture(mx.array([[0, 1]]), cache=cache)
    assert bank.stats["fallback_reasons"]["owned_attn_kv_env"] == 1
    monkeypatch.delenv("MTPLX_OWNED_ATTN_KV")
    monkeypatch.setenv("MTPLX_OWNED_RECURRENT_STATE", "persistent_eval")
    bank.forward_ar_capture(mx.array([[0, 1]]), cache=cache)
    assert bank.stats["fallback_reasons"]["owned_recurrent_state_env"] == 1
    monkeypatch.delenv("MTPLX_OWNED_RECURRENT_STATE")

    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_calls"] == 3
    # The real forward ran for every fallback (prefill + 3).
    assert len(rt.calls) == 4

    # Batched inputs force eager (B=1-only toy: assert via the null runtime).
    null_bank = CompiledVerifyBank(NullRuntime())
    null_bank.forward_ar_capture(mx.array([[0, 1], [1, 2]]), cache=[])
    assert null_bank.stats["fallback_reasons"]["batch_size"] == 1


def test_fallback_reasons_for_unsupported_cache_containers():
    null_rt = NullRuntime()
    bank = CompiledVerifyBank(null_rt)

    # Promotion failure keeps python offsets out of the compiled path.
    class RotatingStub:
        offset = 4
        _idx = 2
        keys = mx.zeros((1, 1, 8, 4))
        values = mx.zeros((1, 1, 8, 4))

    rotating_cache = [RotatingStub()]
    result = bank.forward_ar_capture(mx.array([[0, 1]]), cache=rotating_cache)
    assert result == ("eager-logits", "eager-hidden", {})
    assert (
        bank.stats["fallback_reasons"][
            "promotion_failure:rotating_or_indexed_cache"
        ]
        == 1
    )

    # Unsupported cache container.
    class WeirdCache:
        offset = mx.array(3, dtype=mx.int32)

    weird = [WeirdCache()]
    bank.forward_ar_capture(mx.array([[0, 1]]), cache=weird)
    assert bank.stats["fallback_reasons"]["unsupported_container:WeirdCache"] == 1

    # No cache at all.
    bank.forward_ar_capture(mx.array([[0, 1]]), cache=None)
    assert bank.stats["fallback_reasons"]["no_cache"] == 1

    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_calls"] == 3
    assert null_rt.calls == 3


def test_quantized_paged_entries_fall_back(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    from mtplx.kv_quant import PagedKVQuantConfig

    monkeypatch.delenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", raising=False)
    null_rt = NullRuntime()
    bank = CompiledVerifyBank(null_rt)
    quantized = VllmMetalPagedKVCache(
        block_size=4,
        num_blocks=4,
        kv_quant_config=PagedKVQuantConfig("q8"),
    )
    quantized.update_without_fetch(
        mx.random.normal((1, 2, 5, 16), dtype=mx.float16),
        mx.random.normal((1, 2, 5, 16), dtype=mx.float16),
    )
    cache = [quantized]

    bank.forward_ar_capture(mx.array([[0, 1]]), cache=cache)

    assert bank.stats["fallback_reasons"]["quantized_paged_kv"] == 1
    assert cache[0] is quantized  # never promoted, never densified


def test_post_restore_warmup_defers_first_round_then_promotes(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", "1")
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, restored_tokens=4096)
    cache = _prefill(rt, [0, 1, 2])

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert bank.stats["fallback_reasons"]["post_restore_warmup"] == 1
    assert bank.stats["compiled_calls"] == 0
    # The deferred round must leave the cache unpromoted: the whole point is
    # skipping the O(context) ensure_capacity copy on the TTFT round.
    assert bank.stats["promoted"] == 0

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[1]]), cache=cache)
    assert bank.stats["compiled_calls"] == 1
    assert bank.stats["fallback_calls"] == 1


def test_post_restore_warmup_needs_min_restored_tokens(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", "1")
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, restored_tokens=512)  # below the 2048 floor
    cache = _prefill(rt, [0, 1, 2])

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert "post_restore_warmup" not in bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 1


def test_post_restore_warmup_env_rounds_and_kill_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", "2")
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, restored_tokens=100_000)
    cache = _prefill(rt, [0, 1, 2])
    for window in VERIFY_WINDOWS[:2]:
        bank.forward_ar_capture(mx.array([window]), cache=cache)
    assert bank.stats["fallback_reasons"]["post_restore_warmup"] == 2
    assert bank.stats["compiled_calls"] == 0
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[2]]), cache=cache)
    assert bank.stats["compiled_calls"] == 1

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", "0")
    rt2 = ToyHybridRuntime()
    bank2 = CompiledVerifyBank(rt2, restored_tokens=100_000)
    cache2 = _prefill(rt2, [0, 1, 2])
    bank2.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache2)
    assert "post_restore_warmup" not in bank2.stats["fallback_reasons"]
    assert bank2.stats["compiled_calls"] == 1

    # Default (no env) is OFF: the deferral is opt-in pending a 16k+ restore
    # receipt (see _post_restore_eager_rounds docstring).
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", raising=False)
    assert CompiledVerifyBank(rt2, restored_tokens=100_000)._post_restore_eager_remaining == 0


def test_post_restore_warmup_disabled_under_parity_modes(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", "1")
    rt = ToyHybridRuntime()
    # Parity harnesses must keep full compiled coverage from round 1.
    assert (
        CompiledVerifyBank(
            rt, parity=True, restored_tokens=100_000
        )._post_restore_eager_remaining
        == 0
    )
    assert (
        CompiledVerifyBank(
            rt, parity2=True, restored_tokens=100_000
        )._post_restore_eager_remaining
        == 0
    )


def test_permanent_eager_after_three_repeated_failures():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)

    class ExplodingRuntime(ToyHybridRuntime):
        def forward_ar_capture(self, input_ids, cache=None, **kwargs):
            if bank._shadow is not None and cache is bank._shadow:
                raise RuntimeError("boom inside compiled trace")
            return super().forward_ar_capture(input_ids, cache=cache, **kwargs)

    exploding = ExplodingRuntime(seed=7)
    bank.runtime = exploding
    cache = _prefill(exploding, [0, 1, 2])

    for _ in range(3):
        logits, hidden, captures = bank.forward_ar_capture(
            mx.array([[1, 2]]), cache=cache
        )
        assert logits is not None
    assert bank.stats["fallback_reasons"]["exception:RuntimeError"] == 3
    assert bank.permanent_eager is True

    bank.forward_ar_capture(mx.array([[1, 2]]), cache=cache)
    assert bank.stats["fallback_reasons"]["permanent_eager"] == 1
    assert bank.stats["compiled_calls"] == 0


def test_demote_restores_stock_containers_and_counts():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert isinstance(cache[1], TensorOffsetKVCache)

    count = bank.demote(cache)

    assert count == 1
    assert bank.stats["demotions"] == 1
    assert type(cache[1]) is KVCache
    assert isinstance(cache[1].offset, int)
    assert cache[1].offset == 6
    assert isinstance(cache[0], _arrays_cache_cls())  # GDN entries untouched
    # Compiled closures were dropped with the shadow; the next call rebuilds.
    bank.forward_ar_capture(mx.array([[1, 2]]), cache=cache)
    assert bank.stats["compiled_calls"] == 2
    assert cache[1].size() == 8


def test_to_dict_exposes_stats_and_buckets():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)

    data = bank.to_dict()

    assert data["calls"] == 1
    assert data["compiled_calls"] == 1
    assert data["mode"] == "on"
    assert data["max_verify_len"] == 6
    assert data["permanent_eager"] is False
    assert data["promoted"] == 1
    assert data["compiled_entry_count"] == 1
    assert data["compiled_keys"] == ["m3:default:b0"]  # dense toy: no paged bucket
    assert isinstance(data["fallback_reasons"], dict)
    assert isinstance(data["buckets"], dict)


class _ExactKVRuntime:
    V = 5

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden=False,
        hidden_variant=None,
        capture_backend=None,
    ):
        del hidden_variant, capture_backend
        hidden = input_ids.astype(mx.float32)[..., None]
        kv = hidden[:, None, :, :]
        cache[0].update_and_fetch(kv, kv)
        logits = mx.concatenate((hidden, hidden + 1.0), axis=-1)
        if return_hidden:
            return logits, hidden, {}
        return logits, {}


def test_request_budget_below_ceiling_reserves_exact_budget(monkeypatch):
    """A small explicit budget tightens the grant below the env ceiling."""

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    rt = _ExactKVRuntime()
    cache = [KVCache()]
    rt.forward_ar_capture(mx.array([[0, 1, 2]]), cache=cache)
    bank = CompiledVerifyBank(rt, request_max_tokens=200, parity=True)
    assert bank.growth_reserve_tokens == 206  # budget + one speculative window

    for token_index in range(200):
        bank.forward_ar_capture(
            mx.array([[token_index % rt.V]]),
            cache=cache,
            return_hidden=True,
        )

    stats = bank.to_dict()
    assert stats["request_max_tokens"] == 200
    assert stats["speculative_headroom"] == bank.max_verify_len == 6
    assert stats["compiled_calls"] == 200
    assert stats["fallback_calls"] == 0
    assert stats["growth_demotions"] == 0
    assert stats["parity_failures"] == 0
    assert isinstance(cache[0], TensorOffsetKVCache)
    assert cache[0].size() == 203
    # Grant = 3 prompt + 206 reserve, rounded to one 256-token step — not
    # the 512-token env default, and nowhere near the request ceiling bug.
    assert int(cache[0].keys.shape[2]) == 256


def test_unbounded_request_budget_clamps_to_env_ceiling_and_demotes(monkeypatch):
    """Server-default budgets (whole context window) must not size the grant.

    2.4.0 regression receipt: max_tokens defaulted to ~262k and every
    request materialized a multi-gigabyte KV reserve at first promotion
    (44 GB peak, ~13 tok/s turn opens). The grant clamps to the env
    ceiling; a request that outgrows it demotes to eager for the request
    remainder (the 2026-07-03 contract) and stays exact.
    """

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    rt = _ExactKVRuntime()
    cache = [KVCache()]
    rt.forward_ar_capture(mx.array([[0, 1, 2]]), cache=cache)
    bank = CompiledVerifyBank(rt, request_max_tokens=262_133, parity=True)
    assert bank.growth_reserve_tokens == 512  # env ceiling, not the budget

    for token_index in range(1024):
        bank.forward_ar_capture(
            mx.array([[token_index % rt.V]]),
            cache=cache,
            return_hidden=True,
        )

    stats = bank.to_dict()
    assert stats["request_max_tokens"] == 262_133
    assert stats["calls"] == 1024
    assert stats["growth_demotions"] == 1
    assert stats["growth_handoff_materializations"] == 1
    assert stats["growth_handoff_state_leaves"] == 3
    assert stats["growth_handoff_materialize_time_s"] >= 0.0
    assert stats["fallback_reasons"].get("growth_budget_exhausted", 0) > 0
    assert stats["compiled_calls"] + stats["fallback_calls"] == 1024
    assert stats["parity_failures"] == 0
    # Demoted back to stock entries; the eager path finished the request.
    assert type(cache[0]) is KVCache
    assert cache[0].offset == 1027


def test_growth_handoff_settles_hybrid_state_and_releases_compiled_refs(monkeypatch):
    """Growth demotion must hand eager mode evaluated, independently owned state."""

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "6")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "none")
    rt = ToyHybridRuntime()
    cache = rt.make_cache()
    cache[1].step = 4
    rt.forward_ar_capture(mx.array([[0, 1, 2]]), cache=cache, return_hidden=True)
    bank = CompiledVerifyBank(rt, request_max_tokens=262_133)

    real_eval = mx.eval
    evaluated_batch_sizes: list[int] = []

    def recording_eval(*leaves):
        evaluated_batch_sizes.append(len(leaves))
        return real_eval(*leaves)

    monkeypatch.setattr(mx, "eval", recording_eval)
    for token_index in range(16):
        logits, hidden, _ = bank.forward_ar_capture(
            mx.array([[token_index % rt.V]]),
            cache=cache,
            return_hidden=True,
        )
        mx.eval(logits, hidden)

    stats = bank.to_dict()
    assert stats["growth_demotions"] == 1
    assert stats["growth_handoff_materializations"] == 1
    # Two recurrent leaves plus dense K, V, and tensor offset.
    assert stats["growth_handoff_state_leaves"] == 5
    assert 5 in evaluated_batch_sizes
    assert type(cache[1]) is KVCache
    assert cache[1].offset == 19
    assert bank._held_state_refs == []
    assert bank._shadow is None
    assert bank._spec is None
    assert bank._compiled == {}


def test_env_reserve_raises_ceiling_for_known_budget_runs(monkeypatch):
    """A known 1024-token request must not hit the 512-token cliff when the
    operator widens the ceiling — the original PR #174 win, now env-gated."""

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "2048")
    rt = _ExactKVRuntime()
    cache = [KVCache()]
    rt.forward_ar_capture(mx.array([[0, 1, 2]]), cache=cache)
    bank = CompiledVerifyBank(rt, request_max_tokens=1024, parity=True)
    assert bank.growth_reserve_tokens == 1030  # min(budget + window, env)

    for token_index in range(1024):
        bank.forward_ar_capture(
            mx.array([[token_index % rt.V]]),
            cache=cache,
            return_hidden=True,
        )

    stats = bank.to_dict()
    assert stats["compiled_calls"] == 1024
    assert stats["fallback_calls"] == 0
    assert stats["growth_demotions"] == 0
    assert stats["parity_failures"] == 0
    assert isinstance(cache[0], TensorOffsetKVCache)
    assert cache[0].size() == 1027
    assert int(cache[0].keys.shape[2]) == 1280
    assert int(cache[0].keys.shape[2]) % int(cache[0].step) == 0


def test_parity_mode_passes_on_toy_model_and_commits_eager_state():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, parity=True)
    cache = _prefill(rt, [0, 1, 2])

    logits, hidden, captures = bank.forward_ar_capture(
        mx.array([VERIFY_WINDOWS[0]]), cache=cache
    )

    assert bank.stats["parity_checks"] == 1
    assert bank.stats["parity_failures"] == 0
    assert bank.stats["compiled_calls"] == 1
    mx.eval(logits, hidden)
    # Eager leg is authoritative: it ran on the real cache with rollback set.
    assert cache[1].size() == 6
    assert cache[1].rollback_state[0] is not None
    assert 0 in captures


def test_parity_mode_aborts_on_mismatch():
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt, parity=True)

    class SkewedRuntime(ToyHybridRuntime):
        def forward_ar_capture(self, input_ids, cache=None, **kwargs):
            result = super().forward_ar_capture(input_ids, cache=cache, **kwargs)
            if bank._shadow is not None and cache is not bank._shadow:
                logits, hidden, captures = result
                return logits + 1e-3, hidden, captures
            return result

    skewed = SkewedRuntime(seed=7)
    bank.runtime = skewed
    cache = _prefill(skewed, [0, 1, 2])

    with pytest.raises(CompiledVerifyParityError) as excinfo:
        bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)

    assert bank.stats["parity_failures"] == 1
    assert any(line.startswith("logits") for line in excinfo.value.report)


# -- parity mode #2: compiled authoritative, eager clone tracks ----------------


def test_parity_and_parity2_are_mutually_exclusive():
    with pytest.raises(ValueError):
        CompiledVerifyBank(ToyHybridRuntime(), parity=True, parity2=True)


def test_parity2_commits_compiled_state_and_matches_compiled_only_run():
    """Real entries under parity2 advance bit-identically to a compiled-only
    run through the full accept path — the eager clone leg never perturbs the
    live stream, and the per-call clone rebuild survives structural trims."""
    keep_plan = [3, 2, 1, 3]

    def run(parity2: bool):
        rt = ToyHybridRuntime(seed=7)
        bank = CompiledVerifyBank(rt, parity2=parity2)
        cache = _prefill(rt, [0, 1, 2])
        outputs = []
        for window, keep in zip(VERIFY_WINDOWS, keep_plan):
            logits, hidden, captures = bank.forward_ar_capture(
                mx.array([window]), cache=cache
            )
            committed = commit_captured_prefix(
                cache,
                captures,
                keep_tokens=keep,
                verified_tokens=len(window),
            )
            assert committed is True
            outputs.append(
                {
                    "logits": np.array(logits),
                    "hidden": np.array(hidden),
                    "conv_states": np.array(captures[0]["conv_states"]),
                    "states": np.array(captures[0]["states"]),
                }
            )
        state = [np.array(leaf) for leaf in bank._read_state_leaves(cache)]
        return rt, bank, cache, outputs, state

    rt2, bank2, cache2, outputs2, state2 = run(parity2=True)
    rt0, bank0, cache0, outputs0, state0 = run(parity2=False)

    for step, (got, want) in enumerate(zip(outputs2, outputs0)):
        for name in ("logits", "hidden", "conv_states", "states"):
            assert got[name].shape == want[name].shape, f"step {step}: {name}"
            assert np.array_equal(got[name], want[name]), f"step {step}: {name}"
    assert len(state2) == len(state0)
    for got, want in zip(state2, state0):
        assert np.array_equal(got, want)

    assert bank2.stats["compiled_calls"] == len(VERIFY_WINDOWS)
    assert bank2.stats["parity2_calls"] == len(VERIFY_WINDOWS)
    assert bank2.stats["parity2_divergent_calls"] == 0
    assert bank2.stats["parity2_first_divergence"] is None
    assert bank2.stats["parity_checks"] == 0
    assert bank2.to_dict()["mode"] == "parity2"
    # Mirror-commit semantics (unlike parity#1's eager commit): rollback is
    # cleared, so reject trims take the offset-only branch.
    assert cache2[1].rollback_state == [None, None, None]
    assert int(cache2[1].size()) == int(cache0[1].size())
    # The eager reference leg really ran once per verify call, on the clone.
    assert len(rt2.calls) == len(rt0.calls) + len(VERIFY_WINDOWS)


def _make_parity2_skewed_bank():
    """Bank whose eager CLONE leg (not shadow, not real cache) skews logits."""
    rt = ToyHybridRuntime(seed=7)
    bank = CompiledVerifyBank(rt, parity2=True)
    armed = {"real": None}

    class SkewedRuntime(ToyHybridRuntime):
        def forward_ar_capture(self, input_ids, cache=None, **kwargs):
            result = super().forward_ar_capture(input_ids, cache=cache, **kwargs)
            if (
                armed["real"] is not None
                and cache is not bank._shadow
                and cache is not armed["real"]
            ):
                logits, hidden, captures = result
                return logits + 1e-3, hidden, captures
            return result

    skewed = SkewedRuntime(seed=7)
    bank.runtime = skewed
    cache = _prefill(skewed, [0, 1, 2])
    armed["real"] = cache
    return bank, cache


def test_parity2_divergence_counts_and_logs_without_raising(capsys):
    bank, cache = _make_parity2_skewed_bank()

    logits, hidden, captures = bank.forward_ar_capture(
        mx.array([VERIFY_WINDOWS[0]]), cache=cache
    )  # must NOT raise, unlike parity#1
    mx.eval(logits, hidden)

    assert bank.stats["parity2_calls"] == 1
    assert bank.stats["parity2_divergent_calls"] == 1
    first = bank.stats["parity2_first_divergence"]
    assert first is not None
    assert first["call"] == 1
    assert first["artifact"] == "logits"
    assert first["leaf"] == "logits"
    assert first["mismatched_leaves"] == 1  # only logits were skewed
    assert first["max_abs_diff"] == pytest.approx(1e-3, rel=0.3)
    assert first["context"] == 6  # post-commit offset: 3 prefill + 3 verified
    # Stream continued compiled-authoritative: the real cache advanced.
    assert int(cache[1].size()) == 6
    assert 0 in captures

    # A second divergent call streams on and keeps the first record.
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[1]]), cache=cache)
    assert bank.stats["parity2_divergent_calls"] == 2
    assert bank.stats["parity2_first_divergence"]["call"] == 1
    assert int(cache[1].size()) == 9

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("[parity2]")]
    assert len(lines) == 2
    assert "divergence call=1" in lines[0]
    assert "artifact=logits" in lines[0]
    assert "leaf=logits" in lines[0]
    assert "mismatched_leaves=1" in lines[0]
    assert bank.to_dict()["parity2_first_divergence"]["call"] == 1


def test_parity2_state_leaf_divergence_reports_full_leaf_identity(capsys):
    """A committed-state divergence must name the exact leaf — including the
    colon-bearing 'state[idx:kind].n' identity — not a truncated prefix."""
    rt = ToyHybridRuntime(seed=7)
    bank = CompiledVerifyBank(rt, parity2=True)
    armed = {"real": None}

    class StateSkewedRuntime(ToyHybridRuntime):
        def forward_ar_capture(self, input_ids, cache=None, **kwargs):
            result = super().forward_ar_capture(input_ids, cache=cache, **kwargs)
            if (
                armed["real"] is not None
                and cache is not bank._shadow
                and cache is not armed["real"]
            ):
                # Perturb the clone's committed GDN recurrent state only.
                cache[0][1] = cache[0].cache[1] + 1e-3
            return result

    skewed = StateSkewedRuntime(seed=7)
    bank.runtime = skewed
    cache = _prefill(skewed, [0, 1, 2])
    armed["real"] = cache

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)

    assert bank.stats["parity2_divergent_calls"] == 1
    first = bank.stats["parity2_first_divergence"]
    assert first["artifact"] == "state"
    assert first["leaf"] == "state[0:gdn].1"  # layer idx, kind, leaf index
    assert first["mismatched_leaves"] == 1
    assert first["max_abs_diff"] == pytest.approx(1e-3, rel=0.3)
    out = capsys.readouterr().out
    assert "leaf=state[0:gdn].1" in out
    # Real cache still advanced on the unperturbed compiled state.
    assert int(cache[1].size()) == 6


def test_parity2_divergence_logging_caps_at_ten_calls(capsys):
    bank, cache = _make_parity2_skewed_bank()

    for _ in range(12):
        bank.forward_ar_capture(mx.array([[1, 2]]), cache=cache)

    assert bank.stats["parity2_calls"] == 12
    assert bank.stats["parity2_divergent_calls"] == 12
    out = capsys.readouterr().out
    divergence_lines = [
        line for line in out.splitlines()
        if line.startswith("[parity2] divergence call=")
    ]
    cap_lines = [line for line in out.splitlines() if "log cap reached" in line]
    assert len(divergence_lines) == 10
    assert len(cap_lines) == 1


# -- comparator unit tests (step 4) -------------------------------------------


def test_compare_verify_outputs_equal_is_empty():
    a = {
        "logits": np.ones((1, 3, 5), dtype=np.float32),
        "state[1:fa].2": np.array(7, dtype=np.int32),
        "capture[0].gdn_meta": {"conv_dim": 4},
    }
    b = {
        "logits": np.ones((1, 3, 5), dtype=np.float32),
        "state[1:fa].2": np.array(7, dtype=np.int32),
        "capture[0].gdn_meta": {"conv_dim": 4},
    }
    assert compare_verify_outputs(a, b) == []


def test_compare_verify_outputs_detects_value_shape_dtype_and_missing():
    base = np.zeros((2, 2), dtype=np.float32)
    reference = {
        "logits": base,
        "hidden": base,
        "state[0:gdn].0": base,
        "only_ref": base,
    }
    candidate = {
        "logits": base + 1e-6,
        "hidden": np.zeros((2, 3), dtype=np.float32),
        "state[0:gdn].0": base.astype(np.float16),
        "only_cand": base,
    }

    report = compare_verify_outputs(reference, candidate)

    joined = "\n".join(report)
    assert "logits: value mismatch" in joined
    assert "hidden: shape mismatch" in joined
    assert "state[0:gdn].0: dtype mismatch" in joined
    assert "only_ref: missing from candidate output" in joined
    assert "only_cand: missing from reference output" in joined


def test_compare_verify_outputs_mx_arrays_and_none_leaves():
    a = {"x": mx.array([1.0, 2.0]), "n": None}
    b = {"x": mx.array([1.0, 2.5]), "n": None}
    report = compare_verify_outputs(a, b)
    assert len(report) == 1 and report[0].startswith("x: value mismatch")
    assert compare_verify_outputs({"x": mx.array([1.0])}, {"x": mx.array([1.0])}) == []
    report = compare_verify_outputs({"n": None}, {"n": mx.array([1.0])})
    assert report and report[0].startswith("n: one side is None")


def test_compare_verify_outputs_truncates_report():
    reference = {f"k{i}": np.zeros(1) for i in range(40)}
    candidate = {f"k{i}": np.ones(1) for i in range(40)}
    report = compare_verify_outputs(reference, candidate, max_report_lines=5)
    assert len(report) == 6
    assert report[-1] == "... report truncated ..."


# -- generation wiring (step 3) ------------------------------------------------


def _tiny_mtpk_runtime(*, mtp_token: int = 1):
    """Stub runtime in the style of tests/test_generation_sustained.py."""
    from pathlib import Path
    from types import SimpleNamespace

    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime

    class TinyTokenizer:
        def decode(self, tokens, **_kwargs):
            return "".join(str(int(token)) for token in tokens)

    class TinyMTPModel:
        def __init__(self):
            self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
            self.capture_calls: list[int] = []
            self.mtp_token = int(mtp_token)

        def make_cache(self):
            return []

        def make_mtp_cache(self):
            return []

        def _logits(self, length: int):
            logits = mx.zeros((1, length, 4), dtype=mx.float32)
            return logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)

        def __call__(
            self,
            input_ids,
            *,
            cache=None,
            return_hidden: bool = False,
            hidden_variant: str | None = None,
            **_kwargs,
        ):
            length = int(input_ids.shape[1])
            hidden = mx.zeros((1, length, 2), dtype=mx.float32)
            if return_hidden:
                return self._logits(length), hidden
            return self._logits(length)

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            *,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str | None = None,
            position_offset=None,
        ):
            length = int(next_token_ids.shape[1])
            hidden = mx.zeros((1, length, 2), dtype=mx.float32)
            logits = mx.zeros((1, length, 4), dtype=mx.float32)
            logits = logits + mx.eye(4, dtype=mx.float32)[self.mtp_token]
            if return_hidden:
                return logits, hidden
            return logits

        def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
            return hidden_states

    model = TinyMTPModel()
    rt = MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
    )

    def capture_stub(
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        length = int(input_ids.shape[1])
        model.capture_calls.append(length)
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if return_hidden:
            return model._logits(length), hidden, {}
        return model._logits(length), {}

    rt.forward_ar_capture = capture_stub
    return rt, model


def _run_tiny_mtpk(
    max_tokens: int = 5,
    *,
    verify_strategy: str = "capture_commit",
    mtp_token: int = 1,
):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    rt, model = _tiny_mtpk_runtime(mtp_token=mtp_token)
    out = generate_mtpk(
        rt,
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy=verify_strategy,
        stop_token_ids=set(),
    )
    return out, model


def test_generation_flag_off_instantiates_no_bank(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)

    out, model = _run_tiny_mtpk()

    assert len(out.tokens) == 5
    assert out.tokens == [1, 1, 1, 1, 1]
    assert out.stats.graphbank == {}  # no graphbank, no compiled_verify field
    assert "graphbank" not in out.stats.events[0]
    assert out.stats.verify_calls >= 1


def test_generation_flag_on_attaches_stats_and_matches_flag_off(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    baseline, _ = _run_tiny_mtpk()

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    out, model = _run_tiny_mtpk()

    assert out.tokens == baseline.tokens
    assert out.stats.generated_tokens == baseline.stats.generated_tokens
    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["mode"] == "on"
    assert bank_stats["calls"] == out.stats.verify_calls
    assert bank_stats["compiled_calls"] >= 1
    assert bank_stats["fallback_calls"] == 0
    assert bank_stats["permanent_eager"] is False
    assert all(
        "compiled_verify" not in event.get("graphbank", {})
        for event in out.stats.events
    )
    # No adapters existed in the empty stub cache, so nothing to demote.
    assert bank_stats["demotions"] == 0


def test_generation_target_prefix_compile_is_separately_default_off(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)

    out, _ = _run_tiny_mtpk(verify_strategy="target_prefix")

    assert out.stats.graphbank == {}


def test_generation_flag_on_compiles_target_prefix_without_changing_tokens(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)
    baseline, _ = _run_tiny_mtpk(verify_strategy="target_prefix")

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    out, _ = _run_tiny_mtpk(verify_strategy="target_prefix")

    assert out.tokens == baseline.tokens
    assert out.stats.generated_tokens == baseline.stats.generated_tokens
    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["mode"] == "on"
    assert bank_stats["calls"] == out.stats.verify_calls
    assert bank_stats["compiled_calls"] >= 1
    assert bank_stats["fallback_calls"] == 0


def test_generation_passes_known_output_budget_to_compiled_bank(monkeypatch):
    import mtplx.generation as generation

    real_bank = generation.CompiledVerifyBank
    seen: list[dict] = []

    def recording_bank(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_bank(*args, **kwargs)

    monkeypatch.setattr(generation, "CompiledVerifyBank", recording_bank)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")

    out, _ = _run_tiny_mtpk(
        max_tokens=17,
        verify_strategy="target_prefix",
    )

    assert len(out.tokens) == 17
    assert seen and seen[0]["request_max_tokens"] == 17


def test_generation_target_prefix_compiles_rejection_correction_forward(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_TARGET_PREFIX", raising=False)
    baseline, _ = _run_tiny_mtpk(
        verify_strategy="target_prefix",
        mtp_token=2,
    )

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    out, _ = _run_tiny_mtpk(
        verify_strategy="target_prefix",
        mtp_token=2,
    )

    assert out.tokens == baseline.tokens
    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["calls"] > out.stats.verify_calls
    assert bank_stats["compiled_calls"] == bank_stats["calls"]
    assert bank_stats["fallback_calls"] == 0


def _run_exact_a3b_k1_schedule(
    monkeypatch,
    *,
    target_tokens: list[int],
    max_tokens: int,
    **generation_kwargs,
):
    import mtplx.generation as generation
    from mtplx.a3b_compiled_target_prefix import A3BCompiledTargetPrefixFactory
    from mtplx.gdn_capture import A3BGDNPostconvFactory
    from mtplx.sampling import SamplerConfig

    rt, model = _tiny_mtpk_runtime(mtp_token=1)
    rt.a3b_compiled_target_prefix_factory = (
        A3BCompiledTargetPrefixFactory(
            layer_types=tuple(
                "linear_attention" if index % 4 != 3 else "full_attention"
                for index in range(40)
            ),
            gdn_layers=30,
            full_attention_layers=10,
            hidden_size=2048,
            quantization="affine_q4_group64",
            gdn_postconv=A3BGDNPostconvFactory(
                m1_implementations=tuple(lambda *args: args for _ in range(30)),
                m2_implementations=tuple(lambda *args: args for _ in range(30)),
            ),
        )
    )
    schedule: list[tuple] = []
    primary_states: list[object] = []
    history_appends: list[tuple[list[int], np.ndarray]] = []
    exact_installed = False

    class SpyRoute:
        def __init__(self, cache):
            self.cache = cache

        def _verify(self, input_ids, kind, state_in=None):
            cycle = len(primary_states)
            target_token = int(target_tokens[min(cycle, len(target_tokens) - 1)])
            primary_state = object()
            primary_states.append(primary_state)
            entry = (
                kind,
                tuple(int(token) for token in np.asarray(input_ids).reshape(-1)),
            )
            if state_in is not None:
                entry = entry + (state_in,)
            schedule.append(entry)
            logits = mx.zeros((1, 2, 4), dtype=mx.float32)
            logits = logits + mx.eye(4, dtype=mx.float32)[target_token]
            hidden = mx.stack(
                (
                    mx.full((2,), 10 + cycle, dtype=mx.float32),
                    mx.full((2,), 20 + cycle, dtype=mx.float32),
                ),
                axis=0,
            )[None, ...]
            return logits, hidden, primary_state

        def verify_m2(self, input_ids):
            return self._verify(input_ids, "m2")

        def verify_m2_rebased(self, input_ids, primary_state):
            # Deferred-correction fold: the rejecting cycle's post-primary
            # state is the graph input; no repair_m1 dispatch exists.
            return self._verify(input_ids, "m2r", state_in=primary_state)

        def repair_m1(self, input_ids, primary_state):
            raise AssertionError(
                "repair_m1 must not be dispatched under the deferred-correction fold"
            )

        def final_report(self, *, verify_calls, repair_calls):
            total = int(verify_calls) + int(repair_calls)
            return {
                "calls": total,
                "compiled_calls": total,
                "m2_calls": int(verify_calls),
                "m1_calls": int(repair_calls),
                "fallback_calls": 0,
                "growth_demotions": 0,
            }

        def demote(self):
            return 0

    def install_spy_route(_rt, cache, **_kwargs):
        nonlocal exact_installed
        exact_installed = True
        return SpyRoute(cache)

    monkeypatch.setattr(
        generation,
        "install_a3b_k1_target_prefix_route",
        install_spy_route,
    )

    def forbidden(name):
        def fail(*_args, **_kwargs):
            raise AssertionError(f"exact A3B route must not call {name}")

        return fail

    monkeypatch.setattr(generation, "snapshot_untrimmable_cache", forbidden("snapshot"))
    monkeypatch.setattr(generation, "rollback_after_verify", forbidden("rollback"))
    monkeypatch.setattr(
        generation,
        "_sample_draft_from_logits",
        forbidden("host draft sampler"),
    )
    monkeypatch.setattr(
        generation,
        "trim_verified_window_to_prefix",
        forbidden("trim"),
    )
    real_forward_ar = rt.forward_ar

    def forward_ar_only_before_install(*args, **kwargs):
        if exact_installed:
            raise AssertionError("exact A3B route must not call generic target forward")
        return real_forward_ar(*args, **kwargs)

    monkeypatch.setattr(rt, "forward_ar", forward_ar_only_before_install)

    import mtplx.gdn_capture as gdn_capture

    monkeypatch.setattr(
        gdn_capture,
        "commit_captured_prefix",
        forbidden("capture commit"),
    )

    def record_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        **_kwargs,
    ):
        mx.eval(hidden_states)
        history_appends.append(
            (list(token_ids), np.asarray(hidden_states, dtype=np.float32).copy())
        )
        return 0.0

    monkeypatch.setattr(generation, "_append_mtp_history", record_history)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_TARGET_PREFIX", "1")
    monkeypatch.delenv("MTPLX_STATE_REBASE_EVERY", raising=False)

    out = generation.generate_mtpk(
        rt,
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.5, top_p=1.0, top_k=1),
        speculative_depth=1,
        min_speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="target_prefix",
        stop_token_ids=set(),
        **generation_kwargs,
    )
    return out, schedule, primary_states, history_appends


def test_generation_exact_a3b_k1_rejects_host_only_draft_modifiers_before_prompt(
    monkeypatch,
):
    with pytest.raises(RuntimeError, match="device draft"):
        _run_exact_a3b_k1_schedule(
            monkeypatch,
            target_tokens=[1],
            max_tokens=2,
            online_correction_cache=True,
        )


def test_generation_exact_a3b_k1_rejects_env_forced_loop_guard_before_prompt(
    monkeypatch,
):
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1")
    with pytest.raises(RuntimeError, match="device draft"):
        _run_exact_a3b_k1_schedule(
            monkeypatch,
            target_tokens=[1],
            max_tokens=2,
        )


def test_generation_exact_a3b_k1_accept_keeps_m2_state_without_generic_commit(
    monkeypatch,
):
    out, schedule, _primary_states, _history_appends = _run_exact_a3b_k1_schedule(
        monkeypatch,
        target_tokens=[1],
        max_tokens=5,
    )

    assert out.tokens == [1, 1, 1, 1, 1]
    assert all(call[0] == "m2" for call in schedule)
    assert out.stats.correction_tokens == 0
    report = out.stats.graphbank["compiled_verify"]
    assert report["m2_calls"] == out.stats.verify_calls
    assert report["m1_calls"] == 0


def test_generation_exact_a3b_k1_reject_uses_primary_state_m1_schedule(
    monkeypatch,
):
    out, schedule, primary_states, history_appends = _run_exact_a3b_k1_schedule(
        monkeypatch,
        target_tokens=[2],
        max_tokens=4,
    )

    # Deferred-correction fold: the rejected cycle emits the correction as
    # the pending primary; the NEXT verify is the rebased M2 running from
    # the rejecting cycle's post-primary state.  The token after each
    # correction comes from the rebased verify's pre-sampled row.
    assert out.tokens == [1, 2, 2, 2]
    assert [call[0] for call in schedule] == ["m2", "m2r", "m2r"]
    assert schedule[1][1][0] == 2  # pending correction is the verify primary
    assert schedule[1][2] is primary_states[0]
    assert schedule[2][1][0] == 2
    assert schedule[2][2] is primary_states[1]
    assert out.stats.correction_tokens == 3
    assert out.stats.deferred_correction_repairs == 3
    route_events = [event for event in out.stats.events if "drafts" in event]
    assert any(event.get("pending_primary") == 2 for event in route_events)
    assert route_events[0]["primary_already_emitted"] is False
    assert all(event["primary_already_emitted"] for event in route_events[1:])
    correction_history = [
        hidden for token_ids, hidden in history_appends if token_ids == [2]
    ]
    assert len(correction_history) == 3
    np.testing.assert_array_equal(correction_history[0], np.full((1, 1, 2), 10))
    np.testing.assert_array_equal(correction_history[1], np.full((1, 1, 2), 11))
    np.testing.assert_array_equal(correction_history[2], np.full((1, 1, 2), 12))
    assert not any(np.all(hidden == 90) for hidden in correction_history)
    report = out.stats.graphbank["compiled_verify"]
    assert report["m2_calls"] == out.stats.verify_calls == 3
    assert report["m1_calls"] == 0


def test_generation_exact_a3b_k1_mixed_schedule_keeps_accept_and_reject_ownership(
    monkeypatch,
):
    out, schedule, primary_states, _history_appends = _run_exact_a3b_k1_schedule(
        monkeypatch,
        target_tokens=[1, 2],
        max_tokens=6,
    )

    assert out.tokens[:3] == [1, 1, 1]
    # Accept keeps the plain M2 schedule (state continues from the slots);
    # every rejection folds into a rebased M2 from the rejecting cycle's
    # post-primary state -- ownership of accept vs reject stays distinct.
    assert [call[0] for call in schedule] == ["m2", "m2", "m2r", "m2r"]
    assert schedule[2][2] is primary_states[1]
    assert schedule[3][2] is primary_states[2]
    assert out.stats.accepted_drafts == 1
    assert out.stats.correction_tokens == 3
    assert out.stats.deferred_correction_repairs == 3
    assert out.stats.events[1]["primary_already_emitted"] is True
    assert out.stats.events[2]["primary_already_emitted"] is True


def test_generation_flag_parity_double_runs_each_verify(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")

    out, model = _run_tiny_mtpk()

    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["mode"] == "parity"
    assert bank_stats["parity_checks"] == bank_stats["compiled_calls"]
    assert bank_stats["parity_checks"] >= 1
    assert bank_stats["parity_failures"] == 0
    assert out.tokens == [1, 1, 1, 1, 1]


def test_generation_flag_parity2_compiled_authoritative(monkeypatch):
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    baseline, _ = _run_tiny_mtpk()

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    out, model = _run_tiny_mtpk()

    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["mode"] == "parity2"
    assert bank_stats["parity2_calls"] == bank_stats["compiled_calls"]
    assert bank_stats["parity2_calls"] >= 1
    assert bank_stats["parity2_divergent_calls"] == 0
    assert bank_stats["parity2_first_divergence"] is None
    assert bank_stats["parity_checks"] == 0
    assert out.tokens == baseline.tokens == [1, 1, 1, 1, 1]


def test_generation_other_strategies_ignore_flag(monkeypatch):
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    rt, model = _tiny_mtpk_runtime()
    out = generate_mtpk(
        rt,
        [0],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.stats.graphbank == {}
    assert model.capture_calls == []  # batched verify never uses the capture path


def test_profiles_accept_compiled_verify_env_keys():
    from mtplx.profiles import (
        MODEL_RUNTIME_ENV_OVERRIDE_KEYS,
        normalize_runtime_env_overrides,
    )

    assert "MTPLX_COMPILED_VERIFY" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_COMPILED_VERIFY_MAX_LEN" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_COMPILED_TARGET_PREFIX" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    normalized = normalize_runtime_env_overrides(
        {
            "MTPLX_COMPILED_VERIFY": "parity",
            "MTPLX_COMPILED_VERIFY_MAX_LEN": 6,
            "MTPLX_COMPILED_TARGET_PREFIX": True,
        }
    )
    assert normalized == {
        "MTPLX_COMPILED_VERIFY": "parity",
        "MTPLX_COMPILED_VERIFY_MAX_LEN": "6",
        "MTPLX_COMPILED_TARGET_PREFIX": "1",
    }
    # parity2 is a VALUE of the exact-match MTPLX_COMPILED_VERIFY key, so the
    # existing key list already carries it through contract overrides.
    assert normalize_runtime_env_overrides({"MTPLX_COMPILED_VERIFY": "parity2"}) == {
        "MTPLX_COMPILED_VERIFY": "parity2"
    }


# -- parity abort + gate-script row logic (step 4) -----------------------------


def test_parity_mismatch_aborts_generate_mtpk_stream(monkeypatch):
    """A parity mismatch must abort the stream, not degrade silently."""
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    rt, model = _tiny_mtpk_runtime()

    call_counter = {"n": 0}

    def skewed_capture_stub(
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        # Each invocation returns different logits, so the compiled trace
        # bakes one value and the authoritative eager leg produces another.
        call_counter["n"] += 1
        length = int(input_ids.shape[1])
        logits = mx.full((1, length, 4), float(call_counter["n"]), dtype=mx.float32)
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden, {}
        return logits, {}

    rt.forward_ar_capture = skewed_capture_stub

    with pytest.raises(CompiledVerifyParityError) as excinfo:
        generate_mtpk(
            rt,
            [0],
            max_tokens=5,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="capture_commit",
            stop_token_ids=set(),
        )
    assert any(line.startswith("logits") for line in excinfo.value.report)


def test_parity2_mismatch_does_not_abort_generate_mtpk_stream(monkeypatch, capsys):
    """The same skew that aborts parity#1 must stream to completion under
    parity2, with divergences counted and logged instead of raised."""
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    rt, model = _tiny_mtpk_runtime()

    call_counter = {"n": 0}

    def skewed_capture_stub(
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        call_counter["n"] += 1
        length = int(input_ids.shape[1])
        logits = mx.full((1, length, 4), float(call_counter["n"]), dtype=mx.float32)
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden, {}
        return logits, {}

    rt.forward_ar_capture = skewed_capture_stub

    out = generate_mtpk(
        rt,
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        stop_token_ids=set(),
    )

    assert len(out.tokens) == 5  # stream ran to completion
    bank_stats = out.stats.graphbank["compiled_verify"]
    assert bank_stats["mode"] == "parity2"
    assert bank_stats["parity2_divergent_calls"] >= 1
    first = bank_stats["parity2_first_divergence"]
    assert first is not None and first["artifact"] == "logits"
    assert "[parity2] divergence" in capsys.readouterr().out


def test_exactness_gate_script_row_logic(monkeypatch):
    """The gate script's row builder works on stubs and flags thin coverage."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from mtplx.sampling import SamplerConfig

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "compiled_verify_exactness.py"
    )
    spec = importlib.util.spec_from_file_location("compiled_verify_exactness", script_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    assert gate._csv_ints("1,2,3") == [1, 2, 3]
    with pytest.raises(Exception):
        gate._csv_ints("0")
    tokens, repeated = gate._repeat_tokens([1, 2], 5)
    assert tokens == [1, 2, 1, 2, 1] and repeated is True

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    rt, model = _tiny_mtpk_runtime()
    args = SimpleNamespace(max_tokens=5, min_verify_calls=1, seed=0)
    row = gate._run_case(
        rt,
        [0],
        depth=3,
        sampler_name="greedy",
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        args=args,
    )
    assert row["mismatch"] is False
    assert row["parity_failures"] == 0
    assert row["parity_checks"] >= 1
    assert row["passed"] is True
    # The context manager restored the env after the run.
    assert os.environ.get("MTPLX_COMPILED_VERIFY") is None

    # Thin coverage is inconclusive, not a pass.
    rt2, _model2 = _tiny_mtpk_runtime()
    strict = SimpleNamespace(max_tokens=5, min_verify_calls=99, seed=0)
    row = gate._run_case(
        rt2,
        [0],
        depth=3,
        sampler_name="greedy",
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        args=strict,
    )
    assert row["passed"] is False
    assert "inconclusive" in row["verdict_note"]


def test_compiled_verify_max_context_parses(monkeypatch):
    from mtplx.graphbank import _compiled_verify_max_context

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", raising=False)
    assert _compiled_verify_max_context() == 6144
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "12288")
    assert _compiled_verify_max_context() == 12288
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "0")
    assert _compiled_verify_max_context() == 0
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "junk")
    assert _compiled_verify_max_context() == 6144


def test_compiled_verify_quant_bits_gate(monkeypatch):
    """Turbo promotes compiled verify default-on for the measured-win trunks
    (4-bit Speed and, after the 2026-07-04 re-measure with growth-demote +
    shared traces, 8-bit Quality). Unmeasured quantizations (6-bit) stay
    eager; unquantized test rigs pass; FORCE overrides."""
    from types import SimpleNamespace

    from mtplx.graphbank import (
        CompiledVerifyBank,
        _compiled_verify_bits_gate_ok,
        _runtime_trunk_quant_bits,
    )

    def runtime_with_bits(bits):
        proj = SimpleNamespace(bits=bits) if bits is not None else SimpleNamespace()
        layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=proj))
        inner = SimpleNamespace(layers=[layer])
        model = SimpleNamespace(model=inner)
        return SimpleNamespace(model=model)

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_FORCE", raising=False)
    assert _runtime_trunk_quant_bits(runtime_with_bits(4)) == 4
    assert _runtime_trunk_quant_bits(runtime_with_bits(8)) == 8
    assert _runtime_trunk_quant_bits(runtime_with_bits(None)) is None

    assert _compiled_verify_bits_gate_ok(runtime_with_bits(4)) is True
    assert _compiled_verify_bits_gate_ok(runtime_with_bits(8)) is True
    assert _compiled_verify_bits_gate_ok(runtime_with_bits(None)) is True
    assert _compiled_verify_bits_gate_ok(runtime_with_bits(6)) is False

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_FORCE", "1")
    assert _compiled_verify_bits_gate_ok(runtime_with_bits(6)) is True
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_FORCE", raising=False)

    six_bit_bank = CompiledVerifyBank(runtime_with_bits(6))
    assert six_bit_bank.permanent_eager is True
    q8_bank = CompiledVerifyBank(runtime_with_bits(8))
    assert q8_bank.permanent_eager is False
    four_bit_bank = CompiledVerifyBank(runtime_with_bits(4))
    assert four_bit_bank.permanent_eager is False
    # parity diagnostics bypass the gate deliberately
    parity_bank = CompiledVerifyBank(runtime_with_bits(6), parity2=True)
    assert parity_bank.permanent_eager is False


# -- A2.1 commit-first donation (speed-war Lane A2, 2026-07-06) ----------------


def test_donation_env_default_on(monkeypatch):
    from mtplx.graphbank import _compiled_verify_donation_enabled

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_DONATION", raising=False)
    assert _compiled_verify_donation_enabled() is True
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "0")
    assert _compiled_verify_donation_enabled() is False
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "off")
    assert _compiled_verify_donation_enabled() is False
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "1")
    assert _compiled_verify_donation_enabled() is True


def test_donation_and_legacy_hold_paths_are_bit_identical(monkeypatch):
    """The commit-first ownership handoff must not change a single byte of
    logits, hidden, captures, or committed cache state across a multi-step
    accept-path session (chained pending graphs included)."""

    def run_session(donation: str):
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", donation)
        rt = ToyHybridRuntime(seed=11)
        cache = _prefill(rt, [0, 1, 2])
        bank = CompiledVerifyBank(rt)
        outputs = []
        for window, keep in zip(VERIFY_WINDOWS, [3, 2, 1, 3]):
            logits, hidden, captures = bank.forward_ar_capture(
                mx.array([window]), cache=cache
            )
            committed = commit_captured_prefix(
                cache,
                captures,
                keep_tokens=keep,
                verified_tokens=len(window),
            )
            assert committed is True
            offset = int(cache[1].size())
            outputs.append(
                {
                    "logits": np.array(logits),
                    "hidden": np.array(hidden),
                    "offset": offset,
                    "kv_prefix": np.array(cache[1].cache[0][..., :offset, :]),
                    "v_prefix": np.array(cache[1].cache[1][..., :offset, :]),
                    "gdn_conv": np.array(cache[0].cache[0]),
                    "gdn_state": np.array(cache[0].cache[1]),
                }
            )
        assert bank.stats["compiled_calls"] == len(VERIFY_WINDOWS)
        assert bank.stats["fallback_calls"] == 0
        return outputs

    donated = run_session("1")
    legacy = run_session("0")
    for step, (got, want) in enumerate(zip(donated, legacy)):
        assert got["offset"] == want["offset"], f"step {step}"
        for name in ("logits", "hidden", "kv_prefix", "v_prefix", "gdn_conv", "gdn_state"):
            assert np.array_equal(got[name], want[name]), f"step {step}: {name}"


def test_donation_clears_shadow_leaf_refs_and_held_refs(monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "1")
    rt = ToyHybridRuntime()
    bank = CompiledVerifyBank(rt)
    cache = _prefill(rt, [0, 1, 2])
    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    assert bank.stats["compiled_calls"] == 1
    assert bank._held_state_refs == []
    # Real entries advanced and rollback cleared (mirror-commit semantics).
    entry = cache[1]
    assert isinstance(entry, TensorOffsetKVCache)
    assert entry.rollback_state == [None, None, None]
    assert entry.size() == 6
    # Reject path still trims offset-only, and the next call still works.
    entry.trim(2)
    assert entry.size() == 4
    bank.forward_ar_capture(mx.array([[1, 3]]), cache=cache)
    assert entry.size() == 6
    assert bank.stats["fallback_calls"] == 0


def test_donation_snapshot_views_survive_later_calls(monkeypatch):
    """Zero-copy session-bank-style views taken between verify calls must
    keep their bytes when later calls donate the buffers (COW pinning)."""
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "1")
    rt = ToyHybridRuntime(seed=13)
    cache = _prefill(rt, [0, 1, 2])
    bank = CompiledVerifyBank(rt)

    bank.forward_ar_capture(mx.array([VERIFY_WINDOWS[0]]), cache=cache)
    entry = cache[1]
    snap_keys = entry.cache[0][...]  # lazy zero-copy view (bank pattern)
    snap_vals = entry.cache[1][...]
    expected_keys = np.array(snap_keys)
    expected_vals = np.array(snap_vals)

    for window in VERIFY_WINDOWS[1:]:
        bank.forward_ar_capture(mx.array([window]), cache=cache)
    mx.synchronize()

    assert np.array_equal(np.array(snap_keys), expected_keys)
    assert np.array_equal(np.array(snap_vals), expected_vals)


def test_extended_warmup_env_and_packed_prewarm(monkeypatch):
    """Lane E: extended-warmup gate parses; the packed-GQA pipeline prewarm
    respects the kernel env and never raises."""
    from mtplx.server.openai import (
        _extended_warmup_enabled,
        _prewarm_gqa_packed_pipelines,
    )

    monkeypatch.delenv("MTPLX_WARMUP_EXTENDED", raising=False)
    assert _extended_warmup_enabled() is True
    monkeypatch.setenv("MTPLX_WARMUP_EXTENDED", "0")
    assert _extended_warmup_enabled() is False
    monkeypatch.setenv("MTPLX_WARMUP_EXTENDED", "1")
    assert _extended_warmup_enabled() is True

    monkeypatch.delenv("MTPLX_GQA_PACKED_SDPA", raising=False)
    assert _prewarm_gqa_packed_pipelines() is False  # kernel env off
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    assert _prewarm_gqa_packed_pipelines() in (True, False)  # metal-dependent
