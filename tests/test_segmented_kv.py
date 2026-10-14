"""Segmented KV cache (MTPLX_SEGMENTED_KV): data structure, rollback, seal, merge, snapshots.

Small shapes on the GPU; the kernel and the attention route have their own test files
(test_sdpa_segmented.py, test_segmented_kv_attention.py) and the session bank flow is in
test_segmented_kv_bank.py.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import mtplx.segmented_kv as segmented_kv_module
from mtplx.cache_state import (
    configure_tail_owned_attention_kv_cache,
    restore_cache,
    snapshot_cache,
    snapshot_cache_lazy_hybrid,
)
from mtplx.segmented_kv import (
    KVSegment,
    SegmentedKVCache,
    SegmentedKVState,
    SegRef,
    attend_segments_lse,
    install_segmented_attention_kv_cache,
    merge_segment_outputs,
    route_counts,
    segment_sdpa_lse,
    segmented_kv_enabled,
    unique_state_bytes,
)

HEADS = 2
DIM = 8


@pytest.fixture(autouse=True)
def _no_merge_by_default(monkeypatch):
    # Merge-policy tests opt back in; the others count segments.
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "0")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", "1000")


@pytest.fixture(autouse=True)
def _model_checked(monkeypatch):
    # segmented_kv_enabled() needs a verdict on the loaded model (without one it stays off).
    monkeypatch.setattr(segmented_kv_module, "_MODEL_SUPPORT", {"supported": True, "reasons": []})


def _rows(seed: int, n: int) -> tuple[mx.array, mx.array]:
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed + 50_000)).astype(mx.bfloat16)
    return k, v


def _fill(cache, chunks: list[tuple[int, int]]) -> None:
    for seed, n in chunks:
        cache.update_and_fetch(*_rows(seed, n))


def _same(cache: SegmentedKVCache, control: KVCache) -> bool:
    k, v = cache.gather()
    ck, cv = control.state
    return bool(mx.array_equal(k, ck).item() and mx.array_equal(v, cv).item())


def test_switch_defaults_off_and_is_read_per_call(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_SEGMENTED_KV", raising=False)
    assert segmented_kv_enabled() is False
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    assert segmented_kv_enabled() is True
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "0")
    assert segmented_kv_enabled() is False


def test_switch_off_leaves_the_stock_caches(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_SEGMENTED_KV", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_ATTN", raising=False)
    monkeypatch.delenv("MTPLX_OWNED_ATTN_KV", raising=False)
    cache = [KVCache(), KVCache()]
    configure_tail_owned_attention_kv_cache(cache)
    assert all(type(entry) is KVCache for entry in cache)


def test_switch_on_replaces_only_empty_plain_kv_caches(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_ATTN", raising=False)
    filled = KVCache()
    filled.update_and_fetch(*_rows(1, 4))

    class Recurrent:
        def is_trimmable(self) -> bool:
            return False

    cache = [KVCache(), Recurrent(), filled]
    stats = configure_tail_owned_attention_kv_cache(cache)
    assert isinstance(cache[0], SegmentedKVCache)
    assert isinstance(cache[1], Recurrent)
    assert cache[2] is filled
    assert stats["entries"] == 1 and stats["skipped"] == 2


def test_single_segment_matches_the_stock_cache_bit_for_bit() -> None:
    cache, control = SegmentedKVCache(), KVCache()
    for seed, n in [(1, 300), (2, 1), (3, 7), (4, 600)]:
        got = cache.update_and_fetch(*_rows(seed, n))
        want = control.update_and_fetch(*_rows(seed, n))
        assert cache.offset == control.offset
        assert mx.array_equal(got[0], want[0]).item() and mx.array_equal(got[1], want[1]).item()
    assert cache.segment_count == 1


def test_multiple_segments_gather_equals_the_stock_contents() -> None:
    cache, control = SegmentedKVCache(), KVCache()
    for turn, (seed, n) in enumerate([(1, 300), (2, 5), (3, 700), (4, 9)]):
        for c in (cache, control):
            c.update_and_fetch(*_rows(seed, n))
        cache.seal()  # end of turn
    assert cache.segment_count == 4
    assert cache.offset == control.offset == 1014
    assert _same(cache, control)


def test_make_mask_matches_the_stock_cache() -> None:
    cache = SegmentedKVCache()
    cache.update_and_fetch(*_rows(1, 12))
    control = KVCache()
    control.update_and_fetch(*_rows(1, 12))
    kw = {"return_array": False, "window_size": None}
    assert cache.make_mask(4, **kw) == control.make_mask(4, **kw) == "causal"
    assert cache.make_mask(1, **kw) is None


def test_verify_rollback_stays_in_the_tail_and_never_touches_sealed_segments() -> None:
    cache = SegmentedKVCache()
    _fill(cache, [(1, 400)])
    cache.seal()
    sealed = cache._sealed[0].segment
    before = (sealed.keys, sealed.values)
    _fill(cache, [(2, 10)])
    cache.update_and_fetch(*_rows(3, 4))  # a verify window
    assert cache.offset == 414
    assert cache.trim(3) == 3  # accept one token, roll back three
    assert cache.offset == 411 and cache._tail_n == 11
    cache.update_and_fetch(*_rows(4, 4))
    assert cache.offset == 415
    assert cache._sealed[0].segment is sealed
    assert sealed.keys is before[0] and sealed.values is before[1]
    assert cache.segment_count == 2
    # the control cache saw the same history
    control = KVCache()
    for seed, n in [(1, 400), (2, 10), (3, 4)]:
        control.update_and_fetch(*_rows(seed, n))
    control.trim(3)
    control.update_and_fetch(*_rows(4, 4))
    assert _same(cache, control)


def test_trim_across_sealed_segments_shortens_references_without_a_copy() -> None:
    cache = SegmentedKVCache()
    for seed, n in [(1, 100), (2, 50), (3, 20)]:
        _fill(cache, [(seed, n)])
        cache.seal()
    keys_before = [ref.segment.keys for ref in cache._sealed]
    assert cache.trim(60) == 60  # drops the 20-row segment, shortens the 50-row one
    assert cache.offset == 110
    assert [ref.n for ref in cache._sealed] == [100, 10]
    assert cache._sealed[1].segment.keys is keys_before[1]  # no copy
    assert cache.trim(10_000) == 110
    assert cache.offset == 0 and cache.empty()


def test_seal_copies_only_the_tail_and_gives_it_its_exact_capacity() -> None:
    cache = SegmentedKVCache()
    _fill(cache, [(1, 300)])
    assert cache._tail.capacity > 300  # growth slack
    copies = route_counts.get("seal_copy", 0)
    assert cache.seal() is True
    assert cache.seal() is False  # nothing left to seal
    ref = cache._sealed[0]
    assert ref.segment.sealed and ref.segment.capacity == ref.n == 300
    assert route_counts.get("seal_copy", 0) == copies + 1
    # the history is not copied by a later seal
    _fill(cache, [(2, 40)])
    history = cache._sealed[0].segment.keys
    cache.seal()
    assert cache._sealed[0].segment.keys is history
    assert cache._sealed[1].segment.capacity == 40


def test_state_is_a_list_of_references_and_a_snapshot_never_aliases_the_tail() -> None:
    cache = SegmentedKVCache()
    _fill(cache, [(1, 300)])
    state = cache.state
    assert isinstance(state, SegmentedKVState) and state.rows == 300
    snapshot_k = state.refs[0].segment.keys
    # a later turn writes only into a new tail
    _fill(cache, [(2, 50)])
    assert state.rows == 300 and state.refs[0].segment.keys is snapshot_k
    control = KVCache()
    control.update_and_fetch(*_rows(1, 300))
    assert mx.array_equal(state.to_arrays()[0], control.state[0]).item()


def test_restore_from_a_state_then_extend_equals_a_never_snapshotted_cache() -> None:
    live = SegmentedKVCache()
    control = KVCache()
    for c in (live, control):
        c.update_and_fetch(*_rows(1, 500))
    snapshot = snapshot_cache_lazy_hybrid([live])
    for seed in (10, 11):  # the live cache diverges after the snapshot
        live.update_and_fetch(*_rows(seed, 7))
    restored = SegmentedKVCache()
    restore_cache([restored], snapshot, clone_states=False)
    assert restored.offset == 500
    for c in (restored, control):
        c.update_and_fetch(*_rows(20, 30))
    assert _same(restored, control)
    # the snapshot still serves the original rows
    again = SegmentedKVCache()
    restore_cache([again], snapshot)
    ref = KVCache()
    ref.update_and_fetch(*_rows(1, 500))
    assert again.offset == 500 and _same(again, ref)


def test_fork_inside_a_segment_is_a_reference_without_a_copy() -> None:
    owner = SegmentedKVCache()
    _fill(owner, [(1, 400)])
    state = owner.state
    segment = state.refs[0].segment
    fork = SegmentedKVCache()
    fork.state = state
    fork.trim(150)  # return into the middle of the segment
    assert fork.offset == 250
    assert fork._sealed[0].segment is segment and fork._sealed[0].n == 250
    fork.update_and_fetch(*_rows(7, 20))  # divergent continuation
    control = KVCache()
    control.update_and_fetch(*_rows(1, 400))
    control.trim(150)
    control.update_and_fetch(*_rows(7, 20))
    assert _same(fork, control)
    # the owner's rows are untouched
    ref = KVCache()
    ref.update_and_fetch(*_rows(1, 400))
    assert _same(owner, ref)


def test_state_setter_accepts_a_contiguous_pair() -> None:
    k, v = _rows(5, 64)
    cache = SegmentedKVCache()
    cache.state = (k, v)
    assert cache.offset == 64 and cache.segment_count == 1
    cache.update_and_fetch(*_rows(6, 3))
    assert cache.offset == 67
    cache.state = None
    assert cache.offset == 0


def test_tiered_merge_merges_small_recent_segments_and_leaves_the_history_alone(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "16384")
    cache = SegmentedKVCache()
    _fill(cache, [(1, 4096)])
    cache.seal()  # the big history
    for seed in (2, 3, 4, 5):
        _fill(cache, [(seed, 300)])  # starting a tail runs compact()
        cache.seal()
    # history stays alone; the small segments merged geometrically
    rows = [ref.n for ref in cache._sealed]
    assert rows[0] == 4096
    assert sum(rows) == 4096 + 1200
    assert len(rows) < 5
    control = KVCache()
    for seed, n in [(1, 4096), (2, 300), (3, 300), (4, 300), (5, 300)]:
        control.update_and_fetch(*_rows(seed, n))
    assert _same(cache, control)


def test_merge_also_merges_segments_a_snapshot_holds_and_leaves_the_snapshot_intact(monkeypatch) -> None:
    """Every turn's bank snapshot holds that turn's segments; without merging held segments nothing
    would ever merge and the count would grow by two per turn."""
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "16384")
    cache = SegmentedKVCache()
    for seed in (1, 2):
        _fill(cache, [(seed, 300)])
        cache.seal()
    held = cache.state  # a bank entry references both segments
    held_segments = [ref.segment for ref in held.refs]
    before = [ref.segment for ref in cache._sealed]
    assert before == held_segments
    merges = route_counts.get("merge", 0)
    _fill(cache, [(3, 300)])  # new tail: compact() merges the two equal-sized held segments
    assert len(cache._sealed) == 1 and cache._sealed[0].n == 600
    assert route_counts.get("merge", 0) == merges + 1
    assert held.rows == 600 and [ref.segment for ref in held.refs] == held_segments  # the snapshot is untouched
    control = KVCache()
    for seed in (1, 2):
        control.update_and_fetch(*_rows(seed, 300))
    k, v = held.to_arrays()
    assert mx.array_equal(k, control.state[0]).item() and mx.array_equal(v, control.state[1]).item()
    merged_k = cache._sealed[0].segment.keys[..., :600, :]
    assert mx.array_equal(merged_k, control.state[0]).item()


def test_segment_count_stays_bounded_over_many_turns_with_snapshots(monkeypatch) -> None:
    """Two seals per turn (a prompt snapshot and the final one), every snapshot kept alive."""
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", "16384")
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_TIER", "2")
    cache = SegmentedKVCache()
    _fill(cache, [(1, 20000)])
    cache.seal()
    merges = route_counts.get("merge", 0)
    snapshots = [cache.state]
    peak = 0
    for turn in range(2, 26):
        _fill(cache, [(turn, 700)])
        snapshots.append(cache.state)  # prompt snapshot
        _fill(cache, [(100 + turn, 150)])
        snapshots.append(cache.state)  # final snapshot
        peak = max(peak, cache.segment_count)
    assert peak <= 12, peak
    assert cache.segment_count <= 12
    assert route_counts.get("merge", 0) > merges


def test_nbytes_counts_each_shared_segment_once() -> None:
    cache = SegmentedKVCache()
    _fill(cache, [(1, 256)])
    a = cache.state
    b = cache.state
    assert unique_state_bytes([a, b]) == a.nbytes
    _fill(cache, [(2, 256)])
    c = cache.state
    assert unique_state_bytes([a, b, c]) < a.nbytes + b.nbytes + c.nbytes
    assert unique_state_bytes([c]) == c.nbytes == 2 * a.nbytes


def test_eager_snapshot_of_a_segmented_cache_costs_no_copy() -> None:
    cache = SegmentedKVCache()
    _fill(cache, [(1, 256)])
    snapshot = snapshot_cache([cache])
    assert isinstance(snapshot.states[0], SegmentedKVState)
    assert snapshot.states[0].refs[0].segment.keys is cache._sealed[0].segment.keys


def test_install_skips_a_cache_with_rows_and_keeps_existing_segmented() -> None:
    seg = SegmentedKVCache()
    cache = [seg, KVCache()]
    stats = install_segmented_attention_kv_cache(cache)
    assert cache[0] is seg and isinstance(cache[1], SegmentedKVCache)
    assert stats["entries"] == 2


def test_lse_merge_equals_attention_over_the_concatenated_rows() -> None:
    """The prefill interface: per segment (out, lse) plus one merge is exact (fp32 reference)."""
    heads_q, dim, q_len = 4, 16, 6
    cache = SegmentedKVCache()
    pieces = []
    for seed, n in [(1, 200), (2, 70), (3, 40)]:
        k = mx.random.normal((1, HEADS, n, dim), key=mx.random.key(seed)).astype(mx.bfloat16)
        v = mx.random.normal((1, HEADS, n, dim), key=mx.random.key(seed + 99)).astype(mx.bfloat16)
        cache.update_and_fetch(k, v)
        cache.seal()
        pieces.append((k, v))
    # the last q_len rows of the cache are the new tokens: put a tail with them
    k_new = mx.random.normal((1, HEADS, q_len, dim), key=mx.random.key(7)).astype(mx.bfloat16)
    v_new = mx.random.normal((1, HEADS, q_len, dim), key=mx.random.key(8)).astype(mx.bfloat16)
    cache.update_and_fetch(k_new, v_new)
    q = mx.random.normal((1, heads_q, q_len, dim), key=mx.random.key(9)).astype(mx.bfloat16)
    scale = dim ** -0.5
    got = attend_segments_lse(q, cache, scale=scale)
    keys, values = cache.gather()
    want = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask="causal")
    diff = mx.abs(got.astype(mx.float32) - want.astype(mx.float32))
    assert float(diff.max()) < 0.05  # bf16 output rounding of both routes
    # and exactly equal to an fp32 reference
    kf = mx.repeat(keys.astype(mx.float32), heads_q // HEADS, axis=1)
    vf = mx.repeat(values.astype(mx.float32), heads_q // HEADS, axis=1)
    s = (q.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) * scale
    total = keys.shape[2]
    vis = mx.arange(total)[None, :] <= (total - q_len + mx.arange(q_len))[:, None]
    s = mx.where(vis[None, None], s, -mx.inf)
    ref = mx.softmax(s, axis=-1) @ vf
    assert float(mx.abs(got.astype(mx.float32) - ref).max()) < 0.02


@pytest.fixture
def cpu():
    """The exactness tests run on the CPU: MLX float32 GEMM on a tensor-unit GPU rounds like TF32."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _rounding_case(sizes, q_len, dim=64, heads_q=8):
    cache = SegmentedKVCache()
    for i, n in enumerate(sizes[:-1]):
        k = mx.random.normal((1, HEADS, n, dim), key=mx.random.key(10 + i)).astype(mx.bfloat16)
        v = mx.random.normal((1, HEADS, n, dim), key=mx.random.key(50 + i)).astype(mx.bfloat16)
        cache.update_and_fetch(k, v)
        cache.seal()
    k = mx.random.normal((1, HEADS, sizes[-1], dim), key=mx.random.key(7)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, sizes[-1], dim), key=mx.random.key(8)).astype(mx.bfloat16)
    cache.update_and_fetch(k, v)
    q = mx.random.normal((1, heads_q, q_len, dim), key=mx.random.key(9)).astype(mx.bfloat16)
    keys, values = cache.gather()
    scale = dim ** -0.5
    kf = mx.repeat(keys.astype(mx.float32), heads_q // HEADS, axis=1)
    vf = mx.repeat(values.astype(mx.float32), heads_q // HEADS, axis=1)
    s = (q.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) * scale
    total = keys.shape[2]
    vis = mx.arange(total)[None, :] <= (total - q_len + mx.arange(q_len))[:, None]
    s = mx.where(vis[None, None], s, -mx.inf)
    exact = (mx.softmax(s, axis=-1) @ vf).astype(mx.bfloat16)
    return cache, q, keys, values, scale, exact


def _differing(a, b) -> float:
    return float(mx.mean(a.astype(mx.float32) != b.astype(mx.float32)).item())


@pytest.mark.parametrize("q_len", [1, 6])
@pytest.mark.parametrize("sizes", [[3000, 200], [2000, 1024, 600, 40]])
def test_reference_lse_route_rounds_once(cpu, monkeypatch, q_len, sizes) -> None:
    """The reference route must be the float32 attention rounded once to the query dtype.

    A tolerance of one bf16 ulp does not see a second rounding (every segment rounded to bf16
    before the merge), which puts a quarter or more of the elements one ulp off.
    """
    monkeypatch.setattr(segmented_kv_module, "sdpa_lse_available", lambda: False)
    cache, q, _, _, scale, exact = _rounding_case(sizes, q_len)
    got = attend_segments_lse(q, cache, scale=scale)
    assert got.dtype == mx.bfloat16
    assert _differing(got, exact) <= 0.005


@pytest.mark.parametrize("q_len", [1, 6])
def test_reference_lse_route_is_as_close_to_stock_sdpa_as_the_exact_result(cpu, monkeypatch, q_len) -> None:
    monkeypatch.setattr(segmented_kv_module, "sdpa_lse_available", lambda: False)
    cache, q, keys, values, scale, exact = _rounding_case([3000, 1000, 200], q_len)
    got = attend_segments_lse(q, cache, scale=scale)
    stock = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask="causal" if q_len > 1 else None)
    assert _differing(got, stock) <= _differing(exact, stock) + 0.005


def test_merge_rounds_float32_segment_outputs_once_to_the_requested_dtype(cpu) -> None:
    a = mx.random.normal((1, 2, 3, 4), key=mx.random.key(1))
    b = mx.random.normal((1, 2, 3, 4), key=mx.random.key(2))
    la, lb = mx.zeros((1, 2, 3)), mx.zeros((1, 2, 3))
    merged = merge_segment_outputs([a, b], [la, lb], dtype=mx.bfloat16)
    assert merged.dtype == mx.bfloat16
    assert mx.array_equal(merged, ((a + b) / 2).astype(mx.bfloat16)).item()
    assert merge_segment_outputs([a], [la], dtype=mx.bfloat16).dtype == mx.bfloat16


def test_merge_of_one_segment_is_the_segment_output() -> None:
    out = mx.ones((1, 2, 3, 4), dtype=mx.bfloat16)
    lse = mx.zeros((1, 2, 3))
    assert merge_segment_outputs([out], [lse]) is out


def test_segment_sdpa_lse_is_replaceable() -> None:
    import mtplx.segmented_kv as module

    assert module.segment_sdpa_lse is not None
    assert callable(segment_sdpa_lse)


def test_kvsegment_and_segref_are_plain_references() -> None:
    k, v = _rows(1, 4)
    segment = KVSegment(k, v, sealed=True)
    ref = SegRef(segment, 3)
    assert ref.n == 3 and ref.segment.capacity == 4
    assert segment.nbytes == k.nbytes + v.nbytes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _lse_cache(q_len=4, dim=16):
    cache = SegmentedKVCache()
    for seed, n in [(1, 120), (2, 60)]:
        cache.update_and_fetch(*(mx.random.normal((1, HEADS, n, dim), key=mx.random.key(seed + o)).astype(mx.bfloat16) for o in (0, 99)))
        cache.seal()
    cache.update_and_fetch(*(mx.random.normal((1, HEADS, q_len, dim), key=mx.random.key(7 + o)).astype(mx.bfloat16) for o in (0, 5)))
    q = mx.random.normal((1, 4, q_len, dim), key=mx.random.key(9)).astype(mx.bfloat16)
    return cache, q


def test_stock_mlx_has_no_lse_kernel_and_falls_back_to_the_reference_route(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    monkeypatch.setattr(module, "_SDPA_LSE", None)
    monkeypatch.delenv("MTPLX_SEGMENTED_KV_PREFILL", raising=False)
    if module.sdpa_lse_available():
        pytest.skip("this MLX build has return_lse")
    monkeypatch.setattr(module, "_SDPA_LSE", None)
    assert module.sdpa_lse_available() is False  # stock released MLX in the venv
    assert module.prefill_route() == "gather"
    module.route_counts.clear()
    cache, q = _lse_cache()
    attend_segments_lse(q, cache, scale=0.25)
    assert {k: v for k, v in module.route_counts.items() if k.startswith("sdpa_lse")} == {"sdpa_lse_reference": 3}


def test_lse_kernel_path_plumbing_with_mocked_detection(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    monkeypatch.setattr(module, "_SDPA_LSE", True)
    monkeypatch.delenv("MTPLX_SEGMENTED_KV_PREFILL", raising=False)
    assert module.prefill_route() == "lse"
    calls = []

    def fake_sdpa(q, k, v, *, scale, return_lse=False, mask=None):
        calls.append((int(k.shape[2]), mask))
        out, lse = _segment_sdpa_lse_reference_for_test(q, k, v, scale, mask)
        return out, lse[..., None]  # the MLX kernel returns (B, Hq, Q, 1)

    def _segment_sdpa_lse_reference_for_test(q, k, v, scale, mask):
        return module._segment_sdpa_lse_reference(q, k, v, int(k.shape[2]), scale=scale, causal=mask == "causal")

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake_sdpa)
    module.route_counts.clear()
    cache, q = _lse_cache()
    got = attend_segments_lse(q, cache, scale=0.25)
    assert {k: v for k, v in module.route_counts.items() if k.startswith("sdpa_lse")} == {"sdpa_lse_kernel": 3}
    assert [c[1] for c in calls] == [None, None, "causal"]  # only the last segment is causal
    monkeypatch.setattr(module, "_SDPA_LSE", False)
    want = attend_segments_lse(q, cache, scale=0.25)
    assert float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max()) < 1e-2


def test_prefill_route_env_override(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    monkeypatch.setattr(module, "_SDPA_LSE", False)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV_PREFILL", "lse")
    assert module.prefill_route() == "lse"


def test_real_lse_kernel_matches_the_reference_route() -> None:
    import mtplx.segmented_kv as module

    if not module.sdpa_lse_available():
        pytest.skip("MLX build without return_lse")
    cache, q = _lse_cache()
    module.route_counts.clear()
    got = attend_segments_lse(q, cache, scale=0.25)
    assert module.route_counts.get("sdpa_lse_kernel") == len(cache.attention_segments())
    keys, values = cache.gather()
    want = mx.fast.scaled_dot_product_attention(q, keys, values, scale=0.25, mask="causal")
    assert float(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)).max()) < 2e-2


def _fake_lse_api(calls, tag):
    def fake(q, k, v, *, scale, mask=None, **extra):
        calls.append((tag, mask, extra))
        out = mx.zeros(q.shape, dtype=q.dtype)
        return out, mx.zeros((*q.shape[:-1], 1), dtype=mx.float32)

    return fake


def test_sdpa_with_lse_prefers_the_separate_function(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    calls: list = []
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention_with_lse", _fake_lse_api(calls, "func"), raising=False)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", _fake_lse_api(calls, "kw"))
    q = mx.zeros((1, 1, 1, 64), dtype=mx.bfloat16)
    module.sdpa_with_lse(q, q, q, scale=1.0, mask="causal")
    module.sdpa_with_lse(q, q, q, scale=1.0)
    assert calls == [("func", "causal", {}), ("func", None, {})]
    monkeypatch.setattr(module, "_SDPA_LSE", None)
    assert module.sdpa_lse_available() is True
    assert module.lse_api_found() == module.LSE_API_FUNCTION


def test_sdpa_with_lse_falls_back_to_the_return_lse_keyword(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    calls: list = []
    monkeypatch.delattr(mx.fast, "scaled_dot_product_attention_with_lse", raising=False)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", _fake_lse_api(calls, "kw"))
    q = mx.zeros((1, 1, 1, 64), dtype=mx.bfloat16)
    module.sdpa_with_lse(q, q, q, scale=1.0, mask="causal")
    assert calls == [("kw", "causal", {"return_lse": True})]
    monkeypatch.setattr(module, "_SDPA_LSE", None)
    assert module.sdpa_lse_available() is True
    assert module.lse_api_found() == module.LSE_API_KEYWORD


def test_no_lse_api_reports_unavailable(monkeypatch) -> None:
    import mtplx.segmented_kv as module

    monkeypatch.delattr(mx.fast, "scaled_dot_product_attention_with_lse", raising=False)

    def stock(q, k, v, *, scale, mask=None):
        raise TypeError("unexpected keyword return_lse")

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", stock)
    monkeypatch.setattr(module, "_SDPA_LSE", None)
    assert module.sdpa_lse_available() is False
    assert module.lse_api_found() is None


def test_merge_of_fully_masked_rows_is_zero_not_nan() -> None:
    from mtplx.segmented_kv import merge_segment_outputs

    ninf = float("-inf")
    out_a = mx.array([[[[1.0, 2.0], [float("nan"), float("nan")], [3.0, 4.0]]]])
    out_b = mx.array([[[[5.0, 6.0], [float("nan"), float("nan")], [0.0, 0.0]]]])
    lse_a = mx.array([[[0.0, ninf, 0.0]]])
    lse_b = mx.array([[[0.0, ninf, ninf]]])
    merged = merge_segment_outputs([out_a, out_b], [lse_a, lse_b]).astype(mx.float32)
    assert merged.tolist() == [[[[3.0, 4.0], [0.0, 0.0], [3.0, 4.0]]]]


def test_reference_route_fully_masked_row_is_zero_not_nan() -> None:
    import mtplx.segmented_kv as module

    q = mx.random.normal((1, 2, 4, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 2, 64), key=mx.random.key(2)).astype(mx.bfloat16)
    # 4 queries over a 2-row causal segment: the first two rows see nothing.
    out, lse = module._segment_sdpa_lse_reference(q, k, k, 2, scale=0.125, causal=True)
    assert lse[..., :2].tolist() == [[[float("-inf")] * 2] * 2]
    assert not mx.isnan(out.astype(mx.float32)).any().item()
    assert float(mx.abs(out[:, :, :2].astype(mx.float32)).max()) == 0.0
