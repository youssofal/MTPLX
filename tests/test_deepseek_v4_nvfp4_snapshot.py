import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

_IMPORT_DEVICE = mx.default_device()
mx.set_default_device(mx.cpu)
from mtplx.cache_state import (  # noqa: E402
    CacheSnapshot,
    restore_cache,
    snapshot_cache,
    snapshot_cache_lazy_hybrid,
)
from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache  # noqa: E402
from mtplx.deepseek_v4_mia_engine import MiaTargetCacheArena  # noqa: E402
mx.set_default_device(_IMPORT_DEVICE)


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _new_cache(
    *,
    compress_ratio: int = 4,
    max_batch_tokens: int = 8,
    capacity_tokens: int = 256,
) -> DeepseekV4NVFP4Cache:
    kwargs = {"max_batch_tokens": max_batch_tokens}
    if compress_ratio:
        kwargs["capacity_tokens"] = capacity_tokens
    return DeepseekV4NVFP4Cache(
        window_size=8,
        compress_ratio=compress_ratio,
        head_dim=512,
        rollback_capacity=8,
        **kwargs,
    )


def _projected_rows(*, rows: int, width: int, base: float) -> mx.array:
    values = mx.arange(rows * width, dtype=mx.float32).reshape(1, rows, width)
    return values / 1024.0 + base


def _install_request(
    cache: DeepseekV4NVFP4Cache,
    *,
    base: float,
    rows: int = 7,
) -> None:
    cache.state = None
    cache.window._append_installed_records(
        mx.full((1, rows, 432), int(base) & 0xFF, dtype=mx.uint8),
        absolute_start=0,
    )
    retained_start = max(0, rows - cache.window_size - cache.rollback_capacity)
    cache.window.drop_before(retained_start)
    if cache.compress_ratio:
        compressed_rows = rows // cache.compress_ratio
        cache.compressed._append_installed_records(
            mx.full(
                (1, compressed_rows, 432),
                (int(base) + 1) & 0xFF,
                dtype=mx.uint8,
            )
        )
        if cache.index_compressed is not None:
            cache.index_compressed._append_installed_records(
                mx.full(
                    (1, compressed_rows, 132),
                    (int(base) + 2) & 0xFF,
                    dtype=mx.uint8,
                )
            )
    lanes = []
    if cache.compress_ratio:
        lanes.append((cache.comp, base))
    if cache.index_compressed is not None:
        lanes.append((cache.index_comp, base + 1000.0))
    for lane, lane_base in lanes:
        kv = _projected_rows(
            rows=rows,
            width=lane.state_width,
            base=lane_base,
        )
        score = _projected_rows(
            rows=rows,
            width=lane.state_width,
            base=lane_base + 100.0,
        )
        lane.append_projected_rows(kv, score, offset=0)
        lane.rollback(0, rows)
    cache.offset = rows
    cache.window_start = retained_start
    mx.eval(cache.state)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.array(value)


def _assert_lane_equal(actual, expected) -> None:
    assert actual._journal_end == expected._journal_end
    assert actual._journal_length == expected._journal_length
    for actual_array, expected_array in (
        (actual._journal_kv, expected._journal_kv),
        (actual._journal_score, expected._journal_score),
        (actual.cur_kv, expected.cur_kv),
        (actual.cur_score, expected.cur_score),
        (actual.prev_kv, expected.prev_kv),
        (actual.prev_score, expected.prev_score),
    ):
        if expected_array is None:
            assert actual_array is None
            continue
        np.testing.assert_array_equal(
            _array(actual_array),
            _array(expected_array),
        )


def _fixed_owner_ids(cache: DeepseekV4NVFP4Cache) -> tuple[int, ...]:
    owners = [
        id(cache.window._pages),
        id(cache.window._pool.block_table),
    ]
    for rows in (cache.compressed, cache.index_compressed):
        if rows is not None and hasattr(rows, "pages"):
            owners.extend((id(rows.pages), id(rows.block_table)))
    for lane in (cache.comp, cache.index_comp):
        if hasattr(lane, "_journal_kv"):
            owners.extend((id(lane._journal_kv), id(lane._journal_score)))
    return tuple(owners)


def _array_object_ids(value) -> tuple[int, ...]:
    if isinstance(value, mx.array):
        return (id(value),)
    if isinstance(value, (tuple, list)):
        return tuple(
            object_id
            for item in value
            for object_id in _array_object_ids(item)
        )
    return ()


def _assert_state_tree_equal(actual, expected) -> None:
    if isinstance(expected, mx.array):
        assert isinstance(actual, mx.array)
        np.testing.assert_array_equal(_array(actual), _array(expected))
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_tree_equal(actual_item, expected_item)
        return
    assert actual == expected


def _new_m6_arena():
    layers = tuple(
        SimpleNamespace(
            attn=SimpleNamespace(
                window_size=8,
                compress_ratio=ratio,
                head_dim=512,
            )
        )
        for ratio in (0, 4, 128)
    )
    arena = MiaTargetCacheArena(
        layers,
        capacity_tokens=256,
        max_batch_tokens=6,
    )
    return arena, layers, arena.acquire(layers)


def _stage_m6_lane(lane, schedule, *, base: float) -> None:
    kv = _projected_rows(rows=6, width=lane.state_width, base=base)
    score = _projected_rows(
        rows=6,
        width=lane.state_width,
        base=base + 100.0,
    )
    combined_kv, combined_score = lane.append_m6_projected_rows(
        kv,
        score,
        schedule,
    )
    emitted = schedule.emitted_rows
    filled = emitted * schedule.ratio
    if emitted and lane.overlap:
        lane.prev_kv = combined_kv[:, :filled].reshape(
            1,
            emitted,
            schedule.ratio,
            lane.state_width,
        )[:, -1]
        lane.prev_score = combined_score[:, :filled].reshape(
            1,
            emitted,
            schedule.ratio,
            lane.state_width,
        )[:, -1]
    lane.n_emitted = schedule.first_window + emitted
    total = schedule.prior_rows + 6
    lane.cur_kv = combined_kv[:, filled:] if filled < total else None
    lane.cur_score = combined_score[:, filled:] if filled < total else None


def _stage_m6_cache(cache, schedule, *, base: float) -> None:
    if schedule is not None:
        _stage_m6_lane(cache.comp, schedule, base=base + 10.0)
        cache.compressed._append_m6_records(
            mx.full(
                (1, schedule.emitted_rows, 432),
                int(base + 20.0) & 0xFF,
                dtype=mx.uint8,
            ),
            schedule,
        )
        if cache.compress_ratio == 4:
            _stage_m6_lane(cache.index_comp, schedule, base=base + 30.0)
            cache.index_compressed._append_m6_records(
                mx.full(
                    (1, schedule.emitted_rows, 132),
                    int(base + 40.0) & 0xFF,
                    dtype=mx.uint8,
                ),
                schedule,
            )
    cache.update_window_records(
        mx.full((1, 6, 432), int(base + 50.0) & 0xFF, dtype=mx.uint8)
    )
    cache.advance(6)


@pytest.mark.parametrize("start", [3, 127, 191])
@pytest.mark.parametrize("accepted_rows", range(7))
def test_m6_direct_acceptance_matches_installed_trim_complete_cache_state(
    start: int,
    accepted_rows: int,
) -> None:
    candidate_arena, _candidate_layers, candidate = _new_m6_arena()
    control_arena, _control_layers, control = _new_m6_arena()
    for index, (candidate_cache, control_cache) in enumerate(
        zip(candidate, control, strict=True)
    ):
        _install_request(candidate_cache, base=10.0 + index, rows=start)
        _install_request(control_cache, base=10.0 + index, rows=start)
    candidate_cycle = candidate_arena.begin_verify()
    control_cycle = control_arena.begin_verify()
    for index, (candidate_cache, control_cache) in enumerate(
        zip(candidate, control, strict=True)
    ):
        candidate_schedule = {
            0: None,
            4: candidate_cycle.ratio4,
            128: candidate_cycle.ratio128,
        }[candidate_cache.compress_ratio]
        control_schedule = {
            0: None,
            4: control_cycle.ratio4,
            128: control_cycle.ratio128,
        }[control_cache.compress_ratio]
        _stage_m6_cache(
            candidate_cache,
            candidate_schedule,
            base=100.0 + index * 100.0,
        )
        _stage_m6_cache(
            control_cache,
            control_schedule,
            base=100.0 + index * 100.0,
        )

    acceptance = candidate_arena.commit_verify(accepted_rows)
    for cache in control:
        cache._trim_installed(6 - accepted_rows)
    mx.eval(
        *(cache.state for cache in candidate),
        *(cache.state for cache in control),
    )

    assert acceptance.stop_offset == start + accepted_rows
    assert acceptance.ratio4.compressed_rows == (
        start + accepted_rows
    ) // 4
    assert acceptance.ratio128.compressed_rows == (
        start + accepted_rows
    ) // 128
    assert candidate_arena.current_m6_cycle is None
    for candidate_cache, control_cache in zip(candidate, control, strict=True):
        _assert_state_tree_equal(candidate_cache.state, control_cache.state)
        assert candidate_cache.meta_state == control_cache.meta_state
        assert getattr(candidate_cache.comp, "_pending_m6", None) is None
        assert getattr(candidate_cache.index_comp, "_pending_m6", None) is None


@pytest.mark.parametrize(
    ("snapshot_factory", "clone_states"),
    ((snapshot_cache, True), (snapshot_cache_lazy_hybrid, False)),
    ids=("eager", "lazy-kv"),
)
def test_snapshot_restore_recovers_both_fixed_journals_before_boundary_trim(
    snapshot_factory,
    clone_states: bool,
) -> None:
    restored = _new_cache()
    control = _new_cache()
    _install_request(restored, base=10.0)
    _install_request(control, base=10.0)
    snapshot = snapshot_factory([restored])
    if clone_states:
        mx.eval(snapshot.states)

    # A second request reuses and overwrites the same fixed journal slots.
    _install_request(restored, base=200.0, rows=11)
    restore_cache([restored], snapshot, clone_states=clone_states)

    _assert_lane_equal(restored.comp, control.comp)
    _assert_lane_equal(restored.index_comp, control.index_comp)

    # Rewind 7 -> 3 crosses the ratio-4 emission boundary and must rebuild the
    # attention and indexer frontiers from request A's restored journal bytes.
    restored.trim(4)
    control.trim(4)

    assert restored.offset == control.offset == 3
    _assert_lane_equal(restored.comp, control.comp)
    _assert_lane_equal(restored.index_comp, control.index_comp)


@pytest.mark.parametrize(
    ("snapshot_factory", "clone_states"),
    ((snapshot_cache, True), (snapshot_cache_lazy_hybrid, False)),
    ids=("eager", "lazy-kv"),
)
def test_ratio128_snapshot_restore_recovers_journal_before_boundary_trim(
    snapshot_factory,
    clone_states: bool,
) -> None:
    restored = _new_cache(compress_ratio=128, max_batch_tokens=136)
    control = _new_cache(compress_ratio=128, max_batch_tokens=136)
    _install_request(restored, base=10.0, rows=129)
    _install_request(control, base=10.0, rows=129)
    snapshot = snapshot_factory([restored])
    if clone_states:
        mx.eval(snapshot.states)

    _install_request(restored, base=200.0, rows=130)
    restore_cache([restored], snapshot, clone_states=clone_states)
    _assert_lane_equal(restored.comp, control.comp)

    restored.trim(2)
    control.trim(2)

    assert restored.offset == control.offset == 127
    assert restored.n_compressed == control.n_compressed == 0
    _assert_lane_equal(restored.comp, control.comp)


@pytest.mark.parametrize(
    ("snapshot_factory", "clone_states"),
    ((snapshot_cache, True), (snapshot_cache_lazy_hybrid, False)),
    ids=("eager", "lazy-kv"),
)
@pytest.mark.parametrize(
    ("compress_ratio", "rows", "overwrite_rows", "max_batch_tokens"),
    ((0, 7, 11, 8), (4, 7, 11, 8), (128, 129, 130, 136)),
    ids=("ratio0", "ratio4", "ratio128"),
)
def test_snapshot_restore_preserves_all_fixed_arena_owners(
    snapshot_factory,
    clone_states: bool,
    compress_ratio: int,
    rows: int,
    overwrite_rows: int,
    max_batch_tokens: int,
) -> None:
    cache = _new_cache(
        compress_ratio=compress_ratio,
        max_batch_tokens=max_batch_tokens,
    )
    _install_request(cache, base=10.0, rows=rows)
    owner_ids = _fixed_owner_ids(cache)
    snapshot = snapshot_factory([cache])
    if clone_states:
        mx.eval(snapshot.states)

    _install_request(cache, base=200.0, rows=overwrite_rows)
    restore_cache([cache], snapshot, clone_states=clone_states)

    assert owner_ids == _fixed_owner_ids(cache)
    assert int(cache.window._pages[0, 0, 0].item()) == 10
    assert cache.offset == rows
    assert cache.window_start == max(
        0,
        rows - cache.window_size - cache.rollback_capacity,
    )
    assert len(cache.window) == rows - cache.window_start
    expected_compressed = rows // compress_ratio if compress_ratio else 0
    assert len(cache.compressed) == expected_compressed
    assert cache.n_index_compressed == (
        expected_compressed if compress_ratio == 4 else 0
    )
    if compress_ratio:
        assert int(cache.compressed.pages[0, 0, 0].item()) == 11
    if cache.index_compressed is not None:
        assert int(cache.index_compressed.pages[0, 0, 0].item()) == 12


def test_nvfp4_snapshot_contract_rejects_stale_state_and_meta() -> None:
    cache = _new_cache()
    stale_state = cache.state[1:16]
    wrong_version = list(cache.state)
    wrong_version[0] = "mtplx-deepseek-v4-nvfp4-paged-state-v2"

    with pytest.raises(ValueError, match="unsupported DeepSeek-V4 NVFP4 cache state"):
        cache.state = stale_state
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4 NVFP4 cache state"):
        cache.state = wrong_version
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4 cache meta state"):
        cache.meta_state = (
            "mtplx-deepseek-v4-nvfp4-paged-cache-v2",
            "0",
            "0",
            "0",
            "0",
        )


def test_nvfp4_meta_rejects_physical_frontiers_that_disagree_with_offset() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    pages, block_table, _length = contradictory_state[2]
    contradictory_state[2] = (pages, block_table, 0)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(contradictory_state),),
                meta_states=(source.meta_state,),
            ),
        )

    assert receiver.offset == 0
    assert receiver.window_start == 0
    assert receiver.comp.n_emitted == 0
    assert receiver.index_comp.n_emitted == 0
    assert receiver.comp._journal_end == 0
    assert receiver.index_comp._journal_end == 0


def test_restore_cache_rejection_is_atomic_for_exact_nvfp4_owner() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    _install_request(receiver, base=200.0, rows=11)
    source_snapshot = snapshot_cache([source])
    contradictory_state = list(source_snapshot.states[0])
    pages, block_table, _length = contradictory_state[2]
    contradictory_state[2] = (pages, block_table, 0)
    contradictory_snapshot = CacheSnapshot(
        states=(tuple(contradictory_state),),
        meta_states=source_snapshot.meta_states,
    )
    before = snapshot_cache([receiver])
    mx.eval(before.states)
    before_ids = _array_object_ids(receiver.state)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache([receiver], contradictory_snapshot)

    _assert_state_tree_equal(receiver.state, before.states[0])
    assert receiver.meta_state == before.meta_states[0]
    assert _array_object_ids(receiver.state) == before_ids


def test_restore_cache_preflights_every_exact_entry_before_install() -> None:
    sources = [_new_cache(), _new_cache()]
    receivers = [_new_cache(), _new_cache()]
    _install_request(sources[0], base=10.0)
    _install_request(sources[1], base=20.0)
    _install_request(receivers[0], base=200.0, rows=11)
    _install_request(receivers[1], base=220.0, rows=11)
    snapshot = snapshot_cache(sources)
    states = list(snapshot.states)
    invalid_later = list(states[1])
    pages, block_table, _length = invalid_later[2]
    invalid_later[2] = (pages, block_table, 0)
    states[1] = tuple(invalid_later)
    invalid_snapshot = CacheSnapshot(
        states=tuple(states),
        meta_states=snapshot.meta_states,
    )
    before = snapshot_cache(receivers)
    mx.eval(before.states)
    before_ids = tuple(_array_object_ids(cache.state) for cache in receivers)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache(receivers, invalid_snapshot)

    for index, receiver in enumerate(receivers):
        _assert_state_tree_equal(receiver.state, before.states[index])
        assert receiver.meta_state == before.meta_states[index]
        assert _array_object_ids(receiver.state) == before_ids[index]


@pytest.mark.parametrize(
    ("state_count", "meta_count"),
    ((1, 2), (2, 1), (3, 3)),
    ids=("states-truncated", "meta-truncated", "snapshot-longer-than-cache"),
)
def test_restore_cache_rejects_snapshot_length_mismatch_before_mutation(
    state_count: int,
    meta_count: int,
) -> None:
    sources = [_new_cache(), _new_cache(), _new_cache()]
    receivers = [_new_cache(), _new_cache()]
    for index, source in enumerate(sources):
        _install_request(source, base=10.0 + index * 10.0)
    for index, receiver in enumerate(receivers):
        _install_request(receiver, base=200.0 + index * 20.0, rows=11)
    snapshot = snapshot_cache(sources)
    mismatched = CacheSnapshot(
        states=snapshot.states[:state_count],
        meta_states=snapshot.meta_states[:meta_count],
    )
    before = snapshot_cache(receivers)
    mx.eval(before.states)
    before_ids = tuple(_array_object_ids(cache.state) for cache in receivers)

    with pytest.raises(ValueError, match="cache snapshot length mismatch"):
        restore_cache(receivers, mismatched)

    for index, receiver in enumerate(receivers):
        _assert_state_tree_equal(receiver.state, before.states[index])
        assert receiver.meta_state == before.meta_states[index]
        assert _array_object_ids(receiver.state) == before_ids[index]


def test_restore_cache_rejects_undersized_fixed_window_before_mutation() -> None:
    source = _new_cache(compress_ratio=0)
    receiver = _new_cache(compress_ratio=0)
    _install_request(receiver, base=200.0, rows=11)
    state = list(source.state)
    pages, block_table, _start, _end = state[1]
    state[1] = (pages, block_table, 7, 7)
    meta_state = list(source.meta_state)
    meta_state[1] = "7"
    meta_state[2] = "7"
    before = snapshot_cache([receiver])
    mx.eval(before.states)
    before_ids = _array_object_ids(receiver.state)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(state),),
                meta_states=(tuple(meta_state),),
            ),
        )

    _assert_state_tree_equal(receiver.state, before.states[0])
    assert receiver.meta_state == before.meta_states[0]
    assert _array_object_ids(receiver.state) == before_ids


def test_exact_nvfp4_state_only_restore_rejects_without_mutation() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    _install_request(receiver, base=200.0, rows=11)
    source_snapshot = snapshot_cache([source])
    before = snapshot_cache([receiver])
    mx.eval(before.states)
    before_ids = _array_object_ids(receiver.state)

    with pytest.raises(
        ValueError,
        match=r"DeepSeek-V4 NVFP4 state requires atomic state\+meta restore",
    ):
        restore_cache(
            [receiver],
            source_snapshot,
            restore_meta_state=False,
        )

    _assert_state_tree_equal(receiver.state, before.states[0])
    assert receiver.meta_state == before.meta_states[0]
    assert _array_object_ids(receiver.state) == before_ids


@pytest.mark.parametrize(
    ("state_index", "owner"),
    ((1, "window"), (2, "compressed"), (9, "indexer")),
)
@pytest.mark.parametrize(
    "corruption",
    ("out-of-range", "duplicate", "permutation"),
)
def test_restore_cache_rejects_unsealed_exact_mia_block_table_atomically(
    state_index: int,
    owner: str,
    corruption: str,
) -> None:
    source = _new_cache(max_batch_tokens=64, capacity_tokens=512)
    receiver = _new_cache(max_batch_tokens=64, capacity_tokens=512)
    _install_request(source, base=10.0)
    _install_request(receiver, base=200.0, rows=11)
    source_snapshot = snapshot_cache([source])
    contradictory_state = list(source_snapshot.states[0])
    owner_state = list(contradictory_state[state_index])
    block_table = owner_state[1]
    values = np.array(block_table)
    assert values.tolist() == [0, 1]
    if corruption == "out-of-range":
        values[0] = 999
    elif corruption == "duplicate":
        values[1] = values[0]
    else:
        values = values[::-1].copy()
    owner_state[1] = mx.array(values, dtype=block_table.dtype)
    contradictory_state[state_index] = tuple(owner_state)
    contradictory_snapshot = CacheSnapshot(
        states=(tuple(contradictory_state),),
        meta_states=source_snapshot.meta_states,
    )
    before = snapshot_cache([receiver])
    mx.eval(before.states)
    before_ids = _array_object_ids(receiver.state)

    with pytest.raises(
        ValueError,
        match=f"Mia fixed {owner} block table ownership changed",
    ):
        restore_cache([receiver], contradictory_snapshot)

    _assert_state_tree_equal(receiver.state, before.states[0])
    assert receiver.meta_state == before.meta_states[0]
    assert _array_object_ids(receiver.state) == before_ids


def test_nvfp4_meta_rejects_fixed_window_frontier_mismatch() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    pages, block_table, _start, end = contradictory_state[1]
    contradictory_state[1] = (pages, block_table, 1, end)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(contradictory_state),),
                meta_states=(source.meta_state,),
            ),
        )

    assert receiver.offset == 0
    assert receiver.window_start == 0
    assert receiver.comp.n_emitted == 0
    assert receiver.index_comp.n_emitted == 0


def test_nvfp4_meta_rejects_indexer_page_frontier_mismatch() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    pages, block_table, _length = contradictory_state[9]
    contradictory_state[9] = (pages, block_table, 0)

    with pytest.raises(
        ValueError,
        match="DeepSeek-V4 NVFP4 physical cache frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(contradictory_state),),
                meta_states=(source.meta_state,),
            ),
        )

    assert receiver.offset == 0
    assert receiver.window_start == 0
    assert receiver.comp.n_emitted == 0
    assert receiver.index_comp.n_emitted == 0


def test_nvfp4_state_rejects_malformed_overlap_previous_window() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    contradictory_state[5] = source.comp.prev_kv[:, :3]

    with pytest.raises(
        ValueError,
        match="fixed Mia compressor previous frontier is invalid",
    ):
        receiver.state = contradictory_state

    assert len(receiver.window) == 0
    assert len(receiver.compressed) == 0
    assert len(receiver.index_compressed) == 0


def test_nvfp4_meta_rejects_current_frontier_that_disagrees_with_offset() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    contradictory_state[3] = source.comp.cur_kv[:, :2]
    contradictory_state[4] = source.comp.cur_score[:, :2]

    with pytest.raises(
        ValueError,
        match="fixed Mia compressor current frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(contradictory_state),),
                meta_states=(source.meta_state,),
            ),
        )

    assert receiver.offset == 0
    assert receiver.comp.n_emitted == 0
    assert receiver.comp._journal_end == 0


def test_nvfp4_meta_rejects_missing_overlap_previous_window() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    contradictory_state[5] = None
    contradictory_state[6] = None

    with pytest.raises(
        ValueError,
        match="fixed Mia compressor previous frontier is invalid",
    ):
        restore_cache(
            [receiver],
            CacheSnapshot(
                states=(tuple(contradictory_state),),
                meta_states=(source.meta_state,),
            ),
        )

    assert receiver.offset == 0
    assert receiver.comp.n_emitted == 0
    assert receiver.comp._journal_end == 0


def test_nvfp4_state_rejects_retained_tail_for_fixed_journal_owner() -> None:
    source = _new_cache()
    receiver = _new_cache()
    _install_request(source, base=10.0)
    contradictory_state = list(source.state)
    contradictory_state[7] = source.comp.cur_kv
    contradictory_state[8] = source.comp.cur_score

    with pytest.raises(
        ValueError,
        match="fixed Mia compressor rollback tail must be empty",
    ):
        receiver.state = contradictory_state

    assert len(receiver.window) == 0
    assert len(receiver.compressed) == 0
    assert len(receiver.index_compressed) == 0


@pytest.mark.parametrize(
    ("compress_ratio", "cur_kv_index", "cur_score_index"),
    ((0, 3, 4), (128, 10, 11)),
    ids=("ratio0-attention", "ratio128-indexer"),
)
def test_nvfp4_state_rejects_frontier_for_inactive_compressor_lane(
    compress_ratio: int,
    cur_kv_index: int,
    cur_score_index: int,
) -> None:
    kwargs = {"max_batch_tokens": 8}
    if compress_ratio:
        kwargs["capacity_tokens"] = 256
    cache = DeepseekV4NVFP4Cache(
        window_size=8,
        compress_ratio=compress_ratio,
        head_dim=512,
        rollback_capacity=8,
        **kwargs,
    )
    contradictory_state = list(cache.state)
    contradictory_state[cur_kv_index] = mx.zeros((1, 1, 512), dtype=mx.float32)
    contradictory_state[cur_score_index] = mx.zeros((1, 1, 512), dtype=mx.float32)

    with pytest.raises(
        ValueError,
        match="inactive Mia compressor frontier must be empty",
    ):
        cache.state = contradictory_state


def test_ratio0_state_rejects_inactive_compressed_rows() -> None:
    cache = _new_cache(compress_ratio=0)
    contradictory_state = list(cache.state)
    contradictory_state[2] = mx.zeros((1, 1, 432), dtype=mx.uint8)

    with pytest.raises(
        ValueError,
        match="non-compressed Mia cache cannot restore compressed rows",
    ):
        cache.state = contradictory_state


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_nvfp4_v3_snapshot_round_trips_every_installed_layer_geometry(
    compress_ratio: int,
) -> None:
    cache = _new_cache(compress_ratio=compress_ratio)
    snapshot = snapshot_cache([cache])
    mx.eval(snapshot.states)

    assert len(snapshot.states[0]) == 20
    assert snapshot.states[0][0] == cache._STATE_VERSION
    assert len(snapshot.meta_states[0]) == 9
    assert snapshot.meta_states[0][0] == cache._META_VERSION
    restore_cache([cache], snapshot)


@pytest.mark.parametrize(
    ("compress_ratio", "rows", "max_batch_tokens"),
    ((0, 7, 8), (4, 7, 8), (128, 129, 136)),
    ids=("ratio0", "ratio4", "ratio128"),
)
def test_nvfp4_state_none_resets_frontiers_without_replacing_arenas(
    compress_ratio: int,
    rows: int,
    max_batch_tokens: int,
) -> None:
    cache = _new_cache(
        compress_ratio=compress_ratio,
        max_batch_tokens=max_batch_tokens,
    )
    _install_request(cache, base=10.0, rows=rows)
    owner_ids = _fixed_owner_ids(cache)

    cache.state = None

    assert cache.offset == 0
    assert cache.window_start == 0
    assert len(cache.window) == 0
    assert len(cache.compressed) == 0
    assert cache.n_index_compressed == 0
    for lane in (cache.comp, cache.index_comp):
        assert lane.n_emitted == 0
        assert lane.cur_kv is None
        assert lane.cur_score is None
        assert lane.prev_kv is None
        assert lane.prev_score is None
        assert lane.tail_kv is None
        assert lane.tail_score is None
        if hasattr(lane, "_journal_end"):
            assert lane._journal_end == 0
            assert lane._journal_length == 0
    assert owner_ids == _fixed_owner_ids(cache)
