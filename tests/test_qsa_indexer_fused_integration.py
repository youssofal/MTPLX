"""Focused integration gates for the opt-in fused QSA selector.

These fixtures instantiate only the tiny indexer and synthetic cache state;
they never load checkpoint weights.  The lower-level Metal selector has its
own exhaustive numerical gate.  This module pins the v2.10 QSAIndexer wiring:
the eager kill-switch, output-mode routing, backing-capacity/logical-frontier
separation, stable dense output capacity, sparse engage/transition boundaries,
and query-row chunk reassembly.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import mtplx.kernels.qsa_indexer_compile as compile_module
import mtplx.kernels.qsa_indexer_prepare as prepare_module
import mtplx.kernels.qsa_indexer_select as selector_module
from mtplx.attention_context import attention_phase
from mtplx.models.qwen4_exp import (
    QSACache,
    QSAIndexer,
    TextArgs,
    _qsa_large_prefill_enabled,
    _qsa_prefill_flash_attention_enabled,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="fused QSA integration requires the Metal GPU",
)


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def indexer() -> QSAIndexer:
    mx.random.seed(20260829)
    module = QSAIndexer(_tiny_args())
    # Keep the post-norm queries and pooled keys on the half-precision path;
    # float32 eligibility is intentionally fenced by the NAX host predicate.
    module.q_layernorm.weight = module.q_layernorm.weight.astype(mx.float16)
    module.k_layernorm.weight = module.k_layernorm.weight.astype(mx.float16)
    mx.eval(module.parameters())
    return module


def _inputs(module: QSAIndexer, rows: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    hidden_size = int(module.index_qk_proj.weight.shape[1])
    qk_width = (module.n_heads + module.kv_heads) * module.head_dim
    hidden = (mx.random.normal((1, rows, hidden_size)) * 0.2).astype(mx.float16)
    qk_rows = (mx.random.normal((1, rows, qk_width)) * 0.2).astype(mx.float16)
    mx.eval(hidden, qk_rows)
    return hidden, qk_rows


def _configure_lanes(
    monkeypatch,
    *,
    fused: bool,
    compiled: bool = False,
    mtp_precompute: bool = False,
    flash: bool = False,
    rows: bool = False,
    decode_gather: bool = False,
) -> None:
    monkeypatch.setenv("MTPLX_FUSED_QSA_INDEXER", "1" if fused else "0")
    monkeypatch.setenv("MTPLX_COMPILED_QSA_INDEXER", "1" if compiled else "0")
    monkeypatch.setenv("MTPLX_QSA_MTP_PRECOMPUTE", "1" if mtp_precompute else "0")
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1" if flash else "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1" if rows else "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "1" if decode_gather else "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "8")
    monkeypatch.setenv("MTPLX_QSA_FLASH_MIN_CONTEXT", "0")


def _prime_cache(
    module: QSAIndexer,
    monkeypatch,
    tokens: int,
    *,
    seed: int = 11,
) -> QSACache:
    """Build the indexer streams, then mirror Attention's KV offset advance."""

    _configure_lanes(monkeypatch, fused=False)
    hidden, qk_rows = _inputs(module, tokens, seed)
    cache = QSACache(module.ratio)
    module(hidden, 0, cache, qk_rows=qk_rows)
    leaves = [leaf for leaf in (cache.raw_keys, cache.pooled) if leaf is not None]
    mx.eval(*leaves)
    cache.kv.offset = tokens
    return cache


def _assert_array_equal(actual: mx.array, expected: mx.array) -> None:
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    assert bool(mx.array_equal(actual, expected).item())


def _assert_output_equal(actual, expected) -> None:
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            _assert_array_equal(actual_leaf, expected_leaf)
        return
    assert isinstance(expected, mx.array)
    _assert_array_equal(actual, expected)


def _assert_model_output_equal(actual, expected) -> None:
    """Compare the public QSASelection tree, including route labels."""

    if isinstance(actual, mx.array):
        assert isinstance(expected, mx.array)
        _assert_array_equal(actual, expected)
        return
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            _assert_model_output_equal(actual_leaf, expected_leaf)
        return
    assert actual == expected


def _unexpected_helper(name: str):
    def unexpected(*_args, **_kwargs):
        raise AssertionError(f"unexpected fused {name} helper dispatch")

    return unexpected


def test_cache_capacity_reservation_defers_and_preserves_active_prefix():
    cache = QSACache(compress_ratio=2)
    cache.reserve_indexer_capacity(raw_capacity=512, pooled_capacity=256)
    assert cache.raw_keys is None
    assert cache.pooled is None

    raw = mx.arange(3 * 8, dtype=mx.float16).reshape(1, 3, 8)
    pooled = mx.arange(2 * 8, dtype=mx.float16).reshape(1, 2, 8)
    cache.write_raw(raw)
    cache.write_pooled(pooled, 0, 2)
    assert tuple(cache.raw_keys.shape) == (1, 512, 8)
    assert tuple(cache.pooled.shape) == (1, 256, 8)

    cache.reserve_indexer_capacity(raw_capacity=1024, pooled_capacity=512)
    assert tuple(cache.raw_keys.shape) == (1, 1024, 8)
    assert tuple(cache.pooled.shape) == (1, 512, 8)
    _assert_array_equal(cache.raw_keys[:, :3], raw)
    _assert_array_equal(cache.pooled[:, :2], pooled)


def test_env_off_is_an_exact_eager_kill_switch(indexer, monkeypatch):
    prefix = 12
    cache_default = _prime_cache(indexer, monkeypatch, prefix)
    cache_zero = _prime_cache(indexer, monkeypatch, prefix)
    hidden, qk_rows = _inputs(indexer, 3, 21)

    for name in (
        "qsa_indexer_select_blocks_metal",
        "qsa_indexer_select_dense_mask_metal",
        "qsa_indexer_select_row_tokens_metal",
    ):
        monkeypatch.setattr(selector_module, name, _unexpected_helper(name))
    for name in (
        "qsa_indexer_prepare_queries_metal",
        "qsa_indexer_pool_keys_metal",
    ):
        monkeypatch.setattr(prepare_module, name, _unexpected_helper(name))
    for name in ("select_hidden", "select_qk_rows"):
        monkeypatch.setattr(
            compile_module.QSACompiledIndexerCore,
            name,
            _unexpected_helper(f"compiled.{name}"),
        )

    eager_calls = []
    original_eager = QSAIndexer._select_eager

    def counted_eager(self, *args, **kwargs):
        # args = (q, pos_start, cache, pooled, total): the merged signature
        # threads the cache so _select_eager reads the fp32 pooled mirror.
        eager_calls.append((int(args[0].shape[1]), int(args[3].shape[1])))
        return original_eager(self, *args, **kwargs)

    monkeypatch.setattr(QSAIndexer, "_select_eager", counted_eager)
    _configure_lanes(monkeypatch, fused=False)
    monkeypatch.delenv("MTPLX_FUSED_QSA_INDEXER", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_QSA_INDEXER", raising=False)
    default_out = indexer(hidden, prefix, cache_default, qk_rows=qk_rows)

    monkeypatch.setenv("MTPLX_FUSED_QSA_INDEXER", "0")
    monkeypatch.setenv("MTPLX_COMPILED_QSA_INDEXER", "0")
    zero_out = indexer(hidden, prefix, cache_zero, qk_rows=qk_rows)

    assert len(eager_calls) == 2
    assert isinstance(default_out, mx.array)
    assert isinstance(zero_out, mx.array)
    _assert_array_equal(default_out, zero_out)


def test_env_on_dense_routes_full_backing_and_matches_eager(indexer, monkeypatch):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=31)
    cache_fused = _prime_cache(indexer, monkeypatch, prefix, seed=31)
    hidden, qk_rows = _inputs(indexer, 3, 32)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, mx.array)

    calls = []
    prepare_calls = []
    original_dense = selector_module.qsa_indexer_select_dense_mask_metal
    original_query = prepare_module.qsa_indexer_prepare_queries_metal
    original_pool = prepare_module.qsa_indexer_pool_keys_metal

    def record_dense(q, pooled, **kwargs):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "pooled_shape": tuple(pooled.shape),
                "kwargs": dict(kwargs),
            }
        )
        return original_dense(q, pooled, **kwargs)

    def record_query(raw_q, weight, inv_freq, **kwargs):
        prepare_calls.append(("query", tuple(raw_q.shape), dict(kwargs)))
        return original_query(raw_q, weight, inv_freq, **kwargs)

    def record_pool(raw_keys, weight, inv_freq, **kwargs):
        prepare_calls.append(("pool", tuple(raw_keys.shape), dict(kwargs)))
        return original_pool(raw_keys, weight, inv_freq, **kwargs)

    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_blocks_metal",
        _unexpected_helper("blocks"),
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_row_tokens_metal",
        _unexpected_helper("row_tokens"),
    )
    monkeypatch.setattr(
        selector_module, "qsa_indexer_select_dense_mask_metal", record_dense
    )
    monkeypatch.setattr(
        prepare_module, "qsa_indexer_prepare_queries_metal", record_query
    )
    monkeypatch.setattr(prepare_module, "qsa_indexer_pool_keys_metal", record_pool)

    _configure_lanes(monkeypatch, fused=True)
    actual = indexer(hidden, prefix, cache_fused, qk_rows=qk_rows)
    assert isinstance(actual, mx.array)
    _assert_array_equal(actual, expected)

    assert prepare_calls == [
        (
            "query",
            (1, 3, indexer.n_heads, indexer.head_dim),
            {
                "pos_start": prefix,
                "eps": indexer.rms_norm_eps,
                "attention_scaling": indexer._rope_attention_scaling,
            },
        ),
        (
            "pool",
            (1, indexer.ratio, indexer.head_dim),
            {
                "block_start": prefix // indexer.ratio,
                "compress_ratio": indexer.ratio,
                "eps": indexer.rms_norm_eps,
                "attention_scaling": indexer._rope_attention_scaling,
            },
        ),
    ]

    assert len(calls) == 1
    call = calls[0]
    total = prefix + int(hidden.shape[1])
    logical = total // indexer.ratio
    assert call["q_shape"] == (1, 3, indexer.n_heads, indexer.head_dim)
    assert call["pooled_shape"] == tuple(cache_fused.pooled.shape)
    assert call["pooled_shape"][1] > logical
    assert call["kwargs"] == {
        "pos_start": prefix,
        "total_tokens": total,
        "block_topk": indexer.block_topk,
        "compress_ratio": indexer.ratio,
        "logical_blocks": logical,
        "output_total_tokens": (call["pooled_shape"][1] + 1) * indexer.ratio,
    }
    assert tuple(actual.shape) == (1, 1, 3, total)
    _assert_array_equal(
        cache_fused.raw_keys[:, :total],
        cache_eager.raw_keys[:, :total],
    )
    _assert_array_equal(
        cache_fused.pooled[:, :logical],
        cache_eager.pooled[:, :logical],
    )


def test_dense_capacity_is_reused_across_consecutive_lengths(indexer, monkeypatch):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=35)
    cache_fused = _prime_cache(indexer, monkeypatch, prefix, seed=35)
    steps = [_inputs(indexer, 1, seed) for seed in (36, 37)]

    _configure_lanes(monkeypatch, fused=False)
    expected = []
    position = prefix
    for hidden, qk_rows in steps:
        mask = indexer(hidden, position, cache_eager, qk_rows=qk_rows)
        assert isinstance(mask, mx.array)
        mx.eval(mask)
        expected.append(mask)
        position += 1
        cache_eager.kv.offset = position

    calls = []
    original_dense = selector_module.qsa_indexer_select_dense_mask_metal

    def record_dense(q, pooled, **kwargs):
        calls.append((tuple(pooled.shape), dict(kwargs)))
        return original_dense(q, pooled, **kwargs)

    monkeypatch.setattr(
        selector_module, "qsa_indexer_select_dense_mask_metal", record_dense
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_blocks_metal",
        _unexpected_helper("blocks"),
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_row_tokens_metal",
        _unexpected_helper("row_tokens"),
    )

    _configure_lanes(monkeypatch, fused=True)
    actual = []
    position = prefix
    for hidden, qk_rows in steps:
        mask = indexer(hidden, position, cache_fused, qk_rows=qk_rows)
        assert isinstance(mask, mx.array)
        mx.eval(mask)
        actual.append(mask)
        position += 1
        cache_fused.kv.offset = position

    assert len(calls) == 2
    capacities = [kwargs["output_total_tokens"] for _, kwargs in calls]
    totals = [kwargs["total_tokens"] for _, kwargs in calls]
    backing_shapes = [shape for shape, _ in calls]
    assert totals == [prefix + 1, prefix + 2]
    assert backing_shapes[0] == backing_shapes[1]
    assert capacities == [
        (backing_shapes[0][1] + 1) * indexer.ratio,
        (backing_shapes[0][1] + 1) * indexer.ratio,
    ]
    assert capacities[0] > totals[-1]

    for total, actual_mask, expected_mask in zip(totals, actual, expected, strict=True):
        assert tuple(actual_mask.shape) == (1, 1, 1, total)
        _assert_array_equal(actual_mask, expected_mask)


def test_env_on_flash_routes_blocks_and_matches_eager(indexer, monkeypatch):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=41)
    cache_fused = _prime_cache(indexer, monkeypatch, prefix, seed=41)
    hidden, qk_rows = _inputs(indexer, 1, 42)

    _configure_lanes(monkeypatch, fused=False, flash=True)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, tuple) and expected[0] == "flash"

    calls = []
    original_blocks = selector_module.qsa_indexer_select_blocks_metal

    def record_blocks(q, pooled, **kwargs):
        result = original_blocks(q, pooled, **kwargs)
        calls.append((tuple(pooled.shape), dict(kwargs), result))
        return result

    monkeypatch.setattr(
        selector_module, "qsa_indexer_select_blocks_metal", record_blocks
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_dense_mask_metal",
        _unexpected_helper("dense_mask"),
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_row_tokens_metal",
        _unexpected_helper("row_tokens"),
    )

    _configure_lanes(monkeypatch, fused=True, flash=True)
    actual = indexer(hidden, prefix, cache_fused, qk_rows=qk_rows)
    assert isinstance(actual, tuple) and actual[0] == "flash"
    assert actual[2] == expected[2]
    _assert_array_equal(actual[1], expected[1])

    assert len(calls) == 1
    pooled_shape, kwargs, helper_result = calls[0]
    total = prefix + 1
    logical = total // indexer.ratio
    assert pooled_shape == tuple(cache_fused.pooled.shape)
    assert pooled_shape[1] > logical > indexer.block_topk
    assert kwargs["logical_blocks"] == logical
    assert kwargs["total_tokens"] == total
    _assert_array_equal(actual[1], helper_result[0][0])
    mx.eval(helper_result[1])
    assert helper_result[1].tolist() == [[True] * indexer.block_topk]
    assert tuple(actual[1].shape) == (indexer.block_topk,)
    assert actual[1].dtype == mx.int32


def test_env_on_rows_routes_tokens_and_preserves_exact_selection(indexer, monkeypatch):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=51)
    cache_fused = _prime_cache(indexer, monkeypatch, prefix, seed=51)
    hidden, qk_rows = _inputs(indexer, 4, 52)

    _configure_lanes(monkeypatch, fused=False, rows=True)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, tuple) and expected[0] == "gather_rows"

    calls = []
    original_rows = selector_module.qsa_indexer_select_row_tokens_metal

    def record_rows(q, pooled, **kwargs):
        result = original_rows(q, pooled, **kwargs)
        calls.append((tuple(pooled.shape), dict(kwargs), result))
        return result

    monkeypatch.setattr(
        selector_module, "qsa_indexer_select_row_tokens_metal", record_rows
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_blocks_metal",
        _unexpected_helper("blocks"),
    )
    monkeypatch.setattr(
        selector_module,
        "qsa_indexer_select_dense_mask_metal",
        _unexpected_helper("dense_mask"),
    )

    _configure_lanes(monkeypatch, fused=True, rows=True)
    actual = indexer(hidden, prefix, cache_fused, qk_rows=qk_rows)
    assert isinstance(actual, tuple) and actual[0] == "gather_rows"

    assert len(calls) == 1
    pooled_shape, kwargs, helper_result = calls[0]
    total = prefix + int(hidden.shape[1])
    logical = total // indexer.ratio
    assert pooled_shape == tuple(cache_fused.pooled.shape)
    assert pooled_shape[1] > logical > indexer.block_topk
    assert kwargs["logical_blocks"] == logical
    assert kwargs["total_tokens"] == total

    # QSAIndexer forwards the helper's token-index contract without any
    # reordering or validity reconstruction.
    _assert_array_equal(actual[1], helper_result[0])
    _assert_array_equal(actual[2], helper_result[1])
    assert tuple(actual[1].shape) == (
        4,
        indexer.block_topk * indexer.ratio + indexer.ratio,
    )
    assert actual[1].dtype == mx.int32
    assert actual[2].dtype == mx.bool_

    # The helper reproduces the stock eager argpartition epilogue exactly.
    _assert_array_equal(actual[1], expected[1])
    _assert_array_equal(actual[2], expected[2])


def test_rows_transition_with_visible_blocks_below_topk_is_exact(indexer, monkeypatch):
    # At qpos=6, only three complete blocks are visible even though K=4.
    # The five-row call ends at T=11 (five complete blocks), so the overall
    # call is sparse while its first row needs one invalid selected filler.
    prefix = 6
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=55)
    cache_fused = _prime_cache(indexer, monkeypatch, prefix, seed=55)
    hidden, qk_rows = _inputs(indexer, 5, 56)

    _configure_lanes(monkeypatch, fused=False, rows=True)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, tuple) and expected[0] == "gather_rows"

    _configure_lanes(monkeypatch, fused=True, rows=True)
    actual = indexer(hidden, prefix, cache_fused, qk_rows=qk_rows)
    assert isinstance(actual, tuple) and actual[0] == "gather_rows"
    _assert_array_equal(actual[1], expected[1])
    _assert_array_equal(actual[2], expected[2])

    mx.eval(actual[1], actual[2])
    first_ids = actual[1].tolist()[0]
    first_valid = actual[2].tolist()[0]
    # MLX's N>K partition puts its one selected invalid filler first, then
    # the three valid winners in ascending adjusted-score order. The invalid
    # filler and invalid tail position are both re-pointed to token zero.
    assert first_valid == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert first_ids[:2] == [0, 0]
    assert sorted(first_ids[2:8]) == [0, 1, 2, 3, 4, 5]
    assert first_ids[8:] == [6, 0]


@pytest.mark.parametrize(
    "total,expect_sparse",
    [(8, False), (9, False), (10, True)],
)
def test_dense_sparse_short_circuit_boundary(
    indexer, monkeypatch, total: int, expect_sparse: bool
):
    prefix = total - 1
    cache = _prime_cache(indexer, monkeypatch, prefix, seed=60 + total)
    hidden, qk_rows = _inputs(indexer, 1, 70 + total)
    _configure_lanes(monkeypatch, fused=True)

    fused_calls = []

    def fake_supported(_self, _q, _pooled):
        return True

    def fake_fused(
        _self,
        q,
        pos_start,
        pooled_backing,
        logical_blocks,
        total_tokens,
        mode,
    ):
        fused_calls.append(
            (
                pos_start,
                tuple(pooled_backing.shape),
                logical_blocks,
                total_tokens,
                mode,
            )
        )
        return mx.ones((1, 1, int(q.shape[1]), total_tokens), dtype=mx.bool_)

    def eager_forbidden(*_args, **_kwargs):
        raise AssertionError("sparse boundary unexpectedly routed to eager")

    monkeypatch.setattr(QSAIndexer, "_fused_selector_supported", fake_supported)
    monkeypatch.setattr(QSAIndexer, "_select_fused", fake_fused)
    monkeypatch.setattr(QSAIndexer, "_select_eager", eager_forbidden)

    actual = indexer(hidden, prefix, cache, qk_rows=qk_rows)
    if not expect_sparse:
        assert actual is None
        assert fused_calls == []
        return

    assert isinstance(actual, mx.array)
    assert fused_calls == [
        (
            prefix,
            tuple(cache.pooled.shape),
            total // indexer.ratio,
            total,
            "dense_mask",
        )
    ]
    _assert_array_equal(
        actual,
        mx.ones((1, 1, 1, total), dtype=mx.bool_),
    )


@pytest.mark.parametrize("mode", ["blocks", "dense_mask", "row_tokens"])
def test_forced_multi_chunk_matches_one_dispatch_and_ignores_backing_suffix(
    indexer, monkeypatch, mode: str
):
    rows = 5
    backing_blocks = 13
    logical_blocks = 9
    total = logical_blocks * indexer.ratio + 1
    pos_start = total - rows
    mx.random.seed(81)
    q = (mx.random.normal((1, rows, indexer.n_heads, indexer.head_dim)) * 0.2).astype(
        mx.float16
    )
    pooled = (mx.random.normal((1, backing_blocks, indexer.head_dim)) * 0.2).astype(
        mx.float16
    )
    changed_suffix = mx.full(
        (1, backing_blocks - logical_blocks, indexer.head_dim),
        1000.0,
        dtype=mx.float16,
    )
    pooled_with_changed_suffix = mx.concatenate(
        [pooled[:, :logical_blocks], changed_suffix], axis=1
    )

    kwargs = {
        "pos_start": pos_start,
        "pooled_backing": pooled,
        "logical_blocks": logical_blocks,
        "total": total,
        "mode": mode,
    }
    monkeypatch.setattr(QSAIndexer, "_fused_score_scratch_bytes", 1 << 30)
    one_dispatch = indexer._select_fused(q, **kwargs)
    suffix_changed = indexer._select_fused(
        q,
        **{**kwargs, "pooled_backing": pooled_with_changed_suffix},
    )

    # 8*N bytes permits exactly two rows because each scratch row is 4*N.
    monkeypatch.setattr(
        QSAIndexer,
        "_fused_score_scratch_bytes",
        8 * backing_blocks,
    )
    assert indexer._fused_query_chunk_rows(rows, backing_blocks) == 2
    chunked = indexer._select_fused(q, **kwargs)

    _assert_output_equal(chunked, one_dispatch)
    _assert_output_equal(suffix_changed, one_dispatch)
    if mode == "dense_mask":
        assert tuple(chunked.shape) == (1, 1, rows, total)
    else:
        assert int(chunked[0].shape[0]) == rows


@pytest.mark.parametrize(
    ("public_route", "rows", "compiled_mode"),
    [
        ("dense", 3, "dense_mask"),
        ("flash", 1, "blocks"),
        ("gather_rows", 4, "row_tokens"),
    ],
)
def test_compiled_qk_rows_routes_public_modes_and_matches_eager(
    indexer,
    monkeypatch,
    public_route: str,
    rows: int,
    compiled_mode: str,
):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=201 + rows)
    cache_compiled = _prime_cache(indexer, monkeypatch, prefix, seed=201 + rows)
    hidden, qk_rows = _inputs(indexer, rows, 211 + rows)
    lane_kwargs = {
        "flash": public_route == "flash",
        "rows": public_route == "gather_rows",
    }

    _configure_lanes(monkeypatch, fused=False, **lane_kwargs)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    mx.eval(
        *[
            leaf
            for leaf in (cache_eager.raw_keys, cache_eager.pooled)
            if leaf is not None
        ]
    )

    calls = []
    original_qk_rows = compile_module.QSACompiledIndexerCore.select_qk_rows

    def recorded_qk_rows(self, *args, **kwargs):
        result = original_qk_rows(self, *args, **kwargs)
        calls.append((str(kwargs["mode"]), self))
        return result

    monkeypatch.setattr(
        compile_module.QSACompiledIndexerCore,
        "select_qk_rows",
        recorded_qk_rows,
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("selector while compiled core is active"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_eager",
        _unexpected_helper("eager selector while compiled core is active"),
    )

    _configure_lanes(
        monkeypatch,
        fused=True,
        compiled=True,
        **lane_kwargs,
    )
    actual = indexer(hidden, prefix, cache_compiled, qk_rows=qk_rows)
    _assert_model_output_equal(actual, expected)

    assert len(calls) == 1
    mode, core = calls[0]
    assert mode == compiled_mode
    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 1
    assert report["traces"] == report["entry_count"] == 1
    assert report["qk_rows_calls"] == 1
    assert report["hidden_calls"] == 0
    assert report["modes"][compiled_mode] == 1

    total = prefix + rows
    logical = total // indexer.ratio
    _assert_array_equal(
        cache_compiled.raw_keys[:, :total],
        cache_eager.raw_keys[:, :total],
    )
    _assert_array_equal(
        cache_compiled.pooled[:, :logical],
        cache_eager.pooled[:, :logical],
    )


def test_compiled_hidden_route_captures_projection_and_matches_eager(
    indexer,
    monkeypatch,
):
    # The production artifact supplies FP16 (dense MTP) or quantized trunk
    # index projections.  This tiny dense Linear starts in FP32, so cast it to
    # the production floating dtype before checking the hidden-source graph.
    indexer.index_qk_proj.weight = indexer.index_qk_proj.weight.astype(mx.float16)
    mx.eval(indexer.index_qk_proj.weight)

    prefix = 12
    rows = 3
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=231)
    cache_compiled = _prime_cache(indexer, monkeypatch, prefix, seed=231)
    hidden, _qk_rows = _inputs(indexer, rows, 232)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, prefix, cache_eager)

    calls = []
    original_hidden = compile_module.QSACompiledIndexerCore.select_hidden

    def recorded_hidden(self, *args, **kwargs):
        result = original_hidden(self, *args, **kwargs)
        calls.append((str(kwargs["mode"]), self))
        return result

    monkeypatch.setattr(
        compile_module.QSACompiledIndexerCore,
        "select_hidden",
        recorded_hidden,
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("selector while hidden graph is active"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_eager",
        _unexpected_helper("eager selector while hidden graph is active"),
    )

    _configure_lanes(monkeypatch, fused=True, compiled=True)
    actual = indexer(hidden, prefix, cache_compiled)
    _assert_model_output_equal(actual, expected)

    assert len(calls) == 1
    mode, core = calls[0]
    assert mode == "dense_mask"
    report = core.to_dict()
    assert report["hidden_calls"] == 1
    assert report["qk_rows_calls"] == 0
    assert report["compiled_keys"][0]["source"] == "hidden"

    total = prefix + rows
    logical = total // indexer.ratio
    _assert_array_equal(
        cache_compiled.raw_keys[:, :total],
        cache_eager.raw_keys[:, :total],
    )
    _assert_array_equal(
        cache_compiled.pooled[:, :logical],
        cache_eager.pooled[:, :logical],
    )


def test_prefill_dense_equals_sparse_boundary_stays_off_update_only_graph(
    indexer,
    monkeypatch,
):
    prefix = 0
    rows = indexer.block_topk * indexer.ratio
    cache_eager = QSACache(indexer.ratio)
    cache_compiled = QSACache(indexer.ratio)
    hidden, qk_rows = _inputs(indexer, rows, 242)

    _configure_lanes(monkeypatch, fused=False)
    with attention_phase("prefill"):
        expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert expected is None

    monkeypatch.setattr(
        compile_module.QSACompiledIndexerCore,
        "select_qk_rows",
        _unexpected_helper("qk-row update-only graph during prefill"),
    )
    monkeypatch.setattr(
        compile_module.QSACompiledIndexerCore,
        "select_hidden",
        _unexpected_helper("hidden update-only graph during prefill"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("selector in update-only graph"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_eager",
        _unexpected_helper("eager selector in update-only graph"),
    )

    _configure_lanes(monkeypatch, fused=True, compiled=True)
    with attention_phase("prefill"):
        actual = indexer(hidden, prefix, cache_compiled, qk_rows=qk_rows)
    assert actual is None
    assert indexer._compiled_indexer_core is None

    total = prefix + rows
    logical = total // indexer.ratio
    assert tuple(cache_compiled.raw_keys.shape) == (
        1,
        cache_compiled.step,
        indexer.head_dim,
    )
    assert tuple(cache_compiled.pooled.shape) == (
        1,
        cache_compiled.step,
        indexer.head_dim,
    )
    _assert_array_equal(
        cache_compiled.raw_keys[:, :total],
        cache_eager.raw_keys[:, :total],
    )
    _assert_array_equal(
        cache_compiled.pooled[:, :logical],
        cache_eager.pooled[:, :logical],
    )


def test_prefill_crossovers_use_earliest_query_history(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "2")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "32768")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT", "65536")
    rows = 2048

    with attention_phase("prefill"):
        assert not _qsa_large_prefill_enabled(rows, rows + 32767)
        assert _qsa_large_prefill_enabled(rows, rows + 32768)
        assert not _qsa_prefill_flash_attention_enabled(rows, rows + 65535)
        assert _qsa_prefill_flash_attention_enabled(rows, rows + 65536)

    with attention_phase("decode_verify"):
        assert not _qsa_large_prefill_enabled(rows, rows + 65536)
        assert not _qsa_prefill_flash_attention_enabled(rows, rows + 65536)


def test_compiled_unaligned_prefill_buckets_physical_pool_window(
    indexer,
    monkeypatch,
):
    # Tiny production analogue: S=513/r2 has 256 logical complete blocks but
    # the fixed compiled staging window is ceil(513/2)=257 blocks.  The pooled
    # backing must therefore cross the 256 boundary to 512 even though the
    # logical frontier itself still fits in 256.  Production's corresponding
    # boundary is S=1025/r4.
    rows = 513
    cache_eager = QSACache(indexer.ratio)
    cache_compiled = QSACache(indexer.ratio)
    hidden, qk_rows = _inputs(indexer, rows, 246)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, 0, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, mx.array)

    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("standalone selector on compiled boundary prefill"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_eager",
        _unexpected_helper("eager selector on compiled boundary prefill"),
    )
    # This is a lower-level graph-capacity test, not a production route test.
    # Keep its 513-row synthetic window below the matrix-prefill threshold so
    # the dense-mask compiled primitive remains directly exercisable after
    # production prefill gained a measured context crossover.
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "1024")
    _configure_lanes(monkeypatch, fused=True, compiled=True)
    actual = indexer(hidden, 0, cache_compiled, qk_rows=qk_rows)

    _assert_array_equal(actual, expected)
    logical = rows // indexer.ratio
    assert logical == 256
    assert (rows + indexer.ratio - 1) // indexer.ratio == 257
    assert tuple(cache_compiled.raw_keys.shape) == (1, 1024, indexer.head_dim)
    assert tuple(cache_compiled.pooled.shape) == (1, 512, indexer.head_dim)
    assert cache_compiled.pooled_len == logical
    _assert_array_equal(
        cache_compiled.raw_keys[:, :rows],
        cache_eager.raw_keys[:, :rows],
    )
    _assert_array_equal(
        cache_compiled.pooled[:, :logical],
        cache_eager.pooled[:, :logical],
    )
    core = indexer._compiled_indexer_core
    assert core is not None
    report = core.to_dict()
    assert report["buckets"] == {"raw1024:pool512:out1026": 1}
    assert report["compiled_keys"][0]["pooled_shape"] == (
        1,
        512,
        indexer.head_dim,
    )


@pytest.mark.parametrize(
    ("fused", "compiled", "expected_fused_calls"),
    [
        (False, True, 0),
        (True, False, 1),
    ],
)
def test_compiled_and_fused_flags_are_independent_kill_switches(
    indexer,
    monkeypatch,
    fused: bool,
    compiled: bool,
    expected_fused_calls: int,
):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=251)
    cache_candidate = _prime_cache(indexer, monkeypatch, prefix, seed=251)
    hidden, qk_rows = _inputs(indexer, 3, 252)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)

    for name in ("select_hidden", "select_qk_rows"):
        monkeypatch.setattr(
            compile_module.QSACompiledIndexerCore,
            name,
            _unexpected_helper(f"compiled.{name} with only one flag enabled"),
        )
    fused_calls = []
    original_fused = QSAIndexer._select_fused

    def recorded_fused(self, *args, **kwargs):
        fused_calls.append(str(args[-1] if args else kwargs["mode"]))
        return original_fused(self, *args, **kwargs)

    monkeypatch.setattr(QSAIndexer, "_select_fused", recorded_fused)
    _configure_lanes(monkeypatch, fused=fused, compiled=compiled)
    actual = indexer(hidden, prefix, cache_candidate, qk_rows=qk_rows)

    _assert_model_output_equal(actual, expected)
    assert len(fused_calls) == expected_fused_calls
    assert indexer._compiled_indexer_core is None


def test_compiled_decode_gather_experiment_falls_back_to_eager(
    indexer,
    monkeypatch,
):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=261)
    cache_candidate = _prime_cache(indexer, monkeypatch, prefix, seed=261)
    hidden, qk_rows = _inputs(indexer, 1, 262)

    _configure_lanes(monkeypatch, fused=False, decode_gather=True)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    assert isinstance(expected, mx.array)

    for name in ("select_hidden", "select_qk_rows"):
        monkeypatch.setattr(
            compile_module.QSACompiledIndexerCore,
            name,
            _unexpected_helper(f"compiled.{name} on decode-gather fallback"),
        )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("fixed-shape fused selector on decode-gather fallback"),
    )
    eager_calls = []
    original_eager = QSAIndexer._select_eager

    def recorded_eager(self, *args, **kwargs):
        eager_calls.append(int(args[0].shape[1]))
        return original_eager(self, *args, **kwargs)

    monkeypatch.setattr(QSAIndexer, "_select_eager", recorded_eager)
    _configure_lanes(
        monkeypatch,
        fused=True,
        compiled=True,
        decode_gather=True,
    )
    actual = indexer(hidden, prefix, cache_candidate, qk_rows=qk_rows)

    _assert_array_equal(actual, expected)
    assert eager_calls == [1]
    assert indexer._compiled_indexer_core is None


def test_compiled_route_promotes_restored_non_bucket_backings(
    indexer,
    monkeypatch,
):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=271)
    cache_compiled = _prime_cache(indexer, monkeypatch, prefix, seed=271)
    hidden, qk_rows = _inputs(indexer, 1, 272)

    # A restored/additively-grown v2.10 cache can have a 768-entry backing.
    # The compiled integration must preserve its prefix while promoting it to
    # the shared logarithmic graph bucket rather than refusing or retracing on
    # the additive capacity.
    raw_pad = mx.zeros(
        (1, 768 - int(cache_compiled.raw_keys.shape[1]), indexer.head_dim),
        dtype=cache_compiled.raw_keys.dtype,
    )
    pooled_pad = mx.zeros(
        (1, 768 - int(cache_compiled.pooled.shape[1]), indexer.head_dim),
        dtype=cache_compiled.pooled.dtype,
    )
    cache_compiled.raw_keys = mx.concatenate([cache_compiled.raw_keys, raw_pad], axis=1)
    cache_compiled.pooled = mx.concatenate([cache_compiled.pooled, pooled_pad], axis=1)
    cache_compiled._reserved_raw_capacity = 768
    cache_compiled._reserved_pooled_capacity = 768
    mx.eval(cache_compiled.raw_keys, cache_compiled.pooled)
    assert tuple(cache_compiled.raw_keys.shape) == (1, 768, indexer.head_dim)
    assert tuple(cache_compiled.pooled.shape) == (1, 768, indexer.head_dim)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)
    _configure_lanes(monkeypatch, fused=True, compiled=True)
    actual = indexer(hidden, prefix, cache_compiled, qk_rows=qk_rows)

    _assert_model_output_equal(actual, expected)
    assert tuple(cache_compiled.raw_keys.shape) == (1, 1024, indexer.head_dim)
    assert tuple(cache_compiled.pooled.shape) == (1, 1024, indexer.head_dim)
    core = indexer._compiled_indexer_core
    assert core is not None
    report = core.to_dict()
    assert report["capacity_transitions"] == 0
    assert report["buckets"] == {"raw1024:pool1024:out2050": 1}
    assert report["compiled_keys"][0]["raw_shape"] == (1, 1024, indexer.head_dim)
    assert report["compiled_keys"][0]["pooled_shape"] == (
        1,
        1024,
        indexer.head_dim,
    )


def test_compiled_dispatch_error_is_visible_and_never_falls_back(
    indexer,
    monkeypatch,
):
    prefix = 12
    cache = _prime_cache(indexer, monkeypatch, prefix, seed=281)
    hidden, qk_rows = _inputs(indexer, 1, 282)

    def compiled_failure(*_args, **_kwargs):
        raise RuntimeError("intentional compiled QSA dispatch failure")

    monkeypatch.setattr(
        compile_module.QSACompiledIndexerCore,
        "select_qk_rows",
        compiled_failure,
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_fused",
        _unexpected_helper("fused fallback after compiled failure"),
    )
    monkeypatch.setattr(
        QSAIndexer,
        "_select_eager",
        _unexpected_helper("eager fallback after compiled failure"),
    )
    _configure_lanes(monkeypatch, fused=True, compiled=True)

    with pytest.raises(RuntimeError, match="intentional compiled QSA dispatch failure"):
        indexer(hidden, prefix, cache, qk_rows=qk_rows)


def test_compiled_frontier_gap_fails_closed_before_reservation(
    indexer,
    monkeypatch,
):
    prefix = 12
    cache_eager = _prime_cache(indexer, monkeypatch, prefix, seed=291)
    cache_candidate = _prime_cache(indexer, monkeypatch, prefix, seed=291)
    # A restore that advertises no valid pooled blocks while sitting at token
    # 12 needs six blocks repaired.  A one-row compiled signature can stage at
    # most one block, so eligibility must fail before any backing mutation.
    cache_eager.pooled_len = 0
    cache_candidate.pooled_len = 0
    hidden, qk_rows = _inputs(indexer, 1, 292)

    _configure_lanes(monkeypatch, fused=False)
    expected = indexer(hidden, prefix, cache_eager, qk_rows=qk_rows)

    reservations = []
    original_reserve = QSACache.reserve_indexer_capacity

    def recorded_reserve(self, *, raw_capacity, pooled_capacity):
        reservations.append((int(raw_capacity), int(pooled_capacity)))
        return original_reserve(
            self,
            raw_capacity=raw_capacity,
            pooled_capacity=pooled_capacity,
        )

    monkeypatch.setattr(QSACache, "reserve_indexer_capacity", recorded_reserve)
    for name in ("select_hidden", "select_qk_rows"):
        monkeypatch.setattr(
            compile_module.QSACompiledIndexerCore,
            name,
            _unexpected_helper(f"compiled.{name} across an unsupported gap"),
        )

    before_shapes = (
        tuple(cache_candidate.raw_keys.shape),
        tuple(cache_candidate.pooled.shape),
    )
    _configure_lanes(monkeypatch, fused=True, compiled=True)
    actual = indexer(hidden, prefix, cache_candidate, qk_rows=qk_rows)

    _assert_model_output_equal(actual, expected)
    assert reservations == []
    assert indexer._compiled_indexer_core is None
    assert (
        tuple(cache_candidate.raw_keys.shape),
        tuple(cache_candidate.pooled.shape),
    ) == before_shapes


def test_compiled_rollback_frontier_ignores_stale_pooled_suffix(
    indexer,
    monkeypatch,
):
    rollback_offset = 12
    speculative_offset = 16
    hidden_prefix, qk_prefix = _inputs(indexer, speculative_offset, 301)
    cache_stale = QSACache(indexer.ratio)
    cache_clean = QSACache(indexer.ratio)

    _configure_lanes(monkeypatch, fused=False)
    indexer(hidden_prefix, 0, cache_stale, qk_rows=qk_prefix)
    indexer(
        hidden_prefix[:, :rollback_offset],
        0,
        cache_clean,
        qk_rows=qk_prefix[:, :rollback_offset],
    )
    cache_stale.kv.offset = rollback_offset
    cache_clean.kv.offset = rollback_offset
    cache_stale.pooled_len = rollback_offset // indexer.ratio
    assert int(cache_stale.pooled.shape[1]) > cache_stale.pooled_len
    _assert_array_equal(
        cache_stale.raw_keys[:, :rollback_offset],
        cache_clean.raw_keys[:, :rollback_offset],
    )
    _assert_array_equal(
        cache_stale.pooled[:, : cache_stale.pooled_len],
        cache_clean.pooled[:, : cache_clean.pooled_len],
    )

    # Make the rejected pooled rows unmistakably poisonous.  They remain in
    # the fixed backing after rollback, but logical_blocks/pooled_len must make
    # them invisible to both pool repair and exact top-k selection.
    stale_start = cache_stale.pooled_len
    stale_stop = speculative_offset // indexer.ratio
    poison = mx.full(
        (1, stale_stop - stale_start, indexer.head_dim),
        1000.0,
        dtype=cache_stale.pooled.dtype,
    )
    cache_stale.pooled = mx.concatenate(
        [
            cache_stale.pooled[:, :stale_start],
            poison,
            cache_stale.pooled[:, stale_stop:],
        ],
        axis=1,
    )

    hidden_next, qk_next = _inputs(indexer, 1, 302)
    expected = indexer(
        hidden_next,
        rollback_offset,
        cache_clean,
        qk_rows=qk_next,
    )
    _configure_lanes(monkeypatch, fused=True, compiled=True)
    actual = indexer(
        hidden_next,
        rollback_offset,
        cache_stale,
        qk_rows=qk_next,
    )

    _assert_model_output_equal(actual, expected)
    logical = (rollback_offset + 1) // indexer.ratio
    _assert_array_equal(
        cache_stale.raw_keys[:, : rollback_offset + 1],
        cache_clean.raw_keys[:, : rollback_offset + 1],
    )
    _assert_array_equal(
        cache_stale.pooled[:, :logical],
        cache_clean.pooled[:, :logical],
    )
    _assert_array_equal(
        cache_stale.pooled[:, stale_start:stale_stop],
        poison,
    )
    core = indexer._compiled_indexer_core
    assert core is not None
    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 1
    assert report["modes"]["dense_mask"] == 1
