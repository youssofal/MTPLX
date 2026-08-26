from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import (  # noqa: E402
    FixedMiaNVFP4Window,
    FixedMiaNVFP4WindowRecords,
    PagedMiaNVFP4Records,
)
import mtplx.deepseek_v4_mia_engine as mia_engine  # noqa: E402
from mtplx.kernels import deepseek_v4_nvfp4_mla as mla  # noqa: E402
from mtplx.models import deepseek_v4 as target_model  # noqa: E402
from mtplx.models import deepseek_v4_dspark as draft_model  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_fixed_window_descriptor_separates_logical_wrap_from_physical_padding():
    window = FixedMiaNVFP4Window(capacity_rows=8_416, block_size=64)

    descriptor = window.paged_records(8_400, 8_432)
    again = window.paged_records(8_416, 8_417)

    assert again is descriptor
    assert descriptor.pages is window._pages
    assert descriptor.block_table is window._pool.block_table
    assert descriptor.capacity == 8_416
    assert descriptor.block_size == 64
    assert descriptor.physical_rows == 8_448
    assert descriptor.length == 1

    replacement_pages = mx.zeros((132, 64, 432), dtype=mx.uint8)
    replacement_table = mx.arange(131, -1, -1, dtype=mx.int32)
    window.replace_state((replacement_pages, replacement_table, 8_414, 8_416))
    assert window._paged_records is descriptor
    assert descriptor.pages is window._pages
    assert descriptor.block_table is window._pool.block_table
    assert descriptor.length == 2


@pytest.mark.parametrize(
    "factory",
    [mla._kernel.__wrapped__, mla._prefill_nax_mg16_kernel.__wrapped__],
)
def test_target_mla_source_maps_physical_pages_with_logical_capacity(
    monkeypatch,
    factory,
):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mla.mx.fast, "metal_kernel", capture)
    factory(
        mla._ROUTE_WINDOW,
        1,
        1,
        window_paged=True,
        query_token_major=True,
    )

    assert "paged_window_token_major_v3" in captured["name"]
    assert captured["input_names"][1:6] == [
        "window_records",
        "window_block_table",
        "window_start",
        "window_capacity",
        "window_block_size",
    ]
    assert "constant constexpr bool MTPLX_WINDOW_PAGED = true" in captured["header"]
    assert (
        "constant constexpr bool MTPLX_QUERY_TOKEN_MAJOR = true"
        in captured["header"]
    )
    source = captured["source"]
    assert "absolute_position % size_t(window_capacity)" in source
    assert "window_block_table[logical_block]" in source
    assert "absolute_position % size_t(n_window_records)" not in source
    assert "batch * query_count + query_row" in source


def test_checked_and_exact_mla_kernel_names_cannot_collide(monkeypatch):
    names = []

    def capture(**kwargs):
        names.append(kwargs["name"])
        return object()

    monkeypatch.setattr(mla.mx.fast, "metal_kernel", capture)
    mla._kernel.__wrapped__(mla._ROUTE_WINDOW, 1, 1)
    mla._kernel.__wrapped__(
        mla._ROUTE_WINDOW,
        1,
        1,
        window_paged=True,
        query_token_major=True,
    )

    assert names[0] != names[1]
    assert "contiguous_window_bhmd_v3" in names[0]
    assert "paged_window_token_major_v3" in names[1]


@pytest.mark.parametrize(
    "factory",
    [mla._kernel.__wrapped__, mla._prefill_nax_mg16_kernel.__wrapped__],
)
def test_mla_kernel_names_seal_selected_width_and_block_size(
    monkeypatch,
    factory,
):
    names = []

    def capture(**kwargs):
        names.append(kwargs["name"])
        return object()

    monkeypatch.setattr(mla.mx.fast, "metal_kernel", capture)
    factory(
        mla._ROUTE_INDEXED_PAGED,
        512,
        64,
        window_paged=True,
        query_token_major=True,
    )
    factory(
        mla._ROUTE_INDEXED_PAGED,
        256,
        32,
        window_paged=True,
        query_token_major=True,
    )

    assert names[0] != names[1]
    assert "sw512_bs64" in names[0]
    assert "sw256_bs32" in names[1]


@pytest.mark.parametrize(
    "runner",
    [
        mla._run_installed_window_nvfp4_sparse_mla,
        mla._run_installed_window_nvfp4_prefill_mla,
    ],
)
def test_installed_target_mla_forwards_pages_without_slice_or_transpose(runner):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return (mx.zeros(kwargs["output_shapes"][0], dtype=mx.bfloat16),)

    workspace = SimpleNamespace(
        dummy_record=mx.zeros((1, 1, 432), dtype=mx.uint8),
        dummy_block_table=mx.zeros((1,), dtype=mx.int32),
        indices=lambda rows: mx.zeros((1, rows, 1), dtype=mx.int32),
        lengths=lambda rows: mx.zeros((1, rows), dtype=mx.int32),
    )
    descriptor = FixedMiaNVFP4WindowRecords(
        pages=mx.zeros((132, 64, 432), dtype=mx.uint8),
        block_table=mx.arange(132, dtype=mx.int32),
        length=129,
        block_size=64,
        capacity=8_416,
    )
    queries = mx.zeros((1, 2, 64, 512), dtype=mx.bfloat16)
    output = runner(
        queries,
        descriptor,
        8_287,
        mx.array([8_415, 8_416], dtype=mx.int32),
        None,
        None,
        None,
        mx.zeros((64,), dtype=mx.float32),
        512**-0.5,
        workspace=workspace,
        kernel=kernel,
    )

    inputs = calls[0]["inputs"]
    assert inputs[0] is queries
    assert inputs[1] is descriptor.pages
    assert inputs[2] is descriptor.block_table
    assert inputs[3:6] == [8_287, 8_416, 64]
    assert inputs[14] == 129
    assert tuple(output.shape) == (1, 2, 64, 512)


def test_paged_prefill_checked_adapter_forwards_one_lengths_operand(monkeypatch):
    captured = []

    def storage(*args, **kwargs):
        captured.append((args, kwargs))
        return object()

    monkeypatch.setattr(mla, "_run_nvfp4_prefill_mla_storage", storage)
    compressed = PagedMiaNVFP4Records(
        records=mx.zeros((2, 4, 432), dtype=mx.uint8),
        block_table=mx.arange(2, dtype=mx.int32),
        length=5,
        block_size=4,
    )
    lengths = mx.ones((1, 1), dtype=mx.int32)
    sinks = mx.zeros((64,), dtype=mx.float32)
    scale = 512**-0.5
    workspace = SimpleNamespace(dummy_block_table=mx.zeros((1,), dtype=mx.int32))

    mla._run_paged_nvfp4_prefill_mla(
        mx.zeros((1, 64, 1, 512), dtype=mx.bfloat16),
        mx.zeros((1, 1, 432), dtype=mx.uint8),
        0,
        mx.zeros((1,), dtype=mx.int32),
        compressed,
        None,
        lengths,
        sinks,
        scale,
        workspace=workspace,
    )

    args, kwargs = captured[0]
    assert len(args) == 13
    assert args[10] is lengths
    assert args[11] is sinks
    assert args[12] == scale
    assert kwargs == {"workspace": workspace}


def test_all_installed_target_factories_bind_paged_token_major_variants():
    decode = inspect.getsource(mla.install_nvfp4_sparse_mla)
    prefill = inspect.getsource(mla.install_nvfp4_prefill_mla)
    dspark = inspect.getsource(mla.install_dspark_k5_nvfp4_mla)

    assert "window_paged=True" in decode
    assert "query_token_major=True" in decode
    assert "window_paged=True" in prefill
    assert "query_token_major=True" in prefill
    assert "query_token_major=True" in dspark
    assert "window_paged=True" not in dspark


def test_exact_model_routes_keep_mla_input_and_output_token_major():
    target_qkv = inspect.getsource(
        target_model.DeepseekV4Attention._mia_cached_qkv_records
    )
    target_finish = inspect.getsource(
        target_model.DeepseekV4Attention._mia_finish_cached
    )
    draft_k5 = inspect.getsource(draft_model.DeepseekV4DSparkAttention._run_k5)

    assert "query.transpose" not in target_qkv
    assert "output.transpose" not in target_finish
    assert "query.transpose" not in draft_k5
    assert "output.transpose" not in draft_k5
    assert "query_count = int(queries.shape[1])" in inspect.getsource(
        target_model.DeepseekV4Attention._mia_cached_attention_ratio0
    )


def test_engine_seals_fixed_descriptor_and_token_major_route():
    arena = inspect.getsource(mia_engine.MiaTargetCacheArena.__init__)
    build = inspect.getsource(mia_engine.build_mia_engine_plan)

    assert "FixedMiaNVFP4WindowRecords" in arena
    assert "expected_window_capacity" in arena
    assert "physical_rows" in arena
    assert 'getattr(cache.compressed, "block_size", 0)' in arena
    assert "256 // ratio" in arena
    assert '!= "_trim_fixed_window"' in arena
    assert '"target_swa_stock432_physical_pages"' in build
    assert '"target_fixed_swa_paged_descriptor_8416"' in build
    assert build.count('"BMHD"') >= 4
