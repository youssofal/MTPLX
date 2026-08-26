import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import (  # noqa: E402
    FixedMiaNVFP4Window,
    MIA_NVFP4_RECORD_BYTES,
    MiaNVFP4Rows,
    PagedMiaNVFP4Rows,
)
from mtplx.deepseek_v4_mia_engine import MiaM6RatioTables  # noqa: E402
from mtplx.deepseek_v4_paged_indexer import (  # noqa: E402
    MiaIndexerWorkspace,
    MiaTopKSelection,
    PagedMiaIndexerRows,
    _run_paged_indexer_records_topk,
    paged_indexer_scores,
    paged_indexer_tiled_scores,
)
from mtplx.attention_context import attention_phase  # noqa: E402
from mtplx.models import deepseek_v4 as deepseek_v4_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4NVFP4Cache,
    Indexer,
)
from mtplx.kernels import deepseek_v4_nvfp4_mla as nvfp4_mla_module  # noqa: E402
from mtplx.kernels.deepseek_v4_nvfp4_mla import (  # noqa: E402
    install_dspark_k5_nvfp4_mla,
    install_nvfp4_sparse_mla,
    nvfp4_prefill_mla,
    nvfp4_sparse_mla,
)


def _exact_latent(rows: int = 2) -> mx.array:
    values = mx.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=mx.bfloat16,
    )
    row = mx.tile(values, 32)
    return mx.broadcast_to(row, (1, rows, 512))


def _rope(rows: int = 2) -> mx.array:
    values = (mx.arange(rows * 64, dtype=mx.float32) - 37.0) / 29.0
    return values.reshape(1, rows, 64).astype(mx.bfloat16)


def _post_rope_row(latent: mx.array, rope: mx.array) -> mx.array:
    return mx.concatenate((latent[..., :448], rope), axis=-1)


def _as_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def _exp2(value: mx.array) -> mx.array:
    """Evaluate the base-2 oracle on MLX builds without ``mx.exp2``."""

    return mx.exp(value * np.log(2.0))


def test_mia_decode_kernel_seals_image_h16_bf16_base2_contract() -> None:
    assert nvfp4_mla_module._DECODE_HEADS_PER_GROUP == 16
    assert nvfp4_mla_module._DECODE_CANDIDATE_TILE == 64
    assert nvfp4_mla_module._DECODE_METAL_PANEL == 32
    assert nvfp4_mla_module._DECODE_MATH_THREADS == 256
    assert nvfp4_mla_module._DECODE_NAX_THREADS == 288
    assert "mtplx_dsv4_device_value_bf16" in nvfp4_mla_module._HEADER
    assert "bfloat probability_bf16" in nvfp4_mla_module._DECODE_NAX_SOURCE
    assert "MTPLX_LOG2E" in nvfp4_mla_module._DECODE_NAX_SOURCE
    assert "fast::exp2" in nvfp4_mla_module._DECODE_NAX_SOURCE
    assert "use_indices" not in nvfp4_mla_module._DECODE_NAX_SOURCE
    assert "use_paged_compressed" not in nvfp4_mla_module._DECODE_NAX_SOURCE


def test_tile64_correction_precedes_the_single_bf16_probability_boundary() -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        interim_probability = 2.0**-9.95
        correction = 2.0**-0.03507537688442211
        rounded_too_early = (
            mx.array(interim_probability, dtype=mx.bfloat16).astype(mx.float32)
            * correction
        ).astype(mx.bfloat16)
        source_order = mx.array(
            interim_probability * correction,
            dtype=mx.bfloat16,
        )
        assert float(rounded_too_early.item()) == 0.0009918212890625
        assert float(source_order.item()) == 0.00098419189453125
        assert float(rounded_too_early.item()) != float(source_order.item())
    finally:
        mx.set_default_device(previous)

    source = nvfp4_mla_module._DECODE_NAX_SOURCE
    assert "second_panel_scores" in source
    assert "float corrected_probability" in source
    assert source.index("float corrected_probability") < source.index(
        "bfloat probability_bf16 = bfloat(corrected_probability)"
    )


def test_tile64_score_and_probability_lifetimes_are_disjoint() -> None:
    first = nvfp4_mla_module._DECODE_FIRST_SCORE_RANGE
    second = nvfp4_mla_module._DECODE_SECOND_SCORE_RANGE
    probabilities = nvfp4_mla_module._DECODE_PROBABILITY_RANGE

    def disjoint(left, right) -> bool:
        return left[1] <= right[0] or right[1] <= left[0]

    assert first == (16_384, 18_432)
    assert second == (0, 2_048)
    assert probabilities == (18_432, 20_480)
    assert disjoint(probabilities, first)
    assert disjoint(probabilities, second)
    assert probabilities[1] <= nvfp4_mla_module._DECODE_NAX_SCRATCH_BYTES
    assert (
        "reinterpret_cast<threadgroup bfloat*>(scratch + 18432)"
        in nvfp4_mla_module._DECODE_NAX_SOURCE
    )


def test_decode_reloads_query_inside_every_tile_before_score_alias() -> None:
    source = nvfp4_mla_module._DECODE_NAX_SOURCE
    tile_loop = source.index("for (uint tile_start = 0u;")
    query_reload = source.index("for (uint index = thread_index;", tile_loop)
    qk_read = source.index("auto q_tile = Q.template", query_reload)
    score_alias = source.index(
        "threadgroup float* second_panel_scores =",
        query_reload,
    )

    assert "q_shared[index] =" not in source[:tile_loop]
    assert tile_loop < query_reload < qk_read < score_alias
    assert "Reload the immutable query at every candidate" in source


def test_installed_mla_launchers_use_only_the_prebound_kernel() -> None:
    names = (
        "_run_installed_window_nvfp4_sparse_mla",
        "_run_installed_indexed_paged_nvfp4_sparse_mla",
        "_run_installed_sequential_paged_nvfp4_sparse_mla",
        "_run_installed_window_nvfp4_prefill_mla",
        "_run_installed_indexed_paged_nvfp4_prefill_mla",
        "_run_installed_sequential_paged_nvfp4_prefill_mla",
    )
    for name in names:
        source = inspect.getsource(getattr(nvfp4_mla_module, name))
        assert "kernel=kernel" in source
        assert "_kernel(" not in source
        assert "selected_width" not in source
        assert "compressed_block_size" not in source
        assert "window_records.block_size" in source
    dspark_source = inspect.getsource(nvfp4_mla_module._run_dspark_k5_nvfp4_mla)
    assert "(output,) = kernel(" in dspark_source
    assert "_kernel(" not in dspark_source
    assert "_dspark_placeholder_inputs" not in dspark_source
    installer_source = inspect.getsource(
        nvfp4_mla_module.install_dspark_k5_nvfp4_mla
    )
    assert "_dspark_placeholder_inputs()" in installer_source
    assert "query_positions=query_positions" in installer_source


def test_mia_stock432_record_uses_the_post_rope_row_for_key_and_value() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    latent = _exact_latent()
    rope = -latent[..., 448:]
    post_rope = _post_rope_row(latent, rope)
    rows = MiaNVFP4Rows()
    rows.append(latent[:, :1], rope[:, :1])
    rows.append(latent[:, 1:], rope[:, 1:])
    key, value = rows.decode()

    assert MIA_NVFP4_RECORD_BYTES == 432
    assert rows.shape == (1, 2, 432)
    assert rows.records.dtype == mx.uint8
    assert rows.nbytes == 2 * 432
    np.testing.assert_array_equal(
        np.array(rows.records[..., 288:304]),
        np.zeros((1, 2, 16), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.array(rows.records[..., 256:288]),
        np.full((1, 2, 32), 0x38, dtype=np.uint8),
    )
    assert int(rows.records[0, 0, 0].item()) == 0x10
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(post_rope))
    np.testing.assert_array_equal(_as_numpy(key), _as_numpy(post_rope))


def test_mia_stock432_owner_replaces_truncates_and_restores_whole_records() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    rows = MiaNVFP4Rows()
    rows.append(_exact_latent(4), _rope(4))
    replacement = -_exact_latent(1)
    replacement_rope = -replacement[..., 448:]
    expected = _post_rope_row(replacement, replacement_rope)
    rows.replace(1, replacement, replacement_rope)
    saved = rows.state

    rows.drop_first(1)
    rows.truncate(2)
    assert rows.shape == (1, 2, 432)
    key, value = rows.decode()
    np.testing.assert_array_equal(
        _as_numpy(value[:, :1]),
        _as_numpy(expected),
    )
    np.testing.assert_array_equal(_as_numpy(key[:, :1, 448:]), _as_numpy(replacement_rope))

    rows.replace_state(saved)
    assert rows.shape == (1, 4, 432)
    restored_key, restored_value = rows.decode(1, 2)
    np.testing.assert_array_equal(
        _as_numpy(restored_value),
        _as_numpy(expected),
    )
    np.testing.assert_array_equal(
        _as_numpy(restored_key[..., 448:]),
        _as_numpy(replacement_rope),
    )


def test_paged_mia_stock432_owner_keeps_fixed_pages_across_writes() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    rows = PagedMiaNVFP4Rows(capacity_rows=8, block_size=4)
    pages = rows.pages
    rows.append(_exact_latent(3), _rope(3))
    rows.append(_exact_latent(2), _rope(2))

    assert rows.pages is pages
    assert rows.shape == (1, 5, 432)
    assert rows.paged_records.records is pages
    assert rows.paged_records.length == 5
    assert rows.paged_records.block_size == 4

    replacement = -_exact_latent(1)
    replacement_rope = -replacement[..., 448:]
    expected = _post_rope_row(replacement, replacement_rope)
    rows.replace(1, replacement, replacement_rope)
    rows.truncate(4)
    _key, value = rows.decode()
    np.testing.assert_array_equal(
        _as_numpy(value[:, 1:2]),
        _as_numpy(expected),
    )
    with pytest.raises(ValueError, match="capacity exceeded"):
        rows.append(_exact_latent(5), _rope(5))


def test_target_cache_owns_the_source_exact_post_rope_key_and_value_row() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    cache = DeepseekV4NVFP4Cache(
        window_size=8,
        compress_ratio=0,
        head_dim=512,
    )
    latent = _exact_latent(3)
    rope = -latent[..., 448:]
    post_rope = _post_rope_row(latent, rope)

    records, start = cache.update_window(latent, rope)
    key, value = cache.window.decode()

    assert start == 0
    assert isinstance(cache.window, MiaNVFP4Rows)
    assert cache.window.mode == "nvfp4_stock432"
    assert cache.window.shape == (1, 3, 432)
    assert records.shape == (1, 3, 432)
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(post_rope))
    np.testing.assert_array_equal(_as_numpy(key), _as_numpy(post_rope))


def test_target_compressed_cache_uses_fixed_stock432_pages() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    cache = DeepseekV4NVFP4Cache(
        window_size=128,
        compress_ratio=4,
        head_dim=512,
        capacity_tokens=33,
    )
    pages = cache.compressed.pages
    cache.compressed.append(_exact_latent(5), _rope(5))
    cache.compressed.append(_exact_latent(2), _rope(2))

    assert isinstance(cache.compressed, PagedMiaNVFP4Rows)
    assert cache.compressed.capacity == 9
    assert cache.compressed.pages is pages
    assert cache.attention_compressed().records is pages
    assert isinstance(cache.index_compressed, PagedMiaIndexerRows)
    assert cache.index_compressed.capacity == 9


@pytest.mark.parametrize("offset", [3, 127, 191])
@pytest.mark.parametrize(
    "owner_type,record_bytes,ratio,block_size",
    [
        (PagedMiaNVFP4Rows, 432, 4, 64),
        (PagedMiaNVFP4Rows, 432, 128, 2),
        (PagedMiaIndexerRows, 132, 4, 64),
    ],
)
def test_m6_scheduled_page_append_matches_current_page_bytes(
    offset: int,
    owner_type,
    record_bytes: int,
    ratio: int,
    block_size: int,
) -> None:
    capacity_tokens = 256
    compressed_capacity = (capacity_tokens + ratio - 1) // ratio
    expected = owner_type(
        capacity_rows=compressed_capacity,
        block_size=block_size,
    )
    actual = owner_type(
        capacity_rows=compressed_capacity,
        block_size=block_size,
    )
    tables = MiaM6RatioTables.allocate(
        ratio=ratio,
        rollback_rows=(2 if ratio == 4 else 1) * ratio + 8,
        capacity_tokens=capacity_tokens,
        compressed_capacity=compressed_capacity,
        compressed_block_size=block_size,
        block_table=actual.block_table,
    )
    schedule = tables.slice(offset)
    rng = np.random.default_rng(8_000 + record_bytes + ratio + offset)
    prefix = mx.array(
        rng.integers(
            0,
            256,
            (1, schedule.first_window, record_bytes),
            dtype=np.uint8,
        )
    )
    records = mx.array(
        rng.integers(
            0,
            256,
            (1, schedule.emitted_rows, record_bytes),
            dtype=np.uint8,
        )
    )
    expected._append_installed_records(prefix)
    actual._append_installed_records(prefix)
    expected._append_installed_records(records)
    actual._append_m6_records(records, schedule)
    mx.eval(expected.pages, actual.pages)

    assert len(actual) == len(expected)
    np.testing.assert_array_equal(
        np.array(actual.pages),
        np.array(expected.pages),
    )


def test_paged_mia_indexer_reads_132_byte_fp8_records_directly() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal paged indexer")

    row_values = (
        ((mx.arange(7 * 128, dtype=mx.float32) % 23) - 11) / 9.0
    ).reshape(1, 7, 128).astype(mx.bfloat16)
    rows = PagedMiaIndexerRows(capacity_rows=16, block_size=4)
    pages = rows.pages
    rows.append(row_values[:, :3])
    rows.append(row_values[:, 3:])

    query = (
        ((mx.arange(2 * 64 * 128, dtype=mx.float32) % 29) - 14) / 13.0
    ).reshape(1, 2, 64, 128).astype(mx.bfloat16)
    weights = mx.linspace(-0.2, 0.3, 2 * 64).reshape(1, 2, 64)
    actual = paged_indexer_scores(query, weights, rows.paged_records)
    tiled = paged_indexer_tiled_scores(query, weights, rows.paged_records)

    query_rows = PagedMiaIndexerRows(capacity_rows=128, block_size=64)
    query_rows.append(query.reshape(1, 2 * 64, 128))
    quant_query = query_rows.decode().reshape(1, 2, 64, 128)
    quant_rows = rows.decode()
    dot = mx.einsum("bshd,btd->bsht", quant_query, quant_rows)
    expected = mx.sum(mx.maximum(dot, 0.0) * weights[..., None], axis=2)
    mx.eval(actual, tiled, expected)

    assert rows.pages is pages
    assert rows.paged_records.records is pages
    assert rows.paged_records.record_bytes == 132
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.array(tiled),
        np.array(actual),
        rtol=2e-3,
        atol=2e-3,
    )


def test_mia_indexer_streams_qualified_records_into_compact_topk(
    monkeypatch,
) -> None:
    score_slice_widths = []
    fold_calls = []

    def fake_score_slice(q_records, weights, rows, row_start, row_count):
        del q_records, weights, rows
        score_slice_widths.append(row_count)
        scores = mx.arange(row_start, row_start + row_count, dtype=mx.float32)
        return mx.broadcast_to(scores[None, None], (1, 2, row_count))

    monkeypatch.setattr(
        "mtplx.deepseek_v4_paged_indexer._run_paged_indexer_score_slice",
        fake_score_slice,
    )

    def fake_radix_fold(
        scores,
        carry_values,
        carry_indices,
        causal_lengths,
        *,
        row_start,
        score_indices,
        has_carry,
        sentinel,
    ):
        del scores, causal_lengths, score_indices, sentinel
        fold_calls.append((row_start, has_carry))
        return carry_values, carry_indices

    monkeypatch.setattr(
        "mtplx.deepseek_v4_paged_indexer._run_radix_fold",
        fake_radix_fold,
    )
    rows = SimpleNamespace(length=300)
    workspace = MiaIndexerWorkspace.allocate(
        max_query_rows=2,
        topk=512,
        sentinel=300,
    )
    selection = _run_paged_indexer_records_topk(
        mx.zeros((1, 2, 64, 132), dtype=mx.uint8),
        mx.zeros((1, 2, 64), dtype=mx.float32),
        mx.array([7, 1199], dtype=mx.int32),
        rows,
        topk=512,
        compress_ratio=4,
        workspace=workspace,
        score_chunk_rows=128,
        query_count=2,
        score_slice=fake_score_slice,
        radix_fold=fake_radix_fold,
    )

    assert isinstance(selection, MiaTopKSelection)
    assert score_slice_widths == [128, 128, 44]
    assert fold_calls == [(0, False), (128, True), (256, True)]
    assert tuple(selection.indices.shape) == (1, 2, 512)
    assert tuple(selection.lengths.shape) == (1, 2)
    np.testing.assert_array_equal(np.array(selection.lengths), [[2, 300]])


def test_mia_indexer_install_removes_the_non_source_hadamard(monkeypatch) -> None:
    def installed(*_args):
        return None

    def installed_m6(*_args):
        return None

    monkeypatch.setattr(
        deepseek_v4_module,
        "install_paged_indexer_topk",
        lambda **_kwargs: installed,
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_paged_indexer_m6_topk",
        lambda **_kwargs: installed_m6,
    )
    query_install = {}

    def install_query_records(**kwargs):
        query_install.update(kwargs)
        return object()

    monkeypatch.setattr(
        deepseek_v4_module,
        "install_indexer_query_records",
        install_query_records,
    )
    indexer = Indexer.__new__(Indexer)
    indexer.n_heads = 64
    indexer.head_dim = 128
    indexer.rope_head_dim = 64
    indexer.index_topk = 512
    indexer.compress_ratio = 4
    indexer.softmax_scale = 128**-0.5
    installed_record_modes = []
    indexer.compressor = SimpleNamespace(
        rotate=True,
        install_mia_record_packer=installed_record_modes.append,
    )

    rope_table = object()
    indexer.install_mia_paged_topk(workspace=object(), rope_table=rope_table)

    query = mx.zeros((1, 1, 64, 128), dtype=mx.bfloat16)
    assert installed_record_modes == ["indexer"]
    assert query_install["rope_table"] is rope_table
    assert indexer.compressor.rotate is False
    assert indexer._prepare_query_rows(query) is query
    assert indexer._select_rows is installed
    assert indexer._select_m6_rows is installed_m6


def test_m6_attention_and_indexer_bypass_phase_dispatchers() -> None:
    for method in (
        deepseek_v4_module.DeepseekV4Attention._mia_m6_forward_uncompressed,
        deepseek_v4_module.DeepseekV4Attention._mia_m6_forward_ratio4,
        deepseek_v4_module.DeepseekV4Attention._mia_m6_forward_ratio128,
    ):
        source = inspect.getsource(method)
        assert "_mia_run_installed_attention" not in source
        assert "_nvfp4_sparse_mla" in source
        assert "_mia_qkv_impl" not in source
        assert "_mia_m6_qkv_impl" in source
    qkv_source = inspect.getsource(
        deepseek_v4_module.DeepseekV4Attention._mia_m6_qkv_records
    )
    assert "current_attention_phase" not in qkv_source
    assert ".target_records(" in qkv_source
    ratio4_source = inspect.getsource(
        deepseek_v4_module.DeepseekV4Attention._mia_m6_forward_ratio4
    )
    indexer_source = inspect.getsource(Indexer._mia_m6_select)
    assert "self.indexer(" not in ratio4_source
    assert "_mia_m6_select" in ratio4_source
    assert "_select_rows" not in indexer_source
    assert "_select_m6_rows" in indexer_source


def test_mia_attention_routes_nax_prefill_by_phase(monkeypatch) -> None:
    prefill_result = mx.array([11], dtype=mx.int32)
    direct_result = mx.array([22], dtype=mx.int32)
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_prefill_mla",
        lambda **_kwargs: lambda *_args, **_run_kwargs: prefill_result,
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_sparse_mla",
        lambda **_kwargs: lambda *_args, **_run_kwargs: direct_result,
    )

    class FakeIndexer:
        def install_mia_paged_topk(self) -> None:
            return None

    attn = deepseek_v4_module.DeepseekV4Attention.__new__(
        deepseek_v4_module.DeepseekV4Attention
    )
    attn.head_dim = 512
    attn.rope_head_dim = 64
    attn.n_heads = 64
    attn.window_size = 128
    attn.compress_ratio = 4
    attn.attn_sink = mx.zeros((64,), dtype=mx.float32)
    attn.softmax_scale = 512**-0.5
    attn.indexer = FakeIndexer()
    installed_record_modes = []
    attn.compressor = SimpleNamespace(
        install_mia_record_packer=installed_record_modes.append,
    )
    attn.install_mia_nvfp4_attention()
    assert installed_record_modes == ["nvfp4"]

    selection = MiaTopKSelection(
        indices=mx.zeros((1, 2, 1), dtype=mx.int32),
        lengths=mx.ones((1, 2), dtype=mx.int32),
    )

    def run(query_rows: int):
        return attn._cached_attention_impl(
            mx.zeros((1, query_rows, 64, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 432), dtype=mx.uint8),
            None,
            0,
            mx.arange(query_rows, dtype=mx.int32),
            4,
            MiaTopKSelection(
                indices=selection.indices[:, :query_rows],
                lengths=selection.lengths[:, :query_rows],
            ),
            None,
        )

    with attention_phase("prefill"):
        assert run(1) is prefill_result
        assert run(2) is prefill_result
    with attention_phase("ar_decode"):
        assert run(2) is direct_result


@pytest.mark.parametrize(
    "ratio,entrypoint",
    [
        (0, "_mia_m6_forward_uncompressed"),
        (4, "_mia_m6_forward_ratio4"),
        (128, "_mia_m6_forward_ratio128"),
    ],
)
def test_mia_attention_install_prebinds_ratio_specific_m6_entrypoints(
    monkeypatch,
    ratio: int,
    entrypoint: str,
) -> None:
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_prefill_mla",
        lambda **_kwargs: lambda *_args, **_run_kwargs: None,
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_sparse_mla",
        lambda **_kwargs: lambda *_args, **_run_kwargs: None,
    )
    attention = deepseek_v4_module.DeepseekV4Attention.__new__(
        deepseek_v4_module.DeepseekV4Attention
    )
    attention.head_dim = 512
    attention.rope_head_dim = 64
    attention.n_heads = 64
    attention.window_size = 128
    attention.compress_ratio = ratio
    attention.attn_sink = mx.zeros((64,), dtype=mx.float32)
    attention.softmax_scale = 512**-0.5
    attention.compressor = SimpleNamespace(
        install_mia_record_packer=lambda _mode: None,
    )

    attention.install_mia_nvfp4_attention()

    assert attention._mia_m6_forward_impl.__func__ is getattr(
        deepseek_v4_module.DeepseekV4Attention,
        entrypoint,
    )


@pytest.mark.parametrize("query_rows", [1, 6])
@pytest.mark.parametrize("paged_compressed", [False, True])
@pytest.mark.parametrize(
    "attention_impl",
    [nvfp4_sparse_mla, nvfp4_prefill_mla],
    ids=["direct-decode", "nax-prefill"],
)
def test_sparse_attention_reads_stock432_records_directly(
    query_rows: int,
    paged_compressed: bool,
    attention_impl,
) -> None:
    if not mx.metal.is_available():
        pytest.skip("requires direct Metal NVFP4 attention")

    window_count = 128
    compressed_count = 17
    window_start = 10
    query_positions = mx.arange(131, 131 + query_rows, dtype=mx.int32)
    latent_values = (
        (mx.arange(window_count * 512, dtype=mx.float32) % 31) - 15
    ) / 6.0
    rope_values = (
        (mx.arange(window_count * 64, dtype=mx.float32) % 23) - 11
    ) / 7.0
    window = MiaNVFP4Rows()
    window.append(
        latent_values.reshape(1, window_count, 512).astype(mx.bfloat16),
        rope_values.reshape(1, window_count, 64).astype(mx.bfloat16),
    )
    compressed = (
        PagedMiaNVFP4Rows(capacity_rows=32, block_size=8)
        if paged_compressed
        else MiaNVFP4Rows()
    )
    compressed.append(
        (
            ((mx.arange(compressed_count * 512, dtype=mx.float32) % 19) - 9)
            / 5.0
        ).reshape(1, compressed_count, 512).astype(mx.bfloat16),
        (
            ((mx.arange(compressed_count * 64, dtype=mx.float32) % 13) - 6)
            / 4.0
        ).reshape(1, compressed_count, 64).astype(mx.bfloat16),
    )
    queries = (
        ((mx.arange(64 * query_rows * 512, dtype=mx.float32) % 29) - 14)
        / 17.0
    ).reshape(1, 64, query_rows, 512).astype(mx.bfloat16)
    sinks = mx.linspace(-0.75, 0.5, 64, dtype=mx.float32)
    selected = mx.broadcast_to(
        mx.array([0, 3, 5, 9, 14], dtype=mx.int32),
        (1, query_rows, 5),
    )
    lengths = mx.minimum(
        mx.arange(3, 3 + query_rows, dtype=mx.int32),
        5,
    )[None]
    scale = 512**-0.5

    output = attention_impl(
        queries,
        window.records,
        window_start,
        query_positions,
        compressed.paged_records if paged_compressed else compressed.records,
        selected,
        lengths,
        sinks,
        scale,
    )

    window_key, window_value = window.decode()
    compressed_key, compressed_value = compressed.decode()
    expected_rows = []
    for query_row in range(query_rows):
        query_position = int(query_positions[query_row].item())
        absolute_window = np.arange(
            window_start,
            window_start + window_count,
        )
        valid_window = np.flatnonzero(
            (absolute_window <= query_position)
            & (absolute_window > query_position - 128)
        )
        valid_window = mx.array(valid_window, dtype=mx.int32)
        chosen = selected[0, query_row, : int(lengths[0, query_row].item())]
        key = mx.concatenate(
            [window_key[:, valid_window], compressed_key[:, chosen]],
            axis=1,
        )
        value = mx.concatenate(
            [window_value[:, valid_window], compressed_value[:, chosen]],
            axis=1,
        )
        # SparkInfer dequantizes stock432 operands to BF16, performs QK with
        # FP32 accumulation, and scales only the completed dot product.
        query = queries[:, :, query_row : query_row + 1].astype(mx.bfloat16)
        key_bf16 = key[:, None].astype(mx.bfloat16)
        scores = (
            mx.sum(
                query[..., None, :].astype(mx.float32)
                * key_bf16[:, :, None].astype(mx.float32),
                axis=-1,
            )
            * scale
        )
        scores_base2 = scores * np.log2(np.e)
        sinks_base2 = sinks.reshape(1, 64, 1, 1) * np.log2(np.e)
        maximum = mx.maximum(
            mx.max(scores_base2, axis=-1, keepdims=True),
            sinks_base2,
        )
        weights = _exp2(scores_base2 - maximum)
        denominator = mx.sum(weights, axis=-1, keepdims=True) + _exp2(
            sinks_base2 - maximum
        )
        # The unnormalized P operand crosses a BF16 boundary before P.V.  Its
        # FP32 value still owns the online-softmax denominator.
        probability_bf16 = weights.astype(mx.bfloat16)
        numerator = probability_bf16.astype(mx.float32) @ value[:, None].astype(
            mx.bfloat16
        ).astype(mx.float32)
        expected_rows.append((numerator / denominator).astype(mx.bfloat16))
    expected = mx.concatenate(expected_rows, axis=2)
    mx.eval(output, expected)

    assert output.shape == (1, 64, query_rows, 512)
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )


def test_dspark_k5_attention_uses_bf16_source_math_over_ring_and_all_drafts() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires direct Metal NVFP4 attention")

    prefix_length = 131
    context = MiaNVFP4Rows()
    context.append(
        (
            ((mx.arange(128 * 512, dtype=mx.float32) % 31) - 15) / 7.0
        ).reshape(1, 128, 512).astype(mx.bfloat16),
        (
            ((mx.arange(128 * 64, dtype=mx.float32) % 23) - 11) / 8.0
        ).reshape(1, 128, 64).astype(mx.bfloat16),
    )
    draft = MiaNVFP4Rows()
    draft.append(
        (
            ((mx.arange(5 * 512, dtype=mx.float32) % 19) - 9) / 6.0
        ).reshape(1, 5, 512).astype(mx.bfloat16),
        (
            ((mx.arange(5 * 64, dtype=mx.float32) % 13) - 6) / 5.0
        ).reshape(1, 5, 64).astype(mx.bfloat16),
    )
    queries = (
        ((mx.arange(64 * 5 * 512, dtype=mx.float32) % 29) - 14) / 17.0
    ).reshape(1, 5, 64, 512).astype(mx.bfloat16)
    sinks = mx.linspace(-0.5, 0.75, 64, dtype=mx.float32)
    scale = 512**-0.5

    run = install_dspark_k5_nvfp4_mla(
        heads=64,
        head_dim=512,
        rope_dim=64,
        window_size=128,
        block_size=5,
    )
    output = run(
        queries,
        context.records,
        draft.records,
        prefix_length,
        sinks,
        scale,
    )

    context_key, context_value = context.decode()
    draft_key, draft_value = draft.decode()
    ring_rows = mx.arange(prefix_length - 128, prefix_length, dtype=mx.int32) % 128
    key = mx.concatenate([context_key[:, ring_rows], draft_key], axis=1)
    value = mx.concatenate([context_value[:, ring_rows], draft_value], axis=1)
    scores = (
        mx.sum(
            queries.transpose(0, 2, 1, 3)[..., None, :].astype(mx.float32)
            * key[:, None, None].astype(mx.bfloat16).astype(mx.float32),
            axis=-1,
        )
        * scale
    )
    scores_base2 = scores * np.log2(np.e)
    sinks_base2 = sinks.reshape(1, 64, 1, 1) * np.log2(np.e)
    maximum = mx.maximum(
        mx.max(scores_base2, axis=-1, keepdims=True),
        sinks_base2,
    )
    probabilities = _exp2(scores_base2 - maximum)
    denominator = mx.sum(probabilities, axis=-1, keepdims=True) + _exp2(
        sinks_base2 - maximum
    )
    numerator = probabilities.astype(mx.bfloat16).astype(mx.float32) @ value[
        :, None
    ].astype(mx.bfloat16).astype(mx.float32)
    expected = (numerator / denominator).astype(mx.bfloat16).transpose(
        0, 2, 1, 3
    )
    mx.eval(output, expected)

    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )


def test_ratio128_sequential_paged_decode_crosses_tile64() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires direct Metal NVFP4 attention")

    compressed = PagedMiaNVFP4Rows(capacity_rows=80, block_size=2)
    assert compressed.block_size == 256 // 128
    compressed.append(
        (
            ((mx.arange(65 * 512, dtype=mx.float32) % 31) - 15) / 7.0
        ).reshape(1, 65, 512).astype(mx.bfloat16),
        (
            ((mx.arange(65 * 64, dtype=mx.float32) % 23) - 11) / 8.0
        ).reshape(1, 65, 64).astype(mx.bfloat16),
    )
    queries = (
        ((mx.arange(64 * 512, dtype=mx.float32) % 29) - 14) / 17.0
    ).reshape(1, 1, 64, 512).astype(mx.bfloat16)
    sinks = mx.linspace(-0.5, 0.75, 64, dtype=mx.float32)
    scale = 512**-0.5
    run = install_nvfp4_sparse_mla(
        heads=64,
        head_dim=512,
        rope_dim=64,
        window_size=128,
        compress_ratio=128,
    )
    output = run(
        queries,
        FixedMiaNVFP4Window(
            capacity_rows=8_416,
            block_size=64,
        ).paged_records(0, 0),
        0,
        mx.array([-1], dtype=mx.int32),
        compressed.paged_records,
        None,
        mx.array([[65]], dtype=mx.int32),
        sinks,
        scale,
    )

    key, value = compressed.decode()
    scores = (
        mx.sum(
            queries.transpose(0, 2, 1, 3)[..., None, :].astype(mx.float32)
            * key[:, None, None].astype(mx.bfloat16).astype(mx.float32),
            axis=-1,
        )
        * scale
        * np.log2(np.e)
    )
    sink = sinks.reshape(1, 64, 1, 1) * np.log2(np.e)
    maximum = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
    probabilities = _exp2(scores - maximum)
    denominator = mx.sum(probabilities, axis=-1, keepdims=True) + _exp2(
        sink - maximum
    )
    numerator = probabilities.astype(mx.bfloat16).astype(mx.float32) @ value[
        :, None
    ].astype(mx.bfloat16).astype(mx.float32)
    expected = (numerator / denominator).astype(mx.bfloat16).transpose(
        0, 2, 1, 3
    )
    mx.eval(output, expected)
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
