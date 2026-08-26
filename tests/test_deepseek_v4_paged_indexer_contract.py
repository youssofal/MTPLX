"""CPU/static gates for the installed Mia paged-indexer kernels."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from mtplx import deepseek_v4_paged_indexer as indexer  # noqa: E402
from mtplx.attention_context import attention_phase  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _stable_topk(values, indices, *, topk, sentinel):
    values = np.asarray(values, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    order = np.lexsort((indices, -values))
    selected = order[:topk]
    out_values = np.full((topk,), -np.inf, dtype=np.float32)
    out_indices = np.full((topk,), sentinel, dtype=np.int32)
    out_values[: len(selected)] = values[selected]
    out_indices[: len(selected)] = indices[selected]
    return out_values, out_indices


def test_exact_installer_seals_rope_capacity_and_requires_shared_table():
    undersized = indexer.MiaIndexerRoPETable(
        mx.zeros((2, 64), dtype=mx.float32)
    )
    with pytest.raises(ValueError, match="384005"):
        indexer.install_indexer_query_records(
            heads=64,
            head_dim=128,
            rope_dim=64,
            weight_scale=1.0,
            rope_table=undersized,
        )

    exact_values = SimpleNamespace(
        ndim=2,
        shape=(384_005, 64),
        dtype=mx.float32,
    )
    exact = indexer.install_indexer_query_records(
        heads=64,
        head_dim=128,
        rope_dim=64,
        weight_scale=1.0,
        rope_table=indexer.MiaIndexerRoPETable(exact_values),
    )
    assert exact.func is indexer._run_installed_indexer_query_records
    assert exact.keywords["cos_sin_cache"] is exact_values
    assert indexer.MiaIndexerRoPETable(exact_values).nbytes == 98_305_280

    with pytest.raises(ValueError, match="shared RoPE table"):
        indexer.install_indexer_query_records(
            heads=64,
            head_dim=128,
            rope_dim=64,
            weight_scale=1.0,
        )

    reference = indexer.install_reference_indexer_query_records(
        heads=64,
        head_dim=128,
        rope_dim=64,
        weight_scale=1.0,
    )
    assert reference.func is indexer._run_fused_indexer_query_records

    installed_source = inspect.getsource(
        indexer._run_installed_indexer_query_records
    )
    assert "query_count <= 0" not in installed_source
    assert "at least one query" not in installed_source


def test_exact_query_installer_prebinds_kernel_before_hot_calls(monkeypatch):
    launches = []

    def fake_kernel(**kwargs):
        launches.append(kwargs)
        return tuple(
            mx.zeros(shape, dtype=dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        )

    monkeypatch.setattr(indexer, "_query_rope_quant_kernel", lambda: fake_kernel)
    rope_values = mx.zeros((384_005, 64), dtype=mx.float32)
    installed = indexer.install_indexer_query_records(
        heads=64,
        head_dim=128,
        rope_dim=64,
        weight_scale=1.0,
        rope_table=indexer.MiaIndexerRoPETable(rope_values),
    )

    def fail_factory():
        raise AssertionError("installed query path re-entered the kernel factory")

    monkeypatch.setattr(indexer, "_query_rope_quant_kernel", fail_factory)
    records, weights = installed(
        mx.zeros((1, 1, 64, 128), dtype=mx.bfloat16),
        mx.zeros((1, 1, 64), dtype=mx.bfloat16),
        mx.zeros((1,), dtype=mx.int32),
    )

    assert installed.keywords["kernel"] is fake_kernel
    assert tuple(records.records.shape) == (1, 1, 64, 132)
    assert tuple(weights.shape) == (1, 1, 64)
    assert len(launches) == 1
    installed_source = inspect.getsource(
        indexer._run_installed_indexer_query_records
    )
    assert "_query_rope_quant_kernel" not in installed_source
    assert "records, scaled_weights = kernel(" in installed_source


def test_packaged_q_weight_fold_order_is_mutation_sensitive():
    f32 = np.float32
    weight = f32(8192.0)
    weight_scale = f32(2.0**-20)
    q_scale = f32(2.0**115)
    with np.errstate(over="ignore"):
        packaged = f32(f32(weight * weight_scale) * q_scale)
        reassociated = f32(f32(weight * q_scale) * weight_scale)
    assert np.isfinite(packaged)
    assert np.isinf(reassociated)

    tiny = np.nextafter(f32(0.0), f32(1.0), dtype=np.float32)
    weight = f32(1.5)  # Exactly representable in BF16.
    weight_scale = tiny
    q_scale = f32(1024.0)
    packaged = f32(f32(weight * weight_scale) * q_scale)
    reassociated = f32(f32(weight * q_scale) * weight_scale)
    assert packaged / tiny == 2048
    assert reassociated / tiny == 1536

    for factory in (
        indexer._query_rope_quant_kernel,
        indexer._query_rope_quant_from_inv_freq_kernel,
    ):
        source = inspect.getsource(factory)
        assert "* float(weight_scale)" in source
        assert "folded_weight * scale" in source
        assert "* scale * float(weight_scale)" not in source


def test_index_scores_apply_positive_k_scale_after_head_reduction():
    f32 = np.float32
    raw_dot = f32(28_672.0)
    k_scale = f32(2.0**115)
    weight = f32(2.0**-20)
    with np.errstate(over="ignore"):
        source_sum = f32(0.0)
        mutated_sum = f32(0.0)
        for _head in range(64):
            source_sum = f32(source_sum + f32(max(raw_dot, f32(0.0)) * weight))
            mutated_sum = f32(
                mutated_sum
                + f32(max(f32(raw_dot * k_scale), f32(0.0)) * weight)
            )
        source_order = f32(source_sum * k_scale)
        mutated = mutated_sum
    assert np.isfinite(source_order)
    assert np.isinf(mutated)

    _header, prefill = indexer._tiled_score_source(apply_query_scale=False)
    assert "dot0.thread_elements()[0], 0.0f" in prefill
    assert ") * q_weight;" in prefill
    assert "score0.thread_elements()[0] * k_scales[local_k0]" in prefill
    assert "dot0.thread_elements()[0] * k_scales" not in prefill

    decode = inspect.getsource(indexer._fused_decode_candidates_kernel)
    assert "float dot = simd_sum(partial);" in decode
    assert "local_scores[local_row] = score * k_scale;" in decode
    assert "simd_sum(partial) * q_scale * k_scale" not in decode


def test_raw_oracle_scorers_restore_nonunit_q_record_scale():
    raw_dot = np.float32(7.0)
    q_scale = np.float32(4.0)
    k_scale = np.float32(0.5)
    weight = np.float32(3.0)
    expected = np.float32(max(raw_dot * q_scale * k_scale, 0.0) * weight)
    omitted = np.float32(max(raw_dot * k_scale, 0.0) * weight)
    assert expected == np.float32(42.0)
    assert omitted == np.float32(10.5)

    scalar = inspect.getsource(indexer._oracle_score_kernel)
    assert "float q_scale = mtplx_indexer_record_scale(q_record);" in scalar
    assert "simd_sum(partial) * q_scale * k_scale" in scalar

    exact_header, exact_source = indexer._tiled_score_source(
        apply_query_scale=False
    )
    oracle_header, oracle_source = indexer._tiled_score_source(
        apply_query_scale=True
    )
    assert "q_scales" not in exact_source
    assert "dot0.thread_elements()[0], 0.0f" in exact_source
    assert "6,784" in exact_header
    assert "q_scales" in oracle_source
    assert "dot0.thread_elements()[0] * q_scales[local_q]" in oracle_source
    assert "6,912" in oracle_header
    assert indexer._tiled_score_kernel is not indexer._tiled_score_oracle_kernel


def test_installed_selector_has_no_raw_array_adapter():
    assert not hasattr(indexer, "_installed_query_records")
    assert not hasattr(indexer, "_run_paged_indexer_topk")
    assert not hasattr(indexer, "_run_paged_indexer_decode_topk")
    assert not hasattr(indexer, "_run_paged_indexer_phase_topk")

    installed = inspect.getsource(
        indexer._run_installed_paged_indexer_phase_topk
    )
    assert "_pack_indexer132" not in installed
    assert "isinstance" not in installed


def test_rope_table_nonzero_position_and_pair_layout_reaches_cached_kernel(
    monkeypatch,
):
    inv_freq = mx.array([np.pi / 2, np.pi / 4] + [0.0] * 30, dtype=mx.float32)
    table = indexer.precompute_indexer_rope_table(inv_freq, max_positions=3)
    row = np.array(table.values[1])
    np.testing.assert_allclose(row[[0, 32]], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        row[[1, 33]],
        [2**-0.5, 2**-0.5],
        atol=1e-6,
    )

    captured = {}

    def fake_kernel(*, inputs, **_kwargs):
        captured["position"] = int(inputs[2][0].item())
        captured["rope_row"] = np.array(inputs[3][captured["position"]])
        return (
            mx.zeros((1, 1, 64, 132), dtype=mx.uint8),
            mx.zeros((1, 1, 64), dtype=mx.float32),
        )

    monkeypatch.setattr(indexer, "_query_rope_quant_kernel", lambda: fake_kernel)
    indexer._run_installed_indexer_query_records(
        mx.zeros((1, 1, 64, 128), dtype=mx.bfloat16),
        mx.zeros((1, 1, 64), dtype=mx.bfloat16),
        mx.array([1], dtype=mx.int32),
        cos_sin_cache=table.values,
        weight_scale=mx.array(1.0, dtype=mx.float32),
        kernel=fake_kernel,
    )
    assert captured["position"] == 1
    np.testing.assert_array_equal(captured["rope_row"], row)


def test_q33_prefill_uses_tiled_score_then_three_k_carry_folds_for_tie_membership(
    monkeypatch,
):
    query_count = 33
    row_count = 701
    chunk_rows = 260
    sentinel = row_count
    score_calls = []
    fold_calls = []

    def fake_tiled_score(
        q_records,
        weights,
        rows,
        row_start,
        row_count,
    ):
        del weights, rows
        score_calls.append(
            {
                "queries": int(q_records.shape[1]),
                "row_start": row_start,
                "row_count": row_count,
                "output_shape": (1, query_count, row_count),
            }
        )
        local_values = np.zeros((1, query_count, row_count), dtype=np.float32)
        local_values[:, :, 1::2] = -0.0
        return mx.array(local_values)

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
        assert score_indices is None
        fold_calls.append((row_start, int(scores.shape[2]), has_carry))
        out_values = np.full((1, query_count, 512), -np.inf, dtype=np.float32)
        out_indices = np.full((1, query_count, 512), sentinel, dtype=np.int32)
        for query in range(query_count):
            available = int(causal_lengths[0, query].item()) - row_start
            local_count = max(0, min(available, int(scores.shape[2])))
            values = np.array(scores[0, query, :local_count])
            indices = np.arange(
                row_start, row_start + local_count, dtype=np.int32
            )
            if has_carry:
                valid = np.array(carry_indices[0, query]) != sentinel
                values = np.concatenate(
                    [values, np.array(carry_values[0, query])[valid]]
                )
                indices = np.concatenate(
                    [indices, np.array(carry_indices[0, query])[valid]]
                )
            selected_values, selected_indices = _stable_topk(
                values,
                indices,
                topk=512,
                sentinel=sentinel,
            )
            out_values[0, query] = selected_values
            out_indices[0, query] = selected_indices
        return mx.array(out_values), mx.array(out_indices)

    monkeypatch.setattr(indexer, "_run_paged_indexer_score_slice", fake_tiled_score)
    monkeypatch.setattr(indexer, "_run_radix_fold", fake_radix_fold)
    workspace = indexer.MiaIndexerWorkspace.allocate(
        max_query_rows=query_count,
        topk=512,
        sentinel=sentinel,
    )
    selection = indexer._run_paged_indexer_records_topk(
        mx.zeros((1, query_count, 64, 132), dtype=mx.uint8),
        mx.zeros((1, query_count, 64), dtype=mx.float32),
        mx.full((query_count,), row_count * 4 - 1, dtype=mx.int32),
        SimpleNamespace(length=row_count),
        topk=512,
        compress_ratio=4,
        workspace=workspace,
        score_chunk_rows=chunk_rows,
        query_count=query_count,
        score_slice=fake_tiled_score,
        radix_fold=fake_radix_fold,
    )

    assert [
        (call["row_start"], call["row_count"]) for call in score_calls
    ] == [
        (0, 260),
        (260, 260),
        (520, 181),
    ]
    assert fold_calls == [
        (0, 260, False),
        (260, 260, True),
        (520, 181, True),
    ]
    assert {call["queries"] for call in score_calls} == {33}
    assert {call["output_shape"] for call in score_calls} == {
        (1, 33, 260),
        (1, 33, 181),
    }
    assert tuple(selection.indices.shape) == (1, 33, 512)
    assert set(np.array(selection.indices[0, 0]).tolist()) == set(range(512))


def test_tiled_prefill_leaf_preserves_mma_grid_and_source_score_buffer(monkeypatch):
    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        shape = kwargs["output_shapes"][0]
        return (mx.zeros(shape, dtype=mx.float32),)

    monkeypatch.setattr(indexer, "_tiled_score_kernel", lambda: fake_kernel)
    rows = SimpleNamespace(
        records=mx.zeros((1, 64, 132), dtype=mx.uint8),
        block_table=mx.array([0], dtype=mx.int32),
        block_size=64,
    )
    output = indexer._run_paged_indexer_score_slice(
        mx.zeros((1, 33, 64, 132), dtype=mx.uint8),
        mx.zeros((1, 33, 64), dtype=mx.float32),
        rows,
        0,
        64,
        kernel=fake_kernel,
    )
    assert captured["grid"] == (2 * 1 * 256, 1, 1)
    assert captured["threadgroup"] == (256, 1, 1)
    assert captured["output_shapes"] == [(1, 33, 64)]
    assert tuple(output.shape) == (1, 33, 64)

    assert list(
        indexer._iter_prefill_k_tiles(
            row_count=96_000,
            score_chunk_rows=32_768,
        )
    ) == [
        (0, 32_768),
        (32_768, 32_768),
        (65_536, 30_464),
    ]


def test_prefill_query_split_matches_pinned_512_mib_logits_policy():
    q1024_tiles = list(
        indexer._iter_prefill_query_tiles(
            query_count=1_024,
            row_count=96_000,
            score_chunk_rows=32_768,
        )
    )
    short_q8224_tiles = list(
        indexer._iter_prefill_query_tiles(
            query_count=8_224,
            row_count=2_056,
            score_chunk_rows=32_768,
        )
    )
    long_q8224_tiles = list(
        indexer._iter_prefill_query_tiles(
            query_count=8_224,
            row_count=96_000,
            score_chunk_rows=32_768,
        )
    )
    assert q1024_tiles == [(0, 1_024)]
    assert short_q8224_tiles == [(0, 8_224)]
    assert long_q8224_tiles == [
        (0, 4_096),
        (4_096, 4_096),
        (8_192, 32),
    ]
    assert 1_024 * 32_768 * 4 == 128 * 1024**2
    assert 4_096 * 32_768 * 4 == indexer.INDEXER_PREFILL_MAX_LOGITS_BYTES

    k_chunks = (32_768, 32_768, 30_464)
    k256_tiles = sum((rows + 255) // 256 for rows in k_chunks)
    assert k256_tiles == 375
    assert 32 * k256_tiles == 12_000  # Q1024 Metal scorer threadgroups.
    assert 32 * 188 == 6_016  # Pinned source Q32xK512 geometry.
    assert 2 * len(k_chunks) == 6  # Q1024 score+fold dispatches.
    assert 2 * len(long_q8224_tiles) * len(k_chunks) == 18
    long_q_tiles = sum((rows + 31) // 32 for _start, rows in long_q8224_tiles)
    assert long_q_tiles * k256_tiles == 96_375
    assert 8_224 * len(k_chunks) == 24_672  # Radix threadgroups.


def test_metal_prefill_geometry_keeps_source_head_reduction_in_registers():
    """The Metal port must reduce launch amplification without CUDA geometry."""
    assert indexer.INDEXER_PREFILL_Q_TILE == 32
    assert indexer.INDEXER_PREFILL_K_TILE == 256
    assert indexer.INDEXER_PREFILL_K_SIMD_SPAN == 128
    assert indexer.INDEXER_PREFILL_DIM_PANEL == 8
    assert indexer.INDEXER_PREFILL_SCORE_THREADS == 256

    exact_header, exact_source = indexer._tiled_score_source(
        apply_query_scale=False
    )
    assert "MTPLX_INDEX_Q_TILE = 32" in exact_header
    assert "MTPLX_INDEX_K_TILE = 256" in exact_header
    assert "MTPLX_INDEX_K_SIMD_SPAN = 128" in exact_header
    assert "MTPLX_INDEX_DIM_PANEL = 8" in exact_header
    assert "threadgroup float head_dot" not in exact_source
    assert "threadgroup float scores" not in exact_source
    assert "threadgroup uint physical_rows[MTPLX_INDEX_K_TILE]" in exact_source
    assert "threadgroup float q_weights[MTPLX_INDEX_Q_TILE]" in exact_source
    assert "thread_elements()[0]" in exact_source
    assert "thread_elements()[1]" in exact_source
    assert "* k_scales[local_k0]" in exact_source

    # Q-panel + K-panel + physical rows + K scales + folded Q weights.
    assert 32 * 8 * 2 + 256 * 8 * 2 + 256 * 4 + 256 * 4 + 32 * 4 == 6_784

    k_chunks = (32_768, 32_768, 30_464)
    k256_tiles = sum((rows + 255) // 256 for rows in k_chunks)
    q1024_tiles = 1_024 // 32
    q8224_tiles = 4_096 // 32 + 4_096 // 32 + 1
    assert k256_tiles == 375
    assert q1024_tiles * k256_tiles == 12_000
    assert q8224_tiles * k256_tiles == 96_375


def test_exact_prefill_install_binds_tiled_mma_scorer_and_radix(monkeypatch):
    constructed = []

    def factory(name):
        def build():
            constructed.append(name)
            return object()

        return build

    monkeypatch.setattr(indexer.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(indexer, "_tiled_score_kernel", factory("tiled_mma"))
    monkeypatch.setattr(
        indexer,
        "_tiled_score_oracle_kernel",
        factory("raw_oracle"),
    )
    monkeypatch.setattr(indexer, "_radix_fold_kernel", factory("radix"))
    monkeypatch.setattr(
        indexer,
        "_fused_decode_candidates_kernel",
        factory("decode"),
    )
    workspace = indexer.MiaIndexerWorkspace.allocate(
        max_query_rows=33,
        topk=512,
        sentinel=96_000,
    )
    installed = indexer.install_paged_indexer_topk(
        heads=64,
        head_dim=128,
        topk=512,
        compress_ratio=4,
        workspace=workspace,
    )
    assert installed.func is indexer._run_installed_paged_indexer_phase_topk
    assert constructed == ["tiled_mma", "radix", "decode"]
    installed_source = inspect.getsource(
        indexer._run_paged_indexer_records_topk_query_tile
    )
    assert "scores = score_slice(" in installed_source
    assert "carry_scores, carry_indices = radix_fold(" in installed_source
    assert "_run_fused_prefill_fold(" not in installed_source


def test_exact_topk_installer_prebinds_prefill_decode_and_fold_kernels(monkeypatch):
    constructed = []
    launches = []

    def factory(name):
        def build():
            constructed.append(name)

            def kernel(**kwargs):
                launches.append(name)
                return tuple(
                    mx.zeros(shape, dtype=dtype)
                    for shape, dtype in zip(
                        kwargs["output_shapes"],
                        kwargs["output_dtypes"],
                        strict=True,
                    )
                )

            return kernel

        return build

    monkeypatch.setattr(indexer.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(indexer, "_tiled_score_kernel", factory("score"))
    monkeypatch.setattr(indexer, "_radix_fold_kernel", factory("fold"))
    monkeypatch.setattr(
        indexer,
        "_fused_decode_candidates_kernel",
        factory("decode"),
    )
    workspace = indexer.MiaIndexerWorkspace.allocate(
        max_query_rows=1,
        topk=512,
        sentinel=5_000,
    )
    installed = indexer.install_paged_indexer_topk(
        heads=64,
        head_dim=128,
        topk=512,
        compress_ratio=4,
        workspace=workspace,
    )

    def fail_factory():
        raise AssertionError("installed top-k path re-entered a kernel factory")

    monkeypatch.setattr(indexer, "_tiled_score_kernel", fail_factory)
    monkeypatch.setattr(indexer, "_radix_fold_kernel", fail_factory)
    monkeypatch.setattr(indexer, "_fused_decode_candidates_kernel", fail_factory)
    q_records = indexer.MiaIndexerQueryRecords(
        mx.zeros((1, 1, 64, 132), dtype=mx.uint8)
    )
    rows = SimpleNamespace(
        length=1,
        records=mx.zeros((1, 64, 132), dtype=mx.uint8),
        block_table=mx.zeros((1,), dtype=mx.int32),
        block_size=64,
    )
    weights = mx.zeros((1, 1, 64), dtype=mx.float32)

    with attention_phase("prefill"):
        prefill = installed(
            q_records,
            weights,
            mx.array([3], dtype=mx.int32),
            rows,
        )
    rows.length = 4_097
    with attention_phase("ar_decode"):
        decode = installed(
            q_records,
            weights,
            mx.array([4_097 * 4 - 1], dtype=mx.int32),
            rows,
        )

    assert constructed == ["score", "fold", "decode"]
    assert launches == ["score", "fold", "decode", "fold"]
    assert tuple(prefill.indices.shape) == (1, 1, 512)
    assert tuple(decode.indices.shape) == (1, 1, 512)
    for function, factory_name in (
        (indexer._run_paged_indexer_score_slice, "_tiled_score_kernel"),
        (indexer._run_radix_fold, "_radix_fold_kernel"),
        (
            indexer._run_fused_decode_candidates,
            "_fused_decode_candidates_kernel",
        ),
    ):
        assert factory_name not in inspect.getsource(function)


def test_decode_merge_keeps_sentinels_out_of_valid_signed_zero_winners(monkeypatch):
    row_count = 520
    sentinel = row_count
    values = np.full((1, 1, 2, 512), -np.inf, dtype=np.float32)
    indices = np.full((1, 1, 2, 512), sentinel, dtype=np.int32)
    values[0, 0, 0] = 0.0
    values[0, 0, 0, 1::2] = -0.0
    indices[0, 0, 0] = np.arange(512)
    values[0, 0, 1, :8] = 0.0
    values[0, 0, 1, 1:8:2] = -0.0
    indices[0, 0, 1, :8] = np.arange(512, 520)

    monkeypatch.setattr(
        indexer,
        "_run_fused_decode_candidates",
        lambda *_args: (mx.array(values), mx.array(indices)),
    )

    def fake_radix(
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
        del carry_values, carry_indices, causal_lengths, row_start, has_carry
        selected_values, selected_indices = _stable_topk(
            np.array(scores[0, 0]),
            np.array(score_indices[0, 0]),
            topk=512,
            sentinel=sentinel,
        )
        return (
            mx.array(selected_values[None, None]),
            mx.array(selected_indices[None, None]),
        )

    monkeypatch.setattr(indexer, "_run_radix_fold", fake_radix)
    workspace = indexer.MiaIndexerWorkspace.allocate(
        max_query_rows=1,
        topk=512,
        sentinel=sentinel,
    )
    selection = indexer._run_paged_indexer_records_decode_topk(
        mx.zeros((1, 1, 64, 132), dtype=mx.uint8),
        mx.zeros((1, 1, 64), dtype=mx.float32),
        mx.array([row_count * 4 - 1], dtype=mx.int32),
        SimpleNamespace(length=row_count),
        topk=512,
        compress_ratio=4,
        workspace=workspace,
        query_count=1,
        decode_candidates=lambda *_args: (mx.array(values), mx.array(indices)),
        radix_fold=fake_radix,
    )
    assert int(selection.lengths[0, 0]) == 512
    np.testing.assert_array_equal(np.array(selection.indices[0, 0]), np.arange(512))
    assert sentinel not in np.array(selection.indices[0, 0])


def test_metal_sources_preserve_mma_and_deterministic_winner_membership():
    for factory in (
        indexer._radix_fold_kernel,
        indexer._fused_decode_candidates_kernel,
    ):
        source = inspect.getsource(factory)
        assert "atomic_uint output_counter" not in source
        assert "index_pivot" in source
        assert "<= index_pivot" in source
        assert "value == 0.0f" in source

    header, prefill = indexer._tiled_score_source(apply_query_scale=False)
    assert "MTPLX_INDEX_Q_TILE = 32" in header
    assert "MTPLX_INDEX_K_TILE = 256" in header
    assert "simdgroup_matrix<half, 8, 8>" in prefill
    assert "simdgroup_multiply_accumulate(" in prefill
    assert "score0.thread_elements()[0] += max(" in prefill
    assert "dot0.thread_elements()[0], 0.0f" in prefill
    assert not hasattr(indexer, "_fused_prefill_fold_kernel")
    assert 32 * 8 * 2 + 256 * 8 * 2 + 256 * 4 + 256 * 4 + 32 * 4 == 6_784
