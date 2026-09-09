"""CPU-only gates for the native direct-index sparse-GQA QSA kernel.

Two things are covered, neither of which dispatches Metal:

1. ``mtplx.native.qsa_sparse_gqa_unsupported_reason`` -- the shape/dtype/
   position contract.  MLX arrays are constructed but never evaluated, and
   the reason function only reads metadata, so no kernel is compiled or run.
2. the microbenchmark's scoring logic -- the
   visible-set identity checker and the bf16-ULP tolerance measure.  Those
   are pure host arithmetic and are exercised on numpy inputs, including the
   failure cases, because the whole parity story rests on them.

Numeric Metal parity belongs to the operator-controlled guarded window; see
the harness docstring for the command.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

TOTAL = 16_384
CAPACITY = 32_768
ROWS = 64
TOP_K = 512
RATIO = 4
SCALE = 256 ** -0.5






# ---------------------------------------------------------------------------
# 1. binding validation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def native():
    mx = pytest.importorskip("mlx.core")
    native = pytest.importorskip("mtplx.native")
    if not native.native_qsa_available():
        pytest.skip("the native QSA extension is not built on this host")
    return mx, native


def _inputs(mx, *, rows=ROWS, ids_dtype=None, q_dtype=None):
    ids_dtype = ids_dtype or mx.int32
    q_dtype = q_dtype or mx.bfloat16
    return {
        "queries": mx.zeros((1, 24, rows, 256), q_dtype),
        "keys": mx.zeros((1, 2, CAPACITY, 256), q_dtype),
        "values": mx.zeros((1, 2, CAPACITY, 256), q_dtype),
        "block_ids": mx.zeros((rows, TOP_K), ids_dtype),
    }


def _reason(native_mod, arrays, **kwargs):
    params = {
        "pos_start": TOTAL - ROWS,
        "total_tokens": TOTAL,
        "scale": SCALE,
    }
    params.update(kwargs)
    return native_mod.qsa_sparse_gqa_unsupported_reason(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        **params,
    )


def test_production_contract_is_accepted(native):
    mx, native_mod = native
    assert _reason(native_mod, _inputs(mx)) is None


def test_four_dimensional_block_ids_are_accepted(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = arrays["block_ids"].reshape(1, 1, ROWS, TOP_K)
    assert _reason(native_mod, arrays) is None


def test_uint32_block_ids_are_accepted(native):
    """The metallib instantiates int32 too, so no astype is forced on the lane."""

    mx, native_mod = native
    assert _reason(native_mod, _inputs(mx, ids_dtype=mx.uint32)) is None


@pytest.mark.parametrize(
    "tile,expected_ok",
    [((128, 32), True), ((256, 32), True), ((64, 64), True), ((128, 64), True),
     ((32, 32), False), ((128, 128), False), ((64, 32), False)],
)
def test_only_instantiated_tiles_are_accepted(native, tile, expected_ok):
    mx, native_mod = native
    reason = _reason(
        native_mod, _inputs(mx), key_tile=tile[0], dimension_tile=tile[1]
    )
    assert (reason is None) is expected_ok


def test_float32_queries_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx, q_dtype=mx.float32))
    assert reason is not None and "float16 or bfloat16" in reason


def test_mixed_dtypes_are_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["keys"] = mx.zeros((1, 2, CAPACITY, 256), mx.float16)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "dtypes must match" in reason


def test_wrong_head_count_is_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["queries"] = mx.zeros((1, 16, ROWS, 256), mx.bfloat16)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[1, 24, S, 256]" in reason


def test_block_id_width_must_be_the_budget(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS, 256), mx.int32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[S, 512]" in reason


def test_block_id_rows_must_match_queries(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS + 1, TOP_K), mx.int32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[S, 512]" in reason


def test_float_block_ids_are_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS, TOP_K), mx.float32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "int32 or uint32" in reason


def test_sub_crossover_context_is_refused(native):
    """2,048 tokens is exactly the boundary: 512 blocks is not yet sparse."""

    mx, native_mod = native
    reason = _reason(
        native_mod, _inputs(mx), pos_start=2048 - ROWS, total_tokens=2048
    )
    assert reason is not None and "dense/sparse boundary" in reason


def test_context_beyond_the_backing_is_refused(native):
    mx, native_mod = native
    reason = _reason(
        native_mod,
        _inputs(mx),
        pos_start=CAPACITY + 1,
        total_tokens=CAPACITY + 1 + ROWS,
    )
    assert reason is not None and "backing capacity" in reason


def test_non_causal_suffix_is_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=TOTAL, total_tokens=TOTAL)
    assert reason is not None and "causal suffix" in reason


def test_traced_scalars_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=mx.array(0))
    assert reason is not None and "host integers" in reason


def test_bool_positions_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=True)
    assert reason is not None and "cannot be bool" in reason


def test_supported_predicate_matches_the_reason(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    assert native_mod.qsa_sparse_gqa_supported(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        pos_start=TOTAL - ROWS,
        total_tokens=TOTAL,
        scale=SCALE,
    )
    assert not native_mod.qsa_sparse_gqa_supported(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        pos_start=TOTAL - ROWS,
        total_tokens=TOTAL,
        scale=SCALE,
        key_tile=32,
        dimension_tile=32,
    )


def test_a_refused_call_raises_rather_than_dispatching(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    with pytest.raises(ValueError, match="dense/sparse boundary"):
        native_mod.qsa_sparse_gqa(
            arrays["queries"],
            arrays["keys"],
            arrays["values"],
            arrays["block_ids"],
            pos_start=2048 - ROWS,
            total_tokens=2048,
            scale=SCALE,
        )


# ---------------------------------------------------------------------------
# 2. harness scoring logic (pure numpy)
# ---------------------------------------------------------------------------
def _selector_output(pos_start: int, rows: int, rng: np.random.Generator):
    """Reproduce what ``_select_eager`` guarantees: ascending valid prefix."""

    ids = np.zeros((rows, TOP_K), dtype=np.int64)
    ok = np.zeros((rows, TOP_K), dtype=bool)
    for r in range(rows):
        complete = (pos_start + r + 1) // RATIO
        n = min(TOP_K, complete)
        if n:
            ids[r, :n] = np.sort(rng.choice(complete, size=n, replace=False))
            ok[r, :n] = True
    return ids, ok


def _cell(pos_start: int, rows: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    ids, ok = _selector_output(pos_start, rows, rng)
    return {
        "rows": rows,
        "pos_start": pos_start,
        "block_ids": ids,
        "block_valid": ok,
    }
































# ---------------------------------------------------------------------------
# 3. the seam with the `flash_prefill` selector branch
# ---------------------------------------------------------------------------
# Both lanes read one `_select_eager`.  This kernel's correctness rests on the
# contract the `flash_prefill` branch builds out of `top_idx` (ascending ids,
# prefix validity, count `min(512, (pos+1)//4)`), and that branch never serves
# a single row, so pin both sides.
def test_the_flash_prefill_selector_gate_never_admits_a_single_row():
    qwen4_exp = pytest.importorskip("mtplx.models.qwen4_exp")
    assert qwen4_exp._qsa_prefill_min_rows() >= 2


def test_the_flash_prefill_branch_still_builds_the_prefix_contract():
    """The contract must survive an edit to the scoring branches around it.

    Reading the source is deliberate: the branch's value is a GPU expression,
    but the three operations that establish the invariant (sort ascending,
    gather validity along the SORTED ids, blank invalid ids) are structural
    and a future edit that drops one would be silent wrong attention.
    """

    import inspect

    qwen4_exp = pytest.importorskip("mtplx.models.qwen4_exp")
    source = inspect.getsource(qwen4_exp.QSAIndexer._select_eager)
    branch = source[source.index("_qsa_large_prefill_enabled(S, total)") :]
    branch = branch[: branch.index('return ("flash_prefill"')]
    assert "mx.sort(top_idx" in branch  # chronological
    assert "mx.take_along_axis(" in branch  # validity gathered by SORTED id
    assert "block_ids.astype(mx.int64)" in branch
    assert "mx.where(" in branch  # invalid slots blanked, flag kept separately
