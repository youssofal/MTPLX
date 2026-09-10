"""CPU-only gates for the split-K QSA sparse-GQA DECODE lane.

Nothing here dispatches Metal.  What is covered:

1. The split geometry -- the host arithmetic that sizes the grid and the
   partial buffer.  Every split must have work, and the splits must cover the
   tile range exactly once.
2. The VISIBLE SET.  This is the load-bearing correctness property: the
   kernel must attend exactly the keys the shipped rows-gather lane attends.
   The kernel's per-slot model and a transcription of the shipped
   rows-gather token list's closed form are compared over the interesting
   positions, and the two WRONG predicates a reader might reach for (the
   prefill kernel's leading-prefix cut, and ``candidate <= q_abs``) are shown
   to disagree, so a future edit that adopts either fails here.
3. The kernel SOURCE structure -- that the shipped predicate is the one in
   the MSL, that the normalisation really moved to the merge pass, and that
   the C++ encoder builds the kernel names the metallib instantiates.
4. The parameter-block layout the host and the device both static_assert.
5. The flag parsing.

Numeric Metal parity belongs to the operator-controlled guarded window; see
the sparse-decode microbenchmark for the command.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from mtplx.kernels import qsa_sparse_decode as lane
from mtplx.native import (
    qsa_sparse_gqa_decode_partial_shape,
    qsa_sparse_gqa_decode_split_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "native_extensions" / "qsa_sparse_gqa" / "sparse_gqa"
KERNEL_H = EXT / "steel_qsa_sparse_gqa_decode.h"
PARAMS_H = EXT / "qsa_sparse_gqa_decode_params.h"
DECODE_CPP = EXT / "qsa_sparse_gqa_decode.cpp"
METAL = EXT / "qsa_sparse_gqa.metal"

TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
RATIO = lane.COMPRESS_RATIO
TOP_K = lane.TOP_K


# ---------------------------------------------------------------------------
# 1. split geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_tile", [t[0] for t in TILES])
@pytest.mark.parametrize("key_splits", [1, 2, 3, 4, 6, 8, 12, 16, 17, 33, 64])
def test_every_split_has_work(key_tile, key_splits):
    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, key_tile, key_splits
    )
    assert n_tiles == -(-lane.SELECTED_TOKENS // key_tile)
    assert 1 <= n_splits <= n_tiles
    assert per_split >= 1
    # The last split starts inside the tile range -- no empty threadgroup.
    assert (n_splits - 1) * per_split < n_tiles
    # And the splits reach the end of it.
    assert n_splits * per_split >= n_tiles


@pytest.mark.parametrize("key_tile", [t[0] for t in TILES])
@pytest.mark.parametrize("key_splits", [1, 2, 5, 8, 16, 64])
def test_splits_partition_the_tiles_exactly_once(key_tile, key_splits):
    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, key_tile, key_splits
    )
    covered = []
    for split in range(n_splits):
        t0 = split * per_split
        t1 = min(n_tiles, t0 + per_split)
        assert t0 < t1, "an empty split would write a dead partial state"
        covered.extend(range(t0, t1))
    assert covered == list(range(n_tiles))


def test_split_count_never_exceeds_the_tile_count():
    # 17 tiles at BK=128; asking for 64 splits must not dispatch 64.
    n_tiles, _, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, 64
    )
    assert n_tiles == 17
    assert n_splits == 17


def test_partial_shape_tracks_the_split_count():
    for rows in (1, 4):
        for key_tile, _ in TILES:
            _, _, n_splits = qsa_sparse_gqa_decode_split_geometry(
                lane.SELECTED_TOKENS, key_tile, 8
            )
            shape = qsa_sparse_gqa_decode_partial_shape(rows, key_tile, 8)
            assert shape == (n_splits, lane.Q_HEADS, rows, lane.HEAD_DIM + 2)


def test_partial_buffer_is_small_next_to_what_it_replaces():
    """The split-K partial state must not cost more than the gather it kills.

    The shipped lane writes a [1, 2, rows, 2052, 256] bf16 K/V pair per
    layer; this buffer is the whole extra memory traffic the split-K
    arrangement introduces, and it has to stay an order of magnitude below.
    """

    rows = 4
    shape = qsa_sparse_gqa_decode_partial_shape(rows, 128, 8)
    partial_bytes = shape[0] * shape[1] * shape[2] * shape[3] * 4
    gathered_bytes = 2 * (1 * 2 * rows * (TOP_K * RATIO + RATIO) * 256 * 2)
    assert partial_bytes * 8 < gathered_bytes


def test_split_geometry_rejects_nonsense():
    with pytest.raises(ValueError):
        qsa_sparse_gqa_decode_split_geometry(0, 128, 8)
    with pytest.raises(ValueError):
        qsa_sparse_gqa_decode_split_geometry(2051, 0, 8)


# ---------------------------------------------------------------------------
# 2. the visible set
# ---------------------------------------------------------------------------
def _ids_for(q_abs: int, nb_total: int, seed: int = 7) -> list:
    """512 distinct block ids in an order that is deliberately NOT sorted."""

    state = seed
    pool = list(range(nb_total))
    for i in range(len(pool) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:TOP_K]


POSITIONS = [
    2048, 2049, 2050, 2051, 2052, 2053,
    4093, 4094, 4095, 4096,
    16_383, 16_384, 16_385, 16_386,
    17_407,
]


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_kernel_and_shipped_lane_attend_the_same_keys(q_abs):
    nb_total = (q_abs + 1) // RATIO + 1
    ids = _ids_for(q_abs, max(nb_total, TOP_K + 1))
    assert lane.visible_sets_agree(ids, q_abs, key_length=q_abs + 1)


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_the_dropped_2052nd_slot_is_always_invalid(q_abs):
    """The kernel walks 2,051 slots; the shipped lane builds 2,052.

    The extra one is the tail's fourth, token ``((pos+1)//4)*4 + 3``, which is
    greater than ``pos`` for every residue class.  If that ever stops being
    true the kernel is dropping a visible key.
    """

    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    _, ok = lane.shipped_row_tokens(ids, q_abs)
    assert len(ok) == TOP_K * RATIO + RATIO
    assert ok[-1] is False or ok[-1] == False  # noqa: E712 - mx bool_ compat


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_no_key_past_the_query_position_is_ever_attended(q_abs):
    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    for pos in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1):
        assert pos <= q_abs


def test_the_prefix_cut_the_prefill_kernel_uses_is_wrong_here():
    """The decode selector does not sort, so a leading-prefix cut is wrong.

    ``mtplx.native.qsa_sparse_gqa`` (the prefill entry point) may take the
    first ``min(512, complete_blocks)`` slots as the valid ones, because the
    prefill branch of ``_select_eager`` sorts its top-k ascending.  Its decode
    branch does not.
    This test builds a row where the two disagree, so an edit that adopts the
    prefix cut in the decode kernel fails right here rather than quietly
    changing which keys the model attends.
    """

    # complete_blocks 511 < 512, so a prefix cut really cuts: it keeps the
    # first 511 SLOTS instead of the visible BLOCKS.  Slot 511 holds a
    # perfectly visible block and the cut drops it.
    q_abs = 2043
    complete = lane.visible_block_count(q_abs)
    assert complete == 511
    ids = [511] + list(range(0, 511))
    assert len(ids) == TOP_K
    assert len(set(ids)) == TOP_K
    per_slot = sorted(
        p
        for p in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
        if p >= 0
    )
    prefix_cut = sorted(
        block * RATIO + within
        for slot_block, block in enumerate(ids)
        if slot_block < min(TOP_K, complete)
        for within in range(RATIO)
        if block * RATIO + within <= q_abs
    )
    assert per_slot != prefix_cut
    # And the per-slot answer is the shipped lane's.
    idx, ok = lane.shipped_row_tokens(ids, q_abs)
    shipped = sorted(t for t, good in zip(idx, ok) if good)
    assert per_slot == shipped


def test_causal_only_predicate_would_double_count_the_tail():
    """``candidate <= q_abs`` is implied by the real predicate but not equal.

    For ``block == complete_blocks`` it admits tokens of the INCOMPLETE block,
    which the tail slots already contribute -- so the softmax denominator
    would count them twice.  Pinning this stops a "simplification".
    """

    q_abs = 2049                       # (2050)//4 == 512, tail is non-empty
    complete = lane.visible_block_count(q_abs)
    ids = [complete] + list(range(0, TOP_K - 1))
    real = lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
    lax = []
    for slot in range(TOP_K * RATIO):
        block = ids[slot // RATIO]
        candidate = block * RATIO + (slot % RATIO)
        lax.append(candidate if candidate <= q_abs else -1)
    for within in range(RATIO - 1):
        candidate = complete * RATIO + within
        lax.append(candidate if candidate <= q_abs else -1)
    real_hits = [p for p in real if p >= 0]
    lax_hits = [p for p in lax if p >= 0]
    assert len(real_hits) == len(set(real_hits)), "the real predicate is a set"
    assert len(lax_hits) > len(set(lax_hits)), "the lax one duplicates keys"


def test_a_row_always_has_at_least_one_visible_key():
    """The merge's zero-denominator branch must be unreachable under the gate.

    The gate needs ``total // 4 > 512``, so every query row has at least one
    complete block, and the selector's top-512 of a set whose finite entries
    all outrank the -inf ones must contain at least one of them.
    """

    for q_abs in (2048, 2051, 4096, 17_407):
        complete = lane.visible_block_count(q_abs)
        assert complete >= 1
        ids = _ids_for(q_abs, complete + 1)
        hits = [
            p
            for p in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
            if p >= 0
        ]
        assert hits


def test_key_length_clamps_every_read():
    """kL is a memory-safety bound: nothing may index past it."""

    q_abs = 4096
    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    key_length = 2048  # deliberately shorter than the positions imply
    for pos in lane.kernel_row_tokens(ids, q_abs, key_length=key_length):
        assert pos < key_length


# ---------------------------------------------------------------------------
# 3. kernel source structure
# ---------------------------------------------------------------------------
def test_the_kernel_uses_the_per_slot_predicate():
    src = KERNEL_H.read_text()
    assert "raw_block < long(complete_blocks)" in src
    assert "complete_blocks = (q_abs + 1) / kCompressRatio" in src
    # The prefix cut the prefill kernel uses must NOT appear here.
    assert "valid_blocks" not in src


def test_the_normalisation_moved_to_the_merge_pass():
    src = KERNEL_H.read_text()
    split = src.split("mtplx_qsa_sparse_gqa_decode_merge")[0]
    assert "MtplxQsaDecDivOp" not in split
    assert "row_bin_op<MtplxQsaDecMulOp>" in split
    assert "prow[D] = max_score[0]" in split
    assert "prow[D + 1] = sum_score[0]" in split
    merge = src.split("mtplx_qsa_sparse_gqa_decode_merge")[1]
    assert "denom" in merge and "AccumType(1) / denom" in merge


def test_the_split_pass_walks_only_its_own_tile_range():
    src = KERNEL_H.read_text()
    assert "const int t0 = split * params->tiles_per_split;" in src
    assert "for (int ktile = t0; ktile < t1; ++ktile)" in src


def test_the_split_pass_reads_the_offset_from_a_device_buffer():
    src = KERNEL_H.read_text()
    assert "const device int* QOffset [[buffer(4)]]" in src
    assert "const int q_abs = QOffset[0] + q_pos;" in src


def test_the_online_softmax_initialises_to_finite_min_not_neg_inf():
    """finite_min, so an all-masked tile yields 0 rather than NaN."""

    src = KERNEL_H.read_text()
    assert "max_score[i] = Limits<AccumType>::finite_min;" in src
    assert "AccumType m = Limits<AccumType>::finite_min;" in src


def _expected_kernel_names() -> set:
    names = set()
    for tname in ("float16", "bfloat16"):
        for iname in ("uint32", "int32"):
            for bk, dc in TILES:
                names.add(
                    f"mtplx_qsa_sparse_gqa_decode_split_{tname}_{iname}"
                    f"_bk{bk}_dc{dc}_gqa12_hp16_d256_wm2"
                )
        names.add(f"mtplx_qsa_sparse_gqa_decode_merge_{tname}_d256")
    return names


def test_the_metallib_instantiates_every_name_the_encoder_can_ask_for():
    metal = METAL.read_text()
    # Expand the macro the same way the preprocessor will.
    instantiated = set()
    for tname in ("float16", "bfloat16"):
        for iname in ("uint32", "int32"):
            for bk, dc in TILES:
                instantiated.add(
                    f"mtplx_qsa_sparse_gqa_decode_split_{tname}_{iname}"
                    f"_bk{bk}_dc{dc}_gqa12_hp16_d256_wm2"
                )
        instantiated.add(f"mtplx_qsa_sparse_gqa_decode_merge_{tname}_d256")
    assert instantiated == _expected_kernel_names()
    # And the source really contains the macro pieces that build them.
    assert '"mtplx_qsa_sparse_gqa_decode_split_" #tname "_" #iname' in metal
    assert '"_bk" #bk "_dc" #dc "_gqa12_hp16_d256_wm2"' in metal
    assert '"mtplx_qsa_sparse_gqa_decode_merge_" #tname "_d256"' in metal
    for bk, dc in TILES:
        assert f"iname, itype, {bk}, {dc})" in metal


def test_the_encoder_builds_the_same_name_the_metallib_declares():
    cpp = DECODE_CPP.read_text()
    assert '"mtplx_qsa_sparse_gqa_decode_split_"' in cpp
    assert '"_bk", key_tile_' in cpp
    assert '"_dc", dimension_tile_' in cpp
    assert '"_gqa", kGqa, "_hp", kHeadPad, "_d",' in cpp
    assert '"mtplx_qsa_sparse_gqa_decode_merge_"' in cpp
    # The compiled-in geometry constants must match the macro's literals.
    assert re.search(r"constexpr int kGqa = 12;", cpp)
    assert re.search(r"constexpr int kHeadPad = 16;", cpp)
    assert re.search(r"constexpr int kHeadDim = 256;", cpp)
    assert re.search(r"constexpr int kWarps = 2;", cpp)


def test_the_encoder_dispatches_the_split_grid_on_the_z_axis():
    cpp = DECODE_CPP.read_text()
    assert "MTL::Size(rows, kKvHeads, n_splits)" in cpp
    assert "MTL::Size(32, kWarps, 1)" in cpp
    assert "MTL::Size(kQHeads * rows, 1, 1)" in cpp
    assert "MTL::Size(kHeadDim, 1, 1)" in cpp


def test_the_selected_token_width_agrees_across_the_three_definitions():
    cpp = DECODE_CPP.read_text()
    assert (
        "constexpr int kSelectedTokens = kTopK * kCompressRatio "
        "+ (kCompressRatio - 1);" in cpp
    )
    assert lane.SELECTED_TOKENS == TOP_K * RATIO + (RATIO - 1) == 2051
    src = KERNEL_H.read_text()
    assert "constexpr int kTail = kCompressRatio - 1;" in src
    assert (
        "const int selected_tokens = params->topk * kCompressRatio + kTail;"
        in src
    )


# ---------------------------------------------------------------------------
# 4. parameter-block layout
# ---------------------------------------------------------------------------
def _static_assert_size(header: str, struct: str) -> int:
    match = re.search(
        rf"static_assert\(sizeof\({struct}\) == (\d+)", header
    )
    assert match, f"no sizeof static_assert for {struct}"
    return int(match.group(1))


def test_decode_params_layout_is_pinned_and_arithmetically_right():
    header = PARAMS_H.read_text()
    # 11 ints + 1 float + 1 pad int, then 4 x 3 int64 at 8-byte alignment.
    ints = len(re.findall(r"^  int \w+;", header, flags=re.M))
    body = header.split("struct MtplxQsaSparseGqaDecodeParams")[1].split("};")[0]
    n_int = len(re.findall(r"\bint \w+;", body))
    n_float = len(re.findall(r"\bfloat \w+;", body))
    n_i64_arrays = len(re.findall(r"int64_t \w+\[3\];", body))
    assert (n_int, n_float, n_i64_arrays) == (11, 1, 4)
    scalar = n_int * 4 + n_float * 4
    aligned = -(-scalar // 8) * 8
    assert _static_assert_size(header, "MtplxQsaSparseGqaDecodeParams") == (
        aligned + n_i64_arrays * 24
    )
    assert ints >= 11


def test_merge_params_layout_is_pinned_and_arithmetically_right():
    header = PARAMS_H.read_text()
    body = header.split("struct MtplxQsaSparseGqaMergeParams")[1].split("};")[0]
    n_int = len(re.findall(r"\bint \w+;", body))
    n_i64_arrays = len(re.findall(r"int64_t \w+\[3\];", body))
    assert (n_int, n_i64_arrays) == (6, 1)
    scalar = n_int * 4
    aligned = -(-scalar // 8) * 8
    assert _static_assert_size(header, "MtplxQsaSparseGqaMergeParams") == (
        aligned + n_i64_arrays * 24
    )


def test_both_params_blocks_are_included_by_both_sides():
    assert '#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"' in (
        KERNEL_H.read_text()
    )
    assert '#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"' in (
        DECODE_CPP.read_text()
    )


def test_the_build_compiles_the_new_sources():
    cmake = (
        ROOT / "native_extensions" / "qsa_sparse_gqa" / "CMakeLists.txt"
    ).read_text()
    assert "sparse_gqa/qsa_sparse_gqa_decode.cpp" in cmake
    assert "sparse_gqa/steel_qsa_sparse_gqa_decode.h" in cmake
    assert "sparse_gqa/qsa_sparse_gqa_decode_params.h" in cmake


# ---------------------------------------------------------------------------
# 5. flags
# ---------------------------------------------------------------------------
def test_tile_flag_accepts_only_instantiated_tiles():
    from mtplx.runtime_options import _parse_sparse_decode_tile

    assert _parse_sparse_decode_tile(None) == (128, 32)
    assert _parse_sparse_decode_tile("") == (128, 32)
    for bk, dc in TILES:
        assert _parse_sparse_decode_tile(f"{bk}:{dc}") == (bk, dc)
    for bad in ("128", "128:33", "127:32", "128:32:1", "x:y"):
        with pytest.raises(ValueError):
            _parse_sparse_decode_tile(bad)


def test_splits_flag_is_bounded():
    from mtplx.runtime_options import (
        QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
        _parse_sparse_decode_splits,
    )

    assert _parse_sparse_decode_splits(None) == default == 17
    assert _parse_sparse_decode_splits("17") == 17
    for bad in ("0", "65", "-1", "eight"):
        with pytest.raises(ValueError):
            _parse_sparse_decode_splits(bad)


def test_the_flag_is_off_by_default():
    from mtplx.runtime_options import env_bool

    assert env_bool("MTPLX_QSA_SPARSE_DECODE", default=False, env={}) is False


def test_engagement_reports_a_pending_install_as_not_installed():
    lane.reset_for_tests()
    report = lane.engagement()
    assert report["installed"] is False
    assert report["disabled_reason"] is None
    assert report["verify_kernel"] == 0


def test_parity_thresholds_are_stated_not_implicit():
    assert lane.PARITY_FP32_MAX_ABS_ULPS == 2.0
    assert lane.PARITY_FP32_MAX_REL_L2 == 5.0e-4
    assert lane.PARITY_SHIPPED_MAX_REL_L2 == 5.0e-2
    assert lane.PARITY_MIN_TOP1 == 0.98


# ---------------------------------------------------------------------------
# 7. what the 2026-09-02 runs settled
# ---------------------------------------------------------------------------
def test_the_tight_gate_is_tighter_than_the_measured_shipped_delta():
    """The fp32 gate must be able to FAIL if the attribution is wrong.

    Every configuration measured rel_l2 4.78e-3 against the shipped path.  If
    that delta really is the shipped path's own bf16 score and probability
    casts, the same comparison against the fp32 reference collapses to output
    rounding.  A gate set above 4.78e-3 could not tell those apart, so it
    would certify nothing.
    """

    measured_vs_shipped = 4.78e-3
    assert lane.PARITY_FP32_MAX_REL_L2 < measured_vs_shipped
    # ... while the loose bar stays an order of magnitude above it, because it
    # bounds the reference's quantisation, not the kernel's error.
    assert lane.PARITY_SHIPPED_MAX_REL_L2 > 10 * measured_vs_shipped


def test_the_reference_ladder_has_three_distinct_rungs():
    import inspect

    src = inspect.getsource(lane.reference_attention)
    # The shipped rung must keep BOTH bf16 roundings...
    assert "probs = probs.astype(queries.dtype)" in src
    # ...and the fp32 rung must remove them by UPCASTING (exact), never by
    # changing the operands.
    assert "q_view = q_view.astype(mx.float32)" in src
    assert "v_view = v_view.astype(mx.float32)" in src
    for fn, scores, probs in (
        (lane.stock_reference, False, False),
        (lane.shipped_fp32_probs_reference, False, True),
        (lane.fp32_reference, True, True),
    ):
        wrapped = inspect.getsource(fn)
        assert f"fp32_scores={scores}" in wrapped
        assert f"fp32_probs={probs}" in wrapped


def test_every_reference_rung_returns_the_query_dtype():
    """All three must be comparable element for element against bf16 output."""

    import inspect

    src = inspect.getsource(lane.reference_attention)
    assert src.rstrip().endswith(".astype(queries.dtype)")


def test_the_default_split_target_reaches_one_tile_per_threadgroup():
    """17 is the tile count at BK=128, so it is the knob's saturation point."""

    from mtplx.runtime_options import (
        QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
    )

    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, default
    )
    assert (n_tiles, per_split, n_splits) == (17, 1, 17)
    assert lane.VERIFY_ROWS * lane.KV_HEADS * n_splits == 136


@pytest.mark.parametrize("larger", [18, 32, 33, 64])
def test_split_targets_above_the_tile_count_are_the_same_configuration(larger):
    """Why the first sweep's s17 and s32 rows are one config measured twice.

    Their 5.3% spread is therefore the bench's noise floor, not a result, and
    nothing may be called a winner on a margin under it.
    """

    at_17 = qsa_sparse_gqa_decode_split_geometry(lane.SELECTED_TOKENS, 128, 17)
    assert qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, larger
    ) == at_17




def test_the_lane_and_the_native_wrapper_agree_on_the_default():
    from mtplx import native
    from mtplx.runtime_options import (
        QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
    )

    assert native._DEFAULT_KEY_SPLITS == default
    assert native._DEFAULT_TILE == (128, 32)






# ---------------------------------------------------------------------------
# 6. the nanobind ABI guard
#
# The first guarded run built the extension cleanly and then failed at the
# first call with "incompatible function arguments ... queries:
# mlx::core::array" while invoking with mlx.core.array.  The cause was not the
# stream kwarg: the extension was built against nanobind internals v19 while
# mlx.core uses v21, so the two got separate __nb_internals_<tag>_mlx__
# capsules and no array could ever be cast.  These pin the detector.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402


















def test_the_native_wrapper_uses_the_same_tag_regex():
    """One definition of "what a nanobind ABI tag looks like", two readers."""

    from mtplx import native

    blob = b"\x00v21_system_libcpp_abi1\x00"
    assert [m.group(1) for m in native._NB_ABI_TAG_RE.finditer(blob)] == [b"21"]


def test_the_native_wrapper_reads_a_versions_from_a_binary(tmp_path):
    from mtplx import native

    blob = tmp_path / "fake.so"
    blob.write_bytes(b"v19_system_libcpp_abi1\x00")
    assert native._nanobind_internals_version(blob) == 19
    blob.write_bytes(b"nothing")
    assert native._nanobind_internals_version(blob) is None


def test_the_cmake_guard_is_a_fatal_error_not_a_warning():
    cmake = (
        ROOT / "native_extensions" / "qsa_sparse_gqa" / "CMakeLists.txt"
    ).read_text()
    assert "MTPLX_NANOBIND_DIR" in cmake
    assert "NB_INTERNALS_VERSION" in cmake
    guard = cmake.split("if(MTPLX_NB_INTERNALS AND MTPLX_MLX_NB_INTERNALS)")[1]
    assert "FATAL_ERROR" in guard
    assert "MTPLX_NANOBIND_DIR=" in guard
    # The one wrong fix must stay called out where someone would reach for it.
    assert "NEVER \"fix\" a mismatch by defining NB_INTERNALS_VERSION" in cmake


def test_the_wrappers_omit_a_none_stream():
    """Passing an explicit None was not the bug, but it is one fewer variable."""

    source = (ROOT / "mtplx" / "native" / "__init__.py").read_text()
    assert source.count("if stream is None:") == 2
    assert "stream=stream)" in source
    assert "stream=stream,\n    )" not in source


# ---------------------------------------------------------------------------
# 8. the micro's parity-gate ladder
#
# The two-gate ladder renamed PARITY_MAX_ABS_ULPS -> PARITY_FP32_MAX_ABS_ULPS
# (and PARITY_MAX_REL_L2 -> PARITY_FP32_MAX_REL_L2), and
# The sparse-decode microbenchmark kept reading the old names in its
# report header -- so it crashed with AttributeError at startup, INSIDE a
# guarded window, after the operator had already taken the box.  These pin the
# gate-building surface so that class of drift fails here instead.
#
# Nothing below dispatches Metal: the gate builders are fed stubbed numbers,
# and the one call to the micro's own ``compare`` runs on the CPU stream.
# ---------------------------------------------------------------------------






#: Comfortably inside every threshold, so a test can break exactly one.
PASSING_STATS = {
    "vs_fp32": {"max_abs_bf16_ulps": 1.0, "rel_l2": 1.0e-4, "top1": 1.0},
    "vs_shipped": {"max_abs_bf16_ulps": 9.0, "rel_l2": 4.78e-3, "top1": 1.0},
}


def _stats(reference=None, key=None, value=None):
    """The passing stats, optionally with ONE statistic pushed out of bounds."""

    out = {k: dict(v) for k, v in PASSING_STATS.items()}
    if reference is not None:
        out[reference][key] = value
    return out






















# --- the verdict roll-up, which is what the process status is --------------














