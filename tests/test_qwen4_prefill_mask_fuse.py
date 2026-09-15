"""MTPLX_QWEN4_PREFILL_MASK_FUSE -- the causal arm of the dense QSA lane.

Below the sparse crossover the QSA indexer hands attention **no** selection
whenever the chunk's whole history fits its block budget
(``last_nb <= block_topk``): top-512 of at most 512 candidate blocks is all
of them, so the visible set is exactly causal.  The lane used to materialise
a dense ``[1, 1, S, T]`` bool mask for that case anyway, which is what pushes
MLX off the fused ``steel_attention`` route at ``head_dim`` 256.  This module
pins the replacement:

* **when** the selection is trivially complete (the ``T <= (block_topk + 1) *
  ratio - 1`` frontier, 2,051 tokens on the production pack) and **when** the
  mask is causal-with-offset (chunk k sees all prior context plus a causal
  block within the chunk);
* that MLX 0.32.2's ``mask="causal"`` -- documented lower-right aligned --
  describes that offset case exactly, checked against the installed MLX
  rather than assumed;
* that the visible set of the causal string, of the dense mask the lane
  built, and of ``_qsa_blocks_to_dense_mask`` under a full selection are the
  same set;
* the counters (``mask_fuse_causal`` / ``mask_fuse_bool`` /
  ``mask_fuse_unavailable``), the query-tile composition, and the loud
  one-shot refusal when the build has no fused kernel.

Everything runs on tiny tensors on the CPU stream: the GPU on a development
box is usually holding a guarded benchmark, and ``force_fused=True`` is
resolved while the op is BUILT, so the refusal path is exercised natively
there (MLX raises "the fused kernels require a GPU (Metal) stream").  What
this cannot settle is the fused kernel's own numerics or its speed.
"""

from __future__ import annotations

import io
import contextlib

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.qwen4_prefill_chunk import QUERY_TILE_ENV, query_tile_spans

MASK_FUSE_ENV = "MTPLX_QWEN4_PREFILL_MASK_FUSE"

#: Production pack (``~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-
#: Optimized-Speed/config.json``): indexer_budget 2048, compress_ratio 4 =>
#: block_topk 512.
BLOCK_TOPK = 512
RATIO = 4
#: Largest post-update context whose selection is trivially complete:
#: ``T // ratio <= block_topk``.
TRIVIAL_FRONTIER = (BLOCK_TOPK + 1) * RATIO - 1


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # set_default_device leaks into every later-collected module (pytest
    # shares one process), so restore it.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _clean_lane_state(monkeypatch):
    """Every knob and one-shot in this lane is process-global; reset them."""

    monkeypatch.delenv(MASK_FUSE_ENV, raising=False)
    monkeypatch.delenv(QUERY_TILE_ENV, raising=False)
    saved_counts = dict(qwen4_exp._QSA_PREFILL_COUNTS)
    saved_unavailable = dict(qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE)
    saved_printed = qwen4_exp._MASK_FUSE_REFUSALS_PRINTED[0]
    saved_engaged = qwen4_exp._MASK_FUSE_ENGAGED[0]
    qwen4_exp._QSA_PREFILL_COUNTS.clear()
    qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.clear()
    qwen4_exp._MASK_FUSE_REFUSALS_PRINTED[0] = 0
    qwen4_exp._MASK_FUSE_ENGAGED[0] = False
    try:
        yield
    finally:
        qwen4_exp._QSA_PREFILL_COUNTS.clear()
        qwen4_exp._QSA_PREFILL_COUNTS.update(saved_counts)
        qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.clear()
        qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.update(saved_unavailable)
        qwen4_exp._MASK_FUSE_REFUSALS_PRINTED[0] = saved_printed
        qwen4_exp._MASK_FUSE_ENGAGED[0] = saved_engaged


def _arm(monkeypatch, value: str = "1") -> None:
    monkeypatch.setenv(MASK_FUSE_ENV, value)


def _lane_mask(pos_start: int, rows: int, total: int) -> mx.array:
    """The dense mask the lane builds at ``qwen4_exp.Attention.__call__``."""

    qpos = pos_start + mx.arange(rows, dtype=mx.int32)
    tpos = mx.arange(total, dtype=mx.int32)
    return (tpos[None, :] <= qpos[:, None])[None, None]


def _lower_right_causal(rows: int, total: int) -> mx.array:
    """MLX's documented ``"causal"``: last query aligns with the last key."""

    qpos = (total - rows) + mx.arange(rows, dtype=mx.int32)
    tpos = mx.arange(total, dtype=mx.int32)
    return (tpos[None, :] <= qpos[:, None])[None, None]


# ---------------------------------------------------------------------------
# 1. WHEN is the selection trivially complete / the mask causal-with-offset
# ---------------------------------------------------------------------------


def test_indexer_short_circuits_exactly_at_the_block_budget():
    """``last_nb <= block_topk`` is the whole condition, in both routes."""

    trivial = [t for t in range(1, 4 * TRIVIAL_FRONTIER) if t // RATIO <= BLOCK_TOPK]
    assert max(trivial) == TRIVIAL_FRONTIER == 2051
    # Both the eager route (_call_rows) and the compiled route
    # (_compiled_mode -> "update_only") use this one predicate, so the two
    # cannot disagree about when attention gets no selection.
    source = qwen4_exp.__file__
    text = open(source, encoding="utf-8").read()
    assert text.count("last_nb <= self.block_topk") == 2
    assert "last_nb = T // self.ratio" in text


@pytest.mark.parametrize(
    "chunk, prompt",
    [(2048, 16_384), (4096, 16_384), (2048, 32_768), (4096, 32_768)],
)
def test_which_prefill_chunks_are_trivially_complete(chunk, prompt):
    """Only a chunk whose POST-update context is <= 2,051 gets no selection.

    At the retained prefill width (4,096) that is no chunk at all, at 16K or
    at 32K: the causal arm cannot fire there, and the whole win of the flag
    at those cells is the bool-mask arm.  At the shipped 2,048 width it is
    chunk 0 and nothing else.
    """

    trivial = [
        pos_start
        for pos_start in range(0, prompt, chunk)
        # T == pos_start + S, S == this chunk's rows
        if (pos_start + min(chunk, prompt - pos_start)) // RATIO <= BLOCK_TOPK
    ]
    assert trivial == ([0] if chunk <= TRIVIAL_FRONTIER else [])


@pytest.mark.parametrize(
    "pos_start, rows, total, expected",
    [
        (0, 2048, 2048, True),  # first chunk
        (2048, 2048, 4096, True),  # causal-with-offset: all prior + causal
        (16_384, 4096, 20_480, True),
        (0, 1, 1, True),
        (0, 2048, 4096, False),  # padded/capacity-shaped KV: NOT lower-right
        (2048, 2048, 4097, False),
        (0, 0, 0, False),
    ],
)
def test_causal_string_is_exact_only_when_last_query_is_last_key(
    pos_start, rows, total, expected
):
    assert (
        qwen4_exp._prefill_causal_mask_is_exact(
            pos_start=pos_start, rows=rows, total_keys=total
        )
        is expected
    )


# ---------------------------------------------------------------------------
# 2. The visible sets are the SAME SET (the exactness invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos_start, rows", [(0, 8), (0, 13), (5, 8), (16, 8), (2048 - 16, 16)]
)
def test_lane_mask_equals_mlx_lower_right_causal(pos_start, rows):
    total = pos_start + rows
    lane = np.array(_lane_mask(pos_start, rows, total))
    ref = np.array(_lower_right_causal(rows, total))
    assert np.array_equal(lane, ref)
    # ... and it really is "all prior context + causal inside the chunk".
    dense = lane[0, 0]
    assert dense[:, :pos_start].all()
    within = dense[:, pos_start:]
    assert np.array_equal(within, np.tril(np.ones((rows, rows), dtype=bool)))


def test_full_block_selection_reconstructs_exactly_the_causal_mask():
    """``causal == the full-selection mask`` -- proved on the real function.

    ``_qsa_blocks_to_dense_mask`` is the lane's own reconstruction of a QSA
    selection.  Feed it every complete block (which is what the selector
    returns when ``nb_total <= block_topk``, since ``k_eff = min(block_topk,
    nb_total)``) and it must produce the causal mask, key for key.
    """

    ratio = 4
    for pos_start, rows in ((0, 16), (0, 12), (8, 8), (12, 20)):
        total = pos_start + rows
        logical_blocks = total // ratio
        topk = max(1, logical_blocks)
        # Every row selects all logical blocks; rows whose own complete-block
        # frontier is lower have the surplus ids clipped by the function's own
        # in-range guard, exactly as a real top-k of fewer candidates would.
        block_ids = mx.broadcast_to(
            mx.arange(topk, dtype=mx.int32)[None, :], (rows, topk)
        )
        block_valid = mx.ones((rows, topk), dtype=mx.bool_)
        got = qwen4_exp._qsa_blocks_to_dense_mask(
            block_ids,
            block_valid,
            pos_start=pos_start,
            total_tokens=total,
            compress_ratio=ratio,
        )
        want = _lane_mask(pos_start, rows, total)
        assert np.array_equal(np.array(got), np.array(want)), (pos_start, rows)


@pytest.mark.parametrize("pos_start, rows", [(0, 16), (12, 20), (33, 7)])
def test_mlx_causal_string_and_dense_mask_agree_numerically(pos_start, rows):
    """Against the INSTALLED MLX, not against its documentation."""

    total = pos_start + rows
    rng = np.random.default_rng(20260902)
    shape = (1, 2, rows, 16)
    kshape = (1, 2, total, 16)
    q = mx.array(rng.standard_normal(shape).astype(np.float32))
    k = mx.array(rng.standard_normal(kshape).astype(np.float32))
    v = mx.array(rng.standard_normal(kshape).astype(np.float32))
    scale = 16 ** -0.5
    string_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=qwen4_exp._CAUSAL_MASK
    )
    dense_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=_lane_mask(pos_start, rows, total)
    )
    mx.eval(string_out, dense_out)
    assert np.allclose(np.array(string_out), np.array(dense_out), atol=1e-6)


def test_causal_and_dense_arms_agree_within_bf16_rounding():
    """bf16 inputs: the two arms are the same visible set, one rounding class."""

    pos_start, rows, total = 24, 24, 48
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32)).astype(
        mx.bfloat16
    )
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32)).astype(
        mx.bfloat16
    )
    scale = 16 ** -0.5
    a = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    b = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=scale
    )
    mx.eval(a, b)
    # bf16 has ~3 decimal digits; the visible set being identical is what
    # makes this a tolerance and not a mismatch.
    assert np.allclose(
        np.array(a.astype(mx.float32)),
        np.array(b.astype(mx.float32)),
        atol=6e-3,
        rtol=6e-3,
    )


# ---------------------------------------------------------------------------
# 3. Query-tile composition
# ---------------------------------------------------------------------------


def test_query_tile_spans_keep_every_tile_lower_right_aligned():
    """Why the string survives tiling: each tile's last query is its last key."""

    for total_keys, rows, tile in ((4096, 4096, 2048), (20_480, 4096, 1024)):
        context_before = total_keys - rows
        for r0, r1, keys in query_tile_spans(
            rows, context_before=context_before, tile=tile
        ):
            assert qwen4_exp._prefill_causal_mask_is_exact(
                pos_start=context_before + r0, rows=r1 - r0, total_keys=keys
            )


def test_tiled_causal_string_matches_untiled_and_tiled_dense(monkeypatch):
    pos_start, rows, total = 32, 32, 64
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32))
    scale = 16 ** -0.5
    untiled = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    monkeypatch.setenv(QUERY_TILE_ENV, "8")
    tiled_causal = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    tiled_dense = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=scale
    )
    mx.eval(untiled, tiled_causal, tiled_dense)
    assert qwen4_exp._QSA_PREFILL_COUNTS.get("query_tile") == 2
    assert np.allclose(np.array(tiled_causal), np.array(untiled), atol=1e-6)
    assert np.allclose(np.array(tiled_causal), np.array(tiled_dense), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Counters, gating and the loud refusal
# ---------------------------------------------------------------------------


class _FakeSdpa:
    """Stand-in for a build that DOES have the fused kernels.

    ``fail_on`` refuses a whole mask kind.  ``refuse`` is the interesting
    one: a predicate over the call's own geometry, because that is how MLX
    actually refuses -- ``use_fallback`` reads the query length, the GQA
    factor and the head dims, so one build serves a 4,096-row prefill chunk
    and refuses a 4-row verify step at the very same head dim and mask kind.
    """

    def __init__(self, fail_on=(), refuse=None):
        self.calls = []
        self.geoms = []
        self.fail_on = set(fail_on)
        self.refuse = refuse

    def __call__(self, q, k, v, *, scale, mask=None, force_fused=False, **kw):
        kind = "causal" if isinstance(mask, str) else "bool"
        self.calls.append((kind, bool(force_fused)))
        self.geoms.append((kind, bool(force_fused), int(q.shape[2])))
        if force_fused and (
            kind in self.fail_on
            or (self.refuse is not None and self.refuse(kind, q, k))
        ):
            raise ValueError(
                f"no fused kernel for {kind} at query length {int(q.shape[2])}"
            )
        return mx.zeros(q.shape, dtype=q.dtype)


#: The served geometry: 24 query heads over 2 kv heads at head_dim 256.
PROD_Q_HEADS, PROD_KV_HEADS, PROD_HEAD_DIM = 24, 2, 256
#: MLX's vector kernel caps ``q_len * gqa_factor`` at 32 and its full kernel
#: wants ``q_len > 8``; at GQA 12 that is a dead band at q_len 3..8, which is
#: where an MTP verify step (4 rows) lands and a prefill chunk never does.
VERIFY_ROWS, CHUNK_ROWS = 4, 64


def _prod_qkv(rows: int, total: int):
    q = mx.zeros((1, PROD_Q_HEADS, rows, PROD_HEAD_DIM), dtype=mx.bfloat16)
    kv = mx.zeros((1, PROD_KV_HEADS, total, PROD_HEAD_DIM), dtype=mx.bfloat16)
    return q, kv


def _refuse_short_queries(kind, q, k):
    """The installed MLX's rule, in one line: the vector kernel is the only
    one offered below ``q_len`` 9, and it caps ``q_len * gqa`` at 32."""

    gqa = int(q.shape[1]) // int(k.shape[1])
    return int(q.shape[2]) <= 8 and int(q.shape[2]) * gqa > 32


def _install(monkeypatch, fake):
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake)


def test_flag_off_never_forces_fused_and_never_builds_a_string(monkeypatch):
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0)
    assert fake.calls == [("bool", False)]
    assert "mask_fuse_causal" not in qwen4_exp._QSA_PREFILL_COUNTS
    assert "mask_fuse_bool" not in qwen4_exp._QSA_PREFILL_COUNTS


def test_counters_split_causal_and_bool(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
    )
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0)
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_causal") == 1
    assert counts.get("mask_fuse_bool") == 1
    assert "mask_fuse_unavailable" not in counts
    assert "mask_fuse_dense_causal" not in counts
    assert "mask_fuse_dense_bool" not in counts
    # ONE forced call per arm: the capability question is the real call, at
    # the real geometry, so there is no synthetic probe to answer it wrong.
    assert fake.calls == [("causal", True), ("bool", True)]


def test_engagement_is_announced_once_naming_the_class(monkeypatch):
    """A serving process has no MTPLX_QSA_PREFILL_DEBUG receipt, and the
    absence of a refusal line is not evidence of engagement -- so the first
    class that fuses says so, once, on stderr."""

    _arm(monkeypatch)
    fake = _FakeSdpa(refuse=_refuse_short_queries)
    _install(monkeypatch, fake)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        q_small, kv_small = _prod_qkv(VERIFY_ROWS, VERIFY_ROWS)
        qwen4_exp._qsa_dense_attention(
            q_small, kv_small, kv_small, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
        )
        q_big, kv_big = _prod_qkv(CHUNK_ROWS, CHUNK_ROWS)
        for _ in range(3):
            qwen4_exp._qsa_dense_attention(
                q_big, kv_big, kv_big, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )
            qwen4_exp._qsa_dense_attention(
                q_big,
                kv_big,
                kv_big,
                mask=_lane_mask(0, CHUNK_ROWS, CHUNK_ROWS),
                scale=1.0,
            )
    lines = [line for line in err.getvalue().splitlines() if line.strip()]
    engaged = [line for line in lines if "engaged" in line]
    assert len(engaged) == 1, lines
    assert f"causal-mask q_len {CHUNK_ROWS}" in engaged[0]
    assert f"head_dim {PROD_HEAD_DIM} bfloat16" in engaged[0]
    # The refused verify class is reported too, and separately.
    assert len(lines) == 2, lines
    assert f"q_len {VERIFY_ROWS}" in lines[0]


def test_s1_decode_rows_never_take_the_forced_route(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 1, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(3, 1, 4), scale=1.0)
    assert fake.calls == [("bool", False)]


def test_refusal_is_loud_one_shot_and_per_kind(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa(fail_on={"causal"})
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for _ in range(3):
            qwen4_exp._qsa_dense_attention(
                q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )
        qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0
        )
    message = err.getvalue()
    refusals = [line for line in message.splitlines() if "armed but" in line]
    assert len(refusals) == 1, message
    assert "causal-mask q_len 4" in refusals[0]
    assert "head_dim 256" in refusals[0]
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_unavailable") == 1
    assert "mask_fuse_causal" not in counts
    # All three causal calls went dense, and only the first paid a raise.
    assert counts.get("mask_fuse_dense_causal") == 3
    # The bool arm is unaffected: its class was never asked about.
    assert counts.get("mask_fuse_bool") == 1
    assert "mask_fuse_dense_bool" not in counts
    # One forced causal attempt (refused, sticky for THAT class) + three
    # dense fallbacks, then the bool call.  Never a second forced attempt.
    assert fake.calls == [
        ("causal", True),
        ("causal", False),
        ("causal", False),
        ("causal", False),
        ("bool", True),
    ]


def test_refusal_is_native_on_a_build_without_fused_kernels(monkeypatch):
    """No stub: the CPU stream has no fused kernel at all, so MLX raises."""

    _arm(monkeypatch)
    rows, total = 4, 8
    q, kv = _prod_qkv(rows, total)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
        )
        qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(total - rows, rows, total), scale=1.0
        )
    message = err.getvalue()
    # The refusal is NATIVE: MLX declined force_fused (a CPU stream has no
    # fused kernel), the lane caught it and logged one per-class line with
    # MLX's own message appended. The exact native wording differs by MLX
    # build (0.32.0 said one thing, 0.32.2's CPU stream says "require a GPU
    # (Metal) stream"), so assert the version-independent lane line rather
    # than any one build's native string.
    assert "has no fused SDPA for shape class" in message
    assert "MTPLX_QWEN4_PREFILL_MASK_FUSE" in message
    # Two classes learned -- one per mask kind at this one geometry -- and
    # nothing said about any other shape.
    assert set(qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE) == {
        (
            "causal",
            PROD_HEAD_DIM,
            PROD_HEAD_DIM,
            "bfloat16",
            rows,
            PROD_Q_HEADS // PROD_KV_HEADS,
            True,
        ),
        (
            "bool",
            PROD_HEAD_DIM,
            PROD_HEAD_DIM,
            "bfloat16",
            rows,
            PROD_Q_HEADS // PROD_KV_HEADS,
            True,
        ),
    }
    assert qwen4_exp._QSA_PREFILL_COUNTS.get("mask_fuse_unavailable") == 2


def test_a_short_query_refusal_never_disarms_the_prefill_chunk(monkeypatch):
    """The served-process defect, pinned.

    MLX offers only the VECTOR kernel below query length 9, and that kernel
    caps ``q_len * gqa_factor`` at 32 -- so at this model's GQA 12 a 4-row
    MTP verify step is refused while a prefill chunk of the very same head
    dim, dtype and mask kind is served.  The server's warmup ladder runs a
    verify step before its first wide chunk; the benchmark driver does not.
    A capability keyed by mask kind therefore measured the win in one
    process and silently the control in the other.  Keyed by shape class,
    the verify refusal says nothing about the chunk.
    """

    _arm(monkeypatch)
    fake = _FakeSdpa(refuse=_refuse_short_queries)
    _install(monkeypatch, fake)

    # 1. The verify step goes first, exactly as the warmup ladder runs it.
    q_small, kv_small = _prod_qkv(VERIFY_ROWS, VERIFY_ROWS)
    with contextlib.redirect_stderr(io.StringIO()):
        for _ in range(3):
            qwen4_exp._qsa_dense_attention(
                q_small, kv_small, kv_small, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )

    # 2. Then the wide chunk the flag is actually armed for.
    q_big, kv_big = _prod_qkv(CHUNK_ROWS, CHUNK_ROWS)
    qwen4_exp._qsa_dense_attention(
        q_big, kv_big, kv_big, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
    )
    qwen4_exp._qsa_dense_attention(
        q_big,
        kv_big,
        kv_big,
        mask=_lane_mask(0, CHUNK_ROWS, CHUNK_ROWS),
        scale=1.0,
    )

    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_causal") == 1
    assert counts.get("mask_fuse_bool") == 1
    assert counts.get("mask_fuse_dense_causal") == 3
    assert counts.get("mask_fuse_unavailable") == 1
    # One forced attempt at the refused class, never a second; both chunk
    # classes forced and fused.
    assert fake.geoms == [
        ("causal", True, VERIFY_ROWS),
        ("causal", False, VERIFY_ROWS),
        ("causal", False, VERIFY_ROWS),
        ("causal", False, VERIFY_ROWS),
        ("causal", True, CHUNK_ROWS),
        ("bool", True, CHUNK_ROWS),
    ]


def test_a_prefill_chunk_refusal_routes_only_that_class(monkeypatch):
    """The other direction: a class MLX genuinely cannot serve goes dense
    per call, and the classes it can serve stay fused."""

    _arm(monkeypatch)
    fake = _FakeSdpa(refuse=lambda kind, q, k: int(q.shape[2]) >= CHUNK_ROWS)
    _install(monkeypatch, fake)
    q_big, kv_big = _prod_qkv(CHUNK_ROWS, CHUNK_ROWS)
    q_small, kv_small = _prod_qkv(VERIFY_ROWS, VERIFY_ROWS)
    with contextlib.redirect_stderr(io.StringIO()):
        for _ in range(3):
            qwen4_exp._qsa_dense_attention(
                q_big, kv_big, kv_big, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )
        qwen4_exp._qsa_dense_attention(
            q_small, kv_small, kv_small, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
        )
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_dense_causal") == 3
    assert counts.get("mask_fuse_unavailable") == 1
    # The short class is a different class and is still fused.
    assert counts.get("mask_fuse_causal") == 1
    assert fake.geoms == [
        ("causal", True, CHUNK_ROWS),
        ("causal", False, CHUNK_ROWS),
        ("causal", False, CHUNK_ROWS),
        ("causal", False, CHUNK_ROWS),
        ("causal", True, VERIFY_ROWS),
    ]


def test_the_refusal_line_names_the_class_and_scopes_itself(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa(refuse=_refuse_short_queries)
    _install(monkeypatch, fake)
    q, kv = _prod_qkv(VERIFY_ROWS, VERIFY_ROWS)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(0, VERIFY_ROWS, VERIFY_ROWS), scale=1.0
        )
    message = err.getvalue()
    assert message.count(MASK_FUSE_ENV) == 1, message
    assert "engaged" not in message
    # WHICH class.
    assert f"bool-mask q_len {VERIFY_ROWS}" in message
    assert f"GQA {PROD_Q_HEADS // PROD_KV_HEADS}" in message
    assert f"head_dim {PROD_HEAD_DIM} bfloat16" in message
    # And that it is per-class, not the process-wide disarm it used to be.
    assert "THAT shape class only" in message
    assert "NOT " in message and "process-wide" in message
    # MLX's own reason survives into the line.
    assert "no fused kernel for bool at query length 4" in message


def test_shape_class_reads_what_mlx_reads_and_nothing_else(monkeypatch):
    """A class is the geometry MLX's rules look at -- key length beyond the
    ``q_len <= k_len`` test is not one of them, and the query length is."""

    kind = "causal"
    q, kv = _prod_qkv(CHUNK_ROWS, 4096)
    q2, kv2 = _prod_qkv(CHUNK_ROWS, 8192)
    assert qwen4_exp._prefill_mask_fuse_class(
        kind, q, kv, kv
    ) == qwen4_exp._prefill_mask_fuse_class(kind, q2, kv2, kv2)
    q3, kv3 = _prod_qkv(VERIFY_ROWS, 4096)
    assert qwen4_exp._prefill_mask_fuse_class(
        kind, q3, kv3, kv3
    ) != qwen4_exp._prefill_mask_fuse_class(kind, q, kv, kv)
    # Mask kind, dtype and the GQA factor all split classes too.
    assert qwen4_exp._prefill_mask_fuse_class(
        "bool", q, kv, kv
    ) != qwen4_exp._prefill_mask_fuse_class(kind, q, kv, kv)
    wide_kv = mx.zeros((1, PROD_Q_HEADS, 4096, PROD_HEAD_DIM), dtype=mx.bfloat16)
    assert qwen4_exp._prefill_mask_fuse_class(
        kind, q, wide_kv, wide_kv
    ) != qwen4_exp._prefill_mask_fuse_class(kind, q, kv, kv)


def test_refusal_printing_is_capped_but_counting_is_not(monkeypatch):
    """A build that refuses everything must not own stderr for the life of
    the process; the counters keep the full tally."""

    _arm(monkeypatch)
    fake = _FakeSdpa(fail_on={"causal"})
    _install(monkeypatch, fake)
    classes = qwen4_exp._MASK_FUSE_REFUSAL_PRINT_LIMIT + 3
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for i in range(classes):
            rows = CHUNK_ROWS + i  # a new shape class each time
            q, kv = _prod_qkv(rows, rows)
            qwen4_exp._qsa_dense_attention(
                q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )
    message = err.getvalue()
    refusals = [line for line in message.splitlines() if "armed but" in line]
    assert len(refusals) == qwen4_exp._MASK_FUSE_REFUSAL_PRINT_LIMIT, message
    assert "further shape-class refusals are counted but not printed" in message
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_unavailable") == classes
    assert counts.get("mask_fuse_dense_causal") == classes


def test_armed_flag_on_an_unavailable_build_still_returns_the_dense_answer(
    monkeypatch,
):
    _arm(monkeypatch)
    pos_start, rows, total = 8, 8, 16
    rng = np.random.default_rng(3)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32))
    with contextlib.redirect_stderr(io.StringIO()):
        armed = qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=0.25
        )
    monkeypatch.delenv(MASK_FUSE_ENV, raising=False)
    stock = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=0.25
    )
    mx.eval(armed, stock)
    assert np.array_equal(np.array(armed), np.array(stock))


# ---------------------------------------------------------------------------
# 5. Knobs stay where the harness can reach them
# ---------------------------------------------------------------------------


def test_sparse_crossover_knob_is_documented_and_unchanged(monkeypatch):
    from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS

    assert "MTPLX_QSA_PREFILL_MIN_CONTEXT" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_QSA_PREFILL_DEBUG" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    monkeypatch.delenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", raising=False)
    assert qwen4_exp._qsa_prefill_min_context() == 32_768
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "8192")
    assert qwen4_exp._qsa_prefill_min_context() == 8192
    # Floor: below the 2,048-token budget the indexer has nothing to select.
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "512")
    assert qwen4_exp._qsa_prefill_min_context() == 2049




# ---------------------------------------------------------------------------
# 6. The routing decision at the real call site
# ---------------------------------------------------------------------------

#: A miniature of the production QSA geometry.  block_topk = 8 // 2 = 4, so
#: the trivially-complete frontier is (4 + 1) * 2 - 1 = 9 tokens -- the same
#: arithmetic the 2,048-budget pack runs at 2,051.
TINY = dict(
    hidden_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=2,
)
TINY_FRONTIER = (TINY["indexer_budget"] // TINY["indexer_compress_ratio"] + 1) * TINY[
    "indexer_compress_ratio"
] - 1


def _tiny_attention():
    mx.random.seed(20260902)
    layer = qwen4_exp.Attention(qwen4_exp.TextArgs(**TINY))
    layer.eval()
    mx.eval(layer.parameters())
    return layer


class _MaskRecorder(_FakeSdpa):
    """Records the mask each SDPA call receives, and returns real numbers."""

    def __call__(self, q, k, v, *, scale, mask=None, force_fused=False, **kw):
        kind = "causal" if isinstance(mask, str) else ("none" if mask is None else "bool")
        self.calls.append((kind, bool(force_fused)))
        if force_fused and kind in self.fail_on:
            raise ValueError(f"no fused kernel for {kind}")
        return _REAL_SDPA(q, k, v, scale=scale, mask=mask)


_REAL_SDPA = mx.fast.scaled_dot_product_attention


@pytest.mark.parametrize(
    "rows, expect_kind",
    [
        (TINY_FRONTIER - 1, "causal"),  # T <= frontier: no selection at all
        (TINY_FRONTIER + 1, "bool"),  # past it: a real top-k selection
    ],
)
def test_attention_routes_by_the_trivially_complete_frontier(
    monkeypatch, rows, expect_kind
):
    _arm(monkeypatch)
    layer = _tiny_attention()
    recorder = _MaskRecorder()
    _install(monkeypatch, recorder)
    x = (mx.random.normal((1, rows, TINY["hidden_size"])) * 0.3).astype(mx.bfloat16)
    cache = qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio)
    with contextlib.redirect_stderr(io.StringIO()):
        mx.eval(layer(x, cache))
    kinds = [kind for kind, _ in recorder.calls]
    assert expect_kind in kinds, kinds
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    if expect_kind == "causal":
        assert counts.get("mask_causal_eligible") == 1
    else:
        assert "mask_causal_eligible" not in counts


def test_flag_off_keeps_the_dense_causal_tensor_and_the_same_answer(monkeypatch):
    """Flag off is byte-identical, and the armed causal arm sees the same set."""

    layer = _tiny_attention()
    rows = TINY_FRONTIER - 1
    x = (mx.random.normal((1, rows, TINY["hidden_size"])) * 0.3).astype(mx.bfloat16)

    recorder = _MaskRecorder()
    _install(monkeypatch, recorder)
    off = layer(x, qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio))
    mx.eval(off)
    assert recorder.calls == [("bool", False)]

    _arm(monkeypatch)
    recorder2 = _MaskRecorder()
    _install(monkeypatch, recorder2)
    with contextlib.redirect_stderr(io.StringIO()):
        on = layer(x, qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio))
        mx.eval(on)
    # The recorder stands in for a build that HAS the fused kernel, so the
    # one forced call is the real one -- with the STRING and never a tensor.
    # Its return value is real MLX math, so the equality below is the
    # visible-set claim, evaluated end to end through the layer.
    assert recorder2.calls == [("causal", True)]
    assert np.array_equal(
        np.array(off.astype(mx.float32)), np.array(on.astype(mx.float32))
    )
