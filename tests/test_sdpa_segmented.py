"""Segmented-KV decode/verify kernel (GPU, TensorOps required): exactness against the contiguous
kernel and an fp32 reference, at the real Qwen3.8-27B head geometry."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mtplx.kernels import sdpa_segmented as S
from mtplx.kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
from mtplx.nax_verify import nax_available

HQ, HK, D = 24, 4, 256
SCALE = 1.0 / math.sqrt(D)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not nax_available(),
    reason="TensorOps kernel unavailable on this machine",
)


def _data(total, q_len, peaked=False, seed=3):
    mx.random.seed(seed)
    qs = 2.5 if peaked else 0.5
    q = (mx.random.normal((1, HQ, q_len, D)) * qs).astype(mx.bfloat16)
    k = (mx.random.normal((1, HK, total + 64, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HK, total + 64, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def _split(k, v, lens, spare=0):
    segs, a = [], 0
    for n in lens:
        kk = mx.contiguous(k[:, :, a : a + n + spare])
        vv = mx.contiguous(v[:, :, a : a + n + spare])
        mx.eval(kk, vv)
        segs.append((kk, vv, n))
        a += n
    return segs


def _lens(total, nseg):
    lens = [total // nseg] * nseg
    lens[-1] += total - sum(lens)
    return lens


def _ref(q, k, v, total):
    q_len = q.shape[2]
    kf = mx.repeat(k[:, :, :total].astype(mx.float32), HQ // HK, axis=1)
    vf = mx.repeat(v[:, :, :total].astype(mx.float32), HQ // HK, axis=1)
    s = (q.astype(mx.float32) @ kf.transpose(0, 1, 3, 2)) * SCALE
    vis = mx.arange(total)[None, :] <= (total - q_len + mx.arange(q_len))[:, None]
    s = mx.where(vis[None, None], s, -mx.inf)
    return mx.softmax(s, axis=-1) @ vf


def _ulps(a, b):
    """Max |a - b| in bf16 ulps of the output's largest magnitude (small elements carry the
    cancellation error of their terms, so a per-element ulp would be meaningless)."""
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    top = float(mx.maximum(mx.abs(a).max(), mx.abs(b).max()).item())
    ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
    return float(mx.abs(a - b).max().item()) / ulp


@pytest.mark.parametrize("q_len", [1, 2, 4, 5])
def test_one_segment_is_bit_identical_to_the_contiguous_kernel(q_len) -> None:
    total = 9000
    q, k, v = _data(total, q_len)
    ref = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=total, scale=SCALE)
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=[(k, v, total)], scale=SCALE)
    assert mx.array_equal(out, ref).item()


@pytest.mark.parametrize("nseg", [2, 3, 6, 13, 30])
@pytest.mark.parametrize("q_len", [1, 4, 5])
@pytest.mark.parametrize("peaked", [False, True])
def test_segments_match_contiguous_within_bf16_ulps_and_the_fp32_reference(nseg, q_len, peaked) -> None:
    total = 12000
    q, k, v = _data(total, q_len, peaked)
    contiguous = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=total, scale=SCALE)
    out = S.sdpa_nax_flash_dsplit_segments(
        queries=q, segments=_split(k, v, _lens(total, nseg)), scale=SCALE
    )
    assert out is not None and bool(mx.all(mx.isfinite(out)).item())
    assert _ulps(out, contiguous) <= 3
    ref = _ref(q, k, v, total)
    err = float(mx.abs(out.astype(mx.float32) - ref).max())
    err_contig = float(mx.abs(contiguous.astype(mx.float32) - ref).max())
    assert err <= max(2 * err_contig, 1e-3)
    # argmax over a fixed random projection agrees (the logits' argmax in miniature)
    proj = mx.random.normal((HQ * D, 512), key=mx.random.key(1))
    a = mx.argmax(out.astype(mx.float32).transpose(0, 2, 1, 3).reshape(q_len, -1) @ proj, axis=-1)
    b = mx.argmax(contiguous.astype(mx.float32).transpose(0, 2, 1, 3).reshape(q_len, -1) @ proj, axis=-1)
    assert mx.array_equal(a, b).item()


def test_a_segment_reference_shorter_than_its_buffer_reads_only_its_rows() -> None:
    """A fork inside a segment: (buffer, n') with live rows past n' that must stay invisible."""
    total, q_len = 9000, 4
    q, k, v = _data(total, q_len)
    contiguous = sdpa_nax_flash_dsplit(queries=q, keys=k, values=v, offset=total, scale=SCALE)
    lens = _lens(total, 3)
    segs = _split(k, v, lens, spare=0)
    # make the first segment's buffer longer than its reference, with junk past n'
    n0 = lens[0]
    big_k = mx.concatenate([segs[0][0], mx.ones((1, HK, 300, D), mx.bfloat16) * 9], axis=2)
    big_v = mx.concatenate([segs[0][1], mx.ones((1, HK, 300, D), mx.bfloat16) * 9], axis=2)
    segs[0] = (big_k, big_v, n0)
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=segs, scale=SCALE)
    assert _ulps(out, contiguous) <= 3


def test_only_the_last_segment_is_tail_causal() -> None:
    """An old segment is fully visible to every query row, the last one keeps the mask."""
    total, q_len = 9000, 5
    q, k, v = _data(total, q_len, peaked=True, seed=11)
    ref = _ref(q, k, v, total)
    out = S.sdpa_nax_flash_dsplit_segments(queries=q, segments=_split(k, v, _lens(total, 4)), scale=SCALE)
    assert float(mx.abs(out.astype(mx.float32) - ref).max()) < 5e-3


def test_unsupported_shapes_bail_without_a_kernel_launch() -> None:
    q, k, v = _data(9000, 6)  # 6 rows x GQA 6 = 36 > 32
    before = dict(S.segmented_bail_counts)
    assert S.sdpa_nax_flash_dsplit_segments(queries=q, segments=_split(k, v, _lens(9000, 2)), scale=SCALE) is None
    assert S.segmented_bail_counts.get("m_rows_gt_32", 0) == before.get("m_rows_gt_32", 0) + 1
    assert S.segments_supported(5, 6, 256) and not S.segments_supported(6, 6, 256)
