"""Studio numeric parity for the vendored Steel QSA prefill kernel.

Requires a built ``mtplx_qsa_kernels`` extension. Skips everywhere else.

Bounds are the upstream oMLX ones (Codex §6 / oMLX tests):
normal long-prefix ``max_error <= 5e-3``; early-prefix ``<= 2e-2`` with the
one-visible-token row bit-identical. Do not tighten without repeated M3
results.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import mtplx.kernels.qsa_prefill_direct as direct
from mtplx.models.qwen4_exp import _qsa_prefill_gather_attention

pytestmark = pytest.mark.skipif(
    direct._EXT is None
    or not hasattr(direct._EXT, "qwen4_qsa_sparse_gqa_attention"),
    reason="no built mtplx_qsa_kernels extension on this machine",
)

_SCALE = 0.0625
_RATIO = 4
_TOPK = 512
_NORMAL_BOUND = 5e-3
_EARLY_BOUND = 2e-2


def _selection(pos_start: int, rows: int):
    """Chronological valid prefix, same contract as the production selectors."""

    ids = []
    valid = []
    for r in range(rows):
        complete = (pos_start + r + 1) // _RATIO
        take = min(_TOPK, complete)
        start = max(0, complete - take)
        row = list(range(start, start + take)) + [0] * (_TOPK - take)
        ids.append(row)
        valid.append([True] * take + [False] * (_TOPK - take))
    return mx.array(ids, dtype=mx.int32), mx.array(valid, dtype=mx.bool_)


def _random_qkv(dtype, *, rows: int, total: int, seed: int):
    mx.random.seed(seed)
    q = mx.random.normal((1, 24, rows, 256)).astype(dtype)
    k = mx.random.normal((1, 2, total, 256)).astype(dtype)
    v = mx.random.normal((1, 2, total, 256)).astype(dtype)
    return q, k, v


@pytest.fixture(scope="module")
def native_lane():
    info = direct.qsa_prefill_direct_build_info()
    assert info.get("built_against_mlx") == mx.__version__, info
    assert info.get("metal_library") == "mtplx_qsa_kernels", info
    assert direct.qsa_prefill_direct_preflight() is True, info
    assert direct.qsa_prefill_direct_ready() is True
    return info


def _parity(dtype, *, total: int, rows: int, seed: int, bound: float):
    pos_start = total - rows
    q, k, v = _random_qkv(dtype, rows=rows, total=total, seed=seed)
    ids, valid = _selection(pos_start, rows)
    got = direct.qsa_prefill_direct(
        q,
        k,
        v,
        ids,
        valid,
        pos_start=pos_start,
        total_tokens=total,
        scale=_SCALE,
        compress_ratio=_RATIO,
        block_topk=_TOPK,
    )
    ref = _qsa_prefill_gather_attention(
        q,
        k,
        v,
        ids,
        valid,
        pos_start=pos_start,
        total_tokens=total,
        compress_ratio=_RATIO,
        scale=_SCALE,
        tile_rows=8,
    )
    mx.eval(got, ref)
    diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
    max_err = float(mx.max(diff).item())
    # Argmax over D=256 flips on near-ties inside the 5e-3 bound; report the
    # match rate but do not use it as a gate (Codex: copy oMLX max_error, do
    # not chase bitwise / 1-ULP equality).
    got_am = mx.argmax(got.astype(mx.float32), axis=-1)
    ref_am = mx.argmax(ref.astype(mx.float32), axis=-1)
    match = float(mx.mean((got_am == ref_am).astype(mx.float32)).item())
    return max_err, match, got, ref, ids, valid


def test_build_receipt_matches_runtime_mlx(native_lane):
    assert native_lane["built_against_mlx"] == mx.__version__
    assert native_lane["imported_mlx"] == mx.__version__


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
def test_long_prefix_parity_vs_gather(native_lane, dtype):
    """Production suffix just past the dense/sparse boundary."""

    max_err, argmax_match, *_ = _parity(
        dtype, total=2056, rows=8, seed=121, bound=_NORMAL_BOUND
    )
    print(f"long_prefix {dtype} max_err={max_err:.6g} argmax_match={argmax_match:.4f}")
    assert max_err <= _NORMAL_BOUND, (max_err, argmax_match)


@pytest.mark.parametrize("remainder", [0, 1, 2, 3])
def test_causal_tail_mod4_parity_vs_gather(native_lane, remainder):
    """Last visible token ``T-1 ≡ remainder (mod 4)``."""

    # 2053 % 4 == 1 → T-1 ≡ 0; then 2054, 2055, 2056 cover 1,2,3.
    total = 2053 + remainder
    assert (total - 1) % 4 == remainder
    max_err, argmax_match, *_ = _parity(
        mx.bfloat16, total=total, rows=8, seed=400 + remainder, bound=_NORMAL_BOUND
    )
    assert max_err <= _NORMAL_BOUND, (remainder, max_err, argmax_match)


def test_chunk_start_straddles_a_block(native_lane):
    """pos_start is not block-aligned."""

    total, rows = 2062, 9
    pos_start = total - rows
    assert pos_start % 4 != 0
    max_err, argmax_match, *_ = _parity(
        mx.bfloat16, total=total, rows=rows, seed=77, bound=_NORMAL_BOUND
    )
    assert max_err <= _NORMAL_BOUND, (max_err, argmax_match, pos_start)


def test_selected_block_set_matches_gather_expansion(native_lane):
    """The ids the direct kernel consumes are the same set gather expands."""

    total, rows = 2056, 8
    pos_start = total - rows
    ids, valid = _selection(pos_start, rows)
    mx.eval(ids, valid)
    for r in range(rows):
        qpos = pos_start + r
        complete = (qpos + 1) // _RATIO
        take = min(_TOPK, complete)
        row_ids = [int(x) for x in ids[r, :take].tolist()]
        assert row_ids == list(range(complete - take, complete))
        assert all(0 <= b < complete for b in row_ids)
        assert len(set(row_ids)) == take


def test_capacity_backing_is_rejected_by_the_wrapper(native_lane):
    q, k, v = _random_qkv(mx.bfloat16, rows=8, total=2056, seed=9)
    cap_k = mx.concatenate([k, mx.zeros((1, 2, 128, 256), dtype=k.dtype)], axis=2)
    cap_v = mx.concatenate([v, mx.zeros((1, 2, 128, 256), dtype=v.dtype)], axis=2)
    ids, valid = _selection(2048, 8)
    assert (
        direct.qsa_prefill_direct_supported(
            q,
            cap_k,
            cap_v,
            ids,
            valid,
            pos_start=2048,
            total_tokens=2056,
            scale=_SCALE,
        )
        is False
    )


def test_future_tokens_in_the_logical_cache_cannot_leak(native_lane):
    """Row 0 of a first chunk sees one token. Poisoning the rest must not
    change that row. Uses the native symbol because the Python wrapper
    refuses T below the dense/sparse boundary."""

    mx.random.seed(313)
    rows = 33
    q = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, rows, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, rows, 256)).astype(mx.bfloat16)
    selected = mx.broadcast_to(
        mx.arange(_TOPK, dtype=mx.uint32)[None, None, None, :],
        (1, 1, rows, _TOPK),
    )
    clean = direct._EXT.qwen4_qsa_sparse_gqa_attention(
        q, k, v, selected, _SCALE, 0, key_tile=64, dimension_tile=64
    )
    k_poison = mx.concatenate(
        [k[:, :, :1, :], mx.ones_like(k[:, :, 1:, :]) * 1.0e4], axis=2
    )
    v_poison = mx.concatenate(
        [v[:, :, :1, :], mx.ones_like(v[:, :, 1:, :]) * 1.0e4], axis=2
    )
    poisoned = direct._EXT.qwen4_qsa_sparse_gqa_attention(
        q, k_poison, v_poison, selected, _SCALE, 0, key_tile=64, dimension_tile=64
    )
    mx.eval(clean, poisoned)
    assert mx.array_equal(clean[:, :, :1, :], poisoned[:, :, :1, :]).item()


def test_early_prefix_matches_gather_with_upstream_bound(native_lane):
    """complete < 512 on every row. Native symbol + gather oracle.

    First row is one visible token → bit-identical. Rest: max_error <= 2e-2.
    """

    mx.random.seed(313)
    rows = 33
    q = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, rows, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, rows, 256)).astype(mx.bfloat16)
    ids, valid = _selection(0, rows)
    native = direct._EXT.qwen4_qsa_sparse_gqa_attention(
        q,
        k,
        v,
        mx.contiguous(ids.astype(mx.uint32)[None, None]),
        _SCALE,
        0,
        key_tile=64,
        dimension_tile=64,
    )
    ref = _qsa_prefill_gather_attention(
        q,
        k,
        v,
        ids,
        valid,
        pos_start=0,
        total_tokens=rows,
        compress_ratio=_RATIO,
        scale=_SCALE,
        tile_rows=8,
    )
    mx.eval(native, ref)
    assert mx.array_equal(native[:, :, :1, :], ref[:, :, :1, :]).item()
    err = float(
        mx.max(mx.abs(native.astype(mx.float32) - ref.astype(mx.float32))).item()
    )
    assert err <= _EARLY_BOUND, err


def test_missing_metallib_fails_preflight_in_a_fresh_process(native_lane):
    """FAILED is process-wide, so this drill cannot share the pytest process."""

    ext_dir = Path(direct.__file__).resolve().parents[2] / "native_extensions" / "qsa_kernels"
    metallibs = list((ext_dir / "mtplx_qsa_kernels").glob("*.metallib"))
    assert metallibs, f"no metallib next to the extension in {ext_dir}"
    metallib = metallibs[0]
    child = r"""
import os, sys, traceback
sys.path.insert(0, os.environ["MTPLX_SRC"])
# Hide the metallib before the extension's first eval.
src = os.environ["METALLIB"]
dst = src + ".hidden"
os.rename(src, dst)
try:
    import mlx.core as mx  # noqa: F401
    from mtplx.kernels.qsa_prefill_direct import (
        qsa_prefill_direct_preflight,
        qsa_prefill_direct_ready,
    )
    ok = qsa_prefill_direct_preflight()
    ready = qsa_prefill_direct_ready()
    print(f"preflight={ok} ready={ready}")
    sys.exit(0 if (ok is False and ready is False) else 2)
except Exception:
    traceback.print_exc()
    sys.exit(2)
finally:
    if os.path.exists(dst) and not os.path.exists(src):
        os.rename(dst, src)
"""
    env = os.environ.copy()
    src_root = str(Path(direct.__file__).resolve().parents[2])
    env["MTPLX_SRC"] = src_root
    env["METALLIB"] = str(metallib)
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    hidden = Path(str(metallib) + ".hidden")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", child],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        if hidden.is_file() and not metallib.is_file():
            hidden.rename(metallib)
    assert metallib.is_file(), "child must restore the metallib"
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "preflight=False ready=False" in proc.stdout
