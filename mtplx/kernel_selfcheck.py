"""Load-time exactness self-validation for the turbo kernel lanes.

Why this exists (2026-07-07, turbo-everywhere promotion): turbo became the
default profile for every dense catalog model, which means the MTPLX Metal
kernels now run on Apple GPU families nobody has physically measured (M1, M2,
M3, M3 Ultra). Every turbo lane is plain-SIMD and dtype-templated, so it
*should* be portable — but "should" is not a product guarantee. This module
turns it into one: at model load, each lane that can engage is run once on
tiny synthetic tensors in the model's actual dtype/quant format and compared
against the stock MLX reference. A lane that mismatches is disabled for the
process (serving falls back to the proven stock path for that lane only) and
the verdict is surfaced in ``/health`` as ``kernel_selfcheck``.

The check is a corruption tripwire, not a ULP certifier: thresholds sit ~10x
above the accumulation-order ULP band of each lane and ~10x below the
magnitude a broken kernel produces (wrong indexing, bad intrinsic, garbage
memory). Hot-shape ULP exactness is gated separately by the CI kernel matrix
and the release exactness gates.

The two Ternary Bonsai (Prism) kernels, the fused Hadamard rotation and the
ternary GEMV, are plain SIMD as well and run on every GPU generation. They
are probed whenever a Prism model loads with their switches on, whatever the
profile: the rotation must return the MLX chain's exact bits, the GEMV must
agree with stock ``mx.quantized_matmul`` within float16 rounding.

The four Flash-Next (qwen4_exp) prefill kernels, the hyper-connection read,
the GDN gated norm, the GDN prefill prework and the MoE prefill combine, are
plain SIMD too and on by default on every Mac. They are probed whenever a
Flash-Next model loads with their switches on, whatever the profile, at the
family's geometry and production thread counts: each must return the stock
MLX chain's exact bits.

Env:
- ``MTPLX_KERNEL_SELFCHECK=0`` disables the probe (default: runs whenever a
  turbo kernel env is active, and for every Prism and Flash-Next model).
- ``MTPLX_FORCE_GPU_FAMILY_FALLBACK=1`` (handled in ``nax_verify``) forces the
  G17-gated m16 NAX lane off so newer machines can rehearse the exact
  M1-M4 code path.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

_STATUS_OK = "ok"
_STATUS_FALLBACK = "fallback"
_STATUS_SKIPPED = "skipped"

# lane -> status ("ok" | "fallback" | "skipped"); empty until a run happens.
_LANE_STATUS: dict[str, str] = {}
_DISABLED_LANES: set[str] = set()
_LAST_REPORT: dict[str, Any] = {}

# Max |kernel - stock| tolerated per lane family at the fixture scales below
# (x ~ N(0, 0.5), w ~ N(0, 0.02), K = 1024). The qmm lanes' fp32-accumulate /
# lane-strided reduction differs from stock by low single-digit bf16 ULPs
# (~0.006 at these magnitudes); corruption lands at O(1) or NaN.
_QMM_TOLERANCE = 0.1
_SDPA_TOLERANCE = 0.02
_NORM_TOLERANCE = 0.02
# Ternary GEMV vs stock on its fixture (outputs ~N(0, 0.2), all under 1):
# both sum in float32 and round once to float16, so they differ by a float16
# step or two (<= 1e-3); a wrong decode or index is off by the output scale.
_TERNARY_TOLERANCE = 0.02
# Flash-Next prefill probes: the narrowest width the call sites admit (32 rows)
# at the family's geometry (top-10 experts of hidden 2560), so every kernel
# runs with the thread counts production dispatches.
_FLASH_NEXT_ROWS = 32
_FLASH_NEXT_TOP_K = 10
_FLASH_NEXT_HIDDEN = 2560

_K = 1024  # satisfies every lane's K divisibility contract (%256 for m16)
_N = 1024  # satisfies N%32 (m16/msg) and N%4 (ksplit)


def _env_on(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "on", "yes"}


def selfcheck_enabled(*, prism_ternary: bool = False, flash_next: bool = False) -> bool:
    """Selfcheck runs by default whenever a turbo kernel lane is active.

    ``prism_ternary``: the model is a Prism (Ternary Bonsai) load, whose two
    kernels run under every profile, so it is checked under every profile.
    ``flash_next``: the model is a Flash-Next (qwen4_exp) load, likewise for
    its four prefill kernels.
    """
    raw = str(os.environ.get("MTPLX_KERNEL_SELFCHECK", "")).strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    return (
        prism_ternary
        or flash_next
        or _env_on("MTPLX_NAX_VERIFY")
        or _env_on("MTPLX_GQA_PACKED_SDPA")
        or _env_on("MTPLX_QWEN_ROW_OWNED_ROUTER")
        or _env_on("MTPLX_QWEN_COMBINE_TAIL")
        or _env_on("MTPLX_FUSE_GDN_POST_CONV")
        or _env_on("MTPLX_A3B_WHOLE_MOE_FUSION")
    )


def lane_disabled(lane: str) -> bool:
    return lane in _DISABLED_LANES


def report_for_health() -> dict[str, Any]:
    """JSON-primitive-only payload for the ``/health`` endpoint."""
    payload: dict[str, Any] = {
        "ran": bool(_LAST_REPORT),
    }
    if _LAST_REPORT:
        payload["elapsed_ms"] = float(_LAST_REPORT.get("elapsed_ms", 0.0))
        payload["dtype"] = str(_LAST_REPORT.get("dtype", ""))
        payload["bits"] = int(_LAST_REPORT.get("bits", 0) or 0)
    for lane, status in sorted(_LANE_STATUS.items()):
        payload[lane] = status
    return payload


def _reset_for_tests() -> None:
    _LANE_STATUS.clear()
    _DISABLED_LANES.clear()
    _LAST_REPORT.clear()


def _quantized_fixture(mx, K: int, N: int, bits: int, group_size: int, dtype):
    mx.random.seed(7)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _qmm_reference(mx, x, w_q, scales, biases, *, bits: int, group_size: int):
    return mx.quantized_matmul(
        x,
        w_q,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


def _max_abs_diff(mx, candidate, reference) -> float:
    diff = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
    value = float(diff.max())
    if value != value:  # NaN
        return float("inf")
    return value


def _check_qmm_lane(mx, fn, m: int, bits: int, group_size: int, dtype) -> float:
    w_q, scales, biases = _quantized_fixture(mx, _K, _N, bits, group_size, dtype)
    x = (mx.random.normal((m, _K), dtype=mx.float32) * 0.5).astype(dtype)
    y = fn(x, w_q, scales, biases)
    ref = _qmm_reference(mx, x, w_q, scales, biases, bits=bits, group_size=group_size)
    if tuple(y.shape) != tuple(ref.shape):
        return float("inf")
    return _max_abs_diff(mx, y, ref)


def _check_qwen_row_owned_router(mx, dtype) -> float:
    """Require bitwise stock routing for every installed M1-M16 row count."""

    if dtype != mx.bfloat16:
        return float("inf")
    from .qwen_row_owned_router import qwen_row_owned_route

    fixture = mx.arange(16 * 256, dtype=mx.float32).reshape(16, 256)
    logits = (
        mx.sin(fixture * 0.017) * 0.5
        + mx.cos(fixture * 0.031) * 0.125
    ).astype(dtype)
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    for rows in range(1, 17):
        current = probabilities[:rows]
        stock_ids = mx.argpartition(current, kth=-8, axis=-1)[..., -8:]
        stock_scores = mx.take_along_axis(current, stock_ids, axis=-1)
        stock_scores = stock_scores / stock_scores.sum(axis=-1, keepdims=True)
        candidate_ids, candidate_scores = qwen_row_owned_route(current)
        mx.eval(stock_ids, stock_scores, candidate_ids, candidate_scores)
        if not bool(mx.array_equal(candidate_ids, stock_ids).item()):
            return float("inf")
        if not bool(mx.array_equal(candidate_scores, stock_scores).item()):
            return float("inf")
    return 0.0


def _check_qwen_combine_tail_m1_m2(mx, dtype) -> float:
    """Require bitwise stock arithmetic for every installed combine shape."""

    if dtype != mx.bfloat16:
        return float("inf")
    from .qwen_row_owned_router import (
        qwen_combine_tail_m1,
        qwen_combine_tail_m16,
        qwen_combine_tail_m2,
        qwen_combine_tail_m8,
    )

    for rows, entrypoint in (
        (1, qwen_combine_tail_m1),
        (2, qwen_combine_tail_m2),
        (8, qwen_combine_tail_m8),
        (16, qwen_combine_tail_m16),
    ):
        routed_fixture = mx.arange(
            rows * 8 * 2048, dtype=mx.float32
        ).reshape(1, rows, 8, 2048)
        routed = (
            mx.sin(routed_fixture * 0.013) * 0.5
            + mx.cos(routed_fixture * 0.007) * 0.125
        ).astype(dtype)
        score_fixture = mx.arange(rows * 8, dtype=mx.float32).reshape(
            1, rows, 8
        )
        scores = mx.softmax(
            mx.sin(score_fixture * 0.11)
            + mx.cos(score_fixture * 0.07) * 0.25,
            axis=-1,
        ).astype(dtype)
        stock = (routed * scores[..., None]).sum(axis=-2)
        candidate = entrypoint(routed, scores)
        mx.eval(stock, candidate)
        if tuple(candidate.shape) != (1, rows, 2048):
            return float("inf")
        if not bool(mx.array_equal(candidate, stock).item()):
            return float("inf")
    return 0.0


def _check_gqa_packed(mx, dtype, kernel=None, *, d=128, q_len=4) -> float:
    from .kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail

    hq, hk = (24, 4) if d == 256 else (8, 2)
    capacity, offset = 512, 200
    scale = d**-0.5
    mx.random.seed(11)
    queries = (mx.random.normal((1, hq, q_len, d), dtype=mx.float32) * 0.5).astype(dtype)
    keys = (mx.random.normal((1, hk, capacity, d), dtype=mx.float32) * 0.5).astype(dtype)
    values = (mx.random.normal((1, hk, capacity, d), dtype=mx.float32) * 0.5).astype(dtype)
    out = (kernel or sdpa_gqa_packed_tail)(
        queries=queries,
        keys=keys,
        values=values,
        offset=offset,
        scale=scale,
    )
    if out is None:
        return float("inf")
    # Tail-causal reference: query row j attends to rows n <= offset - q_len + j.
    rows = mx.arange(q_len).reshape(q_len, 1)
    cols = mx.arange(offset).reshape(1, offset)
    mask = cols <= (offset - q_len + rows)
    ref = mx.fast.scaled_dot_product_attention(
        queries,
        keys[:, :, :offset, :],
        values[:, :, :offset, :],
        scale=scale,
        mask=mask,
    )
    return _max_abs_diff(mx, out, ref)


def _check_fused_add_rmsnorm(mx, dtype) -> float:
    from .gdn_capture import _fused_post_norm_tg_override
    from .kernels.fused_norm import fused_add_rmsnorm

    # Probe the production configuration (same threadgroup resolution as the
    # gdn_capture call site) at real model widths. The pre-#319 probe used
    # axis=512 with a hardcoded threadgroup_size=512 — the one width where a
    # forced 512-lane loop matches the reference partition, so it validated a
    # configuration production never hit and stayed green while axes 3072/5120
    # flipped fp16 ULPs from 64 rows up. This lane claims bitwise identity, so
    # its tolerance at the _record call site is 0.0 — never widen it back.
    mx.random.seed(13)
    tg = _fused_post_norm_tg_override()
    worst = 0.0
    for axis in (512, 3072, 5120):
        weight = (mx.random.normal((axis,), dtype=mx.float32) * 0.1 + 1.0).astype(dtype)
        for rows in (1, 4, 128):
            x = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
            residual = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
            eps = 1e-6
            h, normed = fused_add_rmsnorm(x, residual, weight, eps, threadgroup_size=tg)
            ref_h = x + residual
            ref_normed = mx.fast.rms_norm(ref_h, weight, eps).astype(dtype)
            worst = max(
                worst,
                _max_abs_diff(mx, h, ref_h),
                _max_abs_diff(mx, normed, ref_normed),
            )
    return worst


def _check_fused_gdn_norm_gate(mx, dtype) -> float:
    from .kernels.fused_norm import fused_gdn_norm_gate

    mx.random.seed(17)
    rows, axis = 4, 128
    x = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    gate = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    weight = (mx.random.normal((axis,), dtype=mx.float32) * 0.1 + 1.0).astype(dtype)
    eps = 1e-6
    y = fused_gdn_norm_gate(x, gate, weight, eps)
    normed = mx.fast.rms_norm(x, weight, eps)
    gate_f = gate.astype(mx.float32)
    ref = (gate_f * mx.sigmoid(gate_f) * normed.astype(mx.float32)).astype(dtype)
    return _max_abs_diff(mx, y, ref)


def _check_gdn_postconv_inline_g(mx, dtype) -> float:
    """Compare the exact A3B M1/M2 stock captures with their fixed routes."""
    if dtype != mx.bfloat16:
        return float("inf")

    from .gdn_capture import (
        _a3b_compiled_target_gdn_postconv_m1_tgy4,
        _a3b_compiled_target_gdn_postconv_m2_tgy4,
        _stock_gated_delta_capture,
    )

    conv_values = mx.arange(2 * 8192, dtype=mx.float32).reshape(1, 2, 8192)
    conv_out = (mx.sin(conv_values * 0.013) * 0.5).astype(mx.bfloat16)
    gate_values = mx.arange(64, dtype=mx.float32).reshape(1, 2, 32)
    a = (mx.sin(gate_values * 0.11) * 0.5).astype(mx.bfloat16)
    b = (mx.cos(gate_values * 0.07) * 0.5).astype(mx.bfloat16)
    state_values = mx.arange(32 * 128 * 128, dtype=mx.float32).reshape(
        1, 32, 128, 128
    )
    state = mx.sin(state_values * 0.001) * 0.1
    gdn = SimpleNamespace(
        A_log=mx.linspace(0.0, 2.0, 32).astype(dtype),
        dt_bias=mx.linspace(-5.0, -3.0, 32).astype(dtype),
        conv_dim=8192,
        key_dim=2048,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        training=False,
    )
    inv_scale = 128**-0.5
    routes = (
        (1, _a3b_compiled_target_gdn_postconv_m1_tgy4),
        (2, _a3b_compiled_target_gdn_postconv_m2_tgy4),
    )
    differences = []
    for logical_m, route in routes:
        route_conv = conv_out[:, :logical_m]
        route_a = a[:, :logical_m]
        route_b = b[:, :logical_m]
        q, k, v = [
            tensor.reshape(1, logical_m, heads, 128)
            for tensor, heads in zip(
                mx.split(route_conv, [2048, 4096], axis=-1),
                [16, 16, 32],
            )
        ]
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        ref_out, ref_states = _stock_gated_delta_capture(
            q,
            k,
            v,
            route_a,
            route_b,
            state,
            None,
            gdn,
        )
        out, states = route(
            route_conv,
            route_a,
            route_b,
            state,
            A_log=gdn.A_log,
            dt_bias=gdn.dt_bias,
        )
        mx.eval(ref_out, ref_states, out, states)
        if tuple(out.shape) != tuple(ref_out.shape) or tuple(states.shape) != tuple(
            ref_states.shape
        ):
            return float("inf")
        differences.extend(
            (
                _max_abs_diff(mx, out, ref_out),
                _max_abs_diff(mx, states, ref_states),
            )
        )
    return max(differences)


def _check_gdn_postconv_headquarter(mx, dtype) -> float:
    """Compare the exact A3B M1/M2 stock captures with the C1 headquarter routes."""
    if dtype != mx.bfloat16:
        return float("inf")

    from .gdn_capture import (
        _a3b_compiled_target_gdn_postconv_m1_headquarter,
        _a3b_compiled_target_gdn_postconv_m2_headquarter,
        _stock_gated_delta_capture,
    )

    conv_values = mx.arange(2 * 8192, dtype=mx.float32).reshape(1, 2, 8192)
    conv_out = (mx.sin(conv_values * 0.013) * 0.5).astype(mx.bfloat16)
    gate_values = mx.arange(64, dtype=mx.float32).reshape(1, 2, 32)
    a = (mx.sin(gate_values * 0.11) * 0.5).astype(mx.bfloat16)
    b = (mx.cos(gate_values * 0.07) * 0.5).astype(mx.bfloat16)
    state_values = mx.arange(32 * 128 * 128, dtype=mx.float32).reshape(
        1, 32, 128, 128
    )
    state = mx.sin(state_values * 0.001) * 0.1
    gdn = SimpleNamespace(
        A_log=mx.linspace(0.0, 2.0, 32).astype(dtype),
        dt_bias=mx.linspace(-5.0, -3.0, 32).astype(dtype),
        conv_dim=8192,
        key_dim=2048,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        training=False,
    )
    inv_scale = 128**-0.5
    routes = (
        (1, _a3b_compiled_target_gdn_postconv_m1_headquarter),
        (2, _a3b_compiled_target_gdn_postconv_m2_headquarter),
    )
    differences = []
    for logical_m, route in routes:
        route_conv = conv_out[:, :logical_m]
        route_a = a[:, :logical_m]
        route_b = b[:, :logical_m]
        q, k, v = [
            tensor.reshape(1, logical_m, heads, 128)
            for tensor, heads in zip(
                mx.split(route_conv, [2048, 4096], axis=-1),
                [16, 16, 32],
            )
        ]
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        ref_out, ref_states = _stock_gated_delta_capture(
            q,
            k,
            v,
            route_a,
            route_b,
            state,
            None,
            gdn,
        )
        out, states = route(
            route_conv,
            route_a,
            route_b,
            state,
            A_log=gdn.A_log,
            dt_bias=gdn.dt_bias,
        )
        mx.eval(ref_out, ref_states, out, states)
        if tuple(out.shape) != tuple(ref_out.shape) or tuple(states.shape) != tuple(
            ref_states.shape
        ):
            return float("inf")
        differences.extend(
            (
                _max_abs_diff(mx, out, ref_out),
                _max_abs_diff(mx, states, ref_states),
            )
        )
    return max(differences)


def _check_prism_fused_rotation(mx) -> float:
    """Bitwise: the fused rotation must return the MLX chain's exact bits.

    Forward and inverse, three rows of two blocks, with the heavy tails and
    tiny values a residual stream carries. Any differing bit fails the lane,
    and so does a build failure (it raises from the eval).
    """
    from .kernels import hadamard_rotate as hr
    from .models.prism_hadamard_qwen35 import mlx_chain_rotate

    mx.random.seed(19)
    shape = (3, 2 * hr.BLOCK)
    x = mx.random.normal(shape) * 2.0
    x = x + (mx.random.uniform(shape=shape) < 0.01) * mx.random.normal(shape) * 300.0
    x = (x + (mx.random.uniform(shape=shape) < 0.05) * 1e-4).astype(mx.float16)
    signs = mx.where(
        mx.random.uniform(shape=(shape[-1],)) < 0.5, -1.0, 1.0
    ).astype(mx.float32)
    worst = 0.0
    for inverse in (False, True):
        got = hr._launch(x, signs, inverse)
        want = mlx_chain_rotate(x, signs, hr.BLOCK, inverse=inverse)
        mx.eval(got, want)
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            return float("inf")
        if not bool(mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()):
            # Differing bits at zero distance (signed zeros) still fail.
            worst = max(worst, _max_abs_diff(mx, got, want) or float("inf"))
    return worst


def _ternary_fixture(mx, n: int, k: int):
    """A ternary 2-bit/g128 matrix in Prism's layout (biases == -scales)."""
    mx.random.seed(23)
    codes = mx.random.randint(0, 3, (n, k)).astype(mx.uint32)
    shifts = (mx.arange(16, dtype=mx.uint32) * 2)[None, None, :]
    words = (codes.reshape(n, k // 16, 16) << shifts).sum(axis=-1).astype(mx.uint32)
    scales = (0.004 + 0.02 * mx.random.uniform(shape=(n, k // 128))).astype(mx.float16)
    return words, scales, -scales


def _check_bonsai_ternary_qmv(mx) -> float:
    """The ternary GEMV against stock mx.quantized_matmul on a ternary matrix.

    Rows 1 to 4 (decode and verify) in both production geometries (the
    projections' eight simdgroups and the vocabulary head's four), with
    K = 1024 so every lane loops twice. Returns the worst difference relative
    to the output scale; a build failure or a threadgroup MLX refuses on this
    GPU raises from the eval.
    """
    from .kernels import ternary_qmv as tq

    n, k = 2048, 1024
    w, scales, biases = _ternary_fixture(mx, n, k)
    worst = 0.0
    for rows in range(1, tq.MAX_ROWS + 1):
        x = (mx.random.normal((rows, k)) * 0.5).astype(mx.float16)
        ref = mx.quantized_matmul(
            x,
            w,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=tq.GROUP_SIZE,
            bits=tq.BITS,
        )
        for r, sg in (
            (tq.ROWS_PER_SIMDGROUP, tq.SIMDGROUPS),
            (tq.ROWS_PER_SIMDGROUP, tq.HEAD_SIMDGROUPS),
        ):
            got = tq._launch(x, w, scales, rows, r, sg)
            if tuple(got.shape) != tuple(ref.shape):
                return float("inf")
            scale = max(1.0, float(mx.abs(ref.astype(mx.float32)).max()))
            worst = max(worst, _max_abs_diff(mx, got, ref) / scale)
    return worst


def _bitwise_worst(mx, got, want) -> float:
    """0.0 when the bf16 ``got`` carries ``want``'s exact bits, else how far apart.

    A shape or dtype mismatch is infinite, and so is a bit difference at zero
    distance (a signed zero or a NaN).
    """
    mx.eval(got, want)
    if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
        return float("inf")
    if bool(mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()):
        return 0.0
    return _max_abs_diff(mx, got, want) or float("inf")


def _check_qwen4_hc_prefill_read(mx) -> float:
    """Bitwise: the hyper-connection read's grouped norm and sigmoid mix.

    Both kernels at the family geometry (4 streams of 2560, 32 rows), so the
    norm runs its 640-thread groups and the mix its 256-thread groups, against
    ``mx.fast.rms_norm`` times the weight and the mean of ``sigmoid(up)``
    times the norm. The projections between them are stock layers either way.
    A build failure or a thread count this GPU refuses raises from the eval.
    """
    from .kernels import hc_prefill as hc

    mx.random.seed(29)
    rows, streams, hidden = _FLASH_NEXT_ROWS, hc._HC, hc._HIDDEN
    grouped = (1, rows, streams, hidden)
    x = (mx.random.normal((1, rows, streams * hidden)) * 3.0).astype(mx.bfloat16)
    weight = (1.0 + 0.1 * mx.random.normal((streams * hidden,))).astype(mx.bfloat16)
    up = (mx.random.normal(x.shape) * 12.0).astype(mx.bfloat16)
    want_norm = mx.fast.rms_norm(x.reshape(grouped), None, 1e-6).reshape(x.shape) * weight
    want_mix = mx.mean(mx.sigmoid(up).reshape(grouped) * want_norm.reshape(grouped), axis=-2)
    return max(
        _bitwise_worst(mx, hc._normalize(x, weight, 1e-6), want_norm),
        _bitwise_worst(mx, hc._mix(up, want_norm), want_mix),
    )


def _check_qwen4_gdn_gated_norm(mx) -> float:
    """Bitwise: the GDN norm's sigmoid output gate, for bf16 and float32 norms.

    A prefill chunk's ``[1, 32, 48, 128]`` heads in 256-thread groups, the
    gate sliced from a wider projection as in the model, against
    ``(sigmoid(gate.astype(f32)) * x.astype(f32)).astype(bf16)``.
    """
    from .kernels import gdn_gated_norm as gn

    mx.random.seed(31)
    rows = _FLASH_NEXT_ROWS
    heads = (1, rows, 48, 128)
    out = (mx.random.normal(heads) * 2.0).astype(mx.bfloat16)
    projection = (mx.random.normal((1, rows, 16480)) * 3.0).astype(mx.bfloat16)
    gate = projection[..., 10240:16384].reshape(heads)
    worst = 0.0
    for weight_dtype in (mx.bfloat16, mx.float32):
        weight = (1.0 + 0.1 * mx.random.normal((128,))).astype(weight_dtype)
        x = mx.fast.rms_norm(out, weight, 1e-6)
        want = (mx.sigmoid(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(
            mx.bfloat16
        )
        worst = max(worst, _bitwise_worst(mx, gn.sigmoid_gate(x, gate), want))
    return worst


def _check_qwen4_gdn_prefill_prework(mx) -> float:
    """Bitwise: conv + silu + q/k l2norm against the model's staged chain.

    32 rows of the family's q|k|v stream (10240 channels, a strided view of
    the wider projection as the model passes it) after a live conv state, one
    32-thread simdgroup per row and head as in production; q, k and v must
    all carry the chain's exact bits.
    """
    import mlx.nn as nn

    from .kernels import gdn_prefill_prework as pw

    mx.random.seed(37)
    rows, dim, key, head, taps = (
        _FLASH_NEXT_ROWS,
        pw._CONV_DIM,
        pw._KEY_DIM,
        pw._HEAD,
        pw._KERNEL,
    )
    qkv = mx.random.normal((1, rows, 16480)).astype(mx.bfloat16)[..., :dim]
    conv_state = mx.random.normal((1, taps - 1, dim)).astype(mx.bfloat16)
    conv_w = (mx.random.normal((dim, taps, 1)) * 0.5).astype(mx.bfloat16)
    inv_scale = head**-0.5
    got = pw.gdn_prefill_prework(qkv, conv_state, conv_w, inv_scale)

    conv_out = nn.silu(
        mx.conv1d(mx.concatenate([conv_state, qkv], axis=1), conv_w, groups=dim)
    )
    q, k, v = (
        t.reshape(1, rows, -1, head) for t in mx.split(conv_out, [key, 2 * key], -1)
    )

    def l2norm(t):
        tf = t.astype(mx.float32)
        return (tf * mx.rsqrt((tf * tf).sum(-1, keepdims=True) + 1e-6)).astype(t.dtype)

    want = (inv_scale * l2norm(q), l2norm(k), v)
    return max(_bitwise_worst(mx, g, w) for g, w in zip(got, want))


def _check_qwen4_moe_prefill_combine(mx) -> float:
    """Bitwise: the MoE combine against the stock unsort, weight, sum, + shared tail.

    32 rows of Flash-Next's top-10 of 2560 in the kernel's production
    256-thread groups, the expert outputs in a shuffled order as the sorted
    gather leaves them.
    """
    from .kernels import qwen4_moe_prefill_combine as mc

    mx.random.seed(41)
    rows, top_k, hidden = _FLASH_NEXT_ROWS, _FLASH_NEXT_TOP_K, _FLASH_NEXT_HIDDEN
    y_sorted = mx.random.normal((rows * top_k, hidden)).astype(mx.bfloat16)
    inv_order = mx.argsort(mx.random.uniform(shape=(rows * top_k,))).astype(mx.uint32)
    gates = mx.softmax(
        (mx.random.normal((rows, top_k)) * 3.0).astype(mx.bfloat16), axis=-1, precise=True
    )
    scores = gates / gates.sum(axis=-1, keepdims=True)
    shared = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    return _bitwise_worst(
        mx,
        mc._launch(y_sorted, inv_order, scores, shared),
        mc.moe_prefill_combine_reference(y_sorted, inv_order, scores, shared),
    )


def run_kernel_selfcheck(
    dtype,
    bits: int,
    group_size: int,
    *,
    prism_ternary: bool = False,
    flash_next: bool = False,
) -> dict[str, Any]:
    """Probe every turbo lane that can engage for this model configuration.

    ``prism_ternary`` adds the two Ternary Bonsai kernel lanes (a Prism load),
    ``flash_next`` the four Flash-Next prefill kernel lanes (a qwen4_exp load).

    Returns ``{"lanes": {lane: status}, "dmax": {lane: float}, ...}`` and
    updates the process-wide disable registry: lanes reported ``fallback``
    stop engaging (their call sites route the stock path) until the process
    restarts. Idempotent — each run rebuilds the registry from scratch.
    """
    import mlx.core as mx

    from . import nax_verify

    bits = int(bits)
    group_size = int(group_size)
    started = time.perf_counter()

    lanes: dict[str, str] = {}
    dmax: dict[str, float] = {}

    def _record(lane: str, tolerance: float, probe) -> None:
        try:
            value = float(probe())
        except Exception as exc:  # any kernel-side failure means fallback
            logger.warning(
                "[mtplx] kernel selfcheck: %s raised (%s) — falling back to stock",
                lane,
                exc,
            )
            lanes[lane] = _STATUS_FALLBACK
            dmax[lane] = float("inf")
            return
        dmax[lane] = value
        if value <= tolerance:
            lanes[lane] = _STATUS_OK
        else:
            logger.warning(
                "[mtplx] kernel selfcheck: %s mismatched (dmax=%.4g) — "
                "falling back to stock",
                lane,
                value,
            )
            lanes[lane] = _STATUS_FALLBACK

    nax_on = _env_on("MTPLX_NAX_VERIFY")
    if nax_on and bits == 4:
        # 4-bit routes (nax_verify.patched): m4 -> env-selected impl (turbo
        # ships vk_k split-K), m5..6 -> legacy m6 ksplit, m7..16 -> NAX tile.
        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: nax_verify.nax_qmm_m4(x, w, s, b, group_size=group_size),
                4,
                4,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: nax_verify.nax_qmm_m6(x, w, s, b, group_size=group_size),
                6,
                4,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m4_wide"] = _STATUS_SKIPPED  # single m4 impl covers all N at 4-bit
        lanes["qmm_m6_wide"] = _STATUS_SKIPPED
        if nax_verify.m5_rows_lane_enabled():
            # Opt-in exact 5-row lane (MTPLX_M5_ROWS_LANE=1); only probed, and
            # only reported, when the switch is on. A mismatch routes M=5 back
            # to stock. The pipelined m16 tile (MTPLX_NAX_M16_PIPELINED=1) is
            # covered by the qmm_m16_nax probe below, which runs whichever
            # kernel the switch selects.
            _record(
                "qmm_m5",
                _QMM_TOLERANCE,
                lambda: _check_qmm_lane(
                    mx,
                    lambda x, w, s, b: nax_verify.nax_qmm_m5(x, w, s, b, group_size=group_size),
                    5,
                    4,
                    group_size,
                    dtype,
                ),
            )
        if nax_verify.nax_available():
            _record(
                "qmm_m16_nax",
                _QMM_TOLERANCE,
                lambda: _check_qmm_lane(
                    mx,
                    lambda x, w, s, b: nax_verify.nax_qmm_m16(x, w, s, b, group_size=group_size),
                    16,
                    4,
                    group_size,
                    dtype,
                ),
            )
            if nax_verify.m32_tile_enabled():
                _record(
                    "qmm_m32_nax",
                    _QMM_TOLERANCE,
                    lambda: _check_qmm_lane(
                        mx,
                        lambda x, w, s, b: nax_verify.nax_qmm_m32(x, w, s, b, group_size=group_size),
                        24,
                        4,
                        group_size,
                        dtype,
                    ),
                )
        else:
            lanes["qmm_m16_nax"] = _STATUS_SKIPPED
    elif nax_on and bits == 8:
        # 8-bit routes (nax_verify.patched): vk split-K for layer shapes,
        # vk msg wide tile for huge-N (lm_head-class) shapes.
        from .verify_kernels import (
            vk_qmm_m4,
            vk_qmm_m4_ksplit,
            vk_qmm_m6,
            vk_qmm_m6_ksplit,
        )

        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4_ksplit(x, w, s, b, bits=8, group_size=group_size),
                4,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m4_wide",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4(x, w, s, b, bits=8, group_size=group_size),
                4,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6_ksplit(x, w, s, b, bits=8, group_size=group_size),
                6,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6_wide",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6(x, w, s, b, bits=8, group_size=group_size),
                6,
                8,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m16_nax"] = _STATUS_SKIPPED  # 4-bit-only tile
    elif nax_on and bits == 6:
        # 6-bit routes (9B tier, 2026-07-07): split-K hexpack kernels only.
        from .verify_kernels import vk_qmm_m4_ksplit, vk_qmm_m6_ksplit

        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4_ksplit(x, w, s, b, bits=6, group_size=group_size),
                4,
                6,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6_ksplit(x, w, s, b, bits=6, group_size=group_size),
                6,
                6,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m4_wide"] = _STATUS_SKIPPED  # split-K only at 6-bit
        lanes["qmm_m6_wide"] = _STATUS_SKIPPED
        lanes["qmm_m16_nax"] = _STATUS_SKIPPED  # 4-bit-only tile
    else:
        for lane in ("qmm_m4", "qmm_m4_wide", "qmm_m6", "qmm_m6_wide", "qmm_m16_nax"):
            lanes[lane] = _STATUS_SKIPPED

    # lm_head_topk kernels exist but are not routed on the serve path.
    lanes["lm_head_topk"] = _STATUS_SKIPPED

    if _env_on("MTPLX_QWEN_ROW_OWNED_ROUTER"):
        _record(
            "qwen_row_owned_router",
            0.002,
            lambda: _check_qwen_row_owned_router(mx, dtype),
        )
    else:
        lanes["qwen_row_owned_router"] = _STATUS_SKIPPED

    if _env_on("MTPLX_QWEN_COMBINE_TAIL"):
        _record(
            "qwen_combine_tail_m1_m2",
            0.0,
            lambda: _check_qwen_combine_tail_m1_m2(mx, dtype),
        )
    else:
        lanes["qwen_combine_tail_m1_m2"] = _STATUS_SKIPPED

    if _env_on("MTPLX_GQA_PACKED_SDPA"):
        _record("gqa_packed_sdpa", _SDPA_TOLERANCE, lambda: _check_gqa_packed(mx, dtype))
    else:
        lanes["gqa_packed_sdpa"] = _STATUS_SKIPPED

    # NAX attention has its own kernels: validating the scalar packed lane
    # cannot certify these. Probe both physical geometries before serving,
    # and retain the established packed route on non-G17 GPUs / older macOS.
    from .kernels.sdpa_nax_flash import sdpa_nax_flash
    from .kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
    from .kernels.sdpa_nax_tile import sdpa_nax_tile

    for lane, env, kernel, rows in (
        ("nax_flash_sdpa", "MTPLX_NAX_FLASH_ROUTE", sdpa_nax_flash, 8),
        ("nax_flash_dsplit_sdpa", "MTPLX_NAX_FLASH_ROUTE", sdpa_nax_flash_dsplit, 4),
        ("nax_tile_sdpa", "MTPLX_NAX_TILE_ROUTE", sdpa_nax_tile, 8),
    ):
        if _env_on("MTPLX_GQA_PACKED_SDPA") and _env_on(env) and nax_verify.nax_available():
            _record(lane, _SDPA_TOLERANCE,
                    lambda kernel=kernel, rows=rows: _check_gqa_packed(
                        mx, dtype, kernel, d=256, q_len=rows))
        else:
            lanes[lane] = _STATUS_SKIPPED

    if _env_on("MTPLX_FUSE_POST_NORM_RESIDUAL"):
        # Bitwise gate: this lane's contract is exact identity with the
        # unfused reference (#319). _NORM_TOLERANCE stays loose only for
        # fused_gdn_norm_gate, whose fp32 gate/SiLU is legitimately not
        # bitwise.
        _record(
            "fused_add_rmsnorm",
            0.0,
            lambda: _check_fused_add_rmsnorm(mx, dtype),
        )
    else:
        lanes["fused_add_rmsnorm"] = _STATUS_SKIPPED

    if _env_on("MTPLX_FUSE_GDN_NORM_GATE"):
        _record(
            "fused_gdn_norm_gate",
            _NORM_TOLERANCE,
            lambda: _check_fused_gdn_norm_gate(mx, dtype),
        )
    else:
        lanes["fused_gdn_norm_gate"] = _STATUS_SKIPPED

    if _env_on("MTPLX_FUSE_GDN_POST_CONV"):
        from .gdn_capture import _a3b_gdn_postconv_headquarter_requested

        if _a3b_gdn_postconv_headquarter_requested():
            lanes["gdn_postconv_inline_g"] = _STATUS_SKIPPED
            _record(
                "gdn_postconv_headquarter",
                0.03125,
                lambda: _check_gdn_postconv_headquarter(mx, dtype),
            )
        else:
            _record(
                "gdn_postconv_inline_g",
                0.03125,
                lambda: _check_gdn_postconv_inline_g(mx, dtype),
            )
            lanes["gdn_postconv_headquarter"] = _STATUS_SKIPPED
    else:
        lanes["gdn_postconv_inline_g"] = _STATUS_SKIPPED
        lanes["gdn_postconv_headquarter"] = _STATUS_SKIPPED

    # Ternary Bonsai (Prism) kernels. No tensor units, so no GPU-family gate:
    # this probe is what stands between an untested GPU generation and a
    # wrong answer or a failed request. switched_on(), not enabled(): a lane
    # a previous run turned off must be probed again, not skipped.
    from .kernels import hadamard_rotate, ternary_qmv

    for lane, switched_on, tolerance, probe in (
        (
            hadamard_rotate.LANE,
            hadamard_rotate.switched_on(),
            0.0,
            lambda: _check_prism_fused_rotation(mx),
        ),
        (
            ternary_qmv.LANE,
            ternary_qmv.switched_on(),
            _TERNARY_TOLERANCE,
            lambda: _check_bonsai_ternary_qmv(mx),
        ),
    ):
        if prism_ternary and switched_on:
            _record(lane, tolerance, probe)
        else:
            lanes[lane] = _STATUS_SKIPPED

    # Flash-Next (qwen4_exp) prefill kernels, on by default on every Mac with
    # no GPU-family gate either. A GPU or macOS that refuses to build or
    # dispatch one, or rounds it differently, gets the stock chain for the
    # process instead of a failed or changed long prompt. Bitwise, and
    # switched_on() for the same reason as above.
    from .kernels import (
        gdn_gated_norm,
        gdn_prefill_prework,
        hc_prefill,
        qwen4_moe_prefill_combine,
    )

    for lane, switched_on, probe in (
        (
            hc_prefill.LANE,
            hc_prefill.switched_on(),
            lambda: _check_qwen4_hc_prefill_read(mx),
        ),
        (
            gdn_gated_norm.LANE,
            gdn_gated_norm.switched_on(),
            lambda: _check_qwen4_gdn_gated_norm(mx),
        ),
        (
            gdn_prefill_prework.LANE,
            gdn_prefill_prework.switched_on(),
            lambda: _check_qwen4_gdn_prefill_prework(mx),
        ),
        (
            qwen4_moe_prefill_combine.LANE,
            qwen4_moe_prefill_combine.switched_on(),
            lambda: _check_qwen4_moe_prefill_combine(mx),
        ),
    ):
        if flash_next and switched_on:
            _record(lane, 0.0, probe)
        else:
            lanes[lane] = _STATUS_SKIPPED

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    _LANE_STATUS.clear()
    _LANE_STATUS.update(lanes)
    _DISABLED_LANES.clear()
    _DISABLED_LANES.update(
        lane for lane, status in lanes.items() if status == _STATUS_FALLBACK
    )
    dtype_tag = {mx.bfloat16: "bfloat16", mx.float16: "float16"}.get(dtype, str(dtype))
    report = {
        "lanes": dict(lanes),
        "dmax": {lane: float(value) for lane, value in dmax.items()},
        "dtype": dtype_tag,
        "bits": bits,
        "group_size": group_size,
        "elapsed_ms": elapsed_ms,
    }
    _LAST_REPORT.clear()
    _LAST_REPORT.update(report)
    return report


def _model_quant_signature(model: Any):
    """(dtype, bits, group_size) of the first quantized trunk projection."""
    import mlx.core as mx

    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    for layer in getattr(inner, "layers", []) or []:
        for attr_path in (
            ("self_attn", "q_proj"),
            ("mlp", "gate_proj"),
            ("linear_attn", "in_proj_qkvz"),
        ):
            node = layer
            for name in attr_path:
                node = getattr(node, name, None)
                if node is None:
                    break
            bits = getattr(node, "bits", None)
            if bits is None:
                continue
            group_size = int(getattr(node, "group_size", 64) or 64)
            scales = None
            try:
                scales = node["scales"]
            except Exception:
                scales = getattr(node, "scales", None)
            dtype = getattr(scales, "dtype", None) or mx.bfloat16
            return dtype, int(bits), group_size
    return None


def _prism_ternary_model(model: Any) -> bool:
    """A Prism (Ternary Bonsai) load whose loader finished its checks."""
    if not isinstance(getattr(model, "_prism_post_load_report", None), dict):
        return False
    try:
        from .models.prism_hadamard_qwen35 import Model as _PrismModel
    except Exception:  # noqa: BLE001 - never block a load; not a Prism load
        return False
    # isinstance, not the exact class: MTP injection swaps in a subclass.
    return isinstance(model, _PrismModel)


def _flash_next_model(model: Any) -> bool:
    """A Flash-Next (qwen4_exp) load, whose four prefill kernels are on by default."""
    # Never import the model module here: if a load had not imported it, this
    # model cannot be one of its instances.
    module = sys.modules.get(f"{__package__}.models.qwen4_exp")
    model_class = getattr(module, "Model", None)
    # isinstance, not the exact class: MTP injection swaps in a subclass.
    return isinstance(model_class, type) and isinstance(model, model_class)


def maybe_run_model_selfcheck(model: Any) -> dict[str, Any] | None:
    """Run the selfcheck for a freshly loaded model if turbo lanes are active.

    Called once from ``runtime.load()`` before the runtime is returned; any
    failure inside the probe itself must never break model loading.
    """
    prism_ternary = _prism_ternary_model(model)
    flash_next = _flash_next_model(model)
    if not selfcheck_enabled(prism_ternary=prism_ternary, flash_next=flash_next):
        return None
    try:
        signature = _model_quant_signature(model)
        if signature is None:
            # Unquantized trunk: the qmm lanes never engage; still validate
            # the dtype-generic attention/norm lanes with the model dtype.
            import mlx.core as mx

            dtype = mx.bfloat16
            bits = 0
            group_size = 64
        else:
            dtype, bits, group_size = signature
        report = run_kernel_selfcheck(
            dtype,
            bits,
            group_size,
            prism_ternary=prism_ternary,
            flash_next=flash_next,
        )
        fallbacks = sorted(
            lane for lane, status in report["lanes"].items() if status == _STATUS_FALLBACK
        )
        logger.info(
            "[mtplx] kernel selfcheck: %d lanes ok, %d fallback, %d skipped "
            "(%.0f ms, dtype=%s bits=%s)",
            sum(1 for s in report["lanes"].values() if s == _STATUS_OK),
            len(fallbacks),
            sum(1 for s in report["lanes"].values() if s == _STATUS_SKIPPED),
            report["elapsed_ms"],
            report["dtype"],
            report["bits"],
        )
        return report
    except Exception as exc:  # noqa: BLE001 - probe must never block serving
        logger.warning("[mtplx] kernel selfcheck failed to run: %s", exc)
        return None
