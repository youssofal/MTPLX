"""Blocked-sequential Gated DeltaNet prefill kernel.

Ported from omlx (jundot/omlx, omlx/custom_kernels/qwen35_prefill/gdn.py,
kernel S / ``gated_delta_blocked_seq``, v0.5.4rc1) — attribution per house
rules; the algorithm is the exact mlx-lm sequential recurrence, restructured
for Apple-GPU memory traffic:

- The stock mlx-lm ``gated_delta_kernel`` launches ``grid=(32, Dv, B*Hv)``
  threadgroups that each re-read the same k/q rows from device memory once
  per Dv-slice — ~32x redundant traffic (omlx measured ~13 GB per 16k-token
  layer on Qwen3.5/3.6 shapes).
- Here k/q/v/g/beta are cooperatively staged into threadgroup memory in
  TB-token blocks (coalesced), each v-head is split into Dv/32 row blocks so
  8x fewer threadgroups touch the same k/q rows, and each row is read from
  device exactly once per threadgroup.
- The recurrent state lives in registers (``float4 st[4]`` per thread: a
  thread owns one dv row x one 16-wide Dk segment; 8 threads per dv row, all
  in one simdgroup). The two contractions (k.state and q.state) reduce with
  ``simd_shuffle_down`` — no threadgroup barriers inside the token loop.
- omlx's negative result adopted with the port: the chunked WY/FLA
  reformulation costs ~2x the FLOPs and LOSES to blocked-sequential on
  Apple GPUs; do not resurrect it here.

Contract (identical to mlx-lm's scalar-gating kernel path):
  q, k: [B, T, Hk, Dk] (input dtype), v: [B, T, Hv, Dv],
  g, beta: [B, T, Hv] (cast to fp32 here), state: [B, Hv, Dv, Dk] fp32.
  Returns y [B, T, Hv, Dv] in the input dtype and the fp32 final state.

Structural requirements (checked by ``blocked_prefill_eligible``):
  Dk == 128 (8 segments x 16 lanes of fp32 state per thread), Dv % 32 == 0,
  Hv % Hk == 0, scalar gating only, no mask. Anything else must stay on the
  stock path.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_KERNEL_S_SRC = """
    constexpr int TB = 32;                             // time block
    constexpr int DB = 32;                             // dv rows per threadgroup
    const int tid = thread_position_in_threadgroup.x;  // 0..255
    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int dv0 = blk * DB;

    // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    // the same simdgroup (lane = (dv%4)*8 + seg).
    const int dv  = tid / 8;            // 0..31
    const int seg = tid % 8;            // 0..7
    const int d0  = seg * 16;

    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];

    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;

    // state fragment in registers: [dv0+dv][d0..d0+16]
    float4 st[4];
    {
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[i] = S_in[i];
    }

    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // cooperative staging (coalesced): k/q rows, v slice, g/beta
        for (int p = tid; p < tt * Dk; p += 256) {
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int p = tid; p < tt * DB; p += 256) {
            const int r = p / DB, d = p % DB;
            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
        }
        for (int p = tid; p < tt; p += 256) {
            g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            const threadgroup vec<InT,4>* k4 =
                (const threadgroup vec<InT,4>*)&k_s[t][d0];
            const threadgroup vec<InT,4>* q4 =
                (const threadgroup vec<InT,4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            // kv_mem = (g*state) . k ; decay applied to state first
            float4 p4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] *= gt;
                p4 += st[i] * kf[i];
            }
            float part = p4.x + p4.y + p4.z + p4.w;
            // reduce across the 8 segment-threads of this dv row
            part += simd_shuffle_down(part, 4);
            part += simd_shuffle_down(part, 2);
            part += simd_shuffle_down(part, 1);
            const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
            const float delta = ((float)v_s[t][dv] - kv_mem) * bt;

            float4 o4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] += kf[i] * delta;
                o4 += st[i] * float4(q4[i]);
            }
            float out = o4.x + o4.y + o4.z + o4.w;
            out += simd_shuffle_down(out, 4);
            out += simd_shuffle_down(out, 2);
            out += simd_shuffle_down(out, 1);
            if (seg == 0) {
                y_base[(size_t)(t0 + t) * Hv * Dv + dv] = (InT)out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[i];
    }
"""

_SUPPORTED_BLOCK_T = (16, 32, 48)
_kernel_by_tb: dict = {}


def _normalize_block_t(block_t, input_dtype=None) -> int:
    if block_t is None:
        configured = os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_TB")
        if configured is not None:
            block_t = configured
        else:
            # fp32 inputs at TB=32 need 40,192 bytes of threadgroup memory on
            # the 128/128 layout, over Metal's 32 KiB limit; TB=16 fits
            # (20,096 bytes). bf16/fp16 fit at TB=32. (omlx measurement.)
            block_t = 16 if input_dtype == mx.float32 else 32
    block_t = int(block_t)
    if block_t not in _SUPPORTED_BLOCK_T:
        raise ValueError(
            f"MTPLX_GDN_BLOCKED_PREFILL_TB must be one of {_SUPPORTED_BLOCK_T}, got {block_t}"
        )
    return block_t


def _get_kernel(block_t=None, input_dtype=None):
    block_t = _normalize_block_t(block_t, input_dtype)
    kernel = _kernel_by_tb.get(block_t)
    if kernel is None:
        source = _KERNEL_S_SRC.replace(
            "constexpr int TB = 32;", f"constexpr int TB = {block_t};"
        )
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_gdn_blocked_prefill_tb{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernel_by_tb[block_t] = kernel
    return kernel


def blocked_prefill_eligible(
    q: mx.array,
    v: mx.array,
    g: mx.array,
    mask,
    state,
) -> bool:
    """Structural gate: route only shapes the kernel is written for."""
    if mask is not None or g.ndim != 3:
        return False
    if q.ndim != 4 or v.ndim != 4:
        return False
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    if Dk != 128 or Dv % 32 != 0 or Hk <= 0 or Hv % Hk != 0:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if state is not None and state.dtype != mx.float32:
        return False
    return True


def gated_delta_blocked_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    block_t=None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    kernel = _get_kernel(block_t, in_dtype)
    y, state_out = kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // 32), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


# -- runtime patch -----------------------------------------------------------

_PATCH_STATE: dict = {"installed": False, "original": None}

# Component-timing accumulators: branch -> {calls, total_ms, by_t: {T: ms}}.
_COMPONENT_TIMES: dict = {}


def component_timing_report() -> dict:
    """Snapshot of the accumulated GDN component times (diagnostic mode)."""
    return {
        branch: {
            "calls": rec["calls"],
            "total_ms": round(rec["total_ms"], 2),
            "by_t": {t: round(ms, 2) for t, ms in sorted(rec["by_t"].items())},
        }
        for branch, rec in _COMPONENT_TIMES.items()
    }


def _timed_call(branch: str, t_len: int, fn):
    import time as _time

    t0 = _time.perf_counter()
    y, s = fn()
    mx.eval(y, s)
    dt_ms = (_time.perf_counter() - t0) * 1000.0
    rec = _COMPONENT_TIMES.setdefault(
        branch, {"calls": 0, "total_ms": 0.0, "by_t": {}}
    )
    rec["calls"] += 1
    rec["total_ms"] += dt_ms
    rec["by_t"][t_len] = rec["by_t"].get(t_len, 0.0) + dt_ms
    if rec["calls"] % 64 == 0:
        try:
            print(
                f"[gdn-prefill/component-timing] {branch}: {rec['calls']} calls, "
                f"{rec['total_ms']:.0f} ms total",
                flush=True,
            )
        except Exception:
            pass
    return y, s


def blocked_prefill_env_enabled() -> bool:
    return str(os.environ.get("MTPLX_GDN_BLOCKED_PREFILL", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _min_route_t() -> int:
    try:
        return max(2, int(os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_MIN_T", "16")))
    except ValueError:
        return 16


def install_gdn_blocked_prefill_patch() -> dict:
    """Route prefill-scale scalar-gating GDN calls through the blocked kernel.

    Wraps ``mlx_lm.models.gated_delta.gated_delta_update``. Decode-scale calls
    (T below MTPLX_GDN_BLOCKED_PREFILL_MIN_T, default 16), masked calls,
    vectorized gating, and any shape outside the structural gate stay on the
    stock path byte-for-byte. Patches both the defining module and the
    ``qwen3_next`` import site (the name is bound at import time there).
    Idempotent.
    """
    if _PATCH_STATE["installed"]:
        return {"installed": True, "already": True}
    try:
        from mlx_lm.models import gated_delta as _gd
    except Exception as exc:  # pragma: no cover - environment without mlx_lm
        return {"installed": False, "error": f"mlx_lm import failed: {exc}"}

    original = _gd.gated_delta_update
    compute_g = _gd.compute_g
    min_t = _min_route_t()

    debug = str(os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_DEBUG", "")).strip() in {
        "1", "true", "on",
    }
    # Component-timing diagnostic (MTPLX_GDN_PREFILL_COMPONENT_TIMING=1):
    # force-evals the GDN output around each large-T call and accumulates
    # per-branch GPU-inclusive wall time. Serve-to-serve TTFT wobble (+/-7%
    # measured 2026-07-31) cannot resolve the ~2% GDN share, so promotion
    # decisions use this component clock instead. The sync eval perturbs
    # pipeline overlap — DIAGNOSTIC ONLY, never a production default; both
    # the blocked and stock branches are timed identically so the
    # comparison is fair.
    component_timing = str(
        os.environ.get("MTPLX_GDN_PREFILL_COMPONENT_TIMING", "")
    ).strip() in {"1", "true", "on"}
    # Force the stock branch while keeping the wrapper (and its component
    # clock) installed: the fair "stock arm" for component A/Bs.
    force_stock = str(
        os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK", "")
    ).strip() in {"1", "true", "on"}
    debug_state = {"routed": 0, "stock": 0, "logged": 0}
    try:
        debug_max = int(os.environ.get("MTPLX_GDN_BLOCKED_PREFILL_DEBUG_MAX", "6"))
    except ValueError:
        debug_max = 6

    def patched(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    ):
        if (
            use_kernel
            and not force_stock
            and mask is None
            and q.ndim == 4
            and q.shape[1] >= min_t
            and mx.default_device() == mx.gpu
        ):
            beta = mx.sigmoid(b)
            g = compute_g(A_log, a, dt_bias)
            if blocked_prefill_eligible(q, v, g, mask, state):
                if state is None:
                    B = q.shape[0]
                    Hv, Dv = v.shape[2:]
                    Dk = q.shape[3]
                    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
                if debug:
                    debug_state["routed"] += 1
                    if debug_state["logged"] < debug_max:
                        debug_state["logged"] += 1
                        try:
                            print(
                                "[gdn-blocked-prefill/debug] routed call "
                                f"T={q.shape[1]} qdtype={q.dtype} "
                                f"q_flags={getattr(q, 'flags', None)} "
                                f"state_dtype={state.dtype} "
                                f"routed={debug_state['routed']}",
                                flush=True,
                            )
                        except Exception:
                            pass
                if component_timing:
                    return _timed_call(
                        "blocked",
                        int(q.shape[1]),
                        lambda: gated_delta_blocked_prefill(q, k, v, g, beta, state),
                    )
                return gated_delta_blocked_prefill(q, k, v, g, beta, state)
        if debug:
            debug_state["stock"] += 1
            if q.ndim == 4 and q.shape[1] >= 2 and debug_state["logged"] < 2 * debug_max:
                debug_state["logged"] += 1
                try:
                    print(
                        "[gdn-blocked-prefill/debug] STOCK large-T call "
                        f"T={q.shape[1]} mask={mask is not None} "
                        f"g_ndim={'4d-vec' if getattr(a, 'ndim', 0) == 4 else 'scalar'}",
                        flush=True,
                    )
                except Exception:
                    pass
        if component_timing and q.ndim == 4 and q.shape[1] >= min_t:
            return _timed_call(
                "stock",
                int(q.shape[1]),
                lambda: original(
                    q, k, v, a, b, A_log, dt_bias,
                    state=state, mask=mask, use_kernel=use_kernel,
                ),
            )
        return original(
            q, k, v, a, b, A_log, dt_bias, state=state, mask=mask, use_kernel=use_kernel
        )

    _gd.gated_delta_update = patched
    patched_sites = ["mlx_lm.models.gated_delta"]
    # Model modules bind the name at import (`from .gated_delta import
    # gated_delta_update`), so patching the defining module alone misses
    # them. Qwen3.6 loads as model_type qwen3_5 -> mlx_lm.models.qwen3_5
    # holds its own binding (the first integration missed exactly this and
    # every serve-level A/B ran stock on both arms). Sweep every loaded
    # module that holds the original, and import the known model modules
    # first so they exist to be swept even before the model loads.
    import sys as _sys

    for _name in ("mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_next"):
        try:
            __import__(_name)
        except Exception:
            pass
    for _name, _mod in list(_sys.modules.items()):
        if _mod is None or not _name.startswith("mlx_lm"):
            continue
        if getattr(_mod, "gated_delta_update", None) is original:
            try:
                _mod.gated_delta_update = patched
                patched_sites.append(_name)
            except Exception:
                pass
    _PATCH_STATE["installed"] = True
    _PATCH_STATE["original"] = original
    report = {
        "installed": True,
        "already": False,
        "sites": patched_sites,
        "min_t": min_t,
    }
    # logger.info at runtime setup is not captured in the serve console;
    # print the banner so a daemon log proves the route is active.
    try:
        print(f"[mtplx] gdn-blocked-prefill {report}", flush=True)
    except Exception:
        pass
    return report


def uninstall_gdn_blocked_prefill_patch() -> None:
    if not _PATCH_STATE["installed"]:
        return
    original = _PATCH_STATE["original"]
    import sys as _sys

    for _name, _mod in list(_sys.modules.items()):
        if _mod is None or not _name.startswith("mlx_lm"):
            continue
        current = getattr(_mod, "gated_delta_update", None)
        if current is not None and current is not original:
            try:
                _mod.gated_delta_update = original
            except Exception:
                pass
    _PATCH_STATE["installed"] = False
    _PATCH_STATE["original"] = None
