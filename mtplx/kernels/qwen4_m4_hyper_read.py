"""Verify-width (rows 2..8) fused hyper-connection READ for Qwen3.8 Flash-Next.

WHY THIS EXISTS (and why ``hyper_connection.fused_hyper_read`` does not do it)
-----------------------------------------------------------------------------
``GatedResidual.__call__`` is read ~97 times per fixed-M4 verify forward (2 per
layer + the trunk mixer). Eagerly it is 11 dispatches -- grouped rms_norm, the
gamma multiply, the ``[320, 10240]`` down GEMV, silu, the ``[10240, 320]`` up
GEMV, sigmoid, the gated multiply, the hc-mean sum, the mean scale, the
``[4, 10240]`` inject GEMV and its sigmoid -- so 1,067 dispatches/cycle, the
largest zero-byte dispatch family in the compiled verify graph. It is also the
largest non-MoE *byte* consumer: 13.19 MB of bf16 mix weights per read,
1.28 GB/cycle, and the down GEMV measured ~385 GB/s.

``fused_hyper_read`` (kernels/hyper_connection.py) collapses all 11 into ONE
dispatch with ``grid=(1024, S, 1)``: one threadgroup per row. That shape is a
latency trap at S=4 -- 4 threadgroups on a 40-core GPU, each thread walking
~1.6 KB of weights serially, and every weight element re-read once PER ROW
(4 x 13.19 MB = 52.7 MB at S=4). Measured on the M4 verifier 2026-09-01:
13.2 tok/s with ``MTPLX_FUSED_HC=1`` vs 67.8 control, every GPU phase ~5x
slower because the underfilled kernel backs the queue up.

This module is the same arithmetic laid out as a *bandwidth-bound GEMV*:
threadgroups tile the OUTPUT COLUMNS, the row dimension R lives in registers,
and each weight element is read EXACTLY ONCE per call regardless of R.

SHAPE OF THE THREE DISPATCHES
-----------------------------
K0 ``norm``   x[R,10240], gamma[10240]            -> normed[R,10240] (bf16)
              one threadgroup per (row, hc group); 4R threadgroups of
              ``norm_threads``. Materialising ``normed`` once costs 80 KB of
              writes and removes the per-threadgroup re-derivation that the
              v1/v3 kernels pay (v3 recomputes ``x*wn*rms`` inside every
              simdgroup's dot loop -- ~13 MB of L2 traffic at R=1).

K1 ``down``   normed, wd[320,10240], wi[4,10240]  -> mixv[R,320], inject[R,4]
              The inject rows are FOLDED IN as virtual output rows 320..323
              (v3's trick), so one kernel streams a [324, 10240] matrix.
              ``out_per_tg`` output rows per threadgroup, K split across the
              threadgroup's threads: thread t owns k = t, t+NT, t+2NT, ...
              Consecutive threads read consecutive weight addresses (fully
              coalesced), each thread keeps ``normed[r][k]`` for all R rows in
              registers and reuses it for all ``out_per_tg`` weight rows, so
              the activation:weight load ratio is R/out_per_tg, not R:1.
              wd + wi are read once: 6.63 MB per call at any R.

K2 ``up``     normed, mixv, wu[10240,320]         -> mixed[R,2560]
              One simdgroup per (hc group, output d); a threadgroup is
              HC=4 simdgroups covering the four hc partners of the same d, so
              the hc-mean closes inside threadgroup memory with no second
              pass. ``d_per_block`` d's per threadgroup amortises the
              per-lane register copy of ``mixv`` (KUP/32 = 10 j's x R).
              wu is read once: 6.55 MB per call at any R.

Total: 3 dispatches (from 11) and 13.19 MB of weight reads (from R x 13.19 MB
under the (1024, S, 1) kernel) -- the DRAM floor for this read, which at the
M5 Max's 614 GB/s ceiling is ~21.5 us/call, ~2.1 ms/cycle over 97 calls.
There is no arrangement of this arithmetic that goes faster, because every
GatedResidual owns private weights and nothing is reused across the 97 calls.

NUMERICS: WHAT "MATCHES THE EAGER CHAIN" MEANS HERE
---------------------------------------------------
Target is the EAGER bf16 chain (what the compiled verifier runs today), not
the fused v1/v3 kernels. Every op boundary the eager chain rounds at is
rounded here too, in the same order:

    normed  = (T)( (float)(T)(x * rsqrt(ss/D + eps)) * (float)gamma )
    lin     = (T)(dot(normed, wd_row))          # nn.Linear output cast
    t0      = (T)(lin * 0.25f)                  # / hc_count (exact in bf16)
    mixv    = (T)( (float)(T)sigmoid(t0) * t0 ) # nn.silu = x * sigmoid(x)
    inject  = (T)( 2.0f * (float)(T)sigmoid((T)((T)dot(normed, wi_row)*0.25f)) )
    up      = (T)(dot(mixv, wu_row))
    gate    = (T)sigmoid(up)
    prod    = (T)( (float)gate * (float)normed[p] )
    s       = prod[0]; for g in 1..3: s = (T)(s + prod[g])   # mx.sum, bf16 acc
    mixed   = (T)((float)s * 0.25f)                          # mx.mean scale

The op-boundary contract above was checked against MLX 0.32.2 on the CPU
stream (all four assertions live in tests/test_qwen4_hc_m4.py):

  * ``bf16 / 4`` and ``2.0 * bf16`` both stay bf16 (weak scalar promotion),
    so the ``/ hc_count`` and the inject's ``2.0 *`` round in bf16.
  * ``mx.mean(a, axis=-2)`` on a length-4 bf16 axis is bit-identical to
    ``(mx.sum(a, -2) * 0.25)``, and that ``mx.sum`` is bit-identical to a
    SEQUENTIAL bf16 accumulation -- an fp32 accumulation of the same four
    terms differs on 33% of random inputs. Hence the bf16 loop above.
  * ``nn.silu(a)`` is exactly ``a * mx.sigmoid(a)``.

EXPECTED DIFFERENCE CLASS -- rounding only, three named sources:

 1. GEMV REDUCTION ORDER. MLX's ``gemv_wide_bfloat16_nv4_kl32`` tiles
    K=10,240 its own way; this kernel accumulates fp32 per thread over a
    strided k subset, then a simd butterfly (``simd_sum``), then a sequential
    fp32 walk over the NT/32 simdgroup partials. Both are fp32 accumulations
    of the same 10,240 products (MLX's Metal GEMV uses an fp32 AccT for half
    types), so they differ only by fp32 reassociation -- a few ulp in fp32,
    which then usually rounds to the SAME bf16. The residual is a
    1-ulp-of-bf16 flip on outputs sitting near a rounding boundary. Same
    argument for the rms sum of squares.

 2. SIGMOID FLAVOUR -- the dominant term. MLX's own bf16 ``Sigmoid`` is not
    reproducible from any fp32 model: on the CPU backend it differs from
    ``bf16(stable-fp32 sigmoid)`` on ~14% of random bf16 inputs, from
    ``bf16(naive-fp32)`` on ~14%, and from an fp64 evaluation on the same
    ~14% -- i.e. it rounds somewhere inside its own decomposition. This
    kernel uses the stable fp32 form (``y = 1/(1+exp(-|x|))`` mirrored for
    x<0) and rounds once. Expect O(10%) of the ``mixv``/``gate``/``inject``
    elements to land one bf16 ulp away, which is a relative error of at most
    2^-8 on a value in (0, 1) that then multiplies a normed activation.

 3. rsqrt FLAVOUR. ``metal::precise::rsqrt`` here against whatever
    ``mx.fast.rms_norm`` picked -- a bf16-ulp class on ``normed``.

So: NOT bit-identical, and not claimed to be. Adoption gates on acceptance
parity on the real verifier (the same bar `_fused_read_applies` was always
going to need), plus the microbench's max-abs-diff / differing-element counts
against the eager module in the hyper-read microbenchmark.

No silent fallback: ``mtplx.models.qwen4_exp`` raises when MTPLX_QWEN4_HC_M4
is armed and the module geometry or weight dtypes do not match.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import mlx.core as mx

HC = 4
D_HIDDEN = 2560
HCD = HC * D_HIDDEN  # 10240
R_LOWRANK = 320
N_FOLDED = R_LOWRANK + HC  # 324 virtual stage-1 output rows

#: Rows this kernel family is wired for. rows==1 keeps the v3 draft path.
MIN_ROWS = 2
MAX_ROWS = 8

DEFAULT_NORM_THREADS = 256
DEFAULT_OUT_PER_TG = 4
DEFAULT_D_PER_BLOCK = 8

#: Dispatches per read, for the microbench's dispatch-count column.
DISPATCHES_PER_READ = 3
EAGER_DISPATCHES_PER_READ = 11

_HEADER = """
#include <metal_stdlib>
using namespace metal;

// MLX's Sigmoid op, in fp32: y = 1/(1+exp(-|x|)) mirrored for x < 0.
inline float mlx_sigmoid_f(float x) {
    const float y = 1.0f / (1.0f + metal::exp(-metal::abs(x)));
    return (x < 0.0f) ? (1.0f - y) : y;
}
"""

# --------------------------------------------------------------------------
# K0 -- grouped RMS norm + gamma, materialised once.
#
# One threadgroup per (row, hc group): grid.x = norm_threads * (R * HC).
# ``EPS_LITERAL`` is substituted per module eps so the kernel needs no extra
# input array (and therefore no extra array to eval) on the hot path.
# --------------------------------------------------------------------------
_SRC_NORM = """
    constexpr int HC = 4;
    constexpr int D = 2560;
    constexpr int HCD = HC * D;
    constexpr int NT = NTHREADS;
    constexpr int NSG = NT / 32;

    const uint blk = threadgroup_position_in_grid.x;
    const uint r = blk / (uint)HC;
    const uint g = blk % (uint)HC;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;

    device const T* xg = x + (size_t)r * HCD + (size_t)g * D;
    device const T* wg = gamma + (size_t)g * D;
    device T* og = normed + (size_t)r * HCD + (size_t)g * D;

    threadgroup float part[NSG];

    float ss = 0.0f;
    for (int i = (int)tid; i < D; i += NT) {
        const float v = (float)xg[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread walks the NSG partials in the same order, so every thread
    // lands on the identical fp32 scale -- one barrier, and no broadcast slot
    // to leave uninitialized.
    float tot = 0.0f;
    for (int s = 0; s < NSG; ++s) tot += part[s];
    const float sc = metal::precise::rsqrt(tot / (float)D + EPS_LITERAL);
    for (int i = (int)tid; i < D; i += NT) {
        // mx.fast.rms_norm(grouped, None, eps) rounds to T, then the module
        // multiplies by the full-width weight in T.
        const float nv = (float)((T)((float)xg[i] * sc));
        og[i] = (T)(nv * (float)wg[i]);
    }
"""

# --------------------------------------------------------------------------
# K1 -- folded [NTOT, 10240] down/inject GEMV over R rows.
#
# grid.x = down_threads * ceil(NTOT / OUT_PER_TG). Threadgroup ``blk`` owns
# output rows [blk*OUT_PER_TG, +OUT_PER_TG); thread ``t`` owns the k stripe
# {t, t+NT, ...}. Each weight element is loaded once by exactly one thread of
# exactly one threadgroup.
# --------------------------------------------------------------------------
_SRC_DOWN = """
    constexpr int HC = 4;
    constexpr int D = 2560;
    constexpr int HCD = HC * D;
    constexpr int NDOWN = 320;
    constexpr int NTOT = HAS_INJECT ? (NDOWN + HC) : NDOWN;
    constexpr int NT = NTHREADS;
    constexpr int NSG = NT / 32;
    constexpr int OPT = OUT_PER_TG;
    constexpr int RR = ROWS;

    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int o0 = (int)blk * OPT;

    threadgroup float red[NSG * OPT * RR];

    float acc[OPT][RR];
    #pragma clang loop unroll(full)
    for (int o = 0; o < OPT; ++o) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RR; ++r) acc[o][r] = 0.0f;
    }

    for (int k = (int)tid; k < HCD; k += NT) {
        float nv[RR];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RR; ++r) nv[r] = (float)normed[(size_t)r * HCD + k];
        #pragma clang loop unroll(full)
        for (int o = 0; o < OPT; ++o) {
            // Loop-invariant in k: LICM hoists these OPT row pointers into
            // registers. Out-of-range rows (only the tail block, and only
            // when OPT does not divide NTOT) alias row 0 so the inner loop
            // stays branch-free; their partials are dropped in the epilogue.
            const int orow = o0 + o;
            const int safe = (orow < NTOT) ? orow : 0;
            const device T* wrow = (safe < NDOWN)
                ? (wd + (size_t)safe * HCD)
                : (wi + (size_t)(safe - NDOWN) * HCD);
            const float w = (float)wrow[k];
            #pragma clang loop unroll(full)
            for (int r = 0; r < RR; ++r) acc[o][r] += nv[r] * w;
        }
    }

    #pragma clang loop unroll(full)
    for (int o = 0; o < OPT; ++o) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RR; ++r) {
            const float v = simd_sum(acc[o][r]);
            if (lane == 0) red[(size_t)sg * (OPT * RR) + o * RR + r] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr int NPAIR = OPT * RR;
    for (int idx = (int)tid; idx < NPAIR; idx += NT) {
        const int o = idx / RR;
        const int r = idx % RR;
        const int orow = o0 + o;
        if (orow >= NTOT) continue;
        float tot = 0.0f;
        for (int s = 0; s < NSG; ++s) tot += red[(size_t)s * NPAIR + idx];
        const float lin = (float)((T)tot);            // nn.Linear output
        const float t0 = (float)((T)(lin * 0.25f));   // / hc_count
        const float s0 = (float)((T)mlx_sigmoid_f(t0));
        if (orow < NDOWN) {
            mixv[(size_t)r * NDOWN + orow] = (T)(s0 * t0);   // nn.silu
        } else {
            inject[(size_t)r * HC + (orow - NDOWN)] = (T)(2.0f * s0);
        }
    }
"""

# --------------------------------------------------------------------------
# K2 -- up GEMV + sigmoid gate + hc-mean.
#
# Threadgroup = HC simdgroups (128 threads); simdgroup ``g`` owns hc group g,
# the threadgroup owns d in [blk*DPB, +DPB). Every wu row is read once.
# --------------------------------------------------------------------------
_SRC_UP = """
    constexpr int HC = 4;
    constexpr int D = 2560;
    constexpr int HCD = HC * D;
    constexpr int KUP = 320;
    constexpr int JPL = KUP / 32;      // 10 j's per lane
    constexpr int RR = ROWS;
    constexpr int DPB = D_PER_BLOCK;
    constexpr int NT = HC * 32;

    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;          // == hc group
    const uint lane = tid % 32;
    const int d0 = (int)blk * DPB;

    threadgroup float prodv[HC * DPB * RR];

    // Per-lane register copy of the R mixv rows: reused for all DPB d's.
    float m[RR][JPL];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RR; ++r) {
        #pragma clang loop unroll(full)
        for (int jj = 0; jj < JPL; ++jj) {
            m[r][jj] = (float)mixv[(size_t)r * KUP + (int)lane + jj * 32];
        }
    }

    for (int dd = 0; dd < DPB; ++dd) {
        const int d = d0 + dd;
        if (d >= D) break;
        const int p = (int)sg * D + d;
        const device T* wrow = wu + (size_t)p * KUP;
        float a[RR];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RR; ++r) a[r] = 0.0f;
        #pragma clang loop unroll(full)
        for (int jj = 0; jj < JPL; ++jj) {
            const float w = (float)wrow[(int)lane + jj * 32];
            #pragma clang loop unroll(full)
            for (int r = 0; r < RR; ++r) a[r] += m[r][jj] * w;
        }
        #pragma clang loop unroll(full)
        for (int r = 0; r < RR; ++r) {
            const float tot = simd_sum(a[r]);           // uniform: all lanes
            if ((int)lane == r) {
                const float lin = (float)((T)tot);      // nn.Linear output
                const float gate = (float)((T)mlx_sigmoid_f(lin));
                const float nv = (float)normed[(size_t)r * HCD + p];
                prodv[(size_t)sg * (DPB * RR) + dd * RR + r] =
                    (float)((T)(gate * nv));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = (int)tid; idx < DPB * RR; idx += NT) {
        const int dd = idx / RR;
        const int r = idx % RR;
        const int d = d0 + dd;
        if (d >= D) continue;
        // mx.mean(..., axis=-2) is sum-then-scale, and MLX's col_reduce
        // instantiates T == U, so the 4-term sum accumulates IN bf16 and
        // rounds at every add, in axis order. Verified against the CPU
        // backend (mx.sum over a length-4 bf16 axis is bit-identical to a
        // sequential bf16 accumulation; an fp32 accumulation differs on 33%
        // of random inputs, so this ordering is not cosmetic).
        float s = prodv[idx];                            // g == 0
        for (int g = 1; g < HC; ++g) {
            s = (float)((T)(s + prodv[(size_t)g * (DPB * RR) + idx]));
        }
        mixed[(size_t)r * D + d] = (T)(s * 0.25f);       // exact: 2^-2
    }
"""


def _eps_tag(eps: float) -> str:
    """Stable short tag for an eps value, so a kernel name never collides
    across two different eps (MLX caches compiled libraries by name)."""

    return struct.pack("<f", float(eps)).hex()


@lru_cache(maxsize=8)
def _kernel_norm(eps_bits: int):
    eps = struct.unpack("<f", struct.pack("<I", eps_bits))[0]
    src = _SRC_NORM.replace("EPS_LITERAL", f"{eps!r}f")
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_m4_hc_norm_{_eps_tag(eps)}",
        input_names=["x", "gamma"],
        output_names=["normed"],
        header=_HEADER,
        source=src,
    )


@lru_cache(maxsize=2)
def _kernel_down():
    return mx.fast.metal_kernel(
        name="mtplx_qwen4_m4_hc_down",
        input_names=["normed", "wd", "wi"],
        output_names=["mixv", "inject"],
        header=_HEADER,
        source=_SRC_DOWN,
    )


@lru_cache(maxsize=2)
def _kernel_up():
    return mx.fast.metal_kernel(
        name="mtplx_qwen4_m4_hc_up",
        input_names=["normed", "mixv", "wu"],
        output_names=["mixed"],
        header=_HEADER,
        source=_SRC_UP,
    )


def _eps_bits(eps: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(eps)))[0]


def check_shapes(
    x,
    gamma,
    wd,
    wu,
    wi,
    *,
    rows_min: int = MIN_ROWS,
    rows_max: int = MAX_ROWS,
) -> int:
    """Validate the family contract and return R. Raises -- never returns
    False -- so an armed flag on a mismatched pack fails at the call site
    instead of silently reverting to the eager chain.

    ``x`` may carry any leading dims (the module sees ``[B, S, 10240]``); R is
    their product. Taking the unreshaped array keeps this off the graph: no
    reshape node is created just to validate.
    """

    if x.ndim < 1 or x.shape[-1] != HCD:
        raise ValueError(
            f"hyper input must be [..., {HCD}] for the M4 hyper read; got "
            f"{tuple(x.shape)}"
        )
    rows = 1
    for s in x.shape[:-1]:
        rows *= int(s)
    if not (rows_min <= rows <= rows_max):
        raise ValueError(
            f"M4 hyper read is wired for {rows_min}..{rows_max} rows; got {rows}"
        )
    check_weight_shapes(gamma, wd, wu, wi, dtype=x.dtype)
    return rows


def check_weight_shapes(gamma, wd, wu, wi, *, dtype=None) -> None:
    """Validate the WEIGHT half of the family contract.

    Split out of :func:`check_shapes` so the same list can run at INSTALL
    time, where the weights exist but no activation does: a pack the kernel
    cannot read is a deployment error, and it should stop the server coming up
    rather than fail the first request that happens to reach verify width.
    ``dtype`` is the activation dtype when there is one to compare against;
    ``None`` skips only that check.  One list, two callers.
    """

    checks = (
        ("hc_norm.weight", gamma, (HCD,)),
        ("input_mix_weight_down.weight", wd, (R_LOWRANK, HCD)),
        ("input_mix_weight_up.weight", wu, (HCD, R_LOWRANK)),
    )
    if wi is not None:
        checks = checks + (("block_inject_weight.weight", wi, (HC, HCD)),)
    for name, arr, want in checks:
        if tuple(arr.shape) != want:
            raise ValueError(
                f"M4 hyper read: {name} must be {want}; got {tuple(arr.shape)}"
            )
        if dtype is not None and arr.dtype != dtype:
            raise ValueError(
                f"M4 hyper read: {name} dtype {arr.dtype} != hyper input dtype "
                f"{dtype} (the kernel reads the module's unquantized weights)"
            )


def fused_hc_read_m4(
    x2d,
    gamma,
    wd,
    wu,
    wi=None,
    *,
    eps: float = 1e-6,
    norm_threads: int = DEFAULT_NORM_THREADS,
    down_threads: int = DEFAULT_NORM_THREADS,
    out_per_tg: int = DEFAULT_OUT_PER_TG,
    d_per_block: int = DEFAULT_D_PER_BLOCK,
):
    """Fused GatedResidual read for R rows.

    ``x2d`` [R, 10240], ``gamma`` [10240], ``wd`` [320, 10240],
    ``wu`` [10240, 320], ``wi`` [4, 10240] or None (the no-combine trunk
    mixer). Returns ``(mixed [R, 2560], inject [R, 4] or None)``.
    """

    if x2d.ndim != 2:
        raise ValueError(
            f"fused_hc_read_m4 wants a 2-D [R, {HCD}] view; got {tuple(x2d.shape)}"
        )
    rows = check_shapes(x2d, gamma, wd, wu, wi)
    if norm_threads % 32 or not (32 <= norm_threads <= 1024):
        raise ValueError(f"norm_threads={norm_threads}: want a multiple of 32 in [32,1024]")
    if down_threads % 32 or not (32 <= down_threads <= 1024):
        raise ValueError(f"down_threads={down_threads}: want a multiple of 32 in [32,1024]")
    if out_per_tg < 1 or out_per_tg > 32:
        raise ValueError("out_per_tg must be in 1..32")
    if d_per_block < 1 or d_per_block > 64:
        raise ValueError("d_per_block must be in 1..64")

    dt = x2d.dtype
    has_inject = wi is not None
    n_tot = N_FOLDED if has_inject else R_LOWRANK

    (normed,) = _kernel_norm(_eps_bits(eps))(
        inputs=[x2d, gamma],
        template=[("T", dt), ("NTHREADS", norm_threads)],
        grid=(norm_threads * rows * HC, 1, 1),
        threadgroup=(norm_threads, 1, 1),
        output_shapes=[(rows, HCD)],
        output_dtypes=[dt],
    )

    n_blk = (n_tot + out_per_tg - 1) // out_per_tg
    mixv, inject = _kernel_down()(
        inputs=[normed, wd, wi if has_inject else wd],
        template=[
            ("T", dt),
            ("ROWS", rows),
            ("NTHREADS", down_threads),
            ("OUT_PER_TG", out_per_tg),
            ("HAS_INJECT", 1 if has_inject else 0),
        ],
        grid=(down_threads * n_blk, 1, 1),
        threadgroup=(down_threads, 1, 1),
        output_shapes=[(rows, R_LOWRANK), (rows, HC)],
        output_dtypes=[dt, dt],
    )

    d_blk = (D_HIDDEN + d_per_block - 1) // d_per_block
    (mixed,) = _kernel_up()(
        inputs=[normed, mixv, wu],
        template=[("T", dt), ("ROWS", rows), ("D_PER_BLOCK", d_per_block)],
        grid=(HC * 32 * d_blk, 1, 1),
        threadgroup=(HC * 32, 1, 1),
        output_shapes=[(rows, D_HIDDEN)],
        output_dtypes=[dt],
    )
    return mixed, (inject if has_inject else None)


def weight_bytes_per_read(dtype_size: int = 2, *, has_inject: bool = True) -> int:
    """Weight bytes this read streams, independent of R -- the number the
    GB/s column in the hyper-read microbenchmark divides by."""

    n = R_LOWRANK * HCD + HCD * R_LOWRANK + HCD  # down + up + gamma
    if has_inject:
        n += HC * HCD
    return n * dtype_size
