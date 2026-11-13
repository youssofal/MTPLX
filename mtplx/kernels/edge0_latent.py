"""Edge0 latent MoE kernels: the SwiGLU epilogue and the per-group combine.

Both reproduce, bit for bit, the MLX op chains they replace in
:class:`mtplx.models.edge0_latent_moe.LatentMoeBlock` (same float32 operations
in the same order, ``fp contract(off)`` so no multiply and add fold into one
fused multiply-add). The SwiGLU uses ``metal::exp``, which in a JIT-built
kernel is the fast exp: ``mlx.nn.silu`` is itself an ``mx.compile`` function,
so the stock chain's sigmoid is the JIT one, not the precompiled precise one.

Combine. Replaces the ``top_k``-step ``mx.where`` loop of
:meth:`mtplx.models.edge0_latent_moe.LatentMoeBlock.__call__` (about 30 small
dispatches per layer) with one kernel that computes, for every token ``t``,
group ``g`` and latent column ``l``::

    h[t, j, l]  = float(down[t, j, l]) * s_down[ids[t, j]]
    z[t, g, l]  = sum over j = 0 .. K-1 with groups[t, j] == g of  h[t, j, l] * w[t, j]

with the same float32 operations in the same order as the loop (the product
``h * w`` rounded on its own, the sum a left fold over ``j`` that starts from
the first term), so its output is bit-identical to the loop's. ``contract``
is turned off for the expression so the compiler cannot fold a multiply and
the following add into one fused multiply-add.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_SOURCE = """
    #pragma METAL fp contract(off)
    const uint idx = thread_position_in_grid.x;
    const uint l = idx % LAT;
    const uint tg = idx / LAT;
    const uint g = tg % NG;
    const uint t = tg / NG;
    float acc = 0.0f;
    for (int j = 0; j < TOP_K; ++j) {
        const uint r = t * TOP_K + j;
        float v = 0.0f;
        if (groups[r] == int(g)) {
            const float h = float(down[r * LAT + l]) * scalars[ids[r] * 3 + 2];
            v = h * w[r];
        }
        acc = (j == 0) ? v : acc + v;
    }
    z[idx] = acc;
"""


@lru_cache(maxsize=None)
def _kernel(top_k: int, n_groups: int, latent: int):
    header = (
        "using namespace metal;\n"
        f"constant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr uint NG = {n_groups};\n"
        f"constant constexpr uint LAT = {latent};\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_combine_k{top_k}_g{n_groups}_l{latent}",
        input_names=["down", "ids", "scalars", "w", "groups"],
        output_names=["z"],
        header=header,
        source=_SOURCE,
    )


def latent_combine(down: mx.array, ids: mx.array, scalars: mx.array, w: mx.array, groups: mx.array, n_groups: int) -> mx.array:
    """``down [T, K, L]`` (any float dtype), ``ids``/``groups`` ``[T, K]``, ``scalars [E, 3]`` float32 (``s_down`` is
    column 2), ``w`` float32 ``[T, K]`` -> ``z`` float32 ``[T, G, L]``."""
    t, k, lat = (int(d) for d in down.shape)
    total = t * n_groups * lat
    (z,) = _kernel(k, int(n_groups), lat)(
        inputs=[down, ids.astype(mx.int32), scalars.astype(mx.float32), w.astype(mx.float32), groups.astype(mx.int32)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(t, n_groups, lat)],
        output_dtypes=[mx.float32],
    )
    return z


def latent_combine_reference(down: mx.array, ids: mx.array, scalars: mx.array, w: mx.array, groups: mx.array, n_groups: int) -> mx.array:
    """The loop the kernel replaces, in plain MLX ops."""
    h = down.astype(mx.float32) * scalars[ids][..., 2:3]
    hw = h * w[..., None]
    gid = mx.arange(n_groups, dtype=mx.int32)[:, None]
    zero = mx.array(0.0, dtype=mx.float32)
    z = None
    for j in range(down.shape[1]):
        term = mx.where(groups[:, j, None, None] == gid, hw[:, j, None, :], zero)
        z = term if z is None else z + term
    return z


_SWIGLU_SOURCE = """
    #pragma METAL fp contract(off)
    const uint idx = thread_position_in_grid.x;
    const uint r = idx / FF;
    const uint i = idx % FF;
    const int e = ids[r];
    const float g = float(GATE_AT) * scalars[e * 3];
    const float u = float(UP_AT) * scalars[e * 3 + 1];
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(g)));
    const float sig = (g < 0.0f) ? y : 1.0f - y;
    const float silu = g * sig;
    a[idx] = OutT(silu * u);
"""


@lru_cache(maxsize=None)
def _swiglu_kernel(expert_ff: int, merged: bool):
    header = f"using namespace metal;\nconstant constexpr uint FF = {expert_ff};\n"
    if merged:
        src = _SWIGLU_SOURCE.replace("GATE_AT", "gu[r * 2 * FF + i]").replace("UP_AT", "gu[r * 2 * FF + FF + i]")
        names = ["gu", "ids", "scalars"]
    else:
        src = _SWIGLU_SOURCE.replace("GATE_AT", "gate[idx]").replace("UP_AT", "up[idx]")
        names = ["gate", "up", "ids", "scalars"]
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_swiglu_f{expert_ff}_{'m' if merged else 's'}",
        input_names=names,
        output_names=["a"],
        header=header,
        source=src,
    )


def latent_swiglu(gate_up, ids: mx.array, scalars: mx.array, expert_ff: int, out_dtype: mx.Dtype) -> mx.array:
    """``silu(s_gate * gate) * (s_up * up)`` per route, cast to ``out_dtype``.

    ``gate_up`` is either one array ``[..., 2 * I]`` (gate rows then up rows,
    the merged projection) or a pair ``(gate [..., I], up [..., I])``; ``ids``
    holds the expert of each route (``prod(shape[:-1])`` entries) and
    ``scalars [E, 3]`` the float32 per-expert scalars. Output ``[..., I]``.
    """
    merged = not isinstance(gate_up, tuple)
    lead = (gate_up if merged else gate_up[0]).shape[:-1]
    rows = 1
    for d in lead:
        rows *= int(d)
    inputs = [gate_up] if merged else list(gate_up)
    (a,) = _swiglu_kernel(int(expert_ff), merged)(
        inputs=inputs + [ids.reshape(-1).astype(mx.int32), scalars],
        template=[("OutT", out_dtype)],
        grid=(rows * expert_ff, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*lead, expert_ff)],
        output_dtypes=[out_dtype],
    )
    return a


def latent_swiglu_reference(gate_up, ids: mx.array, scalars: mx.array, expert_ff: int, out_dtype: mx.Dtype) -> mx.array:
    """The MLX op chain the kernel replaces."""
    import mlx.nn as nn

    if isinstance(gate_up, tuple):
        gate, up = gate_up
    else:
        gate, up = gate_up[..., :expert_ff], gate_up[..., expert_ff:]
    sc = scalars[ids.reshape(gate.shape[:-1])]
    g = gate.astype(mx.float32) * sc[..., 0:1]
    u = up.astype(mx.float32) * sc[..., 1:2]
    return (nn.silu(g) * u).astype(out_dtype)


__all__ = ["latent_combine", "latent_combine_reference", "latent_swiglu", "latent_swiglu_reference"]


# -- D and U on the groups a token routes to ---------------------------------------
#
# Single-row affine 4-bit GEMVs that skip the work of groups no route of the
# token selected. The dot product is MLX's ``qmv_fast_impl`` (quantized.h,
# MLX 0.32.2) line for line with float accumulation: 2 simdgroups x 4 rows
# per threadgroup, 16 values per lane, K blocks of 512, x pre-divided by
# 16/256/4096 so nibble products need no shifts, one ``simd_sum`` per row. So a
# row it computes is bit-identical to ``mx.quantized_matmul`` on float32
# inputs (x of another dtype is converted exactly on load; scales/biases may
# be float16 when that is lossless, which changes no value).
#
# D (rows): output rows of group g are computed only if g is used; the rest
# are written as zeros (nothing reads them). U (columns): a 512-wide K block
# whose columns all fall in unused groups is skipped; those inputs are exact
# zeros (the combine writes 0.0 there), so skipping them leaves every sum
# unchanged (up to the sign of an exactly-zero sum).

_QMV_HEADER = """
using namespace metal;
constant constexpr int SIMD = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int VPT = PACK_FACTOR * PACKS_PER_THREAD;   // 16 values per lane
constant constexpr int BLOCK = VPT * SIMD;                     // 512
constant constexpr int RPS = 4;                                // rows per simdgroup
constant constexpr int NSG = 2;
constant constexpr int SCALE_STEP = GS / VPT;

template <typename XT>
inline float load_x(const device XT* x, thread float* xt) {
  float sum = 0;
  for (int i = 0; i < VPT; i += 4) {
    const float a = float(x[i]), b = float(x[i + 1]), c = float(x[i + 2]), d = float(x[i + 3]);
    sum += a + b + c + d;
    xt[i] = a;
    xt[i + 1] = b / 16.0f;
    xt[i + 2] = c / 256.0f;
    xt[i + 3] = d / 4096.0f;
  }
  return sum;
}

inline float qdot4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (VPT / 4); i++) {
    accum +=
        (xt[4 * i] * (ws[i] & 0x000f) +
         xt[4 * i + 1] * (ws[i] & 0x00f0) +
         xt[4 * i + 2] * (ws[i] & 0x0f00) +
         xt[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <typename P>
inline bool group_used(P groups, uint m, int g) {  // groups may sit in device or constant memory
  for (int j = 0; j < TOP_K; ++j) {
    if (groups[m * TOP_K + j] == g) return true;
  }
  return false;
}
"""

_QMV_BODY = """
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint m = threadgroup_position_in_grid.z;
  const int out_row = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
  const int in_w = K * BYTES_PER_PACK / PACK_FACTOR;
  const int in_g = K / GS;
  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device ST* sc = scales + out_row * in_g + simd_lid / SCALE_STEP;
  const device ST* bs = biases + out_row * in_g + simd_lid / SCALE_STEP;
  const device XT* xp = x + m * K + simd_lid * VPT;
  device OutT* yp = y + m * N + out_row;
  float xt[VPT];
  float result[RPS] = {0};
ROW_SKIP
  for (int k = 0; k < K; k += BLOCK) {
COL_SKIP
    const float sum = load_x(xp, xt);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_w;
      const float s = float(sc[row * in_g]);
      const float b = float(bs[row * in_g]);
      result[row] += qdot4(wl, xt, s, b, sum);
    }
    ws += BLOCK * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK / GS;
    bs += BLOCK / GS;
    xp += BLOCK;
  }
  for (int row = 0; row < RPS; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      yp[row] = OutT(result[row]);
    }
  }
"""

_ROW_SKIP = """
  if (!group_used(groups, m, out_row / LAT)) {
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) yp[row] = OutT(0);
    }
    return;
  }
"""

_COL_SKIP = """
    if (!group_used(groups, m, k / LAT) && !group_used(groups, m, (k + BLOCK - 1) / LAT)) {
      ws += BLOCK * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK / GS;
      bs += BLOCK / GS;
      xp += BLOCK;
      continue;
    }
"""


@lru_cache(maxsize=None)
def _masked_qmv_kernel(kind: str, k_in: int, n_out: int, latent: int, top_k: int, group_size: int):
    header = (
        f"constant constexpr int K = {k_in};\nconstant constexpr int N = {n_out};\n"
        f"constant constexpr int LAT = {latent};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int GS = {group_size};\n" + _QMV_HEADER
    )
    body = _QMV_BODY.replace("ROW_SKIP", _ROW_SKIP if kind == "rows" else "").replace(
        "COL_SKIP", _COL_SKIP if kind == "cols" else ""
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_qmv_{kind}_k{k_in}_n{n_out}_l{latent}_t{top_k}_g{group_size}",
        input_names=["x", "w", "scales", "biases", "groups"],
        output_names=["y"],
        header=header,
        source=body,
    )


def masked_qmv_eligible(x: mx.array, w: mx.array, scales: mx.array, bits: int, group_size: int, latent: int, by: str, *, max_rows: int = 1) -> bool:
    """Shapes the masked GEMVs serve: up to ``max_rows`` rows (one for :func:`masked_qmv`), 4-bit,
    K a multiple of 512, groups aligned to the 8-row tiles."""
    if bits != 4 or group_size not in (32, 64, 128) or x.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return False
    if scales.dtype not in (mx.float32, mx.float16):
        return False
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    k_in = int(x.shape[-1])
    n_out = int(w.shape[0])
    if not 1 <= rows <= max_rows or k_in % 512 or n_out % 8 or int(w.shape[1]) * 8 != k_in:
        return False
    if max_rows > 1 and (n_out if by == "rows" else k_in) > 32 * latent:  # masked_qmv_rows keeps a 32-bit group mask
        return False
    return (n_out % latent == 0 and latent % 8 == 0) if by == "rows" else k_in % latent == 0


def masked_qmv(x, w, scales, biases, groups, *, by: str, latent: int, group_size: int, out_dtype: mx.Dtype) -> mx.array:
    """``x [1, K] @ dequant(w [N, K]).T`` computing only what the routes' groups need.

    ``by="rows"`` (D): output row ``n`` belongs to group ``n // latent``.
    ``by="cols"`` (U): input column ``k`` belongs to group ``k // latent``,
    and ``x`` must be exactly zero in the columns of unused groups.
    ``groups [1, top_k]`` int32 are the groups of the token's routes.
    """
    k_in = int(x.shape[-1])
    n_out = int(w.shape[0])
    top_k = int(groups.shape[-1])
    kernel = _masked_qmv_kernel(by, k_in, n_out, int(latent), top_k, int(group_size))
    (y,) = kernel(
        inputs=[x.reshape(1, k_in), w, scales, biases, groups.reshape(1, top_k).astype(mx.int32)],
        template=[("XT", x.dtype), ("ST", scales.dtype), ("OutT", out_dtype)],
        grid=(64, n_out // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, n_out)],
        output_dtypes=[out_dtype],
    )
    return y


__all__ += ["masked_qmv", "masked_qmv_eligible"]


# -- the same GEMVs for a few rows (speculative verify) ------------------------------
#
# A verify forward runs 2-4 tokens through each layer. MLX's quantized matmul
# costs as much at 2 rows as at 4 there (D: ~3.0 ms over the 120B's 48 layers,
# against ~1.0 ms for the one-row masked kernel) and rounds differently from its
# one-row kernel. This kernel reads each weight tile once for all ``M``
# rows and computes every row exactly as the one-row kernel above does: the same
# per-row accumulator, the same K-block order, the same skip rule per row (a
# D row block of a group the row does not use is written as zeros; a U column
# block of groups the row does not use is skipped for that row). So row ``m`` of
# its output is bit-identical to ``masked_qmv`` on row ``m`` alone. A weight
# tile is skipped only when no row needs it.

_QMV_ROWS_BODY = """
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int out_row = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
  const int in_w = K * BYTES_PER_PACK / PACK_FACTOR;
  const int in_g = K / GS;
  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_w + simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
  const device ST* sc = scales + out_row * in_g + simd_lid / SCALE_STEP;
  const device ST* bs = biases + out_row * in_g + simd_lid / SCALE_STEP;
  float xt[VPT];
  float result[M][RPS];
  uint gmask[M];
  bool use[M];
  bool any = false;
  for (int m = 0; m < M; m++) {
    for (int row = 0; row < RPS; row++) result[m][row] = 0.0f;
    uint g = 0;
    for (int j = 0; j < TOP_K; ++j) g |= 1u << uint(groups[m * TOP_K + j]);
    gmask[m] = g;
    use[m] = ROW_USED;
    any = any || use[m];
  }
  if (!any) {
    if (simd_lid == 0) {
      for (int m = 0; m < M; m++)
        for (int row = 0; row < RPS; row++) y[m * N + out_row + row] = OutT(0);
    }
    return;
  }
  for (int k = 0; k < K; k += BLOCK) {
    for (int m = 0; m < M; m++) {
      if (!(use[m] && COL_NEEDED)) continue;
      const float sum = load_x(x + m * K + k + simd_lid * VPT, xt);
      for (int row = 0; row < RPS; row++) {
        const float s = float(sc[row * in_g]);
        const float b = float(bs[row * in_g]);
        result[m][row] += qdot4(ws + row * in_w, xt, s, b, sum);
      }
    }
    ws += BLOCK * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK / GS;
    bs += BLOCK / GS;
  }
  for (int m = 0; m < M; m++) {
    for (int row = 0; row < RPS; row++) {
      const float r = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * N + out_row + row] = use[m] ? OutT(r) : OutT(0);
      }
    }
  }
"""

_ROWS_COL_NEEDED = "((((gmask[m] >> uint(k / LAT)) | (gmask[m] >> uint((k + BLOCK - 1) / LAT))) & 1u) != 0u)"

#: Row counts the few-rows kernels are built for (a verify forward is 2..depth+1 rows).
FEW_ROWS_MAX = 8


@lru_cache(maxsize=None)
def _masked_qmv_rows_kernel(kind: str, rows: int, k_in: int, n_out: int, latent: int, top_k: int, group_size: int):
    header = (
        f"constant constexpr int K = {k_in};\nconstant constexpr int N = {n_out};\n"
        f"constant constexpr int LAT = {latent};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int GS = {group_size};\nconstant constexpr int M = {rows};\n"
        + _QMV_HEADER
    )
    row_used = "(((gmask[m] >> uint(out_row / LAT)) & 1u) != 0u)" if kind == "rows" else "true"
    col_needed = _ROWS_COL_NEEDED if kind == "cols" else "true"
    body = _QMV_ROWS_BODY.replace("ROW_USED", row_used).replace("COL_NEEDED", col_needed)
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_qmv_{kind}_m{rows}_k{k_in}_n{n_out}_l{latent}_t{top_k}_g{group_size}",
        input_names=["x", "w", "scales", "biases", "groups"],
        output_names=["y"],
        header=header,
        source=body,
    )


def masked_qmv_rows(x, w, scales, biases, groups, *, by: str, latent: int, group_size: int, out_dtype: mx.Dtype) -> mx.array:
    """``masked_qmv`` for ``x [M, K]``, ``groups [M, top_k]``, ``1 <= M <= FEW_ROWS_MAX``: row ``m`` is
    bit-identical to ``masked_qmv(x[m:m+1], ..., groups[m:m+1])``; each weight tile is read once."""
    k_in = int(x.shape[-1])
    rows = int(x.size // k_in)
    n_out = int(w.shape[0])
    top_k = int(groups.shape[-1])
    kernel = _masked_qmv_rows_kernel(by, rows, k_in, n_out, int(latent), top_k, int(group_size))
    (y,) = kernel(
        inputs=[x.reshape(rows, k_in), w, scales, biases, groups.reshape(rows, top_k).astype(mx.int32)],
        template=[("XT", x.dtype), ("ST", scales.dtype), ("OutT", out_dtype)],
        grid=(64, n_out // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, n_out)],
        output_dtypes=[out_dtype],
    )
    return y


# -- router logits for a few rows ------------------------------------------------------
#
# ``x [M, H] @ W[E, H].T`` in float32 with every weight row read once for all M
# rows. Per row this is a float32 dot product accumulated in float32 (lane
# partial sums over a strided slice of H, then a simdgroup sum): accurate to
# float32 rounding, like the one-row GEMV, but not in its summation order, so
# logits agree with the one-row call to ~1e-7 relative, not bit for bit.
# (MLX's float32 GEMM at 2+ rows may run with TF32 operands on M5, ~1e-3.)

_ROUTER_SOURCE = """
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int e0 = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
  float acc[M][RPS];
  for (int m = 0; m < M; m++)
    for (int r = 0; r < RPS; r++) acc[m][r] = 0.0f;
  for (int h = int(simd_lid) * 4; h < H; h += 32 * 4) {
    float4 xv[M];
    for (int m = 0; m < M; m++) {
      xv[m] = float4(float(x[m * H + h]), float(x[m * H + h + 1]), float(x[m * H + h + 2]), float(x[m * H + h + 3]));
    }
    for (int r = 0; r < RPS; r++) {
      const device WT* wr = wt + (e0 + r) * H + h;
      const float4 wv = float4(float(wr[0]), float(wr[1]), float(wr[2]), float(wr[3]));
      for (int m = 0; m < M; m++) acc[m][r] += dot(xv[m], wv);
    }
  }
  for (int m = 0; m < M; m++) {
    for (int r = 0; r < RPS; r++) {
      const float s = simd_sum(acc[m][r]);
      if (simd_lid == 0) out[m * E + e0 + r] = s;
    }
  }
"""


@lru_cache(maxsize=None)
def _router_rows_kernel(rows: int, hidden: int, n_experts: int):
    header = (
        "using namespace metal;\n"
        f"constant constexpr int M = {rows};\nconstant constexpr int H = {hidden};\n"
        f"constant constexpr int E = {n_experts};\nconstant constexpr int RPS = 4;\nconstant constexpr int NSG = 2;\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_router_rows_m{rows}_h{hidden}_e{n_experts}",
        input_names=["x", "wt"],
        output_names=["out"],
        header=header,
        source=_ROUTER_SOURCE,
    )


def router_rows_eligible(x: mx.array, weight: mx.array) -> bool:
    rows = int(x.size // int(x.shape[-1]))
    hidden, n_experts = int(weight.shape[1]), int(weight.shape[0])
    return (
        2 <= rows <= FEW_ROWS_MAX
        and int(x.shape[-1]) == hidden
        and hidden % 128 == 0
        and n_experts % 8 == 0
        and weight.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and x.dtype in (mx.float32, mx.bfloat16, mx.float16)
    )


def router_logits_rows(x: mx.array, weight: mx.array) -> mx.array:
    """``x [M, H] @ weight [E, H].T`` -> float32 ``[M, E]``, reading ``weight`` once (see above)."""
    hidden = int(x.shape[-1])
    rows = int(x.size // hidden)
    n_experts = int(weight.shape[0])
    (out,) = _router_rows_kernel(rows, hidden, n_experts)(
        inputs=[x.reshape(rows, hidden), weight],
        template=[("WT", weight.dtype)],
        grid=(64, n_experts // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, n_experts)],
        output_dtypes=[mx.float32],
    )
    return out


__all__ += ["FEW_ROWS_MAX", "masked_qmv_rows", "router_logits_rows", "router_rows_eligible"]

