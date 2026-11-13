"""Edge0 latent MoE kernels: the SwiGLU epilogue and the per-group combine; the masked D/U GEMVs;
the fused routing and routed-expert kernels (router GEMV, softmax/top-k, gate_up + SwiGLU,
down + combine) at the end of the module.

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
# Single-row affine 4- or 8-bit GEMVs that skip the work of groups no route of
# the token selected. The dot product is MLX's ``qmv_fast_impl`` (quantized.h,
# MLX 0.32.2) line for line with float accumulation: 2 simdgroups x 4 rows
# per threadgroup, two 32-bit packs per lane (16 values at 4 bits, 8 at 8
# bits), K blocks of 32 lanes' worth (512 / 256 values), and at 4 bits x
# pre-divided by 16/256/4096 so nibble products need no shifts; one
# ``simd_sum`` per row. So a row it computes is bit-identical to
# ``mx.quantized_matmul`` on float32 inputs (x of another dtype is converted
# exactly on load; scales/biases may be float16 when that is lossless, which
# changes no value). Edge0-120B's D/U are 4-bit (ggml Q4_0), Edge0-145B's
# 8-bit (Q8_0).
#
# D (rows): output rows of group g are computed only if g is used; the rest
# are written as zeros (nothing reads them). U (columns): a K block is skipped
# when every group its columns fall in is unused (a block may span any number
# of groups); those inputs are exact zeros (the combine writes 0.0 there), so
# skipping them leaves every sum unchanged (up to the sign of an exactly-zero
# sum).

_QMV_HEADER = """
using namespace metal;
constant constexpr int SIMD = 32;
constant constexpr int PACK_FACTOR = 32 / BITS;                // values per 32-bit pack
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int VPT = PACK_FACTOR * PACKS_PER_THREAD;   // values per lane: 16 (4-bit), 8 (8-bit)
constant constexpr int BLOCK = VPT * SIMD;                     // 512 (4-bit), 256 (8-bit)
constant constexpr int RPS = 4;                                // rows per simdgroup
constant constexpr int NSG = 2;
constant constexpr int SCALE_STEP = GS / VPT;

template <typename XT>
inline float load_x(const device XT* x, thread float* xt) {
  float sum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VPT; i += 4) {
      const float a = float(x[i]), b = float(x[i + 1]), c = float(x[i + 2]), d = float(x[i + 3]);
      sum += a + b + c + d;
      xt[i] = a;
      xt[i + 1] = b / 16.0f;
      xt[i + 2] = c / 256.0f;
      xt[i + 3] = d / 4096.0f;
    }
  } else {  // 8 bits
    for (int i = 0; i < VPT; i++) {
      const float a = float(x[i]);
      sum += a;
      xt[i] = a;
    }
  }
  return sum;
}

inline float qdot(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0;
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (VPT / 4); i++) {
      accum +=
          (xt[4 * i] * (ws[i] & 0x000f) +
           xt[4 * i + 1] * (ws[i] & 0x00f0) +
           xt[4 * i + 2] * (ws[i] & 0x0f00) +
           xt[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else {  // 8 bits
    for (int i = 0; i < VPT; i++) {
      accum += xt[i] * w[i];
    }
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

template <typename P>
inline bool block_used(P groups, uint m, int k) {  // any group the K block [k, k + BLOCK) touches
  for (int g = k / LAT; g <= (k + BLOCK - 1) / LAT; ++g) {
    if (group_used(groups, m, g)) return true;
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
      result[row] += qdot(wl, xt, s, b, sum);
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
    if (!block_used(groups, m, k)) {
      ws += BLOCK * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK / GS;
      bs += BLOCK / GS;
      xp += BLOCK;
      continue;
    }
"""


@lru_cache(maxsize=None)
def _masked_qmv_kernel(kind: str, k_in: int, n_out: int, latent: int, top_k: int, group_size: int, bits: int = 4):
    header = (
        f"constant constexpr int K = {k_in};\nconstant constexpr int N = {n_out};\n"
        f"constant constexpr int LAT = {latent};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int GS = {group_size};\nconstant constexpr int BITS = {bits};\n" + _QMV_HEADER
    )
    body = _QMV_BODY.replace("ROW_SKIP", _ROW_SKIP if kind == "rows" else "").replace(
        "COL_SKIP", _COL_SKIP if kind == "cols" else ""
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_qmv_{kind}_k{k_in}_n{n_out}_l{latent}_t{top_k}_g{group_size}_b{bits}",
        input_names=["x", "w", "scales", "biases", "groups"],
        output_names=["y"],
        header=header,
        source=body,
    )


def masked_qmv_eligible(x: mx.array, w: mx.array, scales: mx.array, bits: int, group_size: int, latent: int, by: str, *, max_rows: int = 1) -> bool:
    """Shapes the masked GEMVs serve: up to ``max_rows`` rows (one for :func:`masked_qmv`), 4- or 8-bit,
    K a multiple of 512 (MLX's fast GEMV, which the kernels reproduce), groups aligned to the 8-row tiles."""
    if bits not in (4, 8) or group_size not in (32, 64, 128) or x.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return False
    if scales.dtype not in (mx.float32, mx.float16):
        return False
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    k_in = int(x.shape[-1])
    n_out = int(w.shape[0])
    if not 1 <= rows <= max_rows or k_in % 512 or n_out % 8 or int(w.shape[1]) * (32 // bits) != k_in:
        return False
    if max_rows > 1 and (n_out if by == "rows" else k_in) > 32 * latent:  # masked_qmv_rows keeps a 32-bit group mask
        return False
    return (n_out % latent == 0 and latent % 8 == 0) if by == "rows" else k_in % latent == 0


def _bits_of(w: mx.array, k_in: int) -> int:
    return 32 * int(w.shape[1]) // k_in


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
    kernel = _masked_qmv_kernel(by, k_in, n_out, int(latent), top_k, int(group_size), _bits_of(w, k_in))
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
        result[m][row] += qdot(ws + row * in_w, xt, s, b, sum);
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

# the bits of groups k / LAT .. (k + BLOCK - 1) / LAT (at most 32 groups, see masked_qmv_eligible)
_ROWS_COL_NEEDED = (
    "((gmask[m] & ((0xFFFFFFFFu >> uint(31 - (k + BLOCK - 1) / LAT)) & (0xFFFFFFFFu << uint(k / LAT)))) != 0u)"
)

#: Row counts the few-rows kernels are built for (a verify forward is 2..depth+1 rows).
FEW_ROWS_MAX = 8


@lru_cache(maxsize=None)
def _masked_qmv_rows_kernel(kind: str, rows: int, k_in: int, n_out: int, latent: int, top_k: int, group_size: int, bits: int = 4):
    header = (
        f"constant constexpr int K = {k_in};\nconstant constexpr int N = {n_out};\n"
        f"constant constexpr int LAT = {latent};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int GS = {group_size};\nconstant constexpr int M = {rows};\n"
        f"constant constexpr int BITS = {bits};\n" + _QMV_HEADER
    )
    row_used = "(((gmask[m] >> uint(out_row / LAT)) & 1u) != 0u)" if kind == "rows" else "true"
    col_needed = _ROWS_COL_NEEDED if kind == "cols" else "true"
    body = _QMV_ROWS_BODY.replace("ROW_USED", row_used).replace("COL_NEEDED", col_needed)
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_latent_qmv_{kind}_m{rows}_k{k_in}_n{n_out}_l{latent}_t{top_k}_g{group_size}_b{bits}",
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
    kernel = _masked_qmv_rows_kernel(by, rows, k_in, n_out, int(latent), top_k, int(group_size), _bits_of(w, k_in))
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



# -- routing and routed experts in four dispatches (rows 1..FEW_ROWS_MAX) -------------------
#
# Per layer the stock op chains spend ~11 dispatches on routing (float32 GEMV,
# softmax, negate, argsort, slice, cast, take, sum, maximum, divide, label
# gather) and ~8 on the routed experts (index arithmetic, gate_up gather_qmm,
# SwiGLU, down gather_qmm, combine). These four kernels do the same work and
# reproduce every value bit for bit:
#
# 1. ``router_logits``: ``x @ W.T`` with MLX's one-row float32 GEMV summation
#    (``gemv.h``: 32 lanes x 4 consecutive columns per 128-column step, an
#    in-order accumulation per lane, then the ``simd_shuffle_down`` ladder).
#    The weight may be held in bfloat16 when that is exact (half the bytes;
#    the products see the same float32 values). Every row of a multi-row call
#    is the one-row result (MLX's multi-row GEMMs and ``router_logits_rows``
#    are not); the block still takes ``router_logits_rows`` (on the bfloat16
#    weight, the same bits) at verify widths, so that verify forwards keep
#    their established numerics and MTP output streams do not change.
# 2. ``route_topk``: per row, MLX's single-row softmax kernel (4 values per
#    thread, ``fast::exp``, the same simd and threadgroup sums), the top-K in
#    the order of a stable ascending argsort of ``-p`` (NaN last, ties to the
#    lower id), the sum of the K values as MLX's small row reduction does it
#    (a left fold from 0), ``maximum(sum, 2**-14)``, the division, and the
#    expert labels.
# 3. ``latent_gate_up_swiglu``: the merged gate_up projection of every route
#    with the SwiGLU epilogue. Each output row is MLX's ``fp_qmv``/
#    ``fp_qmv_fast`` (``fp_quantized.h``, what ``gather_qmm`` runs for one-row
#    routes) line for line: the same lane-to-column mapping, the same
#    in-order block accumulation and ``simd_sum``, rounded to the activation
#    dtype as ``gather_qmm`` returns it; the SwiGLU is ``latent_swiglu``'s.
#    The e2m1 weights are decoded through a 256-entry byte table built with
#    MLX's own conversion, so the products see the same floats. With several
#    rows, the first route of an expert computes every route of that expert
#    (one per row at most), so its weights stream once; the threadgroups of
#    the other routes exit.
# 4. ``latent_down_combine``: the down projection of every route (the same
#    ``fp_qmv`` reproduction, one pass over each distinct expert's rows) and,
#    in the same threadgroup, the per-group combine of ``latent_combine`` for
#    those rows, written straight into ``z [T, G, L]`` for the masked U.
#
# Each row of ``route_topk``, ``latent_gate_up_swiglu`` and ``latent_down_combine``
# is computed exactly as for that row alone (the expert kernels per route, at any
# row count), so with the verify-width router logits of ``router_logits_rows``
# the fused verify path gives the op chain's verify bits.

_FP4_HEADER = """
using namespace metal;
inline float fp4v(uint b) {  // MLX's e2m1 -> float (fp4.h), on the low nibble of b
  half c = as_type<half>(ushort((b & 7) << 9));
  c *= 16384.0;
  return static_cast<float>((b & 8) ? -c : c);
}
inline float e4m3v(uint8_t b) {  // MLX's e4m3 -> float (fp8.h)
  uint16_t v = (b & 127) << 7;
  half c = as_type<half>(v);
  c *= 256.0;
  return static_cast<float>((b & 128) ? -c : c);
}
template <int VPT>
inline float wqdot(const thread float* wf, const thread float* xt, float scale) {  // fp_quantized.h qdot, weights decoded
  float accum = 0;
  for (int i = 0; i < VPT / 4; i++) {
    accum += (xt[4 * i] * wf[4 * i] + xt[4 * i + 1] * wf[4 * i + 1] + xt[4 * i + 2] * wf[4 * i + 2] + xt[4 * i + 3] * wf[4 * i + 3]);
  }
  return scale * accum;
}
template <int PPT>
inline void decode_words(const thread uint* wq, const threadgroup float2* lut, thread float* wf) {
  for (int p = 0; p < PPT; p++) {
    for (int b = 0; b < 4; b++) {
      const float2 v = lut[(wq[p] >> (8 * b)) & 0xffu];
      wf[8 * p + 2 * b] = v.x;
      wf[8 * p + 2 * b + 1] = v.y;
    }
  }
}
"""

_FP4_LUT_INIT = """
  threadgroup float2 lut[256];
  for (int i = int(thread_position_in_threadgroup.x); i < 256; i += NSG * 32) {
    lut[i] = float2(fp4v(uint(i) & 15u), fp4v(uint(i) >> 4));
  }
"""

# ---- 1. router logits

_GEMV_SOURCE = """
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const int e0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * RPS;
  float result[M][RPS];
  for (int m = 0; m < M; m++)
    for (int r = 0; r < RPS; r++) result[m][r] = 0.0f;
  for (int i = 0; i < H / 128; ++i) {
    const int bn = i * 128 + int(lane) * 4;
    float v[M][4];
    for (int m = 0; m < M; m++)
      for (int tn = 0; tn < 4; tn++) v[m][tn] = static_cast<float>(x[m * H + bn + tn]);
    for (int r = 0; r < RPS; r++) {
      float inter[4];
      for (int tn = 0; tn < 4; tn++) inter[tn] = static_cast<float>(wt[(e0 + r) * H + bn + tn]);
      for (int m = 0; m < M; m++)
        for (int tn = 0; tn < 4; tn++) result[m][r] += inter[tn] * v[m][tn];
    }
  }
  for (int m = 0; m < M; m++) {
    for (int r = 0; r < RPS; r++) {
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[m][r] += simd_shuffle_down(result[m][r], sn);
      if (lane == 0) out[m * E + e0 + r] = result[m][r];
    }
  }
"""
_GEMV_RPS, _GEMV_NSG = 2, 4


@lru_cache(maxsize=None)
def _gemv_kernel(rows: int, hidden: int, n_experts: int):
    header = (
        "using namespace metal;\n"
        f"constant constexpr int M = {rows};\nconstant constexpr int H = {hidden};\nconstant constexpr int E = {n_experts};\n"
        f"constant constexpr int RPS = {_GEMV_RPS};\nconstant constexpr int NSG = {_GEMV_NSG};\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_router_gemv_m{rows}_h{hidden}_e{n_experts}",
        input_names=["x", "wt"],
        output_names=["out"],
        header=header,
        source=_GEMV_SOURCE,
    )


def router_logits_eligible(x: mx.array, weight: mx.array) -> bool:
    """Shapes :func:`router_logits` serves: 1..FEW_ROWS_MAX rows, ``H`` a multiple of 128, ``E`` of 8, and
    ``64 < H < 16 E``, where MLX's GEMV takes the layout reproduced here (for ``H >= 16 E`` it splits ``H``
    across simdgroups and sums in another order; matmul.cpp ``gemv_axbpy``)."""
    if x.ndim < 1 or weight.ndim != 2:
        return False
    hidden, n_experts = int(weight.shape[1]), int(weight.shape[0])
    rows = int(x.size // max(int(x.shape[-1]), 1))
    return (
        1 <= rows <= FEW_ROWS_MAX
        and int(x.shape[-1]) == hidden
        and hidden % 128 == 0
        and 64 < hidden < 16 * n_experts
        and n_experts % (_GEMV_RPS * _GEMV_NSG) == 0
        and weight.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and x.dtype in (mx.float32, mx.bfloat16, mx.float16)
    )


def router_logits(x: mx.array, weight: mx.array) -> mx.array:
    """``x [M, H] @ weight [E, H].T`` -> float32 ``[M, E]``; each row bit-identical to
    ``x[m:m+1].astype(float32) @ weight.astype(float32).T`` (MLX's float32 GEMV)."""
    hidden = int(x.shape[-1])
    rows = int(x.size // hidden)
    n_experts = int(weight.shape[0])
    (out,) = _gemv_kernel(rows, hidden, n_experts)(
        inputs=[x.reshape(rows, hidden), weight],
        grid=(32 * _GEMV_NSG * (n_experts // (_GEMV_RPS * _GEMV_NSG)), 1, 1),
        threadgroup=(32 * _GEMV_NSG, 1, 1),
        output_shapes=[(rows, n_experts)],
        output_dtypes=[mx.float32],
    )
    return out


# ---- 2. softmax, top-k, renormalisation, labels (one threadgroup per row)

_TOPK_HEADER = """
using namespace metal;
constant constexpr int E = {E};
constant constexpr int TOP_K = {K};
constant constexpr int TGS = {TGS};
constant constexpr int NSGR = TGS / 32;
constant constexpr int NCAND = NSGR * TOP_K;
constant constexpr int CPL = (NCAND + 31) / 32;
constant constexpr float FLOOR = 6.103515625e-05f;  // ROUTER_WEIGHT_FLOOR, 2**-14
inline bool lt_neg(float a, float b) {{  // MLX's sort LessThan on (-a, -b): NaN sorts last
  const bool an = isnan(a), bn = isnan(b);
  if (an | bn) return (!an) & bn;
  return -a < -b;
}}
inline bool before(float a, int ia, float b, int ib) {{  // a precedes b in a stable ascending argsort of -p
  if (ia == INT_MAX) return false;
  if (ib == INT_MAX) return true;
  if (lt_neg(a, b)) return true;
  if (lt_neg(b, a)) return false;
  return ia < ib;
}}
"""

_TOPK_SOURCE = """
  #pragma METAL fp contract(off)
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const int sgr = int(tid) / 32;
  const int lid = int(tid);
  const int m = int(threadgroup_position_in_grid.x);
  threadgroup float local_max[32];
  threadgroup float local_normalizer[32];
  threadgroup float cand_v[NCAND];
  threadgroup int cand_i[NCAND];
  // softmax: MLX's softmax_single_row (N_READS = 4), float32 in and out
  float ld[4];
  const device float* in = logits + m * E + lid * 4;
  if (lid * 4 + 4 <= E) {
    for (int i = 0; i < 4; i++) ld[i] = in[i];
  } else {
    for (int i = 0; i < 4; i++) ld[i] = (lid * 4 + i < E) ? in[i] : -INFINITY;
  }
  if (sgr == 0) {
    local_max[lane] = -INFINITY;
    local_normalizer[lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = -FLT_MAX;
  for (int i = 0; i < 4; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
  maxval = simd_max(maxval);
  if (lane == 0) local_max[sgr] = maxval;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgr == 0) {
    maxval = simd_max(local_max[lane]);
    if (lane == 0) local_max[0] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < 4; i++) {
    float exp_x = fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (lane == 0) local_normalizer[sgr] = normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgr == 0) {
    normalizer = simd_sum(local_normalizer[lane]);
    if (lane == 0) local_normalizer[0] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  float p[4];
  bool taken[4];
  for (int i = 0; i < 4; i++) {
    p[i] = ld[i] * normalizer;
    taken[i] = lid * 4 + i >= E;
  }
  // top-K, stage 1: each simdgroup's own top-K (the row's top-K lies in their union)
  for (int k = 0; k < TOP_K; k++) {
    float bv = 0.0f;
    int bi = INT_MAX;
    for (int i = 0; i < 4; i++) {
      if (!taken[i] && before(p[i], lid * 4 + i, bv, bi)) { bv = p[i]; bi = lid * 4 + i; }
    }
    for (ushort o = 16; o >= 1; o >>= 1) {
      const float ov = simd_shuffle_xor(bv, o);
      const int oi = simd_shuffle_xor(bi, o);
      if (before(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if (bi != INT_MAX && bi / 4 == lid) taken[bi % 4] = true;
    if (lane == 0) { cand_v[sgr * TOP_K + k] = bv; cand_i[sgr * TOP_K + k] = bi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // stage 2: the first simdgroup takes the top-K of the candidates in argsort order, then renormalises
  if (sgr == 0) {
    float cv[CPL];
    int ci[CPL];
    for (int c = 0; c < CPL; c++) {
      const int q = int(lane) + 32 * c;
      cv[c] = (q < NCAND) ? cand_v[q] : 0.0f;
      ci[c] = (q < NCAND) ? cand_i[q] : INT_MAX;
    }
    float total = 0;
    float my_v = 0.0f;
    int my_i = 0;
    for (int k = 0; k < TOP_K; k++) {
      float bv = 0.0f;
      int bi = INT_MAX;
      int bc = -1;
      for (int c = 0; c < CPL; c++) {
        if (before(cv[c], ci[c], bv, bi)) { bv = cv[c]; bi = ci[c]; bc = c; }
      }
      float wv = bv;
      int wi = bi;
      for (ushort o = 16; o >= 1; o >>= 1) {
        const float ov = simd_shuffle_xor(wv, o);
        const int oi = simd_shuffle_xor(wi, o);
        if (before(ov, oi, wv, wi)) { wv = ov; wi = oi; }
      }
      if (bc >= 0 && bi == wi) ci[bc] = INT_MAX;
      if (int(lane) == k) { my_v = wv; my_i = wi; }
      total = wv + total;  // MLX's row_reduce_small: a left fold from 0
    }
    if (int(lane) < TOP_K) {
      const float denom = isnan(total) ? total : (total > FLOOR ? total : FLOOR);  // MLX's Maximum
      ids[m * TOP_K + lane] = my_i;
      wts[m * TOP_K + lane] = my_v / denom;
      groups[m * TOP_K + lane] = labels[my_i];
    }
  }
"""


def _topk_threads(n_experts: int) -> int:
    return 32 * (((n_experts + 3) // 4 + 31) // 32)


@lru_cache(maxsize=None)
def _topk_kernel(n_experts: int, top_k: int):
    header = _TOPK_HEADER.format(E=n_experts, K=top_k, TGS=_topk_threads(n_experts))
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_route_topk_e{n_experts}_k{top_k}",
        input_names=["logits", "labels"],
        output_names=["ids", "wts", "groups"],
        header=header,
        source=_TOPK_SOURCE,
    )


def route_topk_eligible(logits_shape, labels: mx.array, top_k: int) -> bool:
    """Rows 1..FEW_ROWS_MAX, at most 4096 experts (MLX's one-pass softmax), ``top_k`` <= 32."""
    rows, n_experts = (int(d) for d in logits_shape)
    return 1 <= rows <= FEW_ROWS_MAX and 1 <= top_k <= min(32, n_experts) and n_experts <= 4096 and labels.dtype == mx.int32


def route_topk(logits: mx.array, labels: mx.array, top_k: int) -> tuple[mx.array, mx.array, mx.array]:
    """float32 ``logits [M, E]`` -> ``(ids int32, weights float32, groups int32)``, each ``[M, top_k]``.

    Bit-identical to ``softmax(precise=True)``, ``top_k_lowest_id``, ``w = p / maximum(sum p, 2**-14)``
    and ``labels[ids]`` in MLX ops, row by row.
    """
    rows, n_experts = (int(d) for d in logits.shape)
    tgs = _topk_threads(n_experts)
    ids, wts, groups = _topk_kernel(n_experts, int(top_k))(
        inputs=[logits, labels],
        grid=(tgs * rows, 1, 1),
        threadgroup=(tgs, 1, 1),
        output_shapes=[(rows, top_k)] * 3,
        output_dtypes=[mx.int32, mx.float32, mx.int32],
    )
    return ids, wts, groups


# ---- 3. gate_up + SwiGLU for every route

_GU_HELPERS = """
inline OutT swiglu_one(float gate, float up, float s_gate, float s_up) {  // latent_swiglu, one value
  #pragma METAL fp contract(off)
  const float g = gate * s_gate;
  const float u = up * s_up;
  const float y = 1.0f / (1.0f + metal::exp(metal::abs(g)));
  const float sig = (g < 0.0f) ? y : 1.0f - y;
  const float silu = g * sig;
  return OutT(silu * u);
}

// FPS features (gate rows f0.., up rows FF+f0..) of one expert for MDJ routes, weights read once
// (GP: small inputs may be bound in constant memory)
template <typename XT_, int MDJ, typename GP>
inline void gate_up_routes(const device XT_* xl, const device uint8_t* wb, const device uint8_t* sb, GP groups,
                           thread const int* rl, int f0, uint lane, float s_gate, float s_up, device OutT* a,
                           const threadgroup float2* lut) {
  constexpr int IN_W = LAT / 2;   // bytes per weight row
  constexpr int IN_G = LAT / 16;  // scales per row
  const device XT_* xr[MDJ];
  for (int j = 0; j < MDJ; j++) xr[j] = xl + ((rl[j] / TOP_K) * NG + groups[rl[j]]) * LAT + lane * VPT;
  float res[MDJ][2 * FPS];
  for (int j = 0; j < MDJ; j++)
    for (int q = 0; q < 2 * FPS; q++) res[j][q] = 0.0f;
  for (int k = 0; k < LAT; k += BLK) {
    if (k + int(lane) * VPT >= LAT) break;  // fp_qmv's tail: lanes past K add nothing
    float xt[MDJ][VPT];
    for (int j = 0; j < MDJ; j++)
      for (int i = 0; i < VPT; i++) xt[j][i] = static_cast<float>(xr[j][k + i]);
    for (int q = 0; q < 2 * FPS; q++) {
      const int row = (q < FPS) ? (f0 + q) : (FF + f0 + q - FPS);
      uint wq[PPT];
      for (int p = 0; p < PPT; p++) wq[p] = *(const device uint*)(wb + row * IN_W + k / 2 + p * 4);
      const float sc = e4m3v(sb[row * IN_G + k / 16]);
      float wf[VPT];
      decode_words<PPT>(wq, lut, wf);
      for (int j = 0; j < MDJ; j++) res[j][q] += wqdot<VPT>(wf, xt[j], sc);
    }
  }
  for (int j = 0; j < MDJ; j++) {
    float gv[FPS], uv[FPS];
    for (int q = 0; q < FPS; q++) {
      gv[q] = static_cast<float>(static_cast<OutT>(simd_sum(res[j][q])));
      uv[q] = static_cast<float>(static_cast<OutT>(simd_sum(res[j][FPS + q])));
    }
    if (lane < uint(FPS)) a[rl[j] * FF + f0 + lane] = swiglu_one(gv[lane], uv[lane], s_gate, s_up);
  }
}
"""

_GU_SOURCE = """
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const int r0 = int(threadgroup_position_in_grid.y);
  const int f0 = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * FPS;
LUT_INIT
  const int e = ids[r0];
  int rl[NT];
  int nr = 1;
  rl[0] = r0;
  if (NT > 1) {
    // the first route of an expert computes all its routes (at most one per row); the others exit
    bool dup = false;
    for (int c = 0; c < (NR + 31) / 32; c++) {
      const int q = int(lane) + 32 * c;
      dup = dup || (q < r0 && ids[q] == e);
    }
    if (simd_any(dup)) {
      if (threadgroup_position_in_grid.x == 0 && sg == 0 && lane == 0) info[r0 * (NT + 1)] = 0;
      return;
    }
    for (int c = 0; c < (NR + 31) / 32; c++) {
      const int q = int(lane) + 32 * c;
      uint bits = uint(static_cast<ulong>(simd_ballot(q > r0 && q < NR && ids[min(q, NR - 1)] == e)));
      while (bits != 0u) {
        rl[nr++] = 32 * c + int(ctz(bits));
        bits &= bits - 1u;
      }
    }
    if (threadgroup_position_in_grid.x == 0 && sg == 0 && lane == 0) {
      info[r0 * (NT + 1)] = nr;
      for (int j = 0; j < nr; j++) info[r0 * (NT + 1) + 1 + j] = rl[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int IN_W = LAT / 2;
  constexpr int IN_G = LAT / 16;
  const device uint8_t* wb = (const device uint8_t*)w + size_t(e) * (2 * FF) * IN_W + lane * PPT * 4;
  const device uint8_t* sb = scales + size_t(e) * (2 * FF) * IN_G + lane / (16 / VPT);
  const float s_gate = scalars[e * 3];
  const float s_up = scalars[e * 3 + 1];
  for (int c0 = 0; c0 < nr; c0 += 4) {
    const int n = min(4, nr - c0);
    if (n == 1) gate_up_routes<XT, 1>(xl, wb, sb, groups, rl + c0, f0, lane, s_gate, s_up, a, lut);
    else if (n == 2) gate_up_routes<XT, 2>(xl, wb, sb, groups, rl + c0, f0, lane, s_gate, s_up, a, lut);
    else if (n == 3) gate_up_routes<XT, 3>(xl, wb, sb, groups, rl + c0, f0, lane, s_gate, s_up, a, lut);
    else gate_up_routes<XT, 4>(xl, wb, sb, groups, rl + c0, f0, lane, s_gate, s_up, a, lut);
  }
""".replace("LUT_INIT", _FP4_LUT_INIT)

_MSL_TYPE = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}


def _expert_tiles(rows: int) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """(gate_up: simdgroups, features per simdgroup), (down: simdgroups, rows per threadgroup, rows per item)."""
    if rows == 1:
        return (2, 4), (10, 4, 4)
    return (2, 2), (8, 8, 4)


def _qmv_constants(k_in: int) -> str:
    ppt = 2 if k_in % 512 == 0 else 1  # fp_qmv_fast (K % 512 == 0) or fp_qmv
    return f"constant constexpr int PPT = {ppt};\nconstant constexpr int VPT = 8 * PPT;\nconstant constexpr int BLK = VPT * 32;\n"


@lru_cache(maxsize=None)
def _gate_up_kernel(lat: int, ff: int, top_k: int, n_groups: int, rows: int, nsg: int, fps: int, out_type: str):
    header = (
        _FP4_HEADER
        + f"constant constexpr int LAT = {lat};\nconstant constexpr int FF = {ff};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int NG = {n_groups};\nconstant constexpr int NT = {rows};\nconstant constexpr int NR = {rows * top_k};\n"
        f"constant constexpr int NSG = {nsg};\nconstant constexpr int FPS = {fps};\n"
        + _qmv_constants(lat)
        + f"typedef {out_type} OutT;\n"
        + _GU_HELPERS
        + ("" if rows > 1 else "#define info ((device int*)0)\n")
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_gate_up_swiglu_l{lat}_f{ff}_k{top_k}_g{n_groups}_t{rows}_s{nsg}_p{fps}_{out_type}",
        input_names=["xl", "w", "scales", "ids", "groups", "scalars"],
        output_names=["a"] + (["info"] if rows > 1 else []),
        header=header,
        source=_GU_SOURCE,
    )


def latent_experts_eligible(xl: mx.array, gate_up_w: mx.array, gate_up_s: mx.array, down_w: mx.array, down_s: mx.array, top_k: int) -> bool:
    """Shapes the fused expert kernels serve: ``xl [T, G, L]`` with 1..FEW_ROWS_MAX rows in float32/bfloat16/float16,
    a merged nvfp4 gate_up ``[E, 2I, L/8]`` and down ``[E, L, I/8]`` (uint32 words, uint8 scales, group 16)."""
    if xl.ndim != 3 or xl.dtype not in _MSL_TYPE or gate_up_w.ndim != 3 or down_w.ndim != 3:
        return False
    rows, _, lat = (int(d) for d in xl.shape)
    n_exp, two_ff, words = (int(d) for d in gate_up_w.shape)
    ff = two_ff // 2
    (gu_nsg, gu_fps), (_, dn_rb, dn_rpi) = _expert_tiles(rows)
    return (
        1 <= rows <= FEW_ROWS_MAX
        and gate_up_w.dtype == mx.uint32 and down_w.dtype == mx.uint32
        and gate_up_s.dtype == mx.uint8 and down_s.dtype == mx.uint8
        and words * 8 == lat and lat % 16 == 0 and ff % 16 == 0 and two_ff % 8 == 0 and lat % 8 == 0
        and tuple(gate_up_s.shape) == (n_exp, two_ff, lat // 16)
        and tuple(down_w.shape) == (n_exp, lat, ff // 8)
        and tuple(down_s.shape) == (n_exp, lat, ff // 16)
        and ff % (gu_nsg * gu_fps) == 0 and lat % dn_rb == 0 and dn_rb % dn_rpi == 0
        and 1 <= top_k <= 32
    )


def latent_gate_up_swiglu(xl: mx.array, w: mx.array, scales: mx.array, ids: mx.array, groups: mx.array, scalars: mx.array):
    """``xl [T, G, L]``, merged gate_up nvfp4 ``w [E, 2I, L/8]``/``scales [E, 2I, L/16]``, ``ids``/``groups [T, K]`` int32,
    ``scalars [E, 3]`` float32 -> ``(a [T*K, I] in xl's dtype, info)``.

    ``a`` is bit-identical to ``latent_swiglu(gather_qmm(xl, gate_up, lhs = t*G + g, rhs = ids))``; ``info``
    (``None`` for one row) lists each expert's routes for :func:`latent_down_combine`.
    """
    rows, n_groups, lat = (int(d) for d in xl.shape)
    top_k = int(ids.shape[-1])
    ff = int(w.shape[1]) // 2
    n_routes = rows * top_k
    (nsg, fps), _ = _expert_tiles(rows)
    kernel = _gate_up_kernel(lat, ff, top_k, n_groups, rows, nsg, fps, _MSL_TYPE[xl.dtype])
    outs = kernel(
        inputs=[xl, w, scales, ids.reshape(-1), groups.reshape(-1), scalars],
        template=[("XT", xl.dtype)],
        grid=(32 * nsg * (ff // (nsg * fps)), n_routes, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(n_routes, ff)] + ([(n_routes * (rows + 1),)] if rows > 1 else []),
        output_dtypes=[xl.dtype] + ([mx.int32] if rows > 1 else []),
    )
    return outs[0], (outs[1] if rows > 1 else None)


# ---- 4. down + per-group combine

_DN_HELPERS = """
// RPI down rows of one expert for MDJ routes, weights read once; results rounded to the activation dtype
template <int MDJ>
inline void down_routes(const device XT* a, const device uint8_t* wb, const device uint8_t* sb, thread const int* rl,
                        uint lane, threadgroup float* res, int rq, const threadgroup float2* lut) {
  constexpr int IN_W = FF / 2;
  constexpr int IN_G = FF / 16;
  float acc[MDJ][RPI];
  for (int j = 0; j < MDJ; j++)
    for (int q = 0; q < RPI; q++) acc[j][q] = 0.0f;
  for (int k = 0; k < FF; k += BLK) {
    if (k + int(lane) * VPT >= FF) break;
    float xt[MDJ][VPT];
    for (int j = 0; j < MDJ; j++)
      for (int i = 0; i < VPT; i++) xt[j][i] = static_cast<float>(a[rl[j] * FF + k + lane * VPT + i]);
    for (int q = 0; q < RPI; q++) {
      uint wq[PPT];
      for (int p = 0; p < PPT; p++) wq[p] = *(const device uint*)(wb + q * IN_W + k / 2 + p * 4);
      const float sc = e4m3v(sb[q * IN_G + k / 16]);
      float wf[VPT];
      decode_words<PPT>(wq, lut, wf);
      for (int j = 0; j < MDJ; j++) acc[j][q] += wqdot<VPT>(wf, xt[j], sc);
    }
  }
  for (int j = 0; j < MDJ; j++)
    for (int q = 0; q < RPI; q++) {
      const float v = simd_sum(acc[j][q]);
      if (lane == 0) res[rl[j] * RB + rq + q] = static_cast<float>(static_cast<XT>(v));
    }
}

// latent_combine for the threadgroup's RB rows: a left fold over the routes of (down * s_down) * w
template <typename P1, typename P2, typename P3, typename P4>  // small inputs may be bound in constant memory
inline void combine_rows(const threadgroup float* res, P1 ids, P2 scalars, P3 wts, P4 groups, device float* z, int l0, uint tid) {
  #pragma METAL fp contract(off)
  for (int o = int(tid); o < NT * NG * RB; o += NSG * 32) {
    const int q = o % RB;
    const int g = (o / RB) % NG;
    const int t = o / (RB * NG);
    float acc = 0.0f;
    for (int j = 0; j < TOP_K; ++j) {
      const int r = t * TOP_K + j;
      float v = 0.0f;
      if (groups[r] == g) {
        const float h = res[r * RB + q] * scalars[ids[r] * 3 + 2];
        v = h * wts[r];
      }
      acc = (j == 0) ? v : acc + v;
    }
    z[(t * NG + g) * LAT + l0 + q] = acc;
  }
}
"""

_DN_SOURCE = """
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint tid = thread_position_in_threadgroup.x;
  const int l0 = int(threadgroup_position_in_grid.x) * RB;
  threadgroup float res[NR * RB];
  threadgroup int tinfo[NT > 1 ? NR * (NT + 1) : 1];
  threadgroup int slot_r[NR];
  threadgroup int n_slots[1];
LUT_INIT
  if (NT > 1) {
    for (int i = int(tid); i < NR * (NT + 1); i += NSG * 32) tinfo[i] = info[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (NT > 1) {
    if (sg == 0) {  // the distinct experts in order of first use (latent_gate_up_swiglu marked each one's first route)
      int count = 0;
      for (int c = 0; c < (NR + 31) / 32; c++) {
        const int r = int(lane) + 32 * c;
        const bool first = r < NR && tinfo[min(r, NR - 1) * (NT + 1)] > 0;
        const uint bits = uint(static_cast<ulong>(simd_ballot(first)));
        if (first) slot_r[count + int(popcount(bits & ((1u << lane) - 1u)))] = r;
        count += int(popcount(bits));
      }
      if (lane == 0) n_slots[0] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int nslots = (NT > 1) ? n_slots[0] : TOP_K;
  constexpr int IN_W = FF / 2;
  constexpr int IN_G = FF / 16;
  constexpr int RQS = RB / RPI;
  for (int it = int(sg); it < nslots * RQS; it += NSG) {
    const int s = it / RQS;
    const int rq = (it % RQS) * RPI;
    const int r0 = (NT > 1) ? slot_r[s] : s;
    const int e = ids[r0];
    int rl[NT];
    int nr = 1;
    rl[0] = r0;
    if (NT > 1) {
      nr = tinfo[r0 * (NT + 1)];
      for (int j = 0; j < NT; j++) rl[j] = tinfo[r0 * (NT + 1) + 1 + min(j, nr - 1)];
    }
    const device uint8_t* wb = (const device uint8_t*)w + (size_t(e) * LAT + l0 + rq) * IN_W + lane * PPT * 4;
    const device uint8_t* sb = scales + (size_t(e) * LAT + l0 + rq) * IN_G + lane / (16 / VPT);
    for (int c0 = 0; c0 < nr; c0 += 4) {
      const int n = min(4, nr - c0);
      if (n == 1) down_routes<1>(a, wb, sb, rl + c0, lane, res, rq, lut);
      else if (n == 2) down_routes<2>(a, wb, sb, rl + c0, lane, res, rq, lut);
      else if (n == 3) down_routes<3>(a, wb, sb, rl + c0, lane, res, rq, lut);
      else down_routes<4>(a, wb, sb, rl + c0, lane, res, rq, lut);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  combine_rows(res, ids, scalars, wts, groups, z, l0, tid);
""".replace("LUT_INIT", _FP4_LUT_INIT)


@lru_cache(maxsize=None)
def _down_combine_kernel(lat: int, ff: int, top_k: int, n_groups: int, rows: int, nsg: int, rb: int, rpi: int, x_type: str):
    header = (
        _FP4_HEADER
        + f"constant constexpr int LAT = {lat};\nconstant constexpr int FF = {ff};\nconstant constexpr int TOP_K = {top_k};\n"
        f"constant constexpr int NG = {n_groups};\nconstant constexpr int NT = {rows};\nconstant constexpr int NR = {rows * top_k};\n"
        f"constant constexpr int NSG = {nsg};\nconstant constexpr int RB = {rb};\nconstant constexpr int RPI = {rpi};\n"
        + _qmv_constants(ff)
        + f"typedef {x_type} XT;\n"
        + _DN_HELPERS
        + ("" if rows > 1 else "constant int info[1] = {0};\n")
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_edge0_down_combine_l{lat}_f{ff}_k{top_k}_g{n_groups}_t{rows}_s{nsg}_rb{rb}_rp{rpi}_{x_type}",
        input_names=["a", "w", "scales", "ids", "groups", "scalars", "wts"] + (["info"] if rows > 1 else []),
        output_names=["z"],
        header=header,
        source=_DN_SOURCE,
    )


def latent_down_combine(a, w, scales, ids, groups, scalars, wts, info, n_groups: int) -> mx.array:
    """``a [T*K, I]`` (from :func:`latent_gate_up_swiglu`), down nvfp4 ``w [E, L, I/8]``/``scales [E, L, I/16]``,
    ``ids``/``groups [T, K]``, ``scalars [E, 3]``, float32 ``wts [T, K]`` -> float32 ``z [T, G, L]``.

    Bit-identical to ``latent_combine(gather_qmm(a, down, rhs = ids), ...)``.
    """
    rows, top_k = (int(d) for d in ids.shape)
    ff = int(a.shape[-1])
    lat = int(w.shape[1])
    _, (nsg, rb, rpi) = _expert_tiles(rows)
    kernel = _down_combine_kernel(lat, ff, top_k, int(n_groups), rows, nsg, rb, rpi, _MSL_TYPE[a.dtype])
    inputs = [a, w, scales, ids.reshape(-1), groups.reshape(-1), scalars, wts.reshape(-1)] + ([info] if rows > 1 else [])
    (z,) = kernel(
        inputs=inputs,
        grid=(32 * nsg * (lat // rb), 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(rows, int(n_groups), lat)],
        output_dtypes=[mx.float32],
    )
    return z


__all__ += [
    "latent_down_combine",
    "latent_experts_eligible",
    "latent_gate_up_swiglu",
    "route_topk",
    "route_topk_eligible",
    "router_logits",
    "router_logits_eligible",
]
