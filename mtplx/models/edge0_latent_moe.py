"""The Edge0 latent sparse-MoE block on its natively 4-bit weights.

Edge0-120B is Qwen3.8-Flash-Next with the routed experts replaced by latent
experts: every token is projected into ``G`` group latents by ``D``, each routed
expert runs a SwiGLU on its group's latent, and the weighted expert outputs are
summed per group and mapped back to the hidden size by ``U``. The router, the
shared expert and its sigmoid gate are Flash Next's. Semantics:
:func:`mtplx.edge0.reference.latent_moe` (``combine="per_group"``), which is
the oracle the tests compare against.

Per token ``x [H]``::

    xl[g]   = D[g] @ x                                   all G groups, one affine matmul
    p       = softmax(W_r @ x)  (float32)                top-K by p, ties to the lower id,
    w       = p_sel / max(sum p_sel, 2**-14)             renormalised
    h_j     = s_down * A_down @ (silu(s_gate * A_gate @ xl[g_j]) * (s_up * A_up @ xl[g_j]))
    z[g]    = sum_{j: g_j = g} w_j * h_j                 per-group combine
    out     = U_cat @ concat_g z[g] + sigmoid(w_s . x) * shared_expert(x)

``A_*`` are MLX ``nvfp4`` stacks evaluated with ``gather_qmm`` (never expanded
to float); ``D`` and ``U`` are affine 4- or 8-bit with float32 scales (exact
copies of the ggml Q4_0 / Q8_0 blocks, see :mod:`mtplx.edge0.latent_convert`);
``U_cat`` is ``U`` regrouped to ``[H, G * L]`` so the combine is one matmul.
The per-expert scalars are applied in float32 on the projection outputs
(``s_down`` is ~1e-4: folded into 16-bit weights it would underflow).

The class deliberately does not derive from mlx-lm's or MTPLX's MoE blocks
(installers match those with ``isinstance``), but ``gate``,
``shared_expert.{gate,up,down}_proj`` and ``shared_expert_gate`` have the names
and shapes of ``Qwen3NextSparseMoeBlock``'s, so shell weights load into either.

Everything in ``__call__`` is traceable by ``mx.compile``: no host reads, and
the only Python branches are on the static row count (gather order for large
batches, as in mlx-lm's ``SwitchGLU``; the one-token D/U kernels).

Decode speed (2026-11-13; the 120B at T=1, 48 layers, M5 Max, bf16: 17.4 ->
7.9 ms eager). Each change reproduces the op chain it replaced bit for bit
(:mod:`mtplx.kernels.edge0_latent`):

- the per-group combine and the SwiGLU epilogue (with the ``s_gate``/``s_up``/
  ``s_down`` scalars) are one Metal dispatch each instead of ~40 small ops;
- for one token, ``D`` computes only the rows of the groups its routes use and
  ``U`` skips the 512-column blocks that are all unused groups (exact zeros);
  on the 120B a token's 10 routes touch 4.3 of 8 groups on average;
- after loading, ``gate_proj``/``up_proj`` are stacked into one
  ``gate_up_proj`` (one routed matmul instead of two) and the D/U scales and
  biases are held in float16 where every value round-trips (fewer bytes).

Verify widths (2..``FEW_ROWS_MAX`` tokens, an MTP verify forward; 120B, 4
tokens: 17.3 -> 13.1 ms over 48 layers). ``D`` and ``U`` run the one-token
masked GEMVs for all rows in one dispatch, each weight tile read once, and each
row bit-identical to the one-token kernel on that row
(:func:`mtplx.kernels.edge0_latent.masked_qmv_rows`); the router logits come
from a float32 GEMV kernel that reads the weight once (float32-accurate like
the one-token GEMV, not bit-identical to it). The routed experts stay on
``gather_qmm``: neighbouring tokens share ~40% of their experts, but the
concurrent gathers already hit the same weights in cache (a kernel that gave
each distinct expert one threadgroup measured slower).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from mtplx.attention_math import attention_gate
from mtplx.kernels.edge0_latent import (
    FEW_ROWS_MAX,
    latent_combine,
    latent_combine_reference,
    latent_swiglu,
    latent_swiglu_reference,
    masked_qmv,
    masked_qmv_eligible,
    masked_qmv_rows,
    router_logits_rows,
    router_rows_eligible,
)

ROUTER_WEIGHT_FLOOR = 6.103515625e-05  # 2**-14, mtplx.edge0.reference.ROUTER_WEIGHT_FLOOR
DEFAULT_TOP_K = 10
NVFP4_GROUP = 16
DENSE_GROUP = 32
SORT_THRESHOLD = 64  # routes per call from which the expert matmuls run in expert order


@dataclass(frozen=True)
class LatentGeometry:
    """Sizes of one latent MoE layer; read off the converted arrays, never assumed."""

    hidden: int
    latent_dim: int
    expert_ff: int
    n_groups: int
    n_experts: int
    dense_bits: int = 4

    @classmethod
    def from_arrays(cls, arrays: dict[str, Any], prefix: str = "") -> LatentGeometry:
        """Infer the geometry from :func:`mtplx.edge0.latent_convert.convert_layer` output."""

        def shape(name: str) -> tuple[int, ...]:
            return tuple(int(d) for d in arrays[prefix + name].shape)

        n_experts, expert_ff, lat_groups = shape("experts.gate_proj.scales")
        latent = lat_groups * NVFP4_GROUP
        gl, h_groups = shape("dense_d.scales")
        hidden = h_groups * DENSE_GROUP
        words = shape("dense_d.weight")[1]
        dense_bits = words * 32 // hidden
        if gl % latent:
            raise ValueError(f"dense_d has {gl} rows, not a multiple of the latent size {latent}")
        return cls(hidden=hidden, latent_dim=latent, expert_ff=expert_ff, n_groups=gl // latent, n_experts=n_experts, dense_bits=dense_bits)


class PackedExpertProjection(nn.Module):
    """One stacked expert projection in MLX ``nvfp4``: ``weight`` uint32 ``[E, out, in / 8]``, ``scales`` uint8 ``[E, out, in / 16]``."""

    def __init__(self, n_experts: int, out_dims: int, in_dims: int):
        super().__init__()
        self.weight = mx.zeros((n_experts, out_dims, in_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((n_experts, out_dims, in_dims // NVFP4_GROUP), dtype=mx.uint8)

    def __call__(self, x: mx.array, rhs_indices: mx.array, lhs_indices: mx.array | None = None, sorted_indices: bool = False) -> mx.array:
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=NVFP4_GROUP,
            bits=4,
            mode="nvfp4",
            sorted_indices=sorted_indices,
        )


class LatentExperts(nn.Module):
    """The routed latent experts: three ``nvfp4`` stacks plus float32 ``scalars [E, 3]`` (s_gate, s_up, s_down)."""

    def __init__(self, n_experts: int, latent_dim: int, expert_ff: int):
        super().__init__()
        self.gate_proj = PackedExpertProjection(n_experts, expert_ff, latent_dim)
        self.up_proj = PackedExpertProjection(n_experts, expert_ff, latent_dim)
        self.down_proj = PackedExpertProjection(n_experts, latent_dim, expert_ff)
        self.scalars = mx.ones((n_experts, 3), dtype=mx.float32)

    @property
    def merged(self) -> bool:
        return "gate_up_proj" in self

    def merge_gate_up(self) -> None:
        """Stack ``gate_proj`` and ``up_proj`` row-wise into one ``gate_up_proj [E, 2I, L/8]``.

        Lossless (the ``nvfp4`` blocks are copied, each output row keeps its
        own weights and scales) and bit-identical in output: a row of
        ``gather_qmm`` depends only on that row. One expert matmul per route
        instead of two. Call after the weights are loaded; it evaluates the
        stack and drops the two originals.
        """
        if self.merged:
            return
        e, ff, words = (int(d) for d in self.gate_proj.weight.shape)
        fused = PackedExpertProjection(e, 2 * ff, words * 8)
        fused.weight = mx.concatenate([self.gate_proj.weight, self.up_proj.weight], axis=1)
        fused.scales = mx.concatenate([self.gate_proj.scales, self.up_proj.scales], axis=1)
        mx.eval(fused.weight, fused.scales)
        self.gate_up_proj = fused
        del self["gate_proj"]
        del self["up_proj"]


class PackedDense(nn.Module):
    """An affine-quantized ``[out, in]`` matrix with float32 scales and biases (group 32).

    :meth:`compact_scales` may hold the scales and biases in float16 when
    every value round-trips exactly (the ggml blocks they copy store float16
    scales); the matmul then reads them back as the same float32 values.
    """

    def __init__(self, out_dims: int, in_dims: int, bits: int):
        super().__init__()
        self.weight = mx.zeros((out_dims, in_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((out_dims, in_dims // DENSE_GROUP), dtype=mx.float32)
        self.biases = mx.zeros((out_dims, in_dims // DENSE_GROUP), dtype=mx.float32)
        self._bits = bits

    @property
    def bits(self) -> int:
        return self._bits

    def compact_scales(self) -> bool:
        """Store scales and biases as float16 if that is exact for every value; returns whether it did."""
        if self.scales.dtype != mx.float32:
            return self.scales.dtype == mx.float16
        half = [a.astype(mx.float16) for a in (self.scales, self.biases)]
        exact = all(mx.array_equal(h.astype(mx.float32), a).item() for h, a in zip(half, (self.scales, self.biases)))
        if exact:
            mx.eval(half)
            self.scales, self.biases = half
        return exact

    def __call__(self, x: mx.array) -> mx.array:
        scales, biases = self.scales, self.biases
        if scales.dtype != x.dtype:  # compacted float16: the same values in float32
            scales, biases = scales.astype(x.dtype), biases.astype(x.dtype)
        return mx.quantized_matmul(x, self.weight, scales, biases, transpose=True, group_size=DENSE_GROUP, bits=self._bits)


def _split_tf32(a: mx.array) -> tuple[mx.array, mx.array]:
    """float32 ``a`` -> ``(hi, lo)``, ``hi`` exactly representable in TF32 (10 mantissa bits), ``a = hi + lo``."""
    hi = mx.view(mx.view(a, mx.uint32) & mx.array(0xFFFFE000, dtype=mx.uint32), mx.float32)
    return hi, a - hi


def float32_logits(x: mx.array, weight: mx.array, *, weight_tf32_exact: bool | None = None) -> mx.array:
    """``x @ weight.T`` to float32 accuracy even where MLX runs float32 GEMMs as TF32.

    On GPUs with tensor units (M5) MLX multiplies float32 matrices with TF32
    operands once there is more than one row (unless ``MLX_ENABLE_TF32=0``),
    which moves router logits by ~1e-3 relative and makes routing depend on the
    batch size. One row runs the float32 GEMV, which is exact; for more rows
    ``x`` is split into a TF32-exact high part and the remainder, which
    recovers ~2**-21 relative accuracy. A bfloat16 or float16 weight (or one
    flagged ``weight_tf32_exact``) needs no split, so both parts share one
    matmul; any other float32 weight is split too (three products).
    """
    t = x.shape[0]
    x = x.astype(mx.float32)
    if t == 1:
        return x @ weight.astype(mx.float32).T
    if weight_tf32_exact is None:
        weight_tf32_exact = weight.dtype in (mx.bfloat16, mx.float16)
    xh, xl = _split_tf32(x)
    if weight_tf32_exact:
        both = mx.concatenate([xh, xl], axis=0) @ weight.astype(mx.float32).T
        return both[:t] + both[t:]
    wh, wl = _split_tf32(weight.astype(mx.float32))
    wh_t = wh.T
    return xh @ wh_t + (xh @ wl.T + xl @ wh_t)


def top_k_lowest_id(p: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """The ``k`` largest entries of each row of ``p``, descending, ties to the lower index.

    One stable ascending ``argsort`` of ``-p``: MLX's Metal sort is a stable
    merge sort (odd-even transposition within a thread, merge path taking the
    right run only on strict less), so equal values keep ascending index order,
    and NaN sorts last. ``argpartition`` gives neither the order nor the tie rule.
    Returns ``(ids int32 [..., k], values [..., k])``.
    """
    idx = mx.argsort(-p, axis=-1)[..., :k].astype(mx.int32)
    return idx, mx.take_along_axis(p, idx, axis=-1)


def _use_kernels() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _combine(down: mx.array, ids: mx.array, scalars: mx.array, w: mx.array, groups: mx.array, n_groups: int) -> mx.array:
    """``z [T, G, L]`` float32: per group, the left fold over the routes of ``(float(down) * s_down) * w``.

    One Metal dispatch where available, bit-identical to the ``mx.where`` loop
    it replaced (:func:`mtplx.kernels.edge0_latent.latent_combine_reference`).
    """
    if _use_kernels():
        return latent_combine(down, ids, scalars, w, groups, n_groups)
    return latent_combine_reference(down, ids, scalars, w, groups, n_groups)


def _swiglu(gate_up, ids: mx.array, scalars: mx.array, expert_ff: int, dtype: mx.Dtype) -> mx.array:
    """``(silu(s_gate * gate) * (s_up * up)).astype(dtype)``, one dispatch where available (bit-identical)."""
    if _use_kernels():
        return latent_swiglu(gate_up, ids, scalars, expert_ff, dtype)
    return latent_swiglu_reference(gate_up, ids, scalars, expert_ff, dtype)


class LatentMoeBlock(nn.Module):
    """Drop-in MoE half for an Edge0 decoder layer: ``[B, S, H] -> [B, S, H]`` in the input dtype.

    Parameters (shapes for hidden ``H``, latent ``L``, expert width ``I``,
    ``G`` groups, ``E`` experts, shared width ``I_s``, dense bits ``b``):

    - ``gate.weight [E, H]`` (router), ``shared_expert.{gate,up}_proj.weight [I_s, H]``,
      ``shared_expert.down_proj.weight [H, I_s]``, ``shared_expert_gate.weight [1, H]``
      (plain ``nn.Linear``; quantizable like ``Qwen3NextSparseMoeBlock``'s);
    - ``dense_d.{weight [G*L, H*b/32] u32, scales [G*L, H/32] f32, biases}``;
    - ``dense_u.{weight [H, G*L*b/32] u32, scales [H, G*L/32] f32, biases}``;
    - ``experts.{gate,up}_proj.{weight [E, I, L/8] u32, scales [E, I, L/16] u8}``,
      ``experts.down_proj.{weight [E, L, I/8] u32, scales [E, L, I/16] u8}``,
      ``experts.scalars [E, 3] f32``;
    - ``labels [E] int32`` (group of each expert).

    ``latent_dtype`` is the activation dtype of the expert matmuls (float32 by
    default); the router, the routing weights, the scalars and the combine are
    float32 regardless.
    """

    def __init__(
        self,
        geometry: LatentGeometry,
        *,
        top_k: int = DEFAULT_TOP_K,
        shared_expert_ff: int,
        latent_dtype: mx.Dtype = mx.float32,
        sort_threshold: int = SORT_THRESHOLD,
    ):
        super().__init__()
        g = geometry
        self.geometry = g
        self.top_k = top_k
        self.num_experts = g.n_experts
        self.latent_dtype = latent_dtype
        self.sort_threshold = sort_threshold

        self.gate = nn.Linear(g.hidden, g.n_experts, bias=False)
        self.shared_expert = Qwen3NextMLP(g.hidden, shared_expert_ff)
        self.shared_expert_gate = nn.Linear(g.hidden, 1, bias=False)

        self.dense_d = PackedDense(g.n_groups * g.latent_dim, g.hidden, g.dense_bits)
        self.dense_u = PackedDense(g.hidden, g.n_groups * g.latent_dim, g.dense_bits)
        self.experts = LatentExperts(g.n_experts, g.latent_dim, g.expert_ff)
        self.labels = mx.zeros((g.n_experts,), dtype=mx.int32)
        self._router_tf32_exact = False

    def cast_router_float32(self) -> None:
        """Hold the router weight in float32 (saves a cast per call; ~5 MB per layer for the 120B).

        Call after the weights are loaded. Records whether the values are
        TF32-exact (they are when they came from bfloat16 or float16), which lets
        multi-row calls skip splitting the weight.
        """
        w = self.gate.weight
        if not isinstance(self.gate, nn.Linear) or w.dtype == mx.float32:
            return
        exact = w.dtype in (mx.bfloat16, mx.float16)
        self.gate.weight = w.astype(mx.float32)
        mx.eval(self.gate.weight)
        self._router_tf32_exact = exact

    @classmethod
    def from_model_args(cls, args: Any, geometry: LatentGeometry, **kwargs) -> LatentMoeBlock:
        """Build from Flash Next's ``ModelArgs`` (top-k and shared width) and the converted geometry."""
        if int(args.hidden_size) != geometry.hidden:
            raise ValueError(f"model hidden_size {args.hidden_size} != latent weights' hidden {geometry.hidden}")
        return cls(geometry, top_k=int(args.num_experts_per_tok), shared_expert_ff=int(args.shared_expert_intermediate_size), **kwargs)

    # -- weights ------------------------------------------------------------------

    def load_latent_arrays(self, arrays: dict[str, Any], prefix: str = "") -> None:
        """Install converted arrays (NumPy or MLX), checking every name, shape and dtype."""
        own = dict(tree_flatten(self.parameters()))
        items = []
        for key, value in arrays.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            if name not in own:
                raise KeyError(f"{key}: not a LatentMoeBlock parameter")
            a = value if isinstance(value, mx.array) else mx.array(value)
            ref = own[name]
            if tuple(a.shape) != tuple(ref.shape) or a.dtype != ref.dtype:
                raise ValueError(f"{key}: got {a.dtype}{tuple(a.shape)}, block expects {ref.dtype}{tuple(ref.shape)}")
            items.append((name, a))
        self.update(tree_unflatten(items))

    # -- forward --------------------------------------------------------------------

    def route(self, x2: mx.array) -> tuple[mx.array, mx.array]:
        """``x2 [T, H]`` -> ``(ids int32 [T, K], weights float32 [T, K])``.

        A few rows (speculative verify) take a float32 GEMV kernel that reads
        the router weight once for all rows (:func:`router_logits_rows`):
        float32-accurate like the one-row call, ~5x cheaper than the split
        GEMM :func:`float32_logits` runs at more than one row.
        """
        if isinstance(self.gate, nn.Linear) and "bias" not in self.gate:
            weight = self.gate.weight
            if x2.shape[0] > 1 and _use_kernels() and router_rows_eligible(x2, weight):
                logits = router_logits_rows(x2, weight)
            else:
                logits = float32_logits(x2, weight, weight_tf32_exact=self._router_tf32_exact or None)
        else:  # a quantized router: its own kernel, read in float32
            logits = self.gate(x2.astype(mx.float32)).astype(mx.float32)
        p = mx.softmax(logits, axis=-1, precise=True)
        ids, p_sel = top_k_lowest_id(p, self.top_k)
        denom = mx.maximum(p_sel.sum(axis=-1, keepdims=True), ROUTER_WEIGHT_FLOOR)
        return ids, p_sel / denom

    def _experts(self, xl: mx.array, ids: mx.array, groups: mx.array) -> mx.array:
        """Group latents ``xl [T, G, L]`` -> each route's scaled expert output ``h [T, K, L]`` (float32)."""
        down = self._experts_raw(xl, ids, groups)
        return down.astype(mx.float32) * self.experts.scalars[ids][..., 2:3]

    def _experts_raw(self, xl: mx.array, ids: mx.array, groups: mx.array) -> mx.array:
        """Group latents ``xl [T, G, L]`` -> ``down [T, K, L]`` in ``latent_dtype``.

        ``down`` is the down projection before its ``s_down`` scalar, which the
        combine applies (:func:`_combine`).
        """
        t, n_groups, lat = xl.shape
        k = ids.shape[-1]
        ex = self.experts
        ff = self.geometry.expert_ff
        xl3 = xl.reshape(t * n_groups, 1, lat).astype(self.latent_dtype)
        lhs = (mx.arange(t, dtype=mx.int32)[:, None] * n_groups + groups).astype(mx.uint32)
        do_sort = t * k >= self.sort_threshold
        if do_sort:
            flat = ids.reshape(-1)
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            rid = flat[order]
            xs = xl3[lhs.reshape(-1)[order]]  # [R, 1, L]
            proj = lambda m: m(xs, rid, sorted_indices=True)  # noqa: E731  [R, 1, I or 2I]
        else:
            rid = ids
            proj = lambda m: m(xl3, ids, lhs_indices=lhs)  # noqa: E731  [T, K, 1, I or 2I]
        gate_up = proj(ex.gate_up_proj) if ex.merged else (proj(ex.gate_proj), proj(ex.up_proj))
        a = _swiglu(gate_up, rid, ex.scalars, ff, self.latent_dtype)
        down = ex.down_proj(a, rid, sorted_indices=do_sort)
        if do_sort:
            down = down[inv]
        return down.reshape(t, k, lat)

    def __call__(self, x: mx.array, *, return_routing: bool = False):
        shape = x.shape
        geo = self.geometry
        x2 = x.reshape(-1, shape[-1])
        t = x2.shape[0]

        ids, w = self.route(x2)
        groups = self.labels[ids]  # [T, K]
        # One token: D and U only on the groups its routes use (bit-identical rows, see
        # mtplx.kernels.edge0_latent.masked_qmv); the D rows come out in latent_dtype directly.
        # A few tokens (verify): the same per row, each weight tile read once for all rows.
        kernels = t <= FEW_ROWS_MAX and _use_kernels()
        single = kernels and t == 1
        qmv = masked_qmv if single else masked_qmv_rows
        d, u = self.dense_d, self.dense_u
        if kernels and masked_qmv_eligible(x2, d.weight, d.scales, d.bits, DENSE_GROUP, geo.latent_dim, "rows", max_rows=FEW_ROWS_MAX):
            xl = qmv(x2, d.weight, d.scales, d.biases, groups, by="rows", latent=geo.latent_dim, group_size=DENSE_GROUP, out_dtype=self.latent_dtype)
        else:
            xl = d(x2.astype(mx.float32)).astype(mx.float32)
        xl = xl.reshape(t, geo.n_groups, geo.latent_dim)
        down = self._experts_raw(xl, ids, groups)  # [T, K, L] latent dtype

        # per-group combine: z[g] = sum over routes in group g of w * s_down * down, then one
        # U matmul (elementwise over the K routes, not a matmul: float32 GEMMs may run as TF32)
        z = _combine(down, ids, self.experts.scalars, w, groups, geo.n_groups)  # [T, G, L] float32
        z = z.reshape(t, geo.n_groups * geo.latent_dim)
        if kernels and masked_qmv_eligible(z, u.weight, u.scales, u.bits, DENSE_GROUP, geo.latent_dim, "cols", max_rows=FEW_ROWS_MAX):
            routed = qmv(z, u.weight, u.scales, u.biases, groups, by="cols", latent=geo.latent_dim, group_size=DENSE_GROUP, out_dtype=mx.float32)
        else:
            routed = u(z).astype(mx.float32)

        shared = attention_gate(self.shared_expert(x2), self.shared_expert_gate(x2))
        y = (routed + shared.astype(mx.float32)).astype(x.dtype).reshape(shape)
        if return_routing:
            lead = shape[:-1]
            return y, ids.reshape(*lead, self.top_k), w.reshape(*lead, self.top_k)
        return y


__all__ = [
    "DEFAULT_TOP_K",
    "LatentExperts",
    "LatentGeometry",
    "LatentMoeBlock",
    "PackedDense",
    "PackedExpertProjection",
    "ROUTER_WEIGHT_FLOOR",
    "top_k_lowest_id",
]
