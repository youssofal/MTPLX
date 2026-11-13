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
the only Python branch is on the static row count (gather order for large
batches, as in mlx-lm's ``SwitchGLU``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.qwen3_next import Qwen3NextMLP

from mtplx.attention_math import attention_gate

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


class PackedDense(nn.Module):
    """An affine-quantized ``[out, in]`` matrix with float32 scales and biases (group 32)."""

    def __init__(self, out_dims: int, in_dims: int, bits: int):
        super().__init__()
        self.weight = mx.zeros((out_dims, in_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((out_dims, in_dims // DENSE_GROUP), dtype=mx.float32)
        self.biases = mx.zeros((out_dims, in_dims // DENSE_GROUP), dtype=mx.float32)
        self._bits = bits

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True, group_size=DENSE_GROUP, bits=self._bits)


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
        """``x2 [T, H]`` -> ``(ids int32 [T, K], weights float32 [T, K])``."""
        if isinstance(self.gate, nn.Linear) and "bias" not in self.gate:
            logits = float32_logits(x2, self.gate.weight, weight_tf32_exact=self._router_tf32_exact or None)
        else:  # a quantized router: its own kernel, read in float32
            logits = self.gate(x2.astype(mx.float32)).astype(mx.float32)
        p = mx.softmax(logits, axis=-1, precise=True)
        ids, p_sel = top_k_lowest_id(p, self.top_k)
        denom = mx.maximum(p_sel.sum(axis=-1, keepdims=True), ROUTER_WEIGHT_FLOOR)
        return ids, p_sel / denom

    def _experts(self, xl: mx.array, ids: mx.array, groups: mx.array) -> mx.array:
        """Group latents ``xl [T, G, L]`` -> each route's scaled expert output ``h [T, K, L]`` (float32)."""
        t, n_groups, lat = xl.shape
        k = ids.shape[-1]
        ex = self.experts
        xl3 = xl.reshape(t * n_groups, 1, lat).astype(self.latent_dtype)
        lhs = (mx.arange(t, dtype=mx.int32)[:, None] * n_groups + groups).astype(mx.uint32)
        do_sort = t * k >= self.sort_threshold
        if do_sort:
            flat = ids.reshape(-1)
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            rid = flat[order]
            xs = xl3[lhs.reshape(-1)[order]]  # [R, 1, L]
            gate = ex.gate_proj(xs, rid, sorted_indices=True)
            up = ex.up_proj(xs, rid, sorted_indices=True)
            sc = ex.scalars[rid][:, None, :]  # [R, 1, 3]
        else:
            rid = ids
            gate = ex.gate_proj(xl3, ids, lhs_indices=lhs)  # [T, K, 1, I]
            up = ex.up_proj(xl3, ids, lhs_indices=lhs)
            sc = ex.scalars[ids][..., None, :]  # [T, K, 1, 3]
        g = gate.astype(mx.float32) * sc[..., 0:1]
        u = up.astype(mx.float32) * sc[..., 1:2]
        a = (nn.silu(g) * u).astype(self.latent_dtype)
        h = ex.down_proj(a, rid, sorted_indices=do_sort).astype(mx.float32) * sc[..., 2:3]
        if do_sort:
            h = h[inv]
        return h.reshape(t, k, lat)

    def __call__(self, x: mx.array, *, return_routing: bool = False):
        shape = x.shape
        geo = self.geometry
        x2 = x.reshape(-1, shape[-1])
        t = x2.shape[0]

        ids, w = self.route(x2)
        groups = self.labels[ids]  # [T, K]
        xl = self.dense_d(x2.astype(mx.float32)).astype(mx.float32).reshape(t, geo.n_groups, geo.latent_dim)
        h = self._experts(xl, ids, groups)  # [T, K, L] float32

        # per-group combine: z[g] = sum over routes in group g of w * h, then one U matmul
        # (elementwise over the K routes, not a matmul: float32 GEMMs may run as TF32)
        gid = mx.arange(geo.n_groups, dtype=mx.int32)[:, None]  # [G, 1]
        hw = h * w[..., None]  # [T, K, L]
        zero = mx.array(0.0, dtype=mx.float32)
        z = None
        for j in range(self.top_k):
            term = mx.where((groups[:, j, None, None] == gid), hw[:, j, None, :], zero)  # [T, G, L]
            z = term if z is None else z + term
        routed = self.dense_u(z.reshape(t, geo.n_groups * geo.latent_dim)).astype(mx.float32)

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
