"""NumPy reference for one Edge0 layer's MoE half (the latent sparse MoE).

This is the float oracle the MLX port is tested against. It follows the
semantics of the reference engine (``edge0-strata``, C++/CUDA) as read from its
source, not its arithmetic: the engine quantizes activations to Q8 before D, A
and U and rounds some hyper-connection activations to BF16, so it agrees with
this module to rounding level only.

One token, layer ``l``, hidden ``H``, latent ``L``, expert width ``I``, ``G``
groups, ``E`` experts, top ``K``, ``HC`` residual streams::

    x, inject = gr_read(R)                       # R: [HC, H] residual stack
    ids, w    = route(x, W_router)               # softmax over E, top-K, renormalised
    xl[g]     = D[g] @ x                         # D: [G, L, H]
    h_j       = A_down_j @ (silu(A_gate_j @ xl[g_j]) * (A_up_j @ xl[g_j]))   # g_j = labels[ids[j]]
    routed    = sum_j w_j * (U[g_j] @ h_j)       # U: [G, H, L]
    out       = routed + shared_expert(x)
    R'        = residual_write(R, out, inject)

``A_*`` are the NVFP4 projections already multiplied by the blob tail's
per-expert scalars (``s_gate`` on the gate output, ``s_up`` on the up output,
``s_down`` on the down output; scalars commute, so pre-scaling the matrices is
the same function).

Engine sources (``edge0-strata``):

- ``src/core/verify.cpp`` ``pre``/``post`` lambdas (D stage, router, shared
  expert, latent A and U stages, combine) and ``EPS = 1e-6``;
- ``include/strata/kernels/gr.hpp`` and ``src/kernels/cuda/gr.cu``
  (``gr_read`` / ``gr_write``);
- ``src/kernels/cuda/native_router.cu`` (softmax, top-10, tie rule, weight
  floor);
- ``include/strata/kernels/shared_expert.hpp`` and
  ``src/kernels/cuda/native_moe.cu`` (shared expert, combine);
- ``src/kernels/cuda/iq_kernels.cu`` (NVFP4 tail scales on the projection
  outputs) and ``src/kernels/cuda/latent.cu`` (U stage, group of a route is
  ``labels[expert]``).

The functions take plain arrays. ``layer_weights_from_pack`` is the only place
that knows about the file readers in :mod:`mtplx.edge0.pack`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RMS_EPS = 1e-6
"""The hyper-connection RMSNorm epsilon (``verify.cpp`` ``EPS``; GGUF ``layer_norm_rms_epsilon``)."""

ROUTER_WEIGHT_FLOOR = 6.103515625e-05
"""2**-14: the lower bound on the selected probabilities' sum before renormalising (``native_router.cu``)."""

DEFAULT_TOP_K = 10

ExpertWeights = tuple[np.ndarray, np.ndarray, np.ndarray]
"""(gate [I, L], up [I, L], down [L, I]) of one latent expert, tail scales already applied."""

ExpertSource = Callable[[int], ExpertWeights] | Mapping[int, ExpertWeights] | Sequence[ExpertWeights]


def _f(a, dtype) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- hyper-connection read / write


def gr_read(
    R: np.ndarray,
    w_norm: np.ndarray,
    w_down: np.ndarray,
    w_up: np.ndarray,
    w_inj: np.ndarray | None,
    *,
    eps: float = RMS_EPS,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Collapse the residual stack to the block input ``x`` and the per-stream injection.

    ``R`` is ``[HC, H]`` (or flat ``HC * H``, stream-major). ``w_norm`` is the
    per-stream RMSNorm gamma as the skeleton stores it, already ``1 + w``
    (length ``HC * H``); ``w_down`` is ``[r, HC * H]``, ``w_up`` ``[HC * H, r]``
    and ``w_inj`` ``[HC, HC * H]`` (None for the final mixer). Returns
    ``(mixed [H], inject [HC] or None)``::

        xn     = per-stream RMSNorm(R) * w_norm
        lo     = silu((W_down @ xn) / HC)          # the 1/HC is inside the silu
        mixed  = mean over streams of xn * sigmoid(W_up @ lo)
        inject = W_inj @ xn

    Source: ``include/strata/kernels/gr.hpp`` (``gr_read``), ``src/kernels/cuda/gr.cu``
    (``gr_norm_kernel``: RMS over each stream's ``H`` values, ``rsqrt(mean + eps)``).
    """
    w_norm = _f(w_norm, dtype).reshape(-1)
    w_down = _f(w_down, dtype)
    w_up = _f(w_up, dtype)
    hc_dim = w_norm.shape[0]
    R = _f(R, dtype)
    if R.ndim == 1:
        if w_inj is None:
            raise ValueError("a flat R needs w_inj to tell the stream count; pass R as [HC, H]")
        R = R.reshape(np.shape(w_inj)[0], -1)
    hc, hidden = R.shape
    if hc * hidden != hc_dim:
        raise ValueError(f"R is {R.shape} but w_norm has {hc_dim} values")
    ms = np.mean(R * R, axis=1, keepdims=True)
    xn = (R / np.sqrt(ms + eps)).reshape(-1) * w_norm
    lo = silu((w_down @ xn) / hc)
    gated = xn * sigmoid(w_up @ lo)
    mixed = gated.reshape(hc, hidden).mean(axis=0)
    inject = _f(w_inj, dtype) @ xn if w_inj is not None else None
    return mixed, inject


def residual_write(R: np.ndarray, block_out: np.ndarray, inject: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """``R'[c] = R[c] + block_out * 2 * sigmoid(inject[c] / HC)`` for every stream ``c``.

    The same block output goes into every stream; only the weight is per stream,
    and ``inject = 0`` is a plain residual add. Source: ``gr.hpp`` (``gr_write``),
    ``gr.cu`` (``gr_write_kernel``).
    """
    R = _f(R, dtype)
    inject = _f(inject, dtype).reshape(-1)
    hc = inject.shape[0]
    shape = R.shape
    R2 = R.reshape(hc, -1)
    out = R2 + _f(block_out, dtype).reshape(1, -1) * (2.0 * sigmoid(inject / hc)).reshape(hc, 1)
    return out.reshape(shape)


# --------------------------------------------------------------------------- router


def route(
    x: np.ndarray,
    w_router: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    *,
    dtype=np.float64,
    return_probs: bool = False,
):
    """Top-``k`` experts of one token and their renormalised weights.

    ``logits = W_router @ x`` over all ``E`` experts (the router reads the
    full-width ``x``), ``p = softmax(logits)``, the ``k`` largest ``p`` in
    descending order with ties going to the lower expert id, then
    ``w = p_sel / max(sum(p_sel), 2**-14)``. Returns ``(ids int32 [k], w [k])``
    in rank order, plus ``p`` when ``return_probs``. (A NaN logit makes the
    engine's whole softmax row NaN, which it then maps to ``-FLT_MAX``; that
    degenerate case is not modelled beyond sorting NaN last.)

    Source: ``src/kernels/cuda/native_router.cu`` (the single-row and multi-row
    top-10 kernels). The engine compares float32 probabilities; two experts
    whose logits differ by less than float32 resolution can tie there and not
    here, which is a rounding-level difference.
    """
    x = _f(x, dtype)
    logits = _f(w_router, dtype) @ x
    m = np.max(logits)
    e = np.exp(logits - m)
    p = e / e.sum()
    key = np.where(np.isnan(p), -np.finfo(dtype).max, p)
    order = np.argsort(-key, kind="stable")  # stable: equal values keep ascending expert id
    ids = order[:top_k].astype(np.int32)
    p_sel = p[ids]
    w = p_sel / max(float(p_sel.sum()), ROUTER_WEIGHT_FLOOR)
    if return_probs:
        return ids, w, p
    return ids, w


# --------------------------------------------------------------------------- experts


def latent_expert(
    xl_g: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    *,
    s_gate: float = 1.0,
    s_up: float = 1.0,
    s_down: float = 1.0,
    dtype=np.float64,
) -> np.ndarray:
    """One latent expert on its group's latent vector: ``[L] -> [L]``.

    ``h = s_down * down @ (silu(s_gate * gate @ xl_g) * (s_up * up @ xl_g))``
    with ``gate``/``up`` ``[I, L]`` and ``down`` ``[L, I]``. Leave the scales at 1
    when the matrices are already scaled (``LatentExpert.scaled()``). Source:
    ``iq_kernels.cu`` grouped gate/up kernel (``s_gate``/``s_up`` on the FP32
    outputs) and down kernel (``s_down`` on its output).
    """
    xl_g = _f(xl_g, dtype)
    g = s_gate * (_f(gate, dtype) @ xl_g)
    u = s_up * (_f(up, dtype) @ xl_g)
    return s_down * (_f(down, dtype) @ (silu(g) * u))


@dataclass
class SharedExpertWeights:
    """The full-width shared expert: gate/up ``[I, H]``, down ``[H, I]``, scalar gate row ``[H]``."""

    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gate_inp: np.ndarray


def shared_expert(x: np.ndarray, gate, up=None, down=None, gate_inp=None, *, dtype=np.float64) -> np.ndarray:
    """``sigmoid(gate_inp . x) * down @ (silu(gate @ x) * (up @ x))``, full width, added to the routed sum.

    Accepts the four arrays or a :class:`SharedExpertWeights` as ``gate``.
    Source: ``include/strata/kernels/shared_expert.hpp``.
    """
    if isinstance(gate, SharedExpertWeights):
        gate, up, down, gate_inp = gate.gate, gate.up, gate.down, gate.gate_inp
    x = _f(x, dtype)
    h = silu(_f(gate, dtype) @ x) * (_f(up, dtype) @ x)
    g = sigmoid(_f(gate_inp, dtype).reshape(-1) @ x)
    return g * (_f(down, dtype) @ h)


def _expert(experts: ExpertSource, e: int) -> ExpertWeights:
    if callable(experts):
        got = experts(e)
    else:
        got = experts[e]
    if hasattr(got, "scaled"):
        got = got.scaled()
    return got


# --------------------------------------------------------------------------- the MoE block


@dataclass
class LatentMoEResult:
    """One token's MoE block: ``out = routed + shared``."""

    out: np.ndarray  # [H]
    ids: np.ndarray  # int32 [K], rank order
    weights: np.ndarray  # [K]
    routed: np.ndarray  # [H]
    shared: np.ndarray  # [H] (zeros when no shared expert was given)
    groups: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # int32 [K]: labels[ids]
    latent: list[np.ndarray] = field(default_factory=list)  # K x [L]: each route's h before U


def latent_moe(
    x: np.ndarray,
    *,
    dense_d: np.ndarray,
    dense_u: np.ndarray,
    labels: np.ndarray,
    experts: ExpertSource,
    router: np.ndarray,
    shared: SharedExpertWeights | Mapping[str, np.ndarray] | None,
    top_k: int = DEFAULT_TOP_K,
    combine: str = "per_route",
    dtype=np.float64,
) -> LatentMoEResult:
    """The latent MoE block for one token ``x [H]``.

    ``dense_d [G, L, H]``, ``dense_u [G, H, L]``, ``labels [E]`` (this layer's
    row of ``labels.bin``), ``experts`` (callable, mapping or sequence giving
    scaled ``(gate, up, down)`` for an expert id, or an object with
    ``.scaled()``), ``router [E, H]``, ``shared`` (or None to leave it out).

    ``combine="per_route"`` applies ``U[g_j]`` to every routed expert as the
    engine does (``latent.cu`` ``latent_u_k``: one U product per route);
    ``"per_group"`` sums the weighted latent outputs of each group first and
    applies each group's ``U`` once (at most ``G`` products), which is the same
    function by linearity and is what the MLX port uses.
    """
    if combine not in ("per_route", "per_group"):
        raise ValueError(f"combine must be 'per_route' or 'per_group', got {combine!r}")
    x = _f(x, dtype)
    U = dense_u
    labels = np.asarray(labels)
    ids, w = route(x, router, top_k, dtype=dtype)
    groups = labels[ids].astype(np.int32)
    needed = np.unique(groups)
    # the engine computes xl for all G groups; only the routed groups' rows are ever read
    xl = {int(g): _f(dense_d[g], dtype) @ x for g in needed}
    hs = [latent_expert(xl[int(g)], *_expert(experts, int(e)), dtype=dtype) for e, g in zip(ids, groups)]
    hidden = x.shape[0]
    routed = np.zeros(hidden, dtype=dtype)
    if combine == "per_route":
        for wj, gj, hj in zip(w, groups, hs):
            routed += wj * (_f(U[gj], dtype) @ hj)
    else:
        for g in needed:
            z = sum(wj * hj for wj, gj, hj in zip(w, groups, hs) if gj == g)
            routed += _f(U[g], dtype) @ z
    if shared is None:
        sh = np.zeros(hidden, dtype=dtype)
    elif isinstance(shared, SharedExpertWeights):
        sh = shared_expert(x, shared, dtype=dtype)
    else:
        sh = shared_expert(x, shared["gate"], shared["up"], shared["down"], shared["gate_inp"], dtype=dtype)
    return LatentMoEResult(out=routed + sh, ids=ids, weights=w, routed=routed, shared=sh, groups=groups, latent=hs)


def latent_moe_batch(X: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`latent_moe` over the rows of ``X [T, H]``: returns ``(out [T, H], ids [T, K], weights [T, K])``.

    Tokens are independent in the MoE block; this is a plain loop.
    """
    X = np.asarray(X)
    rows = [latent_moe(x, **kwargs) for x in X]
    return (
        np.stack([r.out for r in rows]),
        np.stack([r.ids for r in rows]),
        np.stack([r.weights for r in rows]),
    )


# --------------------------------------------------------------------------- one layer's MoE half


@dataclass
class HyperConnectionWeights:
    """One half's hyper-connection: ``norm [HC*H]`` (stored ``1 + w``), ``down [r, HC*H]``, ``up [HC*H, r]``, ``inject [HC, HC*H]``."""

    norm: np.ndarray
    down: np.ndarray
    up: np.ndarray
    inject: np.ndarray


@dataclass
class LayerWeights:
    """Everything one layer's MoE half reads. ``experts`` is lazy: ``experts(e)`` decodes one expert."""

    layer: int
    hc: HyperConnectionWeights
    router: np.ndarray  # [E, H]
    shared: SharedExpertWeights
    dense_d: np.ndarray  # [G, L, H]
    dense_u: np.ndarray  # [G, H, L]
    labels: np.ndarray  # int32 [E]
    experts: Callable[[int], ExpertWeights]


def moe_half(R: np.ndarray, weights: LayerWeights, *, top_k: int = DEFAULT_TOP_K, combine: str = "per_route", dtype=np.float64):
    """The MoE half of a layer on the residual stack ``R [HC, H]`` (the attention half's write already applied).

    Returns ``(R_new [HC, H], LatentMoEResult, x [H], inject [HC])``. In the
    engine the read is fused with the previous half's write
    (``verify.cpp`` ``gr_read_group(1, ...)``); the function is the same.
    """
    hcw = weights.hc
    x, inject = gr_read(R, hcw.norm, hcw.down, hcw.up, hcw.inject, dtype=dtype)
    res = latent_moe(
        x,
        dense_d=weights.dense_d,
        dense_u=weights.dense_u,
        labels=weights.labels,
        experts=weights.experts,
        router=weights.router,
        shared=weights.shared,
        top_k=top_k,
        combine=combine,
        dtype=dtype,
    )
    return residual_write(R, res.out, inject, dtype=dtype), res, x, inject


# --------------------------------------------------------------------------- adapter to the file readers


def layer_weights_from_pack(pack: Any, layer: int, *, dense_format: str | None = None) -> LayerWeights:
    """Pull one layer's MoE-half arrays from an :class:`mtplx.edge0.pack.Edge0Pack`.

    Skeleton names are llama.cpp's (``blk.N.ffn_gate_inp.weight`` and so on);
    ``GGUFFile.array`` returns numpy row-major shapes, so the router is
    ``[E, H]``, gate/up shexp ``[I, H]``, down shexp ``[H, I]``, the scalar
    gate ``[1, H]``, ``hc_ffn_down`` ``[r, HC*H]``, ``hc_ffn_up``
    ``[HC*H, r]``, ``hc_ffn_inject`` ``[HC, HC*H]``. ``dense_format`` picks
    ``"q4"`` or ``"q8"`` D/U (default: the model's, which is Q4_0 for the NVFP4
    build).
    """
    sk = pack.skeleton
    p = f"blk.{layer}."
    hc = HyperConnectionWeights(
        norm=sk.array(p + "hc_ffn_norm.weight").reshape(-1),
        down=sk.array(p + "hc_ffn_down.weight"),
        up=sk.array(p + "hc_ffn_up.weight"),
        inject=sk.array(p + "hc_ffn_inject.weight"),
    )
    shared = SharedExpertWeights(
        gate=sk.array(p + "ffn_gate_shexp.weight"),
        up=sk.array(p + "ffn_up_shexp.weight"),
        down=sk.array(p + "ffn_down_shexp.weight"),
        gate_inp=sk.array(p + "ffn_gate_inp_shexp.weight").reshape(-1),
    )

    def experts(e: int) -> ExpertWeights:
        return pack.expert(layer, e).scaled()

    return LayerWeights(
        layer=layer,
        hc=hc,
        router=sk.array(p + "ffn_gate_inp.weight"),
        shared=shared,
        dense_d=pack.dense_d(layer, dense_format),
        dense_u=pack.dense_u(layer, dense_format),
        labels=np.asarray(pack.labels[layer], dtype=np.int32),
        experts=experts,
    )
