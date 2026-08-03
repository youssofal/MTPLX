"""Standalone *alternative* Laguna-S-2.1 decode/prefill runtime.

This is the deliverable of the full mlx.fast Laguna XS2.1 → S-2.1 port (see
``PORT_LEDGER.md``): a second, independent implementation of the whole Laguna
forward that routes through the ported challenge kernels, so it can be
benchmarked head-to-head against MTPLX's reference lane
(``mtplx.laguna_compiled_step.LagunaCompiledLane`` + ``install_from_env``, the
67.4 tok/s path) on identical weights and identical shapes.

Design
------
The reference lane and this lane share the *weights* (the loaded
``mtplx.models.laguna.Model``) and the cache-state machinery (leaves, geometry,
ring arithmetic — all imported from ``laguna_compiled_step`` because that is
plumbing, not a kernel). What differs is the **forward**: every component of the
step is a *span* the config can replace with a ported challenge kernel. That
mirrors how the reference step already swaps ``fused_qk_norm_rope`` in for the
norm→transpose→rope chain — a contiguous span gated by a flag — and generalizes
it to the whole 27-kernel surface.

``AltConfig`` starts with **every kernel off**, so a freshly built alt lane runs
the exact stock spans and is digest-identical to a pure-stock reference forward
(proven on the toy model in ``tests`` / the CPU smoke script). As each kernel is
ported and passes its A/B, its flag is turned on and the span it replaces is
documented against its ``PORT_LEDGER.md`` id. Nothing is skipped as "already
covered": a kernel MTPLX happens to fuse a different way still gets ported here
and measured, because the comparison of the two whole runtimes is the point.

Scope so far: this scaffold implements the **decode** step (T=1, B=1, greedy) as
a faithful parallel of ``build_step`` with the swap surface wired but every span
stock. Prefill (steel flash-attention, prefill gather-GEMM) and the affine MoE
kernels land as their ledger phases are executed; their swap points are marked
below and raise ``NotImplementedError`` only when their flag is turned on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx

# Cache-state plumbing is shared with the reference lane verbatim: it is the
# S-2.1 KV/ring state machine, not a kernel under comparison. Re-deriving it here
# would only risk drift between the two lanes' state, which must be identical for
# an honest A/B.
from .laguna_compiled_step import (
    SLIDING,
    geometry_for,
    kv_plane_mask,
    kv_slot_write,
    next_ring_index,
    pack_kv,
    snapshot_leaves,
    unpack_kv,
)
from mlx_lm.models.base import create_attention_mask

from .kernels.laguna_decode import fused_qk_norm_rope, is_qk_norm_rope_eligible
from .kernels.laguna_prefill_moe_combine import (
    fused_moe_combine_prefill,
    is_moe_combine_prefill_eligible,
)
from .kernels.laguna_residual_router import fused_residual_norm_router
from .kernels.laguna_sdpa_pair import grouped_gqa_sdpa_decode
from .kernels.lm_head_topk import is_qmv8_topk_eligible, qmv8_lm_head_topk
from .models import laguna
from .models.laguna_fused import _router_normalize, _router_weights


# ---------------------------------------------------------------------------
# swap surface — one flag per PORT_LEDGER.md kernel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AltConfig:
    """Which ported challenge kernels this alt lane routes through.

    Every flag defaults to ``False`` == run the stock span, so the default alt
    lane reproduces the reference forward exactly. A flag is flipped on only
    after that kernel is ported AND has passed its correctness + A/B gate under
    the GPU lock. The names map 1:1 to ``PORT_LEDGER.md`` ids.
    """

    # -- decode --
    d1_residual_router: bool = False      # residual+RMSNorm+router-GEMV fusion
    d2_qk_yarn: bool = False              # qk-norm+YaRN rope (full family)
    d3_qk_rope_sliding: bool = False      # qk-norm+rope (sliding family)
    d4_input_qkvg: bool = False          # fused input-norm + QKV + gate proj
    d5_gated_oproj: bool = False         # softplus per-head gate × o_proj
    d6_sdpa_vector: bool = False         # decode SDPA vector (GQA-share)
    d7_shared_swiglu: bool = False       # shared-expert SwiGLU-QMV (affine)
    d8_routed_swiglu: bool = False       # routed SwiGLU-QMV (affine)
    d9_merged_swiglu: bool = False       # 9-slot merged SwiGLU-QMV (affine)
    d10_down_reduce: bool = False        # down+weighted-reduce+scale+add fusion
    d11_dense_layer0: bool = False       # dense layer-0 gate/up/down (affine)
    d12_router_topk: bool = False        # router sigmoid+bias+top-k
    d13_embed_rope_atlas: bool = False   # embedding + rope-atlas
    d14_lm_head_prune: bool = False      # lm_head certified prune (affine int8)
    # -- prefill (Phase 2) --
    p1_prefill_qk_rope: bool = False
    p2_steel_flash: bool = False
    p3_prefill_router_topk: bool = False
    p4_prefill_gather_gemm: bool = False
    p5_prefill_moe_tail: bool = False
    # Size gate: P5 engages only when the flattened token count (batch*length) is
    # >= this. The MoE-combine fusion LOSES at small prefill (the [M,top_k,hidden]
    # intermediate it removes is cheap there) and WINS above ~8k (measured
    # crossover: 0.95x @4k -> 1.04x @16k -> 1.09x @32k, digest-exact). None = no
    # gate (used to SWEEP the crossover); set to the crossover (e.g. 8192) to SHIP.
    prefill_min_tokens: Optional[int] = None

    def any_prefill(self) -> bool:
        return any(
            (
                self.p1_prefill_qk_rope,
                self.p2_steel_flash,
                self.p3_prefill_router_topk,
                self.p4_prefill_gather_gemm,
                self.p5_prefill_moe_tail,
            )
        )


STOCK = AltConfig()  # the all-off config: reproduces the reference forward


def _moe_from_precomputed(
    moe: Any,
    normed: mx.array,
    logits: "mx.array | None",
    residual: mx.array,
    config: "AltConfig" = STOCK,
) -> mx.array:
    """LagunaSparseMoeBlock forward (optionally from D1's precomputed logits).

    Mirrors ``laguna_fused._fused_moe_call`` op-for-op — reusing its own
    ``_router_weights`` / ``_router_normalize`` so the router selection is
    numerically identical. When ``logits`` is given (the D1 path) the ``moe.gate``
    GEMV is skipped, consuming the logits D1 already produced; when ``logits`` is
    ``None`` (the D1-free P5 path) the router logits are computed via ``moe.gate``
    exactly as the stock block does. The expert path (``switch_mlp``) and combine
    are the SAME objects the reference uses.

    Returns the post-MoE residual stream ``residual + moe(normed)`` (the residual
    add is folded in here so P5 can fuse it into the combine dispatch). With
    ``config.p5_prefill_moe_tail`` on and the size gate satisfied,
    ``fused_moe_combine_prefill`` replaces the weighted-reduce + routed_scaling +
    shared-add + residual-add tail with one dispatch; it is bit-exact with the
    stock combine (it consumes the UNSCALED normalized f32 weights and applies
    ``routed_scaling`` in-kernel, matching ``_router_normalize(...).astype``), so
    the fused and unfused branches are digest-identical. D1-free so P5 can be A/B'd
    without D1's prefill penalty (D1 loses at prefill: the router becomes a GEMM).
    """

    batch, length, hidden = normed.shape
    flattened = normed.reshape(-1, hidden)
    residual_flat = residual.reshape(-1, hidden)
    if logits is None:
        logits = moe.gate(flattened)
    logits = logits.reshape(-1, int(logits.shape[-1])).astype(mx.float32)
    if moe.softcap and moe.softcap > 0.0:
        logits = mx.tanh(logits / moe.softcap) * moe.softcap
    scores, scores_for_choice = _router_weights(
        logits, moe.e_score_correction_bias.astype(mx.float32)
    )
    indices = mx.argpartition(
        -scores_for_choice, kth=moe.top_k - 1, axis=-1
    )[..., : moe.top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    expert_out = moe.switch_mlp(flattened, indices)
    shared = moe.shared_expert(flattened)

    # LEDGER P5 — prefill MoE combine tail. Fuse weighted-reduce + routed_scaling
    # + shared-add + residual-add into one dispatch. Only for the normalized-prob
    # path (S-2.1), under the size gate, and when the shapes are covered.
    m_tokens = batch * length
    size_ok = (
        config.prefill_min_tokens is None or m_tokens >= config.prefill_min_tokens
    )
    if (
        config.p5_prefill_moe_tail
        and size_ok
        and moe.norm_topk_prob
        and is_moe_combine_prefill_eligible(
            expert_out,
            (weights / weights.sum(axis=-1, keepdims=True)),
            shared,
            residual_flat,
        )
    ):
        norm_weights = weights / weights.sum(axis=-1, keepdims=True)  # unscaled, f32
        combined = fused_moe_combine_prefill(
            expert_out,
            norm_weights,
            shared,
            residual_flat,
            float(moe.routed_scaling_factor),
        )
        return combined.reshape(batch, length, hidden)

    if moe.norm_topk_prob:
        weights = _router_normalize(
            weights, mx.array(moe.routed_scaling_factor, dtype=mx.float32)
        ).astype(normed.dtype)
    else:
        weights = (weights * moe.routed_scaling_factor).astype(normed.dtype)
    output = laguna.MOE_COMBINE_IMPL(expert_out, weights, shared)
    return residual + output.reshape(batch, length, hidden)


def _is_sparse_moe(mlp: Any) -> bool:
    """A routed MoE block (has a router gate + expert bank); layer-0 dense is not."""

    return hasattr(mlp, "gate") and hasattr(mlp, "switch_mlp")


# ---------------------------------------------------------------------------
# alt PREFILL forward (LEDGER Phase 2)
# ---------------------------------------------------------------------------
def alt_prefill_forward(
    model: Any,
    inputs: mx.array,
    cache: Sequence[Any],
    *,
    config: AltConfig = STOCK,
) -> mx.array:
    """The prefill forward, routed through the ported kernels where they apply.

    Mirrors the eager ``LagunaModel.__call__`` op-for-op (same masks, same
    per-layer residual stream, stock attention which itself uses the installed
    qk-rope/attn-gate kernels), but for the sparse layers replaces the
    post-attention residual-add + RMSNorm + router-GEMV trio with D1's fused
    kernel when ``config.d1_residual_router`` is on — the one ported kernel that
    is prefill-applicable.     Attention stays on ``mx.fast.scaled_dot_product_attention``
    (MLX's flash path, which beat the hand SDPA at decode) and the experts stay on
    stock ``SwitchGLU`` (whose sorted grouped-GEMM the affine hand kernel could not
    beat at any token count). Returns the final-norm hidden ``[B, T, H]``; the
    caller applies the head. Correct-by-construction: with ``STOCK`` it is the
    eager forward exactly.
    """

    # Anti-fake-win contract (same as the decode lane): an enabled flag whose
    # kernel is not wired here must fail loudly, never silently run stock —
    # otherwise an A/B "win" can be measured against a no-op arm.
    unwired_prefill = [
        name
        for name, enabled in (
            ("p1_prefill_qk_rope", config.p1_prefill_qk_rope),
            ("p2_steel_flash", config.p2_steel_flash),
            ("p3_prefill_router_topk", config.p3_prefill_router_topk),
            ("p4_prefill_gather_gemm", config.p4_prefill_gather_gemm),
        )
        if enabled
    ]
    if unwired_prefill:
        raise NotImplementedError(
            "alt_prefill_forward has no wired kernel for enabled flag(s): "
            + ", ".join(unwired_prefill)
        )

    inner = getattr(model, "model", model)
    hidden = inner.embed_tokens(inputs)

    full_mask = create_attention_mask(hidden, cache[inner._first_full])
    if inner._has_swa:
        sliding_mask = create_attention_mask(
            hidden, cache[inner._first_swa], window_size=model.args.sliding_window
        )
    else:
        sliding_mask = full_mask

    rope_memo: dict[int, mx.array] = {}
    for layer, layer_cache in zip(inner.layers, cache):
        mask = sliding_mask if layer.self_attn.is_sliding else full_mask
        attention_out = layer.self_attn(
            layer.input_layernorm(hidden), mask, layer_cache, rope_memo
        )
        moe = layer.mlp
        if config.d1_residual_router and _is_sparse_moe(moe):
            post_ln = layer.post_attention_layernorm
            hidden, normed, logits = fused_residual_norm_router(
                attention_out,
                hidden,
                post_ln.weight,
                moe.gate.weight,
                float(post_ln.eps),
            )
            hidden = _moe_from_precomputed(moe, normed, logits, hidden, config)
        elif config.p5_prefill_moe_tail and _is_sparse_moe(moe):
            # D1-free P5 path: stock attention residual + stock router (moe.gate),
            # but the MoE combine/shared/residual tail fused by P5 (logits=None ->
            # computed via moe.gate). Lets P5 be measured without D1's prefill hit.
            hidden = hidden + attention_out
            normed = layer.post_attention_layernorm(hidden)
            hidden = _moe_from_precomputed(moe, normed, None, hidden, config)
        else:
            hidden = hidden + attention_out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    return inner.norm(hidden)


# ---------------------------------------------------------------------------
# the alt decode step
# ---------------------------------------------------------------------------
def build_alt_step(
    model: Any,
    cap: int,
    *,
    config: AltConfig = STOCK,
    compiled: bool = True,
    packed_kv: bool = False,
) -> Callable[..., tuple[mx.array, ...]]:
    """Build the alt decode step for ``model`` under ``config``.

    Signature matches the reference exactly so the two lanes are drop-in
    swappable in the harness::

        step(token, offset, ring_idx, *leaves)
            -> (next_token, offset_next, ring_idx_next, *leaves_next)

    Each ``# LEDGER Dxx`` marker is a swap point: when ``config`` turns that
    kernel on, the ported impl replaces the stock span directly beneath it. With
    the default ``STOCK`` config every span is the reference op, so the built
    step is digest-identical to :func:`mtplx.laguna_compiled_step.build_step`.
    """

    geometry = geometry_for(model, cap, packed_kv=packed_kv)
    inner = getattr(model, "model", model)
    layers = list(inner.layers)
    window = geometry.window
    packed = geometry.packed_kv
    tied = bool(model.args.tie_word_embeddings)

    positions = mx.arange(geometry.cap, dtype=mx.int32)
    admit = mx.array(0.0, dtype=mx.float32)
    reject = mx.array(-float("inf"), dtype=mx.float32)
    plane = kv_plane_mask()
    mx.eval(positions, admit, reject, plane)

    def _attention(
        attn: Any,
        x: mx.array,
        offset: mx.array,
        start: mx.array,
        kv_state: tuple[mx.array, ...],
        mask_for: Callable[[Any], mx.array] | None,
        gate_impl: Callable[..., mx.array],
        sliding: bool,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        batch, length, _ = x.shape

        # LEDGER D4 — fused QKV(+gate) projection: one GEMM for q/k/v/g instead of
        # four. `install_fused_qkvg` leaves `_qkvg` on the module; the flag GATES
        # its use so the fusion is isolated in the A/B even when installed. Off (or
        # no installer) runs the four separate projections. (The challenge also
        # folds input_layernorm into this dispatch — a further refinement, TODO.)
        qkvg = getattr(attn, "_qkvg", None)
        if config.d4_input_qkvg and qkvg is not None:
            queries, keys, values, gate_logits = qkvg(x)
        else:
            queries, keys, values = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
            gate_logits = None

        values = values.reshape(batch, length, attn.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        # LEDGER D2/D3 — qk-norm + rope (full YaRN / sliding). The ported kernels
        # replace this whole norm→transpose→rope span with one dispatch per
        # family; until then the reference's own fused_qk_norm_rope (when its
        # spec is installed) or the stock chain runs.
        if config.d2_qk_yarn or config.d3_qk_rope_sliding:
            raise NotImplementedError("D2/D3 qk-norm+rope port not yet wired")
        spec = getattr(attn, "_qk_rope_spec", None)
        if is_qk_norm_rope_eligible(
            queries, keys, attn.q_norm.weight, attn.k_norm.weight, spec
        ):
            queries, keys = fused_qk_norm_rope(
                queries,
                keys,
                attn.q_norm.weight,
                attn.k_norm.weight,
                float(attn.q_norm.eps),
                offset,
                spec,
            )
        else:
            queries = attn.q_norm(
                queries.reshape(batch, length, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            keys = attn.k_norm(
                keys.reshape(batch, length, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            queries = attn.rope(queries, offset=offset)
            keys = attn.rope(keys, offset=offset)

        if packed:
            (kv_leaf,) = kv_state
            kv_leaf = kv_slot_write(kv_leaf, pack_kv(keys, values, plane), start)
            keys, values = unpack_kv(kv_leaf)
            updated: tuple[mx.array, ...] = (kv_leaf,)
        else:
            k_leaf, v_leaf = kv_state
            keys = kv_slot_write(k_leaf, keys, start)
            values = kv_slot_write(v_leaf, values, start)
            updated = (keys, values)

        # LEDGER D6 — decode SDPA vector, group-3 GQA KV-reuse (full gqa 6 /
        # sliding gqa 9; 3 divides both). Eligible only when there is no mask —
        # the sliding steady-state layers here; full layers keep their padded-leaf
        # additive mask and fall back to stock SDPA (the kernel has no mask path).
        sdpa_mask = None if mask_for is None else mask_for(queries.dtype)
        output = None
        if config.d6_sdpa_vector:
            output = grouped_gqa_sdpa_decode(
                queries, keys, values, scale=attn.scale, mask=sdpa_mask
            )
        if output is None:
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask=sdpa_mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, length, attn.n_heads * attn.head_dim
        )

        # LEDGER D5 — gated output projection (softplus per-head gate × o_proj).
        if config.d5_gated_oproj:
            raise NotImplementedError("D5 gated o_proj port not yet wired")
        if attn.gating:
            if gate_logits is None:
                gate_logits = attn.g_proj(x)
            if attn.gate_per_head:
                output = gate_impl(output, gate_logits, attn.n_heads, attn.head_dim)
            else:
                gate = mx.logaddexp(
                    gate_logits.astype(mx.float32), mx.array(0.0)
                ).astype(output.dtype)
                output = output * gate

        return attn.o_proj(output), updated

    def _mlp(layer: Any, hidden: mx.array) -> mx.array:
        """Post-attention norm + MLP/MoE, the residual add left to the caller.

        The ported MoE affine kernels (D1 router fusion, D7-D12) and the dense
        layer-0 kernel (D11) replace spans inside here.
        """

        # D1 (residual+RMSNorm+router-GEMV fusion) spans the residual add and so
        # is handled at the loop level, not here; this is the stock (or D1-off)
        # post-attention norm.
        normed = layer.post_attention_layernorm(hidden)

        # LEDGER D7/D8/D9/D10/D11/D12 — affine MoE SwiGLU-QMV + router top-k +
        # down/reduce fusion + dense layer-0. Until ported, the stock module MoE
        # (SwitchGLU + MOE_COMBINE_IMPL) / dense MLP runs.
        if any(
            (
                config.d7_shared_swiglu,
                config.d8_routed_swiglu,
                config.d9_merged_swiglu,
                config.d10_down_reduce,
                config.d11_dense_layer0,
                config.d12_router_topk,
            )
        ):
            raise NotImplementedError("affine MoE kernels (D7-D12) not yet wired")
        return layer.mlp(normed)

    def step(
        token: mx.array,
        offset: mx.array,
        ring_idx: mx.array,
        *leaves: mx.array,
    ) -> tuple[mx.array, ...]:
        if len(leaves) != geometry.n_leaves:
            raise ValueError(f"expected {geometry.n_leaves} leaves, got {len(leaves)}")

        gate_impl = laguna.PER_HEAD_GATE_IMPL

        full_start = mx.reshape(offset, (1,))
        ring_start = mx.reshape(ring_idx, (1,))

        admitted = positions < (offset + 1)
        masks: dict[Any, mx.array] = {}

        def mask_for(dtype: Any) -> mx.array:
            cached = masks.get(dtype)
            if cached is None:
                cached = (
                    mx.where(admitted, admit, reject)
                    .astype(dtype)
                    .reshape(1, 1, 1, geometry.cap)
                )
                masks[dtype] = cached
            return cached

        # LEDGER D13 — embedding + rope-atlas. Stock embedding until ported.
        if config.d13_embed_rope_atlas:
            raise NotImplementedError("D13 embed+rope-atlas not yet wired")
        hidden = inner.embed_tokens(token)

        updated: list[mx.array] = []
        for index, layer in enumerate(layers):
            attn = layer.self_attn
            sliding = geometry.kinds[index] == SLIDING
            attention_out, layer_updated = _attention(
                attn,
                layer.input_layernorm(hidden),
                offset,
                ring_start if sliding else full_start,
                geometry.layer_leaves(leaves, index),
                None if sliding else mask_for,
                gate_impl,
                sliding,
            )
            updated.extend(layer_updated)

            # LEDGER D1 — fused residual-add + post-attn RMSNorm + router GEMV,
            # one dispatch across the residual boundary for the 47 sparse layers.
            # Ineligible shapes (layer-0 dense, non-Metal, wrong axis/experts)
            # fall back inside the kernel to the stock add+norm+matmul, so this
            # branch is correct on any model; the metal kernel only fires at the
            # exact S-2.1 routing shape.
            moe = layer.mlp
            if config.d1_residual_router and _is_sparse_moe(moe):
                post_ln = layer.post_attention_layernorm
                hidden, normed, logits = fused_residual_norm_router(
                    attention_out,
                    hidden,
                    post_ln.weight,
                    moe.gate.weight,
                    float(post_ln.eps),
                )
                # _moe_from_precomputed now folds the residual add (so P5 can fuse
                # it at prefill); at decode T=1 the P5 size gate never engages, so
                # this is the same residual + combine as before.
                hidden = _moe_from_precomputed(moe, normed, logits, hidden, config)
            else:
                hidden = hidden + attention_out
                hidden = hidden + _mlp(layer, hidden)

        output = inner.norm(hidden)

        # LEDGER D14 — lm_head top-1 straight from the 8-bit affine head, one
        # dispatch, no 100352-wide logits materialized + separate argmax. EXACT
        # only if its top-1 equals the stock argmax — the A/B digest is the gate;
        # falls back to the stock head on any ineligible shape (incl. the tied /
        # non-quantized toy head).
        if (
            config.d14_lm_head_prune
            and not tied
            and is_qmv8_topk_eligible(
                output.reshape(-1, output.shape[-1]), model.lm_head, top_k=1
            )
        ):
            # qmv8_lm_head_topk returns a 1-D indices array for the single
            # decode row / top_k=1; the one element is the argmax token.
            _values, indices = qmv8_lm_head_topk(
                output.reshape(-1, output.shape[-1]), model.lm_head, top_k=1
            )
            next_token = indices.reshape(1, 1).astype(mx.uint32)
        else:
            logits = (
                inner.embed_tokens.as_linear(output) if tied else model.lm_head(output)
            )
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]

        ring_next = (
            next_ring_index(ring_idx, window) if window is not None else ring_idx
        )
        return (next_token, offset + 1, ring_next, *updated)

    return mx.compile(step) if compiled else step


# ---------------------------------------------------------------------------
# async-eval decode ladder (LEDGER S1) — the biggest single challenge win
# ---------------------------------------------------------------------------
# The XS2.1 runtime stages async evaluation at token indices 1,7,15,23,31,39
# rather than syncing once per token, measured there as +9.83%. Expressed here
# as a schedule the driver consults; the reference driver syncs every step
# (LADDER_EVERY_STEP), and the ladder is the alternative to A/B against it.
LADDER_EVERY_STEP: tuple[int, ...] = ()
LADDER_XS21: tuple[int, ...] = (1, 7, 15, 23, 31, 39)


# ---------------------------------------------------------------------------
# public lane
# ---------------------------------------------------------------------------
class LagunaAltLane:
    """Drives :func:`build_alt_step` — the alt-runtime twin of LagunaCompiledLane.

    Holds the same tensor state and the same seed/advance contract, so the
    harness can run either lane through one code path. The only additions are
    ``config`` (which ported kernels are live) and ``generate`` (the async-eval
    ladder driver, LEDGER S1).
    """

    def __init__(
        self,
        model: Any,
        cap: int,
        *,
        config: AltConfig = STOCK,
        compiled: bool = True,
        packed_kv: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.compiled = bool(compiled)
        self.packed_kv = bool(packed_kv)
        self.geometry = geometry_for(model, cap, packed_kv=packed_kv)
        self.step = build_alt_step(
            model, cap, config=config, compiled=compiled, packed_kv=packed_kv
        )
        self.token: mx.array | None = None
        self.offset: mx.array | None = None
        self.ring_idx: mx.array | None = None
        self.leaves: tuple[mx.array, ...] = ()
        self._position = 0

    @property
    def cap(self) -> int:
        return self.geometry.cap

    def seed(self, caches: Sequence[Any], token: mx.array) -> "LagunaAltLane":
        self.offset, self.ring_idx, self.leaves = snapshot_leaves(
            self.model, caches, self.geometry.cap, packed_kv=self.packed_kv
        )
        self._position = int(self.offset)
        self.token = mx.array(token, dtype=mx.uint32).reshape(1, 1)
        return self

    def advance(self) -> mx.array:
        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        if self._position >= self.geometry.cap:
            raise ValueError(
                f"the leaves are full at cap {self.geometry.cap}; re-seed with a "
                "larger cap"
            )
        outputs = self.step(self.token, self.offset, self.ring_idx, *self.leaves)
        self.token, self.offset, self.ring_idx = outputs[0], outputs[1], outputs[2]
        self.leaves = tuple(outputs[3:])
        self._position += 1
        return self.token

    def generate(
        self, n: int, *, ladder: tuple[int, ...] = LADDER_EVERY_STEP
    ) -> list[int]:
        """Decode ``n`` tokens, staging ``mx.async_eval`` per the ladder schedule.

        ``ladder`` is the set of step indices (0-based within this call) at which
        the running token is handed to :func:`mx.async_eval` so the host can race
        ahead building the next graph while the GPU finishes the current one. The
        empty schedule (:data:`LADDER_EVERY_STEP`) evaluates every step — the
        reference behaviour. :data:`LADDER_XS21` is the challenge's staging.

        Correctness is independent of the schedule: the tokens produced are
        identical either way (async_eval only changes *when* the host blocks, not
        the values). The schedule is a pure latency lever, which is exactly why
        it can be A/B'd token-for-token.
        """

        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        ladder_set = set(ladder)
        tokens: list[mx.array] = []
        for i in range(n):
            tok = self.advance()
            tokens.append(tok)
            if not ladder_set or i in ladder_set:
                mx.async_eval(tok)
        mx.eval(tokens[-1] if tokens else self.token)
        return [int(t.item()) for t in tokens]

    def remaining_steps(self) -> int:
        if self.token is None:
            raise ValueError("lane has no state; call seed() after a prefill")
        return self.geometry.cap - self._position

    def state(self) -> tuple[mx.array, mx.array, tuple[mx.array, ...]]:
        return self.offset, self.ring_idx, self.leaves
