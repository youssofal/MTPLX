"""Native MLX loader/backend for DeepSeek-V4-Flash (``model_type: deepseek_v4``).

This is a from-scratch port, not a config tweak: no ``deepseek_v4`` loader exists
in mlx-lm.  The scaffold reuses the DeepSeek-V3/V3.2 MoE + shared-expert shape and
the ``noaux_tc`` routing idea, but V4 adds four pieces of genuinely new math over
V3.2, all transcribed here from the authoritative reference
``deepseek-ai/DeepSeek-V4-Flash/inference/model.py`` + ``inference/kernel.py``:

1. **Hyper-Connections (HCA)** — the residual stream is replaced by ``hc_mult=4``
   parallel copies.  Each block runs ``hc_pre`` (collapse 4->1 via Sinkhorn-derived
   pre-weights) around attn/ffn, then ``hc_post`` (expand 1->4 and re-mix with the
   residual copies through a doubly-stochastic ``comb`` matrix).  The ``comb`` matrix
   is produced by ``hc_split_sinkhorn``: one row-softmax + one column-normalise, then
   ``hc_sinkhorn_iters-1`` (=19) further row/column normalisation passes.  ``hc_eps``
   (1e-6) guards every division.  Reference: ``Block.hc_pre/hc_post`` (model.py
   L673-699) and ``hc_split_sinkhorn_kernel`` (kernel.py L371-427).

2. **Compressed Sparse Attention (CSA)** — for layers with ``compress_ratio != 0``,
   a ``Compressor`` builds a second, compressed KV cache by learned gated pooling of
   ``compress_ratio`` consecutive tokens (softmax over a learned gate + absolute
   position embedding ``ape``).  ``compress_ratio==4`` layers pool overlapping
   windows and own an ``Indexer`` that scores compressed positions and returns the
   top-``index_topk`` (512) to attend; ``compress_ratio==128`` layers pool
   non-overlapping windows and use a deterministic strided index.  These layers rope
   with ``compress_rope_theta`` (160000) under YaRN; ``compress_ratio==0`` layers are
   pure sliding-window (``window_size=128``) with base ``rope_theta`` and no YaRN.
   Reference: ``Compressor`` (L279-377), ``Indexer`` (L380-433), ``Attention`` (L436-543).

3. **Output-LoRA (o-LoRA)** — where V3 low-ranks only q and kv, V4 also low-ranks the
   output projection *in groups*.  The ``n_heads*head_dim = 32768`` attention output
   is split into ``o_groups=8`` chunks of 4096; each chunk is projected to
   ``o_lora_rank=1024`` by its own matrix (grouped/block matmul ``wo_a``), the 8
   results concatenate to 8192, then ``wo_b`` maps 8192->dim.  Reference:
   ``Attention.forward`` L536-542.

4. **Hash layers** — the first ``num_hash_layers=3`` layers route each token to a
   fixed expert set determined by token id (``gate.tid2eid`` lookup) instead of
   score-based top-k.  Reference: ``Gate`` (L546-584).

Attention itself is MQA-shaped MLA: ``num_key_value_heads=1``, a single 512-dim KV
latent (``head_dim=512``, ``rope_head_dim=64`` on its tail) shared across all 64
query heads, each head carrying a learned ``attn_sink`` logit.  Routing uses
``scoring_func="sqrtsoftplus"`` (softplus then sqrt) with ``routed_scaling_factor=1.5``.

Weight names mirror the reference module tree, which is exactly what the
``mlx-community/DeepSeek-V4-Flash-4bit`` checkpoint ships:
``model.layers.{i}.attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink}``,
``...attn.compressor.{wkv,wgate,norm,ape}``, ``...attn.indexer.{wq_b,weights_proj,compressor.*}``,
``...ffn.{gate,switch_mlp,shared_experts}``, ``...{attn_hc,ffn_hc}.{fn,base,scale}``,
``model.hc_head.{fn,base,scale}``, ``model.{embed_tokens,norm}``, ``lm_head``.
Quantisation in that checkpoint is mixed: routed experts (``ffn.switch_mlp.*``) are
**mxfp4 group_size 32** (scales, no biases); everything else is **affine 4-bit
group_size 64** (weight/scales/biases).  The MTP block is dropped by the conversion
— see :class:`DeepseekV4MTP` and ``scripts/deepseek_v4_build_mtp_model.py``, which
restores it from the upstream FP8/FP4 checkpoint into a merged model directory.

Status:
  * The four new-math components are numerically gated against the reference
    (tests/test_deepseek_v4_new_math.py) and the WHOLE forward is gated layer-by-layer
    against a reference golden covering every layer type
    (tests/test_deepseek_v4_parity.py) — this is the prefill path.
  * The attention integrates the compressor's compressed KV (overlap ratio-4 and
    non-overlap ratio-128), the window+compressed causal mask, compress-YaRN rope and
    per-head attn_sink; it is the dense-mask equivalent of the reference sparse_attn
    + topk_idxs gather.
  * The ratio-4 :class:`Indexer` top-k filter is wired in both directions
    (tests/test_deepseek_v4_indexer.py): it scores every compressed row against the
    query and masks all but the top ``index_topk``, so the backend is correct past
    ~``index_topk*ratio`` tokens of context, where dense-over-compressed stops being
    the reference's computation.  Below that threshold the filter provably selects
    every causal row, and the scoring path is skipped outright, leaving the short
    regime bit-identical.
  * Streaming decode runs off ``DeepseekV4Cache`` (``make_cache``): a sliding-window
    per-position KV buffer, the growing compressed-KV rows, the compressor's
    in-progress window frontier, and the same pair again for the indexer's own
    compressor lane.  Prompt-prefill + token-by-token decode reproduces the one-shot
    forward (tests/test_deepseek_v4_decode.py), including partial prompt windows,
    both compress ratios, context past ``window_size``, and crossing ``index_topk``
    mid-generation.  The state machine is adapted from ds4.c (antirez/DwarfStar4,
    MIT), which carries ``index_state_kv``/``index_comp_kv`` beside the attention
    lane's for exactly this reason.
  * That cache is **rewindable** (``DeepseekV4Cache.trim``), which is what the
    speculative lane needs on a rejected draft: emitted compressed rows truncate,
    both compressor frontiers rebuild from a bounded journal of their own projected
    rows, and the sliding window retains ``rollback_capacity`` extra rows because
    eviction cannot be undone.  Exactness is gated bit-for-bit against a
    never-speculated arm, on every lane and across every boundary that can break a
    rewind (tests/test_deepseek_v4_spec.py), with four rollback mutations caught.
    Making it exact is also what lets the *engine's* generic all-trimmable
    rejection repair serve this backend
    (``mtplx.cache_state.trim_verified_window_without_snapshot``) instead of a
    bespoke snapshot/restore path.
  * The generic route retains the historical unquantized compressor/indexer oracle.
    The installed Mia route instead owns attention rows in native 432-byte NVFP4
    pages and its indexer rows in the image's 132-byte E4M3 compatibility layout;
    both hot kernels consume physical pages through installed block tables.
  * The MTP draft block (:class:`DeepseekV4MTP`) is implemented and gated against a
    NumPy transcription of the reference ``MTPBlock`` (tests/test_deepseek_v4_mtp.py,
    max_rel ~2e-7 at a shrunk config; nine implementation mutations all caught).
    It binds through the ordinary load path from a checkpoint that ships ``mtp.0.*``
    — no sidecar, no env var — and :meth:`Model.sanitize` drops it from the tree
    when the weights are absent, which is the published mlx-community case and
    keeps the runtime's degrade-to-autoregressive branch reachable unchanged.
  * The speculative lane is wired: :class:`Model` carries the uniform runtime draft
    surface (``__call__(return_hidden=...)``, :meth:`Model.mtp_forward`,
    :meth:`Model.mtp_update_cache`, :meth:`Model.make_mtp_cache`) and
    :func:`inject_deepseek_v4_mtp_support` publishes it, so ``mtplx.generation``
    drives draft/verify/accept/reject/rollback here exactly as it does for every
    other native MTP backend — no parallel loop.  Greedy speculative decode at K =
    1, 2, 3 emits the identical committed sequence as pure AR through the real
    engine (tests/test_deepseek_v4_spec.py); acceptance counters are the engine's
    and come with it.  Not owned here: draft/verify are batch-shaped forwards, so
    the committed row's KV is projected inside a K+1-wide GEMM rather than alone —
    the invariance is committed-sequence exactness, not bitwise-identical logits.
  * The ``swiglu_limit`` clamp (10.0 in the shipped config) is applied in every
    expert, routed and shared, as the reference does (``Expert.forward``, model.py
    L600-602, handed the limit at L624/L627).  The shared expert carries it in
    :class:`DeepseekV4MLP`; the routed experts get it from :class:`ClampedSwiGLU`
    plugged into ``SwitchGLU``'s ``activation`` seam, so the batched expert kernels
    are untouched and one constructor covers trunk, hash and MTP layers alike.
    The clamp is asymmetric — ``up`` clipped to ``[-limit, +limit]``, ``gate`` cut
    only at ``+limit`` — and is gated against a NumPy oracle with the branches
    driven into saturation, with the branch-flip and clamp-removal mutations
    caught (tests/test_deepseek_v4_swiglu_clamp.py).  At ``swiglu_limit=0`` the
    routed path defers to the stock fused ``swiglu``, bit-identically, which is
    where both parity goldens were captured.
    Not yet measured: the activation ranges real V4-Flash weights actually reach,
    i.e. how often the clamp binds in practice.  That needs a checkpoint load and
    is deferred to a GPU window.
  * ``deepseek-v4`` is registered in ``mtplx/backends/registry.py`` so ``mtplx serve``
    resolves the load path.  That arch_id is what BOTH the AR-only mlx-community
    conversions and an MTP-bearing merged directory detect as; the separate
    ``deepseek-v4-mtp`` entry describes vLLM's *split* checkpoint layout, which is a
    different artifact shape MTPLX still has no loader for.

Decode-path bytes (tests/test_deepseek_v4_o_lora.py, tests/test_deepseek_v4_dtypes.py):
  * **o-LoRA weight handling.**  ``wo_a`` is static — ``[8192, 4096]`` on
    DeepSeek-V4-Flash — and the first cut ran ``mx.dequantize`` on it inside every
    ``_o_lora`` call, i.e. 64 MiB of dense bytes written and re-read per layer per
    decoded token, 43 layers deep.  It is now dequantised once and kept
    (``MTPLX_DSV4_O_LORA=cached``, the default, bit-identical to the old path and
    gated as such), which is what the reference does — it holds ``wo_a`` dense and
    just ``view``\\s it (model.py L537).  ``dequant`` restores the per-call
    behaviour as an A/B control; ``gather_qmm`` skips the dense tensor entirely and
    runs the 8 LoRA groups as one quantised block-diagonal matmul — the
    optimisation the reference explicitly leaves on the table (L538-539) — and is
    off by default because it is not bit-identical.  Note the measured risk: the
    grouped kernel tracks the dense einsum to ~1e-6 against fp32 activations but
    loses two orders of magnitude against bf16 ones on the CPU kernel, so its
    accuracy has to be re-measured on Metal before it can be defaulted on.
    What each is worth: ``cached`` vs ``dequant`` on the real checkpoint measured
    +2.1% AR (4.534 -> 4.627 tok/s) with fp32 activation storage, which is inside
    this box's cross-window drift — i.e. not distinguishable from zero, because at
    fp32 the einsum promotes ``wo_a`` anyway and caching removes the dequantize
    but not the cast that followed it.  It is kept because it is bit-identical and
    removes real redundant work, not because it is the speed win; the speed win is
    the activation-dtype fix below.  ``cached`` costs +2.69 GiB resident, and
    ``gather_qmm`` gives that back in full: on Metal at bf16 it measured AR 16.146
    vs 15.954 tok/s (+1.2%), K=3 26.762 vs 25.856 (+3.5%), peak 94.31 vs 96.97
    GiB.  A strictly better speed/memory point whose *quality* is unproven — one
    256-token prompt showed no visible damage, which is not a quality result — so
    it stays env-gated pending its own task eval.
    (bench/deepseek-v4/goal-ab-20260731, configs B/D/A/C.)
  * **Activation dtype.**  The reference keeps the whole attention lane at the model
    dtype and uses fp32 only as arithmetic: ``apply_rotary_emb`` rotates in fp32 and
    copies back into the caller's bf16 tensor (L234/L243), the compressor pools in
    fp32 but casts the row back before the norm (L362, and ``rotate_activation``
    then asserts bf16 at L249), and ``sparse_attn`` is declared ``q/kv/o: BF16``
    with fp32 accumulator fragments, casting the probability block to BF16 before
    the PV gemm (kernel.py L295-297, L305, L340).  This backend stored all three in
    fp32; since ``mx.concatenate`` and ``mx.matmul`` promote, that pulled the KV
    cache, both attention matmuls, the o-LoRA einsum (which then had to upcast
    ``wo_a`` too) and the entire residual stream up to fp32 on *every* layer.  The
    three storage points now follow the reference.  This is a no-op at fp32 — where
    both parity goldens and the decode oracle were captured — so no golden and no
    tolerance moved; ``MTPLX_DSV4_FP32_ACTIVATIONS=1`` restores the promoting path
    as the A/B arm.
    This is where the decode speed in this lane comes from: on the real 2bit-DQ
    checkpoint, in one window, AR 4.534 -> 15.954 tok/s (3.52x) and K=3
    speculative 9.530 -> 25.856 tok/s, with the o-LoRA arm held fixed
    (bench/deepseek-v4/goal-ab-20260731, configs B/D/A).  It is also what costs
    spec==AR byte-identity: at fp32 the precision headroom absorbed the
    batch-width-dependent rounding of a verify-shaped forward, at bf16 it reaches
    the argmax on near-tied tokens.  Speed and the byte gate are not separable
    here; see :mod:`scripts.deepseek_v4_mtpk_bench` for how divergence is
    reported, and the quality evidence is a task eval, not a byte compare.

Dispatch structure (tests/test_deepseek_v4_kernel_paths.py, scripts/deepseek_v4_dispatch_census.py):
  The measured decode cycle is 84.8 ms fixed + 8.9 ms/K with the target forward
  71-81% of it, so what is left to win is the *number* of kernels the host
  encodes per token, not bytes.  ``scripts/deepseek_v4_dispatch_census.py``
  counts them off the Metal dispatch stream itself (the instrumented MLX build in
  ``mlx-profiler``), differencing a 9-step run against a 1-step one so load,
  prefill and compile tracing cancel.  At DeepSeek-V4-Flash's *structure* (43
  layers, hc_mult 4, 20 Sinkhorn iterations, shrunk widths) one bf16 ``s == 1``
  decode step was **19,809 kernel dispatches in 384 command buffers**, and the
  ``cb`` rows put **host encode at 58.8 ms against 34.8 ms of GPU execution** —
  i.e. the encode is not hidden behind the GPU, it *is* the cycle.  ~2.9 us of
  host encode per dispatch, whatever the tensor size.

  * **Hyper-Connections** — the lever.  ``pre`` runs ``2 * n_layers + 1`` times
    per token and almost all of it is 4x4 tensors.  Three changes, all
    bit-identical at the decode shape: the fp32 casts of ``fn``/``base``/``scale``
    are derived once instead of per call (:meth:`HyperConnection._static` — ``fn``
    is 24 x 16384 on the real model); the three affine transforms become one
    ``mixes * scale_vec + base`` over the whole row; and the whole function is a
    module-level pure function of arrays so ``mx.compile`` can hold **one** tape
    for all 87 Hyper-Connection modules.  That collapses the Sinkhorn loop's
    ``divide(add(sum(x), eps))`` triples into one fused kernel each: per decode
    step ``vs_Add`` 3570 -> 43 and ``g2_Divide`` 3408 -> 11, replaced by 3354
    fused dispatches.  See :data:`_HC_COMPILE` for why the whole-forward compile
    receipt does not apply here, and :data:`_HC_COMPILE_MAX_ROWS` for the shape
    cap the tape cache needs.
  * **Attention.**  The sink is now one extra KV column rather than a hand-rolled
    fp32 softmax (:meth:`DeepseekV4Attention._attend`).  Worth 4 dispatches per
    layer at bf16 (the ``maximum``/``max``-reduce/``exp``/``divide`` chain and the
    two fp32 casts around it), 1 at fp32 — small next to the Sinkhorn — but it
    also stops materialising both full-size fp32 temporaries: ``dense`` wrote
    roughly 16 bytes of transient per score element (bf16 block, fp32 upcast,
    fp32 exp, fp32 probabilities, bf16 cast) where ``fused`` writes 6.  At decode
    that is noise; at a 1024-token prefill chunk on the real model it is ~670 MB
    of fp32 traffic per compressed layer that no longer happens.
  * **Together**: 19,809 -> 14,639 dispatches (-26.1%) and 384 -> 288 command
    buffers (-25.0%) per bf16 decode step; 17,733 -> 13,039 (-26.5%) at fp32.
    Roughly 5,200 fewer dispatches per token at ~2.9 us of host encode each.
  * **The Sinkhorn reduction floor (``MTPLX_DSV4_SINKHORN_KERNEL``).**  3,678 of
    the remaining 14,639 (25%) are the Sinkhorn's own row/column ``reduce_sum``
    dispatches — 39 per ``pre`` call, one per normalisation pass — plus the 39
    fused divides and the row-softmax around them.  ``mx.compile`` does not fuse
    reductions and no *stock* op does 20 alternating normalisations in one launch,
    so this is the floor for the stock formulation.  It is not algebraically
    removable — ``mixes = F.linear(x, hc_fn) * rsqrt`` is activation-dependent, so
    ``comb`` cannot be precomputed at load — but the whole 40-pass schedule is a
    deterministic recurrence on a ``[.., 4, 4]`` tensor, which is exactly what a
    hand kernel does in one launch.  :func:`_sinkhorn_kernel_apply` runs the entire
    schedule (softmax + 20 column- + 19 row-normalises) per ``pre`` call as a single
    ``mx.fast.metal_kernel`` dispatch, one thread per matrix, the normalisations in
    registers.  Measured on the shrunk census (bf16, per decode step): total ops
    14,639 -> 7,845, the ~3,354 Sinkhorn ``reduce_sum`` + ~3,354 fused-divide +
    ~86 softmax dispatches collapse to 86 kernel dispatches; command buffers
    288 -> 155.  Bit-identical to the stock loop (:func:`_sinkhorn_ops`) at 1e-6,
    argmax exact — it is env-gated **off** until a real-weights window confirms the
    host-encode saving becomes tok/s.  See :data:`_SINKHORN_KERNEL`.
  * **``mx.fast.scaled_dot_product_attention`` does not fuse this attention, on
    any MLX on this box.**  It takes ``sinks=`` natively and the ``sdpa`` arm uses
    it and is gated exact — but its Metal kernels are only instantiated for head
    dims 64/96/128/256 (0.31.2) and 64/96/128/192/256 (0.32.0 and 0.32.1.dev),
    verified against each shipped ``mlx.metallib``, and DeepSeek-V4's MLA latent
    is 512 wide.  Every call therefore takes MLX's own unfused fallback.  It is
    still the *cheapest measured arm* — 215 dispatches per step below ``fused``,
    because its sink ``concatenate``/``slice`` pair on the score block costs less
    at decode than ``fused``'s ``pad`` of the KV block — but it is not the default
    because those two copies scale with ``s * n_heads`` at prefill where
    ``fused``'s scales with ``n_kv``.  Pick on the real model with the env knob.
    The consequence that survives either way: at bf16 the scores are still
    rounded to bf16 before the softmax, because *something* has to materialise
    them.  Only a kernel instantiated at head_dim 512 fixes that.

Provenance: reference files fetched read-only from
``https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash`` (inference/model.py,
inference/kernel.py, config.json) and
``https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit`` (config.json,
model.safetensors.index.json).  The reference GPU kernels require CUDA/tilelang and
cannot run on this box; the M2 oracle is a faithful transcription of their documented
elementwise math (verified elementwise, not by running the shipped kernel).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU
from mtplx.attention_context import current_attention_phase
from mtplx.deepseek_v4_nvfp4_kv import (
    FixedMiaNVFP4Window,
    FixedMiaNVFP4WindowRecords,
    MiaNVFP4Rows,
    PagedMiaNVFP4Rows,
    decode_stock432,
    install_fixed_window_record_writer,
    install_stock432_record_packer,
)
from mtplx.deepseek_v4_paged_indexer import (
    MiaIndexerQueryRecords,
    MiaIndexerRoPETable,
    MiaIndexerWorkspace,
    MiaTopKSelection,
    PagedMiaIndexerRows,
    install_indexer_query_records,
    install_paged_indexer_m6_topk,
    install_paged_indexer_topk,
)
from mtplx.kernels.deepseek_v4_compressor import (
    fused_indexer_records,
    fused_nvfp4_records,
    install_indexer_record_kernel,
    install_nvfp4_record_kernel,
)
from mtplx.kernels.deepseek_v4_nvfp4_mla import (
    MiaMLAWorkspace,
    install_nvfp4_prefill_mla,
    install_nvfp4_sparse_mla,
    mia_mla_workspace,
)


# Default per-layer compress ratios for DeepSeek-V4-Flash (43 body layers; the
# 44th entry is the dropped MTP layer).  0 = pure sliding-window; 4 = overlapping
# compressor + indexer; 128 = non-overlapping compressor + strided index.
_DEFAULT_COMPRESS_RATIOS = (
    [0, 0]
    + [4, 128] * 20
    + [4, 0]
)

# How many token positions a :class:`DeepseekV4Cache` can un-decode (``trim``).
# Speculative decode only ever rewinds the rejected tail of one verify batch, so
# the real requirement is ``speculative_depth + 1`` (<= 9 for every depth MTPLX
# runs).  The default is set an order of magnitude above that because the cost is
# a handful of retained rows per layer, while the alternative -- discovering the
# bound is too small mid-request -- is a hard failure.  See
# :meth:`DeepseekV4Cache.trim` for what the capacity buys on each lane.
_DEFAULT_ROLLBACK_CAPACITY = 64


# ---------------------------------------------------------------------------
# Decode-path knobs
# ---------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


#: How :meth:`DeepseekV4Attention._o_lora` gets ``wo_a``.
#:
#: ``cached`` (default)
#:     Dequantise the static ``[o_groups*o_lora_rank, n_heads*head_dim/o_groups]``
#:     matrix once and keep the dense result.  Bit-identical to ``dequant``.
#: ``dequant``
#:     Re-run ``mx.dequantize`` on every call — the pre-cache behaviour, kept as
#:     the A/B control and as the oracle the bit-identity gate compares against.
#: ``gather_qmm``
#:     Skip the dense materialisation entirely and run the ``o_groups`` LoRA groups
#:     as one quantised block-diagonal matmul.  *Not* bit-identical (different
#:     accumulation order); off by default until a GPU window says it wins.
_O_LORA_MODES = ("cached", "dequant", "gather_qmm")


def _o_lora_mode_from_env() -> str:
    raw = (os.environ.get("MTPLX_DSV4_O_LORA") or "").strip().lower()
    if not raw:
        return "cached"
    if raw not in _O_LORA_MODES:
        raise ValueError(
            "MTPLX_DSV4_O_LORA must be one of "
            f"{', '.join(_O_LORA_MODES)}; got {raw!r}"
        )
    return raw


#: How :meth:`DeepseekV4Attention._attend` forms the attention block.
#:
#: ``fused`` (default)
#:     One zero row appended to the KV block (so its raw score is exactly 0),
#:     the per-head ``attn_sink`` supplied as an additive column on top of it,
#:     and one ``mx.softmax(..., precise=True)`` over the result.  Removes the
#:     hand-rolled max/exp/sum/divide chain and the two full-size fp32
#:     temporaries it materialised; the softmax keeps fp32 accumulators
#:     internally at bf16 I/O, which is what the reference kernel does.
#: ``sdpa``
#:     ``mx.fast.scaled_dot_product_attention`` with ``sinks=attn_sink`` — the
#:     same semantics expressed as one op.  See :meth:`DeepseekV4Attention._attend`
#:     for why this is *not* the default on MLX 0.31.2.
#: ``dense``
#:     The pre-change path: materialised score block, fp32 softmax with the sink
#:     folded into ``max``/``denom`` by hand.  Kept as the A/B control and as the
#:     oracle the parity gate compares the other two against.
_ATTN_MODES = ("fused", "sdpa", "dense")


def _attn_mode_from_env() -> str:
    raw = (os.environ.get("MTPLX_DSV4_ATTN") or "").strip().lower()
    if not raw:
        return "fused"
    if raw not in _ATTN_MODES:
        raise ValueError(
            "MTPLX_DSV4_ATTN must be one of "
            f"{', '.join(_ATTN_MODES)}; got {raw!r}"
        )
    return raw


#: Whether the Hyper-Connection pre/post/head chains run through ``mx.compile``.
#:
#: The Sinkhorn normalisation is 20 alternating row/column passes over a
#: ``[..., hc, hc]`` tensor — 16 floats at decode — and it runs twice per layer
#: plus once at the head.  Uncompiled that is ~248 primitives per ``pre`` call
#: and roughly two thirds of the entire decode step's graph, all of it host
#: overhead on tensors too small for the GPU to notice.  ``mx.compile`` collapses
#: each ``divide(add(sum(x), eps))`` triple into one fused kernel and replays a
#: prebuilt tape instead of rebuilding the graph from Python on every call.
#:
#: This is *not* the whole-forward compile lever, which is dead on this box: that
#: one lost because the kernels it fused were already bandwidth-bound.  Here the
#: kernels are 4x4.
#:
#: ``MTPLX_DSV4_HC_COMPILE=0`` restores the eager path as the A/B control.  Read
#: at import; tests set the module attribute.
_HC_COMPILE = _env_flag("MTPLX_DSV4_HC_COMPILE", True)

#: Row count (``b * s``) above which the compiled Hyper-Connection variant is
#: bypassed.
#:
#: MLX keeps one compiled tape per distinct input *shape*, in an unbounded list
#: it scans linearly on every call (``CompilerCache::find``).  Decode and
#: speculative verify use a handful of tiny, repeating shapes, so they hit a warm
#: tape every time.  Prefill does not — chunk remainders make ``s`` effectively
#: arbitrary — and it is also the regime where the per-primitive overhead compile
#: removes is already amortised over real work.  Capping the compiled path at a
#: small row count keeps the tape list bounded *and* puts compile only where it
#: pays.
_HC_COMPILE_MAX_ROWS = 32


#: Whether the Sinkhorn alternating-normalisation loop runs as one Metal kernel.
#:
#: After :data:`_HC_COMPILE`, the whole decode step's remaining reduction floor is
#: the Sinkhorn's own ``reduce_sum`` dispatches: 39 per ``pre`` call, one per
#: normalisation pass (20 column-normalises over ``axis=-2`` + 19 row-normalises
#: over ``axis=-1``), plus the 39 fused divides and the row-softmax.  ``mx.compile``
#: does not fuse reductions and no stock op does 20 alternating normalisations in
#: one launch, so at 86 ``pre`` calls per token that is ~3591 ``reduce_sum`` +
#: ~3354 divide + ~86 softmax dispatches the host has to build and encode every
#: step — all of it on a ``[..., hc, hc]`` tensor that is 16 floats at decode.
#:
#: :func:`_sinkhorn_kernel_apply` replaces the entire loop with one
#: ``mx.fast.metal_kernel`` per ``pre`` call: one threadgroup thread per matrix
#: carries the 16 floats in registers and runs all 40 normalisation passes
#: internally, so the whole block collapses to a single dispatch.  The math is the
#: *identical* fp32 arithmetic in the *identical* order as :func:`_sinkhorn_ops`
#: (the loop it replaces); the only thing removed is dispatch count.  Composes with
#: :data:`_HC_COMPILE` (the kernel call is opaque to ``mx.compile`` but sits at the
#: tail of the ``pre`` tape).
#:
#: Default OFF — a pure dispatch-count lever whose win only shows on the real
#: model's GPU window; the parity gate is exact (1e-6, argmax exact) so it can be
#: flipped on the moment that window confirms tok/s.  Read from
#: ``MTPLX_DSV4_SINKHORN_KERNEL`` at import; tests set the module attribute.
_SINKHORN_KERNEL = _env_flag("MTPLX_DSV4_SINKHORN_KERNEL", False)


#: Escape hatch restoring the pre-fix all-fp32 activation path (rope output,
#: compressed KV rows and the attention probability block).  The reference keeps
#: all three at the model dtype — see :func:`_apply_interleaved_rope`,
#: :class:`Compressor` and :meth:`DeepseekV4Attention.__call__` — so this is an
#: A/B control, not a supported serving mode.  Read from
#: ``MTPLX_DSV4_FP32_ACTIVATIONS`` at import; tests set the module attribute.
_FP32_ACTIVATIONS = _env_flag("MTPLX_DSV4_FP32_ACTIVATIONS", False)


#: Fuse only the MoE *tail* after the stock quantised ``SwitchGLU`` projections:
#: ``(routed * weights[..., None].astype(routed.dtype)).sum(axis=-2) + shared``.
#:
#: The kernel is deliberately a narrow DeepSeek-V4-Flash decode/verify lane: BF16
#: activation storage, hidden width 4096, and exactly six routed experts.  It does
#: not alter gate/up/down projection ownership, their Q2 format, their clamp, or
#: routing.  The flag is read once, and an enabled instance receives a prebound
#: callable at the post-load construction boundary; there is no environment or
#: eligibility branch in the token path.  A forced lane on a non-GPU device fails
#: during installation instead of silently falling through to stock.
_MOE_TAIL = _env_flag("MTPLX_DSV4_MOE_TAIL", False)
_MOE_TAIL_TOPK = 6
_MOE_TAIL_HIDDEN = 4096
_MOE_TAIL_EXPERTS = 256
_MOE_TAIL_BODY_LAYERS = 43
_MOE_TAIL_MTP_BLOCKS = 1
_MOE_TAIL_HASH_LAYERS = 3
_MOE_TAIL_INTERMEDIATE = 2048
_MOE_TAIL_SHARED_EXPERTS = 1
_MOE_TAIL_VOCAB = 129280
_MOE_TAIL_KERNEL = None
_MOE_TAIL_SELF_CHECKED = False


# One output owner serially forms the six BF16 products then six BF16 additions.
# MLX's strided reducer association is an implementation detail, so this is not
# asserted equivalent by inspection: :func:`_verify_moe_tail_exact` runs the real
# Metal kernel against the stock expression for M=1 and M=4 before the route can
# be installed.  A mismatch fails construction; it can never become a hot-path
# fallback.
_MOE_TAIL_METAL_SOURCE = r"""
    using namespace metal;
    constexpr uint TOPK = 6;
    constexpr uint HIDDEN = 4096;

    uint i = thread_position_in_grid.x;
    if (i >= n_elements) { return; }
    uint row = i / HIDDEN;
    uint column = i % HIDDEN;
    T mixed = T(0.0f);
    for (uint route = 0; route < TOPK; ++route) {
        T product = T(routed[(row * TOPK + route) * HIDDEN + column]
                      * weights[row * TOPK + route]);
        mixed = T(product + mixed);
    }
    out[i] = T(mixed + shared[i]);
"""


def _validate_moe_tail_config(args: "ModelArgs") -> None:
    """Validate the fixed Q2 tail lane before any generation graph is built."""
    if _FP32_ACTIVATIONS:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires DeepSeek-V4-Flash BF16 activation "
            "storage; MTPLX_DSV4_FP32_ACTIVATIONS is an explicit stock A/B arm"
        )
    if int(args.num_experts_per_tok) != _MOE_TAIL_TOPK:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires DeepSeek-V4-Flash top-k=6; got "
            f"top-k={args.num_experts_per_tok}"
        )
    if int(args.hidden_size) != _MOE_TAIL_HIDDEN:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires DeepSeek-V4-Flash hidden_size=4096; got "
            f"hidden_size={args.hidden_size}"
        )
    if int(args.n_routed_experts) != _MOE_TAIL_EXPERTS:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires DeepSeek-V4-Flash n_routed_experts=256; got "
            f"n_routed_experts={args.n_routed_experts}"
        )
    for field_name, expected in (
        ("num_hidden_layers", _MOE_TAIL_BODY_LAYERS),
        ("num_hash_layers", _MOE_TAIL_HASH_LAYERS),
        ("moe_intermediate_size", _MOE_TAIL_INTERMEDIATE),
        ("n_shared_experts", _MOE_TAIL_SHARED_EXPERTS),
        ("vocab_size", _MOE_TAIL_VOCAB),
        ("num_nextn_predict_layers", _MOE_TAIL_MTP_BLOCKS),
    ):
        actual = int(getattr(args, field_name))
        if actual != expected:
            raise ValueError(
                f"MTPLX_DSV4_MOE_TAIL requires {field_name}={expected}; got {actual}"
            )


def _moe_tail_shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _validate_moe_tail_quantized_projection(
    module,
    *,
    label: str,
    bits: int,
    group_size: int,
    mode: str,
    weight_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    scale_dtype,
    biases: bool,
) -> None:
    """Validate one already-loaded packed projection before route installation."""
    actual_bits = getattr(module, "bits", None)
    actual_group_size = getattr(module, "group_size", None)
    actual_mode = str(getattr(module, "mode", "")).lower()
    if actual_bits != bits:
        raise ValueError(f"{label} requires bits={bits}; got {actual_bits!r}")
    if actual_group_size != group_size:
        raise ValueError(
            f"{label} requires group_size={group_size}; got {actual_group_size!r}"
        )
    if actual_mode != mode:
        raise ValueError(f"{label} requires mode={mode}; got {actual_mode!r}")
    weight = getattr(module, "weight", None)
    if getattr(weight, "dtype", None) != mx.uint32:
        raise ValueError(f"{label} packed weight must use uint32 storage")
    if _moe_tail_shape(weight) != weight_shape:
        raise ValueError(
            f"{label} packed weight shape must be {weight_shape}; "
            f"got {_moe_tail_shape(weight)}"
        )
    scales = getattr(module, "scales", None)
    if (
        _moe_tail_shape(scales) != scale_shape
        or getattr(scales, "dtype", None) != scale_dtype
    ):
        raise ValueError(
            f"{label} scale/bias shape or scale dtype is invalid: "
            f"shape={_moe_tail_shape(scales)} dtype={getattr(scales, 'dtype', None)}"
        )
    offsets = getattr(module, "biases", None)
    if biases:
        if (
            _moe_tail_shape(offsets) != scale_shape
            or getattr(offsets, "dtype", None) != mx.bfloat16
        ):
            raise ValueError(
                f"{label} scale/bias shape or bias dtype is invalid: "
                f"shape={_moe_tail_shape(offsets)} "
                f"dtype={getattr(offsets, 'dtype', None)}"
            )
    elif offsets is not None:
        raise ValueError(f"{label} must not carry affine biases")


def _validate_moe_tail_dense_projection(
    module, *, label: str, weight_shape: tuple[int, ...]
) -> None:
    weight = getattr(module, "weight", None)
    if (
        _moe_tail_shape(weight) != weight_shape
        or getattr(weight, "dtype", None) != mx.bfloat16
        or getattr(module, "scales", None) is not None
        or getattr(module, "biases", None) is not None
    ):
        raise ValueError(
            f"{label} MTP dense shared projection must be BF16 {weight_shape}; "
            f"shape={_moe_tail_shape(weight)} dtype={getattr(weight, 'dtype', None)}"
        )


def _validate_moe_tail_gate(layer, *, layer_id: int, hash_layer: bool) -> None:
    gate = getattr(getattr(layer, "ffn", None), "gate", None)
    if gate is None:
        raise ValueError(f"body layer {layer_id} has no MoE gate")
    for field_name, expected in (
        ("dim", _MOE_TAIL_HIDDEN),
        ("topk", _MOE_TAIL_TOPK),
        ("n_routed", _MOE_TAIL_EXPERTS),
        ("hash", hash_layer),
    ):
        if getattr(gate, field_name, None) != expected:
            raise ValueError(
                f"body layer {layer_id} gate requires {field_name}={expected!r}; "
                f"got {getattr(gate, field_name, None)!r}"
            )
    weight = getattr(gate, "weight", None)
    if (
        _moe_tail_shape(weight) != (_MOE_TAIL_EXPERTS, _MOE_TAIL_HIDDEN)
        or getattr(weight, "dtype", None) != mx.bfloat16
    ):
        raise ValueError(f"body layer {layer_id} gate weight geometry is invalid")
    if hash_layer:
        table = getattr(gate, "tid2eid", None)
        if (
            _moe_tail_shape(table) != (_MOE_TAIL_VOCAB, _MOE_TAIL_TOPK)
            or getattr(table, "dtype", None) != mx.int64
        ):
            raise ValueError(f"body layer {layer_id} hash routing table is invalid")
    else:
        correction = getattr(gate, "e_score_correction_bias", None)
        if (
            _moe_tail_shape(correction) != (_MOE_TAIL_EXPERTS,)
            or getattr(correction, "dtype", None) != mx.float32
        ):
            raise ValueError(f"body layer {layer_id} score correction is invalid")


def _validate_loaded_moe_tail_contract(model, config: dict) -> dict:
    """Prove exact topology and loaded storage before compiling the candidate."""
    if str(getattr(model, "model_type", "")).lower() != "deepseek_v4":
        raise ValueError("MTPLX_DSV4_MOE_TAIL requires loaded model_type=deepseek_v4")
    layers = list(getattr(model, "layers", ()))
    if len(layers) != _MOE_TAIL_BODY_LAYERS:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires exactly 43 body layers; "
            f"got {len(layers)}"
        )
    mtp_blocks = list(getattr(model, "mtp_blocks", ()))
    if len(mtp_blocks) != _MOE_TAIL_MTP_BLOCKS:
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL requires exactly one MTP block; "
            f"got {len(mtp_blocks)}"
        )
    args = getattr(model, "args", None)
    if args is None:
        raise ValueError("MTPLX_DSV4_MOE_TAIL loaded model has no args")
    _validate_moe_tail_config(args)
    expected_config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": _MOE_TAIL_BODY_LAYERS,
        "num_nextn_predict_layers": _MOE_TAIL_MTP_BLOCKS,
        "num_hash_layers": _MOE_TAIL_HASH_LAYERS,
        "n_routed_experts": _MOE_TAIL_EXPERTS,
        "num_experts_per_tok": _MOE_TAIL_TOPK,
        "hidden_size": _MOE_TAIL_HIDDEN,
        "moe_intermediate_size": _MOE_TAIL_INTERMEDIATE,
        "n_shared_experts": _MOE_TAIL_SHARED_EXPERTS,
        "vocab_size": _MOE_TAIL_VOCAB,
    }
    mismatches = {
        field: (config.get(field), expected)
        for field, expected in expected_config.items()
        if config.get(field) != expected
    }
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != 44:
        mismatches["compress_ratios"] = (
            len(ratios) if isinstance(ratios, list) else None,
            44,
        )
    if mismatches:
        raise ValueError(f"MTPLX_DSV4_MOE_TAIL config topology mismatch: {mismatches}")
    quantization = config.get("quantization")
    if not isinstance(quantization, dict) or any(
        quantization.get(field) != expected
        for field, expected in (
            ("bits", 4),
            ("group_size", 64),
            ("mode", "affine"),
        )
    ):
        raise ValueError(
            "MTPLX_DSV4_MOE_TAIL config quantization default must be "
            "4-bit affine group_size=64"
        )

    shared_contract = {
        "gate_proj": ((2048, 512), (2048, 64)),
        "up_proj": ((2048, 512), (2048, 64)),
        "down_proj": ((4096, 256), (4096, 32)),
    }
    for layer_id, layer in enumerate(layers):
        _validate_moe_tail_gate(
            layer, layer_id=layer_id, hash_layer=layer_id < _MOE_TAIL_HASH_LAYERS
        )
        ffn = getattr(layer, "ffn", None)
        switch = getattr(ffn, "switch_mlp", None)
        shared = getattr(ffn, "shared_experts", None)
        gate_group_size = 32 if layer_id < 42 else 64
        gate_scale_groups = 128 if layer_id < 42 else 64
        routed_contract = {
            "gate_proj": (
                (256, 2048, 256),
                (256, 2048, gate_scale_groups),
                gate_group_size,
            ),
            "up_proj": ((256, 2048, 256), (256, 2048, 64), 64),
            "down_proj": ((256, 4096, 128), (256, 4096, 32), 64),
        }
        for projection, (weight_shape, scale_shape, group_size) in routed_contract.items():
            stem = f"model.layers.{layer_id}.ffn.switch_mlp.{projection}"
            expected_spec = {
                "bits": 2,
                "group_size": group_size,
                "mode": "affine",
            }
            actual_spec = quantization.get(stem)
            if not isinstance(actual_spec, dict) or any(
                actual_spec.get(field_name) != expected
                for field_name, expected in expected_spec.items()
            ):
                raise ValueError(
                    f"MTPLX_DSV4_MOE_TAIL config quantization for {stem} "
                    f"must be {expected_spec}; got {actual_spec!r}"
                )
            _validate_moe_tail_quantized_projection(
                getattr(switch, projection, None),
                label=f"body layer {layer_id} routed {projection}",
                bits=2,
                group_size=group_size,
                mode="affine",
                weight_shape=weight_shape,
                scale_shape=scale_shape,
                scale_dtype=mx.bfloat16,
                biases=True,
            )
        for projection, (weight_shape, scale_shape) in shared_contract.items():
            _validate_moe_tail_quantized_projection(
                getattr(shared, projection, None),
                label=f"body shared layer {layer_id} {projection}",
                bits=4,
                group_size=64,
                mode="affine",
                weight_shape=weight_shape,
                scale_shape=scale_shape,
                scale_dtype=mx.bfloat16,
                biases=True,
            )

    mtp = mtp_blocks[0]
    _validate_moe_tail_gate(mtp, layer_id=43, hash_layer=False)
    mtp_switch = mtp.ffn.switch_mlp
    mtp_routed_contract = {
        "gate_proj": ((256, 2048, 512), (256, 2048, 128)),
        "up_proj": ((256, 2048, 512), (256, 2048, 128)),
        "down_proj": ((256, 4096, 256), (256, 4096, 64)),
    }
    for projection, (weight_shape, scale_shape) in mtp_routed_contract.items():
        stem = f"mtp.0.ffn.switch_mlp.{projection}"
        expected_spec = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        actual_spec = quantization.get(stem)
        if not isinstance(actual_spec, dict) or any(
            actual_spec.get(field_name) != expected
            for field_name, expected in expected_spec.items()
        ):
            raise ValueError(
                f"MTPLX_DSV4_MOE_TAIL config quantization for {stem} "
                f"must be {expected_spec}; got {actual_spec!r}"
            )
        _validate_moe_tail_quantized_projection(
            getattr(mtp_switch, projection, None),
            label=f"MTP routed {projection}",
            bits=4,
            group_size=32,
            mode="mxfp4",
            weight_shape=weight_shape,
            scale_shape=scale_shape,
            scale_dtype=mx.uint8,
            biases=False,
        )
    mtp_shared = mtp.ffn.shared_experts
    for projection, weight_shape in {
        "gate_proj": (2048, 4096),
        "up_proj": (2048, 4096),
        "down_proj": (4096, 2048),
    }.items():
        _validate_moe_tail_dense_projection(
            getattr(mtp_shared, projection, None),
            label=f"MTP dense shared {projection}",
            weight_shape=weight_shape,
        )
    return {
        "body_layers": len(layers),
        "mtp_blocks": len(mtp_blocks),
        "body_q2_routed_projections": len(layers) * 3,
        "body_q4_shared_projections": len(layers) * len(shared_contract),
        "mtp_mxfp4_routed_projections": len(mtp_routed_contract),
        "mtp_dense_shared_projections": 3,
    }


def configure_deepseek_v4_moe_tail(model, config: dict) -> dict | None:
    """Install the fixed body route once, after loaded storage is fully known."""
    if not _MOE_TAIL:
        return None
    validated = _validate_loaded_moe_tail_contract(model, config)
    candidate = _install_moe_tail_combine(model.args)
    for layer in model.layers:
        layer.ffn._tail_combine = candidate
    for block in model.mtp_blocks:
        block.ffn._tail_combine = _stock_moe_tail_combine
    return {
        "route": "decode_verify_m4",
        "body_layers_installed": len(model.layers),
        "mtp_layers_stock": len(model.mtp_blocks),
        "verify_rows": 4,
        "repair_rows": 1,
        "topk": _MOE_TAIL_TOPK,
        "hidden_size": _MOE_TAIL_HIDDEN,
        "kernel_selfcheck_exact": True,
        **validated,
    }


def _moe_tail_metal_kernel():
    """Build the one fixed BF16 tail kernel during post-load configuration."""
    global _MOE_TAIL_KERNEL
    if _MOE_TAIL_KERNEL is None:
        _MOE_TAIL_KERNEL = mx.fast.metal_kernel(
            name="mtplx_dsv4_moe_tail_bf16_topk6_h4096",
            input_names=["routed", "weights", "shared", "n_elements"],
            output_names=["out"],
            source=_MOE_TAIL_METAL_SOURCE,
        )
    return _MOE_TAIL_KERNEL


def _moe_tail_apply(kernel, routed: mx.array, weights: mx.array, shared: mx.array) -> mx.array:
    """Dispatch the precompiled fixed tail; ``rows`` is the only varying value."""
    rows = int(routed.shape[0])
    n_elements = rows * _MOE_TAIL_HIDDEN
    (out,) = kernel(
        inputs=[routed, weights.astype(mx.bfloat16), shared, n_elements],
        template=[("T", routed.dtype)],
        grid=((n_elements + 31) // 32 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, _MOE_TAIL_HIDDEN)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def _verify_moe_tail_exact(kernel) -> None:
    """Prove the Metal association against stock on real M=1 and M=4 tensors.

    The values are deterministic and deliberately span signs/exponents.  This is
    an installation boundary self-check, not a per-token proof mechanism.
    """
    global _MOE_TAIL_SELF_CHECKED
    if _MOE_TAIL_SELF_CHECKED:
        return
    for rows in (1, 4):
        n = rows * _MOE_TAIL_TOPK * _MOE_TAIL_HIDDEN
        routed = ((mx.arange(n, dtype=mx.float32) % 29 - 14) / 7).reshape(
            rows, _MOE_TAIL_TOPK, _MOE_TAIL_HIDDEN
        ).astype(mx.bfloat16)
        weights = ((mx.arange(rows * _MOE_TAIL_TOPK, dtype=mx.float32) % 13 - 6) / 5)
        weights = weights.reshape(rows, _MOE_TAIL_TOPK).astype(mx.bfloat16)
        shared = ((mx.arange(rows * _MOE_TAIL_HIDDEN, dtype=mx.float32) % 31 - 15) / 11)
        shared = shared.reshape(rows, _MOE_TAIL_HIDDEN).astype(mx.bfloat16)
        stock = _stock_moe_tail_combine(routed, weights, shared)
        fused = _moe_tail_apply(kernel, routed, weights, shared)
        mx.eval(stock, fused)
        if not mx.array_equal(stock, fused):
            max_abs = float(
                mx.max(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32))).item()
            )
            raise RuntimeError(
                "MTPLX_DSV4_MOE_TAIL failed exact Metal self-check at "
                f"M={rows}: max_abs={max_abs:g}"
            )
    _MOE_TAIL_SELF_CHECKED = True


class _InstalledMoETailRoute:
    """Phase-and-M route over an already validated custom kernel.

    ``current_attention_phase`` is the runtime-owned phase signal already set by
    generation.  It is the only hot decision: no environment, topology, dtype,
    or model metadata is re-read after installation.  Tiny M=1/M=4 prefills are
    therefore explicitly stock despite sharing decode's flattened row count.
    """

    __slots__ = ("kernel",)

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def __call__(
        self, routed: mx.array, weights: mx.array, shared: mx.array
    ) -> mx.array:
        phase = current_attention_phase()
        rows = int(routed.shape[0])
        if phase == "decode_verify" and rows == 4:
            return _moe_tail_apply(self.kernel, routed, weights, shared)
        return _stock_moe_tail_combine(routed, weights, shared)


def _install_moe_tail_combine(args: "ModelArgs"):
    """Return the fixed tail callable, or fail before an enabled generation lane.

    The explicit custom phase route is only M=4 K3 verification.  AR decode,
    prefill (including tiny M=1/M=4 prefills), and every other phase remain the
    stock expression.  Phase + logical M are the sole runtime decisions;
    topology, dtype mode, compilation, and all other routing are fixed here.
    """
    _validate_moe_tail_config(args)
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError(
            "MTPLX_DSV4_MOE_TAIL requires a Metal GPU at post-load installation; "
            "select the explicit stock route on CPU"
        )
    kernel = _moe_tail_metal_kernel()
    _verify_moe_tail_exact(kernel)

    return _InstalledMoETailRoute(kernel)


def _store_dtype(dtype):
    """Dtype an activation is *stored* at (fp32 math is unaffected either way)."""
    return mx.float32 if _FP32_ACTIVATIONS else dtype


class _DerivedCache:
    """Holder for a tensor derived from parameters (e.g. a one-time dequant).

    A plain object rather than a bare ``mx.array`` attribute on purpose:
    ``nn.Module.__setattr__`` routes every ``mx.array``/``dict``/``list``/``tuple``
    into the module's own dict, and only the leading-underscore filter keeps it out
    of ``parameters()``.  Hanging the cache off a plain object keeps it out of the
    module dict altogether, so ``load_weights(strict=True)``, ``save_weights``,
    ``set_dtype`` and ``mx.eval(model)`` cannot see it at all.

    ``src`` holds the parameters the value was derived from, so a later
    ``load_weights``/``update``/``set_dtype`` (which rebinds those arrays)
    invalidates the cache by identity instead of serving a stale copy.
    """

    __slots__ = ("src", "value")

    def __init__(self) -> None:
        self.src: Optional[tuple] = None
        self.value: Optional[mx.array] = None

    def get(self, src: tuple) -> Optional[mx.array]:
        if self.value is None or self.src is None or len(self.src) != len(src):
            return None
        return self.value if all(a is b for a, b in zip(self.src, src)) else None

    def put(self, src: tuple, value: mx.array) -> mx.array:
        self.src = src
        self.value = value
        return value


class _UninstalledGatherOLora:
    """Fail-closed placeholder used until post-load route installation."""

    __slots__ = ()

    def __call__(self, _o: mx.array) -> mx.array:
        raise RuntimeError(
            "gather_qmm o-LoRA was selected but its post-load route was not installed"
        )


def _o_lora_linear_logical_weight_shape(linear) -> tuple[int, ...]:
    """Return ``[output, input]`` without materializing a quantized weight."""

    weight_shape = tuple(getattr(getattr(linear, "weight", None), "shape", ()))
    if len(weight_shape) != 2:
        return ()
    if not isinstance(linear, nn.QuantizedLinear):
        return weight_shape
    try:
        bits = int(linear.bits)
        group_size = int(linear.group_size)
        scales_shape = tuple(linear.scales.shape)
    except (AttributeError, TypeError, ValueError):
        return ()
    packed_divisor = 32 // bits if bits > 0 and 32 % bits == 0 else 0
    if not packed_divisor or group_size <= 0 or len(scales_shape) != 2:
        return ()
    packed_logical = (weight_shape[0], weight_shape[1] * packed_divisor)
    scales_logical = (scales_shape[0], scales_shape[1] * group_size)
    return packed_logical if packed_logical == scales_logical else ()


class _DirectGatherOLora:
    """Prevalidated direct gather route; execution performs no eligibility lookup."""

    __slots__ = (
        "biases",
        "bits",
        "group_size",
        "groups",
        "mode",
        "per_group_input",
        "rank",
        "scales",
        "weight",
        "wo_b",
    )

    def __init__(self, attention: "DeepseekV4Attention", quant: tuple) -> None:
        weight, scales, biases, group_size, bits, mode = quant
        groups = int(attention.n_groups)
        rank = int(attention.o_lora_rank)
        per_group_input = int(
            attention.n_heads * attention.head_dim // attention.n_groups
        )
        output_rows = groups * rank
        if groups <= 0 or rank <= 0 or per_group_input <= 0:
            raise ValueError("gather_qmm o-LoRA geometry must be positive")
        if per_group_input % int(group_size):
            raise ValueError(
                "gather_qmm o-LoRA input width is not divisible by group_size"
            )
        packed_divisor = 32 // int(bits) if int(bits) and 32 % int(bits) == 0 else 0
        if not packed_divisor:
            raise ValueError(f"gather_qmm o-LoRA has unsupported bits={bits!r}")
        expected_weight = (output_rows, per_group_input // packed_divisor)
        expected_scales = (output_rows, per_group_input // int(group_size))
        if tuple(weight.shape) != expected_weight:
            raise ValueError(
                f"gather_qmm o-LoRA packed weight shape {tuple(weight.shape)} "
                f"does not match {expected_weight}"
            )
        if tuple(scales.shape) != expected_scales:
            raise ValueError(
                f"gather_qmm o-LoRA scale shape {tuple(scales.shape)} "
                f"does not match {expected_scales}"
            )
        if biases is not None and tuple(biases.shape) != expected_scales:
            raise ValueError(
                f"gather_qmm o-LoRA bias shape {tuple(biases.shape)} "
                f"does not match {expected_scales}"
            )
        expected_wo_b = (int(attention.dim), output_rows)
        if _o_lora_linear_logical_weight_shape(attention.wo_b) != expected_wo_b:
            raise ValueError("gather_qmm o-LoRA wo_b input geometry is invalid")
        self.groups = groups
        self.rank = rank
        self.per_group_input = per_group_input
        self.weight = weight.reshape(groups, rank, -1)
        self.scales = scales.reshape(groups, rank, -1)
        self.biases = (
            None if biases is None else biases.reshape(groups, rank, -1)
        )
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = mode
        self.wo_b = attention.wo_b

    def __call__(self, o: mx.array) -> mx.array:
        batch, sequence, _ = o.shape
        rows = batch * sequence
        x = o.reshape(rows, self.groups, self.per_group_input).swapaxes(0, 1)
        out = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return self.wo_b(
            out.swapaxes(0, 1).reshape(
                batch, sequence, self.groups * self.rank
            )
        )


@lru_cache(maxsize=1)
def _mia_inverse_rope_output_kernel():
    source = r"""
        uint element = thread_position_in_grid.x;
        constexpr uint HEADS = 64u;
        constexpr uint HEAD_DIM = 512u;
        constexpr uint ROPE_START = 448u;
        constexpr uint ROPE_DIM = 64u;
        uint total = uint(rows) * HEADS * HEAD_DIM;
        if (element >= total) return;
        uint row = element / (HEADS * HEAD_DIM);
        uint within = element % (HEADS * HEAD_DIM);
        uint head = within / HEAD_DIM;
        uint dim = within % HEAD_DIM;
        float value = float(o[element]);
        if (dim >= ROPE_START) {
            uint rope_dim = dim - ROPE_START;
            uint pair = rope_dim / 2u;
            uint partner_dim = dim ^ 1u;
            float partner = float(
                o[(size_t(row) * HEADS + head) * HEAD_DIM + partner_dim]
            );
            float c = float(cos[size_t(row) * (ROPE_DIM / 2u) + pair]);
            float s = float(sin[size_t(row) * (ROPE_DIM / 2u) + pair]);
            value = (rope_dim & 1u) == 0u
                ? value * c + partner * s
                : value * c - partner * s;
        }
        projected_input[element] = T(value);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_inverse_rope_wo_input",
        input_names=["o", "cos", "sin", "rows"],
        output_names=["projected_input"],
        source=source,
        ensure_row_contiguous=True,
    )


class _MiaInverseRopeGatherOLora:
    """Fused inverse-RoPE producer feeding native MXFP8 WO-A/WO-B."""

    __slots__ = ("inverse_rope", "projection")

    def __init__(self, attention: "DeepseekV4Attention", quant: tuple) -> None:
        if (
            attention.n_heads,
            attention.head_dim,
            attention.rope_head_dim,
            attention.n_groups,
        ) != (64, 512, 64, 8):
            raise ValueError("Mia fused WO route requires 64x512 heads in 8 groups")
        if (int(quant[3]), int(quant[4]), str(quant[5])) != (32, 8, "mxfp8"):
            raise ValueError("Mia fused WO route requires native group-32 MXFP8")
        self.inverse_rope = _mia_inverse_rope_output_kernel()
        self.projection = _DirectGatherOLora(attention, quant)

    def __call__(self, o: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        batch, sequence, heads, head_dim = (int(value) for value in o.shape)
        rows = batch * sequence
        elements = rows * heads * head_dim
        (projected_input,) = self.inverse_rope(
            inputs=[
                mx.contiguous(o),
                mx.contiguous(cos),
                mx.contiguous(sin),
                rows,
            ],
            template=[("T", o.dtype)],
            grid=((elements + 255) // 256 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, sequence, heads * head_dim)],
            output_dtypes=[o.dtype],
        )
        return self.projection(projected_input)


class _DirectGatherOLoraWideM4:
    """Construction-bound M4-wide body route plus explicit stock-width routes.

    The canonical body stores eight output-LoRA matrices.  At physical M4 the
    wide entry point owns a threadgroup per stored group and streams one packed
    weight / scale / bias row through all four verifier rows.  Other physical
    widths are deliberately the already-qualified :class:`_DirectGatherOLora`
    route; width is the only value that varies at execution and this is routing,
    not an eligibility check or a fallback.
    """

    __slots__ = ("m4", "stock")

    def __init__(
        self,
        attention: "DeepseekV4Attention",
        quant: tuple,
        *,
        activation_dtype,
    ) -> None:
        if activation_dtype != mx.bfloat16:
            raise ValueError(
                "M4-wide gather o-LoRA activation/output dtype must be bfloat16"
            )
        self.stock = _DirectGatherOLora(attention, quant)
        self.m4 = _GatherQMMWideM4OLora(
            attention, quant, activation_dtype=activation_dtype
        )

    def __call__(self, o: mx.array) -> mx.array:
        batch, sequence, _ = o.shape
        if batch * sequence == 4:
            return self.m4(o)
        return self.stock(o)


class _GatherQMMWideM4OLora:
    """Fixed ``[8, 4, 4096]`` gathered affine-Q4 projection.

    The source is derived for the actual o-LoRA packing, not copied from a
    topology match: logical group ``g`` reads activation ``[row, g, :]`` and
    weight/scale/bias bank ``rhs_ids[g]``.  Every packed nibble is affine
    dequantized once before accumulating all four rows, matching the stock Q4
    association and its eight-lane K ownership.  A construction self-check
    against ``mx.gather_qmm`` is required before this route is published.
    """

    __slots__ = (
        "biases",
        "bits",
        "group_ids",
        "group_size",
        "groups",
        "kernel",
        "mode",
        "per_group_input",
        "rank",
        "scales",
        "weight",
        "wo_b",
    )

    def __init__(
        self,
        attention: "DeepseekV4Attention",
        quant: tuple,
        *,
        activation_dtype,
    ) -> None:
        weight, scales, biases, group_size, bits, mode = quant
        if activation_dtype != mx.bfloat16:
            raise ValueError(
                "M4-wide gather o-LoRA activation/output dtype must be bfloat16"
            )
        if getattr(weight, "dtype", None) != mx.uint32:
            raise ValueError("M4-wide gather o-LoRA packed weight dtype must be uint32")
        if getattr(scales, "dtype", None) != mx.bfloat16:
            raise ValueError("M4-wide gather o-LoRA scales dtype must be bfloat16")
        if biases is None or getattr(biases, "dtype", None) != mx.bfloat16:
            raise ValueError("M4-wide gather o-LoRA biases dtype must be bfloat16")
        groups = int(attention.n_groups)
        rank = int(attention.o_lora_rank)
        per_group_input = int(
            attention.n_heads * attention.head_dim // attention.n_groups
        )
        if (groups, rank, per_group_input, int(group_size), int(bits), mode) != (
            8,
            1024,
            4096,
            64,
            4,
            "affine",
        ):
            raise ValueError("M4-wide gather o-LoRA requires canonical body geometry")
        if tuple(weight.shape) != (groups * rank, 512):
            raise ValueError("M4-wide gather o-LoRA packed weight layout changed")
        if tuple(scales.shape) != (groups * rank, 64):
            raise ValueError("M4-wide gather o-LoRA scale layout changed")
        if tuple(biases.shape) != (groups * rank, 64):
            raise ValueError("M4-wide gather o-LoRA bias layout changed")
        self.groups = groups
        self.rank = rank
        self.per_group_input = per_group_input
        self.weight = weight.reshape(groups, rank, -1)
        self.scales = scales.reshape(groups, rank, -1)
        self.biases = biases.reshape(groups, rank, -1)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = mode
        self.group_ids = mx.arange(groups, dtype=mx.uint32)
        self.wo_b = attention.wo_b
        self.kernel = _gather_qmm_wide_m4_olora_kernel()

    def grouped(self, o_rows: mx.array, rhs_ids: mx.array) -> mx.array:
        """Project exactly four row-major o-LoRA rows with selected group banks."""
        (out,) = self.kernel(
            inputs=[
                o_rows,
                self.weight,
                self.scales,
                self.biases,
                rhs_ids,
            ],
            template=[("T", mx.bfloat16)],
            grid=(32, 256, 8),
            threadgroup=(32, 2, 1),
            output_shapes=[(8, 4, 1024)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    def __call__(self, o: mx.array) -> mx.array:
        batch, sequence, _ = o.shape
        rows = o.reshape(4, self.groups, self.per_group_input)
        out = self.grouped(rows, self.group_ids)
        return self.wo_b(
            out.swapaxes(0, 1).reshape(batch, sequence, self.groups * self.rank)
        )


@lru_cache(maxsize=1)
def _gather_qmm_wide_m4_olora_kernel():
    """Build the exact-shape gathered wide kernel at installation, never decode."""

    source = """
        using namespace metal;
        constexpr int M = 4;
        constexpr int GROUPS = 8;
        constexpr int K = 4096;
        constexpr int N = 1024;
        constexpr int GS = 64;
        constexpr int K_LANES = 8;
        constexpr int RESULTS_PER_SIMDGROUP = 32 / K_LANES;
        constexpr int NUM_SIMDGROUPS = 2;
        constexpr int ROWS_PER_TG = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
        constexpr int SUB = 8;

        uint lane = thread_index_in_simdgroup;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        uint lhs_group = threadgroup_position_in_grid.z;
        short k_lane = short(lane % K_LANES);
        short sg_row = short(lane / K_LANES);
        int out_row = int(tg_n) * ROWS_PER_TG
            + RESULTS_PER_SIMDGROUP * int(simd_gid) + int(sg_row);
        int row = min(out_row, N - 1);
        int rhs_group = int(rhs_ids[lhs_group]);
        int K_by_gs = K / GS;
        int K_bytes = K / 2;
        const device uint8_t* wrow = (const device uint8_t*)w
            + (rhs_group * N + row) * K_bytes;
        const device T* srow = scales + (rhs_group * N + row) * K_by_gs;
        const device T* brow = biases + (rhs_group * N + row) * K_by_gs;

        float result[M] = {0.0f};
        for (int g = int(k_lane); g < K_by_gs; g += K_LANES) {
            float scale = float(srow[g]);
            float bias = float(brow[g]);
            float scaled_hi = scale / 16.0f;
            _Pragma("unroll")
            for (int sc = 0; sc < GS / SUB; ++sc) {
                int k0 = g * GS + sc * SUB;
                const device uint8_t* wc = wrow + k0 / 2;
                float w_dq[SUB];
                w_dq[0] = scale * float(wc[0] & 0x0f) + bias;
                w_dq[1] = scaled_hi * float(wc[0] & 0xf0) + bias;
                w_dq[2] = scale * float(wc[1] & 0x0f) + bias;
                w_dq[3] = scaled_hi * float(wc[1] & 0xf0) + bias;
                w_dq[4] = scale * float(wc[2] & 0x0f) + bias;
                w_dq[5] = scaled_hi * float(wc[2] & 0xf0) + bias;
                w_dq[6] = scale * float(wc[3] & 0x0f) + bias;
                w_dq[7] = scaled_hi * float(wc[3] & 0xf0) + bias;
                _Pragma("unroll")
                for (int v = 0; v < M; ++v) {
                    const device T* xc = x + (v * GROUPS + int(lhs_group)) * K + k0;
                    float acc = 0.0f;
                    _Pragma("unroll")
                    for (int i = 0; i < SUB; ++i) {
                        acc += float(xc[i]) * w_dq[i];
                    }
                    result[v] += acc;
                }
            }
        }
        _Pragma("unroll")
        for (int v = 0; v < M; ++v) {
            result[v] += simd_shuffle_down(result[v], 4);
            result[v] += simd_shuffle_down(result[v], 2);
            result[v] += simd_shuffle_down(result[v], 1);
        }
        if (k_lane == 0 && out_row < N) {
            _Pragma("unroll")
            for (int v = 0; v < M; ++v) {
                y[(int(lhs_group) * M + v) * N + out_row] = T(result[v]);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_olora_gather_qmv_wide_m4_q4_g64",
        input_names=["x", "w", "scales", "biases", "rhs_ids"],
        output_names=["y"],
        source=source,
    )


class _DirectDenseOLora:
    """Prebound dense grouped matmul with no storage/cache decision at execution."""

    __slots__ = ("groups", "per_group_input", "rank", "weight", "wo_b")

    def __init__(self, attention: "DeepseekV4Attention", weight: mx.array) -> None:
        self.groups = int(attention.n_groups)
        self.rank = int(attention.o_lora_rank)
        self.per_group_input = int(
            attention.n_heads * attention.head_dim // attention.n_groups
        )
        self.weight = weight.reshape(
            self.groups, self.rank, self.per_group_input
        )
        self.wo_b = attention.wo_b

    def __call__(self, o: mx.array) -> mx.array:
        batch, sequence, _ = o.shape
        grouped = o.reshape(
            batch, sequence, self.groups, self.per_group_input
        )
        out = mx.einsum("bsgp,grp->bsgr", grouped, self.weight)
        return self.wo_b(out.reshape(batch, sequence, self.groups * self.rank))


class _DirectCachedOLora(_DirectDenseOLora):
    """Known quantized body storage, dequantized and captured exactly once."""

    __slots__ = ()

    def __init__(self, attention: "DeepseekV4Attention", quant: tuple) -> None:
        weight, scales, biases, group_size, bits, mode = quant
        dense = mx.dequantize(
            weight,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        super().__init__(attention, dense)
        mx.eval(self.weight)
        attention._wo_a_cache.put((weight, scales, biases), self.weight)


class _DirectDenseMTPOLora(_DirectDenseOLora):
    """Known dense-BF16 MTP stock math, captured without a quantized lookup."""

    __slots__ = ()


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_hash_layers: int = 3
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    # moe
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    # attention (MQA-shaped MLA)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    window_size: int = 128
    sliding_window: int = 128
    # index / compressed-sparse attention
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_ratios: List[int] = field(default_factory=lambda: list(_DEFAULT_COMPRESS_RATIOS))
    compress_rope_theta: float = 160000.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # norm / rope / yarn
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576
    rope_scaling: Optional[dict] = None
    # yarn (flattened from rope_scaling for convenience; overridden in __post_init__)
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # multi-token prediction (draft head).  DeepSeek-V4-Flash ships one MTP block
    # upstream as ``mtp.0.*``; a conversion that drops it leaves this field at 1
    # while shipping no weights, which :meth:`Model.sanitize` detects and honours.
    num_nextn_predict_layers: int = 0
    # Mia/DeepSpec DSpark signature.  A populated signature installs the native
    # fixed-K5 owner at construction; target-only and legacy one-block artifacts
    # leave every field unset and retain the existing path.
    dspark_block_size: Optional[int] = None
    dspark_markov_rank: Optional[int] = None
    dspark_noise_token_id: Optional[int] = None
    dspark_target_layer_ids: Optional[List[int]] = None
    hybrid_tr3_tail: Optional[dict] = None

    def __post_init__(self):
        # Accept the HF rope_scaling block and mirror it into the flat YaRN fields
        # the reference precompute uses.
        rs = self.rope_scaling or {}
        if rs:
            self.original_seq_len = int(
                rs.get("original_max_position_embeddings", self.original_seq_len)
            )
            self.rope_factor = float(rs.get("factor", self.rope_factor))
            self.beta_fast = int(rs.get("beta_fast", self.beta_fast))
            self.beta_slow = int(rs.get("beta_slow", self.beta_slow))
        # window_size / sliding_window are the same knob under two names.
        self.window_size = int(self.sliding_window or self.window_size)
        if self.hybrid_tr3_tail is not None:
            metadata = self.hybrid_tr3_tail
            observed = (
                metadata.get("format"),
                float(metadata.get("bits", 0.0)),
                metadata.get("codebook"),
                int(metadata.get("tp", 0)),
                int(metadata.get("experts_per_layer", 0)),
                tuple(metadata.get("moe_layers", ())),
                metadata.get("exllamav3_revision"),
                metadata.get("source_revision"),
            )
            expected = (
                "exl3-trellis",
                3.0,
                "mcg",
                1,
                int(self.n_routed_experts),
                (0, int(self.num_hidden_layers) - 1),
                "787d1582267117d6ee83c90014f03b525b14754f",
                "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
            )
            if observed != expected:
                raise ValueError(
                    "unsupported DeepSeek V4 EXL3 target contract: "
                    f"observed={observed!r}, expected={expected!r}"
                )


# ---------------------------------------------------------------------------
# RoPE (YaRN, interleaved / "traditional") — matches reference precompute_freqs_cis
# + apply_rotary_emb, which rope only the last ``rope_head_dim`` dims of q/kv as
# complex pairs (x0+ix1, x2+ix3, ...).
# ---------------------------------------------------------------------------
def _yarn_inv_freq(
    dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> mx.array:
    """Per-(pair) inverse frequencies with the reference's YaRN interpolation ramp.

    Mirrors ``precompute_freqs_cis`` (model.py L199-229): standard inv-freq, then when
    ``original_seq_len > 0`` a smooth linear ramp blends the ``/factor`` (interpolated)
    and un-interpolated frequencies between the beta_fast/beta_slow correction dims.
    """
    half = dim // 2
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len and original_seq_len > 0:
        def correction_dim(num_rot):
            return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (mx.arange(half, dtype=mx.float32) - low) / (high - low)
        ramp = mx.clip(ramp, 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs  # [half]


class MiaRoPETableProvider:
    """One exact-Mia FP32 RoPE graph owner for the active target chunk.

    Token attention and compressor windows deliberately retain their pinned
    source operation orders.  Repeated requests for the same logical slice
    return the same MLX arrays, so every applicable target layer consumes one
    shared graph instead of rebuilding identical trigonometry per layer.
    """

    __slots__ = (
        "inv_freq",
        "max_positions",
        "_token_tables",
        "_window_tables",
    )

    def __init__(self, inv_freq: mx.array, *, max_positions: int) -> None:
        max_positions = int(max_positions)
        if max_positions <= 0:
            raise ValueError("Mia RoPE capacity must be positive")
        if inv_freq.ndim != 1 or int(inv_freq.shape[0]) != 32:
            raise ValueError("Mia RoPE frequencies must be one FP32 32-wide row")
        self.inv_freq = mx.contiguous(inv_freq.astype(mx.float32))
        self.max_positions = max_positions
        self._token_tables: dict[
            tuple[int, int], tuple[mx.array, mx.array, mx.array]
        ] = {}
        self._window_tables: dict[
            tuple[int, int, int], tuple[mx.array, mx.array]
        ] = {}
        mx.eval(self.inv_freq)

    def begin_forward(self) -> None:
        """Drop the previous target chunk's graph references once."""
        self._token_tables.clear()
        self._window_tables.clear()

    def token_tables(
        self,
        start: int,
        count: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return shared token positions/cos/sin in the original integer order."""
        key = (int(start), int(count))
        cached = self._token_tables.get(key)
        if cached is not None:
            return cached
        start, count = key
        if start < 0 or count < 0 or start + count > self.max_positions:
            raise ValueError(
                "Mia token RoPE slice exceeds the installed 384k capacity"
            )
        positions = mx.arange(start, start + count, dtype=mx.int32)
        angles = positions[:, None].astype(mx.float32) * self.inv_freq[None, :]
        tables = (positions, mx.cos(angles), mx.sin(angles))
        self._token_tables[key] = tables
        return tables

    def window_tables(
        self,
        first_window: int,
        count: int,
        *,
        compress_ratio: int,
    ) -> tuple[mx.array, mx.array]:
        """Return shared compressor cos/sin in the pinned FP32 window order."""
        key = (int(first_window), int(count), int(compress_ratio))
        cached = self._window_tables.get(key)
        if cached is not None:
            return cached
        first_window, count, compress_ratio = key
        if first_window < 0 or count < 0 or compress_ratio <= 0:
            raise ValueError("Mia compressor RoPE slice geometry is invalid")
        if count and (first_window + count - 1) * compress_ratio >= self.max_positions:
            raise ValueError(
                "Mia compressor RoPE slice exceeds the installed 384k capacity"
            )
        positions = (
            mx.arange(count, dtype=mx.float32) + float(first_window)
        ) * compress_ratio
        angles = positions[:, None] * self.inv_freq[None, :]
        tables = (mx.cos(angles), mx.sin(angles))
        self._window_tables[key] = tables
        return tables


class MiaStackedMXFP8Projection:
    """One source-faithful MXFP8 GEMM for two row-adjacent projections.

    Mia stores these projections as separate checkpoint names, but both consume
    the same activation with identical native MXFP8 geometry.  The owner keeps
    one concatenated payload while rebinding the named modules to row views, so
    parameter naming survives the loader boundary without retaining a second
    copy of either matrix.
    """

    __slots__ = (
        "weight",
        "scales",
        "split",
        "group_size",
        "bits",
        "mode",
    )

    @staticmethod
    def validate_pair(first: nn.Module, second: nn.Module) -> None:
        expected = (32, 8, "mxfp8", mx.uint32, mx.uint8)
        for label, module in (("first", first), ("second", second)):
            observed = (
                int(getattr(module, "group_size", -1)),
                int(getattr(module, "bits", -1)),
                str(getattr(module, "mode", "")),
                getattr(getattr(module, "weight", None), "dtype", None),
                getattr(getattr(module, "scales", None), "dtype", None),
            )
            if not isinstance(module, nn.QuantizedLinear) or observed != expected:
                raise ValueError(
                    f"Mia stacked {label} projection changed: "
                    f"{observed!r} != {expected!r}"
                )
            if getattr(module, "bias", None) is not None or getattr(
                module, "biases", None
            ) is not None:
                raise ValueError("Mia stacked projections must be bias-free")
        if (
            first.weight.ndim != 2
            or second.weight.ndim != 2
            or int(first.weight.shape[1]) != int(second.weight.shape[1])
            or first.scales.ndim != 2
            or second.scales.ndim != 2
            or int(first.scales.shape[1]) != int(second.scales.shape[1])
            or int(first.weight.shape[0]) != int(first.scales.shape[0])
            or int(second.weight.shape[0]) != int(second.scales.shape[0])
        ):
            raise ValueError("Mia stacked MXFP8 row geometry changed")

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        self.validate_pair(first, second)
        self.split = int(first.scales.shape[0])
        self.group_size = 32
        self.bits = 8
        self.mode = "mxfp8"
        self.weight = mx.contiguous(
            mx.concatenate((first.weight, second.weight), axis=0)
        )
        self.scales = mx.contiguous(
            mx.concatenate((first.scales, second.scales), axis=0)
        )
        mx.eval(self.weight, self.scales)

        # Preserve the original checkpoint/source names as views into the one
        # construction-owned payload.  No duplicate full matrix remains live.
        first.weight = self.weight[: self.split]
        first.scales = self.scales[: self.split]
        second.weight = self.weight[self.split :]
        second.scales = self.scales[self.split :]

    def project_fused(self, values: mx.array) -> mx.array:
        """Return the single fused output buffer for downstream fused consumers."""
        return mx.quantized_matmul(
            values,
            self.weight,
            scales=self.scales,
            biases=None,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def split_fused(self, projected: mx.array) -> tuple[mx.array, mx.array]:
        """Expose the fixed query/KV row ranges as zero-copy slice views."""
        return (
            projected[..., : self.split],
            projected[..., self.split :],
        )

    def __call__(self, values: mx.array) -> tuple[mx.array, mx.array]:
        return self.split_fused(self.project_fused(values))


class MiaStackedDenseProjection:
    """One construction-widened GEMM for source FP32 compressor projections."""

    __slots__ = ("weight", "split")

    @staticmethod
    def validate_pair(first: nn.Module, second: nn.Module) -> None:
        for label, module in (("first", first), ("second", second)):
            if (
                not isinstance(module, nn.Linear)
                or getattr(module, "bias", None) is not None
                or module.weight.ndim != 2
                or module.weight.dtype != mx.bfloat16
            ):
                raise ValueError(
                    f"Mia stacked dense {label} projection changed"
                )
        if int(first.weight.shape[1]) != int(second.weight.shape[1]):
            raise ValueError("Mia stacked dense input geometry changed")

    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        self.validate_pair(first, second)
        self.split = int(first.weight.shape[0])
        self.weight = mx.contiguous(
            mx.concatenate((first.weight, second.weight), axis=0).astype(
                mx.float32
            )
        )
        mx.eval(self.weight)
        first.weight = self.weight[: self.split]
        second.weight = self.weight[self.split :]

    def project_fused(self, values: mx.array) -> mx.array:
        """Return the single fused compressor output buffer."""
        return mx.matmul(values, mx.swapaxes(self.weight, -1, -2))

    def split_fused(self, projected: mx.array) -> tuple[mx.array, mx.array]:
        """Expose the fixed KV/gate row ranges as zero-copy slice views."""
        return (
            projected[..., : self.split],
            projected[..., self.split :],
        )

    def __call__(self, values: mx.array) -> tuple[mx.array, mx.array]:
        return self.split_fused(self.project_fused(values))


def install_mia_stacked_projections(model) -> dict[str, int]:
    """Bind every exact-Mia dense pair without whole-model materialization."""
    if getattr(model, "_mia_stacked_projection_receipt", None) is not None:
        raise ValueError("the Mia stacked projection owners are already installed")

    layers = tuple(model.layers)
    stages = tuple(model.dspark.stages)
    target_attention = tuple(layer.attn for layer in layers)
    draft_attention = tuple(stage.attn for stage in stages)
    shared_expert = tuple(
        layer.ffn.shared_experts for layer in layers + stages
    )
    main_compressor = tuple(
        attention.compressor
        for attention in target_attention
        if int(attention.compress_ratio) > 0
    )
    indexer_compressor = tuple(
        attention.indexer.compressor
        for attention in target_attention
        if int(attention.compress_ratio) == 4
    )
    receipt = {
        "target_attention": len(target_attention),
        "draft_attention": len(draft_attention),
        "shared_expert": len(shared_expert),
        "main_compressor": len(main_compressor),
        "indexer_compressor": len(indexer_compressor),
    }
    expected = {
        "target_attention": 43,
        "draft_attention": 3,
        "shared_expert": 46,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }
    if receipt != expected:
        raise ValueError(
            f"Mia stacked projection topology changed: {receipt!r} != {expected!r}"
        )

    specs = tuple(
        (
            MiaStackedMXFP8Projection,
            attention.wq_a,
            attention.wkv,
            attention.install_mia_stacked_projection,
        )
        for attention in target_attention + draft_attention
    ) + tuple(
        (
            MiaStackedMXFP8Projection,
            expert.gate_proj,
            expert.up_proj,
            expert.install_mia_stacked_gate_up,
        )
        for expert in shared_expert
    ) + tuple(
        (
            MiaStackedDenseProjection,
            compressor.wkv,
            compressor.wgate,
            compressor.install_mia_stacked_projection,
        )
        for compressor in main_compressor + indexer_compressor
    )
    # Prove the entire graph before mutating any module.  Construction then
    # evaluates and releases one pair at a time, bounding transient ownership.
    for owner_type, first, second, _bind in specs:
        owner_type.validate_pair(first, second)
    for owner_type, first, second, bind in specs:
        bind(owner_type(first, second))
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

    model._mia_stacked_projection_receipt = receipt
    return receipt


def _mia_qkv_attentions(model) -> tuple[tuple[object, ...], tuple[object, ...]]:
    target = tuple(layer.attn for layer in model.layers)
    draft = tuple(stage.attn for stage in model.dspark.stages)
    if len(target) != 43 or len(draft) != 3:
        raise ValueError("Mia Q/KV route requires 43 target and three draft attentions")
    if len({id(attention) for attention in target + draft}) != 46:
        raise ValueError("Mia Q/KV attention ownership is not one-to-one")
    return target, draft


def mia_qkv_prologue_receipt(model) -> dict:
    """Return the construction seal for all exact-Mia Q/KV prologues."""
    from mtplx.kernels.deepseek_v4_qkv_prologue import MiaBoundQKVPrologue

    target, draft = _mia_qkv_attentions(model)
    plans = tuple(
        getattr(attention, "_mia_qkv_plan", None)
        for attention in target + draft
    )
    if any(type(plan) is not MiaBoundQKVPrologue for plan in plans):
        raise ValueError("the exact Mia Q/KV plan is absent or changed")
    for index, (attention, plan) in enumerate(
        zip(target + draft, plans, strict=True)
    ):
        if not isinstance(plan.projection_owner, MiaStackedMXFP8Projection):
            raise ValueError(f"Mia Q/KV plan {index} lost its fused projection owner")
        if plan.projection_owner is not attention._mia_input_projection:
            raise ValueError(f"Mia Q/KV plan {index} projection owner changed")
        if plan.q_weight is not attention.q_norm.weight:
            raise ValueError(f"Mia Q/KV plan {index} query weight changed")
        if plan.kv_weight is not attention.kv_norm.weight:
            raise ValueError(f"Mia Q/KV plan {index} KV weight changed")
        if plan.raw.geometry != {
            "q_rank": 1024,
            "heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "target_decode_rows": (1, 6),
            "proposal_rows": 5,
            "context_rows": 128,
            "prefill_tile_rows": 1024,
        }:
            raise ValueError(f"Mia Q/KV plan {index} geometry changed")
    plan_ids = tuple(id(plan) for plan in plans)
    callable_ids = tuple(
        tuple(
            id(getattr(plan, name))
            for name in (
                "project_learned",
                "project_kv",
                "target_records",
                "prefill_records",
                "proposal_records",
                "context_records",
            )
        )
        for plan in plans
    )
    if len(set(plan_ids)) != 46 or len(set(callable_ids)) != 46:
        raise ValueError("Mia Q/KV plans and bound callables must be one-to-one")
    return {
        "route": "mia_fused_qkv_stock432",
        "target_attention": len(target),
        "draft_attention": len(draft),
        "plan_count": len(plans),
        "unique_plan_count": len(set(plan_ids)),
        "plan_type": "MiaBoundQKVPrologue",
        "prefill_cutoff": 1024,
        "proposal_rows": 5,
        "context_rows": 128,
        "plan_ids": plan_ids,
        "callable_ids": callable_ids,
        "q_weight_ids": tuple(id(plan.q_weight) for plan in plans),
        "kv_weight_ids": tuple(id(plan.kv_weight) for plan in plans),
        "projection_owner_ids": tuple(id(plan.projection_owner) for plan in plans),
    }


def install_mia_qkv_prologue_routes(model) -> dict:
    """Bind one exact fused Q/KV plan to each target and draft attention."""
    from mtplx.kernels.deepseek_v4_qkv_prologue import (
        bind_mia_qkv_prologue,
        install_mia_qkv_prologue,
    )

    if getattr(model, "_mia_qkv_prologue_receipt", None) is not None:
        raise ValueError("the Mia Q/KV prologue routes are already installed")
    target, draft = _mia_qkv_attentions(model)
    attentions = target + draft
    plans = []
    for attention in attentions:
        owner = getattr(attention, "_mia_input_projection", None)
        if not isinstance(owner, MiaStackedMXFP8Projection):
            raise ValueError("Mia Q/KV installation requires stacked MXFP8 owners")
        raw = install_mia_qkv_prologue(
            q_rank=attention.q_lora_rank,
            heads=attention.n_heads,
            head_dim=attention.head_dim,
            rope_dim=attention.rope_head_dim,
            proposal_rows=5,
            context_rows=128,
            prefill_tile_rows=1024,
        )
        plans.append(
            bind_mia_qkv_prologue(
                raw,
                projection_owner=owner,
                q_weight=attention.q_norm.weight,
                kv_weight=attention.kv_norm.weight,
                rms_eps=attention.eps,
            )
        )
    for attention, plan in zip(attentions, plans, strict=True):
        attention.install_mia_qkv_prologue(plan)
    receipt = mia_qkv_prologue_receipt(model)
    model._mia_qkv_prologue_receipt = receipt
    return receipt


def install_mia_target_rope_providers(
    model,
    *,
    physical_max_positions: int,
) -> tuple[MiaRoPETableProvider, MiaRoPETableProvider]:
    """Bind exact-Mia RoPE owners with fixed physical M6 tail headroom."""
    if (
        getattr(model, "_mia_base_rope_provider", None) is not None
        or getattr(model, "_mia_compress_rope_provider", None) is not None
        or getattr(model, "_mia_draft_rope_provider", None) is not None
    ):
        raise ValueError("the shared Mia target RoPE providers are already installed")
    args = model.args
    physical_max_positions = int(physical_max_positions)
    base = MiaRoPETableProvider(
        _yarn_inv_freq(
            int(args.qk_rope_head_dim),
            float(args.rope_theta),
            0,
            1.0,
            32,
            1,
        ),
        max_positions=physical_max_positions,
    )
    compress = MiaRoPETableProvider(
        _yarn_inv_freq(
            int(args.qk_rope_head_dim),
            float(args.compress_rope_theta),
            int(args.original_seq_len),
            float(args.rope_factor),
            int(args.beta_fast),
            int(args.beta_slow),
        ),
        max_positions=physical_max_positions,
    )
    # DFlash always verifies one target row plus all five draft rows before
    # slicing a final partial block.  Target and draft providers therefore own
    # the same physical positions while request admission remains logical.
    draft = MiaRoPETableProvider(
        base.inv_freq,
        # ``max_positions`` is an exclusive bound; six rows beginning at
        # position 383999 require storage through inclusive position 384004.
        max_positions=physical_max_positions,
    )
    for layer in model.layers:
        provider = base if int(layer.attn.compress_ratio) == 0 else compress
        layer.attn.install_mia_rope_provider(provider)
    for stage in model.dspark.stages:
        stage.attn.install_mia_rope_provider(draft)
    model._mia_base_rope_provider = base
    model._mia_compress_rope_provider = compress
    model._mia_draft_rope_provider = draft
    return base, compress


def _hadamard_rotate(x: mx.array) -> mx.array:
    """Normalised Walsh-Hadamard rotation of the last axis (power-of-two width).

    Reference ``rotate_activation`` (model.py L247-251) calls
    ``fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)``; this is the same
    map written as the in-place butterfly ds4.c uses
    (``dsv4_hadamard128_inplace_cpu``, antirez/DwarfStar4, MIT), which is bit-for-bit
    the reference's own accumulation order.

    ``H/sqrt(d)`` is **orthogonal**, so it leaves every ``q·k`` the indexer forms
    invariant — see :class:`Indexer` for why it is applied anyway.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard rotation needs a power-of-two width, got {n}")
    y = x.reshape(-1, n)
    stride = 1
    while stride < n:
        y = y.reshape(-1, n // (2 * stride), 2, stride)
        a = y[:, :, 0]
        b = y[:, :, 1]
        y = mx.stack([a + b, a - b], axis=2).reshape(-1, n)
        stride *= 2
    return (y * (n ** -0.5)).reshape(x.shape)


def _topk_mask(key: mx.array, k_row: mx.array, k_max: int) -> mx.array:
    """``True`` for the ``k_row`` largest entries of ``key`` along the last axis.

    ``key`` is ``[..., n]``; ``k_row`` is a *per-row* count broadcastable to
    ``[..., 1]``; ``k_max`` is any upper bound on it (only used to shrink the sort).

    Ties are broken toward the **lowest index**, which is exactly what ds4.c's
    selection does (``indexer_allowed_decode_one``: scan ascending, take over on a
    strict ``>``).  That matters here: the score of a compressed row is a sum of
    ReLU'd dot products, so rows whose every head is negative all score an exact 0
    and collide.  Without a fixed tie-break the one-shot and streaming paths — whose
    score rows have different lengths — could resolve such a collision differently
    and select different rows.

    ``k_row == 0`` selects nothing; ``k_row >= n`` selects everything.
    """
    n = key.shape[-1]
    if k_max <= 0:
        return mx.zeros(key.shape, dtype=mx.bool_)
    if k_max >= n:
        ranked = mx.sort(key, axis=-1)[..., ::-1]
    else:
        ranked = mx.sort(mx.topk(key, k_max, axis=-1), axis=-1)[..., ::-1]
    kth = mx.clip(k_row - 1, 0, ranked.shape[-1] - 1)
    thr = mx.take_along_axis(ranked, kth, axis=-1)          # k_row-th largest
    gt = key > thr
    eq = key == thr
    n_gt = mx.sum(gt.astype(mx.int32), axis=-1, keepdims=True)
    tie_rank = mx.cumsum(eq.astype(mx.int32), axis=-1) - 1  # rank among equals, index order
    return gt | (eq & (tie_rank < (k_row - n_gt)))


def _apply_interleaved_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the last dim of ``x`` (size 2*half) as interleaved complex pairs.

    ``x`` is ``[..., 2*half]``; ``cos``/``sin`` are ``[..., half]`` (broadcastable).
    Pair p = (x[2p], x[2p+1]) -> (x0*cos - x1*sin, x0*sin + x1*cos), matching
    ``apply_rotary_emb`` (model.py L232-244, forward direction).  The inverse
    (de-rotation applied to the attention output) uses cos, -sin.

    **Dtype.**  The rotation is computed in fp32 (``cos``/``sin`` are fp32, so the
    products promote) and *stored back at the input's dtype*, which is precisely
    what the reference does: ``apply_rotary_emb`` rotates ``x.float()`` and then
    ``y.copy_(x)`` into the caller's own bf16 tensor (L234/L243).  Returning fp32
    instead would promote whatever the caller concatenates it with — the roped q,
    the roped per-position KV and the compressor's roped rows all feed tensors the
    rest of the layer is supposed to keep at the model dtype, and one fp32 column
    is enough to drag the KV cache, the attention matmuls, the o-LoRA einsum and
    then the whole residual stream up with it.  ``_FP32_ACTIVATIONS`` restores the
    promoting behaviour for A/B.
    """
    shape = x.shape
    out_dtype = _store_dtype(x.dtype)
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    out = mx.stack([r0, r1], axis=-1)
    return out.reshape(shape).astype(out_dtype)


# ---------------------------------------------------------------------------
# Hyper-Connections
# ---------------------------------------------------------------------------
def _sinkhorn_ops(comb: mx.array, iters: int, eps: float) -> mx.array:
    """The Sinkhorn alternating-normalisation loop as stock MLX ops.

    ``comb`` is the *pre*-softmax ``[..., hc, hc]`` matrix (last axis ``k`` = the
    softmax/row axis, ``axis=-2`` ``j`` = the column axis).  This is the oracle the
    :func:`_sinkhorn_kernel_apply` Metal kernel is gated bit-identical against and
    the path both :func:`hc_split_sinkhorn` and :func:`_hc_pre_impl` take when the
    kernel is off; it is exactly the loop these two functions used to inline.
    """
    comb = mx.softmax(comb, axis=-1) + eps                       # row-softmax
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)       # column normalise
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)   # row normalise
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)   # column normalise
    return comb


#: One compiled ``mx.fast.metal_kernel`` per ``(hc, iters, eps)`` structural triple.
_SINKHORN_KERNELS: dict = {}


def _sinkhorn_metal_kernel(hc: int, iters: int, eps: float):
    """Build (and cache) the Sinkhorn kernel for one structural triple.

    One thread owns one ``[hc, hc]`` matrix, loads its ``hc*hc`` floats into a
    register array, and runs the *entire* normalisation schedule in registers:
    the row-softmax, the first column-normalise, then ``iters-1`` alternating
    row/column normalises.  Every arithmetic step is the same fp32 op in the same
    order as :func:`_sinkhorn_ops` (max-stable ``exp``/sum/divide for the softmax;
    ``sum + eps`` denominators for each normalise), so the result is the same real
    number to fp32 rounding.  ``hc``/``iters``/``eps`` are baked in as compile-time
    constants, so the register array is fixed-size and the loops unroll.
    """
    key = (int(hc), int(iters), float(eps))
    kern = _SINKHORN_KERNELS.get(key)
    if kern is not None:
        return kern
    n = hc * hc
    # eps to full fp32 precision as a Metal float literal (matches the fp32 value
    # ``comb + eps`` used in _sinkhorn_ops, where the python double is cast to fp32).
    eps_lit = f"{float(eps):.9e}f"
    source = f"""
        using namespace metal;
        constexpr uint HC = {hc};
        constexpr uint N = {n};
        constexpr uint ITERS = {iters};
        constexpr float EPS = {eps_lit};

        uint gid = thread_position_in_grid.x;
        if (gid >= nmat) {{ return; }}
        const uint off = gid * N;

        float c[N];
        for (uint i = 0; i < N; ++i) {{ c[i] = comb[off + i]; }}

        // row-softmax over the last axis k (row i = c[i*HC + k]), then + EPS
        for (uint i = 0; i < HC; ++i) {{
            float m = c[i * HC];
            for (uint k = 1; k < HC; ++k) {{ m = metal::max(m, c[i * HC + k]); }}
            float s = 0.0f;
            for (uint k = 0; k < HC; ++k) {{
                float e = metal::exp(c[i * HC + k] - m);
                c[i * HC + k] = e;
                s += e;
            }}
            for (uint k = 0; k < HC; ++k) {{ c[i * HC + k] = c[i * HC + k] / s + EPS; }}
        }}

        // column normalise: sum over rows j (axis=-2), divide by (sum + EPS)
        for (uint k = 0; k < HC; ++k) {{
            float cs = 0.0f;
            for (uint j = 0; j < HC; ++j) {{ cs += c[j * HC + k]; }}
            float den = cs + EPS;
            for (uint j = 0; j < HC; ++j) {{ c[j * HC + k] = c[j * HC + k] / den; }}
        }}

        // iters-1 alternating passes: row normalise then column normalise
        for (uint it = 0; it < (ITERS - 1); ++it) {{
            for (uint i = 0; i < HC; ++i) {{      // row normalise (axis=-1)
                float rs = 0.0f;
                for (uint k = 0; k < HC; ++k) {{ rs += c[i * HC + k]; }}
                float den = rs + EPS;
                for (uint k = 0; k < HC; ++k) {{ c[i * HC + k] = c[i * HC + k] / den; }}
            }}
            for (uint k = 0; k < HC; ++k) {{     // column normalise (axis=-2)
                float cs = 0.0f;
                for (uint j = 0; j < HC; ++j) {{ cs += c[j * HC + k]; }}
                float den = cs + EPS;
                for (uint j = 0; j < HC; ++j) {{ c[j * HC + k] = c[j * HC + k] / den; }}
            }}
        }}

        for (uint i = 0; i < N; ++i) {{ out[off + i] = c[i]; }}
    """
    kern = mx.fast.metal_kernel(
        name=f"mtplx_dsv4_sinkhorn_hc{hc}_it{iters}",
        input_names=["comb", "nmat"],
        output_names=["out"],
        source=source,
    )
    _SINKHORN_KERNELS[key] = kern
    return kern


def _sinkhorn_kernel_apply(comb: mx.array, hc: int, iters: int, eps: float) -> mx.array:
    """Run the whole Sinkhorn loop as one Metal dispatch per ``[..., hc, hc]``.

    Flattens the leading dims to one matrix index, launches one thread per matrix,
    and reshapes back.  Bit-identical (1e-6, argmax exact) to :func:`_sinkhorn_ops`
    — see :data:`_SINKHORN_KERNEL`.
    """
    lead = tuple(int(d) for d in comb.shape[:-2])
    nmat = 1
    for d in lead:
        nmat *= d
    flat = comb.reshape(nmat, hc, hc)
    kern = _sinkhorn_metal_kernel(hc, iters, eps)
    (out,) = kern(
        inputs=[flat, nmat],
        grid=(nmat, 1, 1),
        threadgroup=(min(nmat, 256), 1, 1),
        output_shapes=[(nmat, hc, hc)],
        output_dtypes=[comb.dtype],
    )
    return out.reshape(*lead, hc, hc)


def _install_sinkhorn_normaliser(
    hc: int,
    iters: int,
    eps: float,
    *,
    enabled: bool | None = None,
):
    """Install the fixed Sinkhorn route for one Hyper-Connection instance.

    The experimental lane is deliberately selected at model construction, never
    in ``pre``'s token hot path.  CPU/no-Metal is an explicit stock-oracle route;
    a GPU installation is allowed only for DeepSeek-V4-Flash's proven fp32
    ``hc=4, iters=20, eps=1e-6`` geometry.  A forced flag on any other GPU
    geometry fails here, before measured generation, instead of silently taking a
    differently-shaped kernel or falling back.
    """
    def stock(comb: mx.array) -> mx.array:
        return _sinkhorn_ops(comb, iters, eps)

    selected = _SINKHORN_KERNEL if enabled is None else bool(enabled)
    if not selected:
        return False, stock
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False, stock
    if (hc, iters, eps) != (4, 20, 1e-6):
        raise ValueError(
            "MTPLX_DSV4_SINKHORN_KERNEL requires DeepSeek-V4-Flash's "
            f"fp32 hc=4, iters=20, eps=1e-6 lane; got hc={hc}, "
            f"iters={iters}, eps={eps!r}"
        )
    # Build/cache at installation so a Metal-source failure is reported before a
    # measured forward, rather than as a late per-token fallback.
    _sinkhorn_metal_kernel(hc, iters, eps)

    def kernel(comb: mx.array) -> mx.array:
        return _sinkhorn_kernel_apply(comb, hc, iters, eps)

    return True, kernel


def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc: int,
    iters: int,
    eps: float,
):
    """Transcription of ``hc_split_sinkhorn_kernel`` (kernel.py L371-427).

    ``mixes`` is ``[..., (2+hc)*hc]``.  Returns ``(pre, post, comb)`` with shapes
    ``[..., hc]``, ``[..., hc]``, ``[..., hc, hc]``.  ``comb`` is made (approximately)
    doubly-stochastic by one row-softmax + column-normalise, then ``iters-1`` more
    row/column normalisation passes.
    """
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)  # [..., j, k]
    # This standalone reference transcription is intentionally always stock.  A
    # model instance installs its chosen route once in ``HyperConnection``.
    comb = _sinkhorn_ops(comb, iters, eps)
    return pre, post, comb


def _hc_pre_impl(x, fn_t, base, scale_vec, hc: int, iters: int, eps: float, normalise=None):
    """:meth:`HyperConnection.pre` as one pure function of arrays.

    Identical arithmetic to ``_mixes`` + :func:`hc_split_sinkhorn` + the weighted
    sum, in the same order, and therefore bit-identical to them (gated by
    tests/test_deepseek_v4_hc_compile.py).  Two structural differences, both of
    which only remove primitives:

    * ``fn_t``/``base``/``scale_vec`` arrive already fp32 and already transposed
      / already expanded to one weight per mix column, so the per-call
      ``astype``, ``.T`` and six parameter slices are gone.  All four are pure
      functions of the parameters, so they are derived once (see
      :meth:`HyperConnection._static`).  ``scale_vec`` repeats ``scale[0]`` over
      the ``pre`` columns, ``scale[1]`` over the ``post`` columns and
      ``scale[2]`` over the ``comb`` block, which is exactly the scalar each
      column was multiplied by before.
    * The three affine transforms become one ``mixes * scale_vec + base`` over
      the whole ``[..., (2+hc)*hc]`` row, then sliced — the same multiply and add
      per element.

    Kept a module-level function taking arrays only so ``mx.compile`` can cache
    one tape across all ``2 * n_layers + 1`` Hyper-Connection modules: they share
    every shape and differ only in weight *values*, which are inputs.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32)
    x_flat = xf.reshape(*xf.shape[:-2], -1)
    rsqrt = mx.rsqrt(mx.mean(mx.square(x_flat), axis=-1, keepdims=True) + eps)
    t = ((x_flat @ fn_t) * rsqrt) * scale_vec + base
    pre = mx.sigmoid(t[..., :hc]) + eps
    post = 2.0 * mx.sigmoid(t[..., hc : 2 * hc])
    comb = t[..., 2 * hc :].reshape(*t.shape[:-1], hc, hc)  # [..., j, k]
    if normalise is None:
        def normalise(c):
            return _sinkhorn_ops(c, iters, eps)
    comb = normalise(comb)

    y = mx.sum(pre[..., None] * xf, axis=-2)  # [..., dim]
    return y.astype(dtype), post, comb


def _hc_post_impl(x, residual, post, comb):
    """:meth:`HyperConnection.post` as one pure function of arrays."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    rf = residual.astype(mx.float32)
    term = post[..., None] * xf[..., None, :]           # [..., hc, dim]
    mixed = mx.einsum("...jk,...jd->...kd", comb, rf)   # sum_j comb[j,k] res[j]
    return (term + mixed).astype(dtype)


def _hc_head_impl(x, fn_t, base, scale, eps: float):
    """:class:`HeadHC` as one pure function of arrays."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    x_flat = xf.reshape(*xf.shape[:-2], -1)
    rsqrt = mx.rsqrt(mx.mean(mx.square(x_flat), axis=-1, keepdims=True) + eps)
    mixes = (x_flat @ fn_t) * rsqrt
    pre = mx.sigmoid(mixes * scale + base) + eps
    return mx.sum(pre[..., None] * xf, axis=-2).astype(dtype)


#: One compiled tape per (impl, structural constants) pair.
#:
#: ``mx.compile`` keys its own cache on the *identity* of the function object, so
#: the wrapper has to be built once and reused; rebuilding it per call would
#: retrace every time and leak a cache entry per call.  The structural constants
#: (``hc``, ``iters``, ``eps``) are closed over rather than passed, because they
#: are not arrays and would otherwise be invisible to that cache key.
_HC_COMPILED: dict = {}


def _hc_compiled(kind: str, *consts):
    key = (kind, consts)
    fn = _HC_COMPILED.get(key)
    if fn is None:
        if kind == "pre":
            if len(consts) == 3:
                # Compatibility for existing direct callers of the shared tape
                # helper.  Model instances pass the installed bool below; these
                # test-only callers recreate the same route selection once while
                # building a tape, never from a token hot path.
                hc, iters, eps = consts
                sinkhorn_kernel, _ = _install_sinkhorn_normaliser(hc, iters, eps)
            else:
                hc, iters, eps, sinkhorn_kernel = consts

            if sinkhorn_kernel:
                def normalise(comb):
                    return _sinkhorn_kernel_apply(comb, hc, iters, eps)
            else:
                def normalise(comb):
                    return _sinkhorn_ops(comb, iters, eps)

            def impl(x, fn_t, base, scale_vec):
                return _hc_pre_impl(
                    x, fn_t, base, scale_vec, hc, iters, eps, normalise
                )
        elif kind == "post":
            impl = _hc_post_impl
        elif kind == "head":
            (eps,) = consts

            def impl(x, fn_t, base, scale):
                return _hc_head_impl(x, fn_t, base, scale, eps)
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown Hyper-Connection kernel {kind!r}")
        fn = mx.compile(impl)
        _HC_COMPILED[key] = fn
    return fn


def _hc_use_compile(x: mx.array) -> bool:
    """Is ``x`` in the shape regime the compiled tape is kept for?

    See :data:`_HC_COMPILE_MAX_ROWS`.  Read through the module globals rather
    than captured, so tests (and an operator) can flip either knob after import.
    """
    if not _HC_COMPILE:
        return False
    rows = 1
    for d in x.shape[:-2]:
        rows *= int(d)
    return rows <= _HC_COMPILE_MAX_ROWS


class HyperConnection(nn.Module):
    """Holds a block's ``{fn, base, scale}`` HC parameters and applies pre/post.

    ``fn``: ``[(2+hc)*hc, hc*dim]``   ``base``: ``[(2+hc)*hc]``   ``scale``: ``[3]``.
    Checkpoint keys: ``model.layers.{i}.{attn_hc,ffn_hc}.{fn,base,scale}``.
    """

    def __init__(self, dim: int, hc: int, eps: float, iters: int = 20):
        super().__init__()
        self.dim = dim
        self.hc = hc
        self.eps = eps
        self._iters = iters
        self._sinkhorn_kernel, self._sinkhorn_normalise = _install_sinkhorn_normaliser(
            hc, iters, eps
        )
        mix_hc = (2 + hc) * hc
        self.fn = mx.zeros((mix_hc, hc * dim))
        self.base = mx.zeros((mix_hc,))
        self.scale = mx.zeros((3,))
        # Derived-from-parameters, so a plain object (see _DerivedCache).
        self._static_cache = _DerivedCache()

    def _mixes(self, x: mx.array) -> mx.array:
        # x: [..., hc, dim]
        x_flat = x.reshape(*x.shape[:-2], self.hc * self.dim).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(mx.square(x_flat), axis=-1, keepdims=True) + self.eps)
        return (x_flat @ self.fn.astype(mx.float32).T) * rsqrt

    def _static(self):
        """``(fn.T, base, scale_vec)`` in fp32, derived once from the parameters.

        ``fn`` is ``[(2+hc)*hc, hc*dim]`` — 24 x 16384 on DeepSeek-V4-Flash — and
        the eager path cast it to fp32 inside every call, on every one of the
        ``2 * n_layers`` Hyper-Connections, for a value that never changes.  Same
        shape of waste the ``wo_a`` dequant had, one order of magnitude smaller.
        Keyed on the parameter arrays themselves, so ``load_weights``/``update``/
        ``set_dtype`` invalidate it by identity.
        """
        src = (self.fn, self.base, self.scale)
        hit = self._static_cache.get(src)
        if hit is not None:
            return hit
        hc = self.hc
        s = self.scale.astype(mx.float32)
        scale_vec = mx.concatenate(
            [
                mx.broadcast_to(s[0:1], (hc,)),
                mx.broadcast_to(s[1:2], (hc,)),
                mx.broadcast_to(s[2:3], (hc * hc,)),
            ]
        )
        value = (
            self.fn.astype(mx.float32).T,
            self.base.astype(mx.float32),
            scale_vec,
        )
        return self._static_cache.put(src, value)

    def install_sinkhorn_kernel(self) -> None:
        """Bind the proven fixed-geometry Metal normaliser before execution."""

        installed, normalise = _install_sinkhorn_normaliser(
            self.hc,
            self._iters,
            self.eps,
            enabled=True,
        )
        if not installed:
            raise RuntimeError(
                "DeepSeek-V4 Sinkhorn Metal installation requires the GPU lane"
            )
        self._sinkhorn_kernel = installed
        self._sinkhorn_normalise = normalise

    def pre(self, x: mx.array):
        """Collapse the ``hc`` copies to one; return (y[..., dim], post, comb)."""
        fn_t, base, scale_vec = self._static()
        if _hc_use_compile(x):
            impl = _hc_compiled(
                "pre", self.hc, self._iters, self.eps, self._sinkhorn_kernel
            )
            return impl(x, fn_t, base, scale_vec)
        return _hc_pre_impl(
            x,
            fn_t,
            base,
            scale_vec,
            self.hc,
            self._iters,
            self.eps,
            self._sinkhorn_normalise,
        )

    def post(self, x: mx.array, residual: mx.array, post: mx.array, comb: mx.array):
        """Expand one -> ``hc`` copies and re-mix with the residual copies.

        ``x``: ``[..., dim]``  ``residual``: ``[..., hc, dim]``
        ``post``: ``[..., hc]``  ``comb``: ``[..., hc, hc]``  ->  ``[..., hc, dim]``.
        """
        impl = _hc_compiled("post") if _hc_use_compile(residual) else _hc_post_impl
        return impl(x, residual, post, comb)

class HeadHC(nn.Module):
    """Final head hyper-connection collapse (``ParallelHead.hc_head``, model.py L728).

    Simpler than a block HC: ``pre = sigmoid(mixes*scale + base) + eps`` (no Sinkhorn,
    no post/comb), then weighted sum over the ``hc`` copies.  ``fn``: ``[hc, hc*dim]``.
    Checkpoint keys: ``model.hc_head.{fn,base,scale}``.
    """

    def __init__(self, dim: int, hc: int, eps: float):
        super().__init__()
        self.dim = dim
        self.hc = hc
        self.eps = eps
        self.fn = mx.zeros((hc, hc * dim))
        self.base = mx.zeros((hc,))
        self.scale = mx.zeros((1,))
        self._static_cache = _DerivedCache()

    def _static(self):
        """``(fn.T, base, scale)`` in fp32, derived once (see
        :meth:`HyperConnection._static`)."""
        src = (self.fn, self.base, self.scale)
        hit = self._static_cache.get(src)
        if hit is not None:
            return hit
        value = (
            self.fn.astype(mx.float32).T,
            self.base.astype(mx.float32),
            self.scale.astype(mx.float32),
        )
        return self._static_cache.put(src, value)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., hc, dim]
        fn_t, base, scale = self._static()
        if _hc_use_compile(x):
            return _hc_compiled("head", self.eps)(x, fn_t, base, scale)
        return _hc_head_impl(x, fn_t, base, scale, self.eps)


# ---------------------------------------------------------------------------
# Compressed KV pooling (CSA)
# ---------------------------------------------------------------------------
class Compressor(nn.Module):
    """Learned gated pooling of ``compress_ratio`` consecutive tokens into one
    compressed KV row (reference ``Compressor``, model.py L279-377).

    Full-prefill math (``start_pos == 0``, the path the M2/M3 gates exercise):
        kv    = wkv(x_fp32)                                  # [b,s,coff*head_dim]
        score = wgate(x_fp32)
        (drop the trailing ``s % ratio`` remainder), reshape to windows of ``ratio``,
        add the per-window absolute-position embedding ``ape``, softmax the gate over
        the window and take the gated sum; overlapping windows (ratio==4) additionally
        fold in the previous window's second half.  Then RMSNorm, rope the tail
        ``rope_head_dim`` dims with the compressor's (YaRN) frequencies.

    This component emits normalized rows.  The generic route consumes them directly;
    the installed Mia cache owners quantize them at insertion to native ``stock432``
    attention records or 132-byte E4M3 indexer records.

    Two entry points share one pooling core (:meth:`_pool`): :meth:`__call__` pools a
    whole sequence from position 0 (the parity-gated path), :meth:`step` pools
    incrementally against a :class:`CompressorState` frontier for streaming decode.
    """

    def __init__(
        self, args: ModelArgs, compress_ratio: int, head_dim: int, rotate: bool = False
    ):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rms_norm_eps = float(args.rms_norm_eps)
        # rotate=True is the indexer's copy: the reference Hadamard-rotates its pooled
        # rows before FP4-quantising them (model.py L368-370).  Applied in _pool, so
        # the prefill and streaming paths get it from the same place.
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = mx.zeros((compress_ratio, coff * head_dim))
        self.wkv = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        # Compressor rope uses the compress theta + YaRN (reference passes the
        # compressor its own freqs_cis; window w gets position w*ratio).
        self._inv_freq = _yarn_inv_freq(
            self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._mia_record_impl = None
        self._mia_rope_provider = None
        self._mia_rope_tables_for_windows = None
        self._mia_stacked_projection = None
        self._project_rows_impl = self._stock_project_rows

    def _stock_project_rows(
        self,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.wkv(values), self.wgate(values)

    def install_mia_stacked_projection(
        self,
        owner: MiaStackedDenseProjection,
    ) -> None:
        """Bind the exact-Mia one-GEMM KV/gate owner once."""
        if self._mia_stacked_projection is not None:
            raise ValueError("the Mia compressor projection is already installed")
        self._mia_stacked_projection = owner
        self._project_rows_impl = owner

    def install_mia_rope_provider(self, provider: MiaRoPETableProvider) -> None:
        """Bind the engine-owned compress-theta table provider once."""
        if self._mia_rope_provider is not None:
            raise ValueError("the Mia compressor RoPE provider is already installed")
        self._mia_rope_provider = provider
        self._mia_rope_tables_for_windows = partial(
            provider.window_tables,
            compress_ratio=self.compress_ratio,
        )

    def install_mia_record_packer(self, mode: str) -> None:
        """Bind the source-format fused finalizer before exact-model serving."""
        selected = str(mode)
        if selected == "nvfp4":
            if self.head_dim != 512 or self.compress_ratio not in (4, 128):
                raise ValueError("Mia NVFP4 compressor geometry is not supported")
            kernel = install_nvfp4_record_kernel(
                compress_ratio=self.compress_ratio
            )
            self._mia_record_impl = partial(
                self._nvfp4_record_impl,
                kernel=kernel,
            )
        elif selected == "indexer":
            if self.head_dim != 128 or self.compress_ratio != 4:
                raise ValueError("Mia indexer compressor geometry is not supported")
            self.rotate = False
            self._mia_record_impl = partial(
                self._indexer_record_impl,
                kernel=install_indexer_record_kernel(),
            )
        else:
            raise ValueError(f"unknown Mia compressor record mode {selected!r}")

    def _rope_tables_for_windows(
        self, first_window: int, count: int
    ) -> tuple[mx.array, mx.array]:
        positions = (
            mx.arange(int(count), dtype=mx.float32) + float(first_window)
        ) * self.compress_ratio
        angles = positions[:, None] * self._inv_freq[None, :]
        return mx.cos(angles), mx.sin(angles)

    def _nvfp4_record_impl(
        self,
        kv_windows: mx.array,
        score_windows: mx.array,
        previous_kv: Optional[mx.array],
        previous_score: Optional[mx.array],
        first_window: int,
        has_previous: bool,
        output_rows: int,
        *,
        kernel,
    ) -> mx.array:
        cos, sin = self._mia_rope_tables_for_windows(first_window, output_rows)
        return fused_nvfp4_records(
            kv_windows,
            score_windows,
            kv_windows if previous_kv is None else previous_kv,
            score_windows if previous_score is None else previous_score,
            self.norm.weight,
            cos,
            sin,
            kernel=kernel,
            has_previous=has_previous,
            output_rows=output_rows,
            rms_eps=self.rms_norm_eps,
        )

    def _indexer_record_impl(
        self,
        kv_windows: mx.array,
        score_windows: mx.array,
        previous_kv: Optional[mx.array],
        previous_score: Optional[mx.array],
        first_window: int,
        has_previous: bool,
        output_rows: int,
        *,
        kernel,
    ) -> mx.array:
        cos, sin = self._mia_rope_tables_for_windows(first_window, output_rows)
        return fused_indexer_records(
            kv_windows,
            score_windows,
            kv_windows if previous_kv is None else previous_kv,
            score_windows if previous_score is None else previous_score,
            self.norm.weight,
            cos,
            sin,
            kernel=kernel,
            has_previous=has_previous,
            output_rows=output_rows,
            rms_eps=self.rms_norm_eps,
        )
    def _overlap_transform(
        self, t: mx.array, value: float, prev: Optional[mx.array] = None
    ) -> mx.array:
        """Reference ``overlap_transform`` (model.py L307-314).

        ``t``: ``[b, nwin, ratio, 2*d]`` -> ``[b, nwin, 2*ratio, d]``.  The first
        ``ratio`` slots of window w hold the previous window's tokens under the
        first-half (``:d``) projection (``value`` for w==0); the last ``ratio``
        slots hold the current window's tokens under the second-half (``d:``)
        projection.

        ``prev`` seeds window 0's first half from a window that was pooled in an
        earlier call (streaming decode); ``None`` is the fresh-sequence pad.
        """
        b, nwin, r, _ = t.shape
        d = self.head_dim
        cur = t[..., d:]                       # [b, nwin, ratio, d]  (current, d: half)
        prev_half = t[..., :d]                 # [b, nwin, ratio, d]  (:d half)
        if prev is None:
            seed = mx.full((b, 1, r, d), value, dtype=t.dtype)
        else:
            seed = prev[..., :d][:, None]      # [b, 1, ratio, d]
        prev_shift = mx.concatenate([seed, prev_half[:, :-1]], axis=1)  # w -> window w-1
        return mx.concatenate([prev_shift, cur], axis=2)          # [b, nwin, 2*ratio, d]

    def _pool_components(
        self, kv: mx.array, score: mx.array, first_window: int, out_dtype
    ) -> tuple[mx.array, mx.array]:
        """Return normalized latent rows and their rotated 64-wide RoPE tails.

        ``kv``/``score``: ``[b, nwin, slots, d]`` (``slots`` is ``ratio``, or
        ``2*ratio`` once ``_overlap_transform`` has folded the previous window in).
        Window ``first_window + i`` ropes at absolute position ``(first_window+i)*ratio``
        — its own first token — for both the overlap and non-overlap lanes.

        ``out_dtype`` is the caller's own dtype: the *pooling* is fp32 (the
        reference says so outright — "compression need fp32", L321-322) but the
        emitted row is stored at the model dtype, because the reference casts back
        before the norm (``kv = self.norm(kv.to(dtype))``, L362) and its
        ``rotate_activation`` then asserts the row is bf16 (L249).  These rows are
        concatenated with the per-position KV to form one attention tensor, so an
        fp32 row promotes the whole thing.
        """
        nwin = kv.shape[1]
        rd = self.rope_head_dim
        pooled = mx.sum(kv * mx.softmax(score, axis=2), axis=2)            # [b, nwin, d]
        pooled = self.norm(pooled.astype(out_dtype))
        win_pos = (mx.arange(nwin, dtype=mx.float32) + float(first_window)) * self.compress_ratio
        ang = win_pos[:, None] * self._inv_freq[None, :]
        cos, sin = mx.cos(ang), mx.sin(ang)
        tail = _apply_interleaved_rope(pooled[..., -rd:], cos[None], sin[None])
        return pooled, tail

    def _pool(
        self, kv: mx.array, score: mx.array, first_window: int, out_dtype
    ) -> mx.array:
        """Gated pool + norm + compress-YaRN rope of already-formed windows."""
        pooled, tail = self._pool_components(
            kv,
            score,
            first_window,
            out_dtype,
        )
        out = mx.concatenate([pooled[..., :-self.rope_head_dim], tail], axis=-1)
        return _hadamard_rotate(out) if self.rotate else out

    def _whole_projected_windows(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, int, object]:
        b, s, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        out_dtype = _store_dtype(x.dtype)
        cutoff = s - (s % ratio)
        nwin = cutoff // ratio
        if nwin == 0:
            width = (2 if self.overlap else 1) * d
            empty = mx.zeros((b, 0, ratio, width), dtype=mx.float32)
            return empty, empty, 0, out_dtype
        xf = x.astype(mx.float32)
        kv_rows, score_rows = self._project_rows_impl(xf)
        kv = kv_rows[:, :cutoff].reshape(b, nwin, ratio, -1)          # [b,nwin,ratio,coff*d]
        score = score_rows[:, :cutoff].reshape(b, nwin, ratio, -1) + self.ape
        return kv, score, nwin, out_dtype

    def _whole_components(self, x: mx.array) -> tuple[mx.array, mx.array]:
        b = int(x.shape[0])
        d = self.head_dim
        kv, score, nwin, out_dtype = self._whole_projected_windows(x)
        if nwin == 0:
            return (
                mx.zeros((b, 0, d), dtype=out_dtype),
                mx.zeros((b, 0, self.rope_head_dim), dtype=out_dtype),
            )
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)                          # [b,nwin,2*ratio,d]
            score = self._overlap_transform(score, float("-inf"))
        return self._pool_components(kv, score, 0, out_dtype)

    def mia_records(self, x: mx.array) -> mx.array:
        """Finalize a full prefill directly into the installed record format."""
        if self._mia_record_impl is None:
            raise RuntimeError("Mia compressor record packer was not installed")
        kv, score, nwin, _ = self._whole_projected_windows(x)
        return self._mia_record_impl(kv, score, None, None, 0, False, nwin)

    def __call__(self, x: mx.array) -> mx.array:
        """Whole-sequence pooling from ``start_pos == 0`` (the parity-gated path).

        The incremental equivalent is :meth:`step`; both funnel into the same
        normalized latent and RoPE component builder so the two paths cannot drift.
        """
        pooled, rope = self._whole_components(x)
        out = mx.concatenate(
            [pooled[..., :-self.rope_head_dim], rope],
            axis=-1,
        )
        return _hadamard_rotate(out) if self.rotate else out

    def nvfp4(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Whole-sequence Mia attention-cache inputs."""
        if self.rotate:
            raise ValueError("the Mia attention K/V lane cannot use indexer rotation")
        return self._whole_components(x)

    def _step_projected_windows(
        self,
        x: mx.array,
        state: "CompressorState",
        offset: int,
    ) -> tuple[mx.array, mx.array, object, object, int, object]:
        """Advance the projected frontier and return newly completed windows.

        State machine adapted from ``ds4.c``'s ``compressor_decode_one`` (antirez/
        DwarfStar4, MIT): a token at position ``p`` lands in slot ``p % ratio`` of the
        in-progress window, and a row is emitted exactly when ``(p+1) % ratio == 0``.
        Window ``w`` therefore becomes attendable by query ``p == (w+1)*ratio - 1``,
        which is precisely what the prefill mask's ``c < (i+1)//ratio`` allows, so a
        decode step needs no compressed-column mask at all.

        The buffered frontier is the window's *projected* rows (post-``ape`` for the
        gate), not the raw hidden states, so the emit does the same arithmetic on the
        same values ``__call__`` would have.  For the overlap lane ``state.prev_*``
        keeps the last completed window's full-width rows.  The returned previous
        window lets either the generic pooler or the installed fused record finalizer
        consume the same state transition without doing the projection twice.
        """
        b, s, _ = x.shape
        ratio = self.compress_ratio
        out_dtype = _store_dtype(x.dtype)
        xf = x.astype(mx.float32)
        kv_rows, score_rows = self._project_rows_impl(xf)         # [b, s, coff*d]
        ape_idx = (mx.arange(s) + offset) % ratio                # slot of each token
        score_rows = score_rows + self.ape[ape_idx]
        # Rollback journal: the projected rows are per-position pure functions, so
        # keeping the most recent few is all a rewind needs to rebuild the frontier
        # (:meth:`CompressorState.rollback`).  Pushed BEFORE the frontier concat so
        # the journal stores each row exactly once, as the very array the emit path
        # consumes — a rebuilt frontier is then bit-identical, not merely equal.
        kv_rows, score_rows = state.append_projected_rows(
            kv_rows,
            score_rows,
            offset=offset,
        )
        # kv_rows[:, 0] is at position offset - (offset % ratio), a window boundary.
        total = kv_rows.shape[1]
        nwin = total // ratio
        filled = nwin * ratio
        first_window = state.n_emitted
        previous_kv = state.prev_kv
        previous_score = state.prev_score
        if nwin:
            kv_w = kv_rows[:, :filled].reshape(b, nwin, ratio, -1)
            score_w = score_rows[:, :filled].reshape(b, nwin, ratio, -1)
            if self.overlap:
                state.prev_kv = kv_w[:, -1]                      # [b, ratio, coff*d]
                state.prev_score = score_w[:, -1]
            state.n_emitted += nwin
        else:
            width = (2 if self.overlap else 1) * self.head_dim
            kv_w = mx.zeros((b, 0, ratio, width), dtype=mx.float32)
            score_w = kv_w
        state.cur_kv = kv_rows[:, filled:] if filled < total else None
        state.cur_score = score_rows[:, filled:] if filled < total else None
        return (
            kv_w,
            score_w,
            previous_kv,
            previous_score,
            first_window,
            out_dtype,
        )

    def _step_components(
        self,
        x: mx.array,
        state: "CompressorState",
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        kv_w, score_w, previous_kv, previous_score, first_window, out_dtype = (
            self._step_projected_windows(x, state, offset)
        )
        nwin = int(kv_w.shape[1])
        if nwin == 0:
            return (
                mx.zeros((int(x.shape[0]), 0, self.head_dim), dtype=out_dtype),
                mx.zeros(
                    (int(x.shape[0]), 0, self.rope_head_dim), dtype=out_dtype
                ),
            )
        if self.overlap:
            kv_slots = self._overlap_transform(kv_w, 0.0, previous_kv)
            score_slots = self._overlap_transform(
                score_w, float("-inf"), previous_score
            )
        else:
            kv_slots, score_slots = kv_w, score_w
        pooled, rope = self._pool_components(
            kv_slots,
            score_slots,
            first_window,
            out_dtype,
        )
        return pooled, rope

    def step_records(
        self,
        x: mx.array,
        state: "CompressorState",
        offset: int,
    ) -> mx.array:
        """Advance the frontier and emit source-format cache records directly."""
        if self._mia_record_impl is None:
            raise RuntimeError("Mia compressor record packer was not installed")
        return self._step_records_installed(x, state, offset)

    def _step_records_installed(
        self,
        x: mx.array,
        state: "CompressorState",
        offset: int,
    ) -> mx.array:
        """Advance through the construction-bound fused record finalizer."""
        kv_w, score_w, previous_kv, previous_score, first_window, _ = (
            self._step_projected_windows(x, state, offset)
        )
        nwin = int(kv_w.shape[1])
        has_previous = self.overlap and previous_kv is not None
        return self._mia_record_impl(
            kv_w,
            score_w,
            previous_kv,
            previous_score,
            first_window,
            has_previous,
            nwin,
        )

    def _step_m6_projected_windows(
        self,
        x: mx.array,
        state: "FixedMiaCompressorState",
        schedule,
    ) -> tuple[mx.array, mx.array, object, object, int, object]:
        """Advance one physical-M6 frontier through its request-owned schedule."""
        batch, sequence, _ = x.shape
        ratio = self.compress_ratio
        out_dtype = _store_dtype(x.dtype)
        kv_rows, score_rows = self._project_rows_impl(x.astype(mx.float32))
        score_rows = score_rows + self.ape[schedule.ape_slots]
        kv_rows, score_rows = state.append_m6_projected_rows(
            kv_rows,
            score_rows,
            schedule,
        )
        emitted = schedule.emitted_rows
        filled = emitted * ratio
        previous_kv = state.prev_kv
        previous_score = state.prev_score
        if emitted:
            kv_windows = kv_rows[:, :filled].reshape(
                batch,
                emitted,
                ratio,
                -1,
            )
            score_windows = score_rows[:, :filled].reshape(
                batch,
                emitted,
                ratio,
                -1,
            )
            if self.overlap:
                state.prev_kv = kv_windows[:, -1]
                state.prev_score = score_windows[:, -1]
        else:
            width = (2 if self.overlap else 1) * self.head_dim
            kv_windows = mx.zeros(
                (batch, 0, ratio, width),
                dtype=mx.float32,
            )
            score_windows = kv_windows
        state.n_emitted = schedule.first_window + emitted
        total = schedule.prior_rows + int(sequence)
        state.cur_kv = kv_rows[:, filled:] if filled < total else None
        state.cur_score = score_rows[:, filled:] if filled < total else None
        return (
            kv_windows,
            score_windows,
            previous_kv,
            previous_score,
            schedule.first_window,
            out_dtype,
        )

    def _step_m6_records_installed(
        self,
        x: mx.array,
        state: "FixedMiaCompressorState",
        schedule,
    ) -> mx.array:
        """Emit physical-M6 records without rebuilding position schedules."""
        kv_w, score_w, previous_kv, previous_score, first_window, _ = (
            self._step_m6_projected_windows(x, state, schedule)
        )
        return self._mia_record_impl(
            kv_w,
            score_w,
            previous_kv,
            previous_score,
            first_window,
            self.overlap and previous_kv is not None,
            schedule.emitted_rows,
        )

    def step(self, x: mx.array, state: "CompressorState", offset: int) -> mx.array:
        """Emit the legacy combined row used by the generic DeepSeek path."""
        pooled, rope = self._step_components(x, state, offset)
        out = mx.concatenate(
            [pooled[..., :-self.rope_head_dim], rope],
            axis=-1,
        )
        return _hadamard_rotate(out) if self.rotate else out

    def step_nvfp4(
        self,
        x: mx.array,
        state: "CompressorState",
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        """Emit Mia attention-cache inputs without reconstructing a combined row."""
        if self.rotate:
            raise ValueError("the Mia attention K/V lane cannot use indexer rotation")
        return self._step_components(x, state, offset)


class _UninstalledMiaIndexerTopK:
    __slots__ = ()

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(
            "the Mia indexer workspace was not installed before execution"
        )


class Indexer(nn.Module):
    """Sparse-position selector for ``compress_ratio==4`` layers (reference
    ``Indexer``, model.py L380-433).

    It owns a second, narrower :class:`Compressor` (``index_head_dim`` wide) that
    pools the *same* token windows as the attention compressor, plus
    ``wq_b``/``weights_proj``.  The generic reference lane Hadamard-rotates both
    sides; the exact Mia lane installs its source-native unrotated FP8 contract.
    For each query it scores every compressed row

        ``score[q, c] = sum_h relu(q_h · row_c) * weights[q, h]``
        ``weights     = weights_proj(x) / sqrt(index_head_dim * index_n_heads)``

    and keeps the top ``index_topk`` of the rows that are causally available to that
    query.  The generic route returns a boolean ``[b, s, n_comp]`` mask.  Exact Mia
    installation replaces that construction-time callable with SparkInfer's compact
    ``[b, s, topk]`` indices plus ``[b, s]`` lengths contract.

    Per-query ``k``: the reference prefill takes one global
    ``k = min(index_topk, end_pos // ratio)`` over ``-inf``-masked scores and then
    re-invalidates any non-causal pick (L424-430), which is equivalent to taking
    ``k = min(index_topk, n_causal(q))`` per query — the form ds4.c evaluates
    directly (``indexer_allowed_decode_one``) and the form used here, because it also
    covers the chunked-prefill case the reference has no branch for.

    The generic route keeps its historical full-precision scoring.  The exact Mia
    route replaces the selector callable at installation with the image's padded-cache
    compatibility contract: post-RoPE BF16 queries and K rows are E4M3-quantized
    directly with one power-of-two FP32 scale per 128-wide row, K is retained as paged
    132-byte records, and the score kernel reads those pages through the block table.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
        # The reference hands the indexer the *attention layer's* freqs_cis
        # (model.py L494); on a ratio-4 layer that is compress_rope_theta + YaRN.
        self._inv_freq = _yarn_inv_freq(
            self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._score_rows = self._dense_score_rows
        self._prepare_query_rows = self._hadamard_query_rows
        self._select_rows = self._dense_select_rows
        self._query_components_impl = self._legacy_query_components

    @staticmethod
    def _dense_score_rows(q: mx.array, weights: mx.array, rows: mx.array) -> mx.array:
        score = mx.einsum("bshd,btd->bsht", q, rows.astype(mx.float32))
        return mx.sum(mx.maximum(score, 0.0) * weights[..., None], axis=2)

    @staticmethod
    def _hadamard_query_rows(q: mx.array) -> mx.array:
        return _hadamard_rotate(q.astype(mx.float32))

    @staticmethod
    def _native_query_rows(q: mx.array) -> mx.array:
        return q

    def install_mia_paged_topk(
        self,
        workspace: MiaIndexerWorkspace,
        rope_table: MiaIndexerRoPETable,
    ) -> None:
        # Mia's fused FP8 indexer quantizes post-RoPE BF16 Q/K directly.  The
        # generic DwarfStar lane's orthogonal Hadamard is not in the pinned
        # vLLM/SparkInfer path and would change the post-quantization scores.
        self.compressor.rotate = False
        self.compressor.install_mia_record_packer("indexer")
        self._prepare_query_rows = self._native_query_rows
        self._mia_query_records = install_indexer_query_records(
            heads=self.n_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            weight_scale=self.softmax_scale * self.n_heads**-0.5,
            rope_table=rope_table,
        )
        self._query_components_impl = self._mia_query_components
        self._mia_workspace = workspace
        self._mia_rope_table = rope_table
        self._select_rows = install_paged_indexer_topk(
            heads=self.n_heads,
            head_dim=self.head_dim,
            topk=self.index_topk,
            compress_ratio=self.compress_ratio,
            workspace=workspace,
        )
        self._select_m6_rows = install_paged_indexer_m6_topk(
            heads=self.n_heads,
            head_dim=self.head_dim,
            topk=self.index_topk,
            compress_ratio=self.compress_ratio,
            workspace=workspace,
        )

    def _legacy_query_components(
        self,
        x: mx.array,
        qr: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        b, s, _ = x.shape
        rd = self.rope_head_dim
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        ang = positions[:, None].astype(mx.float32) * self._inv_freq[None, :]
        cos, sin = mx.cos(ang), mx.sin(ang)
        q = mx.concatenate(
            [
                q[..., :-rd],
                _apply_interleaved_rope(
                    q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :]
                ),
            ],
            axis=-1,
        )
        q = self._prepare_query_rows(q)
        weights = self.weights_proj(x).astype(mx.float32) * (
            self.softmax_scale * self.n_heads ** -0.5
        )
        return q, weights

    def _mia_query_components(
        self,
        x: mx.array,
        qr: mx.array,
        positions: mx.array,
    ) -> tuple[MiaIndexerQueryRecords, mx.array]:
        b, s, _ = x.shape
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        weights = self.weights_proj(x)
        return self._mia_query_records(
            q,
            weights,
            positions,
        )

    def _query_components(
        self,
        x: mx.array,
        qr: mx.array,
        positions: mx.array,
    ):
        return self._query_components_impl(x, qr, positions)

    def scores(
        self, x: mx.array, qr: mx.array, positions: mx.array, rows: mx.array
    ) -> mx.array:
        """Per-query relevance of every compressed row (reference L411-421).

        ``x``: ``[b, s, dim]`` (the attention input) — ``weights_proj`` reads it.
        ``qr``: ``[b, s, q_lora_rank]`` — ``q_norm(wq_a(x))``, shared with attention.
        ``positions``: ``[s]`` absolute query positions.
        ``rows``: ``[b, n_comp, index_head_dim]`` every compressed row emitted so far.
        Returns ``[b, s, n_comp]`` fp32.  No causality applied — that is
        :meth:`__call__`'s job.
        """
        q, weights = self._query_components(x, qr, positions)
        return self._score_rows(q, weights, rows)

    def _dense_select_rows(
        self,
        q: mx.array,
        weights: mx.array,
        positions: mx.array,
        rows: mx.array,
    ) -> mx.array:
        b, s = (int(dim) for dim in q.shape[:2])
        n_comp = int(rows.shape[1])
        score = self._score_rows(q, weights, rows)
        causal = (
            mx.arange(n_comp)[None, :]
            < ((positions[:, None] + 1) // self.compress_ratio)
        )[None]
        key = mx.where(causal, score, mx.array(-float("inf"), mx.float32))
        k_row = mx.minimum(
            mx.sum(causal.astype(mx.int32), axis=-1, keepdims=True),
            self.index_topk,
        )
        k_row = mx.broadcast_to(k_row, (b, s, 1))
        return _topk_mask(key, k_row, min(self.index_topk, n_comp)) & causal

    def __call__(
        self, x: mx.array, qr: mx.array, positions: mx.array, rows: mx.array
    ) -> mx.array | MiaTopKSelection:
        q, weights = self._query_components(x, qr, positions)
        return self._select_rows(q, weights, positions, rows)

    def _mia_m6_select(
        self,
        x: mx.array,
        qr: mx.array,
        positions: mx.array,
        rows,
        schedule,
    ) -> MiaTopKSelection:
        q, weights = self._query_components(x, qr, positions)
        return self._select_m6_rows(
            q.records,
            weights,
            rows,
            schedule.causal_lengths,
        )


# ---------------------------------------------------------------------------
# Streaming decode state (sliding-window KV + compressed KV + compressor frontier)
# ---------------------------------------------------------------------------
class CompressorState:
    """Rolling frontier of one compressor lane, plus its rollback journal.

    Mirrors ``ds4.c``'s ``attn_state_kv`` / ``attn_state_score`` row block
    (antirez/DwarfStar4, MIT).  ds4 keeps a fixed ``coff*ratio`` block and clears the
    unfilled tail after prefill (``compressor_finish_prefill_state_cpu``); here the
    filled rows are simply buffered, which is the same state without the -inf padding.

    **Rollback.**  Speculative decode has to un-decode the rejected tail of a verify
    batch, and on this lane that means rewinding the frontier *and* the rows already
    emitted from it.  The emitted rows are trivial — a compressed row is a pure
    function of one completed window, so dropping the rows past the rewind point is
    exact.  The frontier is not: after a window completes, ``cur_*`` is reset to the
    remainder, so a rewind that crosses an emission boundary needs rows the frontier
    no longer holds.  They are also not recomputable without the hidden states, which
    the cache does not keep.

    So this state carries a bounded **journal** of the last ``rollback_rows`` projected
    rows (``tail_kv``/``tail_score``, the same post-``wkv``/post-``ape`` values the emit
    path consumes).  :meth:`rollback` slices the frontier back out of it.  The journal
    is sized to always cover the deepest legal rewind:

        ``rollback_rows = (2 if overlap else 1) * ratio + rollback_capacity``

    — ``ratio`` rows for ``prev_*`` (the last completed window, overlap lane only),
    up to ``ratio - 1`` for ``cur_*``, and ``rollback_capacity`` for the rewind itself.
    """

    def __init__(
        self,
        ratio: int = 0,
        overlap: bool = False,
        rollback_capacity: int = 0,
    ) -> None:
        self.ratio = int(ratio)
        self.overlap = bool(overlap)
        self.rollback_capacity = max(0, int(rollback_capacity))
        self.rollback_rows = (
            0
            if self.ratio <= 0
            else (2 if self.overlap else 1) * self.ratio + self.rollback_capacity
        )
        self.cur_kv: Optional[mx.array] = None      # [b, offset % ratio, coff*head_dim]
        self.cur_score: Optional[mx.array] = None   # same, post-``ape``
        self.prev_kv: Optional[mx.array] = None     # [b, ratio, coff*head_dim] (overlap)
        self.prev_score: Optional[mx.array] = None
        self.tail_kv: Optional[mx.array] = None     # [b, <=rollback_rows, coff*head_dim]
        self.tail_score: Optional[mx.array] = None
        self.n_emitted = 0

    def reset(self) -> None:
        self.cur_kv = None
        self.cur_score = None
        self.prev_kv = None
        self.prev_score = None
        self.tail_kv = None
        self.tail_score = None
        self.n_emitted = 0

    # -- rollback journal --------------------------------------------------
    def push_rollback_rows(self, kv: mx.array, score: mx.array) -> None:
        """Append this step's freshly projected rows to the bounded journal."""
        if self.rollback_rows <= 0 or kv.shape[1] == 0:
            return
        self.tail_kv = kv if self.tail_kv is None else mx.concatenate(
            [self.tail_kv, kv], axis=1
        )
        self.tail_score = score if self.tail_score is None else mx.concatenate(
            [self.tail_score, score], axis=1
        )
        if self.tail_kv.shape[1] > self.rollback_rows:
            self.tail_kv = self.tail_kv[:, -self.rollback_rows:]
            self.tail_score = self.tail_score[:, -self.rollback_rows:]

    def append_projected_rows(
        self,
        kv: mx.array,
        score: mx.array,
        *,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        """Append projected state for the generic retained-array owner."""

        del offset
        self.push_rollback_rows(kv, score)
        if self.cur_kv is None:
            return kv, score
        return (
            mx.concatenate([self.cur_kv, kv], axis=1),
            mx.concatenate([self.cur_score, score], axis=1),
        )

    def rollback(self, n: int, new_offset: int) -> None:
        """Rewind ``n`` token positions; ``new_offset`` is the resulting offset.

        Rebuilds ``cur_*``/``prev_*``/``n_emitted`` from the journal so the state is
        the one this lane would hold had those ``n`` tokens never been stepped —
        bit-identical, because every row it installs is a slice of the same array
        the forward pass produced for that position.
        """
        if self.ratio <= 0:
            return
        held = 0 if self.tail_kv is None else int(self.tail_kv.shape[1])
        kept = held - int(n)
        if kept < 0:
            raise ValueError(
                f"compressor rollback of {n} exceeds the journal ({held} rows held)"
            )
        r = int(new_offset) % self.ratio
        need = r + (self.ratio if (self.overlap and new_offset >= self.ratio) else 0)
        if kept < need:
            raise ValueError(
                f"compressor rollback of {n} leaves {kept} journal rows, "
                f"{need} needed to rebuild the frontier at offset {new_offset}"
            )
        if kept == 0:
            self.tail_kv = None
            self.tail_score = None
        else:
            self.tail_kv = self.tail_kv[:, :kept]
            self.tail_score = self.tail_score[:, :kept]
        self.n_emitted = int(new_offset) // self.ratio
        self.cur_kv = None if r == 0 else self.tail_kv[:, kept - r:]
        self.cur_score = None if r == 0 else self.tail_score[:, kept - r:]
        if self.overlap and self.n_emitted > 0:
            lo = kept - r - self.ratio
            self.prev_kv = self.tail_kv[:, lo: lo + self.ratio]
            self.prev_score = self.tail_score[:, lo: lo + self.ratio]
        else:
            self.prev_kv = None
            self.prev_score = None


@dataclass(frozen=True, slots=True)
class _MiaM6PendingCompressorTransition:
    """Projected rows retained until one physical-M6 verdict is committed."""

    combined_kv: mx.array
    combined_score: mx.array
    previous_kv: mx.array | None
    previous_score: mx.array | None


class FixedMiaCompressorState(CompressorState):
    """Single-sequence circular compressor state derived from vLLM pages.

    vLLM stores every projected compressor row in its request page table, then
    the fused boundary gathers the ratio-sized state window by absolute token
    position.  The Mia MTPLX route owns one sequence and never shares prefixes,
    so only ``(1 + overlap) * ratio + rollback`` recent rows are live.  This
    owner preserves the same absolute-position lookup while bounding storage
    and eliminating incremental ``concatenate`` journals.
    """

    mode = "mia_fixed_compressor_state"

    def __init__(
        self,
        *,
        ratio: int,
        overlap: bool,
        rollback_capacity: int,
        state_width: int,
    ) -> None:
        super().__init__(
            ratio=ratio,
            overlap=overlap,
            rollback_capacity=rollback_capacity,
        )
        self.state_width = int(state_width)
        self._journal_kv = mx.zeros(
            (1, self.rollback_rows, self.state_width),
            dtype=mx.float32,
        )
        self._journal_score = mx.zeros(
            (1, self.rollback_rows, self.state_width),
            dtype=mx.float32,
        )
        self._journal_end = 0
        self._journal_length = 0
        self._pending_m6: _MiaM6PendingCompressorTransition | None = None

    @property
    def journal_buffers(self) -> tuple[mx.array, mx.array]:
        return self._journal_kv, self._journal_score

    def validate_journal_state(
        self,
        value,
    ) -> tuple[mx.array, mx.array]:
        """Validate snapshot bytes before mutating the installed ring owner."""

        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("fixed Mia compressor journal state must contain two arrays")
        restored = []
        for name, candidate, installed in (
            ("kv", value[0], self._journal_kv),
            ("score", value[1], self._journal_score),
        ):
            if not isinstance(candidate, mx.array):
                raise ValueError(f"fixed Mia compressor {name} journal must be an array")
            if tuple(int(dim) for dim in candidate.shape) != tuple(
                int(dim) for dim in installed.shape
            ):
                raise ValueError(
                    f"fixed Mia compressor {name} journal shape changed: "
                    f"expected {tuple(installed.shape)}, got {tuple(candidate.shape)}"
                )
            if candidate.dtype != installed.dtype:
                raise ValueError(
                    f"fixed Mia compressor {name} journal dtype changed: "
                    f"expected {installed.dtype}, got {candidate.dtype}"
                )
            restored.append(candidate)
        return restored[0], restored[1]

    def replace_journal_state(self, value) -> None:
        """Copy snapshot bytes into the construction-owned circular buffers."""

        journal_kv, journal_score = self.validate_journal_state(value)
        self._journal_kv[:] = journal_kv
        self._journal_score[:] = journal_score

    def validate_journal_meta(self, end: int, length: int) -> tuple[int, int]:
        end = int(end)
        length = int(length)
        if end < 0 or length != min(self.rollback_rows, end):
            raise ValueError(
                "fixed Mia compressor journal frontier is incompatible with "
                f"rollback_rows={self.rollback_rows}: end={end}, length={length}"
            )
        return end, length

    def replace_journal_meta(self, end: int, length: int) -> None:
        self._journal_end, self._journal_length = self.validate_journal_meta(
            end,
            length,
        )

    def validate_previous_state(
        self,
        kv,
        score,
    ) -> tuple[mx.array | None, mx.array | None]:
        if kv is None and score is None:
            return None, None
        expected = (1, self.ratio, self.state_width)
        if (
            not self.overlap
            or not isinstance(kv, mx.array)
            or not isinstance(score, mx.array)
            or tuple(int(dim) for dim in kv.shape) != expected
            or tuple(int(dim) for dim in score.shape) != expected
            or kv.dtype != mx.float32
            or score.dtype != mx.float32
        ):
            raise ValueError("fixed Mia compressor previous frontier is invalid")
        return kv, score

    def validate_current_state(self, kv, score, offset: int) -> None:
        rows = int(offset) % self.ratio
        if rows == 0:
            valid = kv is None and score is None
        else:
            expected = (1, rows, self.state_width)
            valid = (
                isinstance(kv, mx.array)
                and isinstance(score, mx.array)
                and tuple(int(dim) for dim in kv.shape) == expected
                and tuple(int(dim) for dim in score.shape) == expected
                and kv.dtype == mx.float32
                and score.dtype == mx.float32
            )
        if not valid:
            raise ValueError("fixed Mia compressor current frontier is invalid")

    def validate_previous_meta(self, kv, score, n_emitted: int) -> None:
        expected = self.overlap and int(n_emitted) > 0
        present = kv is not None or score is not None
        if expected != present:
            raise ValueError("fixed Mia compressor previous frontier is invalid")

    @staticmethod
    def validate_tail_state(kv, score) -> None:
        if kv is not None or score is not None:
            raise ValueError("fixed Mia compressor rollback tail must be empty")

    def reset(self) -> None:
        self.cur_kv = None
        self.cur_score = None
        self.prev_kv = None
        self.prev_score = None
        self.tail_kv = None
        self.tail_score = None
        self.n_emitted = 0
        self._journal_end = 0
        self._journal_length = 0
        self._pending_m6 = None

    def _latest(self, count: int) -> tuple[mx.array, mx.array]:
        positions = (
            mx.arange(self._journal_end - int(count), self._journal_end)
            % self.rollback_rows
        )
        return self._journal_kv[:, positions], self._journal_score[:, positions]

    def append_projected_rows(
        self,
        kv: mx.array,
        score: mx.array,
        *,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        offset = int(offset)
        count = int(kv.shape[1])
        prior = offset % self.ratio
        if prior:
            # ``cur_*`` is the settled frontier retained by the preceding step.
            # Keep the current rows as direct graph inputs so a boundary emit
            # never reads back the same journal slots this call is scattering.
            prior_kv, prior_score = self.cur_kv, self.cur_score
        retained = min(count, self.rollback_rows)
        source_start = count - retained
        absolute_start = offset + source_start
        positions = (
            mx.arange(absolute_start, offset + count, dtype=mx.int32)
            % self.rollback_rows
        )
        self._journal_kv[:, positions] = kv[:, source_start:]
        self._journal_score[:, positions] = score[:, source_start:]
        self._journal_end = offset + count
        self._journal_length = min(self.rollback_rows, self._journal_end)

        if prior == 0:
            return kv, score
        return (
            mx.concatenate((prior_kv, kv), axis=1),
            mx.concatenate((prior_score, score), axis=1),
        )

    def append_m6_projected_rows(
        self,
        kv: mx.array,
        score: mx.array,
        schedule,
    ) -> tuple[mx.array, mx.array]:
        """Journal one physical-M6 projection through precomputed ring slots."""
        prior = schedule.prior_rows
        if prior:
            prior_kv, prior_score = self.cur_kv, self.cur_score
        self._journal_kv[
            schedule.batch_slots,
            schedule.journal_slots,
        ] = kv[0]
        self._journal_score[
            schedule.batch_slots,
            schedule.journal_slots,
        ] = score[0]
        self._journal_end = schedule.start_offset + int(kv.shape[1])
        self._journal_length = min(self.rollback_rows, self._journal_end)
        if prior:
            combined_kv = mx.concatenate((prior_kv, kv), axis=1)
            combined_score = mx.concatenate((prior_score, score), axis=1)
        else:
            combined_kv, combined_score = kv, score
        self._pending_m6 = _MiaM6PendingCompressorTransition(
            combined_kv=combined_kv,
            combined_score=combined_score,
            previous_kv=self.prev_kv,
            previous_score=self.prev_score,
        )
        return combined_kv, combined_score

    def _commit_m6_plain(self, acceptance) -> None:
        transition = self._pending_m6
        current = acceptance.current_rows
        start = acceptance.current_start
        self.cur_kv = (
            transition.combined_kv[:, start : start + current]
            if current
            else None
        )
        self.cur_score = (
            transition.combined_score[:, start : start + current]
            if current
            else None
        )
        self.prev_kv = None
        self.prev_score = None
        self.n_emitted = acceptance.compressed_rows
        self._journal_end = acceptance.stop_offset
        self._journal_length = acceptance.journal_length
        self._pending_m6 = None

    def _commit_m6_overlap(self, acceptance) -> None:
        transition = self._pending_m6
        current = acceptance.current_rows
        start = acceptance.current_start
        self.cur_kv = (
            transition.combined_kv[:, start : start + current]
            if current
            else None
        )
        self.cur_score = (
            transition.combined_score[:, start : start + current]
            if current
            else None
        )
        if acceptance.completed_rows:
            previous_start = acceptance.previous_start
            self.prev_kv = transition.combined_kv[
                :,
                previous_start : previous_start + self.ratio,
            ]
            self.prev_score = transition.combined_score[
                :,
                previous_start : previous_start + self.ratio,
            ]
        else:
            self.prev_kv = transition.previous_kv
            self.prev_score = transition.previous_score
        self.n_emitted = acceptance.compressed_rows
        self._journal_end = acceptance.stop_offset
        self._journal_length = acceptance.journal_length
        self._pending_m6 = None

    def rollback(self, n: int, new_offset: int) -> None:
        del n
        new_offset = int(new_offset)
        self._journal_end = new_offset
        self._journal_length = min(self.rollback_rows, new_offset)
        self.n_emitted = new_offset // self.ratio
        remainder = new_offset % self.ratio
        if remainder:
            self.cur_kv, self.cur_score = self._latest(remainder)
        else:
            self.cur_kv = None
            self.cur_score = None
        if self.overlap and self.n_emitted > 0:
            previous_kv, previous_score = self._latest(remainder + self.ratio)
            self.prev_kv = previous_kv[:, : self.ratio]
            self.prev_score = previous_score[:, : self.ratio]
        else:
            self.prev_kv = None
            self.prev_score = None


class DeepseekV4Cache:
    """Per-layer streaming cache.

    Five pieces, following ``ds4_layer_cache`` (ds4.c, MIT):
      * ``window``  — the rotated per-position KV rows still inside the sliding
        window, sliding by one row once full (``kv_cache_push_raw``).
      * ``compressed`` — every compressed KV row emitted so far
        (``kv_cache_push_comp``, ds4's ``attn_comp_kv``).
      * ``comp`` — the compressor's in-progress window (:class:`CompressorState`).
      * ``index_compressed`` / ``index_comp`` — the same two things for the ratio-4
        indexer's own, narrower compressor: ds4 carries ``index_comp_kv`` beside
        ``attn_comp_kv`` and ``index_state_kv``/``index_state_score`` beside
        ``attn_state_*``.  Maintained on every ratio-4 step regardless of whether the
        filter is currently active, because a row can only be built when its window's
        tokens go past — a context that crosses ``index_topk`` mid-decode would
        otherwise have no rows to score.

    Neither compressed lane is evicted, matching both the reference (a
    ``max_seq_len // ratio`` cache written at ``start_pos // ratio``, model.py L376)
    and ds4 (``comp_cap = ctx/ratio + 2``): the top-k filter bounds how many rows are
    *attended*, not how many are *kept*.  Row storage therefore still grows at
    ``head_dim/ratio`` bytes per token per compressed layer.

    ``offset`` is the absolute position of the next token, i.e. the standard
    mlx-lm cache contract the generate/serve path reads.

    **Rollback (``trim``).**  The speculative lane verifies ``K+1`` tokens in one
    forward and then has to un-decode the rejected tail.  All three lanes here are
    rewindable, by three different mechanisms, each chosen because it is *exact*:

      * emitted compressed rows (both lanes) — **truncate**.  A row is a pure
        function of one completed window, so the rows a shorter context would have
        produced are a prefix of the rows this one did.
      * compressor / indexer frontier — **journal**.  ``cur_*``/``prev_*`` are
        rebuilt from :class:`CompressorState`'s bounded row journal, because a
        rewind across an emission boundary needs rows the frontier itself dropped
        and the cache keeps no hidden states to recompute them from.
      * sliding-window KV — **retention**.  Evicted rows are gone for good, so the
        window simply holds ``rollback_capacity`` rows more than it needs and
        returns only the attendable prefix to attention (which is why the retention
        change is invisible to the forward).  ``trim`` past that bound raises rather
        than silently half-rewinding.

    ``rollback_capacity`` is therefore a hard bound on rewind depth, uniform across
    the three lanes.  It is not a bound on how far back the model can *attend*.
    """

    _META_VERSION = "mtplx-deepseek-v4-cache-v3"

    def __init__(
        self,
        window_size: int,
        compress_ratio: int,
        head_dim: int,
        rollback_capacity: int = _DEFAULT_ROLLBACK_CAPACITY,
    ) -> None:
        self.window_size = int(window_size)
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(head_dim)
        self.rollback_capacity = max(0, int(rollback_capacity))
        self.offset = 0
        self.window: Optional[mx.array] = None      # [b, L, head_dim]
        self.window_start = 0                       # abs position of window[:, 0]
        self.compressed: Optional[mx.array] = None  # [b, n_comp, head_dim]
        overlap = self.compress_ratio == 4
        self.comp = CompressorState(
            ratio=self.compress_ratio,
            overlap=overlap,
            rollback_capacity=self.rollback_capacity,
        )
        self.index_compressed: Optional[mx.array] = None  # [b, n_comp, index_head_dim]
        self.index_comp = CompressorState(
            ratio=self.compress_ratio,
            overlap=overlap,
            rollback_capacity=self.rollback_capacity,
        )

    # -- streaming updates -------------------------------------------------
    @property
    def n_compressed(self) -> int:
        return 0 if self.compressed is None else int(self.compressed.shape[1])

    @property
    def n_index_compressed(self) -> int:
        return 0 if self.index_compressed is None else int(self.index_compressed.shape[1])

    def update_window(self, latent: mx.array, rope: mx.array):
        """Append ``kv`` (positions ``offset..offset+s-1``) and return the rows this
        call can still see, as ``(rows, first_position)``.

        A query at ``p`` attends ``(p - window_size, p]``, so once the oldest query is
        ``offset`` nothing older than ``offset - window_size`` can matter to this
        call: those rows are excluded from the returned slice rather than masked.
        For ``s == 1`` that leaves exactly the attendable set, so the decode step
        needs no mask.

        What is *returned* and what is *retained* are two different sets.  The buffer
        keeps ``window_size + rollback_capacity`` rows so a rewind still has the rows
        the shorter context would have been holding (eviction is irreversible — see
        the class docstring); the extra rows never reach attention, so retention
        depth cannot change the forward.
        """
        kv = mx.concatenate(
            [latent[..., : -int(rope.shape[-1])], rope],
            axis=-1,
        )
        s = int(kv.shape[1])
        if self.window is None:
            buf, buf_start = kv, self.offset
        else:
            buf = mx.concatenate([self.window, kv], axis=1)
            buf_start = self.window_start
        # rows visible to this call's oldest query (position ``offset``)
        first_visible = max(0, self.offset - self.window_size + 1)
        lo = max(0, first_visible - buf_start)
        rows = buf[:, lo:] if lo else buf
        start = buf_start + lo
        keep = self.window_size + self.rollback_capacity
        if buf.shape[1] > keep:
            buf = buf[:, -keep:]
            buf_start = self.offset + s - keep
        self.window = buf
        self.window_start = buf_start
        return (rows, rows), start

    @staticmethod
    def attention_window_length(rows) -> int:
        return int(rows[0].shape[1])

    @staticmethod
    def _grow(rows: Optional[mx.array], new: mx.array) -> Optional[mx.array]:
        if new.shape[1] == 0:
            return rows
        return new if rows is None else mx.concatenate([rows, new], axis=1)

    def update_compressed(self, compressor: Compressor, x: mx.array) -> None:
        """Run the attention compressor's frontier over ``x`` and append its rows."""
        self.compressed = self._grow(
            self.compressed, compressor.step(x, self.comp, self.offset)
        )

    def attention_compressed(self) -> tuple[mx.array, mx.array]:
        if self.compressed is None:
            raise ValueError("DeepSeek-V4 compressed attention cache is empty")
        return self.compressed, self.compressed

    def update_index_compressed(self, compressor: Compressor, x: mx.array) -> None:
        """Same, for the ratio-4 indexer's own compressor lane."""
        self.index_compressed = self._grow(
            self.index_compressed, compressor.step(x, self.index_comp, self.offset)
        )

    def attention_index_compressed(self):
        if self.index_compressed is None:
            raise ValueError("DeepSeek-V4 indexer cache is empty")
        return self.index_compressed

    def advance(self, s: int) -> None:
        self.offset += int(s)

    # -- rollback ----------------------------------------------------------
    @property
    def max_rollback(self) -> int:
        """Deepest legal :meth:`trim`, in token positions."""
        return min(self.rollback_capacity, int(self.offset))

    def trim(self, n: int) -> int:
        """Un-decode the last ``n`` token positions; returns ``n``.

        The mlx-lm cache trim contract (``rollback_after_verify`` /
        ``trim_verified_window_to_prefix`` in ``mtplx.cache_state``), implemented
        exactly: afterwards every field holds what it would hold had those ``n``
        tokens never been passed to the model, so the next forward is bit-identical
        to the one the shorter context would have run.

        Unlike a plain KV cache this trim is *bounded* (:attr:`max_rollback`) — the
        sliding window physically discards evicted rows.  Exceeding the bound raises
        instead of clamping: ``rollback_after_verify`` ignores the return value, so a
        clamped rewind would leave a silently desynced cache decoding on.

        The speculative lane never approaches the bound (it rewinds at most the
        verify width, ``K+1``).  The one caller that can is the session bank's
        near-prefix restore, which trims a restored snapshot down to an arbitrary
        matched prefix (``generation._trim_cache_to_offset``); on this backend that
        depth is not recoverable at all — the rows are gone — so it raises rather
        than returning the False that would let the caller fall back to a cold
        prefill.  Serving V4 behind a session bank therefore needs either a
        ``rollback_capacity`` sized for it or a ``max_rollback`` pre-check in that
        caller.
        """
        n = int(n)
        if n <= 0:
            return 0
        if n > int(self.offset):
            raise ValueError(
                f"cannot trim {n} tokens from a DeepSeek-V4 cache at offset "
                f"{self.offset}"
            )
        if n > self.rollback_capacity:
            raise ValueError(
                f"DeepSeek-V4 cache rollback of {n} exceeds rollback_capacity="
                f"{self.rollback_capacity}: the sliding window has already evicted "
                "the rows that depth would need"
            )
        new_offset = int(self.offset) - n
        if self.window is not None:
            kept = int(self.window.shape[1]) - n
            if kept <= 0:
                self.window = None
                self.window_start = new_offset
            else:
                self.window = self.window[:, :kept]
        if self.compress_ratio:
            n_rows = new_offset // self.compress_ratio
            if self.compressed is not None:
                self.compressed = None if n_rows == 0 else self.compressed[:, :n_rows]
            self.comp.rollback(n, new_offset)
            # The indexer lane only exists on ratio-4 layers; on ratio-128 its
            # state is constructed but never stepped, so there is nothing to rewind.
            if self.compress_ratio == 4:
                if self.index_compressed is not None:
                    self.index_compressed = (
                        None if n_rows == 0 else self.index_compressed[:, :n_rows]
                    )
                self.index_comp.rollback(n, new_offset)
        self.offset = new_offset
        return n

    # -- mlx-lm cache contract --------------------------------------------
    @property
    def state(self):
        return (
            self.window,
            self.compressed,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.comp.tail_kv,
            self.comp.tail_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
            self.index_comp.tail_kv,
            self.index_comp.tail_score,
        )

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.window = None
            self.compressed = None
            self.index_compressed = None
            self.comp.reset()
            self.index_comp.reset()
            self.offset = 0
            self.window_start = 0
            return
        if not isinstance(value, (tuple, list)) or len(value) != 15:
            raise ValueError("DeepSeek-V4 cache state must contain fifteen entries")
        (
            self.window,
            self.compressed,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.comp.tail_kv,
            self.comp.tail_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
            self.index_comp.tail_kv,
            self.index_comp.tail_score,
        ) = value

    def replace_state(self, value) -> None:
        self.state = value

    @property
    def meta_state(self):
        return (
            self._META_VERSION,
            str(self.offset),
            str(self.window_start),
            str(self.comp.n_emitted),
            str(self.index_comp.n_emitted),
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 5
            or value[0] != self._META_VERSION
        ):
            raise ValueError(f"unsupported DeepSeek-V4 cache meta state: {value!r}")
        self.offset = int(value[1])
        self.window_start = int(value[2])
        self.comp.n_emitted = int(value[3])
        self.index_comp.n_emitted = int(value[4])

    def is_trimmable(self) -> bool:
        # :meth:`trim` rewinds all three lanes exactly, which is what lets the
        # engine's snapshot-free rejection repair
        # (``mtplx.cache_state.trim_verified_window_without_snapshot``) serve this
        # backend instead of a bespoke restore path.
        return True

    def size(self) -> int:
        return int(self.offset)

    def empty(self) -> bool:
        return self.offset == 0


@dataclass(frozen=True, slots=True)
class _MiaCompressorSnapshotState:
    cur_kv: object
    cur_score: object
    prev_kv: object
    prev_score: object
    tail_kv: object
    tail_score: object
    journal: tuple[mx.array, mx.array] | None


@dataclass(frozen=True, slots=True)
class _MiaNVFP4SnapshotState:
    window: object
    compressed: object
    comp: _MiaCompressorSnapshotState
    index_compressed: object
    index_comp: _MiaCompressorSnapshotState


@dataclass(frozen=True, slots=True)
class _MiaNVFP4SnapshotMeta:
    offset: int
    window_start: int
    comp_emitted: int
    index_emitted: int
    comp_end: int
    comp_length: int
    index_end: int
    index_length: int


class DeepseekV4NVFP4Cache(DeepseekV4Cache):
    """DeepSeek V4 streaming cache in Mia's native ``stock432`` K/V format."""

    _STATE_VERSION = "mtplx-deepseek-v4-nvfp4-paged-state-v3"
    _META_VERSION = "mtplx-deepseek-v4-nvfp4-paged-cache-v3"

    def __init__(
        self,
        window_size: int,
        compress_ratio: int,
        head_dim: int,
        rollback_capacity: int = _DEFAULT_ROLLBACK_CAPACITY,
        capacity_tokens: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(
            window_size=window_size,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rollback_capacity=rollback_capacity,
        )
        if self.compress_ratio:
            overlap = self.compress_ratio == 4
            self.comp = FixedMiaCompressorState(
                ratio=self.compress_ratio,
                overlap=overlap,
                rollback_capacity=self.rollback_capacity,
                state_width=(2 if overlap else 1) * self.head_dim,
            )
            if self.compress_ratio == 4:
                self.index_comp = FixedMiaCompressorState(
                    ratio=self.compress_ratio,
                    overlap=True,
                    rollback_capacity=self.rollback_capacity,
                    state_width=2 * 128,
                )
        if self.head_dim != 512:
            raise ValueError("Mia stock432 target cache requires head_dim=512")
        if max_batch_tokens is None:
            self._pack_window_records = install_stock432_record_packer(
                head_dim=self.head_dim,
                rope_dim=64,
            )
            self.window = MiaNVFP4Rows()
            self._update_window_impl = self._update_growing_window
            self._trim_window_impl = self._trim_growing_window
        else:
            max_batch_tokens = int(max_batch_tokens)
            if max_batch_tokens <= 0:
                raise ValueError("Mia fixed target window requires max_batch_tokens")
            self.window = FixedMiaNVFP4Window(
                capacity_rows=(
                    max_batch_tokens + self.window_size + self.rollback_capacity
                )
            )
            self._write_window_records = install_fixed_window_record_writer(
                self.window
            )
            self._pack_window_records = None
            self._update_window_impl = self._fixed_window_requires_records
            self._trim_window_impl = self._trim_fixed_window
        if self.compress_ratio:
            if capacity_tokens is None or int(capacity_tokens) <= 0:
                raise ValueError(
                    "Mia compressed stock432 cache requires a positive installed "
                    "capacity_tokens"
                )
            compressed_capacity = (
                int(capacity_tokens) + self.compress_ratio - 1
            ) // self.compress_ratio
            self.compressed = PagedMiaNVFP4Rows(
                capacity_rows=compressed_capacity,
                block_size=max(1, 256 // self.compress_ratio),
            )
            self.index_compressed = (
                PagedMiaIndexerRows(
                    capacity_rows=compressed_capacity,
                    block_size=64,
                )
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressed = MiaNVFP4Rows()

    @property
    def n_compressed(self) -> int:
        return len(self.compressed)

    @property
    def n_index_compressed(self) -> int:
        return 0 if self.index_compressed is None else len(self.index_compressed)

    def update_window(self, latent: mx.array, rope: mx.array):
        return self._update_window_impl(latent, rope)

    def update_window_records(self, records: mx.array):
        """Insert fused stock432 rows and preserve target cache bookkeeping."""
        return self._update_fixed_window_records(records)

    @staticmethod
    def _fixed_window_requires_records(_latent: mx.array, _rope: mx.array):
        raise RuntimeError(
            "the fixed Mia target cache accepts finalized stock432 records only"
        )

    def _update_growing_window(self, latent: mx.array, rope: mx.array):
        s = int(latent.shape[1])
        buf_start = self.offset if len(self.window) == 0 else self.window_start
        records = self._pack_window_records(latent, rope)
        self.window._append_installed_records(
            records,
            prefix=tuple(int(value) for value in records.shape[:-2]),
        )
        total = len(self.window)
        first_visible = max(0, self.offset - self.window_size + 1)
        lo = max(0, first_visible - buf_start)
        records = self.window.records[..., lo:total, :]
        start = buf_start + lo
        keep = self.window_size + self.rollback_capacity
        if total > keep:
            self.window.drop_first(total - keep)
            buf_start = self.offset + s - keep
        self.window_start = buf_start
        return records, start

    def _update_fixed_window(self, latent: mx.array, rope: mx.array):
        records = self._pack_window_records(latent, rope)
        return self._update_fixed_window_records(records)

    def _update_fixed_window_records(self, records: mx.array):
        s = int(records.shape[1])
        self._write_window_records(records, absolute_start=self.offset)
        start = max(0, self.offset - self.window_size + 1)
        visible = self.window.paged_records(start, self.offset + s)
        keep = self.window_size + self.rollback_capacity
        retained_start = max(0, self.offset + s - keep)
        self.window.drop_before(retained_start)
        self.window_start = retained_start
        return visible, start

    def _trim_growing_window(self, n: int, new_offset: int) -> None:
        self.window.truncate(len(self.window) - int(n))
        if len(self.window) == 0:
            self.window_start = int(new_offset)

    def _trim_fixed_window(self, _n: int, new_offset: int) -> None:
        keep = self.window_size + self.rollback_capacity
        self.window_start = self.window.rewind(int(new_offset), keep=keep)

    @staticmethod
    def attention_window_length(
        records: mx.array | FixedMiaNVFP4WindowRecords,
    ) -> int:
        return int(records.shape[1])

    def update_compressed(self, compressor: Compressor, x: mx.array) -> None:
        records = compressor._step_records_installed(x, self.comp, self.offset)
        if records.shape[1]:
            self.compressed._append_installed_records(records)

    def update_index_compressed(self, compressor: Compressor, x: mx.array) -> None:
        records = compressor._step_records_installed(x, self.index_comp, self.offset)
        if records.shape[1]:
            self.index_compressed._append_installed_records(records)

    def update_compressed_m6(self, compressor: Compressor, x: mx.array, schedule) -> None:
        records = compressor._step_m6_records_installed(x, self.comp, schedule)
        self.compressed._append_m6_records(records, schedule)

    def update_index_compressed_m6(
        self,
        compressor: Compressor,
        x: mx.array,
        schedule,
    ) -> None:
        records = compressor._step_m6_records_installed(
            x,
            self.index_comp,
            schedule,
        )
        self.index_compressed._append_m6_records(records, schedule)

    def _commit_m6_window(self, acceptance) -> None:
        self.window._start = acceptance.window_start
        self.window._end = acceptance.stop_offset
        self.window._paged_records.length = (
            acceptance.stop_offset - acceptance.window_start
        )
        self.window_start = acceptance.window_start
        self.offset = acceptance.stop_offset

    def _commit_m6_ratio0(self, acceptance, _ratio_acceptance) -> None:
        self._commit_m6_window(acceptance)

    def _commit_m6_ratio4(self, acceptance, ratio_acceptance) -> None:
        self.compressed._pool.offset = ratio_acceptance.compressed_rows
        self.index_compressed._pool.offset = ratio_acceptance.compressed_rows
        self.comp._commit_m6_overlap(ratio_acceptance)
        self.index_comp._commit_m6_overlap(ratio_acceptance)
        self._commit_m6_window(acceptance)

    def _commit_m6_ratio128(self, acceptance, ratio_acceptance) -> None:
        self.compressed._pool.offset = ratio_acceptance.compressed_rows
        self.comp._commit_m6_plain(ratio_acceptance)
        self._commit_m6_window(acceptance)

    def attention_index_compressed(self):
        if self.index_compressed is None or len(self.index_compressed) == 0:
            raise ValueError("DeepSeek-V4 paged indexer cache is empty")
        return self.index_compressed.paged_records

    def attention_compressed(self):
        if len(self.compressed) == 0:
            raise ValueError("DeepSeek-V4 compressed NVFP4 cache is empty")
        return self.compressed.paged_records

    def trim(self, n: int) -> int:
        n = int(n)
        if n <= 0:
            return 0
        if n > int(self.offset):
            raise ValueError(
                f"cannot trim {n} tokens from a DeepSeek-V4 cache at offset "
                f"{self.offset}"
            )
        if n > self.rollback_capacity:
            raise ValueError(
                f"DeepSeek-V4 cache rollback of {n} exceeds rollback_capacity="
                f"{self.rollback_capacity}: the sliding window has already evicted "
                "the rows that depth would need"
            )
        new_offset = int(self.offset) - n
        self._trim_window_impl(n, new_offset)
        if self.compress_ratio:
            n_rows = new_offset // self.compress_ratio
            self.compressed.truncate(n_rows)
            self.comp.rollback(n, new_offset)
            if self.compress_ratio == 4:
                self.index_compressed.truncate(n_rows)
                self.index_comp.rollback(n, new_offset)
        self.offset = new_offset
        return n

    def _trim_installed(self, n: int) -> int:
        """Rewind a verifier-qualified tail without repeating arena proofs."""

        n = int(n)
        if n <= 0:
            return 0
        new_offset = int(self.offset) - n
        self._trim_window_impl(n, new_offset)
        if self.compress_ratio:
            n_rows = new_offset // self.compress_ratio
            self.compressed.truncate(n_rows)
            self.comp.rollback(n, new_offset)
            if self.compress_ratio == 4:
                self.index_compressed.truncate(n_rows)
                self.index_comp.rollback(n, new_offset)
        self.offset = new_offset
        return n

    @staticmethod
    def _validate_optional_journal_state(
        lane: CompressorState,
        kv,
        score,
    ) -> tuple[mx.array, mx.array] | None:
        if isinstance(lane, FixedMiaCompressorState):
            return lane.validate_journal_state((kv, score))
        if kv is not None or score is not None:
            raise ValueError("non-compressed Mia cache lane cannot restore a journal")
        return None

    @staticmethod
    def _validate_inactive_frontier_state(lane: CompressorState, *values) -> None:
        if not isinstance(lane, FixedMiaCompressorState) and any(
            value is not None for value in values
        ):
            raise ValueError("inactive Mia compressor frontier must be empty")

    @staticmethod
    def _validate_sealed_block_table(block_table, pool, *, label: str) -> None:
        expected = tuple(range(int(pool.num_blocks)))
        observed = tuple(int(value) for value in block_table.tolist())
        if observed != expected:
            raise ValueError(f"Mia fixed {label} block table ownership changed")

    @staticmethod
    def _validate_paged_owner_state(
        owner,
        state,
        *,
        label: str,
        owner_label: str,
    ):
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError(f"invalid {label} state")
        pages, block_table, length = state
        installed_pages = owner.pages
        installed_table = owner.block_table
        if not isinstance(pages, mx.array) or (
            tuple(int(dim) for dim in pages.shape)
            != tuple(int(dim) for dim in installed_pages.shape)
            or pages.dtype != installed_pages.dtype
        ):
            raise ValueError(f"{label} physical page geometry changed")
        if not isinstance(block_table, mx.array) or (
            tuple(int(dim) for dim in block_table.shape)
            != tuple(int(dim) for dim in installed_table.shape)
            or block_table.dtype != installed_table.dtype
        ):
            raise ValueError(f"{label} block table geometry changed")
        DeepseekV4NVFP4Cache._validate_sealed_block_table(
            block_table,
            owner._pool,
            label=owner_label,
        )
        length = int(length)
        if length < 0 or length > owner.capacity:
            raise ValueError(f"{label} frontier is outside installed capacity")
        return pages, block_table, length

    @staticmethod
    def _install_paged_owner_state(owner, state) -> None:
        pages, _block_table, length = state
        owner.pages[:] = pages
        owner._pool.offset = length

    def _validate_window_state(self, state):
        if not isinstance(self.window, FixedMiaNVFP4Window):
            return state
        if not isinstance(state, (tuple, list)) or len(state) != 4:
            raise ValueError("invalid Mia fixed target window state")
        pages, block_table, start, end = state
        if not isinstance(pages, mx.array) or (
            tuple(int(dim) for dim in pages.shape)
            != tuple(int(dim) for dim in self.window._pages.shape)
            or pages.dtype != self.window._pages.dtype
        ):
            raise ValueError("Mia fixed target window physical page geometry changed")
        if not isinstance(block_table, mx.array) or (
            tuple(int(dim) for dim in block_table.shape)
            != tuple(int(dim) for dim in self.window._pool.block_table.shape)
            or block_table.dtype != self.window._pool.block_table.dtype
        ):
            raise ValueError("Mia fixed target window block table geometry changed")
        self._validate_sealed_block_table(
            block_table,
            self.window._pool,
            label="window",
        )
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end - start > self.window.capacity:
            raise ValueError("Mia fixed target window frontier changed")
        return pages, block_table, start, end

    def _install_window_state(self, state) -> None:
        if not isinstance(self.window, FixedMiaNVFP4Window):
            self.window.replace_state(state)
            return
        pages, _block_table, start, end = state
        self.window._pages[:] = pages
        self.window._start = start
        self.window._end = end
        self.window._paged_records.length = end - start

    @property
    def state(self):
        comp_journal = (
            self.comp.journal_buffers
            if isinstance(self.comp, FixedMiaCompressorState)
            else (None, None)
        )
        index_journal = (
            self.index_comp.journal_buffers
            if isinstance(self.index_comp, FixedMiaCompressorState)
            else (None, None)
        )
        return (
            self._STATE_VERSION,
            self.window.state,
            self.compressed.state,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.comp.tail_kv,
            self.comp.tail_score,
            None if self.index_compressed is None else self.index_compressed.state,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
            self.index_comp.tail_kv,
            self.index_comp.tail_score,
            comp_journal[0],
            comp_journal[1],
            index_journal[0],
            index_journal[1],
        )

    def _validate_snapshot_state(self, value) -> _MiaNVFP4SnapshotState:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 20
            or value[0] != self._STATE_VERSION
        ):
            raise ValueError(
                "unsupported DeepSeek-V4 NVFP4 cache state: expected "
                f"{self._STATE_VERSION!r} with twenty entries"
            )
        (
            _state_version,
            raw_window_state,
            raw_compressed_state,
            comp_cur_kv,
            comp_cur_score,
            comp_prev_kv,
            comp_prev_score,
            comp_tail_kv,
            comp_tail_score,
            index_state,
            index_cur_kv,
            index_cur_score,
            index_prev_kv,
            index_prev_score,
            index_tail_kv,
            index_tail_score,
            comp_journal_kv,
            comp_journal_score,
            index_journal_kv,
            index_journal_score,
        ) = value
        window_state = self._validate_window_state(raw_window_state)
        if not self.compress_ratio:
            if raw_compressed_state is not None:
                raise ValueError(
                    "non-compressed Mia cache cannot restore compressed rows"
                )
            compressed_state = None
        else:
            compressed_state = self._validate_paged_owner_state(
                self.compressed,
                raw_compressed_state,
                label="paged Mia stock432",
                owner_label="compressed",
            )
        if self.index_compressed is None:
            if index_state is not None:
                raise ValueError("non-indexed Mia cache cannot restore indexer pages")
            validated_index_state = None
        else:
            validated_index_state = self._validate_paged_owner_state(
                self.index_compressed,
                index_state,
                label="paged Mia indexer",
                owner_label="indexer",
            )
        comp_journal = self._validate_optional_journal_state(
            self.comp,
            comp_journal_kv,
            comp_journal_score,
        )
        index_journal = self._validate_optional_journal_state(
            self.index_comp,
            index_journal_kv,
            index_journal_score,
        )
        if isinstance(self.comp, FixedMiaCompressorState):
            comp_prev_kv, comp_prev_score = self.comp.validate_previous_state(
                comp_prev_kv,
                comp_prev_score,
            )
            self.comp.validate_tail_state(comp_tail_kv, comp_tail_score)
        else:
            self._validate_inactive_frontier_state(
                self.comp,
                comp_cur_kv,
                comp_cur_score,
                comp_prev_kv,
                comp_prev_score,
                comp_tail_kv,
                comp_tail_score,
            )
        if isinstance(self.index_comp, FixedMiaCompressorState):
            index_prev_kv, index_prev_score = (
                self.index_comp.validate_previous_state(
                    index_prev_kv,
                    index_prev_score,
                )
            )
            self.index_comp.validate_tail_state(
                index_tail_kv,
                index_tail_score,
            )
        else:
            self._validate_inactive_frontier_state(
                self.index_comp,
                index_cur_kv,
                index_cur_score,
                index_prev_kv,
                index_prev_score,
                index_tail_kv,
                index_tail_score,
            )
        return _MiaNVFP4SnapshotState(
            window=window_state,
            compressed=compressed_state,
            comp=_MiaCompressorSnapshotState(
                cur_kv=comp_cur_kv,
                cur_score=comp_cur_score,
                prev_kv=comp_prev_kv,
                prev_score=comp_prev_score,
                tail_kv=comp_tail_kv,
                tail_score=comp_tail_score,
                journal=comp_journal,
            ),
            index_compressed=validated_index_state,
            index_comp=_MiaCompressorSnapshotState(
                cur_kv=index_cur_kv,
                cur_score=index_cur_score,
                prev_kv=index_prev_kv,
                prev_score=index_prev_score,
                tail_kv=index_tail_kv,
                tail_score=index_tail_score,
                journal=index_journal,
            ),
        )

    def _install_snapshot_state(self, state: _MiaNVFP4SnapshotState) -> None:
        comp = state.comp
        index_comp = state.index_comp
        self._install_window_state(state.window)
        if isinstance(self.compressed, PagedMiaNVFP4Rows):
            self._install_paged_owner_state(self.compressed, state.compressed)
        else:
            self.compressed.replace_state(state.compressed)
        self.comp.cur_kv = comp.cur_kv
        self.comp.cur_score = comp.cur_score
        self.comp.prev_kv = comp.prev_kv
        self.comp.prev_score = comp.prev_score
        self.comp.tail_kv = comp.tail_kv
        self.comp.tail_score = comp.tail_score
        self.index_comp.cur_kv = index_comp.cur_kv
        self.index_comp.cur_score = index_comp.cur_score
        self.index_comp.prev_kv = index_comp.prev_kv
        self.index_comp.prev_score = index_comp.prev_score
        self.index_comp.tail_kv = index_comp.tail_kv
        self.index_comp.tail_score = index_comp.tail_score
        if comp.journal is not None:
            self.comp.replace_journal_state(comp.journal)
        if index_comp.journal is not None:
            self.index_comp.replace_journal_state(index_comp.journal)
        if self.index_compressed is not None:
            self._install_paged_owner_state(
                self.index_compressed,
                state.index_compressed,
            )

    def prepare_snapshot_restore(
        self,
        state,
        meta_state,
    ) -> tuple[_MiaNVFP4SnapshotState, _MiaNVFP4SnapshotMeta]:
        """Validate one complete snapshot pair without mutating its arena."""
        validated_state = self._validate_snapshot_state(state)
        validated_meta = self._validate_snapshot_meta(meta_state, validated_state)
        return validated_state, validated_meta

    def install_snapshot_restore(
        self,
        prepared: tuple[_MiaNVFP4SnapshotState, _MiaNVFP4SnapshotMeta],
    ) -> None:
        """Install a pair already preflighted by the shared restore boundary."""
        validated_state, validated_meta = prepared
        self._install_snapshot_state(validated_state)
        self._install_snapshot_meta(validated_meta)

    def restore_snapshot_state(self, state, meta_state) -> None:
        """Validate one complete snapshot pair, then install it atomically."""
        self.install_snapshot_restore(
            self.prepare_snapshot_restore(state, meta_state)
        )

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.window.clear()
            self.compressed.clear()
            if self.index_compressed is not None:
                self.index_compressed.clear()
            self.comp.reset()
            self.index_comp.reset()
            self.offset = 0
            self.window_start = 0
            return
        self._validate_snapshot_state(value)
        raise ValueError(
            "DeepSeek-V4 NVFP4 state requires atomic state+meta restore"
        )

    @staticmethod
    def _journal_meta(lane: CompressorState) -> tuple[int, int]:
        if not isinstance(lane, FixedMiaCompressorState):
            return 0, 0
        return lane._journal_end, lane._journal_length

    @property
    def meta_state(self):
        comp_end, comp_length = self._journal_meta(self.comp)
        index_end, index_length = self._journal_meta(self.index_comp)
        return (
            self._META_VERSION,
            str(self.offset),
            str(self.window_start),
            str(self.comp.n_emitted),
            str(self.index_comp.n_emitted),
            str(comp_end),
            str(comp_length),
            str(index_end),
            str(index_length),
        )

    def _validate_snapshot_meta(
        self,
        value,
        state: _MiaNVFP4SnapshotState,
    ) -> _MiaNVFP4SnapshotMeta:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 9
            or value[0] != self._META_VERSION
        ):
            raise ValueError(f"unsupported DeepSeek-V4 cache meta state: {value!r}")
        try:
            (
                offset,
                window_start,
                comp_emitted,
                index_emitted,
                comp_end,
                comp_length,
                index_end,
                index_length,
            ) = (int(item) for item in value[1:])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported DeepSeek-V4 cache meta state: {value!r}"
            ) from exc
        if offset < 0 or window_start < 0 or window_start > offset:
            raise ValueError("DeepSeek-V4 NVFP4 cache frontier is invalid")
        expected_comp_emitted = (
            offset // self.compress_ratio if self.compress_ratio else 0
        )
        expected_index_emitted = (
            offset // self.compress_ratio if self.compress_ratio == 4 else 0
        )
        if (
            comp_emitted != expected_comp_emitted
            or index_emitted != expected_index_emitted
        ):
            raise ValueError("DeepSeek-V4 NVFP4 compressor frontier is invalid")
        if isinstance(self.window, FixedMiaNVFP4Window):
            _pages, _block_table, state_window_start, state_window_end = state.window
            window_matches = (
                state_window_start == window_start
                and state_window_end == offset
                and state_window_end - state_window_start
                >= min(offset, self.window_size)
            )
        elif state.window is None:
            window_matches = offset == window_start
        elif (
            isinstance(state.window, mx.array)
            and state.window.dtype == mx.uint8
            and state.window.ndim >= 2
            and int(state.window.shape[-1]) == 432
        ):
            window_matches = int(state.window.shape[-2]) == offset - window_start
        else:
            window_matches = False
        if not window_matches:
            raise ValueError(
                "DeepSeek-V4 NVFP4 physical cache frontier is invalid"
            )
        if self.compress_ratio and state.compressed[2] != expected_comp_emitted:
            raise ValueError(
                "DeepSeek-V4 NVFP4 physical cache frontier is invalid"
            )
        if self.compress_ratio == 4 and (
            state.index_compressed is None
            or state.index_compressed[2] != expected_index_emitted
        ):
            raise ValueError(
                "DeepSeek-V4 NVFP4 physical cache frontier is invalid"
            )
        for lane, lane_state, emitted, end, length in (
            (self.comp, state.comp, comp_emitted, comp_end, comp_length),
            (
                self.index_comp,
                state.index_comp,
                index_emitted,
                index_end,
                index_length,
            ),
        ):
            if isinstance(lane, FixedMiaCompressorState):
                lane.validate_journal_meta(end, length)
                lane.validate_current_state(
                    lane_state.cur_kv,
                    lane_state.cur_score,
                    offset,
                )
                lane.validate_previous_meta(
                    lane_state.prev_kv,
                    lane_state.prev_score,
                    emitted,
                )
                if end != offset:
                    raise ValueError(
                        "DeepSeek-V4 NVFP4 journal frontier does not match offset"
                    )
            elif end != 0 or length != 0:
                raise ValueError("non-compressed Mia cache lane cannot restore journal meta")

        return _MiaNVFP4SnapshotMeta(
            offset=offset,
            window_start=window_start,
            comp_emitted=comp_emitted,
            index_emitted=index_emitted,
            comp_end=comp_end,
            comp_length=comp_length,
            index_end=index_end,
            index_length=index_length,
        )

    def _install_snapshot_meta(self, meta: _MiaNVFP4SnapshotMeta) -> None:
        self.offset = meta.offset
        self.window_start = meta.window_start
        self.comp.n_emitted = meta.comp_emitted
        self.index_comp.n_emitted = meta.index_emitted
        if isinstance(self.comp, FixedMiaCompressorState):
            self.comp.replace_journal_meta(meta.comp_end, meta.comp_length)
        if isinstance(self.index_comp, FixedMiaCompressorState):
            self.index_comp.replace_journal_meta(meta.index_end, meta.index_length)

    @meta_state.setter
    def meta_state(self, value) -> None:
        state = self._validate_snapshot_state(self.state)
        meta = self._validate_snapshot_meta(value, state)
        self._install_snapshot_meta(meta)


# ---------------------------------------------------------------------------
# Attention (MQA-shaped MLA + sliding window + optional CSA + o-LoRA)
# ---------------------------------------------------------------------------
class DeepseekV4Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.rms_norm_eps
        self.compress_ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim ** -0.5
        self._mia_qkv_plan = None

        self.attn_sink = mx.zeros((self.n_heads,))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        # o-LoRA: grouped down-projection (block matmul) then a dense up-projection.
        # wo_a stores one [n_groups*o_lora_rank, n_heads*head_dim//n_groups] matrix
        # applied group-wise; see GroupedLoRA / __call__.
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)
        # How _o_lora consumes wo_a, and where the one-time dequant lives.  Both
        # are plain (non-array) attributes, so neither reaches the weight tree.
        self.o_lora_mode = _o_lora_mode_from_env()
        self._wo_a_cache = _DerivedCache()
        self._o_lora_impl = (
            _UninstalledGatherOLora()
            if self.o_lora_mode == "gather_qmm"
            else self._o_lora_dense
        )
        # How _attend forms the score block (see _ATTN_MODES).
        self.attn_mode = _attn_mode_from_env()
        # The generic path retains its historical combined-KV arithmetic.  The
        # exact Mia model replaces these callables once at installation, before
        # any measured execution; the hot path never probes model metadata.
        self._uncached_kv_impl = self._legacy_uncached_kv
        self._uncached_compressed_impl = self._legacy_uncached_compressed
        self._cached_attention_impl = self._legacy_cached_attention
        self._cached_mask_impl = self._attn_mask
        self._finalize_attention_impl = self._inverse_rope_output
        self._output_projection_impl = self._stock_output_projection
        self._forward_impl = self._stock_forward
        self._mia_rope_provider = None
        self._mia_token_rope_tables = None
        self._mia_input_projection = None

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

        # rope frequencies: compressor layers use compress_rope_theta + YaRN;
        # ratio==0 layers use base rope_theta with no YaRN.
        if self.compress_ratio:
            inv = _yarn_inv_freq(
                self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
                args.rope_factor, args.beta_fast, args.beta_slow,
            )
        else:
            inv = _yarn_inv_freq(self.rope_head_dim, args.rope_theta, 0, 1.0, 32, 1)
        self._inv_freq = inv  # [rope_head_dim//2]

    def install_mia_rope_provider(
        self,
        provider: MiaRoPETableProvider,
    ) -> None:
        """Bind one engine-owned base- or compress-theta provider."""
        if self._mia_rope_provider is not None:
            raise ValueError("the Mia attention RoPE provider is already installed")
        self._mia_rope_provider = provider
        self._mia_token_rope_tables = provider.token_tables
        if self.compress_ratio:
            self.compressor.install_mia_rope_provider(provider)
        if self.compress_ratio == 4:
            self.indexer.compressor.install_mia_rope_provider(provider)

    def install_mia_stacked_projection(
        self,
        owner: MiaStackedMXFP8Projection,
    ) -> None:
        """Bind the exact-Mia one-GEMM query-rank/KV owner once."""
        if self._mia_input_projection is not None:
            raise ValueError("the Mia attention projection is already installed")
        self._mia_input_projection = owner

    def _rope_tables(self, positions: mx.array):
        # positions: [L] -> cos/sin [L, rope_head_dim//2]
        ang = positions[:, None].astype(mx.float32) * self._inv_freq[None, :]
        return mx.cos(ang), mx.sin(ang)

    def _legacy_uncached_kv(
        self,
        latent: mx.array,
        rope: mx.array,
    ) -> tuple[mx.array, mx.array]:
        combined = mx.concatenate(
            [latent[..., :-self.rope_head_dim], rope],
            axis=-1,
        )
        return combined, combined

    def _mia_uncached_kv(
        self,
        latent: mx.array,
        rope: mx.array,
    ) -> tuple[mx.array, mx.array]:
        rows = MiaNVFP4Rows()
        rows.append(latent, rope)
        return rows.decode()

    def _legacy_uncached_compressed(
        self,
        compressor: Compressor,
        x: mx.array,
    ) -> tuple[mx.array, mx.array]:
        combined = compressor(x)
        return combined, combined

    def _mia_uncached_compressed(
        self,
        compressor: Compressor,
        x: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return decode_stock432(compressor.mia_records(x))

    def _legacy_cached_attention(
        self,
        queries: mx.array,
        window_rows,
        compressed_rows,
        window_start: int,
        query_positions: mx.array,
        ratio: int,
        compressed_selection: Optional[mx.array],
        additive_mask: Optional[mx.array],
    ) -> mx.array:
        del window_start, query_positions, ratio, compressed_selection
        window_key, window_value = window_rows
        full_key, full_value = window_key, window_value
        if compressed_rows is not None:
            compressed_key, compressed_value = compressed_rows
            full_key = mx.concatenate([window_key, compressed_key], axis=1)
            full_value = mx.concatenate([window_value, compressed_value], axis=1)
        return self._attend(queries, full_key, full_value, additive_mask)

    def _mia_run_installed_attention(
        self,
        queries: mx.array,
        window_records: FixedMiaNVFP4WindowRecords,
        compressed_records,
        window_start: int,
        query_positions: mx.array,
        compressed_indices,
        compressed_lengths: mx.array,
    ) -> mx.array:
        runner = (
            self._nvfp4_prefill_mla
            if current_attention_phase() == "prefill"
            else self._nvfp4_sparse_mla
        )
        return runner(
            queries,
            window_records,
            window_start,
            query_positions,
            compressed_records,
            compressed_indices,
            compressed_lengths,
        )

    def _mia_cached_attention_ratio0(
        self,
        queries: mx.array,
        window_records: FixedMiaNVFP4WindowRecords,
        compressed_records,
        window_start: int,
        query_positions: mx.array,
        ratio: int,
        compressed_selection,
        additive_mask,
    ) -> mx.array:
        del compressed_records, ratio, compressed_selection, additive_mask
        query_count = int(queries.shape[1])
        return self._mia_run_installed_attention(
            queries,
            window_records,
            None,
            window_start,
            query_positions,
            None,
            self._mia_mla_workspace.lengths(query_count),
        )

    def _mia_cached_attention_ratio128(
        self,
        queries: mx.array,
        window_records: FixedMiaNVFP4WindowRecords,
        compressed_records,
        window_start: int,
        query_positions: mx.array,
        ratio: int,
        compressed_selection,
        additive_mask,
    ) -> mx.array:
        del compressed_selection, additive_mask
        compressed_lengths = mx.minimum(
            int(compressed_records.length),
            (query_positions + 1) // int(ratio),
        )[None].astype(mx.int32)
        return self._mia_run_installed_attention(
            queries,
            window_records,
            compressed_records,
            window_start,
            query_positions,
            None,
            compressed_lengths,
        )

    def _mia_cached_attention_ratio4(
        self,
        queries: mx.array,
        window_records: FixedMiaNVFP4WindowRecords,
        compressed_records,
        window_start: int,
        query_positions: mx.array,
        ratio: int,
        compressed_selection: MiaTopKSelection,
        additive_mask,
    ) -> mx.array:
        del ratio, additive_mask
        return self._mia_run_installed_attention(
            queries,
            window_records,
            compressed_records,
            window_start,
            query_positions,
            compressed_selection.indices,
            compressed_selection.lengths,
        )

    def _inverse_rope_output(
        self,
        output: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        return mx.concatenate(
            [
                output[..., :-self.rope_head_dim],
                _apply_interleaved_rope(
                    output[..., -self.rope_head_dim :],
                    cos[None, :, None, :],
                    -sin[None, :, None, :],
                ),
            ],
            axis=-1,
        )

    def _stock_output_projection(self, output, cos, sin):
        output = self._finalize_attention_impl(output, cos, sin)
        batch, sequence = (int(value) for value in output.shape[:2])
        return self._o_lora(
            output.reshape(batch, sequence, self.n_heads * self.head_dim)
        )

    def _project_attention_output(self, output, cos, sin):
        return self._output_projection_impl(output, cos, sin)

    @staticmethod
    def _no_additive_mask(
        q_pos: mx.array,
        kv_pos: Optional[mx.array],
        n_win: int,
        n_comp: int,
        ratio: int,
        dtype,
        comp_sel: Optional[mx.array] = None,
    ) -> None:
        del q_pos, kv_pos, n_win, n_comp, ratio, dtype, comp_sel
        return None

    def install_mia_nvfp4_attention(
        self,
        workspace: MiaMLAWorkspace | None = None,
    ) -> None:
        """Bind exact Mia K/V construction and post-RoPE V output at install."""
        if self.head_dim != 512 or self.rope_head_dim != 64:
            raise ValueError(
                "Mia stock432 attention requires head_dim=512 and rope_head_dim=64"
            )
        if self.compress_ratio:
            self.compressor.install_mia_record_packer("nvfp4")
        self._mia_attn_sink = self.attn_sink.astype(mx.float32)
        mx.eval(self._mia_attn_sink)
        workspace = workspace or mia_mla_workspace()
        self._mia_mla_workspace = workspace
        self._mia_mla_query_layout = "BMHD"
        self._mia_mla_output_layout = "BMHD"
        self._uncached_kv_impl = self._mia_uncached_kv
        self._uncached_compressed_impl = self._mia_uncached_compressed
        self._nvfp4_sparse_mla = partial(
            install_nvfp4_sparse_mla(
                heads=self.n_heads,
                head_dim=self.head_dim,
                rope_dim=self.rope_head_dim,
                window_size=self.window_size,
                compress_ratio=self.compress_ratio,
                workspace=workspace,
            ),
            sinks=self._mia_attn_sink,
            scale=self.softmax_scale,
        )
        self._nvfp4_prefill_mla = partial(
            install_nvfp4_prefill_mla(
                heads=self.n_heads,
                head_dim=self.head_dim,
                rope_dim=self.rope_head_dim,
                window_size=self.window_size,
                compress_ratio=self.compress_ratio,
                workspace=workspace,
            ),
            sinks=self._mia_attn_sink,
            scale=self.softmax_scale,
        )
        if self.compress_ratio == 4:
            # The engine installs the indexer only after allocating its single
            # construction-owned workspace and 384K RoPE table.  No layer may
            # materialize either resource independently here.
            self._forward_impl = self._mia_cached_forward_ratio4
            self._mia_m6_forward_impl = self._mia_m6_forward_ratio4
            self._cached_attention_impl = self._mia_cached_attention_ratio4
        elif self.compress_ratio == 128:
            self._forward_impl = self._mia_cached_forward_ratio128
            self._mia_m6_forward_impl = self._mia_m6_forward_ratio128
            self._cached_attention_impl = self._mia_cached_attention_ratio128
        elif self.compress_ratio == 0:
            self._forward_impl = self._mia_cached_forward_uncompressed
            self._mia_m6_forward_impl = self._mia_m6_forward_uncompressed
            self._cached_attention_impl = self._mia_cached_attention_ratio0
        else:
            raise ValueError(
                f"unsupported Mia compression ratio {self.compress_ratio}"
            )
        self._cached_mask_impl = self._no_additive_mask

    def install_mia_qkv_prologue(self, plan) -> None:
        """Bind the construction-qualified fused Q/KV plan exactly once."""
        if self._mia_qkv_plan is not None:
            raise ValueError("the Mia Q/KV prologue is already installed")
        self._mia_qkv_plan = plan
        self._mia_qkv_impl = self._mia_cached_qkv_records
        self._mia_m6_qkv_impl = self._mia_m6_qkv_records
        self._finalize_attention_impl = self._inverse_rope_output

    def _wo_a_quant(self):
        """``wo_a``'s quantised tensors + format, or ``None`` when it is dense.

        ``(weight, scales, biases, group_size, bits, mode)``.  ``biases`` is
        ``None`` for the bias-free modes (mxfp4); ``mode`` is carried through
        rather than assumed so the affine path stays byte-for-byte what it was.
        """
        wo = self.wo_a
        if not isinstance(wo, nn.QuantizedLinear):
            return None
        return (
            wo.weight,
            wo.scales,
            getattr(wo, "biases", None),
            wo.group_size,
            wo.bits,
            getattr(wo, "mode", "affine"),
        )

    def _wo_a_grouped(self) -> mx.array:
        """``wo_a`` as a dense ``[n_groups, o_lora_rank, per]`` tensor.

        ``wo_a`` is **static**: one ``[g*r, per]`` matrix, the same on every token
        of every step.  On the real checkpoint it is 4-bit, and the pre-cache code
        ran ``mx.dequantize`` on it inside every ``_o_lora`` call — on
        DeepSeek-V4-Flash that is a 64 MiB dense tensor written and re-read per
        layer per decoded token, 43 layers deep, for a value that never changes.
        The reference does the dequant once at load and keeps the dense matrix
        (``wo_a = self.wo_a.weight.view(...)``, model.py L537).

        ``cached`` therefore stores exactly what ``mx.dequantize`` returned, so the
        consuming einsum sees the identical values and the path stays bit-identical
        to ``dequant`` (gated by tests/test_deepseek_v4_o_lora.py).  Resident cost
        is one dense copy per layer beside the quantised one it is derived from —
        2.69 GiB across 43 layers on DeepSeek-V4-Flash.

        Do not read the byte count above as a speed claim: measured on the real
        checkpoint at fp32 activation storage, ``cached`` vs ``dequant`` is +2.1%
        AR, inside cross-window drift (bench/deepseek-v4/goal-ab-20260731).  At
        fp32 the einsum promotes ``wo_a`` regardless, so caching removes the
        dequantize and not the cast behind it.  The measured decode win in this
        lane is the activation-dtype fix, not this.
        """
        g = self.n_groups
        r = self.o_lora_rank
        per = self.n_heads * self.head_dim // g
        q = self._wo_a_quant()
        if q is None:
            # Unquantised (the M2/parity path): wo_a is a plain nn.Linear.
            return self.wo_a.weight.reshape(g, r, per)
        w, scales, biases, group_size, bits, mode = q
        src = (w, scales, biases)
        if self.o_lora_mode != "dequant":
            hit = self._wo_a_cache.get(src)
            if hit is not None:
                return hit
        dense = mx.dequantize(
            w, scales, biases, group_size=group_size, bits=bits, mode=mode
        ).reshape(g, r, per)
        if self.o_lora_mode == "dequant":
            return dense
        return self._wo_a_cache.put(src, dense)

    def install_o_lora_route(self, mode: str | None = None) -> dict:
        """Validate and bind one o-LoRA route at an installation boundary."""
        selected = self.o_lora_mode if mode is None else str(mode)
        if selected not in _O_LORA_MODES:
            raise ValueError(f"unsupported o-LoRA route {selected!r}")
        if selected == "gather_qmm":
            quant = self._wo_a_quant()
            if quant is None:
                raise ValueError(
                    "gather_qmm o-LoRA requires a quantized wo_a; dense fallback "
                    "is forbidden"
                )
            installed = _DirectGatherOLora(self, quant)
            output_projection = self._stock_output_projection
            direct = True
        else:
            installed = self._o_lora_dense
            output_projection = self._stock_output_projection
            direct = False
        self.o_lora_mode = selected
        self._o_lora_impl = installed
        self._output_projection_impl = output_projection
        return {
            "mode": selected,
            "direct": direct,
            "groups": int(self.n_groups),
            "rank": int(self.o_lora_rank),
            "per_group_input": int(
                self.n_heads * self.head_dim // self.n_groups
            ),
        }

    def _o_lora_dense(self, o: mx.array) -> mx.array:
        """Grouped output-LoRA (reference model.py L536-542).

        ``o``: ``[b, s, n_heads*head_dim]`` -> reshape ``[b, s, n_groups, per]``;
        each group projects ``per -> o_lora_rank`` by its own slice of ``wo_a``;
        concat to ``n_groups*o_lora_rank`` then ``wo_b`` -> dim.

        See :data:`_O_LORA_MODES` for the three ways ``wo_a`` gets there.
        """
        b, s, _ = o.shape
        g = self.n_groups
        per = self.n_heads * self.head_dim // g
        r = self.o_lora_rank
        og = o.reshape(b, s, g, per)
        w = self._wo_a_grouped()  # [g, r, per]
        # out[b,s,g,r] = sum_p og[...,g,p] * w[g,r,p]
        out = mx.einsum("bsgp,grp->bsgr", og, w)
        out = out.reshape(b, s, g * r)
        return self.wo_b(out)

    def _o_lora(self, o: mx.array) -> mx.array:
        """Execute the prebound route with no enabled-path eligibility branch."""
        return self._o_lora_impl(o)

    def _attn_mask(
        self,
        q_pos: mx.array,
        kv_pos: Optional[mx.array],
        n_win: int,
        n_comp: int,
        ratio: int,
        dtype,
        comp_sel: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Additive ``[b, 1, s, n_win + n_comp]`` mask reproducing the reference
        sparse gather: a query attends the causal sliding window over the per-position
        KV, plus the compressed windows selected for it.

        ``q_pos``/``kv_pos`` are *absolute* positions, so the same rule covers the
        one-shot prefill (both ``arange(s)``) and a cached chunk whose KV rows start
        before the queries.  ``kv_pos is None`` means the caller already dropped every
        unattendable window row (the ``s == 1`` decode step), so that half needs no
        mask.  ``comp_sel`` is the indexer's ``[b, s, n_comp]`` decision; without it
        the compressed half falls back to plain causality, i.e. every compressed row
        the query has completed — which is what the indexer itself returns whenever
        ``n_comp <= index_topk``.

        Returns ``None`` when there is nothing to mask.
        """
        if kv_pos is None and comp_sel is None:
            return None
        s = int(q_pos.shape[0])
        parts = []
        if kv_pos is not None:
            i = q_pos[:, None]
            j = kv_pos[None, :]
            parts.append(((j <= i) & (j > i - self.window_size))[None])   # [1, s, n_win]
        elif n_win:
            parts.append(mx.ones((1, s, n_win), dtype=mx.bool_))
        if n_comp:
            if comp_sel is None:
                c = mx.arange(n_comp)[None, :]
                parts.append((c < ((q_pos[:, None] + 1) // ratio))[None])
            else:
                parts.append(comp_sel)
        b = max(int(p.shape[0]) for p in parts)
        if len(parts) == 1:
            ok = parts[0]
        else:
            ok = mx.concatenate(
                [mx.broadcast_to(p, (b, s, p.shape[2])) for p in parts], axis=-1
            )
        neg = mx.array(mx.finfo(dtype).min, dtype)
        return mx.where(ok, mx.array(0.0, dtype), neg)[:, None]

    def _attend(
        self,
        q_t: mx.array,
        full_key: mx.array,
        full_value: mx.array,
        add,
    ) -> mx.array:
        """``softmax(q.k^T + mask, with attn_sink in the denominator) . v``.

        ``q_t``: ``[b, h, s, head_dim]``.  ``full_key`` and ``full_value`` are
        ``[b, n_kv, head_dim]`` shared over heads (MQA-shaped MLA).
        ``add``: the additive ``[b, 1, s, n_kv]`` window+compressed mask, or
        ``None`` when every column is attendable.

        **The sink.**  ``attn_sink`` is a per-head learned logit that appears only
        in the softmax denominator — the head can decide to attend to nothing.
        Writing it as one extra KV column makes it ordinary attention: the
        appended row is all zeros, so its raw score is *exactly* 0 whatever the
        query is, an additive mask column carries the sink itself, and because
        the same zero row is also the V row it contributes exactly nothing to the
        numerator.  The whole block is then a single softmax, and MLX's is
        ``precise``: fp32 max and fp32 accumulation with bf16 in and out, which is
        what the reference kernel does with its FP32 fragments (kernel.py
        L298/L305/L308-314) and what ``dense`` could only get by materialising the
        entire block in fp32 twice over.

        **Why not ``mx.fast.scaled_dot_product_attention`` by default.**  MLX
        0.31.2 does take ``sinks=`` natively and the ``sdpa`` arm below uses it —
        it is the same computation in one op.  But its fused Metal kernels are
        only instantiated for head dims 64/96/128/256 (vector) and 64/80/128
        (full) — ``ScaledDotProductAttention::use_fallback``, metal/
        scaled_dot_product_attention.cpp L618-636 — and DeepSeek-V4's MLA latent
        is 512 wide, so on this box every call would take MLX's *own* unfused
        fallback: the same matmul/softmax/matmul, plus a ``concatenate`` of the
        sink column and a ``slice`` to remove it again, i.e. two extra passes over
        the full block.  ``fused`` is that fallback minus the two copies.  The arm
        is kept, and kept exact, because the day MLX instantiates head_dim 512
        (or the model is served through an absorbed-MLA rewrite that lands on a
        supported dim) it becomes one kernel with no code change — that is the A/B
        the mlx-0.32 venv arm is for.
        """
        if self.attn_mode == "sdpa":
            # MLX appends and removes the sink column itself; the KV block stays
            # exactly as built.  ``sinks`` must not promote past the value dtype.
            kt = full_key[:, None]
            vt = full_value[:, None]
            return mx.fast.scaled_dot_product_attention(
                q_t,
                kt,
                vt,
                scale=self.softmax_scale,
                mask=add,
                sinks=self.attn_sink.astype(kt.dtype),
            )

        if self.attn_mode == "dense":
            kt = full_key[:, None]
            vt = full_value[:, None]
            scores = (q_t * self.softmax_scale) @ mx.swapaxes(kt, -1, -2)
            if add is not None:
                scores = scores + add
            # attn_sink: per-head learned logit in the softmax denominator.  The
            # softmax itself runs in fp32 — the reference kernel keeps acc_s /
            # scores_max / sum_exp in FP32 fragments and its attn_sink parameter
            # is fp32 (kernel.py L298/L308-314, model.py L457) — but the
            # probability block is cast back to the KV dtype before the PV gemm
            # (``acc_s_cast`` is BF16, kernel.py L305/L340) and ``o`` is written
            # at the model dtype (``o: T.Tensor[(b,m,h,d), BF16]``, L297).
            # Keeping the probabilities fp32 here would promote kt for the second
            # matmul and hand an fp32 ``o`` to the o-LoRA einsum, which then has
            # to upcast wo_a as well.
            sink = self.attn_sink.reshape(1, self.n_heads, 1, 1).astype(mx.float32)
            sf = scores.astype(mx.float32)
            m = mx.maximum(mx.max(sf, axis=-1, keepdims=True), sink)
            ex = mx.exp(sf - m)
            denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
            return (ex / denom).astype(vt.dtype) @ vt

        # "fused": one zero K/V row carries the sink column.
        kt = mx.pad(full_key, [(0, 0), (0, 1), (0, 0)])[:, None]
        vt = mx.pad(full_value, [(0, 0), (0, 1), (0, 0)])[:, None]
        scores = (q_t * self.softmax_scale) @ mx.swapaxes(kt, -1, -2)
        if add is not None:
            scores = scores + mx.pad(add, [(0, 0), (0, 0), (0, 0), (0, 1)])
        sink = self.attn_sink.reshape(1, self.n_heads, 1, 1).astype(scores.dtype)
        scores = scores + mx.pad(
            sink, [(0, 0), (0, 0), (0, 0), (int(full_key.shape[1]), 0)]
        )
        return mx.softmax(scores, axis=-1, precise=True) @ vt

    def _indexer_active(self, n_comp: int) -> bool:
        """Is the top-k filter load-bearing for this call?

        Below the threshold ``min(index_topk, n_comp) == n_comp``, so the indexer would
        select every causally-available row and return exactly the dense causal mask.
        Skipping the whole scoring path there is not just an optimisation: it keeps the
        short-context regime bit-identical to the pre-filter backend (ds4.c takes the
        same early-out — ``if (top_k == n_comp) { all allowed }``).
        """
        return self.compress_ratio == 4 and n_comp > self.indexer.index_topk

    def _mia_cached_qkv_records(self, x: mx.array, cache):
        """Project Q once and finalize exact stock432 cache records."""
        batch, sequence, _ = x.shape
        offset = int(cache.offset)
        positions, cos, sin = self._mia_token_rope_tables(offset, sequence)
        query_rank, latent = self._mia_qkv_plan.project_learned(x)
        query_pre = self.wq_b(query_rank).reshape(
            batch,
            sequence,
            self.n_heads,
            self.head_dim,
        )
        finalizer = (
            self._mia_qkv_plan.prefill_records
            if current_attention_phase() == "prefill"
            else self._mia_qkv_plan.target_records
        )
        query, records = finalizer(
            query_pre,
            latent,
            cos,
            sin,
        )
        return (
            query,
            query_rank,
            records,
            positions,
            cos,
            sin,
        )

    def _mia_m6_qkv_records(self, x: mx.array, cache):
        """Finalize physical-M6 Q/KV records through the target-only plan."""
        batch, sequence, _ = x.shape
        offset = int(cache.offset)
        positions, cos, sin = self._mia_token_rope_tables(offset, sequence)
        query_rank, latent = self._mia_qkv_plan.project_learned(x)
        query_pre = self.wq_b(query_rank).reshape(
            batch,
            sequence,
            self.n_heads,
            self.head_dim,
        )
        query, records = self._mia_qkv_plan.target_records(
            query_pre,
            latent,
            cos,
            sin,
        )
        return (
            query,
            query_rank,
            records,
            positions,
            cos,
            sin,
        )

    def _mia_finish_cached(
        self,
        query: mx.array,
        window_records: FixedMiaNVFP4WindowRecords,
        compressed_records,
        window_start: int,
        positions: mx.array,
        compressed_selection,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        output = self._cached_attention_impl(
            query,
            window_records,
            compressed_records,
            window_start,
            positions,
            self.compress_ratio,
            compressed_selection,
            None,
        )
        return self._project_attention_output(
            output,
            cos,
            sin,
        )

    def _mia_cached_forward_uncompressed(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        del mask
        query, _qr, records, positions, cos, sin = (
            self._mia_qkv_impl(x, cache)
        )
        window_records, window_start = cache.update_window_records(records)
        cache.advance(int(x.shape[1]))
        return self._mia_finish_cached(
            query,
            window_records,
            None,
            window_start,
            positions,
            None,
            cos,
            sin,
        )

    def _mia_cached_forward_ratio128(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        del mask
        query, _qr, records, positions, cos, sin = (
            self._mia_qkv_impl(x, cache)
        )
        cache.update_compressed(self.compressor, x)
        window_records, window_start = cache.update_window_records(records)
        compressed_records = cache.compressed.paged_records
        cache.advance(int(x.shape[1]))
        return self._mia_finish_cached(
            query,
            window_records,
            compressed_records,
            window_start,
            positions,
            None,
            cos,
            sin,
        )

    def _mia_cached_forward_ratio4(
        self,
        x: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        del mask
        query, query_rank, records, positions, cos, sin = (
            self._mia_qkv_impl(x, cache)
        )
        cache.update_compressed(self.compressor, x)
        cache.update_index_compressed(self.indexer.compressor, x)
        window_records, window_start = cache.update_window_records(records)
        compressed_count = cache.n_compressed
        compressed_records = cache.compressed.paged_records
        if compressed_count > self.indexer.index_topk:
            compressed_selection = self.indexer(
                x,
                query_rank,
                positions,
                cache.index_compressed.paged_records,
            )
        else:
            compressed_selection = MiaTopKSelection(
                indices=self._mia_mla_workspace.identity_indices(
                    int(query.shape[1])
                ),
                lengths=mx.minimum(
                    compressed_count,
                    (positions + 1) // 4,
                )[None].astype(mx.int32),
            )
        cache.advance(int(x.shape[1]))
        return self._mia_finish_cached(
            query,
            window_records,
            compressed_records,
            window_start,
            positions,
            compressed_selection,
            cos,
            sin,
        )

    def _mia_m6_forward_uncompressed(
        self,
        x: mx.array,
        *,
        cache,
        schedule,
    ) -> mx.array:
        del schedule
        query, _query_rank, records, positions, cos, sin = (
            self._mia_m6_qkv_impl(x, cache)
        )
        window_records, window_start = cache.update_window_records(records)
        cache.advance(int(x.shape[1]))
        output = self._nvfp4_sparse_mla(
            query,
            window_records,
            window_start,
            positions,
            None,
            None,
            self._mia_mla_workspace.lengths(int(query.shape[1])),
        )
        return self._project_attention_output(output, cos, sin)

    def _mia_m6_forward_ratio128(
        self,
        x: mx.array,
        *,
        cache,
        schedule,
    ) -> mx.array:
        query, _query_rank, records, positions, cos, sin = (
            self._mia_m6_qkv_impl(x, cache)
        )
        cache.update_compressed_m6(self.compressor, x, schedule)
        window_records, window_start = cache.update_window_records(records)
        compressed_records = cache.compressed.paged_records
        cache.advance(int(x.shape[1]))
        output = self._nvfp4_sparse_mla(
            query,
            window_records,
            window_start,
            positions,
            compressed_records,
            None,
            schedule.causal_lengths[None],
        )
        return self._project_attention_output(output, cos, sin)

    def _mia_m6_forward_ratio4(
        self,
        x: mx.array,
        *,
        cache,
        schedule,
    ) -> mx.array:
        query, query_rank, records, positions, cos, sin = (
            self._mia_m6_qkv_impl(x, cache)
        )
        cache.update_compressed_m6(self.compressor, x, schedule)
        cache.update_index_compressed_m6(
            self.indexer.compressor,
            x,
            schedule,
        )
        window_records, window_start = cache.update_window_records(records)
        compressed_records = cache.compressed.paged_records
        if cache.n_compressed > self.indexer.index_topk:
            compressed_selection = self.indexer._mia_m6_select(
                x,
                query_rank,
                positions,
                cache.index_compressed.paged_records,
                schedule,
            )
        else:
            compressed_selection = MiaTopKSelection(
                indices=self._mia_mla_workspace.identity_indices(
                    int(query.shape[1])
                ),
                lengths=schedule.causal_lengths[None],
            )
        cache.advance(int(x.shape[1]))
        output = self._nvfp4_sparse_mla(
            query,
            window_records,
            window_start,
            positions,
            compressed_records,
            compressed_selection.indices,
            compressed_selection.lengths,
        )
        return self._project_attention_output(output, cos, sin)

    def _stock_forward(self, x: mx.array, mask=None, cache=None) -> mx.array:
        # Attends the causal sliding window over per-position KV plus the compressed
        # KV rows the ratio-4 indexer selects (every causal row on ratio-128 layers,
        # and on ratio-4 layers below index_topk) — the dense-mask equivalent of the
        # reference sparse_attn + topk_idxs gather.
        # `cache is None` runs the whole sequence in one shot (the parity-gated path);
        # otherwise the same math runs incrementally off DeepseekV4Cache.  `mask` is
        # built internally either way — it needs the compressed-position columns.
        b, s, _ = x.shape
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        offset = 0 if cache is None else cache.offset
        positions = mx.arange(offset, offset + s)
        cos, sin = self._rope_tables(positions)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        # per-head RMS-like normalisation (no learned weight), reference L498
        q = q * mx.rsqrt(mx.mean(mx.square(q.astype(mx.float32)), axis=-1, keepdims=True) + self.eps)
        q = q.astype(x.dtype)
        q = mx.concatenate(
            [q[..., :-rd], _apply_interleaved_rope(q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :])],
            axis=-1,
        )

        latent = self.kv_norm(self.wkv(x))  # [b, s, head_dim] (single shared MLA row)
        rope = _apply_interleaved_rope(
            latent[..., -rd:],
            cos[None, :, :],
            sin[None, :, :],
        )

        # concat the compressor's compressed KV (reference cats kv + kv_compress)
        comp_sel = None
        if cache is None:
            full_key, full_value = self._uncached_kv_impl(latent, rope)
            n_comp = 0
            n_win = s
            if ratio:
                comp_key, comp_value = self._uncached_compressed_impl(
                    self.compressor,
                    x,
                )
                n_comp = comp_key.shape[1]
                if n_comp:
                    full_key = mx.concatenate([full_key, comp_key], axis=1)
                    full_value = mx.concatenate([full_value, comp_value], axis=1)
                    if self._indexer_active(n_comp):
                        # No cache to keep, so the indexer's compressor only runs when
                        # its rows are actually about to be scored.
                        comp_sel = self.indexer(
                            x, qr, positions, self.indexer.compressor(x)
                        )
            kv_pos = positions
        else:
            # Compressor first: the window a token *completes* is attendable by that
            # same token (mask rule `c < (i+1)//ratio`), so it must land in the cache
            # before this step's scores are formed.  Order copied from ds4.c's decode
            # layer (push raw KV, compressor_decode_one, index compressor_decode_one,
            # indexer selection, then mixed attention).
            if ratio:
                cache.update_compressed(self.compressor, x)
                if ratio == 4:
                    cache.update_index_compressed(self.indexer.compressor, x)
            window_rows, win_start = cache.update_window(latent, rope)
            n_comp = cache.n_compressed
            n_win = cache.attention_window_length(window_rows)
            compressed_rows = cache.attention_compressed() if n_comp else None
            if self._indexer_active(n_comp):
                assert cache.n_index_compressed == n_comp, (
                    "indexer compressor lane desynced from the attention lane: "
                    f"{cache.n_index_compressed} vs {n_comp}"
                )
                comp_sel = self.indexer(
                    x,
                    qr,
                    positions,
                    cache.attention_index_compressed(),
                )
            # s == 1: update_window already dropped every row outside the query's
            # window, so that half needs no mask (the compressed half still does once
            # the indexer is filtering).
            kv_pos = None if s == 1 else mx.arange(
                win_start, win_start + n_win
            )
            cache.advance(s)

        q_t = q.transpose(0, 2, 1, 3)          # [b, h, s, head_dim]
        # K/V are [b, s+n_comp, head_dim] and shared over heads (MQA).
        # q and both blocks always carry the same dtype (both follow x, or both
        # follow the fp32 escape hatch), so either one names the score dtype.
        if cache is None:
            add = self._attn_mask(
                positions,
                kv_pos,
                n_win,
                n_comp,
                ratio,
                q_t.dtype,
                comp_sel=comp_sel,
            )
            o = self._attend(q_t, full_key, full_value, add)
        else:
            # The exact Mia route consumes record owners here; the generic route
            # receives decoded pairs.  Both are installed callables, not an
            # eligibility/fallback decision in the measured path.
            add = self._cached_mask_impl(
                positions,
                kv_pos,
                n_win,
                n_comp,
                ratio,
                q_t.dtype,
                comp_sel=comp_sel,
            )
            o = self._cached_attention_impl(
                q_t,
                window_rows,
                compressed_rows,
                win_start,
                positions,
                ratio,
                comp_sel,
                add,
            )
        o = o.transpose(0, 2, 1, 3)            # [b, s, h, head_dim]
        return self._project_attention_output(o, cos, sin)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        return self._forward_impl(x, mask=mask, cache=cache)


# ---------------------------------------------------------------------------
# MoE (gate: sqrtsoftplus / hash / noaux bias  +  SwitchGLU + shared expert)
# ---------------------------------------------------------------------------
class DeepseekV4MLP(nn.Module):
    """Shared-expert / dense MLP with the reference's swiglu clamp (limit=10).

    Reference ``Expert.forward`` (model.py L596-606), verbatim::

        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up

    Note the asymmetry, which is easy to get wrong in both directions: the *up*
    branch (``w3`` = ``up_proj``) is clipped to ``[-limit, +limit]``, the *gate*
    branch (``w1`` = ``gate_proj``) only has its upper tail cut at ``+limit`` and
    keeps its whole negative range.  Both cuts land on the pre-activation
    projections, before ``silu``.  ``limit <= 0`` disables the clamp entirely,
    which is what both parity goldens were captured at.
    """

    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self._mia_stacked_projection = None
        self._gate_up_impl = self._stock_gate_up

    def _stock_gate_up(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.gate_proj(x), self.up_proj(x)

    def install_mia_stacked_gate_up(
        self,
        owner: MiaStackedMXFP8Projection,
    ) -> None:
        """Bind vLLM's merged shared-expert column projection once."""

        if self._mia_stacked_projection is not None:
            raise ValueError("the Mia shared-expert gate/up owner is already installed")
        self._mia_stacked_projection = owner
        self._gate_up_impl = owner

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = self._gate_up_impl(x)
        if self.limit and self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class ClampedSwiGLU(SwiGLU):
    """``SwitchGLU`` activation carrying the reference's ``swiglu_limit`` clamp.

    The reference applies the clamp inside *every* expert, routed ones included
    (``MoE.__init__`` L624 passes ``swiglu_limit=args.swiglu_limit`` to each
    routed :class:`Expert`, exactly as L627 does for the shared one).  Routed
    experts here run through mlx-lm's :class:`SwitchGLU`, whose only seam is the
    ``activation`` module it calls between the ``up``/``gate`` projections and
    ``down_proj`` — which is precisely where the reference's clamp sits.  So the
    faithful port is an activation, not a fork of ``SwitchGLU``: the batched
    ``gather_mm``/``gather_qmm`` expert kernels are untouched.

    ``SwitchGLU.__call__`` invokes ``self.activation(x_up, x_gate)``, so the
    first argument is the *up* branch and the second is the *gate* branch — the
    opposite of the reading the names suggest.  The clamp is asymmetric between
    them; see :class:`DeepseekV4MLP` for the quoted reference lines.

    At ``limit <= 0`` this defers to :class:`SwiGLU` untouched, so the disabled
    path is the stock fused ``swiglu`` kernel and stays bit-identical to a model
    built without this class at all (both parity goldens were captured there).
    Holds no parameters, so the load path and the weight tree are unchanged.
    """

    def __init__(self, limit: float = 0.0):
        super().__init__()
        self.limit = float(limit or 0.0)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        if self.limit > 0:
            x = mx.clip(x, -self.limit, self.limit)      # up:   two-sided
            gate = mx.minimum(gate, self.limit)          # gate: upper tail only
        return super().__call__(x, gate)


def _stock_moe_tail_combine(
    routed: mx.array, weights: mx.array, shared: mx.array
) -> mx.array:
    """The unfused MoE tail, retained as the construction-time stock route."""
    return (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2) + shared


class MoEGate(nn.Module):
    """Reference ``Gate`` (model.py L546-584): sqrtsoftplus scoring, bias-corrected
    (noaux_tc) top-k for score layers, or fixed tid2eid lookup for hash layers.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.dim = args.hidden_size
        self.topk = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed = args.n_routed_experts
        self.hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((self.n_routed, self.dim))
        if self.hash:
            self.tid2eid = mx.zeros((args.vocab_size, self.topk), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self._route_impl = self._stock_route

    def _score(self, x: mx.array) -> mx.array:
        s = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        if self.score_func == "softmax":
            return mx.softmax(s, axis=-1)
        if self.score_func == "sigmoid":
            return mx.sigmoid(s)
        # sqrtsoftplus
        return mx.sqrt(nn.softplus(s))

    def _stock_route(self, x: mx.array, input_ids: Optional[mx.array] = None):
        scores = self._score(x)  # [n, n_routed]
        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids.reshape(-1)]  # [n, topk]
        else:
            biased = scores + self.e_score_correction_bias
            indices = mx.argpartition(-biased, kth=self.topk - 1, axis=-1)[..., : self.topk]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.score_func != "softmax":
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True))
        weights = weights * self.route_scale
        return indices, weights

    def _mia_score_route(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = self._mia_projection(x, self.weight)
        return self._mia_router(logits, self.e_score_correction_bias, input_ids)

    def _mia_hash_route(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = self._mia_projection(x, self.weight)
        return self._mia_router(logits, self.tid2eid, input_ids)

    def install_mia_router(self) -> None:
        """Install the pinned fused-topk-bias route before generation."""
        if (
            self.score_func != "sqrtsoftplus"
            or not self.norm_topk_prob
            or self.topk != 6
            or self.n_routed not in (64, 216)
            or self.dim != 4096
            or self.route_scale != 1.5
            or tuple(self.weight.shape) != (self.n_routed, self.dim)
            or self.weight.dtype != mx.bfloat16
        ):
            raise ValueError(
                "Mia router requires BF16 K4096 K64/K216 normalized "
                "sqrtsoftplus top-6 routing scaled by 1.5"
            )
        from mtplx.kernels.deepseek_v4_moe_router import (
            install_hash_router,
            install_router_projection,
            install_score_router,
        )

        self._mia_projection = install_router_projection(experts=self.n_routed)

        if self.hash:
            if (
                self.n_routed != 216
                or tuple(self.tid2eid.shape) != (129280, self.topk)
                or self.tid2eid.dtype != mx.int32
            ):
                raise ValueError("Mia hash router table contract changed")
            self._mia_router = install_hash_router(
                experts=self.n_routed,
                topk=self.topk,
                route_scale=self.route_scale,
            )
            self._route_impl = self._mia_hash_route
            route_kind = "hash_tid2eid_int32"
        else:
            if (
                tuple(self.e_score_correction_bias.shape) != (self.n_routed,)
                or self.e_score_correction_bias.dtype != mx.float32
            ):
                raise ValueError("Mia score correction contract changed")
            self._mia_router = install_score_router(
                experts=self.n_routed,
                topk=self.topk,
                route_scale=self.route_scale,
            )
            self._route_impl = self._mia_score_route
            route_kind = "bias_selection_fp32"
        self._mia_router_contract = (
            "bf16xbf16_fp32_k4096",
            "softplus_threshold20_sqrt",
            route_kind,
            "unbiased_normalize_scale1p5",
        )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        return self._route_impl(x, input_ids)


class DeepseekV4MoE(nn.Module):
    """Routed experts + one shared expert (reference ``MoE``, model.py L609-644).

    ``swiglu_limit`` reaches both halves: the routed experts through
    :class:`ClampedSwiGLU` (the ``SwitchGLU`` activation seam) and the shared one
    through :class:`DeepseekV4MLP`, matching L624/L627 where the reference hands
    the same limit to both.  This constructor is the *only* place the backend
    builds routed experts, so trunk score layers, trunk hash layers and the
    :class:`DeepseekV4MTP` draft block (a :class:`DeepseekV4DecoderLayer`
    subclass) are all covered by construction rather than by three call sites
    kept in sync.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.gate = MoEGate(args, layer_id)
        if args.hybrid_tr3_tail is not None and layer_id < args.num_hidden_layers:
            from mtplx.deepseek_v4_exl3 import EXL3SwitchGLU

            self.switch_mlp = EXL3SwitchGLU(
                args.hidden_size,
                args.moe_intermediate_size,
                args.n_routed_experts,
                args.num_experts_per_tok,
                limit=args.swiglu_limit,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.moe_intermediate_size,
                args.n_routed_experts,
                activation=ClampedSwiGLU(args.swiglu_limit),
            )
        self.shared_experts = DeepseekV4MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )
        # Weight storage does not exist yet.  Production keeps this explicit
        # stock route until ``configure_deepseek_v4_moe_tail`` validates the
        # fully loaded model and prebinds the candidate at the runtime boundary.
        self._tail_combine = _stock_moe_tail_combine
        self._forward_impl = self._stock_forward
        self._input_rows_impl = self._optional_input_rows

    @staticmethod
    def _optional_input_rows(input_ids: Optional[mx.array]):
        return input_ids.reshape(-1) if input_ids is not None else None

    @staticmethod
    def _required_input_rows(input_ids: mx.array):
        return input_ids.reshape(-1)

    def _stock_forward(
        self, xf: mx.array, ids: Optional[mx.array]
    ) -> mx.array:
        indices, weights = self.gate(xf, ids)
        y = self.switch_mlp(xf, indices)
        y = self._tail_combine(y, weights, self.shared_experts(xf))
        return y

    def _mia_exl3_forward(
        self, xf: mx.array, ids: Optional[mx.array]
    ) -> mx.array:
        indices, weights = self.gate(xf, ids)
        phase = current_attention_phase()
        if phase == "decode_verify" and int(xf.shape[0]) == 6:
            return self._mia_exl3_m6_fused(
                xf,
                indices,
                weights,
                self.shared_experts(xf),
            )
        return self._mia_exl3_trellis_fused(
            xf,
            indices,
            weights,
            self.shared_experts(xf),
        )

    def install_mia_exl3_runtime(self, *, max_tokens: int) -> None:
        from mtplx.deepseek_v4_exl3 import EXL3SwitchGLU

        if not isinstance(self.switch_mlp, EXL3SwitchGLU):
            raise ValueError("Mia EXL3 runtime requires rank-sliced expert banks")
        self.gate.install_mia_router()
        self.switch_mlp.install_trellis_runtime(max_tokens=max_tokens)
        self._mia_exl3_direct_qmv = self.switch_mlp.direct_qmv
        self._mia_exl3_trellis_fused = self.switch_mlp.fused
        self._mia_exl3_tail_combine = _stock_moe_tail_combine
        self._forward_impl = self._mia_exl3_forward
        self._input_rows_impl = self._required_input_rows

    def install_mia_k64_runtime(self) -> None:
        """Seal the physical-M5 packed K64 route and required token IDs."""

        from mtplx.deepseek_v4_mia_draft_moe import (
            install_mia_k64_packed_switch,
        )

        switch = self.switch_mlp
        observed = (
            tuple(int(value) for value in switch.gate_proj.weight.shape),
            tuple(int(value) for value in switch.gate_proj.scales.shape),
            tuple(int(value) for value in switch.up_proj.weight.shape),
            tuple(int(value) for value in switch.up_proj.scales.shape),
            tuple(int(value) for value in switch.down_proj.weight.shape),
            tuple(int(value) for value in switch.down_proj.scales.shape),
        )
        expected = (
            (64, 2048, 512),
            (64, 2048, 128),
            (64, 2048, 512),
            (64, 2048, 128),
            (64, 4096, 256),
            (64, 4096, 64),
        )
        if observed != expected:
            raise ValueError(
                f"Mia physical-M5 K64 geometry changed: {observed!r} != {expected!r}"
            )

        self.gate.install_mia_router()
        self.switch_mlp = install_mia_k64_packed_switch(switch)
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        self._input_rows_impl = self._required_input_rows

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        ids = self._input_rows_impl(input_ids)
        y = self._forward_impl(xf, ids)
        return y.reshape(shape)


# ---------------------------------------------------------------------------
# Decoder block (Hyper-Connections around attn + MoE)
# ---------------------------------------------------------------------------
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attn = DeepseekV4Attention(args, layer_id)
        self.ffn = DeepseekV4MoE(args, layer_id)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = HyperConnection(
            args.hidden_size, args.hc_mult, args.hc_eps, args.hc_sinkhorn_iters
        )
        self.ffn_hc = HyperConnection(
            args.hidden_size, args.hc_mult, args.hc_eps, args.hc_sinkhorn_iters
        )

    def __call__(self, h: mx.array, mask=None, cache=None, input_ids=None) -> mx.array:
        # h: [b, s, hc, dim]
        residual = h
        x, post, comb = self.attn_hc.pre(h)
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask, cache=cache)
        h = self.attn_hc.post(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc.pre(h)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids=input_ids)
        h = self.ffn_hc.post(x, residual, post, comb)
        return h


# ---------------------------------------------------------------------------
# Multi-token-prediction draft block
# ---------------------------------------------------------------------------
class DeepseekV4MTP(DeepseekV4DecoderLayer):
    """Speculative-decode draft block (reference ``MTPBlock``, model.py L738-766).

    **What it owns.**  ``MTPBlock`` subclasses ``Block``, so the draft head is a
    full decoder layer in its own right: its own attention, its own 256-expert
    MoE, its own ``attn_norm``/``ffn_norm`` and its own two Hyper-Connection
    blocks — none of it shared with the trunk.  On top of a body block it adds
    six pieces (L742-752): ``enorm``/``hnorm`` normalise the two inputs,
    ``e_proj``/``h_proj`` project and sum them, and ``norm`` + ``hc_head`` do the
    final collapse that the trunk does with ``model.norm`` + ``model.hc_head``.
    Every one of those ships upstream under ``mtp.0.*``.

    **What it shares.**  Exactly two things, and it holds no copy of either:
    the token embedding and the output projection.  ``Transformer.__init__``
    L792-793 assigns ``mtp[i].embed = self.embed`` and ``mtp[i].head = self.head``
    after constructing the block, so the draft's logits land in the same
    vocabulary space as the target's — which is what makes accept/reject a
    comparison of like with like.  Both are therefore passed *in* to
    :meth:`__call__` rather than stored, so the 129280-row embedding and lm_head
    are never duplicated in memory.

    **Which layer it is.**  ``layer_id = n_layers + i`` (L791) — 43 on
    DeepSeek-V4-Flash — and that index is what the inherited ``Attention`` and
    ``Gate`` read.  ``compress_ratios[43] == 0`` in the shipped config, so the
    draft block is a **pure sliding-window** attention layer: base ``rope_theta``,
    no YaRN, no :class:`Compressor`, no :class:`Indexer`.  ``43 >=
    num_hash_layers`` (3), so its gate is score-routed (``noaux_tc`` bias), not
    hash-routed.  Both fall out of the inherited constructor rather than being
    re-decided here.

    **Forward** (L757-766).  ``h`` is the trunk's pre-head Hyper-Connection state
    ``[b, s, hc, dim]`` (:meth:`DeepseekV4Model.hc_hidden`), ``input_ids`` are the
    tokens whose *embeddings* get fused in — the caller aligns them, and for
    speculative decode that means position ``i`` of ``input_ids`` is the token the
    trunk predicted *at* ``h[:, i]``, i.e. shifted one ahead of the ids that
    produced ``h``.  The block does not shift anything itself; the reference
    does not either.
    """

    def __init__(self, args: ModelArgs, layer_id: Optional[int] = None):
        layer_id = args.num_hidden_layers if layer_id is None else int(layer_id)
        ratios = list(args.compress_ratios)
        if len(ratios) <= layer_id:
            # The shipped config carries the MTP layer's entry (44 ratios for 43
            # layers, trailing 0).  A config trimmed to the trunk length gets the
            # same value rather than an IndexError out of Attention.__init__.
            ratios = ratios + [0] * (layer_id + 1 - len(ratios))
            args = replace(args, compress_ratios=ratios)
        super().__init__(args, layer_id)
        dim = args.hidden_size
        eps = args.rms_norm_eps
        self.enorm = nn.RMSNorm(dim, eps=eps)
        self.hnorm = nn.RMSNorm(dim, eps=eps)
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.hc_head = HeadHC(dim, args.hc_mult, args.hc_eps)

    def __call__(
        self,
        h: mx.array,
        input_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        cache=None,
        return_hidden: bool = False,
    ) -> mx.array:
        """``h``: ``[b, s, hc, dim]`` -> draft logits ``[b, s, vocab]``.

        ``embed_tokens``/``lm_head`` are the trunk's, per the sharing above.  The
        reference's ``ParallelHead.get_logits`` slices ``x[:, -1]`` before the
        matmul because its caller only ever wants the last row; the full sequence
        is returned here (mlx-lm's convention) and that slice is the caller's.

        ``return_hidden`` additionally returns the block's own pre-head
        Hyper-Connection state ``[b, s, hc, dim]``.  That is the tensor a
        multi-step draft chain feeds back in as ``h``: it occupies exactly the
        position the trunk's :meth:`DeepseekV4Model.hc_hidden` output does, which
        is what makes step ``i+1`` of the chain the same computation step ``i``
        ran.  Depth > 1 is an MTPLX extension either way — the reference ships one
        block and defines only the depth-1 call — and it is the same extension the
        sibling appended-layer backends make (GLM's modulo-into-layers, Hy3's
        single NextN layer reused at every depth).
        """
        e = self.enorm(embed_tokens(input_ids))          # [b, s, dim]
        x = self.hnorm(h)                                # [b, s, hc, dim]
        x = self.e_proj(e)[:, :, None, :] + self.h_proj(x)
        x = super().__call__(x, mask=None, cache=cache, input_ids=input_ids)
        logits = lm_head(self.norm(self.hc_head(x)))
        return (logits, x) if return_hidden else logits


class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hc_head = HeadHC(args.hidden_size, args.hc_mult, args.hc_eps)
        self._hc_hidden_impl = self._stock_hc_hidden
        self._collapse_impl = self._stock_collapse

    def _stock_hc_hidden(self, input_ids: mx.array, cache=None) -> mx.array:
        """Run the body and stop at the Hyper-Connection state ``[b, s, hc, dim]``.

        This is the split point the MTP block needs: the reference keeps ``h`` in
        hc form all the way out of the body and hands *that* tensor to both the
        output head and ``MTPBlock.forward`` (``Transformer.forward`` L806-808 vs
        model.py L757-763).  Collapsing to ``[b, s, dim]`` first — which is what
        :meth:`__call__` returns — would destroy the copies the draft block's own
        ``hnorm``/``h_proj`` read.
        """
        h = self.embed_tokens(input_ids)  # [b, s, dim]
        # expand to hc_mult residual copies
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc_mult, h.shape[-1]))
        # The attention builds its own window + compressed-KV causal mask internally
        # (it needs the compressed-position columns), so no mask is threaded here.
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=None, cache=c, input_ids=input_ids)
        return h

    def _stock_collapse(self, h: mx.array) -> mx.array:
        """Head-side collapse of the hc copies + final norm (``ParallelHead.forward``
        L718-721, minus the ``lm_head`` matmul the caller owns)."""
        return self.norm(self.hc_head(h))

    def _mia_hc_body(self, input_ids: mx.array, cache, tap_ids=()):
        if cache is None:
            cache = [None] * len(self.layers)
        return self._run_mia_hc_body(input_ids, cache, tap_ids)

    def _run_mia_hc_body(self, input_ids: mx.array, cache, tap_ids=()):
        """Execute the installed carried trunk against an owned cache list."""
        lead = tuple(int(value) for value in input_ids.shape)
        first = self.layers[0]
        residual, post, comb, value = self._mia_mhc.pre_broadcast(
            self.embed_tokens(input_ids),
            first.attn_hc,
            first.attn_norm,
        )
        taps = []
        final_reconstructed = None
        for layer_id, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            if layer_id:
                residual, post, comb, value = self._mia_mhc.post_pre_attn(
                    value,
                    residual,
                    post,
                    comb,
                    layer.attn_hc,
                    layer.attn_norm,
                )
            value = layer.attn(
                value.reshape(*lead, self.args.hidden_size),
                mask=None,
                cache=layer_cache,
            )
            residual, post, comb, value = self._mia_mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                layer.ffn_hc,
                layer.ffn_norm,
            )
            value = layer.ffn(
                value.reshape(*lead, self.args.hidden_size),
                input_ids=input_ids,
            )
            if layer_id in tap_ids:
                reconstructed = self._mia_mhc.post(value, residual, post, comb)
                taps.append(
                    mx.mean(
                        reconstructed.reshape(*lead, self.args.hc_mult, self.args.hidden_size),
                        axis=-2,
                    )
                )
                if layer_id == len(self.layers) - 1:
                    final_reconstructed = reconstructed
        hidden = (
            final_reconstructed
            if final_reconstructed is not None
            else self._mia_mhc.post(value, residual, post, comb)
        )
        hidden = hidden.reshape(*lead, self.args.hc_mult, self.args.hidden_size)
        return hidden, tuple(taps)

    def _run_mia_hc_target_tail_taps(self, input_ids: mx.array, cache):
        """Execute the fixed 43-layer Mia target and capture layers 40-42.

        The artifact assigns layer zero the broadcast transition, layers 1-39
        ordinary carried transitions, and the final three layers DSpark tap
        ownership.  Encoding those roles in this installed entrypoint removes
        invariant layer-id and tap-membership decisions from target execution.
        """
        lead = tuple(int(value) for value in input_ids.shape)
        first = self.layers[0]
        residual, post, comb, value = self._mia_mhc.pre_broadcast(
            self.embed_tokens(input_ids),
            first.attn_hc,
            first.attn_norm,
        )
        value = first.attn(
            value.reshape(*lead, self.args.hidden_size),
            mask=None,
            cache=cache[0],
        )
        residual, post, comb, value = self._mia_mhc.post_pre_ffn(
            value,
            residual,
            post,
            comb,
            first.ffn_hc,
            first.ffn_norm,
        )
        value = first.ffn(
            value.reshape(*lead, self.args.hidden_size),
            input_ids=input_ids,
        )

        for layer, layer_cache in zip(self.layers[1:40], cache[1:40]):
            residual, post, comb, value = self._mia_mhc.post_pre_attn(
                value,
                residual,
                post,
                comb,
                layer.attn_hc,
                layer.attn_norm,
            )
            value = layer.attn(
                value.reshape(*lead, self.args.hidden_size),
                mask=None,
                cache=layer_cache,
            )
            residual, post, comb, value = self._mia_mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                layer.ffn_hc,
                layer.ffn_norm,
            )
            value = layer.ffn(
                value.reshape(*lead, self.args.hidden_size),
                input_ids=input_ids,
            )

        taps = []
        reconstructed = None
        for layer, layer_cache in zip(self.layers[40:43], cache[40:43]):
            residual, post, comb, value = self._mia_mhc.post_pre_attn(
                value,
                residual,
                post,
                comb,
                layer.attn_hc,
                layer.attn_norm,
            )
            value = layer.attn(
                value.reshape(*lead, self.args.hidden_size),
                mask=None,
                cache=layer_cache,
            )
            residual, post, comb, value = self._mia_mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                layer.ffn_hc,
                layer.ffn_norm,
            )
            value = layer.ffn(
                value.reshape(*lead, self.args.hidden_size),
                input_ids=input_ids,
            )
            reconstructed = self._mia_mhc.post(value, residual, post, comb)
            taps.append(
                mx.mean(
                    reconstructed.reshape(
                        *lead,
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ),
                    axis=-2,
                )
            )
        hidden = reconstructed.reshape(
            *lead,
            self.args.hc_mult,
            self.args.hidden_size,
        )
        return hidden, tuple(taps)

    def _mia_hc_hidden(self, input_ids: mx.array, cache=None) -> mx.array:
        hidden, _ = self._mia_hc_body(input_ids, cache)
        return hidden

    def mia_hc_hidden_with_taps(self, input_ids, cache, tap_ids):
        return self._mia_hc_body(input_ids, cache, tuple(tap_ids))

    def _mia_collapse(self, h: mx.array) -> mx.array:
        lead = tuple(int(value) for value in h.shape[:-2])
        collapsed = self._mia_mhc.head(h, self.hc_head).reshape(
            *lead, self.args.hidden_size
        )
        return self.norm(collapsed)

    def install_mia_mhc_runtime(self, *, max_tokens: int) -> None:
        from mtplx.kernels.deepseek_v4_mhc import MiaMHCPlan

        if (
            self.args.hidden_size != 4096
            or self.args.hc_mult != 4
            or not self.layers
        ):
            raise ValueError("Mia mHC requires a nonempty hidden=4096, hc=4 trunk")
        self._mia_mhc = MiaMHCPlan(
            max_tokens=max_tokens,
            rms_eps=self.args.rms_norm_eps,
            hc_eps=self.args.hc_eps,
            iters=self.args.hc_sinkhorn_iters,
        )
        self._mia_mhc.install_modules(
            hyper_connections=tuple(
                connection
                for layer in self.layers
                for connection in (layer.attn_hc, layer.ffn_hc)
            ),
            broadcast_connection=self.layers[0].attn_hc,
        )
        self._hc_hidden_impl = self._mia_hc_hidden
        self._collapse_impl = self._mia_collapse

    def install_mia_piecewise_target_runtime(self):
        from mtplx.deepseek_v4_mia_piecewise import (
            install_mia_physical_m6_piecewise_target,
        )

        return install_mia_physical_m6_piecewise_target(self)

    def hc_hidden(self, input_ids: mx.array, cache=None) -> mx.array:
        return self._hc_hidden_impl(input_ids, cache)

    def collapse(self, h: mx.array) -> mx.array:
        return self._collapse_impl(h)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        return self.collapse(self.hc_hidden(input_ids, cache))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._mia_engine_plan = None
        self._mia_base_rope_provider = None
        self._mia_compress_rope_provider = None
        self._mia_draft_rope_provider = None
        # Reference ``Transformer.mtp`` (model.py L789-793): a top-level list, so
        # the parameter paths are ``mtp.{i}.*`` — exactly the upstream checkpoint's
        # names.  Dropped again by :meth:`sanitize` if the weights are not there.
        # Construction-time performance installers may replace this with a
        # typed phase/width router.  The stock callable is explicit and direct;
        # decoder layers never probe candidate eligibility or fall back.
        self._target_hc_hidden_route = self.model.hc_hidden
        dspark_requested = any(
            value is not None
            for value in (
                args.dspark_block_size,
                args.dspark_markov_rank,
                args.dspark_noise_token_id,
                args.dspark_target_layer_ids,
            )
        )
        self.dspark = None
        if dspark_requested:
            from mtplx.models.deepseek_v4_dspark import build_deepseek_v4_dspark

            self.install_dspark_owner(build_deepseek_v4_dspark(args))
        else:
            self.mtp = [
                DeepseekV4MTP(args, args.num_hidden_layers + i)
                for i in range(max(int(args.num_nextn_predict_layers or 0), 0))
            ]
            self._target_forward_route = self._stock_target_forward
            self._target_cache_type = DeepseekV4Cache
            self._has_generic_mtp = bool(self.mtp)

    def install_dspark_owner(self, owner) -> None:
        """Bind one qualified DSpark owner to the target exactly once.

        The split Mia artifact cannot construct the draft from target arguments:
        its routed expert bank has 64 owners while the target has 216.  Loading
        therefore constructs and validates the draft separately, then installs
        this fixed route once before any measured execution.  The ordinary
        same-checkpoint owner remains the constructor's existing route.
        """

        from mtplx.models.deepseek_v4_dspark import (
            DSPARK_BLOCK_SIZE,
            DSPARK_STAGE_COUNT,
            DSPARK_TARGET_LAYER_IDS,
            DSparkTargetRoute,
        )

        if self.args.hybrid_tr3_tail is not None:
            observed = (
                len(owner.stages),
                int(owner.args.n_routed_experts),
                int(owner.block_size),
                tuple(int(value) for value in owner.target_layer_ids),
            )
            expected = (
                DSPARK_STAGE_COUNT,
                64,
                DSPARK_BLOCK_SIZE,
                tuple(DSPARK_TARGET_LAYER_IDS),
            )
            if observed != expected:
                raise ValueError(
                    "unsupported Mia DSpark owner contract: "
                    f"observed={observed!r}, expected={expected!r}"
                )
            self.args.dspark_block_size = int(owner.args.dspark_block_size)
            self.args.dspark_markov_rank = int(owner.args.dspark_markov_rank)
            self.args.dspark_noise_token_id = int(owner.args.dspark_noise_token_id)
            self.args.dspark_target_layer_ids = list(
                owner.args.dspark_target_layer_ids
            )
            target_layer_ids = owner.target_layer_ids
        else:
            if len(owner.stages) != DSPARK_STAGE_COUNT:
                raise ValueError(
                    "DeepSeek V4 DSpark owner must expose exactly three stages"
                )
            target_layer_ids = self.args.dspark_target_layer_ids
        self.dspark = owner
        self.mtp = owner.stages
        target_route = DSparkTargetRoute(target_layer_ids)
        self._target_forward_route = lambda inputs, cache: target_route(
            self, inputs, cache
        )
        for layer in self.model.layers:
            layer.attn.install_mia_nvfp4_attention()
        self._target_cache_type = DeepseekV4NVFP4Cache
        self._has_generic_mtp = False

    def install_mia_mhc_runtime(self, *, max_tokens: int) -> None:
        """Install the carried target and K64 draft mHC state machines."""
        if self.dspark is None:
            raise ValueError("Mia carried mHC installation requires DSpark")
        self.model.install_mia_mhc_runtime(max_tokens=max_tokens)
        self.dspark.install_mia_mhc_runtime(max_tokens=max_tokens)
        if (
            len(self.model.layers) != 43
            or tuple(int(value) for value in self.dspark.target_layer_ids)
            != (40, 41, 42)
        ):
            raise ValueError(
                "Mia fixed target execution requires 43 layers and tail taps 40-42"
            )
        self._target_forward_route = self._mia_target_forward

    def install_mia_piecewise_target_runtime(self):
        from mtplx.deepseek_v4_mia_piecewise import MiaDFlashTargetPhaseRoute

        route = self.model.install_mia_piecewise_target_runtime()
        self._mia_dflash_target_route = MiaDFlashTargetPhaseRoute(
            prefill=self._mia_target_forward,
            decode_verify=self._mia_piecewise_target_forward,
        )
        return route

    def _mia_target_forward(self, inputs: mx.array, cache):
        self._mia_base_rope_provider.begin_forward()
        self._mia_compress_rope_provider.begin_forward()
        self._mia_draft_rope_provider.begin_forward()
        return self.model._run_mia_hc_target_tail_taps(inputs, cache)

    def _mia_piecewise_target_forward(self, inputs: mx.array, cache):
        self._mia_base_rope_provider.begin_forward()
        self._mia_compress_rope_provider.begin_forward()
        self._mia_draft_rope_provider.begin_forward()
        return self.model._mia_piecewise_target_route(inputs, cache)

    def mia_dflash_forward(
        self,
        inputs: mx.array,
        cache,
        *,
        logits_last_only: bool,
    ):
        """Direct target surface for the sealed Mia DFlash execution lane."""
        hidden, taps = self._mia_dflash_target_route(inputs, cache)
        logits_source = hidden[:, -1:] if logits_last_only else hidden
        return self.logits_from_hc_hidden(logits_source), taps

    def install_mia_engine_plan(self, plan) -> None:
        """Seal the exact artifact, cache capacity, and installed callables."""

        if self._mia_engine_plan is not None:
            raise ValueError("the Mia engine plan is already installed")
        self._mia_engine_plan = plan

    def _stock_target_forward(self, inputs: mx.array, cache):
        hidden = self._target_hc_hidden_route(inputs, cache)
        return hidden, hidden

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
        input_embeddings=None,
        hidden_variant: Optional[str] = None,
        emit_logits: bool = True,
        logits_keep: Optional[int] = None,
        **kwargs,
    ):
        """Target forward; also the MTPLX runtime's ``forward_ar`` surface.

        Plain ``model(ids)`` / ``model(ids, cache=cache)`` is unchanged.  The extra
        keywords are the contract ``mtplx.runtime.MTPLXRuntime.forward_ar`` drives
        every MTP backend through:

        * ``return_hidden`` — also return the state the draft block consumes.  For
          this architecture that is the pre-head Hyper-Connection tensor
          ``[b, s, hc, dim]`` (:meth:`hc_hidden`), NOT a ``[b, s, dim]`` hidden:
          collapsing first would destroy the copies ``DeepseekV4MTP.hnorm`` /
          ``h_proj`` read.  Engine code only ever slices axis 1 of this tensor and
          hands it back to :meth:`mtp_forward`, so the extra axis is transparent.
        * ``hidden_variant`` — accepted and ignored.  The variant knob picks
          between a Qwen-style draft's pre-norm/post-norm/fc taps; V4's draft input
          is defined by the reference as exactly one tensor, so there is nothing to
          pick.  Raising instead would break every draft call, since
          ``runtime.draft_mtp`` always resolves the contract default.  Same
          decision as the sibling appended-layer backends (glm_mtp, step3p5, hy3).
        * ``emit_logits`` / ``logits_keep`` — skip, or restrict to the last ``k``
          rows, the ``lm_head`` matmul.  Over a 129280-row vocabulary that matmul
          dominates a prefill chunk, and prefill only needs the final row.
        """
        if input_embeddings is not None:
            raise ValueError(
                "the DeepSeek-V4 backend does not support input_embeddings "
                "(no vision splice path)"
            )
        h, exposed_hidden = self._target_forward_route(inputs, cache)
        logits = None
        if emit_logits:
            source = h
            if logits_keep is not None:
                source = h[:, -max(1, int(logits_keep)):]
            logits = self.logits_from_hc_hidden(source)
        if not return_hidden:
            return logits
        return logits, exposed_hidden

    @property
    def layers(self):
        return self.model.layers

    # -- MTP (speculative draft head) --------------------------------------
    @property
    def mtp_blocks(self) -> list:
        """The draft blocks, however ``mtp`` is currently bound.

        :meth:`__init__` binds ``self.mtp`` to a plain list so the parameter paths
        are the checkpoint's ``mtp.{i}.*``.  ``inject_deepseek_v4_mtp_support``
        rebinds it (post-load) to a container that also answers ``.layers``, which
        is what ``mtplx.mtp_patch.validate_mtp_support`` probes for.  Everything
        else goes through this property so neither binding is load-bearing.
        """
        blocks = getattr(self, "mtp", None)
        if blocks is None:
            return []
        return list(getattr(blocks, "layers", blocks))

    @property
    def has_mtp(self) -> bool:
        return self._has_generic_mtp

    def hc_hidden(self, inputs: mx.array, cache=None) -> mx.array:
        """Trunk forward stopping at the pre-head state the MTP block consumes."""
        return self.model.hc_hidden(inputs, cache)

    def logits_from_hc_hidden(self, h: mx.array) -> mx.array:
        """``[b, s, hc, dim]`` -> target logits; the other half of :meth:`hc_hidden`.

        ``logits_from_hc_hidden(hc_hidden(x)) == self(x)`` — a speculative step
        gets the target's logits and the draft's input from one trunk pass.
        """
        return self.lm_head(self.model.collapse(h))

    def mtp_forward(
        self,
        h: mx.array,
        input_ids: mx.array,
        index: int = 0,
        cache=None,
        *,
        mtp_cache=None,
        concat_order: Optional[str] = None,
        return_hidden: bool = False,
        mtp_hidden_variant: Optional[str] = None,
        position_offset: Optional[int] = None,
        mtp_depth: Optional[int] = None,
    ):
        """Draft logits from the trunk's ``h`` and the next tokens' ids.

        Supplies the two modules the reference assigns onto the block (the trunk
        embedding and lm_head) instead of duplicating them; see
        :class:`DeepseekV4MTP` for the ``input_ids`` alignment contract.

        Two ways to hand it a cache, because it answers to two callers:

        * ``cache`` — the **one** :class:`DeepseekV4Cache` belonging to block
          ``index`` (``make_mtp_cache()[index]``, not the list).  The trunk takes a
          list because it has one entry per layer; a draft block is a single layer
          and takes its own.
        * ``mtp_cache`` — the whole list, which is what
          ``MTPLXRuntime.draft_mtp`` passes; ``index`` selects from it.

        The remaining keywords are the runtime's uniform draft signature.
        ``concat_order`` and ``mtp_hidden_variant`` are Qwen-shaped knobs with no
        V4 counterpart (see :meth:`__call__`) and are accepted and ignored;
        ``mtp_depth`` is informational, as it is for every single-block draft head
        (the one block is reused at every depth); ``position_offset`` is rejected
        rather than ignored, because silently dropping it would put the draft's
        RoPE at the wrong absolute position instead of failing.
        """
        blocks = self.mtp_blocks
        if not blocks:
            raise RuntimeError("this checkpoint ships no MTP block")
        if isinstance(cache, (list, tuple)):
            raise TypeError(
                "mtp_forward takes the MTP block's own cache, not the list: "
                f"pass make_mtp_cache()[{index}]"
            )
        if position_offset is not None:
            raise ValueError(
                "the DeepSeek-V4 draft block takes its RoPE offset from its own "
                "cache; explicit position_offset is not supported"
            )
        if mtp_cache is not None:
            if not isinstance(mtp_cache, (list, tuple)):
                raise TypeError("mtp_cache must be the make_mtp_cache() list")
            if cache is not None:
                raise TypeError("pass either cache= or mtp_cache=, not both")
            cache = mtp_cache[index] if mtp_cache else None
        return blocks[index](
            h,
            input_ids,
            self.model.embed_tokens,
            self.lm_head,
            cache=cache,
            return_hidden=return_hidden,
        )

    def mtp_update_cache(
        self,
        h: mx.array,
        input_ids: mx.array,
        index: int = 0,
        *,
        mtp_cache=None,
        concat_order: Optional[str] = None,
        mtp_hidden_variant: Optional[str] = None,
        position_offset: Optional[int] = None,
        mtp_depth: Optional[int] = None,
        input_embeddings=None,
    ) -> mx.array:
        """Append committed history to the draft cache; returns the draft hidden.

        ``MTPLXRuntime.update_mtp_cache`` drives this to keep the draft block's KV
        in step with the tokens the target committed.  The ``lm_head`` matmul still
        runs — the draft head shares the trunk's 129280-row projection and this
        call is off the hot path (history append, not per-step drafting).
        """
        if input_embeddings is not None:
            raise ValueError(
                "the DeepSeek-V4 backend does not support input_embeddings "
                "(no vision splice path)"
            )
        _logits, hidden = self.mtp_forward(
            h,
            input_ids,
            index,
            mtp_cache=mtp_cache,
            concat_order=concat_order,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            position_offset=position_offset,
            mtp_depth=mtp_depth,
        )
        return hidden

    def make_mtp_cache(self):
        """One :class:`DeepseekV4Cache` per MTP block.

        Separate from :meth:`make_cache`: the draft block's attention is its own
        module with its own KV (reference ``Attention.__init__`` L474 registers a
        per-instance ``kv_cache``), so it must not share the trunk's.  Its
        ``compress_ratio`` is 0, which makes the cache a plain sliding window —
        no compressed rows, no compressor frontier, so ``trim`` there rewinds only
        the window and ``offset``.
        """
        return [
            DeepseekV4Cache(
                window_size=block.attn.window_size,
                compress_ratio=block.attn.compress_ratio,
                head_dim=block.attn.head_dim,
            )
            for block in self.mtp_blocks
        ]

    def make_dspark_cache(self):
        return self.dspark.make_cache()

    def prefill_dspark(self, target_taps, caches) -> None:
        self.dspark.prefill(target_taps, caches)

    def propose_dspark_k5(self, primary_token_ids, caches, *, start_pos: int):
        return self.dspark.propose_k5(
            primary_token_ids,
            self.model.embed_tokens,
            self.lm_head,
            caches,
            start_pos=start_pos,
        )

    def commit_dspark_main(self, target_taps, caches, *, start_pos: int) -> None:
        self.dspark.commit_main(
            target_taps,
            caches,
            start_pos=start_pos,
        )

    def sanitize(self, weights: dict) -> dict:
        """Adapt this module tree to the checkpoint's tensors.

        Confirmed no-ops (the checkpoint already matches the tree):
          * ``ffn.switch_mlp.*`` ships pre-stacked (already ``[n_experts, ...]``) with
            mxfp4 scales and no biases — feed straight into ``SwitchGLU``'s quantised
            path (mode override supplied via config["quantization"]).
          * ``ffn.gate.tid2eid`` (hash layers) loads as int32.

        ``attn.wo_a`` is logically one ``[g*r, per]`` matrix, but the DSpark
        conversion preserves its leading output groups on disk as
        ``[g, r, packed_per]`` (and likewise for scales/biases).  MLX's
        ``QuantizedLinear`` owns the equivalent flattened output dimension, so
        collapse those two leading axes once at load.  This changes only the
        storage view; :meth:`_wo_a_grouped` restores ``[g, r, per]`` before the
        exact grouped einsum.

        The other real adaptation is the MTP block.  ``num_nextn_predict_layers`` is
        not trustworthy on its own: the published MLX conversions declare 1 while
        shipping no ``mtp.*`` tensor at all (which is what
        ``mtplx.artifacts.mtp_weights_present_on_disk`` and the runtime's
        degrade-to-autoregressive branch exist for).  So the *weights* decide —
        a checkpoint that ships the draft head keeps it and binds through the
        ordinary load path, and one that does not drops it from the tree here so
        ``load_weights(strict=True)`` still sees an exact match instead of 58
        spurious "missing" keys.
        """
        args = getattr(self, "args", None)
        if args is not None and args.hybrid_tr3_tail is not None:
            from mtplx.deepseek_v4_exl3 import sanitize_mia_exl3_target_weights

            weights = sanitize_mia_exl3_target_weights(
                weights,
                layers=int(args.num_hidden_layers),
                experts=int(args.n_routed_experts),
            )

        for name, value in tuple(weights.items()):
            if (
                ".attn.wo_a." in str(name)
                and getattr(value, "ndim", 0) == 3
            ):
                weights[name] = value.reshape(
                    int(value.shape[0]) * int(value.shape[1]),
                    int(value.shape[2]),
                )

        mtp_weights_present = any(str(k).startswith("mtp.") for k in weights)
        if self.dspark is not None and not mtp_weights_present:
            raise ValueError("the installed DSpark owner requires mtp.0/1/2 weights")
        if self.dspark is None and self.mtp_blocks and not mtp_weights_present:
            self.mtp = []
            self._has_generic_mtp = False
        return weights

    def make_cache(self, *, capacity_tokens: Optional[int] = None):
        """One :class:`DeepseekV4Cache` per layer (sliding-window KV + compressed KV
        + compressor frontier).  Shapes come off the built attention modules so the
        cache cannot drift from the layer's own compress ratio."""
        if self._mia_engine_plan is not None:
            installed_capacity = int(
                self._mia_engine_plan.context_capacity_tokens
            )
            if capacity_tokens is not None and int(capacity_tokens) > installed_capacity:
                raise ValueError(
                    "the request exceeds the exact Mia cache capacity installed at "
                    "construction: "
                    f"requested={int(capacity_tokens)}, installed={installed_capacity}"
                )
            return self._mia_engine_plan.make_target_cache(self.layers)
        return [
            self._target_cache_type(
                window_size=layer.attn.window_size,
                compress_ratio=layer.attn.compress_ratio,
                head_dim=layer.attn.head_dim,
                **(
                    {"capacity_tokens": capacity_tokens}
                    if self._target_cache_type is DeepseekV4NVFP4Cache
                    else {}
                ),
            )
            for layer in self.layers
        ]


# ---------------------------------------------------------------------------
# MTPLX runtime binding (speculative lane)
# ---------------------------------------------------------------------------
class MTPHead(nn.Module):
    """Post-load container so ``model.mtp`` answers ``.layers``.

    Every other MTP backend is an mlx-lm model that MTPLX *grafts* a draft head
    onto, and ``mtplx.mtp_patch.validate_mtp_support`` probes that graft with
    ``model.mtp.layers``.  This backend owns its draft head natively and binds it
    from the checkpoint's own ``mtp.{i}.*`` paths, which means ``Model.mtp`` has to
    be a plain list at load time — a container would rename every tensor.  So the
    list is wrapped here *after* the weights are bound, holding the very same block
    objects (no copy, no re-load), and :attr:`Model.mtp_blocks` reads through either
    binding.  Same move ``hy_v3_mtp_patch`` makes when it aliases
    ``model.mtp.layers = [model.mtp.layer]``.
    """

    def __init__(self, blocks):
        super().__init__()
        self.layers = list(blocks)


_O_LORA_BODY_COUNT = 43
_O_LORA_MTP_COUNT = 1
_O_LORA_WO_A_LOGICAL_SHAPE = (8192, 4096)
_O_LORA_WO_A_PACKED_SHAPE = (8192, 512)
_O_LORA_WO_A_QUANT_AUX_SHAPE = (8192, 64)
_O_LORA_BODY_WO_B_LOGICAL_SHAPE = (4096, 8192)
_O_LORA_BODY_WO_B_PACKED_SHAPE = (4096, 1024)
_O_LORA_BODY_WO_B_QUANT_AUX_SHAPE = (4096, 128)
_O_LORA_MTP_WO_B_SHAPE = (4096, 8192)
_O_LORA_ATTENTION_GEOMETRY = {
    "n_groups": 8,
    "o_lora_rank": 1024,
    "n_heads": 64,
    "head_dim": 512,
    "dim": 4096,
    "input_width": 32768,
    "per_group_input": 4096,
    "grouped_output_width": 8192,
}
_O_LORA_QUANT_FIELDS = ("scales", "biases", "bits", "group_size", "mode")
_CANONICAL_O_LORA_STORAGE_CONTRACT = {
    "body": {
        "count": 43,
        "attention_geometry": _O_LORA_ATTENTION_GEOMETRY,
        "wo_a": {
            "class": "QuantizedLinear",
            "logical_weight_shape": list(_O_LORA_WO_A_LOGICAL_SHAPE),
            "packed_weight": {
                "shape": list(_O_LORA_WO_A_PACKED_SHAPE),
                "dtype": "uint32",
            },
            "scales": {
                "shape": list(_O_LORA_WO_A_QUANT_AUX_SHAPE),
                "dtype": "bfloat16",
            },
            "biases": {
                "shape": list(_O_LORA_WO_A_QUANT_AUX_SHAPE),
                "dtype": "bfloat16",
            },
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "additive_bias": None,
        },
        "wo_b": {
            "class": "QuantizedLinear",
            "logical_weight_shape": list(_O_LORA_BODY_WO_B_LOGICAL_SHAPE),
            "packed_weight": {
                "shape": list(_O_LORA_BODY_WO_B_PACKED_SHAPE),
                "dtype": "uint32",
            },
            "scales": {
                "shape": list(_O_LORA_BODY_WO_B_QUANT_AUX_SHAPE),
                "dtype": "bfloat16",
            },
            "biases": {
                "shape": list(_O_LORA_BODY_WO_B_QUANT_AUX_SHAPE),
                "dtype": "bfloat16",
            },
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "additive_bias": None,
        },
    },
    "mtp": {
        "count": 1,
        "wo_a": {
            "class": "Linear",
            "weight": {
                "shape": list(_O_LORA_WO_A_LOGICAL_SHAPE),
                "dtype": "bfloat16",
            },
            "additive_bias": None,
            "no_quant_metadata": True,
            "absent_quant_fields": list(_O_LORA_QUANT_FIELDS),
        },
        "wo_b": {
            "class": "Linear",
            "weight": {
                "shape": list(_O_LORA_MTP_WO_B_SHAPE),
                "dtype": "bfloat16",
            },
            "additive_bias": None,
            "no_quant_metadata": True,
            "absent_quant_fields": list(_O_LORA_QUANT_FIELDS),
        },
    },
}


def _require_o_lora_array(
    value, *, label: str, shape: tuple[int, int], dtype
) -> None:
    if tuple(getattr(value, "shape", ())) != shape:
        raise ValueError(
            f"{label} shape {tuple(getattr(value, 'shape', ()))} does not match {shape}"
        )
    if getattr(value, "dtype", None) != dtype:
        raise ValueError(f"{label} dtype is not {dtype}")


def _require_canonical_quantized_linear(
    linear, *, label: str, logical_shape: tuple[int, int]
) -> tuple:
    """Validate exact Q4 storage and derive its logical shape two ways."""

    if not isinstance(linear, nn.QuantizedLinear):
        raise ValueError(f"{label} must be QuantizedLinear")
    for attribute, expected in (
        ("bits", 4),
        ("group_size", 64),
        ("mode", "affine"),
    ):
        observed = getattr(linear, attribute, None)
        if observed != expected:
            raise ValueError(
                f"{label} {attribute}={observed!r}, expected {expected!r}"
            )

    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    weight_shape = tuple(getattr(weight, "shape", ()))
    scales_shape = tuple(getattr(scales, "shape", ()))
    biases_shape = tuple(getattr(biases, "shape", ()))
    if len(weight_shape) != 2:
        raise ValueError(f"{label} packed weight shape {weight_shape} is not rank 2")
    if len(scales_shape) != 2:
        raise ValueError(f"{label} scales shape {scales_shape} is not rank 2")

    packed_divisor = 32 // int(linear.bits)
    packed_logical = (weight_shape[0], weight_shape[1] * packed_divisor)
    scales_logical = (
        scales_shape[0],
        scales_shape[1] * int(linear.group_size),
    )
    expected_output, expected_input = logical_shape
    if packed_logical[0] != expected_output:
        raise ValueError(
            f"{label} packed weight shape {weight_shape} has logical output "
            f"{packed_logical[0]}, expected {expected_output}"
        )
    if packed_logical[1] != expected_input:
        raise ValueError(
            f"{label} packed weight shape {weight_shape} has logical input "
            f"{packed_logical[1]}, expected {expected_input}"
        )
    if scales_logical[0] != expected_output:
        raise ValueError(
            f"{label} scales shape {scales_shape} has logical output "
            f"{scales_logical[0]}, expected {expected_output}"
        )
    if scales_logical[1] != expected_input:
        raise ValueError(
            f"{label} scales shape {scales_shape} has logical input "
            f"{scales_logical[1]}, expected {expected_input}"
        )
    if getattr(weight, "dtype", None) != mx.uint32:
        raise ValueError(f"{label} packed weight dtype is not {mx.uint32}")
    if getattr(scales, "dtype", None) != mx.bfloat16:
        raise ValueError(f"{label} scales dtype is not {mx.bfloat16}")
    if biases_shape != scales_shape:
        raise ValueError(
            f"{label} biases shape {biases_shape} does not match scales shape "
            f"{scales_shape}"
        )
    if getattr(biases, "dtype", None) != mx.bfloat16:
        raise ValueError(f"{label} biases dtype is not {mx.bfloat16}")
    if getattr(linear, "bias", None) is not None:
        raise ValueError(f"{label} additive bias must be absent")
    return (
        weight,
        scales,
        biases,
        linear.group_size,
        linear.bits,
        linear.mode,
    )


def _require_canonical_dense_linear(
    linear, *, label: str, shape: tuple[int, int]
):
    if not isinstance(linear, nn.Linear) or isinstance(linear, nn.QuantizedLinear):
        raise ValueError(f"{label} must be a dense nn.Linear, not QuantizedLinear")
    weight = getattr(linear, "weight", None)
    _require_o_lora_array(
        weight,
        label=f"{label} weight",
        shape=shape,
        dtype=mx.bfloat16,
    )
    if getattr(linear, "bias", None) is not None:
        raise ValueError(f"{label} additive bias must be absent")
    accidental = [
        attribute
        for attribute in _O_LORA_QUANT_FIELDS
        if hasattr(linear, attribute)
    ]
    if accidental:
        raise ValueError(
            f"{label} must not expose quantized metadata: " + ", ".join(accidental)
        )
    return weight


def _validate_canonical_o_lora_topology(trunk, mtp) -> tuple[list[tuple], mx.array]:
    """Fail before timing unless this exact mixed checkpoint layout is loaded.

    The 43 body modules are affine Q4 storage and may take the direct gather
    route.  The one MTP module is intentionally a dense BF16 linear and must
    remain an explicit stock route; treating it as an eligible gather module
    would turn a checkpoint-layout fact into a hot-path fallback.
    """
    if len(trunk) != _O_LORA_BODY_COUNT:
        raise ValueError(f"expected 43 body o-LoRA modules, found {len(trunk)}")
    if len(mtp) != _O_LORA_MTP_COUNT:
        raise ValueError(f"expected exactly one MTP o-LoRA module, found {len(mtp)}")

    body_quant = []
    for index, attention in enumerate(trunk):
        wo_a = getattr(attention, "wo_a", None)
        wo_a_label = f"body {index} wo_a"
        for attribute in ("n_groups", "o_lora_rank", "n_heads", "head_dim", "dim"):
            observed = getattr(attention, attribute, None)
            expected = _O_LORA_ATTENTION_GEOMETRY[attribute]
            if observed != expected:
                raise ValueError(
                    f"body {index} attention {attribute}={observed!r}, "
                    f"expected {expected}"
                )
        input_width = int(attention.n_heads) * int(attention.head_dim)
        per_group_input = input_width // int(attention.n_groups)
        grouped_output_width = int(attention.n_groups) * int(attention.o_lora_rank)
        derived_geometry = {
            "input_width": input_width,
            "per_group_input": per_group_input,
            "grouped_output_width": grouped_output_width,
        }
        for attribute, observed in derived_geometry.items():
            expected = _O_LORA_ATTENTION_GEOMETRY[attribute]
            if observed != expected:
                raise ValueError(
                    f"body {index} attention {attribute}={observed}, expected {expected}"
                )
        body_quant.append(
            _require_canonical_quantized_linear(
                wo_a,
                label=wo_a_label,
                logical_shape=_O_LORA_WO_A_LOGICAL_SHAPE,
            )
        )
        _require_canonical_quantized_linear(
            getattr(attention, "wo_b", None),
            label=f"body {index} wo_b",
            logical_shape=(int(attention.dim), grouped_output_width),
        )

    mtp_weight = _require_canonical_dense_linear(
        getattr(mtp[0], "wo_a", None),
        label="MTP wo_a",
        shape=_O_LORA_WO_A_LOGICAL_SHAPE,
    )
    _require_canonical_dense_linear(
        getattr(mtp[0], "wo_b", None),
        label="MTP wo_b",
        shape=_O_LORA_MTP_WO_B_SHAPE,
    )
    return body_quant, mtp_weight


def _require_bf16_body_activation_output(model):
    """Reify the actual trunk activation dtype before installing the M4 route."""
    trunk = getattr(model, "model", None)
    embedding = getattr(trunk, "embed_tokens", None)
    if embedding is None or not callable(embedding):
        raise ValueError(
            "M4-wide gather o-LoRA requires a callable trunk embedding output"
        )
    try:
        output = embedding(mx.zeros((1, 1), dtype=mx.int32))
    except Exception as exc:
        raise ValueError(
            "M4-wide gather o-LoRA could not reify the trunk embedding output"
        ) from exc
    expected_shape = (1, 1, _O_LORA_ATTENTION_GEOMETRY["dim"])
    if tuple(getattr(output, "shape", ())) != expected_shape:
        raise ValueError(
            "M4-wide gather o-LoRA embedding output shape "
            f"{tuple(getattr(output, 'shape', ()))} does not match {expected_shape}"
        )
    if getattr(output, "dtype", None) != mx.bfloat16:
        raise ValueError(
            "M4-wide gather o-LoRA embedding output dtype must be bfloat16"
        )
    return mx.bfloat16


def _validate_gather_qmm_wide_m4_body_routes(
    body_routes: list[_DirectGatherOLoraWideM4],
) -> None:
    """Prove exact M4 gathered algebra before binding any body route.

    The sentinel layers span the first, a hash-layer boundary, and the final
    body module.  Identity, reordered-distinct, and repeated RHS IDs prove the
    custom group-bank lookup has the same meaning as ``gather_qmm``; production
    uses the authenticated identity IDs held by each installed route.
    """

    if len(body_routes) != _O_LORA_BODY_COUNT:
        raise ValueError("M4-wide gather self-check lacks the 43 body routes")
    rhs_cases = (
        ("identity", (0, 1, 2, 3, 4, 5, 6, 7)),
        ("distinct_reordered", (7, 3, 5, 1, 6, 0, 4, 2)),
        ("repeated", (7, 0, 7, 3, 3, 5, 1, 0)),
    )
    for layer_index in (0, 3, 42):
        route = body_routes[layer_index].m4
        base = mx.arange(4 * 8 * 4096, dtype=mx.float32).reshape(4, 8, 4096)
        probe = ((base % 29.0) - 14.0).astype(mx.bfloat16) / 8.0
        gathered_x = probe.swapaxes(0, 1)
        for case_name, ids in rhs_cases:
            rhs_ids = mx.array(ids, dtype=mx.uint32)
            stock = mx.gather_qmm(
                gathered_x,
                route.weight,
                route.scales,
                route.biases,
                lhs_indices=route.group_ids,
                rhs_indices=rhs_ids,
                transpose=True,
                group_size=route.group_size,
                bits=route.bits,
                mode=route.mode,
            )
            wide = route.grouped(probe, rhs_ids)
            mx.eval(stock, wide)
            exact = bool(mx.array_equal(stock, wide).item())
            if (
                tuple(stock.shape) != (8, 4, 1024)
                or tuple(wide.shape) != (8, 4, 1024)
                or stock.dtype != mx.bfloat16
                or wide.dtype != mx.bfloat16
                or stock.dtype != wide.dtype
                or not exact
            ):
                raise ValueError(
                    "M4-wide gather self-check diverged at body "
                    f"layer {layer_index} ({case_name})"
                )


def install_deepseek_v4_o_lora_routes(
    model, mode: str | None = None, *, canonical_mixed_route: bool = False
) -> dict:
    """Install the canonical 43-Q4-body/one-dense-MTP o-LoRA route.

    Validation and binding are construction-time only.  The direct candidate
    binds gather on body modules and binds MTP to its stock dense callable;
    neither branch has a runtime eligibility check or fallback.
    """
    trunk = [layer.attn for layer in model.layers]
    mtp = [block.attn for block in model.mtp_blocks]
    selected = (
        str(mode)
        if mode is not None
        else _o_lora_mode_from_env()
    )
    if selected not in _O_LORA_MODES:
        raise ValueError(f"unsupported o-LoRA route {selected!r}")
    if not canonical_mixed_route:
        reports = [attention.install_o_lora_route(selected) for attention in trunk + mtp]
        return {
            "mode": selected,
            "module_count": len(reports),
            "trunk_module_count": len(trunk),
            "mtp_module_count": len(mtp),
            "all_direct": bool(reports) and all(report["direct"] for report in reports),
            "all_mode_matches": bool(reports)
            and all(report["mode"] == selected for report in reports),
            "modules": reports,
        }
    if selected not in {"cached", "gather_qmm"}:
        raise ValueError(
            "canonical mixed o-LoRA route supports only cached or gather_qmm"
        )
    if selected == "gather_qmm" and _FP32_ACTIVATIONS:
        raise ValueError(
            "M4-wide gather o-LoRA requires DeepSeek-V4-Flash BF16 activation "
            "storage; MTPLX_DSV4_FP32_ACTIVATIONS is an explicit stock A/B arm"
        )
    body_quant, mtp_weight = _validate_canonical_o_lora_topology(trunk, mtp)
    activation_dtype = (
        _require_bf16_body_activation_output(model)
        if selected == "gather_qmm"
        else None
    )
    body_route_type = (
        _DirectGatherOLoraWideM4
        if selected == "gather_qmm"
        else _DirectCachedOLora
    )
    if selected == "gather_qmm":
        body_impls = [
            body_route_type(
                attention, quant, activation_dtype=activation_dtype
            )
            for attention, quant in zip(trunk, body_quant)
        ]
    else:
        body_impls = [
            body_route_type(attention, quant)
            for attention, quant in zip(trunk, body_quant)
        ]
    if selected == "gather_qmm":
        _validate_gather_qmm_wide_m4_body_routes(body_impls)
    mtp_impls = [_DirectDenseMTPOLora(mtp[0], mtp_weight)]
    for attention, installed in zip(trunk, body_impls):
        attention.o_lora_mode = selected
        attention._o_lora_impl = installed
    # Dense MTP is always explicitly installed stock.  In the candidate arm this
    # is deliberately not a fallback from gather_qmm.
    for attention, installed in zip(mtp, mtp_impls):
        attention.o_lora_mode = "cached"
        attention._o_lora_impl = installed
    body_reports = [
        {
            "mode": selected,
            "direct": selected == "gather_qmm",
            "callable": type(installed).__name__,
        }
        for installed in body_impls
    ]
    mtp_reports = [
        {
            "mode": "cached",
            "direct": False,
            "callable": type(installed).__name__,
        }
        for installed in mtp_impls
    ]
    reports = body_reports + mtp_reports
    route_objects = body_impls + mtp_impls
    callable_census = {
        "body_route_objects": len(body_impls),
        "body_route_kind": (
            "gather_qmm_m4_wide_direct"
            if selected == "gather_qmm"
            else "cached_direct"
        ),
        "body_callable_class": body_route_type.__name__,
        "mtp_route_objects": len(mtp_impls),
        "mtp_route_kind": "dense_bf16_stock_direct",
        "mtp_callable_class": _DirectDenseMTPOLora.__name__,
        "total_route_objects": len(route_objects),
        "unique_route_objects": len({id(installed) for installed in route_objects}),
        "mtp_distinct_type": bool(body_impls and mtp_impls)
        and type(mtp_impls[0]) is not type(body_impls[0]),
    }
    return {
        "mode": selected,
        "module_count": len(reports),
        "trunk_module_count": len(trunk),
        "mtp_module_count": len(mtp),
        "body_direct": sum(report["direct"] for report in body_reports),
        "mtp_stock": sum(
            report["mode"] == "cached" and not report["direct"]
            for report in mtp_reports
        ),
        "body_all_mode_matches": bool(body_reports)
        and all(report["mode"] == selected for report in body_reports),
        "route_plan_matches": bool(body_reports and mtp_reports)
        and all(report["mode"] == selected for report in body_reports)
        and all(
            report["mode"] == "cached" and not report["direct"]
            for report in mtp_reports
        ),
        "storage_contract": _CANONICAL_O_LORA_STORAGE_CONTRACT,
        "callable_census": callable_census,
        "modules": reports,
    }


def _mia_tp1_wo_attentions(model) -> tuple[tuple[object, ...], tuple[object, ...]]:
    target = tuple(layer.attn for layer in model.layers)
    draft = tuple(block.attn for block in model.mtp_blocks)
    if len(target) != 43 or len(draft) != 3:
        raise ValueError(
            "Mia TP1 WO requires 43 target and three draft attentions; "
            f"got {len(target)} and {len(draft)}"
        )
    attentions = target + draft
    if len({id(attention) for attention in attentions}) != 46:
        raise ValueError("Mia TP1 WO attention ownership is not one-to-one")
    return target, draft


def mia_tp1_wo_projection_receipt(model) -> dict:
    """Return the construction seal for the exact 46-owner TP1 WO route."""

    from mtplx.kernels.deepseek_v4_wo_mxfp8 import MiaTP1WOMXFP8Plan

    target, draft = _mia_tp1_wo_attentions(model)
    attentions = target + draft
    plans = tuple(attention._output_projection_impl for attention in attentions)
    expected_storage = (
        ((8192, 1024), mx.uint32),
        ((8192, 128), mx.uint8),
        ((4096, 2048), mx.uint32),
        ((4096, 256), mx.uint8),
    )
    parameter_ids = []
    for index, (attention, plan) in enumerate(
        zip(attentions, plans, strict=True)
    ):
        if type(plan) is not MiaTP1WOMXFP8Plan:
            raise ValueError(
                f"Mia TP1 WO attention {index} owns {type(plan).__name__}, "
                "not MiaTP1WOMXFP8Plan"
            )
        expected_role = "target" if index < len(target) else "draft"
        if plan.owner_role != expected_role:
            raise ValueError(
                f"Mia TP1 WO attention {index} owns role {plan.owner_role!r}, "
                f"not {expected_role!r}"
            )
        geometry = (
            int(attention.n_heads),
            int(attention.head_dim),
            int(attention.rope_head_dim),
            int(attention.n_groups),
            int(attention.o_lora_rank),
            int(attention.dim),
        )
        if geometry != (64, 512, 64, 8, 1024, 4096):
            raise ValueError(
                f"Mia TP1 WO attention {index} geometry changed: {geometry!r}"
            )
        wo_a = attention.wo_a
        wo_b = attention.wo_b
        module_contract = (
            int(getattr(wo_a, "group_size", 0)),
            int(getattr(wo_a, "bits", 0)),
            str(getattr(wo_a, "mode", "")),
            getattr(wo_a, "biases", None),
            int(getattr(wo_b, "group_size", 0)),
            int(getattr(wo_b, "bits", 0)),
            str(getattr(wo_b, "mode", "")),
            getattr(wo_b, "biases", None),
        )
        if module_contract != (32, 8, "mxfp8", None, 32, 8, "mxfp8", None):
            raise ValueError(
                f"Mia TP1 WO attention {index} is not native group-32 MXFP8"
            )
        tensors = (wo_a.weight, wo_a.scales, wo_b.weight, wo_b.scales)
        storage = tuple(
            (tuple(int(dim) for dim in tensor.shape), tensor.dtype)
            for tensor in tensors
        )
        if storage != expected_storage:
            raise ValueError(
                f"Mia TP1 WO attention {index} storage changed: {storage!r}"
            )
        owned = (
            plan.wo_a_weight,
            plan.wo_a_scales,
            plan.wo_b_weight,
            plan.wo_b_scales,
        )
        if any(
            plan_tensor is not module_tensor
            for plan_tensor, module_tensor in zip(owned, tensors, strict=True)
        ):
            raise ValueError(
                f"Mia TP1 WO attention {index} copied or replaced parameters"
            )
        parameter_ids.append(tuple(id(tensor) for tensor in tensors))
    plan_ids = tuple(id(plan) for plan in plans)
    if len(set(plan_ids)) != 46:
        raise ValueError("Mia TP1 WO plans are not one-to-one with attentions")
    max_prefill_rows = {int(plan.max_prefill_rows) for plan in plans}
    if len(max_prefill_rows) != 1:
        raise ValueError("Mia TP1 WO plans disagree on their prefill bound")
    return {
        "route": "mia_tp1_b12x_wo_mxfp8",
        "target_attention": len(target),
        "draft_attention": len(draft),
        "plan_count": len(plans),
        "unique_plan_count": len(set(plan_ids)),
        "plan_type": MiaTP1WOMXFP8Plan.__name__,
        "max_prefill_rows": max_prefill_rows.pop(),
        "storage_contract": tuple(
            (shape, str(dtype)) for shape, dtype in expected_storage
        ),
        "plan_ids": plan_ids,
        "parameter_ids": tuple(parameter_ids),
    }


def install_mia_tp1_wo_projection_routes(
    model,
    *,
    max_prefill_rows: int,
) -> dict:
    """Bind the exact source-derived TP1 WO owner after all weights load."""

    from mtplx.kernels.deepseek_v4_wo_mxfp8 import install_mia_tp1_wo_mxfp8

    if getattr(model, "_mia_wo_projection_receipt", None) is not None:
        raise ValueError("the Mia TP1 WO projection route is already installed")
    target, draft = _mia_tp1_wo_attentions(model)
    plans = tuple(
        install_mia_tp1_wo_mxfp8(
            wo_a_weight=attention.wo_a.weight,
            wo_a_scales=attention.wo_a.scales,
            wo_b_weight=attention.wo_b.weight,
            wo_b_scales=attention.wo_b.scales,
            owner_role="target",
            max_prefill_rows=max_prefill_rows,
        )
        for attention in target
    ) + tuple(
        install_mia_tp1_wo_mxfp8(
            wo_a_weight=attention.wo_a.weight,
            wo_a_scales=attention.wo_a.scales,
            wo_b_weight=attention.wo_b.weight,
            wo_b_scales=attention.wo_b.scales,
            owner_role="draft",
            max_prefill_rows=max_prefill_rows,
        )
        for attention in draft
    )
    attentions = target + draft
    for attention, plan in zip(attentions, plans, strict=True):
        attention._output_projection_impl = plan
    receipt = mia_tp1_wo_projection_receipt(model)
    model._mia_wo_projection_receipt = receipt
    return receipt


def is_deepseek_v4_mtp_config(config: dict) -> bool:
    """Does this artifact declare a DeepSeek-V4 draft head?

    Weight presence is decided later by :meth:`Model.sanitize` (the published
    mlx-community conversions declare the layer and ship no tensors, which is what
    the runtime's degrade-to-autoregressive branch exists for).
    """
    model_type = str((config or {}).get("model_type") or "").lower()
    architectures = [str(a) for a in (config or {}).get("architectures") or []]
    if model_type != "deepseek_v4" and not any(
        a.lower() == "deepseekv4forcausallm" for a in architectures
    ):
        return False
    if any(
        (config or {}).get(field) is not None
        for field in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        )
    ):
        return False
    return int((config or {}).get("num_nextn_predict_layers") or 0) > 0


def inject_deepseek_v4_mtp_support(
    model,
    path=None,
    config: Optional[dict] = None,
    contract=None,
) -> bool:
    """Enable the speculative lane on an already-loaded DeepSeek-V4 model.

    There is nothing to graft: :class:`DeepseekV4MTP` binds through the ordinary
    load path from the checkpoint's ``mtp.0.*`` tensors, and :class:`Model` already
    carries the runtime's draft surface (``__call__(return_hidden=...)``,
    :meth:`Model.mtp_forward`, :meth:`Model.mtp_update_cache`,
    :meth:`Model.make_mtp_cache`).  All this does is publish that fact in the shape
    ``mtplx.mtp_patch.validate_mtp_support`` checks, and report False — the
    degrade-to-autoregressive signal — for a checkpoint whose draft head
    :meth:`Model.sanitize` dropped.

    Returns True when the model can speculate.  The ``path``/``config``/``contract``
    parameters exist to match the sibling ``inject_*_mtp_support`` signature the
    runtime dispatches on; a bare :class:`~mtplx.mtp_patch.MTPContract` needs no
    adaptation here, because the V4 draft input is a single defined tensor with no
    hidden-variant or concat-order choice to make.
    """
    if not is_deepseek_v4_mtp_config(config or {}):
        return False
    blocks = getattr(model, "mtp_blocks", None)
    if not blocks:
        return False
    if getattr(getattr(model, "mtp", None), "layers", None) is None:
        model.mtp = MTPHead(blocks)
    return True
