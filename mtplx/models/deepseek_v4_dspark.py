"""Native fixed-K5 DSpark model components for DeepSeek V4 Flash."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn

from mtplx.deepseek_v4_nvfp4_kv import (
    FixedMiaNVFP4Ring,
    install_fixed_ring_commit_writer,
    install_fixed_ring_context_writer,
    install_stock432_record_packer,
)
from mtplx.kernels.deepseek_v4_nvfp4_mla import (
    install_dspark_k5_nvfp4_mla,
    install_dspark_k5_nvfp4_mla_graph,
)
from mtplx.models.deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    HeadHC,
    ModelArgs,
    _apply_interleaved_rope,
)


DSPARK_STAGE_COUNT = 3
DSPARK_BLOCK_SIZE = 5
DSPARK_NOISE_TOKEN_ID = 128799
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
DSPARK_MARKOV_RANK = 256


class _MarkovHead(Protocol):
    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]: ...


def greedy_future_tokens(
    neural_logits: mx.array,
    primary_token_ids: mx.array,
    markov_head: _MarkovHead,
) -> mx.array:
    """Produce five genuinely future tokens, conditioning row zero on primary."""

    if tuple(neural_logits.shape[:2]) != (
        int(primary_token_ids.shape[0]),
        DSPARK_BLOCK_SIZE,
    ):
        raise ValueError("DSpark K5 logits must have shape [batch, 5, vocab]")
    return _run_greedy_future_tokens_k5(
        neural_logits,
        primary_token_ids,
        markov_head,
    )


def _run_greedy_future_tokens_k5(
    neural_logits: mx.array,
    primary_token_ids: mx.array,
    markov_head: _MarkovHead,
) -> mx.array:
    """Direct sequential Markov decode for the installed physical K5 block."""
    previous = primary_token_ids
    future = []
    for row in range(DSPARK_BLOCK_SIZE):
        bias, _markov_embedding = markov_head(previous)
        previous = mx.argmax(neural_logits[:, row] + bias, axis=-1).astype(
            primary_token_ids.dtype
        )
        future.append(previous)
    return mx.stack(future, axis=1)


class DeepseekV4DSparkCache:
    """One stage's fixed sliding-window context ring in Mia stock432 storage."""

    def __init__(self, *, window_size: int, head_dim: int) -> None:
        self.window_size = int(window_size)
        self.head_dim = int(head_dim)
        if self.head_dim != 512:
            raise ValueError("Mia DSpark cache requires head_dim=512")
        self.ring = FixedMiaNVFP4Ring(capacity_rows=self.window_size)
        self._pack_records = install_stock432_record_packer(
            head_dim=self.head_dim,
            rope_dim=64,
        )
        if self.window_size == 128:
            self._write_initial_records = install_fixed_ring_context_writer(
                self.ring
            )
            self._write_commit_records = install_fixed_ring_commit_writer(
                self.ring
            )
        else:
            self._write_initial_records = None
            self._write_commit_records = None
        self.prefill_length = 0

    def prefill(self, main_latent: mx.array, main_rope: mx.array) -> None:
        batch, sequence_length, width = (int(v) for v in main_latent.shape)
        if width != self.head_dim:
            raise ValueError("DSpark cache head dimension mismatch")
        if tuple(main_rope.shape) != (batch, sequence_length, 64):
            raise ValueError("DSpark cache RoPE rows must have shape [batch, rows, 64]")
        self._prefill(main_latent, main_rope)

    def _prefill(self, main_latent: mx.array, main_rope: mx.array) -> None:
        """Install prompt rows after the DSpark cache contract is sealed."""
        sequence_length = int(main_latent.shape[1])
        if sequence_length > self.window_size:
            main_latent = main_latent[:, -self.window_size :]
            main_rope = main_rope[:, -self.window_size :]
        self._install_prefill_tail(
            main_latent,
            main_rope,
            total_length=sequence_length,
        )

    def _install_prefill_tail(
        self,
        main_latent: mx.array,
        main_rope: mx.array,
        *,
        total_length: int,
    ) -> None:
        """Own an at-most-window prompt tail at its absolute ring positions."""
        batch, tail_length, width = (int(v) for v in main_latent.shape)
        total_length = int(total_length)
        if total_length <= self.window_size:
            latent_padding = mx.zeros(
                (batch, self.window_size - tail_length, width),
                dtype=main_latent.dtype,
            )
            rope_padding = mx.zeros(
                (batch, self.window_size - tail_length, 64),
                dtype=main_rope.dtype,
            )
            latent_rows = mx.concatenate([main_latent, latent_padding], axis=1)
            rope_rows = mx.concatenate([main_rope, rope_padding], axis=1)
        else:
            cutoff = total_length % self.window_size
            latent_rows = (
                main_latent
                if cutoff == 0
                else mx.concatenate(
                    [
                        main_latent[:, self.window_size - cutoff :],
                        main_latent[:, : self.window_size - cutoff],
                    ],
                    axis=1,
                )
            )
            rope_rows = (
                main_rope
                if cutoff == 0
                else mx.concatenate(
                    [
                        main_rope[:, self.window_size - cutoff :],
                        main_rope[:, : self.window_size - cutoff],
                    ],
                    axis=1,
                )
            )
        records = self._pack_records(latent_rows, rope_rows)
        self.ring.clear()
        self.ring._append_installed_records(
            records,
            prefix=tuple(int(value) for value in records.shape[:-2]),
        )
        self.prefill_length = total_length

    def _install_prefill_records(
        self,
        records: mx.array,
        *,
        absolute_start: int,
        total_length: int,
    ) -> None:
        """Install one fused initial prompt tail into its fixed physical ring."""
        self._write_initial_records(records, absolute_start=int(absolute_start))
        self.prefill_length = int(total_length)

    def visible_rows(self) -> tuple[mx.array, mx.array]:
        if len(self.ring) != self.window_size:
            raise RuntimeError("DSpark attention cache has not been prefetched")
        return self.ring.decode()

    def commit_main(
        self,
        start_pos: int,
        main_latent: mx.array,
        main_rope: mx.array,
    ) -> None:
        if len(self.ring) != self.window_size:
            raise RuntimeError("DSpark decode requires attention-only prefill first")
        if (
            main_latent.ndim != 3
            or main_rope.ndim != 3
            or int(main_latent.shape[0]) != int(self.ring.shape[0])
            or tuple(main_latent.shape[:-1]) != tuple(main_rope.shape[:-1])
            or int(main_rope.shape[-1]) != 64
        ):
            raise ValueError("DSpark committed main K/V must match the ring batch")
        count = int(main_latent.shape[1])
        if count <= 0:
            raise ValueError("DSpark committed main K/V width is outside its ring")
        self._commit_main(start_pos, main_latent, main_rope)

    def _commit_main(
        self,
        start_pos: int,
        main_latent: mx.array,
        main_rope: mx.array,
    ) -> None:
        """Advance the installed fixed ring without repeated geometry proofs."""
        count = int(main_latent.shape[1])
        start = int(start_pos) % self.window_size
        first = min(count, self.window_size - start)
        records = self._pack_records(main_latent, main_rope)
        self.ring._replace_installed_records(start, records[:, :first])
        if first < count:
            self.ring._replace_installed_records(0, records[:, first:])
        self.prefill_length = max(self.prefill_length, int(start_pos) + count)

    def _commit_records(self, start_pos: int, records: mx.array) -> None:
        """Scatter one fused authoritative context increment into the ring."""
        count = int(records.shape[1])
        self._write_commit_records(records, absolute_start=int(start_pos))
        self.prefill_length = max(self.prefill_length, int(start_pos) + count)


class DSparkTargetRoute:
    """Construction-bound target route exposing ordered post-layer HC means."""

    def __init__(self, target_layer_ids=DSPARK_TARGET_LAYER_IDS) -> None:
        self.target_layer_ids = tuple(int(layer_id) for layer_id in target_layer_ids)

    def __call__(self, owner, inputs: mx.array, cache):
        hidden = owner.model.embed_tokens(inputs)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], owner.args.hc_mult, hidden.shape[-1]),
        )
        if cache is None:
            cache = [None] * len(owner.model.layers)
        taps = []
        for layer_id, (layer, layer_cache) in enumerate(zip(owner.model.layers, cache)):
            hidden = layer(
                hidden,
                mask=None,
                cache=layer_cache,
                input_ids=inputs,
            )
            if layer_id in self.target_layer_ids:
                taps.append(mx.mean(hidden, axis=2))
        if len(taps) != len(self.target_layer_ids):
            raise RuntimeError("DSpark target route did not observe every required tap")
        return hidden, tuple(taps)


def _validate_dspark_args(args) -> None:
    observed = (
        int(args.dspark_block_size or 0),
        int(args.dspark_noise_token_id or 0),
        tuple(int(value) for value in (args.dspark_target_layer_ids or ())),
        int(args.dspark_markov_rank or 0),
        int(args.num_nextn_predict_layers),
    )
    expected = (
        DSPARK_BLOCK_SIZE,
        DSPARK_NOISE_TOKEN_ID,
        DSPARK_TARGET_LAYER_IDS,
        DSPARK_MARKOV_RANK,
        1,
    )
    if observed != expected:
        raise ValueError(
            f"unsupported DeepSeek V4 DSpark contract: observed={observed!r}, "
            f"expected={expected!r}"
        )
    if int(args.num_hidden_layers) <= DSPARK_TARGET_LAYER_IDS[-1]:
        raise ValueError("DeepSeek V4 DSpark target taps are absent from the trunk")
    if int(args.vocab_size) <= DSPARK_NOISE_TOKEN_ID:
        raise ValueError("DeepSeek V4 DSpark vocabulary omits the noise token")
    ratios = tuple(int(value) for value in args.compress_ratios)
    for layer_id in range(
        int(args.num_hidden_layers),
        int(args.num_hidden_layers) + DSPARK_STAGE_COUNT,
    ):
        if layer_id < len(ratios) and ratios[layer_id] != 0:
            raise ValueError("DeepSeek V4 DSpark stages require uncompressed attention")


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """Official DSpark attention with a Mia stock432 stage context ring."""

    def __init__(self, args: ModelArgs, layer_id: int) -> None:
        super().__init__(args, layer_id)
        if self.compress_ratio != 0:
            raise ValueError("DSpark attention requires compress_ratio=0")
        self._dspark_k5_mla = install_dspark_k5_nvfp4_mla(
            heads=self.n_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            window_size=self.window_size,
            block_size=DSPARK_BLOCK_SIZE,
        )
        self._dspark_k5_mla_graph = None
        self._pack_draft_records = install_stock432_record_packer(
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
        )
        self._project_kv_impl = self._stock_project_kv
        self._prefill_context_impl = self._stock_prefill_context
        self._forward_impl = self._checked_forward_k5

    def install_mia_k5_runtime(self) -> None:
        """Bind the already-qualified fixed K5 attention route."""
        if (
            self.n_heads != 64
            or self.head_dim != 512
            or self.rope_head_dim != 64
            or self.window_size != 128
        ):
            raise ValueError("Mia DSpark attention geometry changed")
        self._mia_attn_sink = self.attn_sink.astype(mx.float32)
        self._mia_mla_query_layout = "BMHD"
        self._mia_mla_output_layout = "BMHD"
        self._mia_draft_position_offsets = mx.arange(
            DSPARK_BLOCK_SIZE,
            dtype=mx.int32,
        )
        mx.eval(self._mia_attn_sink)
        mx.eval(self._mia_draft_position_offsets)
        self._dspark_k5_mla = partial(
            self._dspark_k5_mla,
            sinks=self._mia_attn_sink,
            scale=self.softmax_scale,
        )
        self._dspark_k5_mla_graph = partial(
            install_dspark_k5_nvfp4_mla_graph(
                heads=self.n_heads,
                head_dim=self.head_dim,
                rope_dim=self.rope_head_dim,
                window_size=self.window_size,
                block_size=DSPARK_BLOCK_SIZE,
            ),
            sinks=self._mia_attn_sink,
            scale=self.softmax_scale,
        )

    def install_mia_qkv_prologue(self, plan) -> None:
        super().install_mia_qkv_prologue(plan)
        self._pack_draft_records = None
        self._project_kv_impl = None
        self._project_context_records_impl = self._mia_context_records
        self._prefill_context_impl = self._mia_prefill_context_records
        self._forward_impl = self._run_k5

    def project_kv(
        self,
        hidden: mx.array,
        positions_or_start,
    ) -> tuple[mx.array, mx.array]:
        return self._project_kv_impl(hidden, positions_or_start)

    def _stock_project_kv(
        self,
        hidden: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        rope_dim = self.rope_head_dim
        cos, sin = self._rope_tables(positions)
        latent = self.kv_norm(self.wkv(hidden))
        rope = _apply_interleaved_rope(
            latent[..., -rope_dim:],
            cos[None],
            sin[None],
        )
        return latent, rope

    def _mia_project_kv(
        self,
        hidden: mx.array,
        start_pos: int,
    ) -> tuple[mx.array, mx.array]:
        """Project an exact context slice from the shared base-theta graph."""
        rope_dim = self.rope_head_dim
        _positions, cos, sin = self._mia_token_rope_tables(
            int(start_pos),
            int(hidden.shape[1]),
        )
        _query_rank_raw, latent_raw = self._mia_input_projection(hidden)
        latent = self.kv_norm(latent_raw)
        rope = _apply_interleaved_rope(
            latent[..., -rope_dim:],
            cos[None],
            sin[None],
        )
        return latent, rope

    def project_context_records(
        self,
        hidden: mx.array,
        start_pos: int,
    ) -> mx.array:
        return self._project_context_records_impl(hidden, int(start_pos))

    def _mia_context_records(
        self,
        hidden: mx.array,
        start_pos: int,
    ) -> mx.array:
        _positions, cos, sin = self._mia_token_rope_tables(
            int(start_pos),
            int(hidden.shape[1]),
        )
        latent = self._mia_qkv_plan.project_kv(hidden)
        return self._mia_qkv_plan.context_records(latent, cos, sin)

    def prefill_context(
        self,
        main_hidden: mx.array,
        cache: DeepseekV4DSparkCache,
    ) -> None:
        return self._prefill_context_impl(main_hidden, cache)

    def _stock_prefill_context(
        self,
        main_hidden: mx.array,
        cache: DeepseekV4DSparkCache,
    ) -> None:
        total_length = int(main_hidden.shape[1])
        tail_start = max(0, total_length - self.window_size)
        positions = mx.arange(tail_start, total_length)
        latent, rope = self.project_kv(
            main_hidden[:, tail_start:],
            positions,
        )
        cache._install_prefill_tail(
            latent,
            rope,
            total_length=total_length,
        )

    def _mia_prefill_context_records(
        self,
        main_hidden: mx.array,
        cache: DeepseekV4DSparkCache,
    ) -> None:
        total_length = int(main_hidden.shape[1])
        tail_start = max(0, total_length - self.window_size)
        records = self.project_context_records(
            main_hidden[:, tail_start:], tail_start
        )
        cache._install_prefill_records(
            records,
            absolute_start=tail_start,
            total_length=total_length,
        )

    def __call__(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
    ) -> mx.array:
        return self._forward_impl(hidden, start_pos=start_pos, cache=cache)

    def _checked_forward_k5(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
    ) -> mx.array:
        if int(start_pos) <= 0:
            raise ValueError("DSpark decode attention requires a positive position")
        _batch, block, _width = hidden.shape
        if int(block) != DSPARK_BLOCK_SIZE:
            raise ValueError("DSpark decode requires five neural rows")
        return self._run_k5(hidden, start_pos=start_pos, cache=cache)

    def _run_k5(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
    ) -> mx.array:
        """Installed physical K5 attention with a fixed stock432 union."""
        batch = int(hidden.shape[0])
        block = DSPARK_BLOCK_SIZE
        _positions, cos, sin = self._mia_token_rope_tables(start_pos, block)
        query_rank, draft_latent = self._mia_qkv_plan.project_learned(hidden)
        query_pre = self.wq_b(query_rank).reshape(
            batch,
            block,
            self.n_heads,
            self.head_dim,
        )
        query, draft_records = self._mia_qkv_plan.proposal_records(
            query_pre,
            draft_latent,
            cos,
            sin,
        )
        output = self._dspark_k5_mla(
            query,
            cache.ring.records,
            draft_records,
            int(start_pos),
        )
        return self._project_attention_output(output, cos, sin)

    def _run_k5_graph(
        self,
        hidden: mx.array,
        context_records: mx.array,
        start_position: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        """The same installed K5 arithmetic with graph inputs made explicit."""

        batch = int(hidden.shape[0])
        block = DSPARK_BLOCK_SIZE
        query_rank, draft_latent = self._mia_qkv_plan.project_learned(hidden)
        query_pre = self.wq_b(query_rank).reshape(
            batch,
            block,
            self.n_heads,
            self.head_dim,
        )
        query, draft_records = self._mia_qkv_plan.proposal_records(
            query_pre,
            draft_latent,
            cos,
            sin,
        )
        output = self._dspark_k5_mla_graph(
            query,
            context_records,
            draft_records,
            start_position,
        )
        return self._project_attention_output(output, cos, sin)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embedding = self.markov_w1(token_ids)
        return self.markov_w2(embedding), embedding


class DSparkConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, markov_rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=False)


class DeepseekV4DSparkStage(DeepseekV4DecoderLayer):
    """One stage of the three-layer DSpark owner."""

    def __init__(self, args: ModelArgs, stage_id: int) -> None:
        layer_id = int(args.num_hidden_layers) + int(stage_id)
        ratios = list(args.compress_ratios)
        if len(ratios) <= layer_id:
            ratios.extend([0] * (layer_id + 1 - len(ratios)))
            args = replace(args, compress_ratios=ratios)
        super().__init__(args, layer_id)
        self.attn = DeepseekV4DSparkAttention(args, layer_id)
        self.stage_id = int(stage_id)
        self.main_proj = None
        self.main_norm = None
        self.norm = None
        self.hc_head = None
        self.markov_head = None
        self.confidence_head = None
        if self.stage_id == 0:
            self.main_proj = nn.Linear(
                args.hidden_size * len(args.dspark_target_layer_ids),
                args.hidden_size,
                bias=False,
            )
            self.main_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if self.stage_id == DSPARK_STAGE_COUNT - 1:
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hc_head = HeadHC(args.hidden_size, args.hc_mult, args.hc_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size,
                args.dspark_markov_rank,
            )
            self.confidence_head = DSparkConfidenceHead(
                args.hidden_size,
                args.dspark_markov_rank,
            )

    def fuse_main(self, target_taps: tuple[mx.array, mx.array, mx.array]) -> mx.array:
        return self.fuse_main_rows(mx.concatenate(target_taps, axis=-1))

    def fuse_main_rows(self, target_rows: mx.array) -> mx.array:
        if self.main_proj is None or self.main_norm is None:
            raise RuntimeError("DSpark target-tap fusion belongs to stage zero")
        return self._run_fuse_main_rows(target_rows)

    def _run_fuse_main_rows(self, target_rows: mx.array) -> mx.array:
        return self.main_norm(self.main_proj(target_rows))

    def __call__(
        self,
        hidden: mx.array,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkCache,
        input_ids: mx.array,
    ) -> mx.array:
        residual = hidden
        value, post, combination = self.attn_hc.pre(hidden)
        value = self.attn_norm(value)
        value = self.attn(
            value,
            start_pos=start_pos,
            cache=cache,
        )
        hidden = self.attn_hc.post(value, residual, post, combination)

        residual = hidden
        value, post, combination = self.ffn_hc.pre(hidden)
        value = self.ffn_norm(value)
        value = self.ffn(value, input_ids=input_ids)
        return self.ffn_hc.post(value, residual, post, combination)


@dataclass(frozen=True)
class DSparkModelProposal:
    future_tokens: mx.array
    neural_logits: mx.array


class DeepseekV4DSparkOwner:
    """Three stage-owned DSpark blocks with an unambiguous future-token API."""

    def __init__(self, args, stages) -> None:
        self.args = args
        self.block_size = DSPARK_BLOCK_SIZE
        self.noise_token_id = DSPARK_NOISE_TOKEN_ID
        self.target_layer_ids = DSPARK_TARGET_LAYER_IDS
        self.stages = list(stages)
        if len(self.stages) != DSPARK_STAGE_COUNT:
            raise ValueError("DeepSeek V4 DSpark requires exactly three stages")
        self._propose_impl = self._stock_propose_k5
        self._make_cache_impl = self._new_cache
        self._commit_main_impl = self._stock_commit_main
        self._mia_fullgraph_route = None

    def install_mia_mhc_runtime(self, *, max_tokens: int) -> None:
        from mtplx.kernels.deepseek_v4_mhc import MiaMHCPlan

        self._mia_mhc = MiaMHCPlan(
            max_tokens=max_tokens,
            rms_eps=self.args.rms_norm_eps,
            hc_eps=self.args.hc_eps,
            iters=self.args.hc_sinkhorn_iters,
        )
        self._mia_mhc.install_modules(
            hyper_connections=tuple(
                connection
                for stage in self.stages
                for connection in (stage.attn_hc, stage.ffn_hc)
            ),
            broadcast_connection=self.stages[0].attn_hc,
        )
        for stage in self.stages:
            stage.attn.install_mia_k5_runtime()
            stage.ffn.install_mia_k64_runtime()
        self._mia_noise_tail = mx.full(
            (1, self.block_size - 1),
            self.noise_token_id,
            dtype=mx.uint32,
        )
        mx.eval(self._mia_noise_tail)
        self._mia_cache_arena = tuple(self._new_cache())
        self._mia_cache_leased = False
        self._make_cache_impl = self._acquire_mia_cache
        self._commit_main_impl = self._mia_commit_main
        self._propose_impl = self._mia_propose_k5

    def install_mia_fullgraph_runtime(self, *, embed_tokens, lm_head):
        """Bind Mia's fixed physical-K5 proposal as one compiled graph."""

        from mtplx.deepseek_v4_mia_draft_graph import (
            MiaPhysicalK5FullGraphDraftRoute,
        )

        if self._mia_fullgraph_route is not None:
            raise ValueError("the Mia DSpark full graph is already installed")
        stages = tuple(self.stages)
        providers = tuple(stage.attn._mia_rope_provider for stage in stages)
        if (
            len(stages) != DSPARK_STAGE_COUNT
            or getattr(self, "_mia_mhc", None) is None
            or any(stage.attn._dspark_k5_mla_graph is None for stage in stages)
            or providers[0] is None
            or any(provider is not providers[0] for provider in providers[1:])
        ):
            raise ValueError("the Mia DSpark graph dependencies are incomplete")
        route = MiaPhysicalK5FullGraphDraftRoute(
            self._make_mia_fullgraph_proposal(embed_tokens, lm_head),
        )
        self._mia_fullgraph_route = route
        self._propose_impl = self._mia_fullgraph_propose_k5
        return route

    def _make_mia_fullgraph_proposal(self, embed_tokens, lm_head):
        stages = tuple(self.stages)
        mhc = self._mia_mhc
        hidden_size = int(self.args.hidden_size)
        lead = (1, DSPARK_BLOCK_SIZE)
        noise_tail = self._mia_noise_tail
        position_offsets = stages[0].attn._mia_draft_position_offsets
        inv_freq = stages[0].attn._mia_rope_provider.inv_freq

        def proposal_graph(
            primary_token_ids,
            cache0_records,
            cache1_records,
            cache2_records,
            start_position,
        ):
            input_ids = mx.concatenate(
                [primary_token_ids[:, None], noise_tail],
                axis=1,
            )
            positions = start_position[0] + position_offsets
            angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
            cos = mx.cos(angles)
            sin = mx.sin(angles)
            cache_records = (cache0_records, cache1_records, cache2_records)

            first = stages[0]
            residual, post, comb, value = mhc.pre_broadcast(
                embed_tokens(input_ids),
                first.attn_hc,
                first.attn_norm,
            )
            value = first.attn._run_k5_graph(
                value.reshape(*lead, hidden_size),
                cache_records[0],
                start_position,
                cos,
                sin,
            )
            residual, post, comb, value = mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                first.ffn_hc,
                first.ffn_norm,
            )
            value = first.ffn(
                value.reshape(*lead, hidden_size),
                input_ids=input_ids,
            )

            for stage, records in zip(stages[1:], cache_records[1:], strict=True):
                residual, post, comb, value = mhc.post_pre_attn(
                    value,
                    residual,
                    post,
                    comb,
                    stage.attn_hc,
                    stage.attn_norm,
                )
                value = stage.attn._run_k5_graph(
                    value.reshape(*lead, hidden_size),
                    records,
                    start_position,
                    cos,
                    sin,
                )
                residual, post, comb, value = mhc.post_pre_ffn(
                    value,
                    residual,
                    post,
                    comb,
                    stage.ffn_hc,
                    stage.ffn_norm,
                )
                value = stage.ffn(
                    value.reshape(*lead, hidden_size),
                    input_ids=input_ids,
                )

            hidden = mhc.post(value, residual, post, comb)
            final = stages[-1]
            neural_hidden = mhc.head(hidden, final.hc_head).reshape(
                *lead,
                hidden_size,
            )
            neural_logits = lm_head(final.norm(neural_hidden))
            future_tokens = _run_greedy_future_tokens_k5(
                neural_logits,
                primary_token_ids,
                final.markov_head,
            )
            return future_tokens, neural_logits

        return proposal_graph

    def _mia_fullgraph_propose_k5(
        self,
        primary_token_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> DSparkModelProposal:
        del embed_tokens, lm_head
        future_tokens, neural_logits = self._mia_fullgraph_route(
            primary_token_ids,
            caches[0].ring.records,
            caches[1].ring.records,
            caches[2].ring.records,
            start_pos,
        )
        return DSparkModelProposal(
            future_tokens=future_tokens,
            neural_logits=neural_logits,
        )

    def draft_input_ids(self, primary_token_ids: mx.array) -> mx.array:
        if primary_token_ids.ndim != 1:
            raise ValueError("DSpark primary ids must have shape [batch]")
        return self._draft_input_ids_k5(primary_token_ids)

    def _draft_input_ids_k5(self, primary_token_ids: mx.array) -> mx.array:
        noise = mx.full(
            (primary_token_ids.shape[0], self.block_size - 1),
            self.noise_token_id,
            dtype=primary_token_ids.dtype,
        )
        return mx.concatenate([primary_token_ids[:, None], noise], axis=1)

    def _mia_draft_input_ids_k5(self, primary_token_ids: mx.array) -> mx.array:
        return mx.concatenate(
            [primary_token_ids[:, None], self._mia_noise_tail],
            axis=1,
        )

    def _new_cache(self) -> list[DeepseekV4DSparkCache]:
        return [
            DeepseekV4DSparkCache(
                window_size=stage.attn.window_size,
                head_dim=stage.attn.head_dim,
            )
            for stage in self.stages
        ]

    def _acquire_mia_cache(self) -> list[DeepseekV4DSparkCache]:
        if self._mia_cache_leased:
            raise RuntimeError("Mia DSpark cache arena already owns the active request")
        for cache in self._mia_cache_arena:
            cache.ring.clear()
            cache.prefill_length = 0
        self._mia_cache_leased = True
        return list(self._mia_cache_arena)

    def release_mia_cache(self, caches: list[DeepseekV4DSparkCache]) -> None:
        if not self._mia_cache_leased:
            raise RuntimeError("Mia DSpark cache arena has no active request")
        if len(caches) != len(self._mia_cache_arena) or any(
            observed is not expected
            for observed, expected in zip(caches, self._mia_cache_arena, strict=True)
        ):
            raise ValueError("Mia DSpark cache release does not match its page lease")
        for cache in self._mia_cache_arena:
            cache.ring.clear()
            cache.prefill_length = 0
        self._mia_cache_leased = False

    def make_cache(self) -> list[DeepseekV4DSparkCache]:
        return self._make_cache_impl()

    def prefill(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
    ) -> None:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark prefill requires one cache per stage")
        main_hidden = self.stages[0].fuse_main(target_taps)
        for stage, cache in zip(self.stages, caches):
            stage.attn.prefill_context(main_hidden, cache)

    def commit_main(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> None:
        return self._commit_main_impl(
            target_taps,
            caches,
            start_pos=start_pos,
        )

    def _stock_commit_main(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> None:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark commit requires one cache per stage")
        main_hidden = self.stages[0].fuse_main(target_taps)
        positions = mx.arange(int(start_pos), int(start_pos) + int(main_hidden.shape[1]))
        for stage, cache in zip(self.stages, caches):
            latent, rope = stage.attn.project_kv(main_hidden, positions)
            cache.commit_main(start_pos, latent, rope)

    def _mia_commit_main(
        self,
        target_taps: tuple[mx.array, mx.array, mx.array],
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> None:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark commit requires one cache per stage")
        main_hidden = self.stages[0].fuse_main(target_taps)
        for stage, cache in zip(self.stages, caches):
            records = stage.attn.project_context_records(
                main_hidden, int(start_pos)
            )
            cache._commit_records(start_pos, records)

    def _stock_propose_k5(
        self,
        primary_token_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> DSparkModelProposal:
        if len(caches) != DSPARK_STAGE_COUNT:
            raise ValueError("DSpark proposal requires one cache per stage")
        input_ids = self.draft_input_ids(primary_token_ids)
        hidden = embed_tokens(input_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], self.args.hc_mult, hidden.shape[-1]),
        )
        for stage, cache in zip(self.stages, caches):
            hidden = stage(
                hidden,
                start_pos=start_pos,
                cache=cache,
                input_ids=input_ids,
            )
        final = self.stages[-1]
        if final.hc_head is None or final.norm is None or final.markov_head is None:
            raise RuntimeError("DSpark final stage does not own its output heads")
        neural_logits = lm_head(final.norm(final.hc_head(hidden)))
        return DSparkModelProposal(
            future_tokens=greedy_future_tokens(
                neural_logits,
                primary_token_ids,
                final.markov_head,
            ),
            neural_logits=neural_logits,
        )

    def _mia_propose_k5(
        self,
        primary_token_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> DSparkModelProposal:
        input_ids = self._mia_draft_input_ids_k5(primary_token_ids)
        lead = tuple(int(value) for value in input_ids.shape)
        first = self.stages[0]
        residual, post, comb, value = self._mia_mhc.pre_broadcast(
            embed_tokens(input_ids),
            first.attn_hc,
            first.attn_norm,
        )
        value = first.attn(
            value.reshape(*lead, self.args.hidden_size),
            start_pos=start_pos,
            cache=caches[0],
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
        for stage, cache in zip(self.stages[1:3], caches[1:3]):
            residual, post, comb, value = self._mia_mhc.post_pre_attn(
                value,
                residual,
                post,
                comb,
                stage.attn_hc,
                stage.attn_norm,
            )
            value = stage.attn(
                value.reshape(*lead, self.args.hidden_size),
                start_pos=start_pos,
                cache=cache,
            )
            residual, post, comb, value = self._mia_mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                stage.ffn_hc,
                stage.ffn_norm,
            )
            value = stage.ffn(
                value.reshape(*lead, self.args.hidden_size),
                input_ids=input_ids,
            )
        hidden = self._mia_mhc.post(value, residual, post, comb)
        final = self.stages[-1]
        neural_hidden = self._mia_mhc.head(hidden, final.hc_head).reshape(
            *lead, self.args.hidden_size
        )
        neural_hidden = final.norm(neural_hidden)
        neural_logits = lm_head(neural_hidden)
        return DSparkModelProposal(
            future_tokens=_run_greedy_future_tokens_k5(
                neural_logits,
                primary_token_ids,
                final.markov_head,
            ),
            neural_logits=neural_logits,
        )

    def propose_k5(
        self,
        primary_token_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        caches: list[DeepseekV4DSparkCache],
        *,
        start_pos: int,
    ) -> DSparkModelProposal:
        return self._propose_impl(
            primary_token_ids,
            embed_tokens,
            lm_head,
            caches,
            start_pos=start_pos,
        )


def build_deepseek_v4_dspark(args) -> DeepseekV4DSparkOwner:
    _validate_dspark_args(args)
    return DeepseekV4DSparkOwner(
        args,
        [DeepseekV4DSparkStage(args, stage_id) for stage_id in range(DSPARK_STAGE_COUNT)],
    )
