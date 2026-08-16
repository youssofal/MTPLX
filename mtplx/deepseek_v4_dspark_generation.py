"""DeepSeek-V4-0731 adapter for MTPLX's generic block speculation engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class DeepseekV4DSparkBackend:
    """Construction-selected DSpark operations; target policy stays generic."""

    dspark: Any
    embed_tokens: Any
    lm_head: Any
    target_forward: Callable[..., Any]
    backend_id: str = "deepseek_v4_dspark_0731"
    supported_depths: tuple[int, ...] = (2,)
    prefill_chunk_size: int = 128
    # Absolute zero selects DSpark's attention-only prefill branch.  A
    # one-token prompt therefore needs one explicit target seed before the
    # primary-inclusive fixed-K loop can start from main position one.
    minimum_proposal_target_position: int = 2

    @classmethod
    def bind(
        cls,
        model: Any,
        *,
        supported_depths: tuple[int, ...] = (2,),
    ) -> "DeepseekV4DSparkBackend":
        """Validate ownership once and bind direct hot-path callables."""
        dspark = getattr(model, "_dspark", None)
        inner = getattr(model, "model", None)
        embed_tokens = getattr(inner, "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        if (
            dspark is None
            or not callable(embed_tokens)
            or not callable(lm_head)
            or not callable(model)
        ):
            raise ValueError("DeepSeek-V4 DSpark backend cannot bind model ownership")
        stage_count = len(tuple(getattr(dspark, "stages", ())))
        if stage_count != 3:
            raise ValueError("DeepSeek-V4 DSpark backend requires three owned stages")
        selected_depths = tuple(int(depth) for depth in supported_depths)
        if (
            not selected_depths
            or len(set(selected_depths)) != len(selected_depths)
            or any(depth < 1 or depth > stage_count for depth in selected_depths)
        ):
            raise ValueError(
                "DeepSeek-V4 DSpark construction depths must be unique values from 1 to 3"
            )
        return cls(
            dspark=dspark,
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            target_forward=model,
            supported_depths=selected_depths,
        )

    def make_cache(self, rt: Any) -> Any:
        del rt
        return self.dspark.make_cache()

    def prefill(self, rt: Any, hidden: mx.array, cache: Any) -> None:
        del rt
        self.dspark.prefill(hidden, cache)

    def prefill_chunk(
        self,
        rt: Any,
        hidden: mx.array,
        cache: Any,
        *,
        start_pos: int,
    ) -> None:
        """Install one construction-fixed 0731 sliding-window prompt chunk."""
        del rt
        start_pos = int(start_pos)
        if start_pos == 0:
            self.dspark.prefill(hidden, cache)
        else:
            self.dspark.commit_main(hidden, cache, start_pos=start_pos)
        prefill_length = start_pos + int(hidden.shape[1])
        for entry in cache:
            entry.prefill_length = prefill_length

    def propose(
        self,
        rt: Any,
        hidden: mx.array,
        token_id: int,
        primary_token_id: int,
        cache: Any,
        *,
        start_pos: int,
        width: int,
    ) -> mx.array:
        del rt
        # A logical MTP-N request drafts only N future tokens.  The target has
        # already sampled the primary token from its carried logits, so that
        # authoritative id seeds DSpark's sequential Markov recurrence.  The
        # neural block still evaluates primary + N rows in parallel; the ids-only
        # API returns [last_target, primary, future...], and only future rows
        # leave this adapter.
        #
        # The generic engine names ``start_pos`` as the next target position.
        # DSpark names it as the position of ``hidden``/``token_id`` (upstream
        # passes ``checkpoint.len - 1`` after committing that row), so translate
        # ownership once at this adapter boundary.  Target verification and
        # commit continue to use the unshifted next-target position.
        main_position = int(start_pos) - 1
        draft_ids = self.dspark.forward(
            hidden,
            mx.array([int(token_id)], dtype=mx.int32),
            self.embed_tokens,
            self.lm_head,
            cache,
            start_pos=main_position,
            greedy=True,
            ids_only_width=int(width) + 1,
            forced_first_token_ids=mx.array([int(primary_token_id)], dtype=mx.int32),
        )
        return draft_ids[:, 2:]

    def commit(
        self,
        rt: Any,
        hidden: mx.array,
        cache: Any,
        *,
        start_pos: int,
    ) -> None:
        del rt
        self.dspark.commit_main(hidden, cache, start_pos=int(start_pos))

    def cache_roots(self, cache: Any) -> list[mx.array]:
        return [
            ring
            for entry in cache
            if (ring := getattr(entry, "ring", None)) is not None
        ]

    def bind_target_forward(self, rt: Any) -> Callable[..., Any]:
        """Bind the construction-certified target route before generation."""
        del rt
        return self.target_forward

    def snapshot(self, cache: Any) -> tuple[tuple[mx.array | None, int], ...]:
        """Capture DSpark-owned stage rings before a speculative proposal."""
        return tuple(
            (
                None if entry.ring is None else entry.ring[...],
                int(entry.prefill_length),
            )
            for entry in cache
        )

    def restore(
        self,
        cache: Any,
        snapshot: tuple[tuple[mx.array | None, int], ...],
    ) -> None:
        """Restore only the proposal backend's ring ownership."""
        for entry, (ring, prefill_length) in zip(cache, snapshot):
            entry.ring = ring
            entry.prefill_length = int(prefill_length)

    def rollback_target(self, target_cache: Any, rejected_rows: int) -> None:
        # Eligibility is proven when the DeepSeek-V4 backend is installed: all
        # 43 target cache entries are exact trimmable DeepseekV4Cache instances.
        # The enabled path executes that route directly without probing/fallback.
        for entry in target_cache:
            entry.trim(int(rejected_rows))


def generate_deepseek_v4_dspark(rt: Any, prompt_ids: list[int], **kwargs: Any):
    """Compatibility wrapper for callers that used the old dedicated runner."""
    from .native_block_speculation import generate_native_block_speculative

    return generate_native_block_speculative(
        rt,
        DeepseekV4DSparkBackend.bind(rt.model),
        prompt_ids,
        **kwargs,
    )
