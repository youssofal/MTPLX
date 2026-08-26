"""Construction-bound full physical-K5 proposal replay for Mia DSpark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx


class MiaPhysicalK5FullGraphDraftRoute:
    """Compile one fixed-shape proposal graph and feed it live cache pages."""

    physical_width = 5

    def __init__(
        self,
        proposal_graph: Callable[..., tuple[mx.array, mx.array]],
        *,
        compile_fn: Callable[[Callable[..., Any]], Callable[..., Any]] = mx.compile,
    ) -> None:
        self._compiled = compile_fn(proposal_graph)

    def __call__(
        self,
        primary_token_ids: mx.array,
        cache0_records: mx.array,
        cache1_records: mx.array,
        cache2_records: mx.array,
        start_pos: int,
    ) -> tuple[mx.array, mx.array]:
        start_position = mx.array([int(start_pos)], dtype=mx.int32)
        return self._compiled(
            primary_token_ids,
            cache0_records,
            cache1_records,
            cache2_records,
            start_position,
        )

