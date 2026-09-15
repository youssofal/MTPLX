"""Deferred streamed PLE rows for Qwen4 pipelined autoregressive decode.

The adapter is installed against one validated Qwen4 sidecar.  During graph
construction it supplies three small, MLX-owned packed row leaves.  ``flush``
materializes the sampled token, gathers the exact sidecar bytes into those
leaves, and only then advances the PLE history.  Dequantization stays in the
ordinary model graph, preserving the existing arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np


AR_ROWS = 16


@dataclass(slots=True)
class _Pending:
    handle: Any
    input_ids: Any
    previous: Any
    cache: Any
    state_idx: int


class StreamedArPle:
    """One construction-bound deferred PLE route for S=1 decode."""

    __slots__ = (
        "_native",
        "_mx",
        "_rows",
        "_gather_planes",
        "_eos_id",
        "_context_len",
        "_output_dim",
        "_bits",
        "_group_size",
        "_pending",
        "_failure",
    )

    def __init__(
        self,
        *,
        native_module: Any,
        mx_module: Any,
        rows: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        gather_planes: Callable[
            [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
        eos_id: int,
        context_len: int,
        output_dim: int,
        bits: int,
        group_size: int,
    ) -> None:
        factories = (
            "make_deferred_ar_rows",
            "make_deferred_ar_token",
        )
        missing = [
            name
            for name in factories
            if not callable(getattr(native_module, name, None))
        ]
        if missing:
            raise ValueError(
                "streamed AR PLE requires native factories: " + ", ".join(missing)
            )
        self._native = native_module
        self._mx = mx_module
        self._rows = rows
        self._gather_planes = gather_planes
        self._eos_id = int(eos_id)
        self._context_len = int(context_len)
        self._output_dim = int(output_dim)
        self._bits = int(bits)
        self._group_size = int(group_size)
        self._pending: _Pending | None = None
        self._failure: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def _ensure_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "streamed AR PLE route failed; model reload required"
            ) from self._failure

    def set_active(self, enabled: bool) -> None:
        if enabled:
            self._ensure_healthy()
        if not enabled and self._pending is not None:
            raise RuntimeError("cannot disable streamed AR PLE with a pending leaf")

    def make_token(self) -> Any:
        """Create the mutable token leaf bound to this installed native ABI."""

        return self._native.make_deferred_ar_token()

    def build(self, input_ids: Any, cache: Any, state_idx: int) -> Any:
        """Build the PLE graph on an unfilled packed-row leaf."""

        previous = cache[state_idx]
        if previous is None:
            previous = self._mx.full(
                (1, self._context_len), self._eos_id, dtype=self._mx.int64
            )

        handle = self._native.make_deferred_ar_rows()
        weight, scales, biases = handle.planes()
        embedding = self._mx.dequantize(
            weight,
            scales,
            biases,
            group_size=self._group_size,
            bits=self._bits,
        ).reshape(1, 1, self._output_dim)
        self._pending = _Pending(
            handle=handle,
            input_ids=input_ids,
            previous=previous,
            cache=cache,
            state_idx=int(state_idx),
        )
        return embedding

    def flush(self) -> None:
        """Fill one pending leaf and commit its concrete PLE history."""

        pending = cast(_Pending, self._pending)
        try:
            input_ids = np.asarray(pending.input_ids, dtype=np.int64).reshape(1, 1)
            previous = np.asarray(pending.previous, dtype=np.int64).reshape(
                1, self._context_len
            )
            rows, new_history = self._rows(input_ids, previous)
            flat = np.ascontiguousarray(rows.reshape(-1), dtype=np.int64)
            weight, scales, biases = self._gather_planes(flat)
            pending.handle.fill(weight, scales, biases)
            pending.cache[pending.state_idx] = self._mx.array(new_history)
            self._pending = None
        except BaseException as error:
            self._failure = error
            raise

    def discard(self) -> None:
        """Abandon an unsubmitted graph without committing its history."""

        self._pending = None


__all__ = ["AR_ROWS", "StreamedArPle"]
