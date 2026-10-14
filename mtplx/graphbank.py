"""Speculative decode graph-bank scaffolding for MLX.

The first useful job of this module is to make graph-capture eligibility
explicit.  The current Qwen3.6 MLX cache keeps full-attention positions as
Python integers, so a safe compiled decode graph cannot replay across decode
steps until those offsets become tensor inputs/outputs.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
import weakref
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable

import mlx.core as mx

from .attention_context import attention_phase, compiled_dispatch
from .compile_state import compiled_step_body
from .demotions import note as _note_demotion, note_bank_fallback as _note_bank_fallback
from .errors import is_allocation_failure
from .gdn_capture import resolve_gdn_capture_backend
from .rope_origin import RotaryOrigin, as_rope_delta, stamp_rope_delta


# Process-wide state of the Flash-Next fixed-M4 compiled lane (PX.4):
# "proven" flips after the first dispatch evaluated cleanly on this GPU;
# "retired" holds the one-line reason after a dispatch failure, and the lane
# stays on the eager verifier for the life of the process.
_FIXED_M4_LANE: dict[str, Any] = {"proven": False, "retired": None}
_MISSING = object()


def fixed_m4_lane_retired_reason() -> str | None:
    """Why the fixed-M4 compiled lane is retired for this process, or None."""

    return _FIXED_M4_LANE["retired"]


def retire_fixed_m4_lane(reason: str) -> None:
    """Retire the lane for the process: one printed line, one ledger entry."""

    first = _FIXED_M4_LANE["retired"] is None
    text = " ".join(str(reason).split())[:400]
    if first:
        _FIXED_M4_LANE["retired"] = text
    _note_demotion("fixed_m4_dispatch_retired", _FIXED_M4_LANE["retired"])
    if not first:
        return
    try:
        print(
            "[mtplx] Flash-Next compiled verifier could not dispatch on this GPU "
            f"({text}). Verify runs the eager path for the rest of this process; "
            "output is unchanged. Please report this line with `mtplx doctor "
            "--json`.",
            flush=True,
        )
    except Exception:
        pass


# Process-wide state of compiled copy windows on the same lane
# (MTPLX_FIXED_M4_COPY_WINDOWS=1): the widths whose program traced cleanly in
# this process, and, after a trace failure, the one-line reason every copy
# window runs the eager forward for the rest of the process. The four-row lane
# is not touched by it.
_FIXED_M4_COPY_WINDOWS: dict[str, Any] = {"proven": set(), "retired": None}


def fixed_m4_copy_windows_enabled() -> bool:
    """Compiled copy windows on the fixed-M4 lane (opt-in, MTPLX_FIXED_M4_COPY_WINDOWS=1)."""

    return _env_enabled("MTPLX_FIXED_M4_COPY_WINDOWS")


def fixed_m4_copy_windows_retired_reason() -> str | None:
    """Why compiled copy windows are retired for this process, or None."""

    return _FIXED_M4_COPY_WINDOWS["retired"]


def retire_fixed_m4_copy_windows(reason: str) -> None:
    """Retire compiled copy windows for the process: one printed line, one ledger entry."""

    first = _FIXED_M4_COPY_WINDOWS["retired"] is None
    text = " ".join(str(reason).split())[:400]
    if first:
        _FIXED_M4_COPY_WINDOWS["retired"] = text
    _note_demotion("fixed_m4_copy_windows_retired", _FIXED_M4_COPY_WINDOWS["retired"])
    if not first:
        return
    try:
        print(
            "[mtplx] Flash-Next compiled copy windows could not dispatch on this "
            f"GPU ({text}). Copy rounds run the eager forward for the rest of "
            "this process; output is unchanged.",
            flush=True,
        )
    except Exception:
        pass


# Demotion ledger reasons for the per-round sites: constants, so the verify
# path formats nothing (mtplx/demotions.py cost contract).
_FIXED_M4_OTHER_WIDTH_REASON = (
    "verify width other than 4 has no compiled route: an adaptive stop, the "
    "last round before max_tokens, or a draft depth of 4 or 5"
)
_FIXED_M4_RETIRED_ROUND_REASON = (
    "the compiled verifier was retired for this process after a dispatch "
    "failure, so this round ran the eager forward"
)
_FIXED_M4_NO_HOST_INPUTS_REASON = (
    "a width-4 verify reached the bank without host-owned n-gram inputs (a "
    "copy block of 3), so it ran the eager forward"
)


def _prepare_fixed_m4_materialized(
    prepare_aux,
    cache,
    input_ids,
    _host_input_ids,
    _completion_tokens,
    _committed_count,
):
    """Adapt the materialized PLE route to the fixed-M4 host-input contract."""

    return prepare_aux(input_ids, cache)


@dataclass
class GraphBankStats:
    calls: int = 0
    compiled_calls: int = 0
    fallback_calls: int = 0
    promoted_cache_entries: int = 0
    warmed_lengths: list[int] = field(default_factory=list)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    compile_errors: dict[str, int] = field(default_factory=dict)
    promotion_failures: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecDecodeGraphBank:
    """Fixed-length verify dispatcher with safe fallback instrumentation.

    `mx.compile` can capture array trees, but the stock MLX Qwen3.6 cache also
    stores decode offsets as Python integers.  Replaying a compiled closure that
    captured those integers would use stale RoPE/mask positions, so the safe
    backend refuses to compile until explicit tensor cache state lands.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_verify_len: int = 6,
        allow_python_cache_capture: bool = False,
        promote_tensor_offsets: bool = True,
        capture_backend: str | None = None,
        rope_delta: mx.array | int | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_verify_len = max_verify_len
        self.allow_python_cache_capture = allow_python_cache_capture
        self.promote_tensor_offsets = promote_tensor_offsets
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
        # An image request's rotary delta (None for text): stamped on every
        # cache this bank promotes. The closures below capture each cache's
        # ``compile_state``, which carries the delta as an implicit input.
        self._rope_delta = as_rope_delta(rope_delta)
        self.stats = GraphBankStats()
        self._compiled: dict[tuple[str, int, tuple[int, ...]], Any] = {}

    def forward_ar(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        return self._forward(
            "forward",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        return self._forward(
            "capture",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def _forward(
        self,
        kind: str,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        started = time.perf_counter()
        self.stats.calls += 1
        length = _decode_length(input_ids)
        reason = self._fallback_reason(length, cache)
        if reason is not None:
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=reason,
                started=started,
            )

        try:
            key = (kind, length, str(hidden_variant or ""), _cache_container_signature(cache))
            fn = self._compiled.get(key)
            if fn is None:
                if kind == "capture":
                    fn = self._compile_capture_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                    )
                else:
                    fn = self._compile_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                    )
                self._compiled[key] = fn
            with compiled_dispatch((id(fn), tuple(getattr(input_ids, "shape", ())))):
                result = fn(input_ids)
            self.stats.compiled_calls += 1
            self.stats.elapsed_s += time.perf_counter() - started
            return result
        except Exception as exc:  # pragma: no cover - exercised by real MLX cache probes
            key = type(exc).__name__
            self.stats.compile_errors[key] = self.stats.compile_errors.get(key, 0) + 1
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=f"compile_error:{key}",
                started=started,
            )

    def warm(
        self,
        lengths: range | list[int] | tuple[int, ...],
        *,
        cache_factory,
        token_factory,
    ) -> None:
        """Warm eligible shapes using caller-provided disposable cache/tokens."""
        for length in lengths:
            if length < 1 or length > self.max_verify_len:
                continue
            cache = cache_factory()
            tokens = token_factory(length)
            self.forward_ar(tokens, cache=cache, return_hidden=True)
            if length not in self.stats.warmed_lengths:
                self.stats.warmed_lengths.append(length)

    def to_dict(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data["max_verify_len"] = self.max_verify_len
        data["allow_python_cache_capture"] = self.allow_python_cache_capture
        data["promote_tensor_offsets"] = self.promote_tensor_offsets
        data["capture_backend"] = self.capture_backend
        data["compiled_lengths"] = sorted({length for _, length, _, _ in self._compiled})
        data["compiled_paths"] = [
            f"{kind}:{length}"
            for kind, length in sorted({(kind, length) for kind, length, _, _ in self._compiled})
        ]
        data["compiled_entry_count"] = len(self._compiled)
        data["rope_delta"] = (
            None if self._rope_delta is None else int(self._rope_delta.item())
        )
        return data

    def reset(self) -> None:
        """Drop compiled closures after cache container identity changes."""
        self._compiled.clear()

    def _fallback_reason(self, length: int, cache: Any) -> str | None:
        if length < 1:
            return "invalid_length"
        if length > self.max_verify_len:
            return "length_outside_graphbank"
        if cache is None:
            return None
        if self.promote_tensor_offsets and not self.allow_python_cache_capture:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=length)
            self.stats.promoted_cache_entries += promoted
            for reason, count in failures.items():
                self.stats.promotion_failures[reason] = (
                    self.stats.promotion_failures.get(reason, 0) + count
                )
            stamp_rope_delta(cache, self._rope_delta)
        # A compiled replay writes promoted paged adapters at their traced
        # offset with no Python on the path, and MLX does not clamp that
        # write. Reserve the window here, before the dispatch, for every
        # promoted paged adapter in the list, whether this dispatcher promoted
        # it or received it promoted (the dense adapters are topped up by the
        # promotion above).
        from .cache_state import reserve_paged_window

        reserve_paged_window(cache, length)
        if self.allow_python_cache_capture:
            return None
        if cache_has_python_offsets(cache):
            return "python_cache_offsets"
        return None

    def _fallback(
        self,
        kind: str,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        reason: str,
        started: float,
    ):
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1
        if kind == "capture":
            result = self._runtime_forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
        else:
            result = self.runtime.forward_ar(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
        self.stats.elapsed_s += time.perf_counter() - started
        return result

    def _compile_length(
        self,
        length: int,
        *,
        cache: Any,
        return_hidden: bool,
        hidden_variant: str | None,
    ):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            with compiled_step_body():
                return self.runtime.forward_ar(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                )

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _compile_capture_length(
        self,
        length: int,
        *,
        cache: Any,
        return_hidden: bool,
        hidden_variant: str | None,
    ):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            with compiled_step_body():
                return self._runtime_forward_ar_capture(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                )

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _runtime_forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        if self._capture_accepts_backend:
            return self.runtime.forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=self.capture_backend,
            )
        return self.runtime.forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )


def _decode_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    return int(shape[1])


def _cache_container_signature(cache: Any) -> tuple[int, ...]:
    if cache is None:
        return ()
    signature: list[int] = [id(cache)]
    for entry in cache:
        signature.append(id(entry))
        if entry is None:
            continue
        if hasattr(entry, "compile_state"):
            state = getattr(entry, "compile_state")
            if isinstance(state, list):
                signature.extend(id(item) for item in state)
            continue
        if hasattr(entry, "cache"):
            signature.append(id(getattr(entry, "cache")))
            continue
        state = getattr(entry, "state", None)
        if isinstance(state, list):
            signature.append(id(state))
    return tuple(signature)


def _accepts_capture_backend(runtime: Any) -> bool:
    import inspect

    try:
        signature = inspect.signature(runtime.forward_ar_capture)
    except (AttributeError, TypeError, ValueError):
        return False
    return "capture_backend" in signature.parameters


def _accepts_runtime_keyword(runtime: Any, name: str) -> bool:
    import inspect

    try:
        signature = inspect.signature(runtime.forward_ar_capture)
    except (AttributeError, TypeError, ValueError):
        return False
    return name in signature.parameters


def cache_has_python_offsets(cache: Any) -> bool:
    for entry in cache or []:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            return True
        idx = getattr(entry, "_idx", None)
        if isinstance(idx, int):
            return True
    return False


class TensorOffsetKVCache(RotaryOrigin):
    """Full-attention KV cache adapter with array-backed mutable offset.

    Stock `KVCache.offset` is a Python integer.  In a compiled verify graph that
    integer is graph-constant state, so RoPE and mask positions can silently go
    stale.  This adapter keeps the existing key/value buffers, stores the offset
    in `cache[2]`, and mutates the three-array state through operations visible
    to `mx.compile(inputs=..., outputs=...)`.

    The adapter also owns its ROTARY ORIGIN (``RotaryOrigin``): ``rope_delta``
    is None for a text request and the image request's delta otherwise, and
    ``rope_offset`` is where the attention routes rotate the next row. The
    delta shifts rotary positions only; ``offset`` keeps driving the mask,
    the rollback window and every slice write.
    """

    def __init__(
        self,
        keys: mx.array,
        values: mx.array,
        offset: int | mx.array,
        *,
        step: int = 256,
        rope_delta: mx.array | int | None = None,
    ) -> None:
        offset_array = (
            offset
            if isinstance(offset, mx.array)
            else mx.array(offset, dtype=mx.int32)
        )
        self.cache = [keys, values, offset_array]
        self.rollback_state = [None, None, None]
        self.step = step
        self._init_rotary_origin(rope_delta)
        # Growth-budget tracking (2026-07-03): the first promotion grants
        # headroom (`initial_reserve_tokens`); any capacity expansion AFTER
        # that grant means the compiled verify graph would retrace, so the
        # bank demotes the request to eager. Flag-based so the hot path never
        # adds extra offset evals.
        self._granted = False
        self.growth_after_grant = False

    @classmethod
    def from_kv_cache(cls, entry: Any, *, reserve_tokens: int) -> "TensorOffsetKVCache":
        cache = cls(
            entry.keys,
            entry.values,
            entry.offset,
            step=getattr(entry, "step", 256),
        )
        cache.ensure_capacity(int(entry.offset) + reserve_tokens)
        return cache

    @property
    def keys(self):
        return self.cache[0]

    @keys.setter
    def keys(self, value):
        self.cache[0] = value

    @property
    def values(self):
        return self.cache[1]

    @values.setter
    def values(self, value):
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        self.cache[2] = (
            value
            if isinstance(value, mx.array)
            else mx.array(value, dtype=mx.int32)
        )

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    @property
    def compile_state(self):
        # The rotary origin rides next to the leaves so a closure that
        # captures this tree (the graph bank, the device draft cores) reads
        # the delta at call time, never as a constant of the trace.
        return [self.cache, self.rollback_state, self.rope_state]

    def ensure_capacity(self, needed: int) -> None:
        if self.keys is None or self.values is None:
            return
        capacity = int(self.keys.shape[2])
        if needed <= capacity:
            self._granted = True
            return
        if self._granted:
            self.growth_after_grant = True
        new_capacity = ((needed + self.step - 1) // self.step) * self.step
        extra = new_capacity - capacity
        k_shape = (*self.keys.shape[:2], extra, self.keys.shape[3])
        v_shape = (*self.values.shape[:2], extra, self.values.shape[3])
        self.keys = mx.concatenate(
            [self.keys, mx.zeros(k_shape, dtype=self.keys.dtype)],
            axis=2,
        )
        self.values = mx.concatenate(
            [self.values, mx.zeros(v_shape, dtype=self.values.dtype)],
            axis=2,
        )
        self._granted = True

    def _concrete_offset(self):
        """The host-int offset when it is a materialized value, else None.

        During an ``mx.compile`` trace ``cache[2]`` is a tracer and ``.item()``
        raises; on the eager (demoted) verify path it is a concrete scalar.
        """
        try:
            return int(self.cache[2].item())
        except Exception:
            return None

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        off_i = self._concrete_offset()
        if off_i is not None:
            # Eager (demoted) verify: write the S new rows in place. The
            # functional ``mx.slice_update`` below reallocates the WHOLE
            # [1, 2, capacity, 256] buffer per key and value tensor when the
            # compiled graph cannot donate it (267 MB each at 262K, 6.4 GB
            # across the 12 full-attention layers) -- the transient that OOMs a
            # 261,120-token verify while an S=1 AR decode on the stock KVCache,
            # which writes in place, fits. Mirror the stock in-place write on
            # this path. The rollback snapshot is a materialized independent
            # copy of the pre-write rows, so the in-place write cannot corrupt
            # it and a later ``trim`` restores exactly the same bytes.
            end = off_i + steps
            if end > int(self.cache[0].shape[2]):
                # A write past the granted capacity. The functional path
                # below clamps silently (MLX drops the rows that do not fit
                # while the offset still advances past them), which is how
                # a copy block straddling the growth edge lost its KV rows
                # on the eager copy-block route; an in-place assignment
                # refuses the shape instead. Grow first, as the stock
                # KVCache does. ``ensure_capacity`` flips
                # ``growth_after_grant`` on a granted leaf, so the bank
                # demotes the request rather than replaying a compiled graph
                # over the old buffers.
                self.ensure_capacity(end)
            snap_k = mx.contiguous(self.cache[0][:, :, off_i:end, :])
            snap_v = mx.contiguous(self.cache[1][:, :, off_i:end, :])
            mx.eval(snap_k, snap_v)
            self.rollback_state[0] = self.cache[2]
            self.rollback_state[1] = snap_k
            self.rollback_state[2] = snap_v
            self.cache[0][:, :, off_i:end, :] = keys
            self.cache[1][:, :, off_i:end, :] = values
            self.cache[2] = self.cache[2] + steps
            return self.cache[0], self.cache[1]
        # Compiled trace: keep the compile-visible functional path so
        # mx.compile(inputs=..., outputs=...) can donate the buffers in the
        # replayed graph.
        self.rollback_state[0] = self.cache[2]
        self.rollback_state[1] = mx.slice(
            self.cache[0],
            self.cache[2],
            axes=(2,),
            slice_size=keys.shape,
        )
        self.rollback_state[2] = mx.slice(
            self.cache[1],
            self.cache[2],
            axes=(2,),
            slice_size=values.shape,
        )
        self.cache[0] = mx.slice_update(
            self.cache[0],
            keys,
            self.cache[2],
            axes=(2,),
        )
        self.cache[1] = mx.slice_update(
            self.cache[1],
            values,
            self.cache[2],
            axes=(2,),
        )
        self.cache[2] = self.cache[2] + steps
        return self.cache[0], self.cache[1]

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        del return_array
        if self.keys is None:
            return None
        capacity = int(self.keys.shape[2])
        rinds = mx.arange(capacity)
        linds = self.cache[2] + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask

    def size(self):
        value = self.cache[2]
        mx.eval(value)
        return int(value.item())

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = int(n)
        if (
            self.rollback_state[0] is not None
            and self.rollback_state[1] is not None
            and self.rollback_state[2] is not None
            and int(self.rollback_state[1].shape[2]) == n
        ):
            try:
                back_i = int(self.rollback_state[0].item())
            except Exception:
                back_i = None
            if back_i is not None:
                # Eager restore: put the saved rows back in place instead of
                # reallocating the full buffer (same 6.4 GB avoidance as
                # update_and_fetch's eager write).
                self.cache[0][:, :, back_i:back_i + n, :] = self.rollback_state[1]
                self.cache[1][:, :, back_i:back_i + n, :] = self.rollback_state[2]
                self.cache[2] = self.rollback_state[0]
            else:
                self.cache[0] = mx.slice_update(
                    self.cache[0],
                    self.rollback_state[1],
                    self.rollback_state[0],
                    axes=(2,),
                )
                self.cache[1] = mx.slice_update(
                    self.cache[1],
                    self.rollback_state[2],
                    self.rollback_state[0],
                    axes=(2,),
                )
                self.cache[2] = self.rollback_state[0]
        else:
            self.cache[2] = mx.maximum(
                self.cache[2] - n,
                mx.array(0, dtype=self.cache[2].dtype),
            )
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes + self.cache[2].nbytes

    def demote(self):
        """Restore a stock ``KVCache`` from this adapter.

        The stock container receives the adapter's current key/value buffers
        (no copy) and the materialized integer offset, so downstream consumers
        that expect python-int offsets (postcommit, session bank snapshots)
        never see a tensor-offset adapter.
        """
        from mlx_lm.models.cache import KVCache

        entry = KVCache()
        entry.step = self.step
        entry.keys = self.cache[0]
        entry.values = self.cache[1]
        entry.offset = int(self.size()) if self.cache[0] is not None else 0
        return entry


class TensorOffsetQSACache:
    """Fixed-capacity Qwen4 QSA state for compiled target verification.

    QSA owns five graph leaves: attention keys, attention values, the logical
    token offset, raw index keys, and pooled index keys.  The logical pooled
    length is always ``offset // ratio`` and therefore does not need a second
    mutable offset.  All buffers are granted once when the verifier bank is
    constructed; the enabled path only performs fixed-shape slice updates.

    The bank also owns its ROTARY ORIGIN, next to the logical offset. A text
    request rotates row ``i`` of a forward at ``offset + i``. An image request
    rotates it at ``offset + rope_delta + i``: past the last image every
    position of the request's table is the sequence index plus one constant
    (``mtplx/vision/mrope.py``), so the whole table collapses to that delta
    for every row a decode forward can write. ``rope_delta`` shifts rotary
    positions ONLY. ``offset`` keeps driving the masks, the block selection,
    the tail start and every slice write, which are KV indices.

    Because the origin rides the cache, the fixed lane never reads the
    request context (``attention_context.vision_rope_state``): a table or a
    delta read inside a verify trace would be baked into a graph that every
    request of the process replays.
    """

    fixed_capacity = True
    step = 256

    def __init__(
        self,
        kv: TensorOffsetKVCache,
        raw_keys: mx.array,
        pooled: mx.array,
        *,
        compress_ratio: int,
        rows_gather: bool = False,
        rows_gather_kv_m4: Any,
        rows_gather_enabled: bool = False,
        rows_gather_min_context: int = 0,
        fused_rows_gather_kv_m4: bool = False,
        rope_delta: mx.array | int | None = None,
        capacity_bucket: int = 0,
        dense_capacity: int | None = None,
        indexer_budget: int = 0,
    ) -> None:
        self.kv = kv
        self.raw_keys = raw_keys
        self.pooled = pooled
        self.ratio = max(1, int(compress_ratio))
        self.step = int(getattr(kv, "step", 256))
        self.fixed_rows_gather = bool(rows_gather)
        self.rows_gather_kv_m4 = rows_gather_kv_m4
        self.rows_gather_enabled = bool(rows_gather_enabled)
        self.rows_gather_min_context = max(0, int(rows_gather_min_context))
        self.fused_rows_gather_kv_m4 = bool(fused_rows_gather_kv_m4)
        self.rope_delta = rope_delta
        # Rows the capacity rounds up to while the bank is on the rows-gather
        # lane (_bank_capacity); 0 for every bank but the strict fixed-M4 one.
        self.capacity_bucket = max(0, int(capacity_bucket))
        # The unbucketed parent's grant. Dense SDPA dispatch and rounding
        # depend on this width, even when its extra columns are masked out.
        self.dense_capacity = (
            self.capacity if dense_capacity is None else int(dense_capacity)
        )
        self.indexer_budget = int(indexer_budget)
        # "adopted" when from_qsa_cache took the stock buffers as they were,
        # "copied" when it padded or cut them into new arrays; None for a bank
        # built directly (a shadow twin, a test).
        self.promotion: str | None = None

    @property
    def rope_delta(self) -> mx.array | None:
        """One int32 value for an image request, None for a text request."""

        return self._rope_delta

    @rope_delta.setter
    def rope_delta(self, value: mx.array | int | None) -> None:
        self._rope_delta = as_rope_delta(value)

    @property
    def rope_offset(self):
        """Rotary position of the next row this bank writes.

        For a text request this IS ``offset``, the same array object, so the
        text verify trace holds exactly the nodes it held before image
        requests could reach this lane.
        """

        if self._rope_delta is None:
            return self.kv.offset
        return self.kv.offset + self._rope_delta

    @staticmethod
    def _bank_capacity(
        needed: int, ratio: int, kv_step: int, *, rows_gather: bool, bucket: int = 0
    ) -> int:
        """Rows of a fixed bank that holds ``needed`` tokens.

        Always a multiple of the QSA ratio. On the rows-gather lane also a
        multiple of the K/V cache's growth step, so the K and V banks ARE the
        step-rounded buffers and are never cut from them. MLX lets
        ``slice_update`` write in place only when the array's buffer is at
        most 16,384 bytes larger than the array, while its buffer cache hands
        back a recycled buffer up to two pages (32,768 bytes) larger than
        asked for. A bank cut to a multiple of 4 rows from a 256-row buffer
        landed, after its first copy, in exactly such a buffer whenever the
        cut was 17 to 47 rows short of the step: never donatable again, so
        every K and V bank was copied on every verify round (24 x 135 MB at
        128K: 7.8 ms against 0.6 ms for the 24 writes in isolation, 48
        against 65 tok/s served, 2026-09-20). A capacity on the step is a
        whole number of pages, where the worst recycled buffer is one page
        larger and still donates.

        Rows-gather only, on purpose: there each verify row attends over its
        own selected rows and the index scores are per block, so the padded
        tail never enters a value and the result is bit-identical. The dense
        lane reduces over every row of the bank, and its last float32 bit
        already depends on the padded length (capacity 28 and capacity 64
        differ from the stock cache by 2e-8 on the tiny test geometry), so
        its capacity is left exactly as it was.

        ``bucket`` (the strict fixed-M4 bank's, _fixed_m4_capacity_bucket)
        rounds a rows-gather capacity further up to a multiple of that many
        rows, itself rounded up to the step, so a growing session keeps one
        compiled verify trace across turns. The dense lane ignores it, for a
        reason beyond the last bit above: MLX 0.32.2 picks the vector SDPA
        kernel (one pass or two) and its number of key blocks from the key
        length, which here is the capacity
        (mlx/backend/metal/scaled_dot_product_attention.cpp 487-517, 875-877),
        and the block a key reduces in is its index modulo that count.
        Receipt 2026-09-27, one Flash-Next attention layer on synthetic
        weights with MLX_METAL_GPU_ARCH choosing the dispatch class: capacity
        3,524 against 8,192 differed in 53 to 55% of the bfloat16 outputs of
        each verify step under the base/Pro class, 9,024 against 16,384 in 58
        to 59% under the Ultra class, while the rows-gather lane was bit-equal
        in every cell. The compiled-verify form of the rows-gather invariant is
        tests/test_qwen4_fixed_m4_capacity_bucket.py.
        """

        quantum = max(1, int(ratio))
        if rows_gather:
            quantum = math.lcm(quantum, max(1, int(kv_step)))
            bucket = int(bucket)
            if bucket > 0:
                quantum *= (bucket + quantum - 1) // quantum
        return ((max(1, int(needed)) + quantum - 1) // quantum) * quantum

    @staticmethod
    def _fixed_bank(value: mx.array, capacity: int, axis: int) -> mx.array:
        current = int(value.shape[axis])
        if current == capacity:
            return value
        if current > capacity:
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(0, capacity)
            return value[tuple(slices)]
        shape = list(value.shape)
        shape[axis] = capacity - current
        # A broadcast zero is a view of one scalar: the concatenation writes
        # the pad rows without allocating them first as a block of their own
        # (the 2026-10-01 review measured that block on top of every growth).
        zeros = mx.broadcast_to(mx.array(0, dtype=value.dtype), tuple(shape))
        return mx.concatenate([value, zeros], axis=axis)

    @classmethod
    def from_qsa_cache(
        cls, entry: Any, *, reserve_tokens: int, capacity_bucket: int = 0,
        capacity_plan: FixedM4CapacityPlan | None = None,
    ) -> "TensorOffsetQSACache":
        if capacity_plan is not None:
            reserve_tokens = capacity_plan.reserve_tokens
            capacity_bucket = capacity_plan.bucket
        indexer_budget = (
            capacity_plan.indexer_budget if capacity_plan is not None
            else getattr(entry, "indexer_budget", None)
        )
        if indexer_budget is None:
            capacity_bucket = 0
        reserve_tokens = max(1, int(reserve_tokens))
        offset = int(entry.offset)
        ratio = max(1, int(entry.ratio))
        if entry.raw_keys is None or entry.pooled is None:
            raise ValueError("QSA index state is empty")
        if entry.kv.keys is None or entry.kv.values is None:
            raise ValueError("QSA attention state is empty")

        from .models.qwen4_exp import (
            _qsa_gather_enabled,
            _qsa_gather_min_context,
            _qsa_score_tile_rows,
        )

        rows_gather_enabled = _qsa_gather_enabled()
        rows_gather_min_context = _qsa_gather_min_context()
        rows_gather = rows_gather_enabled and offset >= rows_gather_min_context
        # With fewer than block_topk visible blocks, capacity can change both
        # k_eff and the placement of valid blocks among padded top-k slots.
        # Keep the parent's bank for such an early-gather request.
        if indexer_budget is not None and rows_gather and offset < indexer_budget:
            capacity_bucket = 0
        # A tiled-selector override makes M4 itself a dense consumer. Keep
        # its original allocation and graph signature for the whole request.
        if _qsa_score_tile_rows() > 0:
            capacity_bucket = 0

        logical_capacity = offset + reserve_tokens
        kv_step = getattr(entry.kv, "step", 256)
        dense_capacity = cls._bank_capacity(
            logical_capacity, ratio, kv_step,
            rows_gather=rows_gather,
        )
        raw_capacity = cls._bank_capacity(
            logical_capacity,
            ratio,
            kv_step,
            rows_gather=rows_gather,
            bucket=capacity_bucket,
        )
        adopted = cls.adoptable_rows(
            entry, raw_capacity, kv_step=kv_step, rows_gather=rows_gather,
            bucket=capacity_bucket,
        )
        if adopted is not None:
            raw_capacity = adopted
        pooled_capacity = raw_capacity // ratio

        # Allocate the admitted width directly. Growing on the KV step and
        # then padding again to the bucket can keep both copies live.
        kv = TensorOffsetKVCache(
            cls._fixed_bank(entry.kv.keys, raw_capacity, 2),
            cls._fixed_bank(entry.kv.values, raw_capacity, 2),
            offset,
            step=getattr(entry.kv, "step", 256),
        )
        kv._granted = True
        raw = cls._fixed_bank(entry.raw_keys, raw_capacity, 1)
        pooled = cls._fixed_bank(entry.pooled, pooled_capacity, 1)
        rows_gather_kv_m4 = entry.rows_gather_kv_m4
        fused_rows_gather_kv_m4 = _env_enabled("MTPLX_QSA_M4_FUSED_KV_GATHER")
        if fused_rows_gather_kv_m4:
            expected_shape = (1, 2, raw_capacity, 256)
            if not _env_enabled("MTPLX_QWEN4_FIXED_M4_VERIFY"):
                raise RuntimeError(
                    "QSA fused K/V gather requires the fixed-M4 verifier"
                )
            if not rows_gather_enabled or ratio != 4:
                raise RuntimeError(
                    "QSA fused K/V gather requires the fixed rows-gather ratio-4 lane"
                )
            if (
                tuple(kv.keys.shape) != expected_shape
                or tuple(kv.values.shape) != expected_shape
                or kv.keys.dtype != mx.bfloat16
                or kv.values.dtype != mx.bfloat16
            ):
                raise RuntimeError(
                    "QSA fused K/V gather requires BF16 "
                    "[1,2,capacity,256] cache ownership"
                )
            if rows_gather:
                from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                    bind_qwen4_qsa_m4_fused_kv_gather,
                )

                rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                    capacity=raw_capacity
                )

        bank = cls(
            kv,
            raw,
            pooled,
            compress_ratio=ratio,
            rows_gather=rows_gather,
            rows_gather_kv_m4=rows_gather_kv_m4,
            rows_gather_enabled=rows_gather_enabled,
            rows_gather_min_context=rows_gather_min_context,
            fused_rows_gather_kv_m4=fused_rows_gather_kv_m4,
            capacity_bucket=capacity_bucket,
            dense_capacity=dense_capacity,
            indexer_budget=int(indexer_budget or 0),
        )
        bank.promotion = "adopted" if adopted is not None else "copied"
        return bank

    @staticmethod
    def held_rows(entry: Any) -> int | None:
        """Rows a stock QSA cache's attention and raw index buffers all hold.

        None when a buffer is missing or the three row-indexed buffers
        (keys, values, raw index keys) disagree. The pooled buffer is not
        consulted: it is small, and a short suffix that completes no block
        leaves it at its previous size. The admission checks it separately
        (``one_copy.qsa_buffers_short``) and resizes a short one before the
        bank adopts the others.
        """

        kv = getattr(entry, "kv", None)
        keys = getattr(kv, "keys", None)
        values = getattr(kv, "values", None)
        raw = getattr(entry, "raw_keys", None)
        if keys is None or values is None or raw is None:
            return None
        rows = int(keys.shape[2])
        if int(values.shape[2]) != rows or int(raw.shape[1]) != rows:
            return None
        return rows

    @classmethod
    def adoptable_rows(
        cls, entry: Any, planned: int, *, kv_step: int, rows_gather: bool,
        bucket: int,
    ) -> int | None:
        """The capacity a bank built from ``entry`` can take without a copy.

        The bank used to be a second, padded copy of the conversation: the
        stock buffers were concatenated with zeros up to the planned rows
        even when they already held them. A cache sized for this request
        (qwen4_exp.qsa_rows_target during its prefill) or a lease demoted
        with its capacity kept holds its keys, values and raw index keys in
        buffers of exactly the planned rows, and the bank adopts them as its
        own (the verify writes then donate into the very buffers the prefill
        filled).

        A larger buffer is adopted only where the capacity is not part of the
        arithmetic: on the rows-gather lane with the bucket in force (each
        verify row attends over its own selected rows; the padded tail never
        enters a value; see ``_bank_capacity``), page-aligned like every
        bucketed capacity, and at most one bucket larger, so the selector's
        work over the pooled blocks grows by at most one bucket. The dense
        lane and an unbucketed bank keep their exact width. None means the
        stock pad-or-cut construction runs.
        """

        return cls.adoptable_capacity(
            cls.held_rows(entry), planned,
            ratio=max(1, int(getattr(entry, "ratio", 4) or 4)),
            kv_step=kv_step, rows_gather=rows_gather, bucket=bucket,
        )

    @staticmethod
    def adoptable_capacity(
        held: int | None, planned: int, *, ratio: int, kv_step: int,
        rows_gather: bool, bucket: int,
    ) -> int | None:
        """``adoptable_rows`` on row counts alone (the admission's arithmetic)."""

        planned = int(planned)
        if held is None or int(held) < planned:
            return None
        rows = int(held)
        if rows == planned:
            return rows
        if not rows_gather or int(bucket) <= 0:
            return None
        quantum = math.lcm(max(1, int(ratio)), max(1, int(kv_step)))
        if rows % quantum:
            return None
        bucket_rows = quantum * ((int(bucket) + quantum - 1) // quantum)
        if rows - planned > bucket_rows:
            return None
        return rows

    @property
    def capacity(self) -> int:
        return int(self.raw_keys.shape[1])

    def attention_capacity(self, rows: int) -> int:
        """Read the parent's width on every dense consumer, including S=1."""

        from .models.qwen4_exp import _qsa_score_tile_rows

        tile = _qsa_score_tile_rows()
        if rows == 1 or not self.fixed_rows_gather or 0 < tile < rows:
            return self.dense_capacity
        return self.capacity

    def selector_pooled(self, pooled: mx.array, rows: int) -> mx.array:
        blocks = self.attention_capacity(rows) // self.ratio
        return pooled if blocks == pooled.shape[1] else pooled[:, :blocks]

    def attention_kv(self, keys: mx.array, values: mx.array, rows: int):
        capacity = self.attention_capacity(rows)
        if capacity == keys.shape[2]:
            return keys, values
        return keys[:, :, :capacity], values[:, :, :capacity]

    def growth_rows(self, needed: int, *, bucket: int | None = None) -> int:
        """Rows ``ensure_capacity(needed)`` gives this bank; its capacity if it holds them.

        ``bucket`` overrides the bank's own capacity bucket (0: the
        unbucketed width a refused growth checks next).
        """

        if int(needed) <= self.capacity:
            return self.capacity
        return self._bank_capacity(
            needed,
            self.ratio,
            getattr(self.kv, "step", 256),
            rows_gather=self.fixed_rows_gather,
            bucket=self.capacity_bucket if bucket is None else int(bucket),
        )

    def growth_bytes(self, rows: int) -> tuple[int, int]:
        """(new, old) bytes of the banks growing to ``rows`` rows allocates and frees."""

        new = old = 0
        for leaf, axis, target in (
            (self.kv.keys, 2, rows),
            (self.kv.values, 2, rows),
            (self.raw_keys, 1, rows),
            (self.pooled, 1, int(rows) // self.ratio),
        ):
            if int(leaf.shape[axis]) >= int(target):
                continue
            row_bytes = int(leaf.nbytes) // max(1, int(leaf.shape[axis]))
            new += row_bytes * int(target)
            old += int(leaf.nbytes)
        return new, old

    def ensure_capacity(self, needed: int) -> bool:
        """Grow this installed QSA generation without changing its offset.

        The new banks are written beside the old ones and only then swapped
        in, so an allocation that fails leaves this layer as it was, its
        dense width included; the wait after the swap lets the old banks go
        before anything else is allocated (``eval`` returns before the
        command buffer that read them releases them).
        """

        dense_capacity = self.dense_capacity
        if int(needed) > dense_capacity:
            dense_capacity = self._bank_capacity(
                needed, self.ratio, self.kv.step, rows_gather=self.fixed_rows_gather
            )
        if int(needed) <= self.capacity:
            self.dense_capacity = dense_capacity
            return False
        raw_capacity = self.growth_rows(needed)
        pooled_capacity = raw_capacity // self.ratio
        keys = self._fixed_bank(self.kv.keys, raw_capacity, 2)
        values = self._fixed_bank(self.kv.values, raw_capacity, 2)
        raw = self._fixed_bank(self.raw_keys, raw_capacity, 1)
        pooled = self._fixed_bank(self.pooled, pooled_capacity, 1)
        mx.eval(keys, values, raw, pooled)
        self.kv.keys, self.kv.values = keys, values
        self.raw_keys, self.pooled = raw, pooled
        self.dense_capacity = dense_capacity
        mx.synchronize()
        self.kv._granted = True
        self.kv.growth_after_grant = False
        if self.fixed_rows_gather and self.fused_rows_gather_kv_m4:
            from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                bind_qwen4_qsa_m4_fused_kv_gather,
            )

            self.rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                capacity=raw_capacity
            )
        return True

    def activate_rows_gather(self, logical_end: int) -> bool:
        """Install the construction-validated sparse route at its threshold."""

        if (
            self.fixed_rows_gather
            or not self.rows_gather_enabled
            or int(logical_end) < self.rows_gather_min_context
        ):
            return False
        self.fixed_rows_gather = True
        if self.fused_rows_gather_kv_m4:
            from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                bind_qwen4_qsa_m4_fused_kv_gather,
            )

            self.rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                capacity=self.capacity
            )
        return True

    @property
    def offset(self):
        return self.kv.offset

    @property
    def pooled_len(self):
        return self.kv.offset // self.ratio

    @property
    def state_leaves(self) -> list[mx.array]:
        return [*self.kv.cache, self.raw_keys, self.pooled]

    def pooled_f32_view(self, nb: int) -> mx.array:
        """fp32-transposed [1, 1, D, nb] view of the fixed pooled bank.

        The stock QSACache keeps a lockstep mirror for allocation hygiene;
        the fixed bank has static capacity and lives inside one compiled
        trace, so the cast is a single graph node and needs no mirror.
        """
        return mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[:, None][..., :nb]

    def write_raw(self, keys: mx.array) -> None:
        self.raw_keys = mx.slice_update(
            self.raw_keys, keys, self.kv.offset, axes=(1,)
        )

    def write_pooled(self, blocks: mx.array, nb_start, nb_total) -> None:
        del nb_total
        self.pooled = mx.slice_update(
            self.pooled, blocks, nb_start, axes=(1,)
        )

    def size(self) -> int:
        return self.kv.size()

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return self.kv.trim(n)

    @property
    def nbytes(self) -> int:
        return int(self.kv.nbytes + self.raw_keys.nbytes + self.pooled.nbytes)

    def demote(self, *, compact: bool = True, keep_capacity: bool = False):
        """Hand the state back as a stock ``QSACache``.

        The rotary delta does not travel: keys and pooled keys are stored
        already rotated, and the delta is a function of the request's own
        ids and image grids, recomputed by the next request. Nothing of it
        reaches the session bank, RAM or SSD.

        ``keep_capacity`` hands back the bank's own buffers whole (no cut to
        the dense width, no compaction copy): the one-copy conversation store
        keeps them as the conversation's only copy, and the next request's
        prefill writes into them in place and its bank adopts them again.
        Stock readers only ever read the valid rows below the offset, so the
        extra rows change nothing they compute.
        """

        from .models.qwen4_exp import QSACache

        offset = self.kv.size()
        entry = QSACache(self.ratio)
        entry.indexer_budget = self.indexer_budget
        if keep_capacity:
            entry.kv.step = self.kv.step
            entry.kv.keys = self.kv.keys
            entry.kv.values = self.kv.values
            entry.kv.offset = int(offset)
            entry.raw_keys = self.raw_keys
            entry.pooled = self.pooled
            entry.pooled_len = min(int(self.pooled.shape[1]), offset // self.ratio)
            return entry
        entry.kv = self.kv.demote()
        entry.kv.keys, entry.kv.values = self.attention_kv(entry.kv.keys, entry.kv.values, 1)
        entry.raw_keys = self._fixed_bank(self.raw_keys, self.dense_capacity, 1)
        entry.pooled = self.selector_pooled(self.pooled, 1)
        entry.pooled_len = min(int(self.pooled.shape[1]), offset // self.ratio)
        if compact and self.capacity > self.dense_capacity:
            # Prefix views retain the entire bucket, while SessionBank
            # charges the logical state. Publish owned logical arrays and
            # finish their copies here so lazy snapshots retain no bucket.
            state = tuple(
                mx.asarray(leaf, copy=True) if leaf is not None else None
                for leaf in entry.state
            )
            mx.eval(*state)
            entry.state = state
        return entry


def ensure_eager_window_capacity(cache: Any, window_tokens: int) -> int:
    """Grow every ``TensorOffsetKVCache`` entry so one eager forward of
    ``window_tokens`` rows fits; returns how many entries grew.

    The eager copy-block route writes ``1 + block`` rows into these fixed
    buffers outside the bank's own reservation. The growth has to happen
    before the forward: the attention mask is built once per forward from
    the first full-attention layer's capacity, so a buffer that grows inside
    a layer's update no longer matches the mask. Every full-attention layer
    holds the same token count, so the offset is read once.
    """

    grown = 0
    size = None
    for entry in cache or []:
        if not isinstance(entry, TensorOffsetKVCache) or entry.keys is None:
            continue
        if size is None:
            size = entry.size()
        needed = size + max(1, int(window_tokens))
        if needed > int(entry.keys.shape[2]):
            entry.ensure_capacity(needed)
            grown += 1
    return grown

def promote_kv_cache_offsets(
    cache: Any,
    *,
    reserve_tokens: int,
    preserve_paged: bool | None = None,
    initial_reserve_tokens: int | None = None,
    qsa_capacity_bucket: int = 0,
    qsa_capacity_plan: FixedM4CapacityPlan | None = None,
) -> tuple[int, dict[str, int]]:
    """Replace stock full-attention KV caches with tensor-offset adapters.

    ``preserve_paged`` controls what happens to ``VllmMetalPagedKVCache``
    entries.  When true they are promoted in place to
    ``TensorOffsetVllmMetalPagedKVCache`` (keeping the physical page buffers).
    When false the paged entry falls through to the dense promotion path,
    which reads ``entry.keys`` / ``entry.values`` — the ``.keys`` property on
    the paged cache densifies the whole cache, so paged storage is silently
    lost.  The default (``None``) preserves the historical behavior of the
    ``MTPLX_GRAPHBANK_PRESERVE_PAGED_KV`` env switch; callers that must never
    densify paged KV (e.g. ``CompiledVerifyBank``) pass ``True`` explicitly.

    ``qsa_capacity_bucket`` is the strict fixed-M4 bank's capacity bucket for
    the QSA banks it promotes (``TensorOffsetQSACache._bank_capacity``); every
    other caller leaves it 0.
    """
    promoted = 0
    failures: dict[str, int] = {}
    if cache is None:
        return promoted, failures
    if preserve_paged is None:
        preserve_paged = _env_enabled("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV")
    for idx, entry in enumerate(cache):
        if entry is None:
            continue
        if isinstance(entry, TensorOffsetQSACache):
            continue
        if isinstance(entry, TensorOffsetKVCache):
            entry.ensure_capacity(entry.size() + reserve_tokens)
            continue
        try:
            from .models.qwen4_exp import QSACache
        except Exception:  # pragma: no cover - optional model import
            QSACache = None
        if QSACache is not None and isinstance(entry, QSACache):
            try:
                cache[idx] = TensorOffsetQSACache.from_qsa_cache(
                    entry,
                    reserve_tokens=(
                        initial_reserve_tokens
                        if initial_reserve_tokens is not None
                        else reserve_tokens
                    ),
                    capacity_bucket=qsa_capacity_bucket,
                    capacity_plan=qsa_capacity_plan,
                )
            except (TypeError, ValueError):
                failures["auxiliary_qsa_state"] = (
                    failures.get("auxiliary_qsa_state", 0) + 1
                )
                continue
            promoted += 1
            continue
        if preserve_paged:
            try:
                from .cache_state import (
                    TensorOffsetQuantizedPagedKVCache,
                    TensorOffsetVllmMetalPagedKVCache,
                    VllmMetalPagedKVCache,
                )
            except Exception:  # pragma: no cover - import guard for minimal test envs
                TensorOffsetQuantizedPagedKVCache = None
                TensorOffsetVllmMetalPagedKVCache = None
                VllmMetalPagedKVCache = None
            if (
                VllmMetalPagedKVCache is not None
                and isinstance(entry, VllmMetalPagedKVCache)
            ):
                if entry.key_cache is None or entry.value_cache is None:
                    failures["empty_paged_kv_cache"] = (
                        failures.get("empty_paged_kv_cache", 0) + 1
                    )
                    continue
                if getattr(entry, "turboquant", False):
                    # TurboQuant pages depend on the external vLLM-Metal ops;
                    # no adapter understands them. Keep the eager refusal.
                    failures["quantized_paged_kv_cache"] = (
                        failures.get("quantized_paged_kv_cache", 0) + 1
                    )
                    continue
                if getattr(entry, "kv_quant", False):
                    # kv_quant pages promote to the quantized adapter
                    # (head-major banks + fp32 scale planes, stable leaf
                    # shapes/dtypes for the compiled graph). Fail-closed:
                    # geometry the packed-quant kernel refuses, or the env
                    # kill-switch, keeps the historical eager refusal.
                    if not quantized_paged_bank_enabled():
                        failures["quantized_paged_kv_cache"] = (
                            failures.get("quantized_paged_kv_cache", 0) + 1
                        )
                        continue
                    if (
                        TensorOffsetQuantizedPagedKVCache is None
                        or not TensorOffsetQuantizedPagedKVCache.promotable(entry)
                    ):
                        failures["quantized_paged_kv_geometry"] = (
                            failures.get("quantized_paged_kv_geometry", 0) + 1
                        )
                        continue
                    cache[idx] = TensorOffsetQuantizedPagedKVCache.from_paged_cache(
                        entry
                    )
                    promoted += 1
                    continue
                cache[idx] = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(entry)
                promoted += 1
                continue
        from .segmented_kv import SegmentedKVCache

        if isinstance(entry, SegmentedKVCache):
            # A segmented KV cache (MTPLX_SEGMENTED_KV) has no single buffer for the
            # tensor-offset adapter to adopt; the verify runs eager on it.
            failures["segmented_kv_cache"] = failures.get("segmented_kv_cache", 0) + 1
            continue
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int):
            continue
        if getattr(entry, "_idx", None) is not None:
            failures["rotating_or_indexed_cache"] = (
                failures.get("rotating_or_indexed_cache", 0) + 1
            )
            continue
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            failures["empty_kv_cache"] = failures.get("empty_kv_cache", 0) + 1
            continue
        if (
            len(getattr(keys, "shape", ())) != 4
            or len(getattr(values, "shape", ())) != 4
        ):
            failures["unsupported_kv_shape"] = failures.get("unsupported_kv_shape", 0) + 1
            continue
        cache[idx] = TensorOffsetKVCache.from_kv_cache(
            entry,
            # First promotion may grant extra growth headroom so the compiled
            # verify graph keeps a stable leaf shape for the whole span of a
            # typical agent round; steady-state re-promotion calls above only
            # top up by `reserve_tokens` (the verify length).
            reserve_tokens=(
                initial_reserve_tokens
                if initial_reserve_tokens is not None
                else reserve_tokens
            ),
        )
        promoted += 1
    from .cache_state import link_paged_window_group

    link_paged_window_group(cache)
    return promoted, failures


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def quantized_paged_bank_enabled() -> bool:
    """Whether KV-quantized pages keep the compiled verify bank (default on).

    MTPLX_GRAPHBANK_QUANTIZED_PAGED=0 restores the old eager refusal. One
    reader for the promotion and for the health payload, so the two never
    disagree about a spelling.
    """
    return _env_enabled("MTPLX_GRAPHBANK_QUANTIZED_PAGED", default=True)


def cache_array_tree(cache: Any) -> list[Any]:
    """Return the arrays a compiled closure can legally capture."""
    tree: list[Any] = []
    for entry in cache or []:
        if entry is None:
            tree.append(None)
            continue
        if hasattr(entry, "compile_state"):
            tree.append(getattr(entry, "compile_state"))
            continue
        if hasattr(entry, "cache"):
            tree.append(getattr(entry, "cache"))
            continue
        leaves = []
        for name in ("keys", "values", "left_padding", "lengths", "_lengths"):
            if hasattr(entry, name):
                leaves.append(getattr(entry, name))
        if not leaves and hasattr(entry, "state"):
            leaves.append(entry.state)
        tree.append(leaves)
    return tree


# ---------------------------------------------------------------------------
# W2 compiled verify: pure-function verify step over a shadow cache.
#
# The June-12 poisoning failure compiled the side-effecting forward directly:
# tracer arrays were assigned into the *real* ArraysCache/paged cache lists and
# python offsets were baked into the trace as constants, so the next trace died
# with "eval an array without a primitive".  The firewall here is a persistent
# shadow cache owned by the bank: the compiled function re-seeds every shadow
# leaf from its explicit inputs BEFORE any read, runs the existing runtime
# forward against the shadow containers, and returns every leaf as an explicit
# output.  Tracers therefore never escape into the real cache; the dispatch
# wrapper mirror-commits materialized outputs into the real entries.
# ---------------------------------------------------------------------------

VERIFY_SPEC_KIND_FULL_ATTN = "fa"
VERIFY_SPEC_KIND_GDN = "gdn"
VERIFY_SPEC_KIND_QSA = "qsa"

# Fixed-M4 replay install receipts printed so far in this process (the first
# few installs announce their auxiliary route; the request report carries it
# for every request).
_FIXED_M4_INSTALL_RECEIPTS = 0

TAPE_CAPTURE_KEYS = ("conv_states", "conv_out", "g", "state_in", "tape")
STANDARD_CAPTURE_KEYS = ("conv_states", "states")
_UNSUPPORTED_CAPTURE_BACKENDS = {
    "linear_gdn_final",  # emits {"final_only": True}; nothing to flatten
    "linear_gdn_from_conv_stream_skip0",  # capture_start-shifted layout
}


# Prewarm one-shot (F6, 2026-08-16). The shader/pipeline cache the ladder
# primes is process-global (and OS-persistent), so re-walking buckets that
# are already warm is pure waste — but the OLD one-shot boolean was spent by
# the FIRST compiled dispatch of the process, which is normally the 16-token
# boot warmup: its tiny cache clamped the walk (min paged capacity) and the
# deeper buckets then paid their ~1s compile inside the first MEASURED
# benchmark row. `_PREWARM_DONE` now means "no future walk can add
# coverage" (walk reached the router ceiling, or the cache is structurally
# ladder-free); until then, the first dispatch of each generation retries
# the walk and extends it with whatever new buckets the current cache
# capacity allows, skipping buckets already recorded in
# `_PREWARMED_BUCKETS`. A retry with nothing new to walk is a few python
# comparisons — no compiles, no kernel work.
_PREWARM_DONE = False

# Buckets already walked this process, keyed
# (runtime id, verify length, hidden variant, bucket). A recycled runtime
# id after a model swap can only SKIP a warmup walk (perf miss, never a
# correctness risk — the compiled callables themselves are guarded by the
# weakref check in _shared_or_new_verify_step).
_PREWARMED_BUCKETS: set[tuple[int, int, str, int]] = set()

# Importable prewarm truth for /health (read defensively via getattr).
# "done": no further walk can add coverage; "buckets": bucket sizes warmed
# this process; "walks": ladder walks that executed; "last_report": the most
# recent walk report (same shape as CompiledVerifyBank.stats["prewarm"]).
prewarm_status: dict[str, Any] = {
    "done": False,
    "buckets": [],
    "walks": 0,
    "last_report": None,
}

# Importable compiled-verify degradation truth for /health (F23a).
# "permanent_eager" tracks the most recently constructed bank (flipped True
# by any later runtime flip); "reason"/"flipped_at" keep the LAST flip
# forensics (sticky across requests); "flip_count" counts permanent flips
# process-wide (construction-gate flips count once per distinct reason, not
# once per request); "transient_exception_count" counts per-call exception
# fallbacks that did NOT flip the bank.
compiled_verify_status: dict[str, Any] = {
    "mode": None,
    "permanent_eager": False,
    "reason": None,
    "flipped_at": None,
    "flip_count": 0,
    "transient_exception_count": 0,
}

_PERMANENT_EAGER_LOGGED: set[str] = set()


def _record_permanent_eager(
    reason: str, *, once: bool = False, why: str | None = None
) -> None:
    """Record (and log once per distinct reason) a permanent-eager flip.

    ``once=True`` marks deterministic construction-time flips (per-model
    quant gate): the first bank records and logs; subsequent per-request
    banks only re-assert ``permanent_eager`` without inflating the count.
    ``why`` is appended to the log line for a reason code that needs one.
    """
    already_logged = reason in _PERMANENT_EAGER_LOGGED
    compiled_verify_status["permanent_eager"] = True
    if once and already_logged:
        return
    compiled_verify_status["reason"] = reason
    compiled_verify_status["flipped_at"] = time.time()
    compiled_verify_status["flip_count"] = (
        int(compiled_verify_status.get("flip_count", 0)) + 1
    )
    if not already_logged:
        _PERMANENT_EAGER_LOGGED.add(reason)
        try:
            print(
                "[mtplx] compiled-verify permanent-eager: "
                + reason
                + (f"; {why}" if why else "")
                + " (verify runs the eager path from here)",
                flush=True,
            )
        except Exception:
            pass

# Process-global compiled verify callables, keyed by
# (runtime id, capture backend, state spec, verify length, hidden variant,
# bucket). The bank is per-generation; without sharing, every request pays a
# fresh trace. Values are (compiled_fn, trace_host, runtime_ref), where the
# host's WEAK bank reference is re-pointed before each dispatch so retraces
# (mx.compile re-traces on leaf-shape changes) always use live scratch
# containers, and the host counts the traces the program holds. See
# CompiledVerifyBank._shared_or_new_verify_step.
_SHARED_VERIFY_STEPS: dict[tuple, tuple[Any, dict[str, Any], Any]] = {}

# How many traces one shared program may hold before the next lookup replaces
# it. MLX keeps every trace of a compiled function (its inputs, outputs and
# the whole traced tape, one entry per input shape signature) until the
# function object is destroyed, and a verify graph's state shapes follow the
# request: the fixed-M4 capacity is prompt + reserve, and a dense bank's
# capacity is offset + reserve. An agent session therefore adds one trace per
# turn, 8.6 MiB each on the 48-layer Flash-Next graph (a tiny-width model of
# the same depth, measured), and a program shared for the life of the process
# grew host memory without bound (#546). 16 traces keep a recurring shape
# (a retry, a benchmark ladder, several sessions at once) warm and bound one
# program at about 140 MiB on that graph. Replacing a program erases its
# traces from MLX's compile cache, but MLX 0.32.2 does not return all of a
# trace's host memory then: about 60% of its graph objects stay allocated
# (about 5 MiB of the 8.6 MiB on that graph), outside this process's Python
# objects and compile caches, so the bound slows the growth rather than
# ending it.
_SHARED_VERIFY_TRACES_PER_PROGRAM = 16


def _shared_verify_traces_per_program() -> int:
    """Traces a shared verify program may hold before it is replaced.

    ``MTPLX_COMPILED_VERIFY_TRACES_PER_PROGRAM`` overrides the default for
    workloads that cycle through more distinct shapes than it keeps.
    """

    raw = str(
        os.environ.get("MTPLX_COMPILED_VERIFY_TRACES_PER_PROGRAM", "")
    ).strip()
    if not raw:
        return _SHARED_VERIFY_TRACES_PER_PROGRAM
    try:
        return max(1, int(raw))
    except ValueError:
        return _SHARED_VERIFY_TRACES_PER_PROGRAM


def _prewarm_enabled() -> bool:
    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_PREWARM", "1")).strip().lower()
    return raw not in {"0", "false", "off", ""}


def compiled_verify_mode() -> str:
    """Resolve MTPLX_COMPILED_VERIFY into 'off' | 'on' | 'parity' | 'parity2'.

    ``parity``  — double-run with the eager leg authoritative; abort on the
                  first mismatch (Gate A: per-call bit-exactness).
    ``parity2`` — double-run with the COMPILED leg authoritative and an eager
                  clone tracking it; log mismatches, never abort (Gate B:
                  does compiled-committed state evolution diverge?).
    """
    raw = (os.environ.get("MTPLX_COMPILED_VERIFY") or "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"parity", "parity2"}:
        return raw
    return "on"


def _next_pow2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _owned_state_env_active(name: str) -> bool:
    """True when an owned-state wrapper env is set to any enabling value.

    These envs carry mode names (e.g. ``persistent_eval``) rather than plain
    booleans, so anything other than empty/off counts as active.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def build_verify_state_spec(cache: Any) -> tuple[list[tuple[int, str, int]] | None, str | None]:
    """Ordered (layer_idx, kind, n_leaves) spec over the cache list.

    Full-attention tensor-offset entries contribute every slot of their
    ``cache`` list (three for plain KV/paged adapters, five for the
    quantized paged adapter — payloads, offset, scale planes); GDN
    ``ArraysCache`` entries contribute their two slots.  ``None`` entries
    contribute nothing.  Any other container makes the cache non-compilable
    and returns ``(None, reason)``.
    """
    try:
        from mlx_lm.models.cache import ArraysCache
    except Exception:  # pragma: no cover - mlx_lm always present in product envs
        ArraysCache = None
    try:
        from .cache_state import TensorOffsetVllmMetalPagedKVCache
    except Exception:  # pragma: no cover - import guard for minimal test envs
        TensorOffsetVllmMetalPagedKVCache = None

    spec: list[tuple[int, str, int]] = []
    for idx, entry in enumerate(cache or []):
        if entry is None:
            continue
        if isinstance(entry, TensorOffsetQSACache):
            spec.append((idx, VERIFY_SPEC_KIND_QSA, 5))
            continue
        if isinstance(entry, TensorOffsetKVCache) or (
            TensorOffsetVllmMetalPagedKVCache is not None
            and isinstance(entry, TensorOffsetVllmMetalPagedKVCache)
        ):
            spec.append((idx, VERIFY_SPEC_KIND_FULL_ATTN, len(entry.cache)))
            continue
        if ArraysCache is not None and isinstance(entry, ArraysCache):
            if len(entry.cache) not in (2, 4):
                return None, f"unsupported_container:ArraysCache[{len(entry.cache)}]"
            n_leaves = len(entry.cache)
            if n_leaves == 4 and any(leaf is None for leaf in entry.cache):
                return None, "unsupported_container:ArraysCache[partial_ple]"
            spec.append((idx, VERIFY_SPEC_KIND_GDN, n_leaves))
            continue
        return None, f"unsupported_container:{type(entry).__name__}"
    return spec, None


def _paged_kernel_bucket_eligible(entry: Any, length: int, bucket: int) -> bool:
    """Best-effort eager mirror of the compiled paged-attention kernel gates.

    Plain adapters mirror ``sdpa_2pass_paged_tail_dynamic_offset``; the
    quantized adapter mirrors ``sdpa_gqa_packed_tail_quant``. A miss here is
    a performance decision, not a correctness one: inside the compiled
    function the kernel declining simply routes to the pure dense
    ``cache.state`` math, which stays trace-safe.
    """
    key_cache = entry.cache[0]
    value_cache = entry.cache[1]
    if key_cache is None or value_cache is None:
        return False
    if not mx.metal.is_available():
        return False
    try:
        from .cache_state import TensorOffsetQuantizedPagedKVCache
    except Exception:  # pragma: no cover - import guard for minimal test envs
        TensorOffsetQuantizedPagedKVCache = None
    if TensorOffsetQuantizedPagedKVCache is not None and isinstance(
        entry, TensorOffsetQuantizedPagedKVCache
    ):
        # Packed-quant kernel gates (head-major banks): two query banks cap
        # the verify window at 8 rows; payload dtype must match the bits;
        # head dims come from the adapter's own metadata. GQA legality
        # (32 * factor <= 1024) needs the query head count and stays a
        # per-call kernel gate inside the graph.
        if length > 8:
            return False
        bits = int(entry.kv_bits)
        expect_kv = mx.int8 if bits == 8 else mx.uint8
        if key_cache.dtype != expect_kv or value_cache.dtype != expect_kv:
            return False
        head_dim = int(entry.head_dims[0])
        if head_dim not in (64, 128, 256) or int(entry.head_dims[1]) != head_dim:
            return False
        from .kernels.sdpa_gqa_packed_quant import _static_blocks

        blocks = _static_blocks(int(entry.capacity), int(bucket) or None)
        return blocks > 0 and blocks % 32 == 0
    if key_cache.dtype not in (mx.bfloat16, mx.float16):
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    if int(entry.block_size) != int(key_cache.shape[1]):
        return False
    head_dim = int(key_cache.shape[3])
    if head_dim != int(value_cache.shape[3]) or head_dim not in {64, 96, 128, 256}:
        return False
    max_q = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16")
    if length > max_q:
        return False
    from .kernels.sdpa_2pass import _compute_blocks

    blocks = _compute_blocks(max(1, int(length)), int(bucket))
    return blocks > 0 and blocks % 32 == 0


def _as_numpy(value: Any):
    import numpy as np

    try:
        import mlx.core as mx

        if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
            # numpy has no bf16 buffer support; widening to float32 is exact
            # (every bf16 maps to a unique float32), so bit-equality on the
            # widened arrays is bit-equality on the originals.
            return np.asarray(value.astype(mx.float32))
    except Exception:
        pass
    return np.asarray(value)


def _copy_state_leaf(leaf: Any) -> Any:
    """Materialized copy of a cache state leaf.

    ``mx.array(existing)`` allocates a fresh buffer (dtype-preserving, immune
    to donation of the source), which is what lets the parity2 eager clone
    replay a verify step without sharing a single buffer with the live
    compiled-authoritative stream.
    """
    if isinstance(leaf, mx.array):
        return mx.array(leaf)
    return leaf


def _artifact_kind(name: str) -> str:
    """Map a compare_verify_outputs leaf name to its artifact family."""
    if name == "logits":
        return "logits"
    if name == "hidden":
        return "hidden"
    if name.startswith("capture["):
        return "capture"
    if name.startswith("state["):
        return "state"
    return "other"


def _leaf_max_abs_diff(reference: Any, candidate: Any) -> float | None:
    """Max-abs difference between two leaves, or None when incomparable."""
    import numpy as np

    if reference is None or candidate is None:
        return None
    if not hasattr(reference, "shape") or not hasattr(candidate, "shape"):
        return None
    ref_np = _as_numpy(reference)
    cand_np = _as_numpy(candidate)
    if ref_np.shape != cand_np.shape:
        return None
    try:
        diff = np.asarray(ref_np, dtype=np.float64) - np.asarray(
            cand_np, dtype=np.float64
        )
    except (TypeError, ValueError):
        return None
    if not diff.size:
        return 0.0
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(diff)))


def _compiled_verify_max_context() -> int:
    """Context ceiling for the compiled verify step (tokens). Beyond it the
    bank falls back to eager for that call. Default 6144 = the highest
    context Gate A has proven bit-exact AND the ABBA showed +4.8%; past it
    the 2026-07-02 long-form pair measured -28% with a seed-0 trajectory
    fork (boundary materialization scales with context; bucket-crossing
    numerics untested). 0 disables the ceiling (experiments only)."""
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "6144")).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 6144
    return max(0, value)


_FIXED_M4_DONATION_PROBE_RAW = str(
    __import__("os").environ.get("MTPLX_FIXED_M4_DONATION_PROBE", "")
).strip().lower()
_FIXED_M4_DONATION_PROBE = _FIXED_M4_DONATION_PROBE_RAW in (
    "1", "true", "yes", "on", "all"
)
_FIXED_M4_DONATION_PROBE_ALL = _FIXED_M4_DONATION_PROBE_RAW == "all"


def _compiled_verify_boundary() -> str:
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")).strip().lower()
    return raw if raw in ("both", "pre", "post", "none") else "both"


def _compiled_verify_donation_enabled() -> bool:
    """A2.1 commit-first ownership handoff (speed-war Lane A2, 2026-07-06).

    Donation of a KV buffer into its in-graph ``slice_update`` requires the
    graph to hold the ONLY reference when the graph is scheduled.  The
    historical dispatch order (async_eval outputs -> mirror-commit) kept the
    real cache entries and the ``state_in`` list alive at schedule time, so
    every compiled verify call materialized a full copy of every full-attn
    K and V buffer: measured 16.5 ms at 64k / ~33 ms at 128k per call
    (compiled_copy_tax_probe.py arms A vs G, 2026-07-06).  Committing the
    output leaves into the real cache FIRST and dropping the dispatcher
    reference before ``async_eval`` unblocks donation with byte-identical
    results (chained-pending + snapshot-COW proof:
    compiled_copy_tax_correctness.py).  Default ON; env kill-switch for
    A/B and emergency revert.
    """
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_DONATION", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _batch_paged_offsets_enabled() -> bool:
    """Batch-materialize paged-KV offsets before the bucket walk (#318 port).

    ``TensorOffsetVllmMetalPagedKVCache.size()`` does ``mx.eval(cache[2])``
    per entry, so after a trim/rollback (offsets left lazy) the bucket walk
    forces one serial host sync per full-attention entry.  Evaluating every
    offset in one ``mx.eval`` first turns N syncs into one; ``mx.eval``
    cannot change values, so the result is exact by construction.  Neutral
    on non-trimming workloads (offsets already materialized).  Ported from
    grzracz PR #318 with the env read hoisted out of the hot call.  Default
    ON since the night-20260822 round-4 ruling (n=4 counterbalanced ABBA
    blend +2.7% mean, byte-identity held greedy+sampled); "0" opts out.
    """
    import os

    raw = str(os.environ.get("MTPLX_BATCH_PAGED_OFFSETS", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


_BATCH_PAGED_OFFSETS = _batch_paged_offsets_enabled()

# Long-context fence for the #318 default (night-20260822 quad: the trio
# stack measured −2.9%/−2.7% at 16k/32k while short/mid rungs blend
# +2.5..+9.8). generation's per-request prebind sets this from the shared
# MTPLX_GREEDY_TRIO_MAX_CONTEXT fence; requests that never prebind (batch
# lane) keep the last-set/default value — that lane pays at most the
# pre-#318 serial-sync behavior, never a correctness change.
from contextvars import ContextVar

_PAGED_OFFSETS_CONTEXT_OK: ContextVar[bool] = ContextVar(
    "mtplx_paged_offsets_context_ok", default=True
)


def set_paged_offsets_context_ok(allowed: bool):
    """Per-request fence stamp from generation's trio prebind."""
    return _PAGED_OFFSETS_CONTEXT_OK.set(bool(allowed))


def paged_offsets_context_ok() -> bool:
    """Read the current request's fence stamp (receipts/trace)."""
    return _PAGED_OFFSETS_CONTEXT_OK.get()


def materialize_paged_offsets(entries: Any) -> None:
    """Evaluate every paged entry's offset in one ``mx.eval`` (#318).

    Only under the batching switch and the per-request long-context fence
    above; otherwise the offsets stay as they are and each reader's
    ``size()``/``.item()`` syncs its own, the pre-#318 behaviour. ``mx.eval``
    cannot change a value, so either way the offsets read the same. Inside an
    ``mx.compile`` trace the eval raises MLX's trace refusal, which the
    reservation path reads as "traced".
    """

    if not (_BATCH_PAGED_OFFSETS and _PAGED_OFFSETS_CONTEXT_OK.get()):
        return
    offsets = []
    for entry in entries or []:
        entry_state = getattr(entry, "cache", None)
        if isinstance(entry_state, (list, tuple)) and len(entry_state) > 2:
            entry_offset = entry_state[2]
            if isinstance(entry_offset, mx.array):
                offsets.append(entry_offset)
    if offsets:
        mx.eval(*offsets)


def _ccopy_bank_max_len() -> int:
    """Ceiling for extended-window (context-copy block) compiled dispatch.

    Copy blocks are proposed at their native ladder lengths (block 8-32 ->
    T=9-33); the bank verifies them one-shot so the trajectory is byte-equal
    to the eager copy lane (v1's cap-to-bank-window changed the proposal and
    was falsified as a net win, MEASUREMENTS 2026-08-25 12:05). Default 33
    covers the full default ladder; longer custom MTPLX_CONTEXT_COPY_K
    proposals fall back eager per call.
    """
    raw = os.environ.get("MTPLX_CCOPY_BANK_MAX_LEN", "").strip()
    try:
        return max(1, int(raw)) if raw else 33
    except ValueError:
        return 33


def _compiled_verify_growth_reserve() -> int:
    """Dense-leaf growth headroom granted at first promotion (tokens).

    Sized so a typical agent tool round (40-500 generated tokens) completes
    inside one stable leaf shape: one trace per (length, capacity) class,
    zero mid-round retraces. Long generations exceed the grant and demote to
    eager for the request remainder.
    """

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "512")).strip()
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 512


def _fixed_m4_initial_growth_reserve() -> int:
    """Construction-time reserve for the strict Qwen4 fixed-M4 lane."""

    if "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE" in os.environ:
        return _compiled_verify_growth_reserve()
    return 1024


#: Rows a strict fixed-M4 rows-gather bank's capacity is rounded up to
#: (MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET); see _fixed_m4_capacity_bucket.
_FIXED_M4_CAPACITY_BUCKET = 8192


def _fixed_m4_capacity_bucket() -> int:
    """Rows a strict fixed-M4 bank on the rows-gather lane is rounded up to.

    The compiled verify step keeps one trace per input-shape signature for as
    long as it lives, and the bank capacity is part of that signature. Sized
    prompt + reserve on the 256-row step, every turn of a growing agent
    session arrived with a capacity the process had not traced and paid a
    fresh trace (and a fresh fused K/V gather kernel). Rounded up to a bucket
    edge, consecutive turns replay one trace until the session crosses the
    edge. The price is the extra rows per QSA layer up to the edge, which the
    lane's memory gate prices (generation._qwen4_fixed_m4_lane_fits).

    Rows-gather only; see ``TensorOffsetQSACache._bank_capacity`` for why the
    dense lane keeps its exact capacity. 0 turns the bucket off (the 256-row
    step alone); a value that is not a multiple of the step is rounded up to
    one. 8,192 rows by default, at most 7,936 rows over the 256-row step
    (215 MiB at the Flash-Next geometry's 28,416 bytes a row). Replaying the
    capacities of 1,202 logged requests through the rule (2026-09-27) left
    0.03 capacities per request the process had not seen between 16K and
    64K, where the 256-row step left 0.61, and 0.11 above 64K, where it left
    1.20.
    """

    from .models.qwen4_exp import _qsa_score_tile_rows

    if _qsa_score_tile_rows() > 0:
        return 0
    raw = str(os.environ.get("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "")).strip()
    if not raw:
        return _FIXED_M4_CAPACITY_BUCKET
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _FIXED_M4_CAPACITY_BUCKET


@dataclass
class FixedM4CapacityPlan:
    """One request's reserve and bucket, shared by admission and allocation.

    Admission may remove the bucket when only the parent's allocation fits.
    It never changes the reserve or the parent's dense reduction width.
    ``resize_rows`` is the bank's rows when the admission priced resizing the
    conversation's held buffers to them before the bank adopts them
    (one_copy.resize_qsa_buffers); 0 when they are adopted or copied as they
    are. ``admit_growth(need)`` says whether ``need`` more bytes fit under the
    lane's line now: an installed bank asks it for its growth's bill before
    any leaf changes, then for each layer's new banks before that layer is
    written (``CompiledVerifyBank.reserve_fixed_m4_window``).
    """

    reserve_tokens: int
    bucket: int
    indexer_budget: int = 2048
    admit_growth: Callable[[int], bool] | None = field(default=None, repr=False, compare=False)
    resize_rows: int = 0

    @classmethod
    def for_request(cls, max_tokens: int | None, *, width: int = 4, runtime: Any = None):
        reserve = _fixed_m4_initial_growth_reserve()
        if max_tokens is not None:
            reserve = min(max(0, int(max_tokens)) + width, max(reserve, width))
        model = getattr(runtime, "model", None)
        text = getattr(model, "language_model", model)
        args = getattr(text, "args", None) or getattr(getattr(text, "model", None), "args", None)
        ratio = max(1, int(getattr(args, "indexer_compress_ratio", 4) or 4))
        budget = int(getattr(args, "indexer_budget", 2048) or 2048)
        return cls(max(width, reserve), _fixed_m4_capacity_bucket(), (budget // ratio) * ratio)

    def rows(self, offset: int, ratio: int, kv_step: int = 256) -> int:
        from .models.qwen4_exp import _qsa_gather_enabled, _qsa_gather_min_context

        rows_gather = _qsa_gather_enabled() and offset >= _qsa_gather_min_context()
        if rows_gather and offset < self.indexer_budget:
            self.bucket = 0
        return TensorOffsetQSACache._bank_capacity(
            offset + self.reserve_tokens, ratio, kv_step,
            rows_gather=rows_gather, bucket=self.bucket,
        )


_FIXED_M4_MAX_GROWTH_TOKENS = 16384


def _next_fixed_m4_growth_tokens(current: int) -> int:
    """Next construction-owned grant for an overrun fixed-M4 generation."""

    current = max(1, int(current))
    return min(
        current * 2,
        max(current, _FIXED_M4_MAX_GROWTH_TOKENS),
    )


def _fixed_m4_capacity_growth(
    *,
    capacity: int,
    required_end: int,
    growth_tokens: int,
    capacity_limit: int | None,
) -> tuple[int, int]:
    """Resolve one host-boundary capacity transition and its next grant."""

    next_capacity = max(
        int(required_end),
        int(capacity) + max(1, int(growth_tokens)),
    )
    if capacity_limit is not None:
        next_capacity = max(
            int(required_end),
            min(next_capacity, int(capacity_limit)),
        )
    return next_capacity, _next_fixed_m4_growth_tokens(growth_tokens)


class FixedM4GrowthRefused(MemoryError):
    """An installed fixed-M4 bank could not grow for the next write.

    Raised by ``reserve_fixed_m4_window`` when the memory admission refuses
    the growth's bill at both the bucketed and the unbucketed width (no
    cache leaf has changed), or refuses a layer partway, or the memory
    refuses a layer's allocation while it is written: each layer is then at
    its old capacity or its new one, whole. The admission is the session
    bank's too (generation._qwen4_fixed_m4_layer_fits). The committed state
    is whole either way: a caller between rounds can end the answer there
    instead of failing it.
    """

    def __init__(
        self, *, capacity: int, required_end: int, rows: int,
        bill_bytes: int | None = None, layers_left: int | None = None,
        allocation_error: str | None = None,
    ) -> None:
        super().__init__("fixed-M4 bank growth exceeds memory admission")
        self.receipt = {
            "reason": "fixed_m4_growth_refused",
            "capacity_tokens": int(capacity),
            "required_tokens": int(required_end),
            "requested_rows": int(rows),
        }
        if bill_bytes is not None:
            self.receipt["bill_bytes"] = int(bill_bytes)
        if layers_left is not None:
            self.receipt["layers_left"] = int(layers_left)
        if allocation_error is not None:
            self.receipt["allocation_error"] = str(allocation_error)


def _fixed_m4_growing(
    entries: Any, needed: int, *, bucket: int | None = None,
) -> list[tuple[int, TensorOffsetQSACache]]:
    """(rows, bank) for each installed QSA bank that must grow to hold ``needed`` tokens."""

    growing = []
    for entry in entries:
        rows = entry.growth_rows(needed, bucket=bucket)
        if rows > entry.capacity:
            growing.append((rows, entry))
    return growing


def _fixed_m4_growth_bill(growing: list[tuple[int, TensorOffsetQSACache]]) -> int:
    """Bytes a layer-by-layer growth adds, at its peak, to what the banks hold.

    The rows the layers grown before the peak gained, plus one layer's new
    banks written beside its old ones, which are let go before the next
    layer is admitted (``TensorOffsetQSACache.ensure_capacity``).
    """

    gained = peak = 0
    for rows, entry in growing:
        new, old = entry.growth_bytes(rows)
        peak = max(peak, gained + new)
        gained += new - old
    return peak


def _post_restore_eager_rounds() -> int:
    """Verify rounds routed eager after a large session-bank restore (opt-in).

    A restored cache (clone or bank reference lease) arrives with exact-size
    KV buffers, so the first compiled-route promotion ensure_capacity ->
    mx.concatenate's the restored KV per full-attention layer before the
    round can run. Deferring the first round(s) to eager moves that copy off
    the TTFT path; promotion happens one round later, mid-stream.

    DEFAULT 0 (off). Clean-room A/B 2026-08-06 (4k restore, fresh server):
    the promotion copy measured sub-milliseconds at 4k context (the 08-05
    turbo warm anomaly was dominated by first-shape-in-process compile
    traces plus postcommit stacking, not the copy), while the deferral's
    eager->compiled transition introduced one novel verify-shape trace
    (~100-200ms once per process). Net: no receipt that the deferral helps
    at agent-scale contexts, one measured cost. The copy grows linearly
    with restored context (~2 GB at 32k), so the lever may still pay at
    16k+ restores — enable via env and gate before flipping any default.
    """

    raw = os.environ.get(
        "MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", ""
    ).strip()
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    return 0


def _post_restore_min_tokens() -> int:
    """Restored-prefix size below which the post-restore deferral stays off.

    Small restores copy little (a 512-token prefix is ~tens of MB across the
    full-attention layers); the deferral only earns its round for mid/long
    contexts where the concatenate cost is user-visible.
    """

    raw = os.environ.get(
        "MTPLX_COMPILED_VERIFY_POST_RESTORE_MIN_TOKENS", ""
    ).strip()
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 2048
    return 2048


def _runtime_trunk_quant_bits(runtime: Any) -> int | None:
    """Bits of the first quantized trunk projection, or None if unquantized.

    Used by the turbo-profile per-model gate. 4-bit (Optimized-Speed) and
    8-bit (Optimized-Quality) trunks are measured wins with the growth-demote
    + shared-traces bank (2026-07-04 re-measure: q8 +10% bare / flat @7k /
    +6% rules-context, parity2 zero divergences — the 07-02 sprint's q8
    -15/-18% verdict was the per-request trace tax, since removed). Other
    quantizations (6-bit 9B) stay eager until measured.
    """

    try:
        model = getattr(runtime, "model", None)
        text_model = getattr(model, "language_model", model)
        inner = getattr(text_model, "model", text_model)
        for layer in getattr(inner, "layers", []) or []:
            for attr_path in (
                ("self_attn", "q_proj"),
                ("mlp", "gate_proj"),
                ("linear_attn", "in_proj_qkvz"),
            ):
                node = layer
                for name in attr_path:
                    node = getattr(node, name, None)
                    if node is None:
                        break
                bits = getattr(node, "bits", None)
                if bits is not None:
                    return int(bits)
        return None
    except Exception:
        return None


def _prism_ternary_fp16_loaded(runtime: Any) -> bool:
    """True for a Ternary Bonsai (Prism) pack that finished its float16 load checks.

    Its 2-bit trunk is the Prism ternary layout behind in-module Hadamard
    rotations, not a generic 2-bit affine trunk: the verify graph is the dense
    Qwen3.5 one with each projection's rotation inside its call. Admission
    needs the loader's own post-load report (rotation metadata, packed module
    count, float16 activations with float32 rotation signs).
    """

    model = getattr(runtime, "model", None)
    try:
        from .models.prism_hadamard_qwen35 import Model as _PrismModel
    except Exception:
        return False
    # isinstance, not the exact class: MTP injection swaps the loaded model's
    # class for a subclass (mtp_patch._MTPLXOuterModel) at load time.
    if not isinstance(model, _PrismModel):
        return False
    report = getattr(model, "_prism_post_load_report", None)
    records = getattr(model, "_prism_records", ()) or ()
    return (
        isinstance(report, dict)
        and report.get("model_type") == "prism_hadamard_qwen35"
        and bool(records)
        and report.get("packed_modules") == len(records)
        and report.get("float_dtypes") == ["float16", "float32"]
    )


def _compiled_verify_bits_gate_ok(runtime: Any) -> bool:
    if _env_enabled("MTPLX_COMPILED_VERIFY_FORCE"):
        return True
    bits = _runtime_trunk_quant_bits(runtime)
    # Measured-win allowlist: 4-bit and 8-bit affine trunks engage;
    # unquantized (None) passes for test rigs and bf16 research models.
    # Unmeasured quantizations (e.g. the 6-bit 9B) stay eager. The Prism
    # ternary 2-bit trunk (Ternary Bonsai 2) engages after its loader's
    # float16 checks: parity2 and speed receipts in the Bonsai night report.
    if bits == 2 and _prism_ternary_fp16_loaded(runtime):
        return True
    return bits is None or bits in (4, 8)


def _fused_kernel_keeps_float32(value: float) -> bool:
    """Whether float32(value) survives as a constant of a fused MLX kernel.

    MLX 0.32.2 writes a fused kernel's scalar constants into its Metal source
    with std::numeric_limits<float>::digits10 + 1 = 7 significant digits
    (mlx/backend/common/compiled.h); a float32 needs 9 to read back exactly.
    """

    import numpy as np

    exact = np.float32(value)
    return bool(np.float32(float(f"{float(exact):.7g}")) == exact)


def _runtime_trunk_activation_dtype(runtime: Any) -> Any:
    """The residual stream's dtype, from the loaded embedding and layer norms.

    The embedding (its scales when quantized) sets the stream's dtype and
    every RMSNorm promotes it to its weight's dtype (MLX rms_norm returns
    ``result_type(x, weight)``), so one float32 norm weight makes the rest of
    the trunk float32. None when the model has no recognizable embedding.
    """

    try:
        model = getattr(runtime, "model", None)
        text_model = getattr(model, "language_model", model)
        inner = getattr(text_model, "model", text_model)
        embed = getattr(inner, "embed_tokens", None)
        source = getattr(embed, "scales", None)
        if source is None:
            source = getattr(embed, "weight", None)
        if source is None:
            return None
        dtypes = [source.dtype]
        for layer in getattr(inner, "layers", []) or []:
            for name in ("input_layernorm", "post_attention_layernorm"):
                weight = getattr(getattr(layer, name, None), "weight", None)
                if weight is not None:
                    dtypes.append(weight.dtype)
        return mx.float32 if mx.float32 in dtypes else dtypes[0]
    except Exception:
        return None


def _float32_gdn_key_scale_head_dim(runtime: Any) -> int | None:
    """A GDN key head size whose float32 scales a fused kernel would change.

    The GatedDeltaNet verify forward scales its queries by head_k_dim ** -1
    and its keys by head_k_dim ** -0.5, as Python floats (mlx_lm qwen3_5 and
    qwen3_next, mtplx/gdn_capture.py; Flash-Next scales its queries by
    head_k_dim ** -0.5). With a bfloat16 or float16 stream they become
    half-precision constants, which 7 digits hold exactly; with a float32
    stream they are float32 constants, and at head size 128 the compiled
    verifier's fused kernel reads 128 ** -0.5 = 0.0883883461 back as
    0.0883883536, so its GDN states, hidden states and logits differ from the
    eager verifier's from the first round. Returns the first such head size
    of a float32 trunk, else None.
    """

    try:
        if _runtime_trunk_activation_dtype(runtime) != mx.float32:
            return None
        model = getattr(runtime, "model", None)
        text_model = getattr(model, "language_model", model)
        inner = getattr(text_model, "model", text_model)
        for layer in getattr(inner, "layers", []) or []:
            head_k_dim = getattr(
                getattr(layer, "linear_attn", None), "head_k_dim", None
            )
            if head_k_dim is None:
                continue
            head_k_dim = int(head_k_dim)
            if not (
                _fused_kernel_keeps_float32(head_k_dim ** -0.5)
                and _fused_kernel_keeps_float32(1.0 / head_k_dim)
            ):
                return head_k_dim
        return None
    except Exception:
        return None


def _float32_gdn_key_scale_why(head_k_dim: int) -> str:
    """The logged reason for a float32 GDN key-scale demotion."""

    return (
        "float32 activations, and MLX writes the GDN key scale "
        f"{head_k_dim}**-0.5 into fused kernels with 7 significant digits, so "
        "the compiled verifier would not match the eager one"
    )


def compare_verify_outputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_report_lines: int = 24,
) -> list[str]:
    """Exact-equality diff between two named verify output trees.

    Both arguments are flat mappings ``name -> leaf`` where leaves are arrays
    (mx or numpy) or plain python values.  Returns human-readable mismatch
    lines; an empty list means bit-exact agreement.
    """
    import numpy as np

    lines: list[str] = []

    def add(line: str) -> None:
        if len(lines) < max_report_lines:
            lines.append(line)
        elif len(lines) == max_report_lines:
            lines.append("... report truncated ...")

    for name in sorted(set(reference) | set(candidate)):
        if name not in reference:
            add(f"{name}: missing from reference output")
            continue
        if name not in candidate:
            add(f"{name}: missing from candidate output")
            continue
        ref = reference[name]
        cand = candidate[name]
        if ref is None or cand is None:
            if ref is not cand:
                add(f"{name}: one side is None ({type(ref).__name__} vs {type(cand).__name__})")
            continue
        if not hasattr(ref, "shape") and not hasattr(cand, "shape"):
            if ref != cand:
                add(f"{name}: value mismatch ({ref!r} vs {cand!r})")
            continue
        ref_np = _as_numpy(ref)
        cand_np = _as_numpy(cand)
        if ref_np.shape != cand_np.shape:
            add(f"{name}: shape mismatch ({ref_np.shape} vs {cand_np.shape})")
            continue
        if ref_np.dtype != cand_np.dtype:
            add(f"{name}: dtype mismatch ({ref_np.dtype} vs {cand_np.dtype})")
            continue
        if not np.array_equal(ref_np, cand_np):
            both = np.asarray(ref_np, dtype=np.float64) - np.asarray(cand_np, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                max_abs = float(np.nanmax(np.abs(both))) if both.size else 0.0
            mismatched = int(np.sum(ref_np != cand_np))
            add(
                f"{name}: value mismatch (elements={mismatched}/{ref_np.size}, "
                f"max_abs_diff={max_abs:.3e})"
            )
    return lines


class CompiledVerifyParityError(RuntimeError):
    """Raised in parity mode when compiled and eager verify outputs diverge."""

    def __init__(self, report: list[str]) -> None:
        self.report = list(report)
        super().__init__(
            "compiled verify parity mismatch:\n" + "\n".join(self.report)
        )


class CompiledVerifyBank:
    """Compiled speculative-verify dispatcher with a shadow-cache firewall.

    ``verify_step(input_ids, *state_in) -> (logits, hidden, *captures_flat,
    *state_out)`` is a pure function: every piece of cache state enters as an
    explicit input leaf and leaves as an explicit output leaf.  The dispatch
    wrapper reads the leaves from the real (promoted) cache entries, calls the
    compiled function, and mirror-commits the outputs back into the real
    entries with ``rollback_state`` cleared so the untouched accept
    (``commit_captured_prefix``) and reject (``rollback_after_verify`` ->
    offset-only ``trim``) paths keep working unchanged.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_verify_len: int | None = None,
        request_max_tokens: int | None = None,
        capture_backend: str | None = None,
        parity: bool = False,
        parity2: bool = False,
        restored_tokens: int = 0,
        rope_delta: mx.array | int | None = None,
        capacity_plan: FixedM4CapacityPlan | None = None,
    ) -> None:
        self.runtime = runtime
        # The request's rotary delta (an image request on the QSA family) or
        # None (every text request). The bank is built per request, so this is
        # request state: every fixed QSA bank it promotes carries the delta
        # (``TensorOffsetQSACache.rope_delta``) and every verify step it
        # compiles for this request takes it as a graph INPUT, never as a
        # captured constant: two image requests with different deltas replay
        # one trace.
        self._rope_delta: mx.array | None = None
        self._rope_delta_value: int | None = None
        self._adopt_rope_delta(rope_delta)
        if max_verify_len is None:
            raw = os.environ.get("MTPLX_COMPILED_VERIFY_MAX_LEN", "").strip()
            max_verify_len = int(raw) if raw else 6
        self.max_verify_len = int(max_verify_len)
        self.request_max_tokens = (
            None if request_max_tokens is None else max(0, int(request_max_tokens))
        )
        self.speculative_headroom = (
            self.max_verify_len if self.request_max_tokens is not None else 0
        )
        self.strict_no_fallback = bool(
            getattr(runtime, "qwen4_fixed_m4_compiled_verify", False)
        )
        # The strict lane's QSA banks round their rows-gather capacity up to
        # this bucket so consecutive requests replay one verify trace
        # (_fixed_m4_capacity_bucket); generic banks keep the exact rule.
        self.capacity_plan = (
            capacity_plan or FixedM4CapacityPlan.for_request(
                self.request_max_tokens, width=self.max_verify_len, runtime=runtime
            )
        ) if self.strict_no_fallback else None
        self.fixed_m4_capacity_bucket = self.capacity_plan.bucket if self.capacity_plan else 0
        # Generic banks let the request budget only TIGHTEN the reserve; it
        # never raises it past the env ceiling. Server requests default max_tokens to the
        # whole remaining context window (~262k on a 256k model), and
        # granting that verbatim made every request materialize a
        # multi-gigabyte KV reserve across all promoted leaves at first
        # promotion: +17 GB active / 44 GB peak, decode opening at ~13 tok/s
        # for the first ~150 tokens of every turn, and 8.8x commit cost
        # (2.4.0 short-turn regression, root-caused 2026-07-31). A bounded
        # grant restores the growth-demotion contract below: agent-length
        # rounds run fully compiled, longer generations demote to eager for
        # the request remainder (measured flat vs eager-only). Explicit
        # small budgets still reserve exactly budget + one speculative
        # window; raise MTPLX_COMPILED_VERIFY_GROWTH_RESERVE to widen the
        # stable-capacity generation. The construction-owned Qwen4 fixed-M4
        # lane has its own 1K default, then grows and reinstalls its graph at
        # capacity boundaries instead of demoting to eager. An explicit env
        # reserve remains authoritative for both lanes.
        reserve_ceiling = (
            _fixed_m4_initial_growth_reserve()
            if self.strict_no_fallback
            else _compiled_verify_growth_reserve()
        )
        self.growth_reserve_tokens = (
            min(
                self.request_max_tokens + self.speculative_headroom,
                max(
                    reserve_ceiling,
                    self.max_verify_len,
                ),
            )
            if self.request_max_tokens is not None
            else reserve_ceiling
        )
        if self.capacity_plan is not None:
            self.growth_reserve_tokens = self.capacity_plan.reserve_tokens
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self.parity = bool(parity)
        self.parity2 = bool(parity2)
        if self.parity and self.parity2:
            raise ValueError(
                "CompiledVerifyBank: parity and parity2 are mutually exclusive"
            )
        self.permanent_eager = False
        self.permanent_eager_reason: str | None = None
        compiled_verify_status["mode"] = (
            "parity" if self.parity else ("parity2" if self.parity2 else "on")
        )
        compiled_verify_status["permanent_eager"] = False
        if not parity and not parity2 and not _compiled_verify_bits_gate_ok(runtime):
            # Per-model promotion gate: 4-bit and 8-bit affine trunks engage
            # (both parity2-validated; q8's early -15/-18% reading predated
            # the 2.4.0 compiled stack — measured 2026-07-31: q8 304/304
            # compiled, 0 fallbacks, 41.3 tok/s at league parity). Unmeasured
            # quantizations (e.g. the 6-bit 9B) stay eager.
            self.permanent_eager = True
            self.permanent_eager_reason = (
                f"quant_bits_gate:bits={_runtime_trunk_quant_bits(runtime)}"
            )
            _record_permanent_eager(self.permanent_eager_reason, once=True)
        if not parity and not parity2 and not self.permanent_eager:
            # Committed state must equal the eager verifier's. A float32
            # stream through GDN layers whose key scale a fused kernel
            # rewrites cannot (see _float32_gdn_key_scale_head_dim); no
            # shipping pack streams float32. The parity diagnostics bypass
            # this gate, as they do the bits gate, to measure the mismatch.
            head_k_dim = _float32_gdn_key_scale_head_dim(runtime)
            if head_k_dim is not None:
                self.permanent_eager = True
                self.permanent_eager_reason = (
                    f"float32_gdn_key_scale:head_k_dim={head_k_dim}"
                )
                # A bank that may never fall back (the Flash-Next fixed-M4
                # lane's) verifies eagerly here instead of refusing the
                # request: eager is the decision, and it is logged once.
                self.strict_no_fallback = False
                _record_permanent_eager(
                    self.permanent_eager_reason,
                    once=True,
                    why=_float32_gdn_key_scale_why(head_k_dim),
                )
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
        capture_layout = getattr(runtime, "_mtplx_capture_layout", None)
        self._capture_layout_override = (
            None
            if capture_layout is None
            else tuple(str(name) for name in capture_layout)
        )
        self._extra_capture_layout = tuple(
            (int(layer_index), tuple(str(name) for name in names))
            for layer_index, names in tuple(
                getattr(runtime, "_mtplx_capture_extra_layout", ()) or ()
            )
        )
        prepare_aux = getattr(runtime, "prepare_compiled_verify_aux", None)
        self._prepare_compiled_aux = prepare_aux if callable(prepare_aux) else None
        build_fixed_aux = getattr(runtime, "build_fixed_m4_compiled_verify_aux", None)
        self._build_fixed_m4_aux = (
            build_fixed_aux if callable(build_fixed_aux) else None
        )
        commit_captures = getattr(runtime, "commit_compiled_verify_captures", None)
        self._commit_compiled_captures = (
            commit_captures if callable(commit_captures) else None
        )
        self._runtime_accepts_compiled_aux = _accepts_runtime_keyword(
            runtime, "compiled_aux"
        )
        if self._prepare_compiled_aux is not None and not self._runtime_accepts_compiled_aux:
            raise TypeError(
                "compiled verify auxiliary preparation requires a compiled_aux input"
            )
        self._compiled: dict[tuple[int, str, int, int], Any] = {}
        # Trace hosts of the shared programs in _compiled, by the same key
        # (see _bind_program).
        self._program_hosts: dict[tuple[int, str, int, int], dict[str, Any]] = {}
        self._spec: list[tuple[int, str, int]] | None = None
        self._shadow: list[Any] | None = None
        self._shadow_signature: tuple[Any, ...] | None = None
        self._gdn_meta_cache: dict[int, dict[str, int] | None] = {}
        self._exception_failures = 0
        self._held_state_refs: list = []
        # The Qwen4 fixed-M4 lane installs one construction-owned replay plan
        # after prompt prefill.  Production calls then bypass the generic
        # eligibility, promotion, bucket, shadow, and fallback machinery.
        # Parity modes intentionally stay on the generic dispatcher because
        # they need its eager comparison paths.
        self._fixed_m4_dispatch: dict[str, Any] | None = None
        # Generic lanes demote when dense leaves outgrow the initial grant.
        # The fixed-M4 lane instead performs explicit capacity-generation
        # transitions while keeping its installed direct route.
        self._growth_demoted = False
        # Per-call dispatch receipt.  Aggregate counters answer "how often";
        # these fields answer the more important Route Tape question: which
        # path served the most recent call.  Keep this on the bank so callers
        # never have to infer execution from counter deltas.
        self.last_dispatch_kind = "not_run"
        self.last_fallback_reason: str | None = None
        self.last_fallback_transition = False
        self._growth_budget_fallback_reported = False
        self._dense_capacity_grant: dict[int, int] | None = None
        # Post-restore warmup: a session-bank restore hands this generation
        # exact-size KV buffers, so the first promotion concatenate-copies the
        # whole restored context (see _post_restore_eager_rounds). Parity
        # modes keep full compiled coverage for the exactness harnesses.
        self._post_restore_eager_remaining = (
            _post_restore_eager_rounds()
            if (
                int(restored_tokens or 0) >= _post_restore_min_tokens()
                and not parity
                and not parity2
                and not self.strict_no_fallback
            )
            else 0
        )
        self.stats: dict[str, Any] = {
            "calls": 0,
            "compiled_calls": 0,
            "extended_calls": 0,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "buckets": {},
            "promoted": 0,
            "demotions": 0,
            "traces": 0,
            # Shared programs this bank found full and replaced; its first
            # dispatch after one re-traced (see _shared_or_new_verify_step).
            "shared_programs_retired": 0,
            "parity_checks": 0,
            "parity_failures": 0,
            "parity2_calls": 0,
            "parity2_divergent_calls": 0,
            "parity2_first_divergence": None,
            "growth_demotions": 0,
            "growth_handoff_materializations": 0,
            "growth_handoff_state_leaves": 0,
            "growth_handoff_materialize_time_s": 0.0,
            "fixed_m4_capacity_transitions": 0,
            "fixed_m4_route_transitions": 0,
        }

    # -- public API ---------------------------------------------------------

    def last_dispatch_route(self, compiled_route: str = "compiled_bank") -> str:
        """Return the actual route taken by the most recent public call."""
        if self.last_dispatch_kind == "eager":
            return f"bank_eager:{self.last_fallback_reason or 'unknown'}"
        if self.last_dispatch_kind == "compiled":
            return compiled_route
        return "not_run"

    def _adopt_rope_delta(self, value: mx.array | int | None) -> None:
        """Bind this request's rotary delta once; a second value must agree."""

        delta = as_rope_delta(value)
        if delta is None:
            return
        number = int(value) if isinstance(value, int) else int(delta.item())
        if self._rope_delta is not None:
            if number != self._rope_delta_value:
                raise ValueError(
                    "this verify bank already carries rope_delta="
                    f"{self._rope_delta_value}; a request has one delta, got {number}"
                )
            return
        mx.eval(delta)
        self._rope_delta = delta
        self._rope_delta_value = number

    def _stamp_rope_delta(self, cache: Any) -> None:
        """Every promoted container of this request owns its rotary origin.

        Fixed QSA banks (Flash-Next) and the full-attention tensor-offset
        adapters (the dense families, plain and paged) alike.
        """

        stamp_rope_delta(cache, self._rope_delta)

    def _verify_key(
        self, length: int, hidden_variant: str | None, bucket: int
    ) -> tuple[int, str, int, int]:
        """Per-bank key of one compiled verify step.

        The last element says whether the step takes the rotary delta as an
        input. A text request and an image request never share a trace: the
        text trace has no such input and stays the graph it always was.
        """

        return (
            int(length),
            str(hidden_variant or ""),
            int(bucket),
            int(self._rope_delta is not None),
        )

    def _rope_args(self) -> tuple[mx.array, ...]:
        """The delta as a positional graph input, or nothing for text."""

        return () if self._rope_delta is None else (self._rope_delta,)

    def install_fixed_m4(
        self,
        cache: Any,
        *,
        prompt_ids,
        hidden_variant: str | None,
        rope_delta: mx.array | int | None = None,
    ) -> None:
        """Install the exact Qwen4 physical-M4 replay once after prefill.

        ``rope_delta`` admits an image request: every QSA bank of the cache
        rotates at ``offset + rope_delta`` and the replayed graph takes the
        delta as an input. None is a text request and installs the text
        trace.
        """

        if not self.strict_no_fallback:
            raise ValueError("fixed-M4 installation requires the Qwen4 runtime route")
        if self.parity:
            raise ValueError(
                "fixed-M4 direct replay has no eager-authoritative parity mode; "
                "MTPLX_COMPILED_VERIFY=parity2 compares every round against the "
                "eager verifier"
            )
        self._adopt_rope_delta(rope_delta)

        class _M4Shape:
            shape = (1, 4)

        reason = self._fallback_reason(_M4Shape(), cache, True)
        if reason is not None:
            raise RuntimeError(f"qwen4 fixed-M4 installation refused: {reason}")
        bucket = self._resolve_bucket(cache, 4)
        if bucket != 0:
            raise RuntimeError(
                f"qwen4 fixed-M4 installation requires dense state; bucket={bucket}"
            )
        self._ensure_shadow(cache)
        state_plan = tuple(
            (kind, cache[idx], n_leaves)
            for idx, kind, n_leaves in self._spec or ()
        )
        if not state_plan or any(
            kind not in (VERIFY_SPEC_KIND_QSA, VERIFY_SPEC_KIND_GDN)
            for kind, _entry, _n in state_plan
        ):
            raise RuntimeError("qwen4 fixed-M4 installation found unsupported state")
        qsa_entries = tuple(
            entry for kind, entry, _n in state_plan if kind == VERIFY_SPEC_KIND_QSA
        )
        if not qsa_entries:
            raise RuntimeError("qwen4 fixed-M4 installation found no QSA state")
        self.fixed_m4_capacity_bucket = min(entry.capacity_bucket for entry in qsa_entries)
        if self.capacity_plan is not None:
            self.capacity_plan.bucket = self.fixed_m4_capacity_bucket
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        key = self._verify_key(4, hidden_variant, route_key)
        fn = self._verify_program(key, 4, hidden_variant)
        pending_route_thresholds = tuple(
            entry.rows_gather_min_context
            for entry in qsa_entries
            if entry.rows_gather_enabled and not entry.fixed_rows_gather
        )

        capture_plan = []
        capture_pos = 0
        for idx, names in self._extra_capture_layout:
            capture_plan.append((cache[idx], capture_pos, len(names)))
            capture_pos += len(names)

        boundary = _compiled_verify_boundary()
        if self._build_fixed_m4_aux is not None and boundary in ("both", "pre"):
            prepare_aux = self._build_fixed_m4_aux(cache, prompt_ids)
            aux_route = "staged_sidecar"
            aux_inputs = "host_ledger"
        else:
            prepare_aux = partial(
                _prepare_fixed_m4_materialized,
                self._prepare_compiled_aux,
                cache,
            )
            aux_route = "materialized"
            aux_inputs = "device_history"
        from .models.qwen4_exp import _qsa_stock_rows_gather_kv

        kv_gather = (
            "fused_m4"
            if any(
                kind == VERIFY_SPEC_KIND_QSA
                and entry.rows_gather_kv_m4 is not _qsa_stock_rows_gather_kv
                for kind, entry, _n in state_plan
            )
            else "stock"
        )
        initial_growth_tokens = max(
            self.max_verify_len,
            self.growth_reserve_tokens,
        )
        capacity_limit = (
            None
            if self.request_max_tokens is None
            else (
                len(prompt_ids)
                + self.request_max_tokens
                + self.speculative_headroom
            )
        )
        self._fixed_m4_dispatch = {
            "fn": fn,
            "host": self._program_hosts.get(key),
            "prepare_aux": prepare_aux,
            "aux_route": aux_route,
            "aux_inputs": aux_inputs,
            "kv_gather": kv_gather,
            "state_plan": state_plan,
            "state_leaves": sum(n for _kind, _entry, n in state_plan),
            "capture_plan": tuple(capture_plan),
            "capture_leaves": capture_pos,
            "boundary": boundary,
            "base_offset": len(prompt_ids),
            "capacity": min(entry.capacity for entry in qsa_entries),
            "dense_capacity": min(entry.dense_capacity for entry in qsa_entries),
            "capacity_bucket": self.fixed_m4_capacity_bucket,
            "growth_tokens": _next_fixed_m4_growth_tokens(
                initial_growth_tokens
            ),
            "capacity_limit": capacity_limit,
            "hidden_variant": hidden_variant,
            "qsa_entries": qsa_entries,
            "route_transition_at": (
                min(pending_route_thresholds)
                if pending_route_thresholds
                else None
            ),
            # parity2 reads the pre-round leaves after the replay, so nothing
            # may be donated into it.
            "donate": (
                _compiled_verify_donation_enabled()
                and boundary in ("both", "post")
                and not self.parity2
            ),
            # The replay's positional inputs after the auxiliary: the delta
            # of an image request, nothing for text. Built once here so the
            # round itself assembles no tuple and the text call is the call
            # it always was.
            "rope_args": self._rope_args(),
            "rope_delta": self._rope_delta_value,
            "parity2": bool(self.parity2),
        }
        # Engagement receipt (counters law): the request report carries the
        # bound auxiliary route (to_dict -> compiled_verify.fixed_m4) and the
        # first installs per process announce it in the serve log, so an A/B
        # can prove the staged sidecar lane ran rather than the materialized
        # PLE embedding.
        global _FIXED_M4_INSTALL_RECEIPTS
        if _FIXED_M4_INSTALL_RECEIPTS < 3:
            _FIXED_M4_INSTALL_RECEIPTS += 1
            print(
                "[qwen4-fixed-M4-verify] replay installed: "
                f"aux={aux_route} inputs={aux_inputs} boundary={boundary} "
                f"donate={self._fixed_m4_dispatch['donate']} kv_gather={kv_gather} "
                "positions="
                + ("text" if self._rope_delta is None else "vision_delta"),
                flush=True,
            )

    def fixed_m4_capacity_receipt(self) -> dict[str, Any] | None:
        """The installed fixed-M4 bank's capacity, for the Route Tape.

        None when no fixed-M4 replay is installed. ``capacity_bucket`` is the
        configured bucket; it shapes the capacity only while ``rows_gather``.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return None
        return {
            "capacity": int(dispatch["capacity"]),
            "capacity_bucket": int(dispatch["capacity_bucket"]),
            "rows_gather": all(
                entry.fixed_rows_gather for entry in dispatch["qsa_entries"]
            ),
            "base_offset": int(dispatch["base_offset"]),
        }

    def prefetch_fixed_m4_aux(
        self,
        host_input_prefix,
        *,
        completion_tokens,
        committed_count: int,
    ) -> int:
        """Warm the n-gram rows of the next verify window's first tokens.

        The verify replay opens with a host row gather that needs the draft
        tokens, so the GPU sits idle for it every round.  The draft loop calls
        this as each token becomes known, while the GPU is still drafting; the
        replay's own gather then reads hot rows.  An instrument of timing
        only: it returns a count, produces no array, and a failure here is
        recorded and ignored because the gather does not depend on it.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return 0
        prefetch = getattr(dispatch.get("prepare_aux"), "prefetch", None)
        if prefetch is None:
            return 0
        try:
            fetched = int(
                prefetch(host_input_prefix, completion_tokens, committed_count)
            )
        except Exception as error:  # noqa: BLE001 - never on the decode path
            self.stats["fixed_m4_aux_prefetch_error"] = repr(error)[:200]
            return 0
        self.stats["fixed_m4_aux_prefetch_calls"] = (
            int(self.stats.get("fixed_m4_aux_prefetch_calls", 0)) + 1
        )
        self.stats["fixed_m4_aux_prefetch_rows"] = (
            int(self.stats.get("fixed_m4_aux_prefetch_rows", 0)) + fetched
        )
        return fetched

    def reserve_fixed_m4_round(
        self,
        cache: Any,
        *,
        committed_count: int,
        copy_window: int = 0,
    ) -> None:
        """Grow an installed fixed-M4 bank for the round about to run, first.

        The verify's own reservation (a four-row window at the same committed
        count) is made here, before the round samples, drafts or writes, so
        every forward sees the capacity it saw before. On the rows-gather
        lane, where the bank's capacity changes no value, the widest copy
        window is reserved too. A refusal (``FixedM4GrowthRefused``) then
        comes between rounds, with the committed state untouched.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return
        window = 4
        if int(copy_window) > window and all(
            entry.fixed_rows_gather for entry in dispatch["qsa_entries"]
        ):
            window = int(copy_window)
        self.reserve_fixed_m4_window(
            cache, committed_count=committed_count, window_tokens=window
        )

    def reserve_fixed_m4_window(
        self,
        cache: Any,
        *,
        committed_count: int | None = None,
        window_tokens: int = 4,
        final_capture: bool = False,
    ) -> None:
        """Reserve every target write into an installed fixed-capacity bank.

        D1/D2 and copy windows use these same buffers even when their forward
        runs eager. They must renew capacity before writing, too. Generation
        supplies its host ledger to avoid a device sync; standalone callers
        without that ledger use the live cache offset. Generic banks are
        unchanged. Verification keeps four rows for a possible lazy bonus,
        even at width one. Only an explicit final capture has no following
        write and reserves just its window, preserving an existing grant.

        A growth the memory refuses raises ``FixedM4GrowthRefused`` with
        every layer whole, and the caller ends the answer between rounds. Any
        other exception while the banks grow (a cancellation, a programming
        error) first hands the cache back as stock containers, converted on a
        copy of the list and published in one step, so the prompt's lease can
        rewind it. If that hand-back fails as well, every adapter stays and
        the session bank will not serve the lease. The original exception
        always propagates.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return
        logical_start = (
            int(dispatch["base_offset"]) + max(0, int(committed_count))
            if committed_count is not None
            else dispatch["qsa_entries"][0].size()
        )
        window_tokens = int(window_tokens)
        required_end = logical_start + (
            window_tokens if final_capture else max(4, window_tokens)
        )
        # Follow the parent's grants independently of the larger allocation.
        # A rejection trims the offset, not these already granted widths.
        capacity_needed = required_end > int(dispatch["dense_capacity"])
        route_transition_at = dispatch["route_transition_at"]
        route_needed = (
            route_transition_at is not None
            and required_end >= int(route_transition_at)
        )
        if not capacity_needed and not route_needed:
            return

        qsa_entries = dispatch["qsa_entries"]
        capacity_changed = False
        next_growth_tokens = int(dispatch["growth_tokens"])
        if capacity_needed:
            try:
                next_capacity, next_growth_tokens = _fixed_m4_capacity_growth(
                    capacity=int(dispatch["dense_capacity"]),
                    required_end=required_end,
                    growth_tokens=int(dispatch["growth_tokens"]),
                    capacity_limit=dispatch["capacity_limit"],
                )
                admit = self.capacity_plan.admit_growth if self.capacity_plan else None
                growing = _fixed_m4_growing(qsa_entries, next_capacity)
                if growing:
                    # The banks grow one layer at a time below: each layer's new
                    # banks are written beside its old ones, swapped in, and the
                    # old ones let go before the next layer is admitted, so the
                    # transition holds the layers already grown plus one layer's
                    # new banks, never a second bank (the 09-29 refusals priced
                    # the whole new bank and were 0.51 GB short where one layer
                    # needed 0.22 GB). That is the bill ``admit`` takes before
                    # any leaf changes; one layer's new banks alone (the price
                    # until the 2026-10-01 review) left out the rows the earlier
                    # layers kept.
                    if admit is not None and not admit(_fixed_m4_growth_bill(growing)):
                        # Check the unbucketed width before giving up the request.
                        # No cache leaf changes before this admission.
                        growing = _fixed_m4_growing(qsa_entries, next_capacity, bucket=0)
                        bill = _fixed_m4_growth_bill(growing)
                        if not admit(bill):
                            raise FixedM4GrowthRefused(
                                capacity=int(dispatch["dense_capacity"]),
                                required_end=required_end,
                                rows=max(rows for rows, _entry in growing),
                                bill_bytes=bill,
                            )
                        for entry in qsa_entries:
                            entry.capacity_bucket = 0
                        self.fixed_m4_capacity_bucket = self.capacity_plan.bucket = 0
                        dispatch["capacity_bucket"] = 0
                    # Nothing compiled may still hold an old leaf: a shadow twin or
                    # a retained input list would keep every old layer alive until
                    # the whole transition ends.
                    self._clear_shadow_leaf_refs()
                    self._held_state_refs.clear()
                grown = 0
                failure = None
                for entry in qsa_entries:
                    rows = entry.growth_rows(next_capacity)
                    if (
                        rows > entry.capacity
                        and admit is not None
                        and not admit(entry.growth_bytes(rows)[0])
                    ):
                        # Memory taken since the bill was admitted: stop between
                        # layers, every one whole.
                        raise FixedM4GrowthRefused(
                            capacity=int(dispatch["dense_capacity"]),
                            required_end=required_end,
                            rows=rows,
                            layers_left=len(growing) - grown,
                        )
                    try:
                        grew = entry.ensure_capacity(next_capacity)
                    except Exception as exc:
                        # The memory refusing a layer's new banks while they are
                        # written leaves that layer as it was: the answer ends
                        # between rounds like a refused growth, its lease whole.
                        # Anything else is not the growth's to absorb.
                        if not is_allocation_failure(exc):
                            raise
                        failure = f"{type(exc).__name__}: {exc}"
                        break
                    grown += int(grew)
                    capacity_changed = grew or capacity_changed
                if failure is not None:
                    # The failed layer's new banks went with the exception: wait
                    # for the GPU to let go of what it was writing into them, then
                    # hand them back to the system.
                    mx.synchronize()
                    mx.clear_cache()
                    raise FixedM4GrowthRefused(
                        capacity=int(dispatch["dense_capacity"]),
                        required_end=required_end,
                        rows=max(rows for rows, _entry in growing),
                        layers_left=len(growing) - grown,
                        allocation_error=failure,
                    )
                dispatch["dense_capacity"] = min(entry.dense_capacity for entry in qsa_entries)
                dispatch["growth_tokens"] = next_growth_tokens
            except FixedM4GrowthRefused:
                # The caller ends at a committed boundary and performs the
                # normal final capture and demotion.
                raise
            except BaseException:
                # Growth publishes one complete layer at a time. Even an
                # interrupted allocation leaves every layer's prefix valid;
                # return stock containers so the prompt lease can rewind it.
                try:
                    # Convert references first, without copying tensor data.
                    # If a second error interrupts demotion, the original
                    # adapters still mark the whole lease as unavailable.
                    restored = list(cache)
                    self.demote(restored, compact=False, keep_capacity=True)
                    cache[:] = restored
                except Exception as error:
                    print(
                        f"[mtplx] cache hand-back after interrupted growth failed ({error}); "
                        "the original error is preserved",
                        file=sys.stderr,
                    )
                raise
        route_changed = False
        if route_needed:
            for entry in qsa_entries:
                if logical_start < entry.indexer_budget:
                    entry.capacity_bucket = 0
                route_changed = (
                    entry.activate_rows_gather(required_end) or route_changed
                )
            pending_route_thresholds = tuple(
                entry.rows_gather_min_context
                for entry in qsa_entries
                if entry.rows_gather_enabled and not entry.fixed_rows_gather
            )
            dispatch["route_transition_at"] = (
                min(pending_route_thresholds)
                if pending_route_thresholds
                else None
            )
            self.fixed_m4_capacity_bucket = min(entry.capacity_bucket for entry in qsa_entries)
            dispatch["capacity_bucket"] = self.fixed_m4_capacity_bucket
            if self.capacity_plan is not None:
                self.capacity_plan.bucket = self.fixed_m4_capacity_bucket
        if not capacity_changed and not route_changed:
            return

        self._clear_shadow_leaf_refs()
        self._held_state_refs.clear()
        self._shadow = None
        self._shadow_signature = None
        self._ensure_shadow(cache)
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        key = self._verify_key(4, dispatch["hidden_variant"], route_key)
        dispatch["fn"] = self._verify_program(key, 4, dispatch["hidden_variant"])
        dispatch["host"] = self._program_hosts.get(key)
        dispatch["capacity"] = min(entry.capacity for entry in qsa_entries)
        if capacity_changed:
            dispatch["growth_tokens"] = next_growth_tokens
            self.stats["fixed_m4_capacity_transitions"] += 1
        if route_changed:
            # Receipt refresh: the rows-gather activation may have bound the
            # fused kernel, so the request report must not keep carrying the
            # install-time gather label.
            from .models.qwen4_exp import _qsa_stock_rows_gather_kv

            dispatch["kv_gather"] = (
                "fused_m4"
                if any(
                    entry.rows_gather_kv_m4 is not _qsa_stock_rows_gather_kv
                    for entry in qsa_entries
                )
                else "stock"
            )
            self.stats["fixed_m4_route_transitions"] += 1

    def _fixed_m4_program(self, width: int):
        """The installed replay's compiled step for ``width`` rows, and its trace host.

        Four rows is the verify round's program, bound at install and at every
        capacity or route transition. Any other width (a copy window) is the
        same verify step traced at that width: the same state plan, capture
        layout, auxiliary and rotary inputs, keyed like the four-row program
        on the route, so it is shared by every bank of the process and MLX
        keeps one trace per bank capacity.
        """

        dispatch = self._fixed_m4_dispatch
        if width == 4:
            return dispatch["fn"], dispatch["host"]
        route_key = int(all(entry.fixed_rows_gather for entry in dispatch["qsa_entries"]))
        key = self._verify_key(width, dispatch["hidden_variant"], route_key)
        fn = self._verify_program(key, width, dispatch["hidden_variant"])
        return fn, self._program_hosts.get(key)

    def _forward_installed_fixed_m4(
        self,
        input_ids,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache: Any,
    ):
        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        # A verify round replays four rows (its reservation keeps the lazy
        # bonus row too); a copy window replays its own width.
        width = _decode_length(input_ids)
        self.reserve_fixed_m4_window(
            cache,
            committed_count=committed_count,
            window_tokens=width,
        )
        # After the reservation: a capacity or route transition there binds
        # the programs of the grown bank.
        fn, host = self._fixed_m4_program(width)
        boundary = dispatch["boundary"]
        donate = dispatch["donate"]
        if donate:
            self._clear_shadow_leaf_refs()

        state_in: list[Any] = []
        # Every input but the QSA bank buffers: the ones the pre-dispatch
        # evaluation below may have to run (a commit after a rejection leaves
        # the recurrent state and the offsets pending).
        pending: list[Any] = []
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                state_in.extend(
                    (
                        entry.kv.cache[0],
                        entry.kv.cache[1],
                        entry.kv.cache[2],
                        entry.raw_keys,
                        entry.pooled,
                    )
                )
                pending.append(entry.kv.cache[2])
            else:
                state_in.extend(entry.cache[:n_leaves])
                pending.extend(entry.cache[:n_leaves])

        # Host split of the replay (four clock reads per round): the n-gram row
        # gather, the input eval, the graph replay and the encode. The verify
        # forward is most of a decode round and it is one opaque host call
        # otherwise; the request log carries these as compiled_verify
        # fixed_m4_host_s (copy windows: fixed_m4_copy_window_host_s) so a
        # slow round can be read, not guessed.
        host_split = self.stats.setdefault(
            "fixed_m4_host_s" if width == 4 else "fixed_m4_copy_window_host_s",
            {"aux": 0.0, "input_eval": 0.0, "replay": 0.0, "encode": 0.0},
        )
        clock = time.perf_counter
        t0 = clock()
        compiled_aux = dispatch["prepare_aux"](
            input_ids,
            host_input_ids,
            completion_tokens,
            committed_count,
        )
        t1 = clock()
        if boundary in ("both", "pre"):
            # A pending evaluation holds every array it is handed until it
            # completes, and the compiled step's bank writes are encoded right
            # after: a K, V or raw bank held then is copied, not written in
            # place (MLX donates a buffer only to its sole holder). A copy
            # window therefore hands over only what can be pending; the banks
            # are materialized or, if not, run by their own kernels as inputs
            # of the step, so no value changes. A verify round keeps the call
            # it shipped with.
            mx.async_eval(compiled_aux, *(state_in if width == 4 else pending))
        t2 = clock()
        self._bind_program(host)
        # Argument order is the trace's contract (``_make_verify_step``):
        # ids, the auxiliary, the rotary delta of an image request, the state.
        identity = (id(fn), tuple(getattr(input_ids, "shape", ())))
        with compiled_dispatch(identity):
            outputs = fn(input_ids, compiled_aux, *dispatch["rope_args"], *state_in)
        t3 = clock()
        host_split["aux"] += t1 - t0
        host_split["input_eval"] += t2 - t1
        host_split["replay"] += t3 - t2

        capture_end = 2 + dispatch["capture_leaves"]
        logits, hidden = outputs[:2]
        captures_flat = outputs[2:capture_end]
        state_out = outputs[capture_end:]

        if dispatch.get("parity2"):
            # The real entries still hold the pre-round leaves here: parity2
            # installs without donation and the commit below has not run.
            self._fixed_m4_parity2_round(
                dispatch,
                cache,
                input_ids,
                compiled_aux=compiled_aux,
                candidate_logits=logits,
                candidate_hidden=hidden,
                candidate_captures=self._named_fixed_m4_captures(captures_flat),
                candidate_state=list(state_out),
                committed_count=committed_count,
                dispatch_kind="compiled",
            )

        if not donate and boundary in ("both", "post"):
            mx.async_eval(*outputs)
            self._held_state_refs.clear()
        elif not donate:
            self._held_state_refs.append(state_in)
            if len(self._held_state_refs) > 3:
                self._held_state_refs.pop(0)

        state_pos = 0
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[state_pos]
                entry.kv.cache[1] = state_out[state_pos + 1]
                entry.kv.cache[2] = state_out[state_pos + 2]
                entry.raw_keys = state_out[state_pos + 3]
                entry.pooled = state_out[state_pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[state_pos + slot]
            state_pos += n_leaves

        for entry, start, count in dispatch["capture_plan"]:
            entry._mtplx_verify_rows = tuple(captures_flat[start : start + 6])
            if count > 6:
                entry._mtplx_verify_ple = tuple(
                    captures_flat[start + 6 : start + count]
                )

        if donate:
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)
        host_split["encode"] += clock() - t3

        if _FIXED_M4_DONATION_PROBE:
            self._probe_fixed_m4_donation(dispatch)

        # Route Tape receipt (main contract): the installed replay is a
        # compiled dispatch, so last_dispatch_route() reports it as such.
        self.last_dispatch_kind = "compiled"
        self.last_fallback_reason = None
        self.last_fallback_transition = False
        self.stats["compiled_calls"] += 1
        self.stats["buckets"]["0"] = self.stats["buckets"].get("0", 0) + 1
        return logits, hidden, {}

    def _probe_fixed_m4_donation(self, dispatch) -> None:
        """MTPLX_FIXED_M4_DONATION_PROBE=1: did the verify write land in place?

        A measuring instrument (it forces a sync, so it is never on in a
        product lane).  After each installed replay it reads the device
        address of the first QSA layer's key bank.  A donated buffer keeps its
        address from one round to the next; a bank that was copied into a
        fresh allocation moves.  ``fixed_m4_kv_bank_moves`` over
        ``fixed_m4_kv_bank_probes`` is the share of rounds that paid a whole
        -bank copy.
        """

        import numpy as np

        entry = dispatch["qsa_entries"][0]
        keys = entry.kv.cache[0]
        mx.eval(keys)
        address = int(
            np.asarray(keys.view(mx.uint16), copy=False).__array_interface__["data"][0]
        )
        previous = dispatch.get("_probe_key_address")
        dispatch["_probe_key_address"] = address
        self.stats["fixed_m4_kv_bank_probes"] = (
            int(self.stats.get("fixed_m4_kv_bank_probes", 0)) + 1
        )
        if previous is not None and previous != address:
            self.stats["fixed_m4_kv_bank_moves"] = (
                int(self.stats.get("fixed_m4_kv_bank_moves", 0)) + 1
            )
        if _FIXED_M4_DONATION_PROBE_ALL:
            self._probe_fixed_m4_donation_all_leaves(dispatch)

    def _probe_fixed_m4_donation_all_leaves(self, dispatch) -> None:
        """MTPLX_FIXED_M4_DONATION_PROBE=all: which state leaves move per round?

        The first-key-bank probe above cleared one leaf of one layer.  Every
        other leaf the replay owns (values, the third KV leaf, raw keys, the
        pooled bank, the GDN state) can still lose donation on its own, and a
        context-sized leaf that is copied every round is a cost that grows
        with the prompt.  Per leaf class this records how many rounds saw a
        new device address and how many bytes those rounds re-wrote.
        """

        import numpy as np

        names = ("kv0", "kv1", "kv2", "raw_keys", "pooled")
        leaves: list[tuple[str, Any]] = []
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                leaves.extend(
                    zip(
                        names,
                        (
                            entry.kv.cache[0],
                            entry.kv.cache[1],
                            entry.kv.cache[2],
                            entry.raw_keys,
                            entry.pooled,
                        ),
                    )
                )
            else:
                leaves.extend(("gdn", entry.cache[slot]) for slot in range(n_leaves))
        mx.eval([leaf for _name, leaf in leaves if leaf is not None])
        previous = dispatch.get("_probe_leaf_addresses")
        current: list[int | None] = []
        moves = self.stats.setdefault("fixed_m4_leaf_moves", {})
        moved_bytes = self.stats.setdefault("fixed_m4_leaf_moved_bytes", {})
        for index, (name, leaf) in enumerate(leaves):
            # Scalars (the logical offset) are rebuilt every round by design
            # and cannot be viewed as bytes; only buffers are of interest.
            if leaf is None or leaf.ndim == 0 or leaf.nbytes < 4096:
                current.append(None)
                continue
            try:
                flat = leaf.view(mx.uint8) if leaf.dtype != mx.uint8 else leaf
                address = int(
                    np.asarray(flat, copy=False).__array_interface__["data"][0]
                )
            except Exception as exc:  # noqa: BLE001 - an instrument never fails a round
                self.stats["fixed_m4_leaf_probe_error"] = f"{name}: {exc}"
                current.append(None)
                continue
            current.append(address)
            if (
                previous is not None
                and index < len(previous)
                and previous[index] is not None
                and previous[index] != address
            ):
                moves[name] = int(moves.get(name, 0)) + 1
                moved_bytes[name] = int(moved_bytes.get(name, 0)) + int(leaf.nbytes)
        dispatch["_probe_leaf_addresses"] = current
        self.stats["fixed_m4_leaf_probe_rounds"] = (
            int(self.stats.get("fixed_m4_leaf_probe_rounds", 0)) + 1
        )
        self.stats["fixed_m4_leaf_count"] = len(leaves)

    def forward_fixed_m4(
        self,
        input_ids,
        *,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        """Run the installed physical-M4 route with host-owned n-gram inputs.

        The first dispatch of the process runs under a guard
        (``_prove_fixed_m4_dispatch``): a kernel an older GPU refuses retires
        the lane to the eager verifier for the life of the process instead
        of failing the request. Once proven, this is the unguarded replay.
        """

        del return_hidden
        self.stats["calls"] += 1
        if _FIXED_M4_LANE["retired"] is not None:
            return self._fixed_m4_retired_forward(
                input_ids,
                cache=cache,
                committed_count=committed_count,
                hidden_variant=hidden_variant,
            )
        if _FIXED_M4_LANE["proven"]:
            return self._forward_installed_fixed_m4(
                input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
                cache,
            )
        return self._prove_fixed_m4_dispatch(
            input_ids,
            host_input_ids,
            completion_tokens,
            committed_count,
            cache,
            hidden_variant=hidden_variant,
        )

    # -- copy windows on the installed bank (MTPLX_FIXED_M4_COPY_WINDOWS=1) -----
    #
    # A context-copy block round forwards the primary and a block of prompt
    # tokens (9, 13, 17 or 25 rows by default) over the conversation. The
    # four-row replay was this lane's only compiled forward, so every block
    # round built the eager graph of the whole model and paid the eager bank
    # write's two syncs per QSA layer (the offset read and the rollback copy):
    # 116 to 146 ms per round at 125K+ in the 2026-09-29 Pi session, the
    # stutter of a file rewrite. Here the same verify step replays at the
    # block's width, with the same inputs, captures and in-place state.

    def fixed_m4_copy_window_refusal(self, width: int) -> str | None:
        """Why a copy window of ``width`` rows would run eager now, or None.

        A window replays the four-row lane's verify step traced at its width,
        so it needs that lane installed, proven on this GPU and not retired.
        It needs a bank whose capacity is bucketed on the rows-gather lane:
        there one trace per width serves every round and turn up to the next
        bucket edge, where the dense lane's exact per-request capacity would
        trace every width once per request. And it needs a width the verify
        routes serve in every layer (``qwen4_exp.verify_route_max_rows``),
        where the compiled trace is the eager forward bit for bit.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return "not_installed"
        if _FIXED_M4_LANE["retired"] is not None:
            return "lane_retired"
        if not _FIXED_M4_LANE["proven"]:
            return "lane_not_proven"
        if _FIXED_M4_COPY_WINDOWS["retired"] is not None:
            return "copy_windows_retired"
        if int(dispatch["capacity_bucket"]) <= 0 or not all(
            entry.fixed_rows_gather for entry in dispatch["qsa_entries"]
        ):
            return "capacity_not_bucketed"
        from .models.qwen4_exp import verify_route_max_rows

        if not 2 <= int(width) <= verify_route_max_rows():
            return "width_past_verify_routes"
        return None

    def replay_fixed_m4_copy_window(
        self,
        input_ids,
        *,
        widths,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache,
    ):
        """Replay a copy window on the installed bank: ``(logits, hidden)``, or None.

        ``widths`` are the request's full-length copy widths
        (``context_copy.ladder_widths``); a cut-short block keeps the eager
        forward, so the ladder bounds the traces to one per width and bucket.
        None means the caller runs its eager forward over an untouched cache;
        the reason is counted in ``fixed_m4_copy_windows["eager"]``. Otherwise
        the window ran like a verify round: the captures the family commit
        reads are on the cache entries and the state was written in place.
        """

        width = _decode_length(input_ids)
        receipt = self.stats.setdefault(
            "fixed_m4_copy_windows", {"compiled": {}, "traces": {}, "eager": {}}
        )
        traces_before = int(self.stats.get("traces", 0))
        refusal = (
            self.fixed_m4_copy_window_refusal(width)
            if width in widths
            else "width_off_the_ladder"
        )
        if refusal is None and width not in _FIXED_M4_COPY_WINDOWS["proven"]:
            refusal = self._prove_fixed_m4_copy_window(
                input_ids, host_input_ids, completion_tokens, committed_count, cache
            )
        if refusal is not None:
            receipt["eager"][refusal] = receipt["eager"].get(refusal, 0) + 1
            return None
        self.stats["calls"] += 1
        logits, hidden, _captures = self._forward_installed_fixed_m4(
            input_ids, host_input_ids, completion_tokens, committed_count, cache
        )
        key = str(width)
        receipt["compiled"][key] = receipt["compiled"].get(key, 0) + 1
        traced = int(self.stats.get("traces", 0)) - traces_before
        if traced:
            receipt["traces"][key] = receipt["traces"].get(key, 0) + traced
        return logits, hidden

    def _prove_fixed_m4_copy_window(
        self, input_ids, host_input_ids, completion_tokens, committed_count, cache
    ) -> str | None:
        """First window of a width in this process: trace it, or retire copy windows.

        The four-row lane's guard evaluates its first replay with the input
        leaves held, which blocks their donation: the bank is copied once.
        That is the warmup's small bank for the four-row lane, but a width's
        first copy window can come at 128K, where one copy of the QSA state
        is about 3.7 GB. So this guard traces the width's program alone
        (``_trace_fixed_m4_copy_window``: nothing evaluated, nothing rebound)
        and the replay that follows donates as every round does. A trace that
        raises (a host read at this width, a route the tiny test pack never
        reaches) retires compiled copy windows for the process, with one
        printed line and a ledger entry, and this window runs the eager
        forward on the state it found. A failure after the trace, when the
        graph runs, is the same failure a proven four-row round would raise.
        """

        width = _decode_length(input_ids)
        self.reserve_fixed_m4_window(
            cache, committed_count=committed_count, window_tokens=width
        )
        try:
            self._trace_fixed_m4_copy_window(
                input_ids, host_input_ids, completion_tokens, committed_count
            )
        except Exception as exc:  # noqa: BLE001 - retired, printed and counted; the round runs eager
            retire_fixed_m4_copy_windows(f"{type(exc).__name__}: {exc}")
            return "copy_windows_retired"
        _FIXED_M4_COPY_WINDOWS["proven"].add(width)
        return None

    def _trace_fixed_m4_copy_window(
        self, input_ids, host_input_ids, completion_tokens, committed_count
    ) -> None:
        """Call this width's program on the live inputs and drop its outputs.

        A compiled step traces on its first call at a shape. Made here, the
        call evaluates nothing (no GPU work, no allocation) and rebinds
        nothing, so a trace that raises leaves the round as it was; the
        replay that follows reuses the trace.
        """

        dispatch = self._fixed_m4_dispatch
        fn, host = self._fixed_m4_program(_decode_length(input_ids))
        compiled_aux = dispatch["prepare_aux"](
            input_ids, host_input_ids, completion_tokens, committed_count
        )
        self._bind_program(host)
        with compiled_dispatch((id(fn), tuple(getattr(input_ids, "shape", ())))):
            fn(
                input_ids,
                compiled_aux,
                *dispatch["rope_args"],
                *self._fixed_m4_state_leaves(),
            )

    def _fixed_m4_state_refs(self) -> list[tuple[Any, str, tuple, tuple, tuple]]:
        """References (never copies) to every leaf the replay rebinds.

        Holding them keeps MLX from donating the input buffers for this one
        dispatch, so a failed dispatch leaves the pre-round state intact and
        restoring it is a rebind.
        """

        dispatch = self._fixed_m4_dispatch
        saved: list[tuple[Any, str, tuple, tuple, tuple]] = []
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                leaves = (
                    entry.kv.cache[0],
                    entry.kv.cache[1],
                    entry.kv.cache[2],
                    entry.raw_keys,
                    entry.pooled,
                )
                rollback = tuple(entry.kv.rollback_state)
            else:
                leaves = tuple(entry.cache[:n_leaves])
                rollback = ()
            captured = (
                getattr(entry, "_mtplx_verify_rows", _MISSING),
                getattr(entry, "_mtplx_verify_ple", _MISSING),
            )
            saved.append((entry, kind, leaves, rollback, captured))
        return saved

    @staticmethod
    def _restore_fixed_m4_state(saved: list[tuple[Any, str, tuple, tuple, tuple]]) -> None:
        for entry, kind, leaves, rollback, captured in saved:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0], entry.kv.cache[1], entry.kv.cache[2] = leaves[:3]
                entry.raw_keys = leaves[3]
                entry.pooled = leaves[4]
                for slot, value in enumerate(rollback):
                    entry.kv.rollback_state[slot] = value
            else:
                for slot, value in enumerate(leaves):
                    entry.cache[slot] = value
            for name, value in zip(("_mtplx_verify_rows", "_mtplx_verify_ple"), captured):
                if value is _MISSING:
                    if hasattr(entry, name):
                        try:
                            delattr(entry, name)
                        except AttributeError:
                            pass
                else:
                    setattr(entry, name, value)

    def _prove_fixed_m4_dispatch(
        self,
        input_ids,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache: Any,
        *,
        hidden_variant: str | None,
    ):
        """First dispatch of the process: prove the lane or retire it.

        Every receipt behind the fixed-M4 kernels is an M5 Max, the lane is
        armed by model geometry alone, and its dispatch had no try/except: a
        kernel that an older GPU refuses to build or dispatch (the 896-thread
        pipeline cap of issue #400 is the known shape of that failure) was a
        failed request. The generic bank already falls back after an
        exception streak; this lane now does the same on its first dispatch.

        The round is evaluated inside the guard (the generation loop blocks
        on these logits next anyway), with the input leaves held so nothing
        is donated: on a failure the pre-round state is restored by rebinding
        and the same verify window runs the eager forward, the route every
        width other than 4 already takes through this bank.
        """

        self.reserve_fixed_m4_window(cache, committed_count=committed_count)
        saved = self._fixed_m4_state_refs()
        compiled_before = int(self.stats.get("compiled_calls", 0))
        bucket_before = int(self.stats["buckets"].get("0", 0))
        try:
            result = self._forward_installed_fixed_m4(
                input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
                cache,
            )
            mx.eval(result[0], result[1])
        except Exception as exc:  # noqa: BLE001 - any dispatch failure retires the lane
            self._restore_fixed_m4_state(saved)
            self._held_state_refs.clear()
            # The replay counts itself before the guard's blocking eval; a
            # round that failed there was not a compiled round.
            self.stats["compiled_calls"] = compiled_before
            self.stats["buckets"]["0"] = bucket_before
            retire_fixed_m4_lane(f"{type(exc).__name__}: {exc}")
            return self._fixed_m4_retired_forward(
                input_ids,
                cache=cache,
                committed_count=committed_count,
                hidden_variant=hidden_variant,
            )
        _FIXED_M4_LANE["proven"] = True
        return result

    def _fixed_m4_retired_forward(
        self,
        input_ids,
        *,
        cache: Any,
        committed_count: int | None,
        hidden_variant: str | None,
    ):
        """The eager verify forward through an installed bank (lane retired)."""

        self.reserve_fixed_m4_window(
            cache,
            committed_count=committed_count,
            window_tokens=_decode_length(input_ids),
        )
        self.last_dispatch_kind = "eager"
        self.last_fallback_reason = "fixed_m4_dispatch_retired"
        self.last_fallback_transition = False
        self.stats["fallback_calls"] = int(self.stats.get("fallback_calls", 0)) + 1
        _note_demotion("fixed_m4_uncompiled_round", _FIXED_M4_RETIRED_ROUND_REASON)
        return self._runtime_forward(
            input_ids,
            cache=cache,
            return_hidden=True,
            hidden_variant=hidden_variant,
        )

    # -- parity2 on the installed fixed-M4 lane ---------------------------------
    #
    # MTPLX_COMPILED_VERIFY=parity2 on this lane: the installed bank stays
    # authoritative (the stream advances on its state) and every forward that
    # runs over it, the compiled width-4 replay and the eager short windows
    # alike, is replayed once more the way the EAGER VERIFIER would have run
    # it: on a stock-cache copy of the pre-round state, in the request context
    # the caller has open. Logits, hidden rows, captured rows and every state
    # leaf are compared on the device; only scalars cross to the host unless
    # a round diverges. A measuring instrument: it copies the whole QSA state
    # every round and blocks on it, so it never runs in a product lane.
    #
    # The reference takes its positions from the CALLER'S scope (the
    # generation loop opens it from the request's own splice), never from the
    # delta this bank was handed. The two are independent on purpose: a wrong
    # delta at the admission, or one baked into a trace, shows up as a
    # divergence instead of being mirrored by its own reference.

    def _named_fixed_m4_captures(self, captures_flat) -> dict[str, Any]:
        named: dict[str, Any] = {}
        pos = 0
        for idx, names in self._extra_capture_layout:
            for key_name in names:
                named[f"capture[{idx}].{key_name}"] = captures_flat[pos]
                pos += 1
        return named

    def _named_eager_captures(self, captures: Any) -> dict[str, Any]:
        named: dict[str, Any] = {}
        if not isinstance(captures, dict):
            return named
        for idx, names in self._extra_capture_layout:
            layer_capture = captures.get(idx) or {}
            for key_name in names:
                named[f"capture[{idx}].{key_name}"] = layer_capture.get(key_name)
        return named

    def _fixed_m4_state_leaves(self) -> list[Any]:
        """The installed plan's live leaves, in replay output order."""

        leaves: list[Any] = []
        for kind, entry, n_leaves in self._fixed_m4_dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                leaves.extend(
                    (
                        entry.kv.cache[0],
                        entry.kv.cache[1],
                        entry.kv.cache[2],
                        entry.raw_keys,
                        entry.pooled,
                    )
                )
            else:
                leaves.extend(entry.cache[:n_leaves])
        return leaves

    def _fixed_m4_parity2_clone(self, cache: Any) -> list[Any]:
        """The pre-round state as the eager verifier would hold it.

        Leaf copies of every entry (``_parity2_clone_cache``), with each fixed
        QSA bank demoted to the stock ``QSACache``: a host offset, no rotary
        delta, the stock gather. Materialized at once, because the forward
        that follows may write the source buffers in place.
        """

        clone = self._parity2_clone_cache(cache, 0)
        for idx, entry in enumerate(clone):
            if isinstance(entry, TensorOffsetQSACache):
                clone[idx] = entry.demote()
        leaves: list[mx.array] = []
        for entry in clone:
            if entry is None:
                continue
            kv = getattr(entry, "kv", None)
            if kv is not None:
                leaves.extend(
                    leaf
                    for leaf in (kv.keys, kv.values, entry.raw_keys, entry.pooled)
                    if isinstance(leaf, mx.array)
                )
            else:
                leaves.extend(
                    leaf for leaf in entry.cache if isinstance(leaf, mx.array)
                )
        if leaves:
            mx.eval(*leaves)
        return clone

    def _fixed_m4_parity2_record(self, dispatch) -> dict[str, Any]:
        """This request's parity receipt (``compiled_verify.fixed_m4_parity2``)."""

        rope_delta = dispatch["rope_delta"]
        return self.stats.setdefault(
            "fixed_m4_parity2",
            {
                "positions": "text" if rope_delta is None else "vision_delta",
                "rope_delta": rope_delta,
                # Rounds compared, and how many of them differed anywhere.
                "rounds": 0,
                "rounds_by_width": {},
                # A width-4 call can still run eager without host inputs.
                # Count the dispatch that actually produced the candidate.
                "compiled_rounds": 0,
                "eager_rounds": 0,
                "compiled_exact_rounds": 0,
                "eager_exact_rounds": 0,
                "divergent_rounds": 0,
                "divergent_rounds_by_width": {},
                # Rounds of an image request NOT compared, because their
                # caller had no request scope open (see
                # ``_fixed_m4_parity2_reference_ready``).
                "reference_scope_missing_rounds": 0,
                "reference_scope_missing_by_width": {},
                "compared_leaves_per_round": 0,
                "logits_max_abs_diff": 0.0,
                "logits_max_kl": 0.0,
                "hidden_max_abs_diff": 0.0,
                "state_max_abs_diff": 0.0,
                "capture_max_abs_diff": 0.0,
                "kl_scope": "logits_only",
                "state_metrics": ["max_abs_diff", "differing_elements"],
                "round_diagnostics": [],
                "first_divergence": None,
            },
        )

    def _fixed_m4_parity2_reference_ready(self, dispatch, input_ids) -> bool:
        """Whether this round has a reference it can be compared with.

        The reference of an image round is the stock cache roped from the
        REQUEST's scope, the one the generation loop opens from the splice.
        It is read from the caller's context, here on the host in the
        instrument (the fixed lane itself never reads it), so it cannot
        depend on the delta under test: a wrong delta at the admission would
        otherwise be compared with itself and pass. A caller with no scope
        open leaves nothing independent to compare with. Its reference would
        rotate at text positions and the round would read as divergent
        whatever the lane did, so it is counted and not compared, and the
        maxima below stay the maxima of real comparisons.
        """

        if dispatch["rope_delta"] is None:
            return True
        from .attention_context import vision_rope_state

        if vision_rope_state() is not None:
            return True
        record = self._fixed_m4_parity2_record(dispatch)
        width = str(_decode_length(input_ids))
        record["reference_scope_missing_rounds"] += 1
        record["reference_scope_missing_by_width"][width] = (
            record["reference_scope_missing_by_width"].get(width, 0) + 1
        )
        return False

    def _fixed_m4_parity2_round(self, dispatch, cache, input_ids, **candidate) -> None:
        if not self._fixed_m4_parity2_reference_ready(dispatch, input_ids):
            return
        self._fixed_m4_parity2_compare(
            dispatch, self._fixed_m4_parity2_clone(cache), input_ids, **candidate
        )

    def _fixed_m4_parity2_compare(
        self,
        dispatch,
        clone: list[Any],
        input_ids,
        *,
        compiled_aux,
        candidate_logits,
        candidate_hidden,
        candidate_captures: dict[str, Any],
        candidate_state: list[Any],
        committed_count: int | None,
        dispatch_kind: str,
    ) -> None:
        # The reference leg: the stock cache in the caller's ambient scope
        # (``_fixed_m4_parity2_reference_ready`` checked that an image round
        # has one).
        with attention_phase("decode_verify"):
            eager_logits, eager_hidden, eager_captures = self._runtime_forward(
                input_ids,
                cache=clone,
                return_hidden=True,
                hidden_variant=dispatch["hidden_variant"],
                compiled_aux=compiled_aux,
            )

        # (name, reference leaf, candidate leaf); the two lanes may hold
        # different capacities, but both must store and compare ALL live rows.
        pairs: list[tuple[str, Any, Any]] = [
            ("logits", eager_logits, candidate_logits),
            ("hidden", eager_hidden, candidate_hidden),
        ]
        reference_captures = self._named_eager_captures(eager_captures)
        for name in sorted(set(reference_captures) | set(candidate_captures)):
            pairs.append(
                (name, reference_captures.get(name), candidate_captures.get(name))
            )
        offsets: list[tuple[str, int, int]] = []
        storage_shortfalls: dict[str, dict[str, Any]] = {}
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            entry = clone[idx]
            prefix = f"state[{idx}:{kind}]"
            if kind == VERIFY_SPEC_KIND_QSA:
                reference = (
                    entry.kv.keys, entry.kv.values, None, entry.raw_keys, entry.pooled
                )
                reference_offset = int(entry.kv.offset)
                candidate_offset = int(candidate_state[pos + 2].item())
                offsets.append((f"{prefix}.2", reference_offset, candidate_offset))
                for leaf_idx, axis in ((0, 2), (1, 2), (3, 1), (4, 1)):
                    ref = reference[leaf_idx]
                    cand = candidate_state[pos + leaf_idx]
                    name = f"{prefix}.{leaf_idx}"
                    divisor = int(entry.ratio) if leaf_idx == 4 else 1
                    ref_rows = reference_offset // divisor
                    cand_rows = candidate_offset // divisor
                    if ref.shape[axis] < ref_rows or cand.shape[axis] < cand_rows:
                        storage_shortfalls[name] = {
                            "reason": "live_storage_shortfall",
                            "reference_storage_rows": int(ref.shape[axis]),
                            "reference_live_rows": ref_rows,
                            "candidate_storage_rows": int(cand.shape[axis]),
                            "candidate_live_rows": cand_rows,
                        }
                    ref_index = [slice(None)] * ref.ndim
                    cand_index = [slice(None)] * cand.ndim
                    ref_index[axis] = slice(0, ref_rows)
                    cand_index[axis] = slice(0, cand_rows)
                    pairs.append(
                        (name, ref[tuple(ref_index)], cand[tuple(cand_index)])
                    )
            else:
                for leaf_idx in range(n_leaves):
                    pairs.append(
                        (
                            f"{prefix}.{leaf_idx}",
                            entry.cache[leaf_idx],
                            candidate_state[pos + leaf_idx],
                        )
                    )
            pos += n_leaves

        comparable = [
            (name, ref, cand)
            for name, ref, cand in pairs
            if name not in storage_shortfalls
            and isinstance(ref, mx.array)
            and isinstance(cand, mx.array)
            and ref.shape == cand.shape
            and ref.dtype == cand.dtype
        ]
        # Retain scalar diagnostics for every leaf of every compared round.
        # No cache arrays cross to the host, even when many leaves diverge.
        measurements = {}
        for name, ref, cand in comparable:
            # Integer leaves must not lose differences above float32's exact
            # integer range. The logical QSA offset is measured as a host int.
            dtype = mx.int64 if mx.issubdtype(ref.dtype, mx.integer) else mx.float32
            measurements[name] = (
                mx.max(mx.abs(ref.astype(dtype) - cand.astype(dtype)))
                if ref.size else mx.array(0.0),
                mx.sum(ref != cand),
            )
        ref32 = eager_logits.astype(mx.float32)
        cand32 = candidate_logits.astype(mx.float32)
        log_p = ref32 - mx.logsumexp(ref32, axis=-1, keepdims=True)
        log_q = cand32 - mx.logsumexp(cand32, axis=-1, keepdims=True)
        max_kl = mx.max(mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1))
        mx.eval(list(measurements.values()), max_kl)
        leaves: dict[str, dict[str, Any]] = {}
        for name, ref, cand in pairs:
            if name in storage_shortfalls:
                metric = {
                    "max_abs_diff": None, "differing_elements": None,
                    **storage_shortfalls[name],
                }
            elif name in measurements:
                diff, count = measurements[name]
                differing = int(count.item())
                metric = {
                    "max_abs_diff": float(diff.item()),
                    "differing_elements": differing,
                    "reason": "value_mismatch" if differing else None,
                }
            elif ref is None and cand is None:
                metric = {"max_abs_diff": 0.0, "differing_elements": 0, "reason": None}
            else:
                if ref is None or cand is None:
                    reason = "missing_leaf"
                elif ref.shape != cand.shape:
                    reason = "shape_mismatch"
                else:
                    reason = "dtype_mismatch"
                metric = {
                    "max_abs_diff": None, "differing_elements": None, "reason": reason,
                }
            leaves[name] = metric
        for name, ref, cand in offsets:
            leaves[name] = {
                "max_abs_diff": abs(ref - cand),
                "differing_elements": int(ref != cand),
                "reason": "offset_mismatch" if ref != cand else None,
            }
        leaves["logits"]["max_kl"] = float(max_kl.item())
        mismatched = [name for name, metric in leaves.items() if metric["reason"]]

        width = str(_decode_length(input_ids))
        record = self._fixed_m4_parity2_record(dispatch)
        record["rounds"] += 1
        record["rounds_by_width"][width] = record["rounds_by_width"].get(width, 0) + 1
        record[f"{dispatch_kind}_rounds"] += 1
        record["compared_leaves_per_round"] = len(pairs) + len(offsets)
        context = (
            int(dispatch["base_offset"]) + int(committed_count)
            if committed_count is not None else None
        )
        record["round_diagnostics"].append({
            "round": int(record["rounds"]), "width": int(width),
            "dispatch_kind": dispatch_kind, "context": context, "leaves": leaves,
        })
        for name, metric in leaves.items():
            family = _artifact_kind(name)
            if family in ("logits", "hidden", "state", "capture"):
                diff = metric["max_abs_diff"]
                if diff is not None:
                    key = f"{family}_max_abs_diff"
                    record[key] = max(record[key], diff)
        record["logits_max_kl"] = max(record["logits_max_kl"], float(max_kl.item()))
        self.stats["parity2_calls"] += 1
        if not mismatched:
            record[f"{dispatch_kind}_exact_rounds"] += 1
            return

        record["divergent_rounds"] += 1
        record["divergent_rounds_by_width"][width] = (
            record["divergent_rounds_by_width"].get(width, 0) + 1
        )
        self.stats["parity2_divergent_calls"] += 1
        if record["first_divergence"] is None:
            report = [f"{name}: {leaves[name]}" for name in mismatched]
            first_name = mismatched[0]
            record["first_divergence"] = {
                "round": int(record["rounds"]),
                "width": int(width),
                "context": context,
                "leaf": first_name,
                "reason": leaves[first_name]["reason"],
                "artifact": _artifact_kind(first_name),
                "mismatched_leaves": len(mismatched),
                "report": report,
            }
            self.stats["parity2_first_divergence"] = {
                "call": int(self.stats["calls"]),
                "context": record["first_divergence"]["context"],
                "artifact": _artifact_kind(first_name),
                "leaf": first_name,
                "reason": leaves[first_name]["reason"],
                "max_abs_diff": leaves[first_name]["max_abs_diff"],
                "mismatched_leaves": len(mismatched),
            }
        if record["divergent_rounds"] <= 10:
            print(
                f"[parity2] fixed-M4 divergence round={record['rounds']} "
                f"width={width} positions={record['positions']} "
                f"leaf={mismatched[0]} mismatched_leaves={len(mismatched)} "
                f"logits_max_abs_diff={leaves['logits']['max_abs_diff']} "
                f"kl={float(max_kl.item()):.3e}",
                flush=True,
            )

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
        extended_window: bool = False,
        committed_count: int | None = None,
    ):
        """Compiled verify dispatch.

        ``extended_window`` (context-copy block rounds, 2026-08-26 v2) admits
        lengths above ``max_verify_len`` up to ``MTPLX_CCOPY_BANK_MAX_LEN``.
        The extended lane changes ROUTING only, never the request's memory
        contract: the speculative reserve stays keyed to ``max_verify_len``,
        a dense-capacity preflight refuses (falls back eager) instead of
        growing a granted KV leaf, and paged capacity overflow falls back as
        before.  ``_paged_ineligibility`` is skipped for extended lengths:
        that gate is a performance router for windows whose eager alternative
        is cheap, while a block round's eager alternative costs ~380 ms flat
        at long context (MEASUREMENTS 2026-08-25 11:26) — inside the traced
        graph the paged kernel declining is shape-deterministic and routes to
        the same dense math the eager forward takes at the same T, so
        exactness is unaffected either way.
        """
        global _PREWARM_DONE
        if self._fixed_m4_dispatch is not None:
            self.stats["calls"] += 1
            self.reserve_fixed_m4_window(
                cache,
                committed_count=committed_count,
                window_tokens=_decode_length(input_ids),
            )
            if _decode_length(input_ids) == 4:
                # Without host-owned n-gram inputs (any caller other than the
                # generation loop's forward_fixed_m4 entrypoint, for example a
                # bank-routed copy block of the same width) the installed
                # replay cannot run. Main's bank contract is to refuse
                # internally and take the identical runtime forward, so route
                # the call eager and let the Route Tape show it.
                self.last_dispatch_kind = "eager"
                self.last_fallback_reason = "fixed_m4_host_inputs_missing"
                self.last_fallback_transition = False
                _note_demotion("fixed_m4_uncompiled_round", _FIXED_M4_NO_HOST_INPUTS_REASON)
            else:
                # Shorter adaptive windows take the normal family capture
                # route. Not a fallback (no counter), but the Route Tape must
                # still see that this call ran eager through the bank.
                self.last_dispatch_kind = "eager"
                self.last_fallback_reason = "fixed_m4_short_window"
                self.last_fallback_transition = False
                _note_demotion("fixed_m4_uncompiled_round", _FIXED_M4_OTHER_WIDTH_REASON)
            # Either way the forward runs eager ON the promoted bank, where an
            # image request rotates at the bank's own origin. parity2 checks
            # that forward too, against the stock cache in the request scope.
            parity2_clone = (
                self._fixed_m4_parity2_clone(cache)
                if self._fixed_m4_dispatch.get("parity2")
                and return_hidden
                and self._fixed_m4_parity2_reference_ready(
                    self._fixed_m4_dispatch, input_ids
                )
                else None
            )
            result = self._runtime_forward(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
            if parity2_clone is not None:
                self._fixed_m4_parity2_compare(
                    self._fixed_m4_dispatch,
                    parity2_clone,
                    input_ids,
                    compiled_aux=None,
                    candidate_logits=result[0],
                    candidate_hidden=result[1],
                    candidate_captures=self._named_eager_captures(result[2]),
                    candidate_state=self._fixed_m4_state_leaves(),
                    committed_count=committed_count,
                    dispatch_kind="eager",
                )
            return result
        if (
            not _PREWARM_DONE
            and not self.parity
            and not self.parity2
            and not extended_window
            and _prewarm_enabled()
        ):
            # First compiled dispatch of a generation while coverage is
            # incomplete (the first one of the process is normally the
            # startup warmup generation): walk the PAGED bucket ladder so
            # those graphs (and their Metal pipelines) exist before any
            # user-facing generation — paged bucket crossings were the bulk
            # of the −28% unrouted long-form cost (MEASUREMENTS 2026-07-02).
            # On the dense path this is a deliberate no-op
            # ("no_paged_entries", marks the walk complete): dense KV
            # retraces every 256 tokens of growth (5 traces per 1.3k-token
            # chat answer, measured 2026-07-02 21:25) and pre-walking ~24
            # shape classes to 6k is startup-prohibitive — the designed fix
            # there is pow2-bucketized dense leaves, not a longer prewarm.
            # F6 (2026-08-16): a walk CLAMPED by the current cache's paged
            # capacity (the 16-token boot warmup) no longer spends the
            # one-shot — later generations with more capacity (the server
            # warmup ladder rungs) extend the walk over the still-missing
            # buckets, so their compiles land in warmup, not in measured
            # rows. The walk is best-effort by design: a failure is
            # recorded visibly and the organic dispatch below handles the
            # same condition through its own fallback accounting.
            try:
                report = self.prewarm_ladder(
                    cache, input_ids, hidden_variant=hidden_variant
                )
            except Exception as exc:  # visible, never fatal (see docstring)
                report = {
                    "buckets": [],
                    "skipped": [f"walk_error:{type(exc).__name__}"],
                    "elapsed_s": 0.0,
                    "complete": False,
                }
            self.stats["prewarm"] = report
            _PREWARM_DONE = bool(report.get("complete"))
            prewarm_status["done"] = _PREWARM_DONE
            prewarm_status["walks"] = int(prewarm_status.get("walks", 0)) + 1
            prewarm_status["last_report"] = report
            prewarm_status["buckets"] = sorted(
                {bucket for _rt, _len, _var, bucket in _PREWARMED_BUCKETS}
            )
            if report.get("buckets") or int(prewarm_status["walks"]) == 1:
                # One line per walk that actually compiled something (plus
                # the first walk of the process); silent no-op retries stay
                # off the console.
                try:
                    import json as _json

                    print(
                        "[mtplx] compiled-verify prewarm " + _json.dumps(report),
                        flush=True,
                    )
                except Exception:
                    pass
        self.last_dispatch_kind = "compiled"
        self.last_fallback_reason = None
        self.last_fallback_transition = False
        self.stats["calls"] += 1
        reason = self._fallback_reason(
            input_ids, cache, return_hidden, extended_window=extended_window
        )
        if reason is not None:
            return self._fallback(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=reason,
            )
        length = _decode_length(input_ids)
        try:
            bucket = self._resolve_bucket(cache, length)
            if bucket is None:
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="capacity_overflow",
                )
            max_ctx = _compiled_verify_max_context()
            if max_ctx and getattr(self, "_last_context_estimate", 0) > max_ctx:
                # Context-scaled router: compiled verify is proven bit-exact
                # and +4.8% only up to ~6k ctx; beyond, eager wins and the
                # exactness corpus has no coverage. Fall back per call.
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="context_above_threshold",
                )
            if not (extended_window and length > self.max_verify_len):
                # Extended block windows skip this performance router (see
                # the method docstring); every other call keeps it verbatim.
                ineligible = self._paged_ineligibility(cache, length, bucket)
                if ineligible is not None:
                    return self._fallback(
                        input_ids,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                        reason=ineligible,
                    )
            self._ensure_shadow(cache)
            self._apply_bucket(cache, bucket)
            # Boundary policy (experiment knob, 2026-07-02 sprint):
            #   pre  — materialize pending input state with the eager kernels
            #          before entering the compiled function. Exactness
            #          boundary: a lazy upstream graph absorbed into compiled
            #          execution computes with fused-kernel numerics (~1e-6),
            #          breaking bit-parity with the eager reference.
            #   post — schedule evaluation of outputs while the input leaves
            #          are still referenced by the real cache. Buffer-safety
            #          boundary: without it, mirror-commit drops the last
            #          input references while the compiled graph is pending
            #          and the allocator reuses their buffers.
            # MTPLX_COMPILED_VERIFY_BOUNDARY = both (default) | pre | post |
            # none. When 'post' is dropped, buffer safety is preserved by
            # holding the input references until the NEXT dispatch instead
            # (self._held_state_refs) — no numerics cost, no forced batch.
            boundary = _compiled_verify_boundary()
            donate = (
                _compiled_verify_donation_enabled()
                and not self.parity
                and not self.parity2
                and boundary in ("both", "post")
            )
            if donate:
                # A2.1: the shadow twins hold promotion-time leaf refs that
                # (a) pin one full stale KV buffer set for the generation and
                # (b) alias the first call's input buffers, blocking their
                # donation. The traced body re-seeds every slot from the
                # explicit inputs before any read, so the held refs are dead.
                self._clear_shadow_leaf_refs()
            key = self._verify_key(length, hidden_variant, bucket)
            fn = self._verify_program(key, length, hidden_variant)
            state_in = self._read_state_leaves(cache)
            if state_in is None:
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="empty_state_leaf",
                )
            compiled_aux = (
                self._prepare_compiled_aux(input_ids, cache)
                if self._prepare_compiled_aux is not None
                else None
            )
            if boundary in ("both", "pre"):
                mx.async_eval(
                    *((compiled_aux,) if compiled_aux is not None else ()),
                    *state_in,
                )
            # Same positional contract as the installed replay: ids, the
            # auxiliary when the runtime prepares one, the rotary delta of an
            # image request, the state leaves.
            with compiled_dispatch((id(fn), tuple(getattr(input_ids, "shape", ())))):
                outputs = (
                    fn(input_ids, compiled_aux, *self._rope_args(), *state_in)
                    if compiled_aux is not None
                    else fn(input_ids, *self._rope_args(), *state_in)
                )
            logits, hidden, captures_flat, state_out = self._unpack_outputs(outputs)
            if donate:
                # A2.1 commit-first ownership handoff — commit + schedule
                # happen AFTER this fallback-safe block (see below): once the
                # real cache is rebound to the outputs, an eager fallback
                # would double-apply the verify window.
                pass
            elif boundary in ("both", "post"):
                mx.async_eval(*outputs)
                self._held_state_refs.clear()
            else:
                # Keep inputs alive across a 3-generation window: with the
                # deferred serve path, call N-1's graph may still be pending
                # when call N dispatches, so a single-slot hold can release
                # buffers the allocator then reuses. Three generations covers
                # the deepest deferred chain the serve path produces
                # (experiment probe; production would release on evidence).
                self._held_state_refs.append(state_in)
                if len(self._held_state_refs) > 3:
                    self._held_state_refs.pop(0)
        except Exception as exc:
            self._exception_failures += 1
            compiled_verify_status["transient_exception_count"] = (
                int(compiled_verify_status.get("transient_exception_count", 0)) + 1
            )
            if self._exception_failures >= 3:
                self.permanent_eager = True
                self.permanent_eager_reason = (
                    f"exception_streak:{type(exc).__name__}"
                )
                _record_permanent_eager(self.permanent_eager_reason)
            return self._fallback(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=f"exception:{type(exc).__name__}",
            )
        self._exception_failures = 0
        self.stats["compiled_calls"] += 1
        if extended_window and length > self.max_verify_len:
            self.stats["extended_calls"] += 1
        bucket_key = str(int(bucket))
        self.stats["buckets"][bucket_key] = self.stats["buckets"].get(bucket_key, 0) + 1
        captures = self._rebuild_captures(captures_flat)
        if self.parity:
            return self._parity_check(
                input_ids,
                cache=cache,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
                state_in=state_in,
                compiled_logits=logits,
                compiled_hidden=hidden,
                compiled_captures=captures,
                compiled_state_out=state_out,
            )
        if self.parity2:
            return self._parity2_check(
                input_ids,
                cache=cache,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
                bucket=int(bucket),
                compiled_logits=logits,
                compiled_hidden=hidden,
                compiled_captures=captures,
                compiled_state_out=state_out,
            )
        self._mirror_commit(cache, state_out)
        if self._commit_compiled_captures is not None:
            self._commit_compiled_captures(cache, captures)
        if donate:
            # A2.1 commit-first ownership handoff: the real cache is already
            # rebound to the output leaves, so dropping the dispatcher's
            # ``state_in`` list makes the pending graph the ONLY holder of
            # each input KV buffer at schedule time.  MLX then donates the
            # buffer into the in-graph ``slice_update`` instead of
            # materializing a full copy of every full-attn K/V buffer per
            # verify call (measured 16.5 ms @64k, ~33 ms @128k — probe arms
            # A vs G, outputs/ivanbench-20260705/compiled_copy_tax_probe.py).
            # Byte-exactness across chained pending calls and snapshot-COW
            # pinning proven in compiled_copy_tax_correctness.py; buffers
            # shared with a bank entry (restore/postcommit views) simply COW
            # once, exactly as before.  (A freshly built shadow still holds
            # the promotion-time leaves, so the first call of a generation
            # pays one copy; calls 2+ donate because the shadow's stale refs
            # never alias the current inputs.)
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)
        return logits, hidden, captures

    def prewarm_ladder(
        self,
        cache: Any,
        input_ids,
        *,
        hidden_variant: str | None = None,
        max_context: int | None = None,
    ) -> dict[str, Any]:
        """Compile-and-execute the verify step once per pow2 bucket up to
        the router boundary, priming the Metal shader cache.

        Outputs are discarded and state is never committed (`verify_step`
        is a pure function of its state leaves), so the caller's cache is
        untouched apart from the static bucket ceiling, which is restored
        to its natural value before returning. Failures are recorded per
        bucket and never flip ``permanent_eager`` — a bucket that cannot
        prewarm simply pays its organic compile later.

        ``report["complete"]`` is the one-shot verdict (F6): True when no
        future walk could add coverage (the ladder reached the router
        ceiling, or the cache is structurally ladder-free), False when the
        walk was clamped by the current cache's paged capacity or skipped
        for a transient reason — the trigger then retries on a later
        generation whose cache reaches further. Buckets warmed by earlier
        walks are skipped (``report["already"]``), so a retry with nothing
        new to add costs a few python comparisons.
        """
        report: dict[str, Any] = {
            "buckets": [],
            "skipped": [],
            "already": [],
            "elapsed_s": 0.0,
            "complete": False,
        }
        started = time.perf_counter()

        def _finish() -> dict[str, Any]:
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            return report

        if self.permanent_eager:
            # Structural for this process/model (quant gate) or already a
            # terminal degradation — nothing a later walk could add.
            report["skipped"].append("permanent_eager")
            report["complete"] = True
            return _finish()
        reason = self._fallback_reason(
            input_ids, cache, True, consume_post_restore=False
        )
        if reason is not None:
            report["skipped"].append(reason)
            return _finish()
        length = _decode_length(input_ids)
        try:
            natural = self._resolve_bucket(cache, length)
        except Exception as exc:
            report["skipped"].append(f"resolve:{type(exc).__name__}")
            return _finish()
        if not natural:
            report["skipped"].append(
                "capacity_overflow" if natural is None else "no_paged_entries"
            )
            # Dense caches have no paged bucket ladder by design (see the
            # trigger comment): the walk is complete, not clamped.
            report["complete"] = natural is not None
            return _finish()
        boundary = (
            int(max_context)
            if max_context is not None
            else _compiled_verify_max_context()
        )
        if boundary <= 0:
            # Router disabled: only the natural bucket is reachable cheaply;
            # deeper buckets appear at unbounded context growth and warming
            # them all is unbounded work.
            boundary = int(natural)
        min_capacity: int | None = None
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if hasattr(entry, "capacity"):
                cap = int(entry.capacity)
                min_capacity = cap if min_capacity is None else min(min_capacity, cap)
        ceiling = _next_pow2(boundary + length + 512)
        if int(natural) > ceiling:
            # This call's context is already above the compiled-verify
            # router: every dispatch of this generation falls back per call
            # ("context_above_threshold"), so walking (and compiling) its
            # bucket would burn ~1s on a graph no compiled row can use.
            report["skipped"].append("context_above_router")
            return _finish()
        ladder: list[int] = []
        bucket = int(natural)
        while True:
            if min_capacity is not None:
                bucket = min(bucket, min_capacity)
            if bucket not in ladder:
                ladder.append(bucket)
            if min_capacity is not None and bucket >= min_capacity:
                break
            if bucket >= ceiling:
                break
            bucket *= 2
        # Complete = the ladder reached the router ceiling. A walk clamped
        # below it by min_capacity leaves the one-shot unspent so a later,
        # larger cache (server warmup ladder rungs) extends the coverage.
        report["complete"] = bool(ladder) and int(ladder[-1]) >= ceiling
        variant_key = str(hidden_variant or "")
        runtime_id = id(self.runtime)
        pending = [
            bucket
            for bucket in ladder
            if (runtime_id, length, variant_key, int(bucket))
            not in _PREWARMED_BUCKETS
        ]
        report["already"] = [
            int(bucket) for bucket in ladder if bucket not in pending
        ]
        if not pending:
            return _finish()
        self._ensure_shadow(cache)
        state_in = self._read_state_leaves(cache)
        if state_in is None:
            report["skipped"].append("empty_state_leaf")
            report["complete"] = False
            return _finish()
        for bucket in pending:
            if self._paged_ineligibility(cache, length, bucket) is not None:
                report["skipped"].append(f"b{bucket}:paged_kernel_ineligible")
                continue
            try:
                self._apply_bucket(cache, bucket)
                key = self._verify_key(length, hidden_variant, bucket)
                # Shared-registry compile (F6): a bare per-bank mx.compile
                # primed the Metal pipelines but kept the trace private to the
                # warmup bank, so the first real request at the same shapes
                # re-traced every bucket (~1s each) inside its measured row.
                # The shared step is exactly what organic dispatch consults.
                fn = self._verify_program(key, length, hidden_variant)
                bucket_started = time.perf_counter()
                outputs = fn(input_ids, *self._rope_args(), *state_in)
                # Synchronous eval: the compile cost is paid HERE, and no
                # graph is left pending, so no held-reference bookkeeping
                # is needed. Outputs are dropped, never committed.
                mx.eval(*outputs)
                report["buckets"].append(
                    {
                        "bucket": int(bucket),
                        "s": round(time.perf_counter() - bucket_started, 3),
                    }
                )
                _PREWARMED_BUCKETS.add((runtime_id, length, variant_key, int(bucket)))
            except Exception as exc:
                report["skipped"].append(f"b{bucket}:{type(exc).__name__}")
        try:
            restored = self._resolve_bucket(cache, length)
            if restored:
                self._apply_bucket(cache, restored)
        except Exception:
            pass
        return _finish()

    def prewarm_extended_lengths(
        self,
        cache: Any,
        lengths: list[int],
        *,
        hidden_variant: str | None = None,
    ) -> dict[str, Any]:
        """Trace the extended (context-copy block) windows ahead of use.

        Optional, driven by ``MTPLX_CCOPY_BANK_PREWARM`` at the ccopy site.
        Without it the first block round per (length, bucket) pays the fresh
        ``mx.compile`` trace organically — once per PROCESS (shared traces),
        which was the recurring-looking "~240 ms/call" in the 8-round v1 cell
        (one ~1s first-trace amortized over 8 rounds; the dispatch layer
        itself re-clones nothing, probe receipts 2026-08-26). A/B cells that
        time steady-state block rounds should enable this so first-trace cost
        lands in warmup, not in a measured row.

        Same firewall economics as ``prewarm_ladder``: the compiled function
        is pure, outputs are dropped and never mirror-committed, so the live
        cache is untouched. The dry run cannot donate its input buffers (the
        real cache still holds every leaf), so each traced length transiently
        materializes one copy of the full-attn KV set — the same one-time
        copy the first organic call of a generation pays.
        """
        report: dict[str, Any] = {"lengths": [], "skipped": [], "elapsed_s": 0.0}
        started = time.perf_counter()

        def _finish() -> dict[str, Any]:
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            self.stats["extended_prewarm"] = report
            return report

        if self.permanent_eager or self.parity or self.parity2:
            report["skipped"].append("bank_mode")
            return _finish()
        ceiling = max(self.max_verify_len, _ccopy_bank_max_len())
        variant_key = str(hidden_variant or "")
        runtime_id = id(self.runtime)
        for length in sorted({int(item) for item in lengths}):
            if length <= self.max_verify_len or length > ceiling:
                report["skipped"].append(f"m{length}:outside_extended_window")
                continue
            probe = mx.zeros((1, length), dtype=mx.int32)
            reason = self._fallback_reason(
                probe,
                cache,
                True,
                consume_post_restore=False,
                extended_window=True,
            )
            if reason is not None:
                report["skipped"].append(f"m{length}:{reason}")
                continue
            try:
                bucket = self._resolve_bucket(cache, length)
                if bucket is None:
                    report["skipped"].append(f"m{length}:capacity_overflow")
                    continue
                if (
                    runtime_id,
                    length,
                    variant_key,
                    int(bucket),
                ) in _PREWARMED_BUCKETS:
                    report["skipped"].append(f"m{length}:already")
                    continue
                self._ensure_shadow(cache)
                self._apply_bucket(cache, bucket)
                state_in = self._read_state_leaves(cache)
                if state_in is None:
                    report["skipped"].append(f"m{length}:empty_state_leaf")
                    continue
                key = self._verify_key(length, hidden_variant, bucket)
                fn = self._verify_program(key, length, hidden_variant)
                length_started = time.perf_counter()
                outputs = fn(probe, *self._rope_args(), *state_in)
                mx.eval(*outputs)
                report["lengths"].append(
                    {
                        "m": int(length),
                        "bucket": int(bucket),
                        "s": round(time.perf_counter() - length_started, 3),
                    }
                )
                _PREWARMED_BUCKETS.add((runtime_id, length, variant_key, int(bucket)))
            except Exception as exc:
                report["skipped"].append(f"m{length}:{type(exc).__name__}")
        try:
            # Politeness restore (mirrors prewarm_ladder): an eager forward
            # between this walk and the next dispatch should see a natural
            # static ceiling, not the last extended length's. Any ceiling
            # >= offset + T is topology-valid, and dispatch re-applies its
            # own bucket before every compiled call.
            restored = self._resolve_bucket(cache, 1)
            if restored:
                self._apply_bucket(cache, restored)
        except Exception:
            pass
        return _finish()

    def _materialize_growth_handoff_state(self, cache: Any) -> int:
        """Settle compiled state before the eager tail takes ownership.

        Compiled dispatch schedules every output asynchronously. Merely
        replacing the tensor-offset cache containers leaves their KV and
        recurrent leaves attached to that deferred graph. The eager tail then
        inherits the compiled dependency chain, so long generations pay the
        old work through later verify-output evaluations instead of crossing a
        clean ownership boundary.

        Growth demotion is a once-per-request transition. Evaluate the current
        state exactly once here, while the compiled state spec is still valid,
        then let ``demote`` replace the containers and release compiled refs.
        """
        state = self._read_state_leaves(cache)
        if state is None:
            raise RuntimeError(
                "compiled verify growth handoff has incomplete cache state"
            )
        leaves: list[mx.array] = []
        seen: set[int] = set()
        for leaf in state:
            if not isinstance(leaf, mx.array):
                continue
            identity = id(leaf)
            if identity in seen:
                continue
            seen.add(identity)
            leaves.append(leaf)
        started = time.perf_counter()
        if leaves:
            mx.eval(*leaves)
        self.stats["growth_handoff_materializations"] = (
            int(self.stats.get("growth_handoff_materializations", 0)) + 1
        )
        self.stats["growth_handoff_state_leaves"] = (
            int(self.stats.get("growth_handoff_state_leaves", 0)) + len(leaves)
        )
        self.stats["growth_handoff_materialize_time_s"] = float(
            self.stats.get("growth_handoff_materialize_time_s", 0.0)
        ) + (time.perf_counter() - started)
        return len(leaves)

    def demote(
        self, cache: Any, *, compact: bool = True, keep_capacity: bool = False,
    ) -> int:
        """Restore stock containers for every tensor-offset adapter in place.

        Mandatory before postcommit / final-state capture: downstream cache
        consumers must never see promoted adapters. Skip publication copies
        when the caller will discard the state. ``keep_capacity`` hands the
        fixed QSA banks back whole, for the one-copy conversation store
        (``TensorOffsetQSACache.demote``).
        """
        try:
            from .cache_state import TensorOffsetVllmMetalPagedKVCache
        except Exception:  # pragma: no cover - import guard for minimal test envs
            TensorOffsetVllmMetalPagedKVCache = None
        # Retire all owners before conversion. Otherwise the installed plan
        # pins every old QSA bank while compact copies accumulate beside it.
        # Replacing cache entries below now releases each layer in turn.
        self._clear_shadow_leaf_refs()
        self._held_state_refs.clear()
        self._fixed_m4_dispatch = None
        self._shadow = None
        self._shadow_signature = None
        self._spec = None
        self._compiled.clear()
        self._program_hosts.clear()
        count = 0
        for idx, entry in enumerate(cache or []):
            if isinstance(entry, TensorOffsetQSACache):
                cache[idx] = entry.demote(compact=compact, keep_capacity=keep_capacity)
                count += 1
            elif isinstance(entry, TensorOffsetKVCache):
                cache[idx] = entry.demote()
                count += 1
            elif TensorOffsetVllmMetalPagedKVCache is not None and isinstance(
                entry, TensorOffsetVllmMetalPagedKVCache
            ):
                cache[idx] = entry.demote()
                count += 1
        if count:
            self.stats["demotions"] += count
        return count

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.stats)
        data["fallback_reasons"] = dict(self.stats["fallback_reasons"])
        data["buckets"] = dict(self.stats["buckets"])
        first_divergence = self.stats.get("parity2_first_divergence")
        data["parity2_first_divergence"] = (
            dict(first_divergence) if isinstance(first_divergence, dict) else None
        )
        if self.parity2:
            data["mode"] = "parity2"
        else:
            data["mode"] = "parity" if self.parity else "on"
        data["max_verify_len"] = self.max_verify_len
        data["request_max_tokens"] = self.request_max_tokens
        data["speculative_headroom"] = self.speculative_headroom
        data["growth_reserve_tokens"] = self.growth_reserve_tokens
        data["capture_backend"] = self.capture_backend
        data["permanent_eager"] = self.permanent_eager
        data["permanent_eager_reason"] = getattr(
            self, "permanent_eager_reason", None
        )
        data["compiled_entry_count"] = len(self._compiled)
        data["compiled_keys"] = [
            f"m{length}:{variant or 'default'}:b{bucket}"
            + (":rope_delta" if rope_delta_input else "")
            for length, variant, bucket, rope_delta_input in sorted(self._compiled)
        ]
        # The generic lane's own receipt: an image request replays the trace
        # that takes the rotary delta as an input; a text request the text
        # trace. (The fixed-M4 lane repeats it under ``fixed_m4`` below.)
        data["rope_delta_input"] = self._rope_delta is not None
        data["rope_delta"] = self._rope_delta_value
        parity2 = self.stats.get("parity2")
        if isinstance(parity2, dict):
            data["parity2"] = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in parity2.items()
            }
        copy_windows = self.stats.get("fixed_m4_copy_windows")
        if isinstance(copy_windows, dict):
            # Compiled copy windows (MTPLX_FIXED_M4_COPY_WINDOWS=1): replays
            # and traces by width, and the copy rounds that ran eager, by why.
            data["fixed_m4_copy_windows"] = {
                **{key: dict(value) for key, value in copy_windows.items()},
                "retired": _FIXED_M4_COPY_WINDOWS["retired"],
            }
        dispatch = self._fixed_m4_dispatch
        if dispatch is not None:
            data["fixed_m4"] = {
                "installed": True,
                "aux_route": dispatch["aux_route"],
                "aux_inputs": dispatch["aux_inputs"],
                "kv_gather": dispatch["kv_gather"],
                "boundary": dispatch["boundary"],
                "donate": bool(dispatch["donate"]),
                "state_leaves": int(dispatch["state_leaves"]),
                "capture_leaves": int(dispatch["capture_leaves"]),
                # Growth receipts: the installed bank's prompt offset, its
                # current QSA capacity and the next grant. Read together with
                # the fixed_m4_capacity_transitions counter they prove that a
                # request outgrew its first grant and stayed on the lane.
                "base_offset": int(dispatch["base_offset"]),
                "capacity": int(dispatch["capacity"]),
                "dense_capacity": int(dispatch["dense_capacity"]),
                "growth_tokens": int(dispatch["growth_tokens"]),
                # The bucket the capacity rounds up to while every QSA bank is
                # on the rows-gather lane (the dense lane keeps its exact
                # capacity); the trace count above says whether this request
                # replayed a capacity the process had already traced.
                "capacity_bucket": int(dispatch["capacity_bucket"]),
                "rows_gather": all(
                    entry.fixed_rows_gather for entry in dispatch["qsa_entries"]
                ),
                # Image requests replay the trace that takes the rotary delta
                # as an input; text requests replay the text trace.
                "rope_delta_input": bool(dispatch["rope_args"]),
                "rope_delta": dispatch["rope_delta"],
            }
        return data

    # -- dispatch preconditions ----------------------------------------------

    def _fallback_reason(
        self,
        input_ids,
        cache,
        return_hidden: bool,
        *,
        consume_post_restore: bool = True,
        extended_window: bool = False,
    ) -> str | None:
        if self.permanent_eager:
            return "permanent_eager"
        if not return_hidden:
            return "hidden_not_requested"
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) != 2:
            return "invalid_input_shape"
        if int(shape[0]) != 1:
            return "batch_size"
        length = int(shape[1])
        if length < 1:
            return "invalid_length"
        window_ceiling = (
            max(self.max_verify_len, _ccopy_bank_max_len())
            if extended_window
            else self.max_verify_len
        )
        if length > window_ceiling:
            return "length_outside_bank"
        if extended_window and length > self.max_verify_len:
            # Dense-capacity preflight: an extended window must never grow a
            # granted dense KV leaf (`promote_kv_cache_offsets` below would
            # call `ensure_capacity(size + length)` and flip
            # `growth_after_grant`). When the window cannot fit the grant,
            # run the SAME once-per-request growth-demotion transition the
            # MTP lane runs at grant exhaustion — the MTP top-up would trip
            # it within `max_verify_len` tokens anyway — so the eager
            # fallback verifies against stock containers that grow natively.
            # (Falling back onto the still-granted adapter would overflow
            # its fixed buffer inside `update_and_fetch`; the route-off
            # eager lane shares that narrow dense-edge exposure today.)
            for entry in cache or []:
                if not isinstance(entry, TensorOffsetKVCache):
                    continue
                if entry.keys is None:
                    continue
                if entry.size() + length > int(entry.keys.shape[2]):
                    self._growth_demoted = True
                    self.stats["growth_demotions"] = (
                        int(self.stats.get("growth_demotions", 0)) + 1
                    )
                    self._materialize_growth_handoff_state(cache)
                    self.demote(cache)
                    return "block_window_capacity"
        if self.capture_backend in _UNSUPPORTED_CAPTURE_BACKENDS:
            return "unsupported_capture_backend"
        if _owned_state_env_active("MTPLX_OWNED_ATTN_KV"):
            return "owned_attn_kv_env"
        if _owned_state_env_active("MTPLX_OWNED_RECURRENT_STATE"):
            return "owned_recurrent_state_env"
        if cache is None:
            return "no_cache"
        if self._growth_demoted:
            # Cache was demoted back to stock entries when the growth budget
            # tripped; the plain eager path owns the rest of this request.
            return "growth_budget_exhausted"
        if self._post_restore_eager_remaining > 0:
            # Keep the restored cache unpromoted for the first round(s) so the
            # O(context) ensure_capacity copy lands after the first token is
            # already on the wire, not inside warm TTFT. Non-consuming probes
            # (prewarm eligibility) must not tick the counter — and must still
            # skip, or the probe itself would promote and pay the copy.
            if consume_post_restore:
                self._post_restore_eager_remaining -= 1
            return "post_restore_warmup"
        promoted, failures = promote_kv_cache_offsets(
            cache,
            reserve_tokens=length,
            preserve_paged=True,
            initial_reserve_tokens=max(length, self.growth_reserve_tokens),
            qsa_capacity_bucket=self.fixed_m4_capacity_bucket,
            qsa_capacity_plan=self.capacity_plan,
        )
        self.stats["promoted"] += promoted
        self._stamp_rope_delta(cache)
        for entry in cache:
            if isinstance(entry, TensorOffsetKVCache) and entry.growth_after_grant:
                # A dense leaf outgrew its first-promotion grant: every
                # further growth step would retrace the compiled graph, and
                # eager-on-adapter pays capacity-wide masks + non-donatable
                # slice updates (measured -15% vs clean eager at 7k). Demote
                # to stock entries NOW and stay eager for the rest of this
                # request (the bank is per-request, so the next round
                # re-grants fresh headroom).
                self._growth_demoted = True
                self.stats["growth_demotions"] = (
                    int(self.stats.get("growth_demotions", 0)) + 1
                )
                self._materialize_growth_handoff_state(cache)
                self.demote(cache)
                return "growth_budget_exhausted"
        if failures:
            if "quantized_paged_kv_cache" in failures:
                return "quantized_paged_kv"
            if "quantized_paged_kv_geometry" in failures:
                return "quantized_paged_kv_geometry"
            return "promotion_failure:" + ",".join(sorted(failures))
        if cache_has_python_offsets(cache):
            return "python_cache_offsets"
        spec, spec_reason = build_verify_state_spec(cache)
        if spec is None:
            return spec_reason or "unsupported_container"
        self._spec = spec
        if self.capture_backend == "linear_gdn_from_conv_tape":
            for idx, kind, _n in spec:
                if kind == VERIFY_SPEC_KIND_GDN and self._gdn_meta(idx) is None:
                    return "gdn_meta_unavailable"
        return None

    def _resolve_bucket(self, cache: Any, length: int) -> int | None:
        """Static paged-attention ceiling for this call, or None on overflow."""
        # One eval for every paged offset instead of a serial sync per entry
        # inside size() below (#318; helper docstring has the mechanism).
        # Mirrors this loop's own iteration exactly.
        materialize_paged_offsets(
            cache[spec_idx]
            for spec_idx, spec_kind, _n in self._spec or []
            if spec_kind == VERIFY_SPEC_KIND_FULL_ATTN
            and hasattr(cache[spec_idx], "capacity")
        )
        max_needed = 0
        min_capacity: int | None = None
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if not hasattr(entry, "capacity"):
                continue  # dense adapter: grows via ensure_capacity instead
            offset = int(entry.size())
            capacity = int(entry.capacity)
            max_needed = max(max_needed, offset + length)
            min_capacity = capacity if min_capacity is None else min(min_capacity, capacity)
        self._last_context_estimate = max_needed
        if min_capacity is None:
            return 0  # no paged entries; bucket unused
        if max_needed > min_capacity:
            return None
        bucket = min(min_capacity, _next_pow2(max_needed + 512))
        if max_needed > bucket:  # hard precondition: offset+M <= bucket
            bucket = min_capacity
        return bucket

    def _paged_ineligibility(self, cache: Any, length: int, bucket: int) -> str | None:
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if not hasattr(entry, "capacity"):
                continue
            if not _paged_kernel_bucket_eligible(entry, length, bucket):
                return "paged_kernel_ineligible"
        return None

    def _apply_bucket(self, cache: Any, bucket: int) -> None:
        """Pin the per-instance static ceiling on shadow and real paged entries.

        The two-pass paged kernel's reduction topology depends on the static
        ceiling, so the real entries get the same bucket: eager fallback calls
        and parity's authoritative eager run then use the identical kernel
        shape, which is what makes bit-exact comparison meaningful.
        """
        if not bucket:
            return
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if hasattr(entry, "static_max_offset"):
                entry.static_max_offset = int(bucket)
            shadow_entry = self._shadow[idx] if self._shadow else None
            if shadow_entry is not None and hasattr(shadow_entry, "static_max_offset"):
                shadow_entry.static_max_offset = int(bucket)

    # -- shadow cache ---------------------------------------------------------

    def _container_signature(self, cache: Any) -> tuple[Any, ...]:
        signature: list[Any] = []
        for entry in cache or []:
            if entry is None:
                signature.append(None)
                continue
            meta = (
                (int(entry.block_size), int(entry.num_blocks))
                if hasattr(entry, "num_blocks")
                else ()
            )
            signature.append((id(entry), type(entry).__name__, meta))
        return tuple(signature)

    def _ensure_shadow(self, cache: Any) -> None:
        signature = self._container_signature(cache)
        if self._shadow is not None and signature == self._shadow_signature:
            return
        from .cache_state import (
            TensorOffsetQuantizedPagedKVCache,
            TensorOffsetVllmMetalPagedKVCache,
        )

        shadow: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                kv = TensorOffsetKVCache(
                    entry.kv.cache[0],
                    entry.kv.cache[1],
                    entry.kv.cache[2],
                    step=entry.kv.step,
                )
                twin = TensorOffsetQSACache(
                    kv,
                    entry.raw_keys,
                    entry.pooled,
                    compress_ratio=entry.ratio,
                    rows_gather=entry.fixed_rows_gather,
                    rows_gather_kv_m4=entry.rows_gather_kv_m4,
                    rows_gather_enabled=entry.rows_gather_enabled,
                    rows_gather_min_context=entry.rows_gather_min_context,
                    fused_rows_gather_kv_m4=entry.fused_rows_gather_kv_m4,
                    rope_delta=entry.rope_delta,
                    capacity_bucket=entry.capacity_bucket,
                    dense_capacity=entry.dense_capacity,
                    indexer_budget=entry.indexer_budget,
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                # The twin carries the request's rotary origin like every
                # other slot; the traced body re-seeds it from the delta input.
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        entry.cache[0],
                        entry.cache[1],
                        entry.cache[2],
                        step=entry.step,
                        rope_delta=entry.rope_delta,
                    )
                elif isinstance(entry, TensorOffsetQuantizedPagedKVCache):
                    twin = TensorOffsetQuantizedPagedKVCache(
                        key_cache=entry.cache[0],
                        value_cache=entry.cache[1],
                        offset=entry.cache[2],
                        key_scale_cache=entry.cache[3],
                        value_scale_cache=entry.cache[4],
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                        kv_quant_config=entry.kv_quant_config,
                        source_dtypes=entry.source_dtypes,
                        head_dims=entry.head_dims,
                        rope_delta=entry.rope_delta,
                    )
                else:
                    twin = TensorOffsetVllmMetalPagedKVCache(
                        key_cache=entry.cache[0],
                        value_cache=entry.cache[1],
                        offset=entry.cache[2],
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                        rope_delta=entry.rope_delta,
                    )
            else:
                twin = type(entry)(len(entry.cache))
                for slot, leaf in enumerate(entry.cache):
                    twin[slot] = leaf
            shadow[idx] = twin
        self._shadow = shadow
        self._shadow_signature = signature
        # New shadow objects invalidate closures compiled over the old ones.
        self._compiled.clear()
        self._program_hosts.clear()

    # -- compiled function ------------------------------------------------------

    def _verify_program(self, key, length: int, hidden_variant: str | None):
        """This bank's compiled verify step for ``key``, bound to this bank.

        The first call for a key takes the process-wide program, or compiles
        one (``_shared_or_new_verify_step``), and the bank keeps it until its
        state containers change. A program the registry retires meanwhile
        stays this bank's program.
        """

        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(key, length, hidden_variant)
            self._compiled[key] = fn
        self._bind_program(self._program_hosts.get(key))
        return fn

    def _bind_program(self, host: dict[str, Any] | None) -> None:
        """Point a shared program's trace host at this bank before a call.

        A call at state shapes the program has not traced runs its Python
        body, which takes the shadow containers and the stats sink from the
        host. The registry points the host at the bank that looked the
        program up last, which need not be the calling bank: another bank may
        have looked it up since, that bank may have ended, and a retired
        program is not looked up again at all. Binding before every call keeps
        each trace on the calling bank's own containers.
        """

        if host is not None and host["bank_ref"]() is not self:
            host["bank_ref"] = weakref.ref(self)

    def _shared_or_new_verify_step(self, key, length: int, hidden_variant: str | None):
        """Reuse one compiled verify callable per process for a logical key.

        The bank is constructed per generation, so a per-instance compile dict
        pays a fresh trace (~1s wall at 7k leaves, measured 2026-07-03 as the
        whole compiled-vs-eager gap on long generations) for every request.
        The traced graph depends only on the runtime, capture layout, state
        spec, verify length, and hidden variant — mx.compile re-traces
        internally when leaf shapes change and caches per shape signature —
        so callables are shared process-wide. The closure's shadow containers
        are trace-time scratch: the re-seed firewall assigns every leaf from
        the explicit inputs before any read, so a retrace under a different
        bank/request is safe. Each program's trace host points its shadow and
        stats sink at the bank that calls it (``_bind_program``), so a retrace
        never touches another request's containers.

        MLX keeps every trace until the callable is destroyed, and the state
        shapes change with nearly every request, so a callable that has
        traced ``_shared_verify_traces_per_program()`` shapes is replaced
        here. Banks still replaying it keep it until they end; then MLX
        erases its traces. The replacement re-traces only the shapes that
        recur, which are the only shapes the old traces could have served.
        """

        rope_delta_input = bool(key[3])
        if not _env_enabled("MTPLX_COMPILED_VERIFY_SHARED_TRACES", default=True):
            return mx.compile(
                self._make_verify_step(
                    length, hidden_variant, rope_delta_input=rope_delta_input
                )
            )
        spec_sig = tuple(self._spec or [])
        from .attention_context import exact_verify_required

        global_key = (
            id(self.runtime),
            self.capture_backend,
            self._capture_layout_override,
            self._extra_capture_layout,
            self._prepare_compiled_aux is not None,
            spec_sig,
            int(length),
            str(hidden_variant or ""),
            int(key[2]),
            # Kernel-route dimension: a trace compiled under the sampled
            # (vk/nax) verify route bakes those kernels into the graph; a
            # greedy (t<=0, stock-route) request must never replay it, and
            # vice versa. Without this key a t=0.6 request's shared trace
            # would silently serve a t=0 request with non-exact kernels.
            bool(exact_verify_required()),
            # Input-signature dimension: an image request's step takes the
            # rotary delta as one more graph input. The VALUE never keys
            # anything, so every image request of the process shares one
            # trace whatever its delta, and text requests keep theirs.
            rope_delta_input,
        )
        entry = _SHARED_VERIFY_STEPS.get(global_key)
        if entry is not None:
            fn, host, runtime_ref = entry
            # id() can be recycled after a model swap frees the old runtime;
            # a stale callable would replay graphs bound to freed weights.
            if runtime_ref() is not self.runtime:
                _SHARED_VERIFY_STEPS.pop(global_key, None)
            elif host["traces"] >= _shared_verify_traces_per_program():
                _SHARED_VERIFY_STEPS.pop(global_key, None)
                self.stats["shared_programs_retired"] += 1
            else:
                host["bank_ref"] = weakref.ref(self)
                self._program_hosts[key] = host
                return fn
        # Programs may outlive requests. Keeping the bank here also keeps its
        # shadow KV and traced state alive after the request/session is gone.
        # The dispatch owns the bank; the process cache owns only the program.
        host = {"bank_ref": weakref.ref(self), "traces": 0}
        fn = mx.compile(
            self._make_verify_step(
                length,
                hidden_variant,
                trace_host=host,
                rope_delta_input=rope_delta_input,
            )
        )
        def release_program(reference, *, key=global_key):
            # Compiled graphs may hold weight arrays even after their Python
            # runtime is gone. Drop them at unload, not only on id() reuse.
            entry = _SHARED_VERIFY_STEPS.get(key)
            if entry is not None and entry[2] is reference:
                _SHARED_VERIFY_STEPS.pop(key, None)

        _SHARED_VERIFY_STEPS[global_key] = (
            fn, host, weakref.ref(self.runtime, release_program)
        )
        self._program_hosts[key] = host
        return fn

    def _make_verify_step(
        self,
        length: int,
        hidden_variant: str | None,
        trace_host: dict[str, Any] | None = None,
        rope_delta_input: bool = False,
    ):
        """The pure verify step: ``(input_ids, [aux], [rope_delta], *state)``.

        ``rope_delta_input`` adds the rotary delta of an image request as a
        graph input, after the auxiliary and before the state leaves. It must
        be an input: a delta closed over as a Python value would be folded
        into the trace and replayed, stale, for the next image request.
        """

        spec = list(self._spec or [])
        if rope_delta_input and not any(
            kind in (VERIFY_SPEC_KIND_QSA, VERIFY_SPEC_KIND_FULL_ATTN)
            for _idx, kind, _n in spec
        ):
            raise ValueError(
                "rope_delta has no consumer: only fixed QSA banks and "
                "full-attention tensor-offset adapters carry a rotary origin"
            )
        layout = self._capture_layout()
        static_host = {"bank_ref": weakref.ref(self), "traces": 0}
        host = trace_host if trace_host is not None else static_host

        def verify_step(input_ids, *args):
            # Python body executes at trace time only; replays skip it.
            live = host["bank_ref"]()
            if live is None:
                raise RuntimeError("compiled verifier traced without a live request bank")
            shadow = live._shadow
            if live._prepare_compiled_aux is not None:
                compiled_aux, *state_in = args
            else:
                compiled_aux = None
                state_in = args
            if rope_delta_input:
                rope_delta, *state_in = state_in
            else:
                rope_delta = None
            live.stats["traces"] += 1
            host["traces"] += 1
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            # (1) Re-seed firewall: every shadow leaf is assigned from the
            # explicit inputs BEFORE any read, so nothing stale and no tracer
            # from a previous trace can leak into this graph.
            pos = 0
            for idx, kind, n_leaves in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_QSA:
                    entry.kv.cache[0] = state_in[pos]
                    entry.kv.cache[1] = state_in[pos + 1]
                    entry.kv.cache[2] = state_in[pos + 2]
                    entry.raw_keys = state_in[pos + 3]
                    entry.pooled = state_in[pos + 4]
                    for slot in range(len(entry.kv.rollback_state)):
                        entry.kv.rollback_state[slot] = None
                    # The rotary origin is re-seeded like every other leaf:
                    # the traced input for an image request, None on the text
                    # trace, never whatever the twin held before.
                    entry.rope_delta = rope_delta
                elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                    for slot in range(len(entry.rollback_state)):
                        entry.rollback_state[slot] = None
                    # The rotary origin of a dense adapter is re-seeded the
                    # same way: the traced delta input, or None on the text
                    # trace.
                    entry.rope_delta = rope_delta
                else:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            # (2) The existing runtime forward, on shadow containers only.
            with attention_phase("decode_verify"), compiled_step_body():
                result = live._runtime_forward(
                    input_ids,
                    cache=shadow,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
                    compiled_aux=compiled_aux,
                )
            logits, hidden, captures = result
            # (3) Read every leaf back out and return it explicitly.
            captures_flat: list[Any] = []
            for idx, kind, _n in spec:
                if kind != VERIFY_SPEC_KIND_GDN:
                    continue
                layer_capture = captures[idx]
                for key_name in layout:
                    captures_flat.append(layer_capture[key_name])
            for idx, names in live._extra_capture_layout:
                layer_capture = captures[idx]
                for key_name in names:
                    captures_flat.append(layer_capture[key_name])
            state_out: list[Any] = []
            for idx, kind, _n in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_QSA:
                    state_out.extend(entry.state_leaves)
                elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    state_out.extend(entry.cache[slot] for slot in range(_n))
                else:
                    state_out.extend(entry.cache[slot] for slot in range(_n))
            return (logits, hidden, *captures_flat, *state_out)

        return verify_step

    def _capture_layout(self) -> tuple[str, ...]:
        if self._capture_layout_override is not None:
            return self._capture_layout_override
        if self.capture_backend == "linear_gdn_from_conv_tape":
            return TAPE_CAPTURE_KEYS
        return STANDARD_CAPTURE_KEYS

    def _unpack_outputs(self, outputs):
        spec = self._spec or []
        layout = self._capture_layout()
        n_captures = sum(
            len(layout) for _idx, kind, _n in spec if kind == VERIFY_SPEC_KIND_GDN
        )
        n_captures += sum(len(names) for _idx, names in self._extra_capture_layout)
        n_state = sum(n for _idx, _kind, n in spec)
        expected = 2 + n_captures + n_state
        if len(outputs) != expected:
            raise ValueError(
                f"compiled verify returned {len(outputs)} leaves, expected {expected}"
            )
        logits = outputs[0]
        hidden = outputs[1]
        captures_flat = list(outputs[2 : 2 + n_captures])
        state_out = list(outputs[2 + n_captures :])
        return logits, hidden, captures_flat, state_out

    def _rebuild_captures(self, captures_flat: list[Any]) -> dict[int, dict[str, Any]]:
        layout = self._capture_layout()
        captures: dict[int, dict[str, Any]] = {}
        pos = 0
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_GDN:
                continue
            layer_capture = {
                key_name: captures_flat[pos + key_pos]
                for key_pos, key_name in enumerate(layout)
            }
            pos += len(layout)
            if self.capture_backend == "linear_gdn_from_conv_tape":
                layer_capture["gdn_meta"] = self._gdn_meta(idx)
            captures[idx] = layer_capture
        for idx, names in self._extra_capture_layout:
            layer_capture = captures.setdefault(idx, {})
            for key_name in names:
                layer_capture[key_name] = captures_flat[pos]
                pos += 1
        return captures

    def _gdn_meta(self, layer_idx: int) -> dict[str, int] | None:
        if layer_idx in self._gdn_meta_cache:
            return self._gdn_meta_cache[layer_idx]
        meta: dict[str, int] | None = None
        try:
            from .gdn_capture import _gdn_tape_meta

            model = getattr(self.runtime, "model", None)
            text_model = getattr(model, "language_model", model)
            inner = getattr(text_model, "model", None)
            layer = inner.layers[layer_idx]
            meta = _gdn_tape_meta(layer.linear_attn)
        except Exception:
            meta = None
        self._gdn_meta_cache[layer_idx] = meta
        return meta

    # -- state movement -----------------------------------------------------------

    def _read_state_leaves(self, cache: Any) -> list[Any] | None:
        leaves: list[Any] = []
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                layer_leaves = tuple(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                layer_leaves = tuple(entry.cache[slot] for slot in range(_n))
            else:
                layer_leaves = tuple(entry.cache[slot] for slot in range(_n))
            if any(leaf is None for leaf in layer_leaves):
                return None
            leaves.extend(layer_leaves)
        return leaves

    def _mirror_commit(self, cache: Any, state_out: list[Any]) -> None:
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[pos]
                entry.kv.cache[1] = state_out[pos + 1]
                entry.kv.cache[2] = state_out[pos + 2]
                entry.raw_keys = state_out[pos + 3]
                entry.pooled = state_out[pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[pos + slot]
                # Cleared rollback forces trim() onto the offset-only branch,
                # which is the correct reject semantics for a batched verify.
                for slot in range(len(entry.rollback_state)):
                    entry.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[pos + slot]
            pos += n_leaves

    def _clear_shadow_leaf_refs(self) -> None:
        """Drop leaf references held by the shadow twins (A2.1 donation).

        The traced verify body re-seeds every shadow slot from the explicit
        inputs before any read, so whatever the twins hold between calls —
        promotion-time leaves right after ``_ensure_shadow``, stale tracers
        after a trace — is dead weight.  Promotion-time refs additionally
        alias the first call's input buffers, which would block their
        donation and pin one full stale KV/GDN buffer set for the whole
        generation.
        """
        for entry in self._shadow or []:
            if entry is None:
                continue
            if isinstance(entry, TensorOffsetQSACache):
                for slot in range(len(entry.kv.cache)):
                    entry.kv.cache[slot] = None
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
                entry.raw_keys = None
                entry.pooled = None
                # A twin holds the traced delta after a trace; like the other
                # slots it is re-seeded from the inputs before any read.
                entry.rope_delta = None
                continue
            cache_list = getattr(entry, "cache", None)
            if isinstance(cache_list, list):
                for slot in range(len(cache_list)):
                    cache_list[slot] = None
            rollback = getattr(entry, "rollback_state", None)
            if isinstance(rollback, list):
                for slot in range(len(rollback)):
                    rollback[slot] = None
            if isinstance(entry, RotaryOrigin):
                entry.rope_delta = None

    # -- eager paths ---------------------------------------------------------------

    def _runtime_forward(
        self,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        compiled_aux=None,
    ):
        kwargs = (
            {"compiled_aux": compiled_aux}
            if self._runtime_accepts_compiled_aux
            else {}
        )
        if self._capture_accepts_backend:
            return self.runtime.forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=self.capture_backend,
                **kwargs,
            )
        return self.runtime.forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )

    def _fallback(
        self,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        reason: str,
    ):
        if self.strict_no_fallback:
            raise RuntimeError(f"qwen4 fixed-M4 verifier refused: {reason}")
        _note_bank_fallback(reason)
        self.last_dispatch_kind = "eager"
        self.last_fallback_reason = reason
        growth_transition = (
            reason == "growth_budget_exhausted"
            and not self._growth_budget_fallback_reported
        )
        self.last_fallback_transition = growth_transition
        self.stats["fallback_calls"] += 1
        # Growth exhaustion is a one-way request state, not a fresh failure
        # on every eager tail round.  Count its transition once while keeping
        # fallback_calls as the honest per-call total.
        if reason != "growth_budget_exhausted" or growth_transition:
            self.stats["fallback_reasons"][reason] = (
                self.stats["fallback_reasons"].get(reason, 0) + 1
            )
        if growth_transition:
            self._growth_budget_fallback_reported = True
        return self._runtime_forward(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def _parity_check(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        compiled_aux,
        state_in: list[Any],
        compiled_logits,
        compiled_hidden,
        compiled_captures,
        compiled_state_out,
    ):
        """Double-run: compiled pure step already ran; eager is authoritative."""
        self.stats["parity_checks"] += 1
        with attention_phase("decode_verify"):
            eager_logits, eager_hidden, eager_captures = self._runtime_forward(
                input_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
            )
        eager_state = []
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                eager_state.extend(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                eager_state.extend(entry.cache[slot] for slot in range(_n))
            else:
                eager_state.extend(entry.cache[slot] for slot in range(_n))
        reference = self._named_outputs(eager_logits, eager_hidden, eager_captures, eager_state)
        candidate = self._named_outputs(
            compiled_logits, compiled_hidden, compiled_captures, compiled_state_out
        )
        report = compare_verify_outputs(reference, candidate)
        if report:
            self.stats["parity_failures"] += 1
            raise CompiledVerifyParityError(report)
        return eager_logits, eager_hidden, eager_captures

    def _parity2_check(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        compiled_aux,
        bucket: int,
        compiled_logits,
        compiled_hidden,
        compiled_captures,
        compiled_state_out,
    ):
        """Inverted parity: COMPILED is authoritative; an eager CLONE tracks it.

        Parity mode #1 proved per-call bit-exactness at fixed contexts, but its
        eager leg re-commits the real cache on every call, so compiled-committed
        state never compounds across steps — exactly the multi-step evolution
        the live-stream fork hypothesis points at.  Here the real stream keeps
        running on the compiled mirror-commit, and the eager reference replays
        the same single step on a fresh leaf-copy clone of the pre-step cache.
        The clone is rebuilt from the real entries every call, so accept-path
        commits/trims on the real cache between calls can never drift the clone
        structurally: each comparison is one verify step given identical
        (compiled-committed) inputs.  A mismatch is logged and counted — never
        raised — so streaming continues compiled-authoritative.
        """
        self.stats["parity2_calls"] += 1
        record = self._parity2_record()
        width = str(_decode_length(input_ids))
        if not self._parity2_reference_ready(record, width):
            self._mirror_commit(cache, compiled_state_out)
            return compiled_logits, compiled_hidden, compiled_captures
        # Seed the clone BEFORE mirror-commit: the real entries still hold the
        # pre-step leaves here (the compiled step ran purely on the shadow).
        clone = self._parity2_clone_cache(cache, bucket)
        with attention_phase("decode_verify"), self._parity2_reference_scope():
            eager_logits, eager_hidden, eager_captures = self._runtime_forward(
                input_ids,
                cache=clone,
                return_hidden=True,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
            )
        # Compiled is authoritative: the live stream advances on compiled state.
        self._mirror_commit(cache, compiled_state_out)
        clone_state: list[Any] = []
        for idx, kind, _n in self._spec or []:
            entry = clone[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                clone_state.extend(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                clone_state.extend(entry.cache[slot] for slot in range(_n))
            else:
                clone_state.extend(entry.cache[slot] for slot in range(_n))
        reference = self._named_outputs(
            eager_logits, eager_hidden, eager_captures, clone_state
        )
        candidate = self._named_outputs(
            compiled_logits, compiled_hidden, compiled_captures, compiled_state_out
        )
        self._parity2_compare(record, width, reference, candidate, cache=cache)
        return compiled_logits, compiled_hidden, compiled_captures

    def _parity2_clone_cache(self, cache: Any, bucket: int) -> list[Any]:
        """Fresh eager-leg clone: real container classes over leaf COPIES.

        Mirrors ``_ensure_shadow``'s twin construction but with materialized
        ``mx.array`` copies instead of shared refs, so the eager forward's
        writes (functional slice_updates and slot reassignments) can never
        interact with the buffers the compiled-authoritative stream holds.
        """
        from .cache_state import (
            TensorOffsetQuantizedPagedKVCache,
            TensorOffsetVllmMetalPagedKVCache,
        )

        clone: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                kv = TensorOffsetKVCache(
                    _copy_state_leaf(entry.kv.cache[0]),
                    _copy_state_leaf(entry.kv.cache[1]),
                    _copy_state_leaf(entry.kv.cache[2]),
                    step=entry.kv.step,
                )
                twin = TensorOffsetQSACache(
                    kv,
                    _copy_state_leaf(entry.raw_keys),
                    _copy_state_leaf(entry.pooled),
                    compress_ratio=entry.ratio,
                    rows_gather=entry.fixed_rows_gather,
                    rows_gather_kv_m4=entry.rows_gather_kv_m4,
                    rows_gather_enabled=entry.rows_gather_enabled,
                    rows_gather_min_context=entry.rows_gather_min_context,
                    fused_rows_gather_kv_m4=entry.fused_rows_gather_kv_m4,
                    # The eager leg of the generic parity modes runs on this
                    # fixed twin, so it rotates where the real entry does.
                    rope_delta=entry.rope_delta,
                    capacity_bucket=entry.capacity_bucket,
                    dense_capacity=entry.dense_capacity,
                    indexer_budget=entry.indexer_budget,
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                # No rotary delta on the reference leg, on purpose: the eager
                # forward on this clone positions every row from the
                # request's host table (``_parity2_reference_scope``), which
                # is independent of the delta under test, on the same padded
                # buffers the compiled step reduced over.
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        _copy_state_leaf(entry.cache[0]),
                        _copy_state_leaf(entry.cache[1]),
                        _copy_state_leaf(entry.cache[2]),
                        step=entry.step,
                    )
                elif isinstance(entry, TensorOffsetQuantizedPagedKVCache):
                    twin = TensorOffsetQuantizedPagedKVCache(
                        key_cache=_copy_state_leaf(entry.cache[0]),
                        value_cache=_copy_state_leaf(entry.cache[1]),
                        offset=_copy_state_leaf(entry.cache[2]),
                        key_scale_cache=_copy_state_leaf(entry.cache[3]),
                        value_scale_cache=_copy_state_leaf(entry.cache[4]),
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                        kv_quant_config=entry.kv_quant_config,
                        source_dtypes=entry.source_dtypes,
                        head_dims=entry.head_dims,
                    )
                else:
                    twin = TensorOffsetVllmMetalPagedKVCache(
                        key_cache=_copy_state_leaf(entry.cache[0]),
                        value_cache=_copy_state_leaf(entry.cache[1]),
                        offset=_copy_state_leaf(entry.cache[2]),
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                    )
                if bucket and hasattr(twin, "static_max_offset"):
                    # Same static ceiling as the real/shadow entries so the
                    # eager paged kernel runs the identical reduction topology
                    # (what makes bit-exact comparison meaningful).
                    twin.static_max_offset = int(bucket)
            else:
                twin = type(entry)(len(entry.cache))
                for slot, leaf in enumerate(entry.cache[:_n]):
                    twin[slot] = _copy_state_leaf(leaf)
            clone[idx] = twin
        return clone

    def _parity2_record(self) -> dict[str, Any]:
        """This request's parity receipt (``compiled_verify.parity2``).

        The shape the fixed-M4 lane's ``fixed_m4_parity2`` has, so one reader
        serves both families: rounds compared, rounds that differed anywhere,
        the largest differences, and the first divergent leaf.
        """

        return self.stats.setdefault(
            "parity2",
            {
                "positions": "text" if self._rope_delta is None else "vision_delta",
                "rope_delta": self._rope_delta_value,
                "rounds": 0,
                "rounds_by_width": {},
                "divergent_rounds": 0,
                "divergent_rounds_by_width": {},
                # Rounds of an image request NOT compared, because their
                # caller had no request scope open (``_parity2_reference_ready``).
                "reference_scope_missing_rounds": 0,
                "reference_scope_missing_by_width": {},
                "compared_leaves_per_round": 0,
                "logits_max_abs_diff": 0.0,
                "logits_max_kl": 0.0,
                "hidden_max_abs_diff": 0.0,
                "state_max_abs_diff": 0.0,
                "capture_max_abs_diff": 0.0,
                "first_divergence": None,
            },
        )

    def _parity2_reference_ready(self, record: dict[str, Any], width: str) -> bool:
        """Whether this round has a reference independent of the delta.

        A text round always has one. The reference of an image round is the
        REQUEST's own position table, read from the caller's scope by the
        eager leg (the dense adapter through ``_parity2_reference_scope``,
        the Flash-Next attention through ``vision_rope_state``); the bank's
        delta is never consulted, or a wrong delta would be compared with
        itself and pass. A caller with no scope open leaves nothing
        independent to compare with: the round is counted, not compared, so
        the maxima stay the maxima of real comparisons.
        """

        if self._rope_delta is None:
            return True
        from .attention_context import vision_rope_state
        from .dense_mrope import dense_mrope_state

        if dense_mrope_state() is not None or vision_rope_state() is not None:
            return True
        record["reference_scope_missing_rounds"] += 1
        record["reference_scope_missing_by_width"][width] = (
            record["reference_scope_missing_by_width"].get(width, 0) + 1
        )
        return False

    def _parity2_reference_scope(self):
        """The eager leg of an image round positions rows from the host table.

        The clone carries no rotary delta (``_parity2_clone_cache``); under
        this scope the dense adapter resolves the clone's concrete offset
        through the request's table, exactly as the eager route on a stock
        cache does, on the same padded buffers the compiled step used.
        """

        if self._rope_delta is None:
            return contextlib.nullcontext()
        from .dense_mrope import host_positions_for_tensor_offsets

        return host_positions_for_tensor_offsets()

    def _parity2_compare(
        self,
        record: dict[str, Any],
        width: str,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        *,
        cache: Any,
    ) -> None:
        """Bit-for-bit on the device; the host sees only a divergence."""

        names = sorted(set(reference) | set(candidate))
        comparable: list[tuple[str, Any, Any]] = []
        mismatched: list[str] = []
        for name in names:
            ref, cand = reference.get(name), candidate.get(name)
            if ref is None and cand is None:
                continue
            if (
                isinstance(ref, mx.array)
                and isinstance(cand, mx.array)
                and ref.shape == cand.shape
                and ref.dtype == cand.dtype
            ):
                comparable.append((name, ref, cand))
            elif not hasattr(ref, "shape") and not hasattr(cand, "shape"):
                if ref != cand:
                    mismatched.append(name)
            else:
                mismatched.append(name)
        flags = [mx.array_equal(ref, cand) for _name, ref, cand in comparable]
        ref32 = reference["logits"].astype(mx.float32)
        cand32 = candidate["logits"].astype(mx.float32)
        log_p = ref32 - mx.logsumexp(ref32, axis=-1, keepdims=True)
        log_q = cand32 - mx.logsumexp(cand32, axis=-1, keepdims=True)
        max_kl = mx.max(mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1))
        logits_diff = mx.max(mx.abs(ref32 - cand32))
        hidden_diff = mx.max(
            mx.abs(
                reference["hidden"].astype(mx.float32)
                - candidate["hidden"].astype(mx.float32)
            )
        )
        mx.eval(flags, max_kl, logits_diff, hidden_diff)
        mismatched.extend(
            name for (name, _ref, _cand), flag in zip(comparable, flags)
            if not bool(flag.item())
        )
        mismatched.sort(key=names.index)

        record["rounds"] += 1
        record["rounds_by_width"][width] = record["rounds_by_width"].get(width, 0) + 1
        record["compared_leaves_per_round"] = len(names)
        record["logits_max_abs_diff"] = max(
            record["logits_max_abs_diff"], float(logits_diff.item())
        )
        record["logits_max_kl"] = max(record["logits_max_kl"], float(max_kl.item()))
        record["hidden_max_abs_diff"] = max(
            record["hidden_max_abs_diff"], float(hidden_diff.item())
        )
        if not mismatched:
            return

        record["divergent_rounds"] += 1
        record["divergent_rounds_by_width"][width] = (
            record["divergent_rounds_by_width"].get(width, 0) + 1
        )
        self.stats["parity2_divergent_calls"] += 1
        # Size of every divergence, still on the device: one scalar per leaf.
        sized = [
            (
                name,
                mx.max(
                    mx.abs(
                        reference[name].astype(mx.float32)
                        - candidate[name].astype(mx.float32)
                    )
                ),
            )
            for name in mismatched
            if isinstance(reference.get(name), mx.array)
            and isinstance(candidate.get(name), mx.array)
            and reference[name].shape == candidate[name].shape
        ]
        mx.eval([diff for _name, diff in sized])
        for name, diff in sized:
            family = _artifact_kind(name)
            if family in ("state", "capture"):
                key = f"{family}_max_abs_diff"
                record[key] = max(record[key], float(diff.item()))

        ordinal = int(self.stats["calls"])
        context = self._parity2_context_estimate(cache)
        first_name = mismatched[0]
        artifact = _artifact_kind(first_name)
        max_abs = _leaf_max_abs_diff(reference.get(first_name), candidate.get(first_name))
        if record["first_divergence"] is None:
            # The readable lines cross to the host, so only the first
            # divergent round writes them and only for its first few leaves
            # (one K bank is hundreds of megabytes at long context).
            detailed = mismatched[:8]
            record["first_divergence"] = {
                "round": int(record["rounds"]),
                "width": int(width),
                "context": context,
                "leaf": first_name,
                "artifact": artifact,
                "mismatched_leaves": len(mismatched),
                "report": compare_verify_outputs(
                    {name: reference.get(name) for name in detailed},
                    {name: candidate.get(name) for name in detailed},
                    max_report_lines=len(detailed) + 2,
                ),
            }
        if self.stats["parity2_first_divergence"] is None:
            self.stats["parity2_first_divergence"] = {
                "call": ordinal,
                "context": context,
                "artifact": artifact,
                "leaf": first_name,
                "max_abs_diff": max_abs,
                "mismatched_leaves": len(mismatched),
            }
        count = int(self.stats["parity2_divergent_calls"])
        if count <= 10:
            max_abs_text = "n/a" if max_abs is None else f"{max_abs:.3e}"
            print(
                f"[parity2] divergence call={ordinal} context={context} "
                f"artifact={artifact} leaf={first_name} "
                f"max_abs_diff={max_abs_text} mismatched_leaves={len(mismatched)}",
                flush=True,
            )
            if count == 10:
                print(
                    "[parity2] divergence log cap reached (10); further "
                    "divergent calls are counted in stats only "
                    "(parity2_divergent_calls)",
                    flush=True,
                )

    def _parity2_context_estimate(self, cache: Any) -> int:
        """Context/offset estimate for divergence reports (tokens).

        Paged entries already produced offset+M in ``_resolve_bucket``; dense
        adapters (no ``capacity``) fall through to the post-commit offset.
        Best-effort diagnostics only — never load-bearing.
        """
        estimate = int(getattr(self, "_last_context_estimate", 0) or 0)
        if estimate:
            return estimate
        best = 0
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            try:
                best = max(best, int(entry.size()))
            except Exception:
                continue
        return best

    def _named_outputs(
        self,
        logits,
        hidden,
        captures: dict[int, dict[str, Any]],
        state_leaves: list[Any],
    ) -> dict[str, Any]:
        named: dict[str, Any] = {"logits": logits, "hidden": hidden}
        layout = self._capture_layout()
        for layer_idx in sorted(k for k in captures if isinstance(k, int)):
            layer_capture = captures[layer_idx]
            for key_name in layout:
                named[f"capture[{layer_idx}].{key_name}"] = layer_capture.get(key_name)
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            for leaf_idx in range(n_leaves):
                named[f"state[{idx}:{kind}].{leaf_idx}"] = state_leaves[pos + leaf_idx]
            pos += n_leaves
        return named
