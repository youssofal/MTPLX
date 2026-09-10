"""Speculative decode graph-bank scaffolding for MLX.

The first useful job of this module is to make graph-capture eligibility
explicit.  The current Qwen3.6 MLX cache keeps full-attention positions as
Python integers, so a safe compiled decode graph cannot replay across decode
steps until those offsets become tensor inputs/outputs.
"""

from __future__ import annotations

import os
import time
import weakref
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import mlx.core as mx

from .attention_context import attention_phase
from .gdn_capture import resolve_gdn_capture_backend
# Module-level so a test (and the twin-construction guard) can see one name;
# read at each call so the process-frozen flag is honoured and monkeypatchable.
from .runtime_options import qsa_sparse_decode_enabled


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
    ) -> None:
        self.runtime = runtime
        self.max_verify_len = max_verify_len
        self.allow_python_cache_capture = allow_python_cache_capture
        self.promote_tensor_offsets = promote_tensor_offsets
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
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
        if self.allow_python_cache_capture:
            return None
        if self.promote_tensor_offsets:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=length)
            self.stats.promoted_cache_entries += promoted
            for reason, count in failures.items():
                self.stats.promotion_failures[reason] = (
                    self.stats.promotion_failures.get(reason, 0) + count
                )
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


class TensorOffsetKVCache:
    """Full-attention KV cache adapter with array-backed mutable offset.

    Stock `KVCache.offset` is a Python integer.  In a compiled verify graph that
    integer is graph-constant state, so RoPE and mask positions can silently go
    stale.  This adapter keeps the existing key/value buffers, stores the offset
    in `cache[2]`, and mutates the three-array state through operations visible
    to `mx.compile(inputs=..., outputs=...)`.
    """

    def __init__(
        self,
        keys: mx.array,
        values: mx.array,
        offset: int | mx.array,
        *,
        step: int = 256,
    ) -> None:
        offset_array = (
            offset
            if isinstance(offset, mx.array)
            else mx.array(offset, dtype=mx.int32)
        )
        self.cache = [keys, values, offset_array]
        self.rollback_state = [None, None, None]
        self.step = step
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
        return [self.cache, self.rollback_state]

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

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
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
        qsa_sparse_decode: bool = False,
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
        # MTPLX_QSA_SPARSE_DECODE: the native split-K sparse-GQA attention
        # lane. Validated ONCE, here, at cache install: this is model-build
        # time and outside any mx.compile trace, which is what lets the
        # install run a real parity probe. The indexer reads the row count
        # below and never re-derives the decision, so one trace of the verify
        # graph cannot disagree with the next about which attention it holds.
        #
        # The gate is asymmetric on purpose. install() RAISES when the
        # contract cannot be met (an armed flag that cannot apply is a
        # configuration error, including a missing native extension) and
        # DISABLES when the numerical probe fails (this kernel is
        # rounding-class; a parity miss is a measurement, and turning a
        # measurement into an outage helps nobody). The server auto-arm only
        # STAMPS the default when the extension is built, so a wheel without
        # it serves the stock decode path; an explicit operator export of
        # MTPLX_QSA_SPARSE_DECODE=1 still reaches this fail-closed install.
        self.qsa_sparse_decode = bool(qsa_sparse_decode)
        self.qsa_sparse_decode_rows = 0
        # An armed flag that reaches a cache constructed WITHOUT it is the
        # armed-but-inert failure mode, and it is silent: the cache carries
        # qsa_sparse_decode_rows = 0 and every routing decision declines.
        # Every construction site that can be reached with the flag armed
        # passes it through (from_qsa_cache reads the frozen flag), so a site
        # that forgot is a bug in THIS file and dies here.
        if qsa_sparse_decode_enabled() and not self.qsa_sparse_decode:
            raise RuntimeError(
                "MTPLX_QSA_SPARSE_DECODE is armed but this QSA cache was "
                "constructed without the lane; the construction site did not "
                "pass qsa_sparse_decode through, so the kernel would be inert "
                "on this cache"
            )
        if self.qsa_sparse_decode:
            from .kernels import qsa_sparse_decode as _qsa_sparse

            if not _qsa_sparse.install(
                self.kv.keys,
                self.kv.values,
                compress_ratio=self.ratio,
                verify=True,
            ):
                # install() recorded the measured deltas before returning
                # False; this turns them into a build failure instead of a
                # silent revert to the stock chain: an armed arm that runs the
                # stock chain is worse than an outage because it looks like a
                # result.
                raise RuntimeError(
                    "MTPLX_QSA_SPARSE_DECODE is armed but the split-K lane "
                    "declined to install: "
                    + (
                        _qsa_sparse.disabled_reason()
                        or "the lane returned no verdict"
                    )
                    + " -- read the deltas off the [mtplx] qsa_sparse_decode "
                    "stderr line, then unarm the flag deliberately"
                )
            self.qsa_sparse_decode_rows = _qsa_sparse.VERIFY_ROWS

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
        return mx.concatenate(
            [value, mx.zeros(tuple(shape), dtype=value.dtype)], axis=axis
        )

    @classmethod
    def from_qsa_cache(
        cls, entry: Any, *, reserve_tokens: int
    ) -> "TensorOffsetQSACache":
        reserve_tokens = max(1, int(reserve_tokens))
        offset = int(entry.offset)
        ratio = max(1, int(entry.ratio))
        if entry.raw_keys is None or entry.pooled is None:
            raise ValueError("QSA index state is empty")
        if entry.kv.keys is None or entry.kv.values is None:
            raise ValueError("QSA attention state is empty")

        logical_capacity = offset + reserve_tokens
        raw_capacity = ((logical_capacity + ratio - 1) // ratio) * ratio
        pooled_capacity = raw_capacity // ratio

        kv = TensorOffsetKVCache.from_kv_cache(
            entry.kv, reserve_tokens=reserve_tokens
        )
        kv.keys = cls._fixed_bank(kv.keys, raw_capacity, 2)
        kv.values = cls._fixed_bank(kv.values, raw_capacity, 2)
        raw = cls._fixed_bank(entry.raw_keys, raw_capacity, 1)
        pooled = cls._fixed_bank(entry.pooled, pooled_capacity, 1)
        from .models.qwen4_exp import (
            _qsa_gather_enabled,
            _qsa_gather_min_context,
        )

        rows_gather_enabled = _qsa_gather_enabled()
        rows_gather_min_context = _qsa_gather_min_context()
        rows_gather = rows_gather_enabled and offset >= rows_gather_min_context
        # MTPLX_QSA_SPARSE_DECODE: read the process-frozen flag here so the
        # cache validates the native split-K lane once, at install; the
        # indexer never re-derives it. Passed to __init__ below.
        qsa_sparse_decode = qsa_sparse_decode_enabled()
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

        return cls(
            kv,
            raw,
            pooled,
            compress_ratio=ratio,
            rows_gather=rows_gather,
            rows_gather_kv_m4=rows_gather_kv_m4,
            rows_gather_enabled=rows_gather_enabled,
            rows_gather_min_context=rows_gather_min_context,
            fused_rows_gather_kv_m4=fused_rows_gather_kv_m4,
            qsa_sparse_decode=qsa_sparse_decode,
        )

    @property
    def capacity(self) -> int:
        return int(self.raw_keys.shape[1])

    def ensure_capacity(self, needed: int) -> bool:
        """Grow this installed QSA generation without changing its offset."""

        raw_capacity = (
            (max(1, int(needed)) + self.ratio - 1) // self.ratio
        ) * self.ratio
        if raw_capacity <= self.capacity:
            return False
        pooled_capacity = raw_capacity // self.ratio
        self.kv.keys = self._fixed_bank(self.kv.keys, raw_capacity, 2)
        self.kv.values = self._fixed_bank(self.kv.values, raw_capacity, 2)
        self.raw_keys = self._fixed_bank(self.raw_keys, raw_capacity, 1)
        self.pooled = self._fixed_bank(self.pooled, pooled_capacity, 1)
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

    def demote(self):
        from .models.qwen4_exp import QSACache

        offset = self.kv.size()
        entry = QSACache(self.ratio)
        entry.kv = self.kv.demote()
        entry.raw_keys = self.raw_keys
        entry.pooled = self.pooled
        entry.pooled_len = min(int(self.pooled.shape[1]), offset // self.ratio)
        return entry


def promote_kv_cache_offsets(
    cache: Any,
    *,
    reserve_tokens: int,
    preserve_paged: bool | None = None,
    initial_reserve_tokens: int | None = None,
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
                    if not _env_enabled(
                        "MTPLX_GRAPHBANK_QUANTIZED_PAGED", default=True
                    ):
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
    return promoted, failures


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _record_permanent_eager(reason: str, *, once: bool = False) -> None:
    """Record (and log once per distinct reason) a permanent-eager flip.

    ``once=True`` marks deterministic construction-time flips (per-model
    quant gate): the first bank records and logs; subsequent per-request
    banks only re-assert ``permanent_eager`` without inflating the count.
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
# containers. See CompiledVerifyBank._shared_or_new_verify_step.
_SHARED_VERIFY_STEPS: dict[tuple, tuple[Any, dict[str, Any], Any]] = {}


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


def _compiled_verify_bits_gate_ok(runtime: Any) -> bool:
    if _env_enabled("MTPLX_COMPILED_VERIFY_FORCE"):
        return True
    bits = _runtime_trunk_quant_bits(runtime)
    # Measured-win allowlist: 4-bit and 8-bit affine trunks engage;
    # unquantized (None) passes for test rigs and bf16 research models.
    # Unmeasured quantizations (e.g. the 6-bit 9B) stay eager.
    return bits is None or bits in (4, 8)


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
    ) -> None:
        self.runtime = runtime
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
        self._compiled: dict[tuple[int, str, int], Any] = {}
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

    def install_fixed_m4(
        self,
        cache: Any,
        *,
        prompt_ids,
        hidden_variant: str | None,
    ) -> None:
        """Install the exact Qwen4 physical-M4 replay once after prefill."""

        if not self.strict_no_fallback:
            raise ValueError("fixed-M4 installation requires the Qwen4 runtime route")
        if self.parity or self.parity2:
            raise ValueError("fixed-M4 direct replay is disabled in parity modes")

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
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        key = (4, str(hidden_variant or ""), route_key)
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(key, 4, hidden_variant)
            self._compiled[key] = fn
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
            "donate": (
                _compiled_verify_donation_enabled()
                and boundary in ("both", "post")
            ),
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
                f"donate={self._fixed_m4_dispatch['donate']} kv_gather={kv_gather}",
                flush=True,
            )

    def reserve_fixed_m4_window(
        self,
        cache: Any,
        *,
        committed_count: int | None = None,
        window_tokens: int = 4,
    ) -> None:
        """Reserve every target write into an installed fixed-capacity bank.

        D1/D2 and copy windows use these same buffers even when their forward
        runs eager. They must renew capacity before writing, too. Generation
        supplies its host ledger to avoid a device sync; standalone callers
        without that ledger use the live cache offset. Generic banks are
        unchanged. Keep four rows reserved for a possible lazy bonus write.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            return
        logical_start = (
            int(dispatch["base_offset"]) + max(0, int(committed_count))
            if committed_count is not None
            else dispatch["qsa_entries"][0].size()
        )
        required_end = logical_start + max(4, int(window_tokens))
        capacity_needed = required_end > int(dispatch["capacity"])
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
            next_capacity, next_growth_tokens = _fixed_m4_capacity_growth(
                capacity=int(dispatch["capacity"]),
                required_end=required_end,
                growth_tokens=int(dispatch["growth_tokens"]),
                capacity_limit=dispatch["capacity_limit"],
            )
            for entry in qsa_entries:
                capacity_changed = (
                    entry.ensure_capacity(next_capacity) or capacity_changed
                )
        route_changed = False
        if route_needed:
            for entry in qsa_entries:
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
        if not capacity_changed and not route_changed:
            return

        self._clear_shadow_leaf_refs()
        self._held_state_refs.clear()
        self._shadow = None
        self._shadow_signature = None
        self._ensure_shadow(cache)
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        key = (4, str(dispatch["hidden_variant"] or ""), route_key)
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(
                key,
                4,
                dispatch["hidden_variant"],
            )
            self._compiled[key] = fn
        dispatch["fn"] = fn
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
        self.reserve_fixed_m4_window(
            cache,
            committed_count=committed_count,
        )
        boundary = dispatch["boundary"]
        donate = dispatch["donate"]
        if donate:
            self._clear_shadow_leaf_refs()

        state_in: list[Any] = []
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
            else:
                state_in.extend(entry.cache[:n_leaves])

        compiled_aux = dispatch["prepare_aux"](
            input_ids,
            host_input_ids,
            completion_tokens,
            committed_count,
        )
        if boundary in ("both", "pre"):
            mx.async_eval(compiled_aux, *state_in)
        outputs = dispatch["fn"](input_ids, compiled_aux, *state_in)

        capture_end = 2 + dispatch["capture_leaves"]
        logits, hidden = outputs[:2]
        captures_flat = outputs[2:capture_end]
        state_out = outputs[capture_end:]

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

        # Route Tape receipt (main contract): the installed replay is a
        # compiled dispatch, so last_dispatch_route() reports it as such.
        self.last_dispatch_kind = "compiled"
        self.last_fallback_reason = None
        self.last_fallback_transition = False
        self.stats["compiled_calls"] += 1
        self.stats["buckets"]["0"] = self.stats["buckets"].get("0", 0) + 1
        return logits, hidden, {}

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
        """Run the installed physical-M4 route with host-owned n-gram inputs."""

        del return_hidden, hidden_variant
        self.stats["calls"] += 1
        return self._forward_installed_fixed_m4(
            input_ids,
            host_input_ids,
            completion_tokens,
            committed_count,
            cache,
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
                return self._runtime_forward(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                )
            # Shorter adaptive windows take the normal family capture route.
            # Not a fallback (no counter), but the Route Tape must still
            # see that this call ran eager through the bank.
            self.last_dispatch_kind = "eager"
            self.last_fallback_reason = "fixed_m4_short_window"
            self.last_fallback_transition = False
            return self._runtime_forward(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
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
            key = (length, str(hidden_variant or ""), int(bucket))
            fn = self._compiled.get(key)
            if fn is None:
                fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                self._compiled[key] = fn
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
            outputs = (
                fn(input_ids, compiled_aux, *state_in)
                if compiled_aux is not None
                else fn(input_ids, *state_in)
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
                key = (length, variant_key, int(bucket))
                fn = self._compiled.get(key)
                if fn is None:
                    # Shared-registry compile (F6): a bare per-bank
                    # mx.compile primed the Metal pipelines but kept the
                    # trace private to the warmup bank, so the first real
                    # request at the same shapes re-traced every bucket
                    # (~1s each) inside its measured row. The shared step
                    # is exactly what organic dispatch consults.
                    fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                    self._compiled[key] = fn
                bucket_started = time.perf_counter()
                outputs = fn(input_ids, *state_in)
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
                key = (length, variant_key, int(bucket))
                fn = self._compiled.get(key)
                if fn is None:
                    fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                    self._compiled[key] = fn
                length_started = time.perf_counter()
                outputs = fn(probe, *state_in)
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

    def demote(self, cache: Any) -> int:
        """Restore stock containers for every tensor-offset adapter in place.

        Mandatory before postcommit / final-state capture: downstream cache
        consumers must never see promoted adapters.
        """
        try:
            from .cache_state import TensorOffsetVllmMetalPagedKVCache
        except Exception:  # pragma: no cover - import guard for minimal test envs
            TensorOffsetVllmMetalPagedKVCache = None
        count = 0
        for idx, entry in enumerate(cache or []):
            if isinstance(entry, TensorOffsetQSACache):
                cache[idx] = entry.demote()
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
            # Container identity changed; compiled closures bound the old
            # shadow, which no longer mirrors the cache list.
            self._clear_shadow_leaf_refs()
            self._held_state_refs.clear()
            self._shadow = None
            self._shadow_signature = None
            self._spec = None
            self._compiled.clear()
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
            for length, variant, bucket in sorted(self._compiled)
        ]
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
                "growth_tokens": int(dispatch["growth_tokens"]),
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
        )
        self.stats["promoted"] += promoted
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
        if _BATCH_PAGED_OFFSETS and _PAGED_OFFSETS_CONTEXT_OK.get():
            # One eval for every paged offset instead of a serial sync per
            # entry inside size() below (#318; helper docstring has the
            # mechanism). Mirrors this loop's own iteration exactly.
            paged_offsets = []
            for spec_idx, spec_kind, _n in self._spec or []:
                if spec_kind != VERIFY_SPEC_KIND_FULL_ATTN:
                    continue
                spec_entry = cache[spec_idx]
                if not hasattr(spec_entry, "capacity"):
                    continue
                entry_state = getattr(spec_entry, "cache", None)
                if isinstance(entry_state, (list, tuple)) and len(entry_state) > 2:
                    entry_offset = entry_state[2]
                    if isinstance(entry_offset, mx.array):
                        paged_offsets.append(entry_offset)
            if paged_offsets:
                mx.eval(*paged_offsets)
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
                    # A twin that dropped this would silently revert to the
                    # stock QSA chain -- the armed-but-inert failure mode.
                    qsa_sparse_decode=entry.qsa_sparse_decode,
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        entry.cache[0],
                        entry.cache[1],
                        entry.cache[2],
                        step=entry.step,
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
                    )
                else:
                    twin = TensorOffsetVllmMetalPagedKVCache(
                        key_cache=entry.cache[0],
                        value_cache=entry.cache[1],
                        offset=entry.cache[2],
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
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

    # -- compiled function ------------------------------------------------------

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
        bank/request is safe. `_TRACE_HOSTS` keeps each callable's shadow and
        stats sink pointed at the LIVE bank so retraces never touch a dead
        request's containers.
        """

        if not _env_enabled("MTPLX_COMPILED_VERIFY_SHARED_TRACES", default=True):
            return mx.compile(self._make_verify_step(length, hidden_variant))
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
        )
        entry = _SHARED_VERIFY_STEPS.get(global_key)
        if entry is not None:
            fn, host, runtime_ref = entry
            # id() can be recycled after a model swap frees the old runtime;
            # a stale callable would replay graphs bound to freed weights.
            if runtime_ref() is self.runtime:
                host["bank_ref"] = weakref.ref(self)
                return fn
            _SHARED_VERIFY_STEPS.pop(global_key, None)
        # Programs may outlive requests. Keeping the bank here also keeps its
        # shadow KV and traced state alive after the request/session is gone.
        # The dispatch owns the bank; the process cache owns only the program.
        host = {"bank_ref": weakref.ref(self)}
        fn = mx.compile(
            self._make_verify_step(length, hidden_variant, trace_host=host)
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
        return fn

    def _make_verify_step(
        self,
        length: int,
        hidden_variant: str | None,
        trace_host: dict[str, Any] | None = None,
    ):
        spec = list(self._spec or [])
        layout = self._capture_layout()
        static_host = {"bank_ref": weakref.ref(self)}
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
            live.stats["traces"] += 1
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
                elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                    for slot in range(len(entry.rollback_state)):
                        entry.rollback_state[slot] = None
                else:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            # (2) The existing runtime forward, on shadow containers only.
            #
            # Sample the sparse-decode lane's route counters ACROSS the
            # forward. The routing decision is host-side and happens in THIS
            # python body, so a trace that ends with neither a route hit nor a
            # short-context decline is an armed flag that is not in the graph
            # -- and the graph is what the next few hundred cycles replay.
            # Raising here costs one trace; the alternative was a whole window.
            from .kernels import qsa_sparse_decode as _qsa_sparse_lane

            sparse_route_before = _qsa_sparse_lane.route_snapshot()
            with attention_phase("decode_verify"):
                result = live._runtime_forward(
                    input_ids,
                    cache=shadow,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
                    compiled_aux=compiled_aux,
                )
            # Trace-time engagement check on the graph this body just built,
            # fatal because the graph is what the next few hundred cycles
            # replay: an armed sparse-decode flag that never routed is an inert
            # arm. A no-op when the lane is unarmed (route_snapshot stays 0).
            _qsa_sparse_lane.assert_traced(
                length, before=sparse_route_before, where="compiled verify"
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
                continue
            cache_list = getattr(entry, "cache", None)
            if isinstance(cache_list, list):
                for slot in range(len(cache_list)):
                    cache_list[slot] = None
            rollback = getattr(entry, "rollback_state", None)
            if isinstance(rollback, list):
                for slot in range(len(rollback)):
                    rollback[slot] = None

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
        # Seed the clone BEFORE mirror-commit: the real entries still hold the
        # pre-step leaves here (the compiled step ran purely on the shadow).
        clone = self._parity2_clone_cache(cache, bucket)
        with attention_phase("decode_verify"):
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
        # Uncapped compare so mismatched_leaves is a true count, not a preview.
        report = compare_verify_outputs(
            reference,
            candidate,
            max_report_lines=len(reference) + len(candidate) + 8,
        )
        if report:
            self._record_parity2_divergence(report, reference, candidate, cache)
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
                    # A twin that dropped this would silently revert to the
                    # stock QSA chain -- the armed-but-inert failure mode.
                    qsa_sparse_decode=entry.qsa_sparse_decode,
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
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

    def _record_parity2_divergence(
        self,
        report: list[str],
        reference: dict[str, Any],
        candidate: dict[str, Any],
        cache: Any,
    ) -> None:
        self.stats["parity2_divergent_calls"] += 1
        ordinal = int(self.stats["calls"])
        context = self._parity2_context_estimate(cache)
        # Split on ": " (not ":"): state leaf names embed a colon, e.g.
        # "state[1:fa].2: value mismatch (...)".
        first_name = report[0].split(": ", 1)[0]
        artifact = _artifact_kind(first_name)
        max_abs = _leaf_max_abs_diff(
            reference.get(first_name), candidate.get(first_name)
        )
        mismatched = sum(1 for line in report if not line.startswith("... "))
        record = {
            "call": ordinal,
            "context": context,
            "artifact": artifact,
            "leaf": first_name,
            "max_abs_diff": max_abs,
            "mismatched_leaves": mismatched,
        }
        if self.stats["parity2_first_divergence"] is None:
            self.stats["parity2_first_divergence"] = record
        count = int(self.stats["parity2_divergent_calls"])
        if count <= 10:
            max_abs_text = "n/a" if max_abs is None else f"{max_abs:.3e}"
            print(
                f"[parity2] divergence call={ordinal} context={context} "
                f"artifact={artifact} leaf={first_name} "
                f"max_abs_diff={max_abs_text} mismatched_leaves={mismatched}",
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
