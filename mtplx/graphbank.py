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
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

import mlx.core as mx

from .attention_context import attention_phase
from .gdn_capture import resolve_gdn_capture_backend


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
        if isinstance(entry, TensorOffsetKVCache):
            entry.ensure_capacity(entry.size() + reserve_tokens)
            continue
        if preserve_paged:
            try:
                from .cache_state import (
                    TensorOffsetVllmMetalPagedKVCache,
                    VllmMetalPagedKVCache,
                )
            except Exception:  # pragma: no cover - import guard for minimal test envs
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
                if getattr(entry, "turboquant", False) or getattr(entry, "kv_quant", False):
                    # The tensor-offset adapter only understands plain bf16/fp16
                    # pages; promoting quantized pages would corrupt them.
                    failures["quantized_paged_kv_cache"] = (
                        failures.get("quantized_paged_kv_cache", 0) + 1
                    )
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
    "last_exception": None,
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
# (runtime id, route fingerprint, capture backend, state spec, verify length,
# hidden variant, bucket). The bank is per-generation; without sharing, every request pays a
# fresh trace. Values are (compiled_fn, trace_host) where trace_host["bank"]
# is re-pointed to the live bank before each dispatch so internal retraces
# (mx.compile re-traces on leaf-shape changes) always use live scratch
# containers. See CompiledVerifyBank._shared_or_new_verify_step.
_SHARED_VERIFY_STEPS: dict[tuple, tuple[Any, dict[str, Any]]] = {}


def _compiled_verify_route_fingerprint(runtime: Any) -> str:
    """Separate shared traces whose model-side optimization bindings differ."""

    route = getattr(runtime, "qwen38_route", None)
    return str(getattr(route, "fingerprint", "") or "")


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

    Full-attention tensor-offset entries contribute their three ``cache[0..2]``
    leaves; GDN ``ArraysCache`` entries contribute their two slots.  ``None``
    entries contribute nothing.  Any other container makes the cache
    non-compilable and returns ``(None, reason)``.
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
        if isinstance(entry, TensorOffsetKVCache) or (
            TensorOffsetVllmMetalPagedKVCache is not None
            and isinstance(entry, TensorOffsetVllmMetalPagedKVCache)
        ):
            spec.append((idx, VERIFY_SPEC_KIND_FULL_ATTN, 3))
            continue
        if ArraysCache is not None and isinstance(entry, ArraysCache):
            if len(entry.cache) != 2:
                return None, f"unsupported_container:ArraysCache[{len(entry.cache)}]"
            spec.append((idx, VERIFY_SPEC_KIND_GDN, 2))
            continue
        return None, f"unsupported_container:{type(entry).__name__}"
    return spec, None


def _paged_kernel_bucket_eligible(entry: Any, length: int, bucket: int) -> bool:
    """Best-effort eager mirror of sdpa_2pass_paged_tail_dynamic_offset gates.

    A miss here is a performance decision, not a correctness one: inside the
    compiled function the kernel declining simply routes to the pure dense
    ``cache.state`` math, which stays trace-safe.
    """
    key_cache = entry.cache[0]
    value_cache = entry.cache[1]
    if key_cache is None or value_cache is None:
        return False
    if not mx.metal.is_available():
        return False
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
_PAGED_OFFSETS_CONTEXT_OK: ContextVar[bool] = ContextVar(
    "mtplx_paged_offsets_context_ok", default=True
)


def set_paged_offsets_context_ok(allowed: bool):
    """Per-request fence stamp from generation's trio prebind."""
    return _PAGED_OFFSETS_CONTEXT_OK.set(bool(allowed))


def paged_offsets_context_ok() -> bool:
    """Read the current request's fence stamp (receipts/trace)."""
    return _PAGED_OFFSETS_CONTEXT_OK.get()


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
        # The request budget can only TIGHTEN the reserve, never raise it
        # past the env ceiling. Server requests default max_tokens to the
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
        # ceiling for known-budget batch runs.
        self.growth_reserve_tokens = (
            min(
                self.request_max_tokens + self.speculative_headroom,
                max(
                    _compiled_verify_growth_reserve(),
                    self.max_verify_len,
                ),
            )
            if self.request_max_tokens is not None
            else _compiled_verify_growth_reserve()
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
        self._compiled: dict[tuple[int, str, int], Any] = {}
        self._spec: list[tuple[int, str, int]] | None = None
        self._shadow: list[Any] | None = None
        self._shadow_signature: tuple[Any, ...] | None = None
        self._gdn_meta_cache: dict[int, dict[str, int] | None] = {}
        self._exception_failures = 0
        self._held_state_refs: list = []
        # Dense leaves that outgrow the capacity granted at first promotion
        # would retrace the compiled graph on every cache-growth step. A
        # generation request supplies its known output budget plus one maximum
        # speculative window; TensorOffsetKVCache.ensure_capacity rounds the
        # final offset + reserve to each entry's own step geometry. Standalone
        # callers without a request budget retain the legacy env reserve.
        self._growth_demoted = False
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
            )
            else 0
        )
        self.stats: dict[str, Any] = {
            "calls": 0,
            "compiled_calls": 0,
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
        }

    # -- public API ---------------------------------------------------------

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        global _PREWARM_DONE
        if (
            not _PREWARM_DONE
            and not self.parity
            and not self.parity2
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
        self.stats["calls"] += 1
        reason = self._fallback_reason(input_ids, cache, return_hidden)
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
            if boundary in ("both", "pre"):
                mx.async_eval(*state_in)
            outputs = fn(input_ids, *state_in)
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
            exception_detail = f"{type(exc).__name__}: {exc}"
            compiled_verify_status["transient_exception_count"] = (
                int(compiled_verify_status.get("transient_exception_count", 0)) + 1
            )
            compiled_verify_status["last_exception"] = exception_detail
            if self._exception_failures == 1:
                print(
                    "[mtplx] compiled-verify exception: " + exception_detail,
                    flush=True,
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
        bucket_key = str(int(bucket))
        self.stats["buckets"][bucket_key] = self.stats["buckets"].get(bucket_key, 0) + 1
        captures = self._rebuild_captures(captures_flat)
        if self.parity:
            return self._parity_check(
                input_ids,
                cache=cache,
                hidden_variant=hidden_variant,
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
                bucket=int(bucket),
                compiled_logits=logits,
                compiled_hidden=hidden,
                compiled_captures=captures,
                compiled_state_out=state_out,
            )
        self._mirror_commit(cache, state_out)
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
            if isinstance(entry, TensorOffsetKVCache):
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
        return data

    # -- dispatch preconditions ----------------------------------------------

    def _fallback_reason(
        self,
        input_ids,
        cache,
        return_hidden: bool,
        *,
        consume_post_restore: bool = True,
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
        if length > self.max_verify_len:
            return "length_outside_bank"
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
        from .cache_state import TensorOffsetVllmMetalPagedKVCache

        shadow: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        entry.cache[0],
                        entry.cache[1],
                        entry.cache[2],
                        step=entry.step,
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
        global_key = (
            id(self.runtime),
            _compiled_verify_route_fingerprint(self.runtime),
            self.capture_backend,
            spec_sig,
            int(length),
            str(hidden_variant or ""),
            int(key[2]),
        )
        entry = _SHARED_VERIFY_STEPS.get(global_key)
        if entry is not None:
            fn, host, runtime_ref = entry
            # id() can be recycled after a model swap frees the old runtime;
            # a stale callable would replay graphs bound to freed weights.
            if runtime_ref() is self.runtime:
                host["bank"] = self
                return fn
            _SHARED_VERIFY_STEPS.pop(global_key, None)
        host = {"bank": self}
        fn = mx.compile(
            self._make_verify_step(length, hidden_variant, trace_host=host)
        )
        _SHARED_VERIFY_STEPS[global_key] = (fn, host, weakref.ref(self.runtime))
        return fn

    def _make_verify_step(
        self,
        length: int,
        hidden_variant: str | None,
        trace_host: dict[str, Any] | None = None,
    ):
        spec = list(self._spec or [])
        layout = self._capture_layout()
        bank = self
        static_host = {"bank": self}
        host = trace_host if trace_host is not None else static_host

        del bank

        def verify_step(input_ids, *state_in):
            # Python body executes at trace time only; replays skip it.
            live = host["bank"]
            shadow = live._shadow
            live.stats["traces"] += 1
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            # (1) Re-seed firewall: every shadow leaf is assigned from the
            # explicit inputs BEFORE any read, so nothing stale and no tracer
            # from a previous trace can leak into this graph.
            pos = 0
            for idx, kind, n_leaves in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    entry.cache[0] = state_in[pos]
                    entry.cache[1] = state_in[pos + 1]
                    entry.cache[2] = state_in[pos + 2]
                    entry.rollback_state[0] = None
                    entry.rollback_state[1] = None
                    entry.rollback_state[2] = None
                else:
                    entry.cache[0] = state_in[pos]
                    entry.cache[1] = state_in[pos + 1]
                pos += n_leaves
            # (2) The existing runtime forward, on shadow containers only.
            with attention_phase("decode_verify"):
                result = live._runtime_forward(
                    input_ids,
                    cache=shadow,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
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
            state_out: list[Any] = []
            for idx, kind, _n in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    state_out.extend((entry.cache[0], entry.cache[1], entry.cache[2]))
                else:
                    state_out.extend((entry.cache[0], entry.cache[1]))
            return (logits, hidden, *captures_flat, *state_out)

        return verify_step

    def _capture_layout(self) -> tuple[str, ...]:
        if self.capture_backend == "linear_gdn_from_conv_tape":
            return TAPE_CAPTURE_KEYS
        return STANDARD_CAPTURE_KEYS

    def _unpack_outputs(self, outputs):
        spec = self._spec or []
        layout = self._capture_layout()
        n_captures = sum(
            len(layout) for _idx, kind, _n in spec if kind == VERIFY_SPEC_KIND_GDN
        )
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
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                layer_leaves = (entry.cache[0], entry.cache[1], entry.cache[2])
            else:
                layer_leaves = (entry.cache[0], entry.cache[1])
            if any(leaf is None for leaf in layer_leaves):
                return None
            leaves.extend(layer_leaves)
        return leaves

    def _mirror_commit(self, cache: Any, state_out: list[Any]) -> None:
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                entry.cache[0] = state_out[pos]
                entry.cache[1] = state_out[pos + 1]
                entry.cache[2] = state_out[pos + 2]
                # Cleared rollback forces trim() onto the offset-only branch,
                # which is the correct reject semantics for a batched verify.
                entry.rollback_state[0] = None
                entry.rollback_state[1] = None
                entry.rollback_state[2] = None
            else:
                entry.cache[0] = state_out[pos]
                entry.cache[1] = state_out[pos + 1]
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

    def _fallback(
        self,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        reason: str,
    ):
        self.stats["fallback_calls"] += 1
        self.stats["fallback_reasons"][reason] = (
            self.stats["fallback_reasons"].get(reason, 0) + 1
        )
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
            )
        eager_state = []
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                eager_state.extend((entry.cache[0], entry.cache[1], entry.cache[2]))
            else:
                eager_state.extend((entry.cache[0], entry.cache[1]))
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
            )
        # Compiled is authoritative: the live stream advances on compiled state.
        self._mirror_commit(cache, compiled_state_out)
        clone_state: list[Any] = []
        for idx, kind, _n in self._spec or []:
            entry = clone[idx]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                clone_state.extend((entry.cache[0], entry.cache[1], entry.cache[2]))
            else:
                clone_state.extend((entry.cache[0], entry.cache[1]))
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
        from .cache_state import TensorOffsetVllmMetalPagedKVCache

        clone: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_FULL_ATTN:
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        _copy_state_leaf(entry.cache[0]),
                        _copy_state_leaf(entry.cache[1]),
                        _copy_state_leaf(entry.cache[2]),
                        step=entry.step,
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
                for slot, leaf in enumerate(entry.cache):
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
