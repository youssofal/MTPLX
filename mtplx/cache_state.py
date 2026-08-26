"""Conservative cache snapshot helpers for correctness gates."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Any

from .attention_context import current_attention_phase
from .paged_cache import PagedCachePlan, PagedCachePool

SUPPORTED_DETACH_MODES = {
    "eval_only",
    "contiguous_eval",
    "selected_slice_contiguous_eval",
    "metal_copy_leaf",
}


@dataclass(frozen=True)
class CacheSnapshot:
    states: tuple[Any, ...]
    meta_states: tuple[Any, ...]


@dataclass(frozen=True)
class _PagedGQARouteDecision:
    route: str
    reason: str
    requested_route: str
    min_context: int
    min_q: int
    max_q: int


def _normalize_detach_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_DETACH_MODES:
        raise ValueError(
            "detach mode must be one of "
            f"{sorted(SUPPORTED_DETACH_MODES)}; got {mode!r}"
        )
    return normalized


class TailOwnedKVCache:
    """KV cache that owner-copies the newly produced attention tail.

    Stock MLX KV cache stores each new K/V slice directly into a persistent
    cache. This diagnostic keeps the same logical cache contract but cuts the
    tail tensor's lazy lineage before insertion, which is far cheaper than
    periodically copying the whole historical KV buffer.
    """

    def __init__(
        self,
        *,
        mode: str = "contiguous_eval",
        step: int = 256,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
    ) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)
        self.step = int(step)
        self.mode = _normalize_detach_mode(mode)
        self.tail_owner_updates = 0
        self.tail_owner_arrays = 0
        self.tail_owner_bytes = 0
        self.tail_owner_time_s = 0.0

    @classmethod
    def from_cache(cls, entry: Any, *, mode: str, step: int | None = None) -> "TailOwnedKVCache":
        return cls(
            mode=mode,
            step=int(step or getattr(entry, "step", 256)),
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
        )

    def _own_tail(self, keys: Any, values: Any) -> tuple[Any, Any]:
        started = time.perf_counter()
        owned_keys = detach_array_leaf(keys, mode=self.mode)
        owned_values = detach_array_leaf(values, mode=self.mode)
        self.tail_owner_time_s += time.perf_counter() - started
        self.tail_owner_updates += 1
        self.tail_owner_arrays += 2
        self.tail_owner_bytes += int(owned_keys.nbytes) + int(owned_values.nbytes)
        return owned_keys, owned_values

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        keys, values = self._own_tail(keys, values)
        prev = self.offset
        steps = int(keys.shape[2])
        if self.keys is None or (prev + steps) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + steps - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += steps
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self) -> int:
        return int(self.offset)

    @property
    def state(self):
        if self.keys is None or self.values is None:
            return self.keys, self.values
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, value) -> None:
        self.keys, self.values = value
        self.offset = 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.step), str(self.offset), self.mode)

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.step = int(value[0])
        self.offset = int(value[1])
        if len(value) > 2:
            self.mode = _normalize_detach_mode(str(value[2]))

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return int(self.keys.nbytes) + int(self.values.nbytes)

    def tail_owner_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "updates": int(self.tail_owner_updates),
            "arrays": int(self.tail_owner_arrays),
            "bytes": int(self.tail_owner_bytes),
            "time_s": float(self.tail_owner_time_s),
        }


class BlockOwnedKVCache(TailOwnedKVCache):
    """Full-attention KV cache with independent physical token blocks."""

    def __init__(
        self,
        *,
        mode: str = "contiguous_eval",
        block_size: int = 1024,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.key_blocks: list[Any] = []
        self.value_blocks: list[Any] = []
        self._pending_keys = None
        self._pending_values = None
        self._tail_shape: tuple[int, int, int] | None = None
        self._tail_dtypes: tuple[Any, Any] | None = None
        super().__init__(mode=mode, step=block_size, offset=0)
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(offset))

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        mode: str,
        block_size: int | None = None,
    ) -> "BlockOwnedKVCache":
        return cls(
            mode=mode,
            block_size=int(block_size or getattr(entry, "step", 1024) or 1024),
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
        )

    def _record_shape(self, keys: Any, values: Any) -> None:
        self._tail_shape = (
            int(keys.shape[0]),
            int(keys.shape[1]),
            int(keys.shape[3]),
        )
        self._tail_dtypes = (keys.dtype, values.dtype)

    def _new_blocks_like(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = int(values.shape[3])
        key_block = mx.zeros(
            (B, n_kv_heads, self.block_size, k_head_dim),
            dtype=keys.dtype,
        )
        value_block = mx.zeros(
            (B, n_kv_heads, self.block_size, v_head_dim),
            dtype=values.dtype,
        )
        mx.eval(key_block, value_block)
        return key_block, value_block

    def _ensure_capacity_for(self, absolute_pos: int, keys: Any, values: Any) -> None:
        needed_blocks = (int(absolute_pos) // self.block_size) + 1
        while len(self.key_blocks) < needed_blocks:
            key_block, value_block = self._new_blocks_like(keys, values)
            self.key_blocks.append(key_block)
            self.value_blocks.append(value_block)

    def _finalize_block(self, block_index: int) -> None:
        import mlx.core as mx

        key_block = mx.contiguous(self.key_blocks[block_index])
        value_block = mx.contiguous(self.value_blocks[block_index])
        mx.eval(key_block, value_block)
        self.key_blocks[block_index] = key_block
        self.value_blocks[block_index] = value_block

    def _load_contiguous_state(self, keys: Any, values: Any, offset: int) -> None:
        self.key_blocks = []
        self.value_blocks = []
        self._pending_keys = None
        self._pending_values = None
        self.offset = 0
        total = int(offset)
        if total <= 0:
            return
        self._record_shape(keys, values)
        cursor = 0
        while cursor < total:
            take = min(self.block_size, total - cursor)
            key_block, value_block = self._new_blocks_like(keys, values)
            key_tail = keys[..., cursor : cursor + take, :]
            value_tail = values[..., cursor : cursor + take, :]
            key_tail, value_tail = self._own_tail(key_tail, value_tail)
            key_block[..., :take, :] = key_tail
            value_block[..., :take, :] = value_tail
            self.key_blocks.append(key_block)
            self.value_blocks.append(value_block)
            if take == self.block_size:
                self._finalize_block(len(self.key_blocks) - 1)
            cursor += take
        self.offset = total

    @property
    def keys(self):
        return self._active_arrays()[0]

    @keys.setter
    def keys(self, value) -> None:
        if value is None:
            self.key_blocks = []
            self.value_blocks = []
            self.offset = 0
            self._pending_keys = None
            return
        self._pending_keys = value
        if self._pending_values is not None:
            self._load_contiguous_state(
                self._pending_keys,
                self._pending_values,
                int(value.shape[2]),
            )

    @property
    def values(self):
        return self._active_arrays()[1]

    @values.setter
    def values(self, value) -> None:
        if value is None:
            self._pending_values = None
            return
        self._pending_values = value
        if self._pending_keys is not None:
            self._load_contiguous_state(
                self._pending_keys,
                self._pending_values,
                int(value.shape[2]),
            )

    def _active_arrays(self) -> tuple[Any | None, Any | None]:
        import mlx.core as mx

        if self.offset <= 0 or not self.key_blocks:
            return None, None
        full_blocks = self.offset // self.block_size
        partial = self.offset % self.block_size
        key_parts = list(self.key_blocks[:full_blocks])
        value_parts = list(self.value_blocks[:full_blocks])
        if partial:
            key_parts.append(self.key_blocks[full_blocks][..., :partial, :])
            value_parts.append(self.value_blocks[full_blocks][..., :partial, :])
        if len(key_parts) == 1:
            return key_parts[0], value_parts[0]
        return mx.concatenate(key_parts, axis=2), mx.concatenate(value_parts, axis=2)

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        active_keys, active_values = self._active_arrays()
        return active_keys, active_values

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        self._record_shape(keys, values)
        steps = int(keys.shape[2])
        cursor = 0
        while cursor < steps:
            absolute_pos = self.offset
            block_index = absolute_pos // self.block_size
            in_block = absolute_pos % self.block_size
            take = min(steps - cursor, self.block_size - in_block)
            self._ensure_capacity_for(absolute_pos, keys, values)
            key_tail = keys[..., cursor : cursor + take, :]
            value_tail = values[..., cursor : cursor + take, :]
            key_tail, value_tail = self._own_tail(key_tail, value_tail)
            self.key_blocks[block_index][..., in_block : in_block + take, :] = key_tail
            self.value_blocks[block_index][..., in_block : in_block + take, :] = value_tail
            self.offset += take
            cursor += take
            if in_block + take == self.block_size:
                self._finalize_block(block_index)

    def active_block_slices(self) -> list[tuple[int, Any, Any]]:
        """Return active physical KV block slices as ``(start, keys, values)``."""
        blocks = []
        if self.offset <= 0 or not self.key_blocks:
            return blocks
        full_blocks = self.offset // self.block_size
        partial = self.offset % self.block_size
        for block_index in range(full_blocks):
            blocks.append(
                (
                    block_index * self.block_size,
                    self.key_blocks[block_index],
                    self.value_blocks[block_index],
                )
            )
        if partial:
            blocks.append(
                (
                    full_blocks * self.block_size,
                    self.key_blocks[full_blocks][..., :partial, :],
                    self.value_blocks[full_blocks][..., :partial, :],
                )
            )
        return blocks

    @property
    def state(self):
        return self._active_arrays()

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_blocks = []
        self.value_blocks = []
        self.offset = 0
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(keys.shape[2]))

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.block_size), str(self.offset), self.mode)

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.block_size = int(value[0])
        self.step = self.block_size
        self.offset = int(value[1])
        if len(value) > 2:
            self.mode = _normalize_detach_mode(str(value[2]))

    def empty(self) -> bool:
        return not self.key_blocks or self.offset <= 0

    @property
    def nbytes(self) -> int:
        total = 0
        for key_block, value_block in zip(self.key_blocks, self.value_blocks):
            total += int(key_block.nbytes) + int(value_block.nbytes)
        return total


def _vllm_metal_reference_path() -> Path:
    override = os.environ.get("MTPLX_VLLM_METAL_REPO")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "REFERENCES:TOOLS" / "vllm-metal"


def _paged_attention_impl_from_env() -> str:
    return (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "")
        .strip()
        .lower()
        .replace("-", "_")
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _paged_gqa_sdpa_route_decision_from_env(
    *,
    q_len: int,
    offset: int,
    query_heads: int,
    kv_heads: int,
) -> _PagedGQARouteDecision:
    raw = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE")
        or os.environ.get("MTPLX_PAGED_GQA_SDPA_ROUTE")
        or ""
    ).strip()
    if not raw and _env_truthy("MTPLX_VLLM_METAL_PAGED_GQA_SDPA"):
        raw = "auto"
    route = raw.lower().replace("-", "_")
    min_context = _env_int("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", 65536)
    min_q = _env_int("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q", 4)
    max_q = _env_int("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q", 5)
    if route in {"", "0", "false", "no", "off", "disabled"}:
        return _PagedGQARouteDecision("", "disabled", route, min_context, min_q, max_q)
    if route in {"1", "true", "yes", "on"}:
        route = "auto"
    if route not in {"auto", "grouped", "per_head", "async_per_head"}:
        return _PagedGQARouteDecision(
            "", "unsupported_route", route, min_context, min_q, max_q
        )
    q_len = int(q_len)
    offset = int(offset)
    query_heads = int(query_heads)
    kv_heads = int(kv_heads)
    if kv_heads <= 0 or query_heads <= 0 or query_heads == kv_heads:
        return _PagedGQARouteDecision("", "not_gqa", route, min_context, min_q, max_q)
    if query_heads % kv_heads:
        return _PagedGQARouteDecision(
            "", "query_heads_not_multiple_of_kv_heads", route, min_context, min_q, max_q
        )
    if offset < min_context:
        return _PagedGQARouteDecision(
            "", "context_lt_min", route, min_context, min_q, max_q
        )
    if q_len < min_q:
        return _PagedGQARouteDecision("", "q_len_lt_min", route, min_context, min_q, max_q)
    if q_len > max_q:
        return _PagedGQARouteDecision("", "q_len_gt_max", route, min_context, min_q, max_q)
    if route == "auto":
        route = "async_per_head"
    return _PagedGQARouteDecision(route, "enabled", route, min_context, min_q, max_q)


def _paged_gqa_sdpa_route_from_env(
    *,
    q_len: int,
    offset: int,
    query_heads: int,
    kv_heads: int,
) -> str:
    return _paged_gqa_sdpa_route_decision_from_env(
        q_len=q_len,
        offset=offset,
        query_heads=query_heads,
        kv_heads=kv_heads,
    ).route


_PAGED_GQA_SDPA_STREAMS: dict[int, list[Any]] = {}
_PAGED_GQA_SDPA_MASK_CACHE_MAX = 8
_PAGED_GQA_SDPA_MASK_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _paged_gqa_tail_causal_mask(q_len: int, kv_len: int) -> Any:
    import mlx.core as mx

    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    return k_pos <= q_pos


def _cached_paged_gqa_mask(key: tuple[Any, ...], source: Any, mask: Any) -> Any:
    if len(_PAGED_GQA_SDPA_MASK_CACHE) >= _PAGED_GQA_SDPA_MASK_CACHE_MAX:
        _PAGED_GQA_SDPA_MASK_CACHE.clear()
    _PAGED_GQA_SDPA_MASK_CACHE[key] = (source, mask)
    return mask


def _repeat_paged_gqa_mask(mask: Any, *, q_len: int, kv_len: int, gqa: int) -> Any:
    import mlx.core as mx

    if mask is None or (isinstance(mask, str) and mask == "causal"):
        key = ("causal", int(q_len), int(kv_len), int(gqa))
        cached = _PAGED_GQA_SDPA_MASK_CACHE.get(key)
        if cached is not None:
            return cached[1]
        mask = _paged_gqa_tail_causal_mask(q_len, kv_len)
        reps = [1] * mask.ndim
        reps[-2] = int(gqa)
        return _cached_paged_gqa_mask(key, None, mx.tile(mask, tuple(reps)))
    if not isinstance(mask, mx.array):
        return mask
    if int(mask.shape[-2]) != q_len:
        return mask
    key = (
        "array",
        id(mask),
        tuple(int(dim) for dim in mask.shape),
        str(mask.dtype),
        int(gqa),
    )
    cached = _PAGED_GQA_SDPA_MASK_CACHE.get(key)
    if cached is not None and cached[0] is mask:
        return cached[1]
    reps = [1] * mask.ndim
    reps[-2] = int(gqa)
    return _cached_paged_gqa_mask(key, mask, mx.tile(mask, tuple(reps)))


def _paged_gqa_sdpa_streams_for(kv_heads: int) -> list[Any]:
    import mlx.core as mx

    if kv_heads not in _PAGED_GQA_SDPA_STREAMS:
        _PAGED_GQA_SDPA_STREAMS[kv_heads] = [mx.new_stream(mx.gpu) for _ in range(kv_heads)]
    return _PAGED_GQA_SDPA_STREAMS[kv_heads]


def _paged_gqa_sdpa(
    *,
    queries: Any,
    keys: Any,
    values: Any,
    scale: float,
    mask: Any,
    route: str,
) -> Any | None:
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    if int(batch_size) != 1 or kv_heads <= 0 or query_heads % kv_heads:
        return None
    gqa = int(query_heads) // int(kv_heads)
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    grouped_mask = _repeat_paged_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
    if route == "grouped":
        output = scaled_dot_product_attention(
            grouped_queries,
            keys,
            values,
            cache=None,
            scale=scale,
            mask=grouped_mask,
        )
    elif route == "per_head":
        output = mx.concatenate(
            [
                scaled_dot_product_attention(
                    grouped_queries[:, head : head + 1, :, :],
                    keys[:, head : head + 1, :, :],
                    values[:, head : head + 1, :, :],
                    cache=None,
                    scale=scale,
                    mask=grouped_mask,
                )
                for head in range(kv_heads)
            ],
            axis=1,
        )
    elif route == "async_per_head":
        outputs = []
        for head, stream in enumerate(_paged_gqa_sdpa_streams_for(kv_heads)):
            with mx.stream(stream):
                output = scaled_dot_product_attention(
                    grouped_queries[:, head : head + 1, :, :],
                    keys[:, head : head + 1, :, :],
                    values[:, head : head + 1, :, :],
                    cache=None,
                    scale=scale,
                    mask=grouped_mask,
                )
                mx.async_eval(output)
                outputs.append(output)
        output = mx.concatenate(outputs, axis=1)
    else:
        return None
    return output.reshape(batch_size, kv_heads, gqa, q_len, head_dim).reshape(
        batch_size,
        query_heads,
        q_len,
        head_dim,
    )


def _dynamic_paged_num_blocks(*, block_size: int, configured_blocks: int) -> int:
    if not _env_truthy("MTPLX_DYNAMIC_PAGED_KV"):
        return int(configured_blocks)
    min_blocks = max(
        int(configured_blocks),
        _env_int("MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS", int(configured_blocks)),
    )
    request_tokens = max(0, _env_int("MTPLX_DYNAMIC_PAGED_KV_TOKENS", 0))
    previous_high_water = max(
        0,
        _env_int("MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER", 0),
    )
    session_tokens = int((previous_high_water * 3 + 1) // 2)
    margin = max(0, _env_int("MTPLX_DYNAMIC_PAGED_KV_MARGIN", 128))
    needed = max(request_tokens, session_tokens) + margin
    if needed <= margin:
        return min_blocks
    required_blocks = (needed + int(block_size) - 1) // int(block_size)
    return max(min_blocks, required_blocks)


def _paged_attention_requires_external_ops(
    *,
    turboquant_config: Any | None = None,
    kv_quant_config: Any | None = None,
) -> bool:
    if turboquant_config is not None:
        return True
    if kv_quant_config is not None:
        return False
    impl = _paged_attention_impl_from_env()
    if impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
        return False
    if impl == "sdpa_2pass_paged":
        return False
    if impl == "mlx_vector_paged":
        return _env_truthy("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN")
    return True


def _load_vllm_metal_ops():
    errors: list[str] = []
    override = os.environ.get("MTPLX_VLLM_METAL_REPO", "").strip()
    if override:
        repo = Path(override).expanduser()
        if not repo.exists():
            raise RuntimeError(f"MTPLX_VLLM_METAL_REPO does not exist: {repo}")
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        for name in ("vllm_metal.metal", "vllm_metal"):
            sys.modules.pop(name, None)
        try:
            from vllm_metal.metal import get_ops

            return get_ops()
        except Exception as exc:
            raise RuntimeError(
                "failed to load vllm-metal ops from MTPLX_VLLM_METAL_REPO="
                f"{repo}: {exc}"
            ) from exc

    try:
        metal = importlib.import_module("vllm_metal.metal")
        return metal.get_ops()
    except Exception as exc:
        errors.append(f"vendored vllm_metal.metal: {exc}")

    repo = _vllm_metal_reference_path()
    if repo.exists():
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        for name in ("vllm_metal.metal", "vllm_metal"):
            sys.modules.pop(name, None)
        try:
            from vllm_metal.metal import get_ops

            return get_ops()
        except Exception as exc:
            errors.append(f"reference checkout {repo}: {exc}")
    else:
        errors.append(f"reference checkout missing: {repo}")

    raise RuntimeError(
        "vllm-metal paged-attention ops are unavailable; "
        + "; ".join(errors)
        + ". Install MTPLX with its Darwin/arm64 dependencies or set "
        "MTPLX_VLLM_METAL_REPO to a working vllm-metal checkout."
    )


# Module-level latch so we only warn once per process when the optional
# vllm-metal external ops can't load. Subsequent calls stay silent.
_VLLM_METAL_OPS_UNAVAILABLE_WARNED = False


def _warn_vllm_metal_ops_unavailable(exc: BaseException, *, context: str) -> None:
    """Emit a single, plain-stderr warning when external ops are missing.

    We deliberately avoid the ``warnings`` module here because MTPLX runs under
    request-scoped warning filters that can swallow the message; a direct
    ``print`` to stderr is what the operator actually sees in the server log.
    """

    global _VLLM_METAL_OPS_UNAVAILABLE_WARNED
    if _VLLM_METAL_OPS_UNAVAILABLE_WARNED:
        return
    _VLLM_METAL_OPS_UNAVAILABLE_WARNED = True
    print(
        f"mtplx: vllm-metal external ops unavailable ({context}): {exc}; "
        "falling back to packaged paged-attention path. Install nanobind and "
        "build vllm_metal/paged_ops.cpp, or set MTPLX_VLLM_METAL_REPO, to "
        "re-enable the external ops.",
        file=sys.stderr,
        flush=True,
    )


def _load_vllm_metal_ops_optional(*, context: str):
    """Load vllm-metal ops if available; return ``None`` (and warn once) if not.

    Use this on the graceful-fallback paths where the caller has a valid
    in-tree alternative.  Use ``_load_vllm_metal_ops`` directly only for
    diagnostic / explicitly-required paths.
    """

    try:
        return _load_vllm_metal_ops()
    except RuntimeError as exc:
        _warn_vllm_metal_ops_unavailable(exc, context=context)
        return None


class VllmMetalPagedKVCache:
    """Preallocated full-attention KV pages backed by vLLM-Metal primitives.

    This is an opt-in diagnostic cache for Step 11B/11C.  It stores accepted K/V
    in physical token pages shaped like vLLM-Metal's paged attention kernel:
    ``[num_blocks, block_size, num_kv_heads, head_dim]``.  Logical positions are
    currently contiguous from zero, so the block table is trivial while still
    exercising the native paged read path.
    """

    def __init__(
        self,
        *,
        block_size: int = 16,
        num_blocks: int = 1024,
        keys: Any | None = None,
        values: Any | None = None,
        offset: int = 0,
        turboquant_config: Any | None = None,
        kv_quant_config: Any | None = None,
        allow_growth: bool | None = None,
        growth_limit_tokens: int | None = None,
    ) -> None:
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.offset = 0
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self._page_pool: PagedCachePool | None = None
        self._allow_growth = (
            _env_truthy("MTPLX_DYNAMIC_PAGED_KV")
            if allow_growth is None
            else bool(allow_growth)
        )
        self._growth_limit_tokens = (
            _env_int("MTPLX_CONTEXT_WINDOW_TOKENS", 0)
            if growth_limit_tokens is None
            else max(0, int(growth_limit_tokens))
        )
        self.turboquant_config = turboquant_config
        self.turboquant = turboquant_config is not None
        self.kv_quant_config = kv_quant_config
        self.kv_quant = kv_quant_config is not None
        self._shape: tuple[int, int, int] | None = None
        self._dtypes: tuple[Any, Any] | None = None
        self.update_calls = 0
        self.paged_attention_calls = 0
        self.partitioned_attention_calls = 0
        self.turboquant_attention_calls = 0
        self.kv_quant_attention_calls = 0
        self.gqa_sdpa_calls = 0
        self.gqa_sdpa_calls_by_route: dict[str, int] = {}
        self.gqa_sdpa_calls_by_phase: dict[str, int] = {}
        self.gqa_sdpa_route_misses_by_phase_reason: dict[str, int] = {}
        self.gqa_sdpa_route_misses_by_q_len: dict[str, int] = {}
        self.gqa_sdpa_last_route_miss: dict[str, int | str] = {}
        self.active_array_calls = 0
        self.active_array_time_s = 0.0
        self.kv_quant_dequant_calls = 0
        self.kv_quant_dequant_time_s = 0.0
        self.kv_quant_dequant_tokens = 0
        self.kv_quant_dequant_memo_hits = 0
        self.kv_quant_dequant_memo_rebuilds = 0
        self.kv_quant_kernel_calls = 0
        # Incremental dequant memo: a bf16 mirror of the quantized cache that
        # is extended tail-only per step. Without it every attention call on
        # the dequant fallback re-dequantized the whole prefix — O(context)
        # per token, the q8/q4 decode collapse. q8-only, sized to the offset
        # (geometric growth, never the paged capacity), and released when a
        # request latches the q8-kernel route — a persistent capacity-sized
        # mirror inverted the feature's memory promise (quantized + full
        # bf16 > plain bf16). q4 can never kernel, so it keeps no mirror at
        # all. dict: mirror_k, mirror_v (flat [rows, heads, dim]), tokens
        # (valid prefix rows).
        self._dequant_memo: dict[str, Any] | None = None
        # Per-request numerics route for kv_quant attention: None until the
        # request's first attention call latches "kernel" or "dequant" from
        # the offset it starts attending at. Deciding once per request keeps
        # temp-0 outputs on ONE math path — a per-call offset check switched
        # numerics mid-generation when a request crossed the two-pass
        # threshold.
        self._kv_quant_route: str | None = None
        self._kv_quant_route_offset = -1
        self.dense_fallback_calls = 0
        self.dense_fallback_calls_by_phase: dict[str, int] = {}
        self.paged_attention_bailouts_by_phase_reason: dict[str, int] = {}
        self.paged_attention_last_bailout: dict[str, int | str] = {}
        self.paged_attention_large_q_path = ""
        self.large_q_split_sdpa_fallback_calls = 0
        self.large_q_split_sdpa_fallback_calls_by_phase: dict[str, int] = {}
        self.partitioned_paged_calls = 0
        self.partitioned_paged_calls_by_phase: dict[str, int] = {}
        self.grow_events = 0
        self.cache_write_time_s = 0.0
        self.attention_time_s = 0.0
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(offset))

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        block_size: int = 16,
        num_blocks: int = 1024,
        turboquant_config: Any | None = None,
        kv_quant_config: Any | None = None,
    ) -> "VllmMetalPagedKVCache":
        return cls(
            block_size=block_size,
            num_blocks=num_blocks,
            keys=getattr(entry, "keys", None),
            values=getattr(entry, "values", None),
            offset=int(getattr(entry, "offset", 0)),
            turboquant_config=turboquant_config,
            kv_quant_config=kv_quant_config,
        )

    @property
    def allocated_blocks(self) -> int | None:
        """Physical block count of the live pages (None before allocation)."""
        return None if self.key_cache is None else int(self.key_cache.shape[0])

    @property
    def capacity(self) -> int:
        # Capacity is a fact about the allocated pages, not about the mutable
        # num_blocks claim — re-configs and snapshot restores stomp the claim
        # without reallocating, and a lying capacity skips the growth guard
        # into silent scatter truncation (#310).
        allocated_blocks = self.allocated_blocks
        return int(self.block_size) * int(
            self.num_blocks if allocated_blocks is None else allocated_blocks
        )

    @property
    def page_pool(self) -> PagedCachePool | None:
        return self._page_pool

    def _page_buffers(self) -> dict[str, Any]:
        buffers = {
            "key": self.key_cache,
            "value": self.value_cache,
            "key_scale": self.key_scale_cache,
            "value_scale": self.value_scale_cache,
            "key_zero": self.key_zero_cache,
        }
        return {name: value for name, value in buffers.items() if value is not None}

    def _rebuild_page_pool(self) -> None:
        buffers = self._page_buffers()
        if not buffers:
            self._page_pool = None
            return
        pool = PagedCachePool(
            PagedCachePlan.contiguous(
                block_size=self.block_size,
                num_blocks=self.num_blocks,
                array_names=tuple(buffers),
            ),
            offset=self.offset,
        )
        for name, buffer in buffers.items():
            pool.bind(
                name,
                row_shape=tuple(int(dim) for dim in buffer.shape[2:]),
                dtype=buffer.dtype,
                buffer=buffer,
            )
        self._page_pool = pool

    def _active_block_table(self, used_blocks: int) -> Any:
        if self._page_pool is None:
            raise RuntimeError("paged KV physical owner was not installed")
        return self._page_pool.block_table[: int(used_blocks)][None, :]

    def _grow_to_capacity(self, required_tokens: int) -> bool:
        if not self._allow_growth:
            return False
        allocated_blocks = self.allocated_blocks
        current_blocks = (
            int(self.num_blocks) if allocated_blocks is None else int(allocated_blocks)
        )
        required_blocks = (int(required_tokens) + self.block_size - 1) // self.block_size
        grown_blocks = max(
            required_blocks,
            int((current_blocks * 3 + 1) // 2),
            int(current_blocks) + 1,
        )
        window_tokens = int(self._growth_limit_tokens)
        if window_tokens > 0:
            # Geometric growth must not overshoot the serving context window
            # (#150: the 1.5x step at 100k+ ctx allocates GiBs of blocks no
            # request can ever address). A genuinely larger requirement still
            # wins — correctness over the clamp.
            window_blocks = (int(window_tokens) + self.block_size - 1) // self.block_size
            if window_blocks >= required_blocks:
                grown_blocks = min(
                    grown_blocks, max(window_blocks, int(current_blocks))
                )
        if grown_blocks <= current_blocks:
            self.num_blocks = int(current_blocks)
            return True
        if self.key_cache is None or self.value_cache is None:
            self.num_blocks = int(grown_blocks)
            self.grow_events += 1
            return True

        import mlx.core as mx

        extra_blocks = int(grown_blocks) - int(current_blocks)
        key_extra = mx.zeros(
            (extra_blocks, *self.key_cache.shape[1:]),
            dtype=self.key_cache.dtype,
        )
        value_extra = mx.zeros(
            (extra_blocks, *self.value_cache.shape[1:]),
            dtype=self.value_cache.dtype,
        )
        grown_arrays = [key_extra, value_extra]
        self.key_cache = mx.concatenate([self.key_cache, key_extra], axis=0)
        self.value_cache = mx.concatenate([self.value_cache, value_extra], axis=0)
        grown_arrays.extend([self.key_cache, self.value_cache])
        if self.key_scale_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.key_scale_cache.shape[1:]),
                dtype=self.key_scale_cache.dtype,
            )
            self.key_scale_cache = mx.concatenate([self.key_scale_cache, extra], axis=0)
            grown_arrays.extend([extra, self.key_scale_cache])
        if self.value_scale_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.value_scale_cache.shape[1:]),
                dtype=self.value_scale_cache.dtype,
            )
            self.value_scale_cache = mx.concatenate([self.value_scale_cache, extra], axis=0)
            grown_arrays.extend([extra, self.value_scale_cache])
        if self.key_zero_cache is not None:
            extra = mx.zeros(
                (extra_blocks, *self.key_zero_cache.shape[1:]),
                dtype=self.key_zero_cache.dtype,
            )
            self.key_zero_cache = mx.concatenate([self.key_zero_cache, extra], axis=0)
            grown_arrays.extend([extra, self.key_zero_cache])
        self.num_blocks = int(grown_blocks)
        self.grow_events += 1
        mx.eval(*grown_arrays)
        self._rebuild_page_pool()
        return True

    def _ensure_allocated(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        if int(keys.shape[0]) != 1:
            raise ValueError("VllmMetalPagedKVCache currently supports batch size 1")
        shape = (int(keys.shape[1]), int(keys.shape[3]), int(values.shape[3]))
        dtypes = (keys.dtype, values.dtype)
        if self.key_cache is not None:
            if self._shape != shape or self._dtypes != dtypes:
                raise ValueError(
                    "paged KV cache shape/dtype changed: "
                    f"had {self._shape}/{self._dtypes}, got {shape}/{dtypes}"
                )
            if self._page_pool is None:
                self._rebuild_page_pool()
            return
        # Fresh allocation: any surviving dequant mirror belongs to the
        # previous buffer's contents and must not be served against the new
        # one (single choke point for every reset -> reallocate path). The
        # kv_quant numerics route re-latches with the new contents too.
        self._invalidate_dequant_memo()
        self._reset_kv_quant_route()
        n_kv_heads, k_head_dim, v_head_dim = shape
        # Defense-in-depth: if a TurboQuant cache reaches first allocation but
        # the external vllm-metal ops can't load, gracefully degrade to the
        # plain paged layout instead of crashing the in-flight request. The
        # install-time path already drops turboquant_config when ops are
        # missing, but downstream callers (e.g. snapshot restore) may still
        # construct a TurboQuant cache without going through that gate.
        if self.turboquant and _load_vllm_metal_ops_optional(
            context="TurboQuant cache allocation"
        ) is None:
            self.turboquant = False
            self.turboquant_config = None
        if self.turboquant:
            from .turboquant import (
                SCALE_GROUP_SIZE,
                packed_dim,
                value_centroids,
                validate_head_dim,
            )

            validate_head_dim(k_head_dim)
            validate_head_dim(v_head_dim)
            cfg = self.turboquant_config
            key_dtype = mx.int8 if cfg.key_dtype_name == "int8" else mx.uint8
            self.key_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(k_head_dim, int(cfg.key_bits)),
                ),
                dtype=key_dtype,
            )
            self.value_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(v_head_dim, int(cfg.value_bits)),
                ),
                dtype=mx.uint8,
            )
            scale_shape = (
                self.num_blocks,
                self.block_size,
                n_kv_heads,
                k_head_dim // SCALE_GROUP_SIZE,
            )
            self.key_scale_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self.value_scale_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self.key_zero_cache = mx.zeros(scale_shape, dtype=mx.float16)
            self._turboquant_v_centroids = mx.array(
                value_centroids(int(cfg.value_bits)), dtype=mx.float32
            )
            mx.eval(
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
                self._turboquant_v_centroids,
            )
        elif self.kv_quant:
            from .kv_quant import packed_dim

            cfg = self.kv_quant_config
            bits = int(cfg.bits)
            cache_dtype = mx.int8 if bits == 8 else mx.uint8
            self.key_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(k_head_dim, bits),
                ),
                dtype=cache_dtype,
            )
            self.value_cache = mx.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    n_kv_heads,
                    packed_dim(v_head_dim, bits),
                ),
                dtype=cache_dtype,
            )
            scale_shape = (self.num_blocks, self.block_size, n_kv_heads, 1)
            # fp32 scales: quantize_symmetric computes them in fp32 and every
            # consumer multiplies in fp32; storing fp16 only added rounding
            # error (see kv_quant.quantize_symmetric).
            self.key_scale_cache = mx.zeros(scale_shape, dtype=mx.float32)
            self.value_scale_cache = mx.zeros(scale_shape, dtype=mx.float32)
            self.key_zero_cache = None
            mx.eval(
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
            )
        else:
            self.key_cache = mx.zeros(
                (self.num_blocks, self.block_size, n_kv_heads, k_head_dim),
                dtype=keys.dtype,
            )
            self.value_cache = mx.zeros(
                (self.num_blocks, self.block_size, n_kv_heads, v_head_dim),
                dtype=values.dtype,
            )
            mx.eval(self.key_cache, self.value_cache)
        self._shape = shape
        self._dtypes = dtypes
        self._rebuild_page_pool()

    def _write_tail(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        self._ensure_allocated(keys, values)
        steps = int(keys.shape[2])
        if self.offset + steps > self.capacity:
            required = self.offset + steps
            if not self._grow_to_capacity(required):
                raise ValueError(
                    f"paged KV cache capacity exceeded: {required} > {self.capacity}"
                )
        started = time.perf_counter()
        if self._page_pool is None:
            raise RuntimeError("paged KV physical owner was not installed")
        physical_blocks, block_offsets = self._page_pool.slot_mapping(
            self.offset, steps
        )
        slot_mapping = physical_blocks * self.block_size + block_offsets
        k_3d = mx.contiguous(keys[0].transpose(1, 0, 2))
        v_3d = mx.contiguous(values[0].transpose(1, 0, 2))
        if self.turboquant:
            if (
                self.key_scale_cache is None
                or self.value_scale_cache is None
                or self.key_zero_cache is None
            ):
                raise RuntimeError("TurboQuant scale caches were not allocated")
            cfg = self.turboquant_config
            ops = _load_vllm_metal_ops_optional(context="TurboQuant tq_encode")
            if ops is None or not hasattr(ops, "tq_encode"):
                # Graceful fallback: external ops are missing or stripped down.
                # Drop the TurboQuant snapshot path for this cache and reroute
                # to the plain paged layout. The cache is still empty for this
                # write (no prior tq_encode could have succeeded without ops),
                # so it is safe to re-allocate.
                if ops is not None and not hasattr(ops, "tq_encode"):
                    _warn_vllm_metal_ops_unavailable(
                        RuntimeError(
                            "local vLLM-Metal ops do not expose tq_encode"
                        ),
                        context="TurboQuant tq_encode",
                    )
                self.turboquant = False
                self.turboquant_config = None
                self.key_cache = None
                self.value_cache = None
                self.key_scale_cache = None
                self.value_scale_cache = None
                self.key_zero_cache = None
                self._page_pool = None
                self._shape = None
                self._dtypes = None
                self._ensure_allocated(keys, values)
                assert self._page_pool is not None
                self._page_pool.write_tail({"key": k_3d, "value": v_3d})
                self.offset = self._page_pool.offset
                self.update_calls += 1
                self.cache_write_time_s += time.perf_counter() - started
                return
            (
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
            ) = ops.tq_encode(
                k_3d,
                v_3d,
                self.key_cache,
                self.value_cache,
                self.key_scale_cache,
                self.value_scale_cache,
                self.key_zero_cache,
                slot_mapping,
                self._turboquant_v_centroids,
                int(cfg.value_bits),
                int(cfg.key_bits),
                bool(cfg.key_signed),
            )
            self._page_pool.replace_state(
                self._page_buffers(), self.offset + steps
            )
        elif self.kv_quant:
            if self.key_scale_cache is None or self.value_scale_cache is None:
                raise RuntimeError("paged KV quantization scale caches were not allocated")
            from .kv_quant import quantize_symmetric

            bits = int(self.kv_quant_config.bits)
            qk, k_scale = quantize_symmetric(k_3d, bits=bits)
            qv, v_scale = quantize_symmetric(v_3d, bits=bits)
            self._page_pool.write_tail(
                {
                    "key": qk,
                    "value": qv,
                    "key_scale": k_scale,
                    "value_scale": v_scale,
                }
            )
        else:
            self._page_pool.write_tail({"key": k_3d, "value": v_3d})
        self.offset = self._page_pool.offset
        self.update_calls += 1
        if self.kv_quant:
            phase = current_attention_phase()
            if phase == "prefill" or (phase == "unknown" and steps > 1):
                # A new prompt is being written: the per-request numerics
                # route re-latches at this request's first attention call.
                # Decode/verify appends (single-token steps, decode phases)
                # stay inside the current request's latched route.
                self._reset_kv_quant_route()
        self.cache_write_time_s += time.perf_counter() - started

    def _load_contiguous_state(self, keys: Any, values: Any, offset: int) -> None:
        self._invalidate_dequant_memo()
        self._reset_kv_quant_route()
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self._page_pool = None
        self._shape = None
        self._dtypes = None
        self.offset = 0
        total = min(int(offset), int(keys.shape[2]))
        if total <= 0:
            return
        self._write_tail(keys[..., :total, :], values[..., :total, :])

    def _partition_threshold(self) -> int:
        return _env_int("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", 2048)

    def _partitioned_attention_enabled(self) -> bool:
        return _env_truthy("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN")

    @staticmethod
    def _safe_2pass_paged_q_len(*, query_heads: int, kv_heads: int) -> int:
        """Max q_len that keeps the packaged paged-tail Metal threadgroup legal."""

        query_heads = max(1, int(query_heads))
        kv_heads = max(1, int(kv_heads))
        if query_heads % kv_heads:
            return 0
        gqa_factor = max(1, query_heads // kv_heads)
        return max(1, 1024 // max(1, 32 * gqa_factor))

    @staticmethod
    def _kv_quant_kernel_enabled() -> bool:
        """Kill-switch for the inline-dequant q8 kernel (default on).

        MTPLX_KV_QUANT_2PASS_KERNEL=0 restores the dequant-fallback dispatch
        for one release in case a field regression needs the old path.
        """
        raw = os.environ.get("MTPLX_KV_QUANT_2PASS_KERNEL")
        if raw is None or not raw.strip():
            return True
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _reset_kv_quant_route(self) -> None:
        self._kv_quant_route = None
        self._kv_quant_route_offset = -1

    def _kv_quant_route_decision(self, queries: Any, *, sliding_window: int) -> str:
        """Choose this request's kv_quant numerics path, once.

        Latched at the request's first attention call and held until a new
        prompt write or a buffer reload resets it: a request must not hop
        between kernel math and dequant math because its offset crossed the
        two-pass threshold mid-generation (temp-0 exactness — one request,
        one math path). trim() deliberately does NOT reset the route:
        speculative-verify rejections retract rows mid-request, and
        re-latching there would reintroduce the switch at the threshold
        boundary. Structural no-gos (q4, kill-switch, sliding window, GQA
        shapes the kernel refuses) latch "dequant"; otherwise the starting
        offset decides: below the threshold the memoized dequant path is
        cheap and the kernel has no KV-bandwidth advantage to harvest.
        """
        if (
            not self.kv_quant
            or self.turboquant
            or int(self.kv_quant_config.bits) != 8
            or not self._kv_quant_kernel_enabled()
            or int(sliding_window) > 0
            or self.key_cache is None
        ):
            return "dequant"
        if (
            self._safe_2pass_paged_q_len(
                query_heads=int(queries.shape[1]),
                kv_heads=int(self.key_cache.shape[2]),
            )
            < 1
        ):
            return "dequant"
        two_pass_threshold = int(
            os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD",
                "1024",
            )
            or "1024"
        )
        return "kernel" if int(self.offset) >= two_pass_threshold else "dequant"

    def _kv_quant_2pass_attention(
        self,
        queries: Any,
        *,
        scale: float,
        mask: Any | None,
        sliding_window: int,
        q_len: int,
    ) -> Any | None:
        """Decode/verify attention reading q8 pages directly (no dequant).

        This is the lane that makes q8 an actual memory feature during
        decode: no bf16 materialization at all. Only requests routed
        "kernel" (see _kv_quant_route_decision — the offset-vs-threshold
        call happens once per request, not per call) dispatch here;
        per-call eligibility mirrors the dense two-pass tail (causal,
        batch 1, no window, q_len within the threadgroup budget); anything
        else falls back for that call. The kernel's June closed-lane
        verdict (~0.8x) was measured against the DENSE kernel on
        unquantized caches — irrelevant here, where the alternative is the
        dequant fallback.
        """
        if (
            not self.kv_quant
            or self.turboquant
            or int(self.kv_quant_config.bits) != 8
            or not self._kv_quant_kernel_enabled()
        ):
            return None
        if mask is not None and not (isinstance(mask, str) and mask == "causal"):
            return None
        if int(sliding_window) > 0:
            return None
        if (
            self.key_cache is None
            or self.value_cache is None
            or self.key_scale_cache is None
            or self.value_scale_cache is None
        ):
            return None
        safe_q = self._safe_2pass_paged_q_len(
            query_heads=int(queries.shape[1]),
            kv_heads=int(self.key_cache.shape[2]),
        )
        if q_len > safe_q:
            return None
        from .kernels.sdpa_2pass_paged_q8 import sdpa_2pass_paged_q8_tail

        return sdpa_2pass_paged_q8_tail(
            queries=queries,
            key_q=self.key_cache,
            key_scales=self.key_scale_cache[..., 0],
            value_q=self.value_cache,
            value_scales=self.value_scale_cache[..., 0],
            offset=int(self.offset),
            block_size=int(self.block_size),
            scale=float(scale),
            max_q_len=safe_q,
        )

    def _long_context_dense_fallback_forbidden(self) -> bool:
        if _env_truthy("MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK"):
            return False
        if _env_truthy("MTPLX_ALLOW_PAGED_ACTIVE_ARRAY_SNAPSHOT"):
            return False
        # Only the QA assertion env var should turn the dense-fallback path
        # into a hard error. MTPLX_SUSTAINED_PREFILL is a *product* profile
        # flag (set by `--profile sustained`) and must not abort production
        # requests when the partitioned-paged kernel returns None - the
        # caller in attention_split.py already handles None by routing to
        # scaled_dot_product_attention with cache.state, which is correct.
        asserted = _env_truthy("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS")
        return asserted and int(self.offset) >= self._partition_threshold()

    def long_context_dense_fallback_forbidden(self) -> bool:
        return self._long_context_dense_fallback_forbidden()

    def _record_paged_bailout(
        self,
        reason: str,
        *,
        impl: str = "",
        offset: int | None = None,
        q_len: int | None = None,
        max_q_len: int | None = None,
        sliding_window: int | None = None,
        partitioned_enabled: bool | None = None,
        partition_threshold: int | None = None,
    ) -> None:
        phase = current_attention_phase()
        normalized_reason = (reason or "unknown").strip().lower() or "unknown"
        key = f"{phase}:{normalized_reason}"
        self.paged_attention_bailouts_by_phase_reason[key] = (
            int(self.paged_attention_bailouts_by_phase_reason.get(key, 0)) + 1
        )
        self.paged_attention_last_bailout = {
            "phase": phase,
            "reason": normalized_reason,
            "impl": str(impl or ""),
            "offset": int(self.offset if offset is None else offset),
            "q_len": int(q_len or 0),
            "max_q_len": int(max_q_len or 0),
            "block_size": int(self.block_size),
            "sliding_window": int(-1 if sliding_window is None else sliding_window),
            "partitioned_enabled": int(
                self._partitioned_attention_enabled()
                if partitioned_enabled is None
                else bool(partitioned_enabled)
            ),
            "partition_threshold": int(
                self._partition_threshold()
                if partition_threshold is None
                else partition_threshold
            ),
        }
        if _env_truthy("MTPLX_PAGED_ATTENTION_TRACE"):
            print(
                "mtplx_paged_attention_bailout "
                + " ".join(f"{k}={v}" for k, v in self.paged_attention_last_bailout.items()),
                file=sys.stderr,
            )

    def _record_gqa_route_miss(
        self,
        decision: _PagedGQARouteDecision,
        *,
        offset: int,
        q_len: int,
        query_heads: int,
        kv_heads: int,
    ) -> None:
        if not decision.requested_route or decision.reason in {"disabled", "enabled"}:
            return
        phase = current_attention_phase()
        reason = decision.reason.strip().lower() or "unknown"
        key = f"{phase}:{reason}"
        self.gqa_sdpa_route_misses_by_phase_reason[key] = (
            int(self.gqa_sdpa_route_misses_by_phase_reason.get(key, 0)) + 1
        )
        q_key = f"{phase}:q{int(q_len)}:{reason}"
        self.gqa_sdpa_route_misses_by_q_len[q_key] = (
            int(self.gqa_sdpa_route_misses_by_q_len.get(q_key, 0)) + 1
        )
        self.gqa_sdpa_last_route_miss = {
            "phase": phase,
            "reason": reason,
            "requested_route": str(decision.requested_route),
            "offset": int(offset),
            "q_len": int(q_len),
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "min_context": int(decision.min_context),
            "min_q": int(decision.min_q),
            "max_q": int(decision.max_q),
        }
        if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
            print(
                "mtplx_prefill_route "
                f"path=paged_gqa_sdpa_miss phase={phase} reason={reason} "
                f"route={decision.requested_route} offset={int(offset)} "
                f"q_len={int(q_len)} q_heads={int(query_heads)} kv_heads={int(kv_heads)} "
                f"min_context={int(decision.min_context)} "
                f"min_q={int(decision.min_q)} max_q={int(decision.max_q)}",
                file=sys.stderr,
            )

    def record_dense_fallback(self) -> None:
        self.dense_fallback_calls += 1
        phase = current_attention_phase()
        self.dense_fallback_calls_by_phase[phase] = (
            int(self.dense_fallback_calls_by_phase.get(phase, 0)) + 1
        )

    def _paged_range_flat(self, start: int, end: int) -> tuple[Any, Any]:
        """Rows [start:end) as flat [tokens, heads, dim], dequantized."""
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("paged KV cache is not allocated")
        start = max(0, int(start))
        end = min(int(end), int(self.offset))
        if end <= start:
            raise ValueError(f"invalid paged KV range: {start}:{end}")
        flat_k = self.key_cache.reshape(
            -1,
            int(self.key_cache.shape[2]),
            int(self.key_cache.shape[3]),
        )[start:end]
        flat_v = self.value_cache.reshape(
            -1,
            int(self.value_cache.shape[2]),
            int(self.value_cache.shape[3]),
        )[start:end]
        if self.kv_quant:
            if (
                self.key_scale_cache is None
                or self.value_scale_cache is None
                or self._shape is None
                or self._dtypes is None
            ):
                raise RuntimeError("paged KV quantization cache is incomplete")
            from .kv_quant import dequantize_symmetric

            bits = int(self.kv_quant_config.bits)
            flat_ks = self.key_scale_cache.reshape(
                -1,
                int(self.key_scale_cache.shape[2]),
                int(self.key_scale_cache.shape[3]),
            )[start:end]
            flat_vs = self.value_scale_cache.reshape(
                -1,
                int(self.value_scale_cache.shape[2]),
                int(self.value_scale_cache.shape[3]),
            )[start:end]
            flat_k = dequantize_symmetric(
                flat_k,
                flat_ks,
                bits=bits,
                head_dim=int(self._shape[1]),
            ).astype(self._dtypes[0])
            flat_v = dequantize_symmetric(
                flat_v,
                flat_vs,
                bits=bits,
                head_dim=int(self._shape[2]),
            ).astype(self._dtypes[1])
        return flat_k, flat_v

    def _paged_range(self, start: int, end: int) -> tuple[Any, Any]:
        flat_k, flat_v = self._paged_range_flat(start, end)
        return flat_k.transpose(1, 0, 2)[None, ...], flat_v.transpose(1, 0, 2)[None, ...]

    def _invalidate_dequant_memo(self) -> None:
        self._dequant_memo = None

    def _dequant_active_arrays(self) -> tuple[Any, Any]:
        """Full active K/V for kv_quant, dequantizing only the unseen tail.

        The bf16 mirror is a q8-only working set sized to the offset
        (geometric growth clamped to the paged capacity — never allocated
        AT capacity: a capacity-sized mirror inverted the feature's memory
        promise). It extends tail-only per step; trim() truncates its
        valid-token count (retracted rows are rewritten through _write_tail
        and re-dequantized); every buffer reallocation path drops it via
        _invalidate_dequant_memo; and the kernel route-latch drops it once
        per request. Growing the quantized store keeps it: flat row indices
        are append-stable. q4 can never reach the q8 kernel, so it keeps no
        mirror at all — a persistent bf16 copy on top of the quantized
        store would defeat the point of q4 — and materializes transiently
        instead.
        """
        import mlx.core as mx

        offset = int(self.offset)
        if int(self.kv_quant_config.bits) != 8:
            self.kv_quant_dequant_tokens += offset
            return self._paged_range(0, offset)
        if self._shape is None or self._dtypes is None:
            raise RuntimeError("paged KV quantization cache is incomplete")
        memo = self._dequant_memo
        if memo is None:
            memo = {"tokens": 0, "mirror_k": None, "mirror_v": None}
            self._dequant_memo = memo
            self.kv_quant_dequant_memo_rebuilds += 1
        if int(memo["tokens"]) > offset:
            # The offset moved backwards without trim() (meta_state rewind):
            # the mirror prefix below the new offset is still the dequant of
            # unchanged rows; anything above re-dequantizes when the offset
            # re-advances over rewrites.
            memo["tokens"] = offset
        valid = int(memo["tokens"])
        mirror_k = memo["mirror_k"]
        mirror_rows = 0 if mirror_k is None else int(mirror_k.shape[0])
        if offset > mirror_rows:
            capacity_rows = int(self.key_cache.shape[0]) * int(self.key_cache.shape[1])
            grown_rows = min(
                capacity_rows,
                max(offset, (mirror_rows * 3) // 2, int(self.block_size)),
            )
            grown_k = mx.zeros(
                (grown_rows, int(self.key_cache.shape[2]), int(self._shape[1])),
                dtype=self._dtypes[0],
            )
            grown_v = mx.zeros(
                (grown_rows, int(self.value_cache.shape[2]), int(self._shape[2])),
                dtype=self._dtypes[1],
            )
            if valid > 0:
                grown_k[:valid] = memo["mirror_k"][:valid]
                grown_v[:valid] = memo["mirror_v"][:valid]
            memo["mirror_k"] = grown_k
            memo["mirror_v"] = grown_v
        if valid < offset:
            tail_k, tail_v = self._paged_range_flat(valid, offset)
            memo["mirror_k"][valid:offset] = tail_k
            memo["mirror_v"][valid:offset] = tail_v
            memo["tokens"] = offset
            self.kv_quant_dequant_tokens += offset - valid
        else:
            self.kv_quant_dequant_memo_hits += 1
        keys = memo["mirror_k"][:offset].transpose(1, 0, 2)[None, ...]
        values = memo["mirror_v"][:offset].transpose(1, 0, 2)[None, ...]
        return keys, values

    def _large_q_split_sdpa_fallback(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int,
        mask: Any | None,
    ):
        import mlx.core as mx

        if self.turboquant:
            self._record_paged_bailout(
                "turboquant_unsupported",
                impl="large_q_split_sdpa",
                offset=int(self.offset),
                q_len=int(queries.shape[2]),
                sliding_window=int(sliding_window),
            )
            return None
        if mask is not None and not (isinstance(mask, str) and mask == "causal"):
            self._record_paged_bailout(
                "unsupported_mask",
                impl="large_q_split_sdpa",
                offset=int(self.offset),
                q_len=int(queries.shape[2]),
                sliding_window=int(sliding_window),
            )
            return None

        q_len = int(queries.shape[2])
        if _env_truthy("MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK"):
            raise RuntimeError(
                "large-q split SDPA fallback was invoked while "
                "MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK=1 "
                f"phase={current_attention_phase()} offset={int(self.offset)} "
                f"q_len={q_len} threshold={self._partition_threshold()}"
            )
        cached_prefix_len = max(0, int(self.offset) - q_len)
        query_heads = int(queries.shape[1])
        q_chunk_size = max(
            1,
            _env_int("MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE", 2048),
        )
        kv_chunk_size = max(
            1,
            _env_int("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", 1024),
        )
        key_start = 0
        if int(sliding_window) > 0:
            key_start = max(0, int(self.offset) - int(sliding_window))
        outputs: list[Any] = []
        very_negative = mx.array(-1.0e30, dtype=mx.float32)
        eps = mx.array(1.0e-20, dtype=mx.float32)
        # kv-quant stores values packed; _paged_range dequantizes them back to
        # the logical head dim recorded in _shape, so the accumulator must be
        # sized to the dequantized width, not the packed storage width (#150,
        # q4 crash on the paged split-SDPA path).
        if self.kv_quant and self._shape is not None:
            value_dim = int(self._shape[2])
        else:
            value_dim = int(self.value_cache.shape[3])

        for q_start in range(0, q_len, q_chunk_size):
            q_end = min(q_len, q_start + q_chunk_size)
            q = queries[:, :, q_start:q_end, :].astype(mx.float32)
            q_positions = cached_prefix_len + mx.arange(q_start, q_end)
            max_key_for_chunk = int(cached_prefix_len + q_end)
            running_max = mx.full(
                (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), 1),
                very_negative,
                dtype=mx.float32,
            )
            running_denom = mx.zeros_like(running_max)
            running_acc = mx.zeros(
                (
                    int(q.shape[0]),
                    int(q.shape[1]),
                    int(q.shape[2]),
                    value_dim,
                ),
                dtype=mx.float32,
            )
            for k_start in range(key_start, min(int(self.offset), max_key_for_chunk), kv_chunk_size):
                k_end = min(int(self.offset), max_key_for_chunk, k_start + kv_chunk_size)
                if k_end <= k_start:
                    continue
                keys, values = self._paged_range(k_start, k_end)
                kv_heads = int(keys.shape[1])
                if kv_heads != query_heads and query_heads % kv_heads:
                    return None
                k = keys.astype(mx.float32)
                v = values.astype(mx.float32)
                repeat = query_heads // kv_heads if kv_heads != query_heads else 1
                if repeat > 1:
                    q_for_scores = q.reshape(
                        int(q.shape[0]),
                        kv_heads,
                        repeat,
                        int(q.shape[2]),
                        int(q.shape[3]),
                    )
                    k_for_scores = k[:, :, None, :, :]
                    scores = mx.matmul(
                        q_for_scores,
                        k_for_scores.transpose(0, 1, 2, 4, 3),
                    ).reshape(
                        int(q.shape[0]),
                        query_heads,
                        int(q.shape[2]),
                        int(k.shape[2]),
                    ) * float(scale)
                else:
                    scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * float(scale)
                # The native paged kernels are causal when no explicit mask is
                # supplied. Preserve that contract in the in-tree fallback;
                # treating ``None`` as unmasked would let a multi-token query
                # read later keys from the same update.
                if mask is None or mask == "causal":
                    key_positions = mx.arange(k_start, k_end)
                    allowed = q_positions[:, None] >= key_positions[None, :]
                    valid = mx.any(allowed, axis=-1, keepdims=True)
                    scores = mx.where(allowed[None, None, :, :], scores, very_negative)
                else:
                    valid = mx.ones(scores.shape[:-1] + (1,), dtype=mx.bool_)
                local_max = mx.max(scores, axis=-1, keepdims=True)
                local_max = mx.where(valid, local_max, very_negative)
                weights = mx.where(valid, mx.exp(scores - local_max), 0.0)
                local_denom = mx.sum(weights, axis=-1, keepdims=True)
                if repeat > 1:
                    local_acc = mx.matmul(
                        weights.reshape(
                            int(q.shape[0]),
                            kv_heads,
                            repeat,
                            int(q.shape[2]),
                            int(k.shape[2]),
                        ),
                        v[:, :, None, :, :],
                    ).reshape(
                        int(q.shape[0]),
                        query_heads,
                        int(q.shape[2]),
                        int(v.shape[3]),
                    )
                else:
                    local_acc = mx.matmul(weights, v)
                new_max = mx.maximum(running_max, local_max)
                old_scale = mx.exp(running_max - new_max)
                new_scale = mx.exp(local_max - new_max)
                new_scale = mx.where(valid, new_scale, 0.0)
                running_acc = running_acc * old_scale + local_acc * new_scale
                running_denom = running_denom * old_scale + local_denom * new_scale
                running_max = new_max
            outputs.append(running_acc / mx.maximum(running_denom, eps))

        if not outputs:
            return None
        self.large_q_split_sdpa_fallback_calls += 1
        phase = current_attention_phase()
        self.large_q_split_sdpa_fallback_calls_by_phase[phase] = (
            int(self.large_q_split_sdpa_fallback_calls_by_phase.get(phase, 0)) + 1
        )
        self.paged_attention_large_q_path = "large_q_split_sdpa_fallback"
        if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
            print(
                "mtplx_prefill_route "
                f"path=large_q_split_sdpa_fallback phase={phase} "
                f"offset={int(self.offset)} q_len={q_len} "
                f"q_chunk={q_chunk_size} kv_chunk={kv_chunk_size}",
                file=sys.stderr,
            )
        return mx.concatenate(outputs, axis=2).astype(queries.dtype)

    def _active_arrays(self) -> tuple[Any | None, Any | None]:
        started = time.perf_counter()
        self.active_array_calls += 1
        try:
            if (
                _env_truthy("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS")
                and self._long_context_dense_fallback_forbidden()
            ):
                raise RuntimeError(
                    "Paged KV cache attempted to materialize active K/V arrays in "
                    f"the long-context path phase={current_attention_phase()} "
                    f"offset={int(self.offset)} threshold={self._partition_threshold()}"
                )
            if self.key_cache is None or self.value_cache is None or self.offset <= 0:
                return None, None
            if self.turboquant:
                return None, None
            if self.kv_quant:
                dequant_started = time.perf_counter()
                keys, values = self._dequant_active_arrays()
                self.kv_quant_dequant_calls += 1
                self.kv_quant_dequant_time_s += time.perf_counter() - dequant_started
                return keys, values
            flat_k = self.key_cache.reshape(
                -1,
                int(self.key_cache.shape[2]),
                int(self.key_cache.shape[3]),
            )[: self.offset]
            flat_v = self.value_cache.reshape(
                -1,
                int(self.value_cache.shape[2]),
                int(self.value_cache.shape[3]),
            )[: self.offset]
            return flat_k.transpose(1, 0, 2)[None, ...], flat_v.transpose(1, 0, 2)[None, ...]
        finally:
            self.active_array_time_s += time.perf_counter() - started

    @property
    def keys(self):
        return self._active_arrays()[0]

    @keys.setter
    def keys(self, value) -> None:
        if value is None:
            self.key_cache = None
            self.value_cache = None
            self.key_scale_cache = None
            self.value_scale_cache = None
            self.key_zero_cache = None
            self._page_pool = None
            self.offset = 0
            return
        if self.value_cache is not None:
            self._load_contiguous_state(value, self.values, int(value.shape[2]))

    @property
    def values(self):
        return self._active_arrays()[1]

    @values.setter
    def values(self, value) -> None:
        if value is None:
            self.key_cache = None
            self.value_cache = None
            self.key_scale_cache = None
            self.value_scale_cache = None
            self.key_zero_cache = None
            self._page_pool = None
            self.offset = 0
            return
        if self.key_cache is not None:
            self._load_contiguous_state(self.keys, value, int(value.shape[2]))

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        return self._active_arrays()

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        self._write_tail(keys, values)

    def size(self) -> int:
        return int(self.offset)

    @property
    def state(self):
        return self._active_arrays()

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_cache = None
        self.value_cache = None
        self.key_scale_cache = None
        self.value_scale_cache = None
        self.key_zero_cache = None
        self._page_pool = None
        self.offset = 0
        self._shape = None
        self._dtypes = None
        if keys is not None and values is not None:
            self._load_contiguous_state(keys, values, int(keys.shape[2]))

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (str(self.block_size), str(self.num_blocks), str(self.offset))

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        if self.key_cache is None:
            self.block_size = int(value[0])
            self.num_blocks = int(value[1])
            self.offset = int(value[2])
            return
        # `state` already rebuilt these pages; the snapshot's block count
        # describes a buffer that no longer exists (#310). The live pages own
        # the geometry — only the offset is restored, and it must fit them.
        self.num_blocks = int(self.key_cache.shape[0])
        offset = int(value[2])
        if offset > self.capacity:
            raise ValueError(
                "restored paged KV offset exceeds page capacity: "
                f"{offset} > {self.capacity}"
            )
        self.offset = offset
        # The page pool owns the live logical-to-physical cursor. Rebuild it
        # over the existing buffers after restoring the validated offset so a
        # subsequent write starts at the restored position, not the position
        # at which ``state`` rebuilt the pages.
        self._rebuild_page_pool()

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        if self._page_pool is not None:
            self._page_pool.truncate(self.offset)
        if self._dequant_memo is not None:
            # Retracted rows are rewritten via _write_tail before reuse; the
            # mirror prefix below the new offset is still exact.
            self._dequant_memo["tokens"] = min(
                int(self._dequant_memo["tokens"]), int(self.offset)
            )
        # The kv_quant numerics route deliberately survives trim():
        # speculative-verify rejections retract rows mid-request, and
        # re-latching here would switch math when a rejection lands the
        # offset back across the two-pass threshold. The next prompt write
        # is the request boundary that re-latches.
        return n

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self.key_cache is None or self.offset <= 0

    @property
    def nbytes(self) -> int:
        if self.key_cache is None or self.value_cache is None:
            return 0
        total = int(self.key_cache.nbytes) + int(self.value_cache.nbytes)
        for extra in (
            self.key_scale_cache,
            self.value_scale_cache,
            self.key_zero_cache,
        ):
            if extra is not None:
                total += int(extra.nbytes)
        memo = self._dequant_memo
        if memo is not None and memo.get("mirror_k") is not None:
            # The live dequant mirror is real memory; hiding it from the
            # bytes stat is how the kv-quant memory inversion went unnoticed.
            total += int(memo["mirror_k"].nbytes) + int(memo["mirror_v"].nbytes)
        return total

    def _effective_sliding_window(self, requested: int) -> int:
        raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW")
        if raw is None or not raw.strip():
            return int(requested)
        return int(raw)

    def _active_attention_arrays(self, sliding_window: int) -> tuple[Any | None, Any | None]:
        keys, values = self._active_arrays()
        if keys is None or values is None or int(sliding_window) <= 0:
            return keys, values
        take = min(int(sliding_window), int(keys.shape[2]))
        return keys[..., -take:, :], values[..., -take:, :]

    def paged_attention(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int = -1,
        mask: Any | None = None,
        impl_override: str | None = None,
    ):
        import mlx.core as mx

        started = time.perf_counter()
        q_len = int(queries.shape[2]) if hasattr(queries, "shape") and len(queries.shape) >= 3 else 0
        sliding_window = self._effective_sliding_window(sliding_window)
        impl_source = (
            impl_override
            if impl_override is not None
            else os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "")
        )
        impl = impl_source.strip().lower().replace("-", "_")
        max_q_len = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16")
        partitioned_enabled = self._partitioned_attention_enabled()
        partition_threshold = self._partition_threshold()

        def bailout(reason: str) -> None:
            self._record_paged_bailout(
                reason,
                impl=impl,
                offset=int(self.offset),
                q_len=q_len,
                max_q_len=max_q_len,
                sliding_window=int(sliding_window),
                partitioned_enabled=partitioned_enabled,
                partition_threshold=partition_threshold,
            )
            return None

        def run_partitioned_paged(*, force_fp32_paged: bool = False):
            if self.turboquant:
                return bailout("turboquant_unsupported")
            if self.kv_quant:
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.kv_quant_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
                return bailout("kv_quant_partitioned_unsupported")
            if self.key_cache is None or self.value_cache is None:
                return bailout("empty_cache")
            kernel_queries = queries.astype(mx.float32) if force_fp32_paged else queries
            kernel_key_cache = (
                self.key_cache.astype(mx.float32)
                if force_fp32_paged
                else self.key_cache
            )
            kernel_value_cache = (
                self.value_cache.astype(mx.float32)
                if force_fp32_paged
                else self.value_cache
            )
            q_3d = mx.contiguous(kernel_queries[0].transpose(1, 0, 2))
            used_blocks = (int(self.offset) + int(self.block_size) - 1) // int(self.block_size)
            if used_blocks <= 0:
                return bailout("blocks_invalid")
            block_tables = self._active_block_table(used_blocks)
            seq_lens = mx.array([int(self.offset)], dtype=mx.int32)
            cu_seqlens_q = mx.array([0, q_len], dtype=mx.int32)
            partition_size = int(
                os.environ.get("MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE") or "512"
            )
            max_num_partitions = max(
                1,
                (int(self.offset) + partition_size - 1) // partition_size,
            )
            try:
                ops = _load_vllm_metal_ops()
            except RuntimeError:
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
                return bailout("partitioned_unavailable")
            if _env_truthy("MTPLX_VLLM_METAL_PAGED_USE_PRIMITIVE") and hasattr(
                ops,
                "paged_attention_partitioned_primitive",
            ):
                out = mx.array(0)
                ops.paged_attention_partitioned_primitive(
                    q_3d,
                    kernel_key_cache,
                    kernel_value_cache,
                    int(kernel_key_cache.shape[2]),
                    float(scale),
                    0.0,
                    block_tables,
                    seq_lens,
                    cu_seqlens_q,
                    int(self.block_size),
                    int(self.offset),
                    int(sliding_window),
                    out,
                )
                self.paged_attention_calls += 1
                self.partitioned_attention_calls += 1
                self.partitioned_paged_calls += 1
                phase = current_attention_phase()
                self.partitioned_paged_calls_by_phase[phase] = (
                    int(self.partitioned_paged_calls_by_phase.get(phase, 0)) + 1
                )
                self.attention_time_s += time.perf_counter() - started
                self.paged_attention_large_q_path = "partitioned_paged"
                if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
                    print(
                        "mtplx_prefill_route "
                        f"path=partitioned_paged phase={phase} "
                        f"offset={int(self.offset)} q_len={q_len} "
                        f"partition_size=primitive",
                        file=sys.stderr,
                    )
                out = out.transpose(1, 0, 2)[None, ...]
                return out.astype(queries.dtype) if force_fp32_paged else out
            if not hasattr(ops, "paged_attention_v2_online_partitioned"):
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
                return bailout("partitioned_unavailable")
            out = mx.zeros(q_3d.shape, dtype=q_3d.dtype)
            exp_sums = mx.zeros(
                (q_len, int(q_3d.shape[1]), max_num_partitions),
                dtype=mx.float32,
            )
            max_logits = mx.zeros(
                (q_len, int(q_3d.shape[1]), max_num_partitions),
                dtype=mx.float32,
            )
            tmp_out = mx.zeros(
                (
                    q_len,
                    int(q_3d.shape[1]),
                    max_num_partitions,
                    int(q_3d.shape[2]),
                ),
                dtype=q_3d.dtype,
            )
            mx.eval(
                out,
                q_3d,
                self.key_cache,
                self.value_cache,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                exp_sums,
                max_logits,
                tmp_out,
            )
            ops.paged_attention_v2_online_partitioned(
                out,
                q_3d,
                kernel_key_cache,
                kernel_value_cache,
                int(kernel_key_cache.shape[2]),
                float(scale),
                0.0,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                int(self.block_size),
                int(self.offset),
                int(sliding_window),
                exp_sums,
                max_logits,
                tmp_out,
            )
            mx.synchronize()
            self.paged_attention_calls += 1
            self.partitioned_attention_calls += 1
            self.partitioned_paged_calls += 1
            phase = current_attention_phase()
            self.partitioned_paged_calls_by_phase[phase] = (
                int(self.partitioned_paged_calls_by_phase.get(phase, 0)) + 1
            )
            self.attention_time_s += time.perf_counter() - started
            self.paged_attention_large_q_path = "partitioned_paged"
            if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
                print(
                    "mtplx_prefill_route "
                    f"path=partitioned_paged phase={phase} "
                    f"offset={int(self.offset)} q_len={q_len} "
                    f"partition_size={partition_size} partitions={max_num_partitions}",
                    file=sys.stderr,
                )
            out = out.transpose(1, 0, 2)[None, ...]
            return out.astype(queries.dtype) if force_fp32_paged else out

        if self.key_cache is None or self.value_cache is None or self.offset <= 0:
            return bailout("empty_cache")
        if int(queries.shape[0]) != 1:
            return bailout("batch_not_1")
        if q_len <= 0:
            return bailout("q_len_invalid")
        if self.turboquant and impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
            raise ValueError("TurboQuant cannot use exact-gather attention")
        if not self.turboquant and impl in {"fast_sdpa_gather", "sdpa_gather", "exact_gather"}:
            from mlx_lm.models.base import scaled_dot_product_attention

            keys, values = self._active_attention_arrays(sliding_window)
            if keys is None or values is None:
                return bailout("empty_cache")
            out = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=None,
                scale=scale,
                mask=mask,
            )
            self.paged_attention_calls += 1
            if self.kv_quant:
                # This branch serves kv_quant traffic through the dequant
                # gather; without the increment the dashboard undercounted
                # quantized attention calls on exactly this hot path.
                self.kv_quant_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
            return out
        if not self.turboquant and not self.kv_quant and impl in {"sdpa_2pass_paged", "mlx_vector_paged"}:
            from .kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail

            two_pass_threshold = int(
                os.environ.get(
                    "MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD",
                    "1024",
                )
                or "1024"
            )
            if int(self.offset) < two_pass_threshold:
                from mlx_lm.models.base import scaled_dot_product_attention

                keys, values = self._active_attention_arrays(sliding_window)
                if keys is None or values is None:
                    return None
                out = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=scale,
                    mask=mask,
                )
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return out
            gqa_decision = _paged_gqa_sdpa_route_decision_from_env(
                q_len=q_len,
                offset=int(self.offset),
                query_heads=int(queries.shape[1]),
                kv_heads=int(self.key_cache.shape[2]),
            )
            gqa_route = gqa_decision.route
            if gqa_route:
                keys, values = self._active_attention_arrays(sliding_window)
                if keys is not None and values is not None:
                    out = _paged_gqa_sdpa(
                        queries=queries,
                        keys=keys,
                        values=values,
                        scale=float(scale),
                        mask=mask,
                        route=gqa_route,
                    )
                    if out is not None:
                        self.paged_attention_calls += 1
                        self.gqa_sdpa_calls += 1
                        self.gqa_sdpa_calls_by_route[gqa_route] = (
                            int(self.gqa_sdpa_calls_by_route.get(gqa_route, 0)) + 1
                        )
                        phase = current_attention_phase()
                        self.gqa_sdpa_calls_by_phase[phase] = (
                            int(self.gqa_sdpa_calls_by_phase.get(phase, 0)) + 1
                        )
                        self.attention_time_s += time.perf_counter() - started
                        if _env_truthy("MTPLX_PREFILL_ROUTE_TRACE"):
                            print(
                                "mtplx_prefill_route "
                                f"path=paged_gqa_sdpa route={gqa_route} phase={phase} "
                                f"offset={int(self.offset)} q_len={q_len}",
                                file=sys.stderr,
                        )
                        return out
            else:
                self._record_gqa_route_miss(
                    gqa_decision,
                    offset=int(self.offset),
                    q_len=q_len,
                    query_heads=int(queries.shape[1]),
                    kv_heads=int(self.key_cache.shape[2]),
                )
            safe_tail_q_len = self._safe_2pass_paged_q_len(
                query_heads=int(queries.shape[1]),
                kv_heads=int(self.key_cache.shape[2]),
            )
            effective_max_q_len = min(max_q_len, safe_tail_q_len)
            if q_len > effective_max_q_len:
                if partitioned_enabled and int(self.offset) >= partition_threshold:
                    self.paged_attention_large_q_path = "partitioned_paged"
                    return run_partitioned_paged(force_fp32_paged=False)
                return bailout("q_len_gt_max")
            out = sdpa_2pass_paged_tail(
                queries=queries,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                offset=int(self.offset),
                block_size=int(self.block_size),
                scale=float(scale),
                mask=mask,
                max_q_len=effective_max_q_len,
                sliding_window=int(sliding_window),
            )
            if out is not None:
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return out
            return bailout("kernel_unavailable")
        if self.kv_quant:
            if self._kv_quant_route is None:
                self._kv_quant_route = self._kv_quant_route_decision(
                    queries, sliding_window=int(sliding_window)
                )
                self._kv_quant_route_offset = int(self.offset)
                if self._kv_quant_route == "kernel":
                    # The kernel owns this request's decode: any prefill-era
                    # bf16 mirror is dead weight, released exactly once,
                    # here at latch. A later shape-driven dequant call (a
                    # verify burst past the kernel's q budget, an exotic
                    # mask) may rebuild it and keep it tail-extended —
                    # kernel calls never re-release, which would thrash
                    # full-prefix rebuild stalls.
                    self._invalidate_dequant_memo()
            if self._kv_quant_route == "kernel":
                kernel_out = self._kv_quant_2pass_attention(
                    queries,
                    scale=scale,
                    mask=mask,
                    sliding_window=int(sliding_window),
                    q_len=q_len,
                )
                if kernel_out is not None:
                    self.paged_attention_calls += 1
                    self.kv_quant_attention_calls += 1
                    self.kv_quant_kernel_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return kernel_out
            elif int(self.kv_quant_config.bits) != 8:
                # q4 keeps no mirror, so the full-width bf16 fallback below
                # would re-materialize offset-sized K/V on every step. The
                # chunked online-softmax path dequantizes in bounded
                # windows and is the lane that keeps q4 an actual memory
                # feature; the rare shapes it declines (non-causal array
                # masks, ragged GQA) fall through to the transient
                # full-width path.
                split_out = self._large_q_split_sdpa_fallback(
                    queries,
                    scale=scale,
                    sliding_window=int(sliding_window),
                    mask=mask,
                )
                if split_out is not None:
                    self.paged_attention_calls += 1
                    self.kv_quant_attention_calls += 1
                    self.attention_time_s += time.perf_counter() - started
                    return split_out
            from mlx_lm.models.base import scaled_dot_product_attention

            gqa_decision = _paged_gqa_sdpa_route_decision_from_env(
                q_len=q_len,
                offset=int(self.offset),
                query_heads=int(queries.shape[1]),
                kv_heads=int(self._shape[0]) if self._shape is not None else 0,
            )
            gqa_route = gqa_decision.route
            keys, values = self._active_attention_arrays(sliding_window)
            if keys is None or values is None:
                return bailout("empty_cache")
            if gqa_route:
                out = _paged_gqa_sdpa(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=float(scale),
                    mask=mask,
                    route=gqa_route,
                )
                if out is not None:
                    self.paged_attention_calls += 1
                    self.kv_quant_attention_calls += 1
                    self.gqa_sdpa_calls += 1
                    self.gqa_sdpa_calls_by_route[gqa_route] = (
                        int(self.gqa_sdpa_calls_by_route.get(gqa_route, 0)) + 1
                    )
                    phase = current_attention_phase()
                    self.gqa_sdpa_calls_by_phase[phase] = (
                        int(self.gqa_sdpa_calls_by_phase.get(phase, 0)) + 1
                    )
                    self.attention_time_s += time.perf_counter() - started
                    return out
            else:
                self._record_gqa_route_miss(
                    gqa_decision,
                    offset=int(self.offset),
                    q_len=q_len,
                    query_heads=int(queries.shape[1]),
                    kv_heads=int(self._shape[0]) if self._shape is not None else 0,
                )
            if q_len > max_q_len and partitioned_enabled and int(self.offset) >= partition_threshold:
                return run_partitioned_paged(force_fp32_paged=False)
            out = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=None,
                scale=scale,
                mask=mask,
            )
            self.paged_attention_calls += 1
            self.kv_quant_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
            return out
        force_fp32_paged = impl in {"fp32_paged", "paged_fp32"}
        kernel_queries = queries.astype(mx.float32) if force_fp32_paged else queries
        kernel_key_cache = (
            self.key_cache.astype(mx.float32) if force_fp32_paged else self.key_cache
        )
        kernel_value_cache = (
            self.value_cache.astype(mx.float32) if force_fp32_paged else self.value_cache
        )
        q_3d = mx.contiguous(kernel_queries[0].transpose(1, 0, 2))
        used_blocks = (self.offset + self.block_size - 1) // self.block_size
        block_tables = self._active_block_table(used_blocks)
        seq_lens = mx.array([self.offset], dtype=mx.int32)
        cu_seqlens_q = mx.array([0, q_len], dtype=mx.int32)
        if self.turboquant:
            if (
                self.key_scale_cache is None
                or self.value_scale_cache is None
                or self.key_zero_cache is None
            ):
                return bailout("turboquant_unsupported")
            tq_ops = _load_vllm_metal_ops_optional(
                context="TurboQuant paged_attention_primitive"
            )
            if tq_ops is None or not hasattr(tq_ops, "paged_attention_primitive"):
                return bailout("turboquant_unsupported")
            cfg = self.turboquant_config
            out = mx.array(0)
            tq_ops.paged_attention_primitive(
                q_3d,
                self.key_cache,
                self.value_cache,
                int(self.key_cache.shape[2]),
                float(scale),
                0.0,
                block_tables,
                seq_lens,
                cu_seqlens_q,
                int(self.block_size),
                int(self.offset),
                int(sliding_window),
                out,
                key_scale_cache=self.key_scale_cache,
                value_scale_cache=self.value_scale_cache,
                key_zero_cache=self.key_zero_cache,
                v_centroids=self._turboquant_v_centroids,
                use_turboquant=True,
                quant_type=str(cfg.key_quant),
                v_bits=int(cfg.value_bits),
            )
            self.paged_attention_calls += 1
            self.turboquant_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
            return out.transpose(1, 0, 2)[None, ...]
        if partitioned_enabled and int(self.offset) >= partition_threshold:
            return run_partitioned_paged(force_fp32_paged=force_fp32_paged)
        out = mx.array(0)
        # The bare vllm_metal default path needs external ops. If they aren't
        # available (no nanobind, no JIT-built paged_ops.so, no
        # MTPLX_VLLM_METAL_REPO), fall through to the in-tree split-sdpa
        # fallback used by the partitioned/turboquant paths so the request
        # does not crash mid-stream. Operators who want the optimal kernel
        # install nanobind + run vllm_metal/metal/build.py.
        ops = _load_vllm_metal_ops_optional(
            context="vllm_metal default paged_attention"
        )
        if ops is None:
            split_out = self._large_q_split_sdpa_fallback(
                queries,
                scale=scale,
                sliding_window=int(sliding_window),
                mask=mask,
            )
            if split_out is not None:
                self.paged_attention_calls += 1
                self.attention_time_s += time.perf_counter() - started
                return split_out
            return bailout("vllm_metal_ops_unavailable")
        ops.paged_attention_primitive(
            q_3d,
            kernel_key_cache,
            kernel_value_cache,
            int(kernel_key_cache.shape[2]),
            float(scale),
            0.0,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            int(self.block_size),
            int(self.offset),
            int(sliding_window),
            out,
        )
        self.paged_attention_calls += 1
        self.attention_time_s += time.perf_counter() - started
        out = out.transpose(1, 0, 2)[None, ...]
        return out.astype(queries.dtype) if force_fp32_paged else out

    def paged_stats(self) -> dict[str, Any]:
        mode = "vllm_metal_paged"
        if self.turboquant:
            mode = "vllm_metal_paged_turboquant"
        elif self.kv_quant:
            mode = f"vllm_metal_paged_kv_{self.kv_quant_config.normalized_mode}"
        return {
            "mode": mode,
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "capacity": int(self.capacity),
            "offset": int(self.offset),
            "updates": int(self.update_calls),
            "paged_attention_calls": int(self.paged_attention_calls),
            "partitioned_attention_calls": int(self.partitioned_attention_calls),
            "turboquant_attention_calls": int(self.turboquant_attention_calls),
            "kv_quant_attention_calls": int(self.kv_quant_attention_calls),
            "gqa_sdpa_calls": int(self.gqa_sdpa_calls),
            "gqa_sdpa_calls_by_route": dict(self.gqa_sdpa_calls_by_route),
            "gqa_sdpa_calls_by_phase": dict(self.gqa_sdpa_calls_by_phase),
            "gqa_sdpa_route_misses_by_phase_reason": dict(
                self.gqa_sdpa_route_misses_by_phase_reason
            ),
            "gqa_sdpa_route_misses_by_q_len": dict(
                self.gqa_sdpa_route_misses_by_q_len
            ),
            "gqa_sdpa_last_route_miss": dict(self.gqa_sdpa_last_route_miss),
            "active_array_calls": int(self.active_array_calls),
            "active_array_time_s": float(self.active_array_time_s),
            "kv_quant_dequant_calls": int(self.kv_quant_dequant_calls),
            "kv_quant_dequant_time_s": float(self.kv_quant_dequant_time_s),
            "kv_quant_dequant_tokens": int(self.kv_quant_dequant_tokens),
            "kv_quant_dequant_memo_hits": int(self.kv_quant_dequant_memo_hits),
            "kv_quant_dequant_memo_rebuilds": int(
                self.kv_quant_dequant_memo_rebuilds
            ),
            "kv_quant_kernel_calls": int(self.kv_quant_kernel_calls),
            "kv_quant_route": str(self._kv_quant_route or ""),
            "kv_quant_route_offset": int(self._kv_quant_route_offset),
            "dense_fallback_calls": int(self.dense_fallback_calls),
            "prefill_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("prefill", 0)
            ),
            "decode_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("decode_verify", 0)
            ),
            "ar_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("ar_decode", 0)
            ),
            "postcommit_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase.get("postcommit", 0)
            ),
            "paged_attention_bailouts_by_phase_reason": dict(
                self.paged_attention_bailouts_by_phase_reason
            ),
            "paged_attention_large_q_path": str(self.paged_attention_large_q_path),
            "large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls
            ),
            "large_q_split_sdpa_fallback_calls_by_phase": dict(
                self.large_q_split_sdpa_fallback_calls_by_phase
            ),
            "prefill_large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls_by_phase.get("prefill", 0)
            ),
            "decode_large_q_split_sdpa_fallback_calls": int(
                self.large_q_split_sdpa_fallback_calls_by_phase.get("decode_verify", 0)
            ),
            "partitioned_paged_calls": int(self.partitioned_paged_calls),
            "partitioned_paged_calls_by_phase": dict(
                self.partitioned_paged_calls_by_phase
            ),
            "prefill_partitioned_paged_calls": int(
                self.partitioned_paged_calls_by_phase.get("prefill", 0)
            ),
            "decode_partitioned_paged_calls": int(
                self.partitioned_paged_calls_by_phase.get("decode_verify", 0)
            ),
            "grow_events": int(self.grow_events),
            "turboquant": int(bool(self.turboquant)),
            "turboquant_k_quant": (
                str(self.turboquant_config.key_quant) if self.turboquant else ""
            ),
            "turboquant_v_quant": (
                str(self.turboquant_config.value_quant) if self.turboquant else ""
            ),
            "kv_quant": int(bool(self.kv_quant)),
            "kv_quant_mode": (
                str(self.kv_quant_config.normalized_mode) if self.kv_quant else ""
            ),
            "sliding_window": int(
                os.environ.get("MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW") or "-1"
            ),
            "bytes": int(self.nbytes),
            "cache_write_time_s": float(self.cache_write_time_s),
            "attention_time_s": float(self.attention_time_s),
        }


class TensorOffsetVllmMetalPagedKVCache:
    """GraphBank-safe paged KV cache with an array-backed offset.

    ``VllmMetalPagedKVCache`` stores the decode offset as a Python integer,
    which is unsafe for ``mx.compile`` replay.  This adapter preserves the
    physical page buffers but makes the offset and rollback window part of the
    compiled array state.
    """

    def __init__(
        self,
        *,
        key_cache: Any,
        value_cache: Any,
        offset: int | Any,
        block_size: int,
        num_blocks: int,
    ) -> None:
        import mlx.core as mx

        self.cache = [
            key_cache,
            value_cache,
            offset if isinstance(offset, mx.array) else mx.array(offset, dtype=mx.int32),
        ]
        self.rollback_state = [None, None, None]
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        # Per-instance static attention ceiling for the dynamic-offset paged
        # kernel.  When set it wins over MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET
        # so a compiled-verify bucket can pin the kernel's static block count
        # without mutating process-global env state.
        self.static_max_offset: int | None = None
        self.update_calls = 0
        self.paged_attention_calls = 0
        self.cache_write_time_s = 0.0
        self.attention_time_s = 0.0

    @classmethod
    def from_paged_cache(cls, entry: VllmMetalPagedKVCache) -> "TensorOffsetVllmMetalPagedKVCache":
        if entry.key_cache is None or entry.value_cache is None:
            raise ValueError("cannot promote empty paged KV cache")
        return cls(
            key_cache=entry.key_cache,
            value_cache=entry.value_cache,
            offset=int(entry.offset),
            block_size=int(entry.block_size),
            num_blocks=int(entry.num_blocks),
        )

    @property
    def key_cache(self):
        return self.cache[0]

    @key_cache.setter
    def key_cache(self, value) -> None:
        self.cache[0] = value

    @property
    def value_cache(self):
        return self.cache[1]

    @value_cache.setter
    def value_cache(self, value) -> None:
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value) -> None:
        import mlx.core as mx

        self.cache[2] = value if isinstance(value, mx.array) else mx.array(value, dtype=mx.int32)

    @property
    def capacity(self) -> int:
        return int(self.block_size) * int(self.num_blocks)

    @property
    def compile_state(self):
        return [self.cache, self.rollback_state]

    def _flat_key_cache(self):
        return self.cache[0].reshape(-1, int(self.cache[0].shape[2]), int(self.cache[0].shape[3]))

    def _flat_value_cache(self):
        return self.cache[1].reshape(-1, int(self.cache[1].shape[2]), int(self.cache[1].shape[3]))

    def update_without_fetch(self, keys: Any, values: Any) -> None:
        import mlx.core as mx

        steps = int(keys.shape[2])
        started = time.perf_counter()
        k_3d = mx.contiguous(keys[0].transpose(1, 0, 2))
        v_3d = mx.contiguous(values[0].transpose(1, 0, 2))
        flat_k = self._flat_key_cache()
        flat_v = self._flat_value_cache()
        self.rollback_state[0] = self.cache[2]
        self.rollback_state[1] = mx.slice(
            flat_k,
            self.cache[2],
            axes=(0,),
            slice_size=k_3d.shape,
        )
        self.rollback_state[2] = mx.slice(
            flat_v,
            self.cache[2],
            axes=(0,),
            slice_size=v_3d.shape,
        )
        flat_k = mx.slice_update(flat_k, k_3d, self.cache[2], axes=(0,))
        flat_v = mx.slice_update(flat_v, v_3d, self.cache[2], axes=(0,))
        self.cache[0] = flat_k.reshape(self.cache[0].shape)
        self.cache[1] = flat_v.reshape(self.cache[1].shape)
        self.cache[2] = self.cache[2] + steps
        self.update_calls += 1
        self.cache_write_time_s += time.perf_counter() - started

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        self.update_without_fetch(keys, values)
        return self.state

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        import mlx.core as mx

        del return_array
        rinds = mx.arange(self.capacity)
        linds = self.cache[2] + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask

    def paged_attention(
        self,
        queries: Any,
        *,
        scale: float,
        sliding_window: int = -1,
        mask: Any | None = None,
        impl_override: str | None = None,
    ):
        del impl_override
        if int(sliding_window) > 0:
            return None
        if int(queries.shape[0]) != 1:
            return None
        static_max_offset = self._static_attention_max_offset()
        started = time.perf_counter()
        from .kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail_dynamic_offset

        out = sdpa_2pass_paged_tail_dynamic_offset(
            queries=queries,
            key_cache=self.cache[0],
            value_cache=self.cache[1],
            offset=self.cache[2],
            block_size=int(self.block_size),
            scale=float(scale),
            mask=mask,
            max_q_len=int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16"),
            max_offset=static_max_offset,
        )
        if out is not None:
            self.paged_attention_calls += 1
            self.attention_time_s += time.perf_counter() - started
        return out

    def _static_attention_max_offset(self) -> int | None:
        if self.static_max_offset is not None:
            return int(self.static_max_offset)
        raw = os.environ.get("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET")
        if raw is None or not raw.strip():
            return None
        return int(raw)

    @property
    def state(self):
        flat_k = self._flat_key_cache()
        flat_v = self._flat_value_cache()
        keys = flat_k.transpose(1, 0, 2)[None, ...]
        values = flat_v.transpose(1, 0, 2)[None, ...]
        return keys, values

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self.key_cache = None
        self.value_cache = None
        self.offset = 0
        if keys is None or values is None:
            return
        paged = VllmMetalPagedKVCache(
            block_size=int(self.block_size),
            num_blocks=int(self.num_blocks),
        )
        paged.update_without_fetch(keys, values)
        self.cache = [
            paged.key_cache,
            paged.value_cache,
            self.offset,
        ]

    def size(self) -> int:
        import mlx.core as mx

        mx.eval(self.cache[2])
        return int(self.cache[2].item())

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        import mlx.core as mx

        n = int(n)
        if (
            self.rollback_state[0] is not None
            and self.rollback_state[1] is not None
            and self.rollback_state[2] is not None
            and int(self.rollback_state[1].shape[0]) == n
        ):
            flat_k = self._flat_key_cache()
            flat_v = self._flat_value_cache()
            flat_k = mx.slice_update(
                flat_k,
                self.rollback_state[1],
                self.rollback_state[0],
                axes=(0,),
            )
            flat_v = mx.slice_update(
                flat_v,
                self.rollback_state[2],
                self.rollback_state[0],
                axes=(0,),
            )
            self.cache[0] = flat_k.reshape(self.cache[0].shape)
            self.cache[1] = flat_v.reshape(self.cache[1].shape)
            self.cache[2] = self.rollback_state[0]
        else:
            self.cache[2] = mx.maximum(
                self.cache[2] - n,
                mx.array(0, dtype=self.cache[2].dtype),
            )
        return n

    def empty(self) -> bool:
        return self.key_cache is None or self.value_cache is None

    @property
    def meta_state(self) -> tuple[str, ...]:
        return (
            str(self.block_size),
            str(self.num_blocks),
            str(int(self.size())),
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        if not value:
            return
        self.block_size = int(value[0])
        self.num_blocks = int(value[1])
        self.offset = int(value[2])

    def to_paged_cache(self) -> "VllmMetalPagedKVCache":
        """Restore a stock ``VllmMetalPagedKVCache`` from this adapter.

        The stock container receives the adapter's current physical page
        buffers (no copy, no densify) and the materialized integer offset.
        Shape/dtype metadata is rebuilt so the next ``update_without_fetch``
        appends in place instead of re-allocating.
        """
        paged = VllmMetalPagedKVCache(
            block_size=int(self.block_size),
            num_blocks=int(self.num_blocks),
        )
        if self.cache[0] is None or self.cache[1] is None:
            return paged
        paged.key_cache = self.cache[0]
        paged.value_cache = self.cache[1]
        paged.offset = int(self.size())
        paged._shape = (
            int(self.cache[0].shape[2]),
            int(self.cache[0].shape[3]),
            int(self.cache[1].shape[3]),
        )
        paged._dtypes = (self.cache[0].dtype, self.cache[1].dtype)
        return paged

    def demote(self) -> "VllmMetalPagedKVCache":
        """Alias for :meth:`to_paged_cache` (bank-facing demotion API)."""
        return self.to_paged_cache()

    @property
    def nbytes(self) -> int:
        if self.key_cache is None or self.value_cache is None:
            return 0
        return int(self.key_cache.nbytes) + int(self.value_cache.nbytes) + int(self.cache[2].nbytes)

    def paged_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": "tensor_offset_vllm_metal_paged",
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "capacity": int(self.capacity),
            "offset": int(self.size()),
            "static_max_offset": int(self._static_attention_max_offset() or self.capacity),
            "updates": int(self.update_calls),
            "paged_attention_calls": int(self.paged_attention_calls),
            "bytes": int(self.nbytes),
            "cache_write_time_s": float(self.cache_write_time_s),
            "attention_time_s": float(self.attention_time_s),
        }


class OwnedRecurrentStateCache:
    """Fixed-shape recurrent cache with persistent owned state buffers.

    Qwen3Next GDN layers keep only two recurrent leaves: the causal-conv tail
    and the GDN matrix state.  Stock ``ArraysCache`` replaces those leaves with
    whatever expression produced the newest state.  This diagnostic instead
    keeps stable buffers and copies each accepted state into them, forcing the
    official cache entry to be owned data at the commit boundary.
    """

    def __init__(
        self,
        size: int = 2,
        *,
        mode: str = "persistent_eval",
        initial: list[Any] | tuple[Any, ...] | None = None,
        left_padding: Any | None = None,
        lengths: Any | None = None,
    ) -> None:
        self.cache = [None] * int(size)
        self.mode = str(mode).strip().lower().replace("-", "_") or "persistent_eval"
        if self.mode not in {"persistent_eval"}:
            raise ValueError("owned recurrent state mode must be 'persistent_eval'")
        self._owned_buffers = [None] * int(size)
        self.left_padding = left_padding
        self.lengths = lengths
        self.owner_updates = 0
        self.owner_arrays = 0
        self.owner_allocations = 0
        self.owner_inplace_updates = 0
        self.owner_bytes = 0
        self.owner_time_s = 0.0
        if initial is not None:
            self.replace_state(initial)

    @classmethod
    def from_cache(
        cls,
        entry: Any,
        *,
        mode: str = "persistent_eval",
    ) -> "OwnedRecurrentStateCache":
        return cls(
            size=len(getattr(entry, "cache", getattr(entry, "state", [None, None]))),
            mode=mode,
            initial=list(getattr(entry, "state", [None, None])),
            left_padding=getattr(entry, "left_padding", None),
            lengths=getattr(entry, "lengths", None),
        )

    def __getitem__(self, idx: int) -> Any:
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        # Model forward writes speculative state through ``cache[i] = ...``.
        # Keep those cheap; commit/restore paths call ``replace_state`` to force
        # the owned-copy boundary only for authoritative state.
        self.cache[idx] = value

    @property
    def state(self) -> list[Any]:
        return self.cache

    @state.setter
    def state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        self.replace_state(value)

    def replace_state(self, value: list[Any] | tuple[Any, ...] | None) -> None:
        if value is None:
            self.cache = [None] * len(self.cache)
            self._owned_buffers = [None] * len(self._owned_buffers)
            return
        for idx, item in enumerate(value):
            if idx >= len(self.cache):
                break
            self.cache[idx] = self._own_value(idx, item)
        for idx in range(len(value), len(self.cache)):
            self.cache[idx] = None

    def restore_masked(
        self,
        snapshot_state: list[Any] | tuple[Any, ...] | None,
        row_mask: Any,
    ) -> None:
        """Per-row masked restore of the recurrent leaves (fold-in REPLAY rewind).

        Rows where ``row_mask`` is ``True`` revert each leaf (batch-major
        ``[conv_tail, gdn_matrix]``, axis 0 == batch) to ``snapshot_state``; rows
        where it is ``False`` keep their current advanced state.  This is the
        per-row analogue of ``rollback_after_verify``'s whole-batch restore,
        following the same snapshot-in / restore-out convention (the snapshot
        comes from ``snapshot_untrimmable_cache``), and it covers a REPLAY row's
        rewind: the conv tail is sliding-window / positional, so its missed-cycle
        pollution is undone here (a test pins this bitwise).

        REBINDS ``self.cache[idx]`` with a lazy ``mx.where`` expression -- it does
        NOT route through ``replace_state``/``_own_value`` (those force an
        ``mx.eval`` into the owned buffer), so the restore stays a device op that
        adds no sync to the single-sync fold-in loop.  Signature is additive;
        nothing existing changes.
        """
        import mlx.core as mx

        if snapshot_state is None:
            return
        mask = row_mask if isinstance(row_mask, mx.array) else mx.array(row_mask)
        mask = mask.astype(mx.bool_).reshape(-1)
        for idx in range(len(self.cache)):
            cur = self.cache[idx]
            snap = snapshot_state[idx] if idx < len(snapshot_state) else None
            if cur is None or snap is None:
                continue
            # broadcast the [B] row selector across each leaf's trailing dims.
            m = mask.reshape((int(mask.size),) + (1,) * (int(cur.ndim) - 1))
            self.cache[idx] = mx.where(m, snap, cur)

    def zero_rows(self, row_mask: Any) -> None:
        """Per-row masked ZERO of the recurrent leaves (refill admission).

        Rows where ``row_mask`` is ``True`` have every leaf reset to zeros —
        the recurrent fresh-start value (the causal-conv tail and the GDN
        matrix state both zero-initialize), so an admission prefill over those
        rows reproduces a from-scratch prefill.  Same lazy rebind contract as
        :meth:`restore_masked`: a device-side ``mx.where``, no sync, additive.
        """
        import mlx.core as mx

        mask = row_mask if isinstance(row_mask, mx.array) else mx.array(row_mask)
        mask = mask.astype(mx.bool_).reshape(-1)
        for idx in range(len(self.cache)):
            cur = self.cache[idx]
            if cur is None:
                continue
            m = mask.reshape((int(mask.size),) + (1,) * (int(cur.ndim) - 1))
            self.cache[idx] = mx.where(m, mx.zeros_like(cur), cur)

    @property
    def meta_state(self) -> tuple[str, str]:
        return ("owned_recurrent_state", self.mode)

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        if not value:
            return
        if isinstance(value, (list, tuple)) and len(value) > 1:
            mode = str(value[1]).strip().lower().replace("-", "_")
            if mode in {"persistent_eval"}:
                self.mode = mode

    @property
    def batch_size(self) -> int:
        for item in self.cache:
            if item is not None:
                return int(item.shape[0])
        if self.left_padding is not None:
            return int(self.left_padding.size)
        if self.lengths is not None:
            return int(self.lengths.size)
        return 1

    def _own_value(self, idx: int, value: Any) -> Any:
        import mlx.core as mx

        if value is None or not isinstance(value, mx.array):
            return value
        started = time.perf_counter()
        existing = self._owned_buffers[idx]
        if existing is value:
            mx.eval(existing)
            self.owner_updates += 1
            self.owner_arrays += 1
            self.owner_bytes += int(existing.nbytes)
            self.owner_time_s += time.perf_counter() - started
            return existing
        reusable = (
            isinstance(existing, mx.array)
            and tuple(existing.shape) == tuple(value.shape)
            and existing.dtype == value.dtype
        )
        if reusable:
            target = existing
            self.owner_inplace_updates += 1
        else:
            target = mx.zeros(value.shape, dtype=value.dtype)
            self.owner_allocations += 1
        full_slice = tuple(slice(None) for _ in range(len(value.shape)))
        target[full_slice] = value
        mx.eval(target)
        self._owned_buffers[idx] = target
        self.owner_updates += 1
        self.owner_arrays += 1
        self.owner_bytes += int(target.nbytes)
        self.owner_time_s += time.perf_counter() - started
        return target

    def filter(self, batch_indices: Any) -> None:
        self.replace_state(
            [item[batch_indices] if item is not None else None for item in self.cache]
        )
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other: Any) -> None:
        import mlx.core as mx

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a: Any, b: Any) -> Any:
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype
            if shape is None:
                return None
            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)
            return mx.concatenate([a, b])

        self.replace_state([cat(c, o) for c, o in zip(self.cache, other.cache)])
        self.left_padding = cat(self.left_padding, getattr(other, "left_padding", None))
        self.lengths = cat(self.lengths, getattr(other, "lengths", None))

    def extract(self, idx: int) -> "OwnedRecurrentStateCache":
        return OwnedRecurrentStateCache(
            len(self.cache),
            mode=self.mode,
            initial=[item[idx : idx + 1] if item is not None else None for item in self.cache],
            left_padding=(
                self.left_padding[idx : idx + 1]
                if self.left_padding is not None
                else None
            ),
            lengths=self.lengths[idx : idx + 1] if self.lengths is not None else None,
        )

    def prepare(self, lengths=None, **kwargs) -> None:
        import mlx.core as mx

        if lengths is not None:
            self.lengths = mx.array(lengths)

    def finalize(self) -> None:
        self.lengths = None
        self.left_padding = None

    def advance(self, N: int) -> None:
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        import mlx.core as mx

        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        return None

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return all(item is None for item in self.cache)

    @property
    def nbytes(self) -> int:
        return sum(int(item.nbytes) for item in self.cache if item is not None)

    def owner_stats(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "updates": int(self.owner_updates),
            "arrays": int(self.owner_arrays),
            "allocations": int(self.owner_allocations),
            "inplace_updates": int(self.owner_inplace_updates),
            "bytes": int(self.owner_bytes),
            "time_s": float(self.owner_time_s),
        }


def replace_recurrent_cache_state(entry: Any, state: list[Any] | tuple[Any, ...]) -> None:
    if hasattr(entry, "replace_state"):
        entry.replace_state(state)
        return
    if hasattr(entry, "__setitem__") and len(state) >= 2:
        entry[0] = state[0]
        entry[1] = state[1]
        return
    current = getattr(entry, "state", None)
    if isinstance(current, list) and len(current) == len(state):
        current[:] = list(state)
    else:
        entry.state = list(state)


def install_owned_recurrent_state_cache(
    cache: list[Any],
    *,
    mode: str = "persistent_eval",
) -> dict[str, int | str]:
    """Replace stock fixed recurrent caches with persistent owner caches."""
    normalized = str(mode).strip().lower().replace("-", "_") or "persistent_eval"
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
    }
    for idx, entry in enumerate(cache or []):
        if entry is None or _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, OwnedRecurrentStateCache):
            entry.mode = normalized
            stats["entries"] = int(stats["entries"]) + 1
            continue
        state = getattr(entry, "state", None)
        if not isinstance(state, list) or len(state) != 2:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if hasattr(entry, "keys") or hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = OwnedRecurrentStateCache.from_cache(entry, mode=normalized)
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def configure_owned_recurrent_state_cache(cache: list[Any]) -> dict[str, int | str]:
    raw = os.environ.get("MTPLX_OWNED_RECURRENT_STATE") or ""
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {"1", "true", "yes", "on", "persistent", "persistent_eval"}:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    mode = os.environ.get("MTPLX_OWNED_RECURRENT_STATE_MODE") or "persistent_eval"
    return install_owned_recurrent_state_cache(cache, mode=mode)


def owned_recurrent_state_stats(cache: list[Any] | None) -> dict[str, int | float | str]:
    aggregate: dict[str, int | float | str] = {
        "enabled": 0,
        "entries": 0,
        "updates": 0,
        "arrays": 0,
        "allocations": 0,
        "inplace_updates": 0,
        "bytes": 0,
        "time_s": 0.0,
        "mode": "disabled",
    }
    for entry in cache or []:
        if not isinstance(entry, OwnedRecurrentStateCache):
            continue
        stats = entry.owner_stats()
        aggregate["enabled"] = 1
        aggregate["entries"] = int(aggregate["entries"]) + 1
        aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
        aggregate["arrays"] = int(aggregate["arrays"]) + int(stats["arrays"])
        aggregate["allocations"] = int(aggregate["allocations"]) + int(stats["allocations"])
        aggregate["inplace_updates"] = int(aggregate["inplace_updates"]) + int(
            stats["inplace_updates"]
        )
        aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
        aggregate["time_s"] = float(aggregate["time_s"]) + float(stats["time_s"])
        aggregate["mode"] = str(stats["mode"])
    return aggregate


def install_tail_owned_attention_kv_cache(
    cache: list[Any],
    *,
    mode: str = "contiguous_eval",
    step: int | None = None,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with tail-owner caches."""
    normalized = _normalize_detach_mode(mode)
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
    }
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            entry.mode = normalized
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = TailOwnedKVCache.from_cache(
            entry,
            mode=normalized,
            step=step,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def install_block_owned_attention_kv_cache(
    cache: list[Any],
    *,
    mode: str = "contiguous_eval",
    block_size: int = 1024,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with block-owner caches."""
    normalized = _normalize_detach_mode(mode)
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": normalized,
        "entries": 0,
        "skipped": 0,
        "block_size": int(block_size),
    }
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, BlockOwnedKVCache):
            entry.mode = normalized
            entry.block_size = int(block_size)
            entry.step = int(block_size)
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = BlockOwnedKVCache.from_cache(
            entry,
            mode=normalized,
            block_size=block_size,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def install_vllm_metal_paged_attention_kv_cache(
    cache: list[Any],
    *,
    block_size: int = 16,
    num_blocks: int = 1024,
    turboquant_config: Any | None = None,
    kv_quant_config: Any | None = None,
) -> dict[str, int | str]:
    """Replace stock full-attention KV caches with vLLM-Metal paged caches."""
    fallback_kv_quant_config = kv_quant_config
    if turboquant_config is not None:
        kv_quant_config = None
    mode = "vllm_metal_paged"
    if turboquant_config is not None:
        mode = "vllm_metal_paged_turboquant"
    elif kv_quant_config is not None:
        mode = f"vllm_metal_paged_kv_{kv_quant_config.normalized_mode}"
    stats: dict[str, int | str] = {
        "enabled": 1,
        "mode": mode,
        "entries": 0,
        "skipped": 0,
        "block_size": int(block_size),
        "num_blocks": int(num_blocks),
        "turboquant": int(bool(turboquant_config)),
        "kv_quant": int(bool(kv_quant_config)),
        "attention_impl": _paged_attention_impl_from_env() or "vllm_metal",
    }
    external_ops_required = _paged_attention_requires_external_ops(
        turboquant_config=turboquant_config,
        kv_quant_config=kv_quant_config,
    )
    stats["external_ops_required"] = int(external_ops_required)
    if turboquant_config is not None:
        stats["turboquant_k_quant"] = str(turboquant_config.key_quant)
        stats["turboquant_v_quant"] = str(turboquant_config.value_quant)
    if kv_quant_config is not None:
        stats["kv_quant_mode"] = str(kv_quant_config.normalized_mode)
    # Validate the optional dependency once at install time only for paths that
    # actually dispatch into the external vLLM-Metal ops. The packaged
    # mlx_vector_paged and sdpa_2pass_paged paths are in-tree and must survive a
    # clean product checkout without REFERENCES:TOOLS.
    #
    # When the install is for a TurboQuant snapshot but external ops can't be
    # loaded (no nanobind, no JIT-built paged_ops.so, no MTPLX_VLLM_METAL_REPO),
    # we MUST NOT raise: that would crash any in-flight request that triggered
    # the install. Instead, downgrade gracefully to the plain paged-cache
    # layout. The cache stays functional via the in-tree mlx_vector_paged /
    # sdpa_2pass_paged kernels, just without the TurboQuant compression.
    if external_ops_required:
        try:
            _load_vllm_metal_ops()
        except RuntimeError as exc:
            if turboquant_config is not None:
                _warn_vllm_metal_ops_unavailable(
                    exc, context="TurboQuant install"
                )
                turboquant_config = None
                kv_quant_config = fallback_kv_quant_config
                stats["mode"] = (
                    f"vllm_metal_paged_kv_{kv_quant_config.normalized_mode}"
                    if kv_quant_config is not None
                    else "vllm_metal_paged"
                )
                stats["turboquant"] = 0
                stats["kv_quant"] = int(bool(kv_quant_config))
                stats["external_ops_required"] = int(
                    _paged_attention_requires_external_ops(
                        turboquant_config=None,
                        kv_quant_config=kv_quant_config,
                    )
                )
                stats["turboquant_disabled_reason"] = "vllm_metal_ops_unavailable"
                stats.pop("turboquant_k_quant", None)
                stats.pop("turboquant_v_quant", None)
                if kv_quant_config is not None:
                    stats["kv_quant_mode"] = str(kv_quant_config.normalized_mode)
            else:
                raise
    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, VllmMetalPagedKVCache):
            if entry.key_cache is None:
                entry.block_size = int(block_size)
                entry.num_blocks = int(num_blocks)
            else:
                # Live pages own the geometry; a re-config is a request for
                # room satisfied by an explicit grow — never a claim of blocks
                # that do not exist (#310). block_size stays put too: changing
                # it would reinterpret the live buffer.
                entry.num_blocks = int(entry.key_cache.shape[0])
                wanted = int(block_size) * int(num_blocks)
                if wanted > entry.capacity:
                    entry._grow_to_capacity(wanted)
            entry.turboquant_config = turboquant_config
            entry.turboquant = turboquant_config is not None
            entry.kv_quant_config = kv_quant_config
            entry.kv_quant = kv_quant_config is not None
            stats["entries"] = int(stats["entries"]) + 1
            continue
        if isinstance(entry, TailOwnedKVCache):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not _is_trimmable(entry):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if getattr(entry, "_idx", None) is not None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not hasattr(entry, "keys") or not hasattr(entry, "values"):
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        cache[idx] = VllmMetalPagedKVCache.from_cache(
            entry,
            block_size=block_size,
            num_blocks=num_blocks,
            turboquant_config=turboquant_config,
            kv_quant_config=kv_quant_config,
        )
        stats["entries"] = int(stats["entries"]) + 1
    return stats


def configure_tail_owned_attention_kv_cache(cache: list[Any]) -> dict[str, int | str]:
    paged_raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN") or ""
    if paged_raw.strip().lower() in {"1", "true", "yes", "on"}:
        from .kv_quant import config_from_env as kv_quant_config_from_env
        from .turboquant import config_from_env as turboquant_config_from_env

        block_size = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE") or "16")
        configured_blocks = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS") or "1024")
        num_blocks = _dynamic_paged_num_blocks(
            block_size=block_size,
            configured_blocks=configured_blocks,
        )
        turboquant_config = turboquant_config_from_env()
        return install_vllm_metal_paged_attention_kv_cache(
            cache,
            block_size=block_size,
            num_blocks=num_blocks,
            turboquant_config=turboquant_config,
            kv_quant_config=kv_quant_config_from_env(),
        )
    raw = os.environ.get("MTPLX_OWNED_ATTN_KV") or ""
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {
        "1",
        "true",
        "yes",
        "on",
        "tail",
        "tail_owned",
        "block",
        "block_owned",
    }:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    mode = os.environ.get("MTPLX_OWNED_ATTN_KV_MODE") or "contiguous_eval"
    if normalized in {"block", "block_owned"}:
        block_raw = (
            os.environ.get("MTPLX_OWNED_ATTN_KV_BLOCK_SIZE")
            or os.environ.get("MTPLX_OWNED_ATTN_KV_STEP")
            or "1024"
        )
        return install_block_owned_attention_kv_cache(
            cache,
            mode=mode,
            block_size=int(block_raw),
        )
    step_raw = os.environ.get("MTPLX_OWNED_ATTN_KV_STEP")
    step = int(step_raw) if step_raw else None
    return install_tail_owned_attention_kv_cache(cache, mode=mode, step=step)


def configure_mtp_attention_kv_cache(cache: list[Any]) -> dict[str, int | str]:
    """Optionally put the native MTP layer on the vLLM-Metal paged KV path.

    Trunk paged attention is controlled by ``MTPLX_VLLM_METAL_PAGED_ATTN``.
    The MTP layer is kept behind a separate flag because it changes the draft
    proposal path and therefore needs its own speed and parity evidence.
    """

    raw = os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_ATTN") or ""
    if raw.strip().lower() not in {"1", "true", "yes", "on"}:
        return {"enabled": 0, "entries": 0, "skipped": 0, "mode": "disabled"}
    block_size = int(
        os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE")
        or os.environ.get("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE")
        or "16"
    )
    num_blocks = int(
        os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS")
        or os.environ.get("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS")
        or "1024"
    )
    num_blocks = _dynamic_paged_num_blocks(
        block_size=block_size,
        configured_blocks=num_blocks,
    )
    stats = install_vllm_metal_paged_attention_kv_cache(
        cache,
        block_size=block_size,
        num_blocks=num_blocks,
        turboquant_config=None,
    )
    stats["mode"] = "vllm_metal_paged_mtp"
    return stats


def tail_owned_attention_kv_stats(cache: list[Any] | None) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "enabled": 0,
        "entries": 0,
        "updates": 0,
        "arrays": 0,
        "bytes": 0,
        "time_s": 0.0,
        "mode": "disabled",
    }
    for entry in cache or []:
        if isinstance(entry, VllmMetalPagedKVCache):
            stats = entry.paged_stats()
            aggregate["enabled"] = 1
            aggregate["entries"] = int(aggregate["entries"]) + 1
            aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
            aggregate["arrays"] = int(aggregate["arrays"]) + int(
                stats["paged_attention_calls"]
            )
            aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
            aggregate["time_s"] = float(aggregate["time_s"]) + float(
                stats["cache_write_time_s"]
            ) + float(stats["attention_time_s"])
            aggregate["mode"] = str(stats["mode"])
            aggregate["block_size"] = int(stats["block_size"])
            aggregate["num_blocks"] = int(stats["num_blocks"])
            aggregate["capacity"] = int(stats["capacity"])
            aggregate["partitioned_attention_calls"] = int(
                aggregate.get("partitioned_attention_calls", 0)
            ) + int(stats.get("partitioned_attention_calls", 0))
            aggregate["turboquant_attention_calls"] = int(
                aggregate.get("turboquant_attention_calls", 0)
            ) + int(stats.get("turboquant_attention_calls", 0))
            aggregate["kv_quant_attention_calls"] = int(
                aggregate.get("kv_quant_attention_calls", 0)
            ) + int(stats.get("kv_quant_attention_calls", 0))
            aggregate["gqa_sdpa_calls"] = int(
                aggregate.get("gqa_sdpa_calls", 0)
            ) + int(stats.get("gqa_sdpa_calls", 0))
            aggregate["active_array_calls"] = int(
                aggregate.get("active_array_calls", 0)
            ) + int(stats.get("active_array_calls", 0))
            aggregate["active_array_time_s"] = float(
                aggregate.get("active_array_time_s", 0.0)
            ) + float(stats.get("active_array_time_s", 0.0))
            aggregate["kv_quant_dequant_calls"] = int(
                aggregate.get("kv_quant_dequant_calls", 0)
            ) + int(stats.get("kv_quant_dequant_calls", 0))
            aggregate["kv_quant_dequant_time_s"] = float(
                aggregate.get("kv_quant_dequant_time_s", 0.0)
            ) + float(stats.get("kv_quant_dequant_time_s", 0.0))
            aggregate["kv_quant_dequant_tokens"] = int(
                aggregate.get("kv_quant_dequant_tokens", 0)
            ) + int(stats.get("kv_quant_dequant_tokens", 0))
            aggregate["kv_quant_dequant_memo_hits"] = int(
                aggregate.get("kv_quant_dequant_memo_hits", 0)
            ) + int(stats.get("kv_quant_dequant_memo_hits", 0))
            aggregate["kv_quant_dequant_memo_rebuilds"] = int(
                aggregate.get("kv_quant_dequant_memo_rebuilds", 0)
            ) + int(stats.get("kv_quant_dequant_memo_rebuilds", 0))
            aggregate["kv_quant_kernel_calls"] = int(
                aggregate.get("kv_quant_kernel_calls", 0)
            ) + int(stats.get("kv_quant_kernel_calls", 0))
            aggregate["dense_fallback_calls"] = int(
                aggregate.get("dense_fallback_calls", 0)
            ) + int(stats.get("dense_fallback_calls", 0))
            for key in (
                "prefill_dense_fallback_calls",
                "decode_dense_fallback_calls",
                "ar_dense_fallback_calls",
                "postcommit_dense_fallback_calls",
                "large_q_split_sdpa_fallback_calls",
                "prefill_large_q_split_sdpa_fallback_calls",
                "decode_large_q_split_sdpa_fallback_calls",
                "partitioned_paged_calls",
                "prefill_partitioned_paged_calls",
                "decode_partitioned_paged_calls",
            ):
                aggregate[key] = int(aggregate.get(key, 0)) + int(stats.get(key, 0))
            for dict_key in (
                "large_q_split_sdpa_fallback_calls_by_phase",
                "partitioned_paged_calls_by_phase",
                "gqa_sdpa_calls_by_phase",
                "gqa_sdpa_calls_by_route",
                "gqa_sdpa_route_misses_by_phase_reason",
                "gqa_sdpa_route_misses_by_q_len",
            ):
                phase_counts = stats.get(dict_key) or {}
                if isinstance(phase_counts, dict):
                    merged = dict(aggregate.get(dict_key) or {})
                    for phase, count in phase_counts.items():
                        merged[str(phase)] = int(merged.get(str(phase), 0)) + int(count)
                    aggregate[dict_key] = merged
            last_gqa_miss = stats.get("gqa_sdpa_last_route_miss") or {}
            if isinstance(last_gqa_miss, dict) and last_gqa_miss:
                aggregate["gqa_sdpa_last_route_miss"] = dict(last_gqa_miss)
            bailouts = stats.get("paged_attention_bailouts_by_phase_reason") or {}
            if isinstance(bailouts, dict):
                merged = dict(aggregate.get("paged_attention_bailouts_by_phase_reason") or {})
                for reason_key, count in bailouts.items():
                    merged[str(reason_key)] = int(merged.get(str(reason_key), 0)) + int(count)
                aggregate["paged_attention_bailouts_by_phase_reason"] = merged
            large_q_path = str(stats.get("paged_attention_large_q_path") or "")
            if large_q_path:
                aggregate["paged_attention_large_q_path"] = large_q_path
            aggregate["grow_events"] = int(
                aggregate.get("grow_events", 0)
            ) + int(stats.get("grow_events", 0))
            aggregate["turboquant"] = int(
                aggregate.get("turboquant", 0)
            ) or int(stats.get("turboquant", 0))
            aggregate["kv_quant"] = int(
                aggregate.get("kv_quant", 0)
            ) or int(stats.get("kv_quant", 0))
            if stats.get("turboquant_k_quant"):
                aggregate["turboquant_k_quant"] = str(stats["turboquant_k_quant"])
            if stats.get("turboquant_v_quant"):
                aggregate["turboquant_v_quant"] = str(stats["turboquant_v_quant"])
            if stats.get("kv_quant_mode"):
                aggregate["kv_quant_mode"] = str(stats["kv_quant_mode"])
            continue
        if not isinstance(entry, TailOwnedKVCache):
            continue
        stats = entry.tail_owner_stats()
        aggregate["enabled"] = 1
        aggregate["entries"] = int(aggregate["entries"]) + 1
        aggregate["updates"] = int(aggregate["updates"]) + int(stats["updates"])
        aggregate["arrays"] = int(aggregate["arrays"]) + int(stats["arrays"])
        aggregate["bytes"] = int(aggregate["bytes"]) + int(stats["bytes"])
        aggregate["time_s"] = float(aggregate["time_s"]) + float(stats["time_s"])
        aggregate["mode"] = str(stats["mode"])
    return aggregate


def _clone_tree(value: Any) -> Any:
    import mlx.core as mx

    if value is None:
        return None
    if isinstance(value, mx.array):
        # `mx.array(existing_array)` can preserve storage identity for mutable
        # cache buffers. Force a new array expression so later KV writes cannot
        # mutate the saved snapshot behind our back.
        return value + mx.zeros((), dtype=value.dtype)
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    return value


def snapshot_cache(cache: list[Any]) -> CacheSnapshot:
    return CacheSnapshot(
        states=tuple(_clone_tree(getattr(c, "state", None)) for c in cache),
        meta_states=tuple(_clone_tree(getattr(c, "meta_state", None)) for c in cache),
    )


def _lazy_state_view(value: Any) -> Any:
    """Zero-copy retention of a cache leaf.

    A fresh slice expression references the array's current *value*, so later
    container writes — whether rebind-style (`self.cache[0] = mx.slice_update`)
    or setitem-style (`self.keys[..., a:b, :] = tail`) — can never mutate it:
    MLX only donates a buffer when it holds the sole reference
    (tests/test_lazy_snapshot_cow.py pins this). No GPU work happens here.
    """
    import mlx.core as mx

    if isinstance(value, mx.array):
        return value[...]
    if isinstance(value, tuple):
        return tuple(_lazy_state_view(v) for v in value)
    if isinstance(value, list):
        return [_lazy_state_view(v) for v in value]
    if isinstance(value, dict):
        return {k: _lazy_state_view(v) for k, v in value.items()}
    return value


def snapshot_cache_lazy_hybrid(cache: list[Any]) -> CacheSnapshot:
    """Snapshot with zero-copy views for trimmable KV, clones for the rest.

    Trimmable attention KV carries the GB-scale bytes; retaining lazy views
    makes commit O(1) and defers the single divergence copy to MLX's COW at
    the next container write. Recurrent/owned containers mutate their buffers
    in place (`OwnedRecurrentStateCache._own_value` setitem path), so their
    small states are still eagerly cloned.
    """
    states = []
    meta_states = []
    for entry in cache:
        state = getattr(entry, "state", None)
        if _is_trimmable(entry):
            states.append(_lazy_state_view(state))
        else:
            states.append(_clone_tree(state))
        meta_states.append(_clone_tree(getattr(entry, "meta_state", None)))
    return CacheSnapshot(states=tuple(states), meta_states=tuple(meta_states))


def snapshot_untrimmable_cache(cache: list[Any]) -> CacheSnapshot:
    """Snapshot only recurrent/non-trimmable cache state.

    Attention KV caches can roll back by trimming their offset. GDN recurrent
    caches cannot, so those are the states we copy before speculative verify.
    """
    states = []
    meta_states = []
    for entry in cache:
        if _is_trimmable(entry):
            states.append(None)
            meta_states.append(None)
        else:
            states.append(_clone_tree(getattr(entry, "state", None)))
            meta_states.append(_clone_tree(getattr(entry, "meta_state", None)))
    return CacheSnapshot(states=tuple(states), meta_states=tuple(meta_states))


def snapshot_untrimmable_cache_lazy(cache: list[Any]) -> CacheSnapshot:
    """Zero-copy-view variant of :func:`snapshot_untrimmable_cache`.

    Identical entry selection (trimmable KV -> ``None``; recurrent/non-trimmable
    -> captured), but each recurrent leaf is retained as a lazy view
    (:func:`_lazy_state_view`, ``value[...]``, zero kernel) instead of a
    materialized clone (:func:`_clone_tree`, ``value + mx.zeros`` -- a full
    device copy of the whole batch's GDN matrix state every cycle).

    COW-safety basis (why the view can never be mutated behind our back on the
    fold-in loop):

    * The GDN forward REBINDS the recurrent cache slots
      (``cache[idx] = new_state``; ``gdn_capture.py`` ->
      ``OwnedRecurrentStateCache.__setitem__``) rather than writing in place, so
      advancing the state leaves the retained view pointing at the pre-forward
      array's value.
    * The per-row REPLAY rewind (:func:`restore_untrimmable_cache_masked` ->
      ``OwnedRecurrentStateCache.restore_masked``) also REBINDS via a fresh
      ``mx.where`` expression, never a setitem into the snapshot's buffer.

    Only the authoritative commit path (``replace_state`` / ``_own_value``'s
    in-place ``target[:] = value``) writes a recurrent buffer in place, and that
    path is not on the fold-in decode forward.  Meta-states are tiny string
    tuples and are still cloned.  :func:`snapshot_untrimmable_cache` (eager) is
    left byte-for-byte unchanged for every other caller (serial/pipelined
    scalar-repair lanes, ``generation.py``).
    """
    states = []
    meta_states = []
    for entry in cache:
        if _is_trimmable(entry):
            states.append(None)
            meta_states.append(None)
        else:
            states.append(_lazy_state_view(getattr(entry, "state", None)))
            meta_states.append(_clone_tree(getattr(entry, "meta_state", None)))
    return CacheSnapshot(states=tuple(states), meta_states=tuple(meta_states))


def restore_cache(
    cache: list[Any],
    snapshot: CacheSnapshot,
    *,
    restore_meta_state: bool = True,
    clone_states: bool = True,
) -> None:
    cache_count = len(cache)
    state_count = len(snapshot.states)
    meta_count = len(snapshot.meta_states)
    if cache_count != state_count or cache_count != meta_count:
        raise ValueError(
            "cache snapshot length mismatch: "
            f"cache={cache_count}, states={state_count}, meta_states={meta_count}"
        )

    atomic_pairs = tuple(
        (
            getattr(entry, "prepare_snapshot_restore", None),
            getattr(entry, "install_snapshot_restore", None),
        )
        for entry in cache
    )
    if (
        restore_meta_state
        and atomic_pairs
        and all(callable(prepare) and callable(install) for prepare, install in atomic_pairs)
    ):
        pairs = tuple(zip(snapshot.states, snapshot.meta_states, strict=True))
        if all(state is None and meta_state is None for state, meta_state in pairs):
            return
        if any(state is None or meta_state is None for state, meta_state in pairs):
            raise ValueError("atomic cache snapshot requires complete state/meta pairs")
        prepared = []
        for entry, (state, meta_state), (prepare, install) in zip(
            cache,
            pairs,
            atomic_pairs,
            strict=True,
        ):
            install_view = not clone_states and _is_trimmable(entry)
            prepared_state = (
                _lazy_state_view(state) if install_view else _clone_tree(state)
            )
            prepared.append(
                (install, prepare(prepared_state, _clone_tree(meta_state)))
            )
        for install, prepared_pair in prepared:
            install(prepared_pair)
        return

    for entry, state, meta_state in zip(cache, snapshot.states, snapshot.meta_states):
        atomic_restore = getattr(entry, "restore_snapshot_state", None)
        if (
            state is not None
            and restore_meta_state
            and meta_state is not None
            and callable(atomic_restore)
        ):
            install_view = not clone_states and _is_trimmable(entry)
            prepared_state = (
                _lazy_state_view(state) if install_view else _clone_tree(state)
            )
            atomic_restore(prepared_state, _clone_tree(meta_state))
            continue
        if state is not None:
            install_view = not clone_states and _is_trimmable(entry)
            _restore_state_preserving_container(entry, state, clone=not install_view)
        if restore_meta_state and meta_state is not None:
            entry.meta_state = _clone_tree(meta_state)


def _restore_state_preserving_container(entry: Any, state: Any, *, clone: bool = True) -> None:
    # clone=False (lazy snapshots into trimmable KV) must still never install
    # the snapshot's own array objects. A setitem-style container write
    # (`self.keys[..., a:b, :] = tail`) mutates the installed *object* itself,
    # and a rebind-style write may donate its buffer (the lone shared object
    # counts as uniquely referenced) — either way the borrower's suffix
    # prefill/decode lands inside the bank entry's stored span and every later
    # restore serves the poisoned pages (issue #247). Installing a fresh
    # zero-copy view keeps restore O(1) while restoring the two-object
    # geometry commit relies on: the retained snapshot reference blocks buffer
    # donation, so the borrower's first write pays the single deferred
    # divergence copy (COW rules pinned by tests/test_lazy_snapshot_cow.py).
    # Containers with replace_state (owned recurrent) copy into owned buffers
    # and must always receive a full clone they are free to consume.
    cloned = _clone_tree(state) if clone else _lazy_state_view(state)
    if hasattr(entry, "replace_state"):
        entry.replace_state(cloned)
        return
    current = getattr(entry, "state", None)
    if isinstance(current, list) and isinstance(cloned, list) and len(current) == len(cloned):
        current[:] = cloned
        return
    entry.state = cloned


def rollback_after_verify(cache: list[Any], snapshot: CacheSnapshot, verified_tokens: int) -> None:
    """Undo a speculative target verify pass."""
    for entry in cache:
        if _is_trimmable(entry) and hasattr(entry, "trim"):
            entry.trim(verified_tokens)
    restore_cache(cache, snapshot)


def restore_untrimmable_cache_masked(
    cache: list[Any],
    snapshot: CacheSnapshot,
    row_mask: Any,
) -> None:
    """Per-row masked restore of every non-trimmable (recurrent) entry.

    The fold-in decode loop's REPLAY rewind: rows selected by ``row_mask`` revert
    their recurrent state to ``snapshot`` (the pre-verify snapshot of the cycle
    they missed, from :func:`snapshot_untrimmable_cache`); every other row keeps
    advancing.  This is the per-row companion to :func:`rollback_after_verify`'s
    whole-batch restore.

    Trimmable KV carries a ``None`` snapshot state here (see
    :func:`snapshot_untrimmable_cache`) and is skipped -- the ragged fold-in KV
    lane rolls a missed row back by OVERWRITING its stale draft slot on the replay
    write, not by snapshot restore, so only the recurrent leaves need this.

    Entries exposing ``restore_masked`` (``OwnedRecurrentStateCache``) take the
    lazy device rebind path (no sync).  A plain list/array-state recurrent entry
    (e.g. the CPU test fake) falls back to a generic per-row selection so the same
    loop drives both.
    """
    for entry, state in zip(cache, snapshot.states):
        if state is None:
            continue
        restore_masked = getattr(entry, "restore_masked", None)
        if callable(restore_masked):
            restore_masked(state, row_mask)
        else:
            entry.state = _select_rows_masked(getattr(entry, "state", None), state, row_mask)


def _select_rows_masked(current: Any, snapshot: Any, row_mask: Any) -> Any:
    """Per-row masked blend of ``current`` and ``snapshot`` recurrent state.

    Fallback for entries WITHOUT ``restore_masked`` (array-state recurrent caches
    take the class method instead).  Handles the two shapes such an entry's
    ``state`` can take:

    * a single batch-major ``mx.array`` (``[B, ...]``) -> ``mx.where`` on axis 0;
    * a per-row Python container (``list[row]`` -- the CPU test fake's histories)
      -> pick whole rows by the host mask, COPYING reverted rows so a later
      in-place append can't mutate the retained snapshot.

    A ``list``-of-arrays *leaves* container (``[conv_tail, gdn_matrix]``) is NOT
    handled here on purpose -- that is ``OwnedRecurrentStateCache.restore_masked``'s
    job, and the fold-in make-cache converts every real recurrent entry to that
    class, so only per-row list state ever reaches this fallback.
    """
    import mlx.core as mx

    if current is None or snapshot is None:
        return current if current is not None else snapshot
    if isinstance(current, mx.array) and isinstance(snapshot, mx.array):
        mask = row_mask if isinstance(row_mask, mx.array) else mx.array(row_mask)
        mask = mask.astype(mx.bool_).reshape((-1,) + (1,) * (int(current.ndim) - 1))
        return mx.where(mask, snapshot, current)
    if isinstance(current, (list, tuple)) and isinstance(snapshot, (list, tuple)):
        flags = row_mask.tolist() if isinstance(row_mask, mx.array) else list(row_mask)
        out = []
        for r in range(len(current)):
            revert = bool(flags[r]) if r < len(flags) else False
            src = snapshot[r] if revert else current[r]
            out.append(list(src) if isinstance(src, list) else src)
        return type(current)(out)
    return current


def trim_verified_window_to_prefix(
    cache: list[Any],
    snapshot: CacheSnapshot,
    *,
    verified_tokens: int,
    keep_tokens: int,
) -> bool:
    """Keep a verified target prefix by trimming only uncommitted KV tail tokens.

    This is deliberately stricter than ``rollback_after_verify``. It is only
    valid when the pre-verify snapshot contains no recurrent/non-trimmable
    state, because those caches cannot be advanced partially from an ordinary
    batched verify pass.
    """
    verified_tokens = int(verified_tokens)
    keep_tokens = int(keep_tokens)
    if keep_tokens < 0 or verified_tokens < keep_tokens:
        return False
    trim_tokens = verified_tokens - keep_tokens
    if any(state is not None for state in snapshot.states):
        return False
    if any(meta_state is not None for meta_state in snapshot.meta_states):
        return False
    if trim_tokens <= 0:
        return True
    if not cache:
        return False

    before_offsets: list[int] = []
    for entry in cache:
        trim = getattr(entry, "trim", None)
        offset = _entry_offset(entry)
        if not _is_trimmable(entry) or not callable(trim) or offset is None:
            return False
        if offset < trim_tokens:
            return False
        before_offsets.append(offset)

    for entry, before_offset in zip(cache, before_offsets):
        trimmed = entry.trim(trim_tokens)
        if trimmed is not None and int(trimmed) != trim_tokens:
            return False
        after_offset = _entry_offset(entry)
        if after_offset is None or before_offset - after_offset != trim_tokens:
            return False
    return True


def trim_verified_window_without_snapshot(
    cache: list[Any],
    *,
    verified_tokens: int,
    keep_tokens: int,
) -> bool:
    """Snapshot-free ``trim_verified_window_to_prefix``.

    The snapshot's only role in the trim path is proving that no
    recurrent/non-trimmable state needs restoring; when every cache entry is
    trimmable that property holds by construction, so a skipped verify
    snapshot (MTPLX_SKIP_VERIFY_SNAPSHOT=1) must not strand the repair.
    Returns False for any cache carrying non-trimmable entries — those
    genuinely need the snapshot.
    """

    if not cache:
        return False
    if any(not _is_trimmable(entry) for entry in cache):
        return False
    empty = CacheSnapshot(
        states=tuple(None for _ in cache),
        meta_states=tuple(None for _ in cache),
    )
    return trim_verified_window_to_prefix(
        cache,
        empty,
        verified_tokens=verified_tokens,
        keep_tokens=keep_tokens,
    )


def _entry_offset(entry: Any) -> int | None:
    offset = getattr(entry, "offset", None)
    if offset is None:
        return None
    try:
        if hasattr(offset, "item"):
            return int(offset.item())
        return int(offset)
    except Exception:
        return None


def detach_array_leaf(value: Any, *, mode: str) -> Any:
    """Return an evaluated cache leaf according to the configured detach mode."""
    import mlx.core as mx

    if not isinstance(value, mx.array):
        return value
    normalized = _normalize_detach_mode(mode)
    if normalized == "eval_only":
        mx.eval(value)
        return value
    if normalized == "metal_copy_leaf":
        from .kernels.copy_leaf import metal_copy_leaf

        return metal_copy_leaf(value)
    leaf = mx.contiguous(value)
    mx.eval(leaf)
    return leaf


def detach_recurrent_cache_state(
    cache: list[Any],
    *,
    components: set[str],
    mode: str,
) -> dict[str, int]:
    """Detach official recurrent cache state in-place.

    This is intentionally limited to non-trimmable recurrent entries. Attention
    KV caches remain on their normal trim/update path until attribution says the
    tail needs its own owner-copy implementation.
    """
    import mlx.core as mx

    requested = {item.strip().lower().replace("-", "_") for item in components if item}
    supported = {"conv", "gdn"}
    requested &= supported
    stats = {"entries": 0, "arrays": 0, "bytes": 0}
    if not requested:
        return stats

    for entry in cache:
        if _is_trimmable(entry):
            continue
        state = getattr(entry, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) < 2:
            continue
        mutable = list(state)
        changed = False
        for index, component in ((0, "conv"), (1, "gdn")):
            if component not in requested:
                continue
            value = mutable[index]
            if not isinstance(value, mx.array):
                continue
            detached = detach_array_leaf(value, mode=mode)
            mutable[index] = detached
            changed = True
            stats["arrays"] += 1
            stats["bytes"] += int(detached.nbytes)
        if not changed:
            continue
        stats["entries"] += 1
        if hasattr(entry, "replace_state"):
            entry.replace_state(mutable)
        elif isinstance(state, list):
            state[:] = mutable
        else:
            entry.state = tuple(mutable)
    return stats


def detach_attention_cache_state(
    cache: list[Any],
    *,
    mode: str,
) -> dict[str, int]:
    """Evaluate or owner-copy attention KV cache arrays in-place."""
    import mlx.core as mx

    stats = {"entries": 0, "arrays": 0, "bytes": 0}
    normalized = _normalize_detach_mode(mode)
    for entry in cache:
        if not _is_trimmable(entry):
            continue
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
            continue
        if normalized == "eval_only":
            mx.eval(keys, values)
            detached_keys, detached_values = keys, values
        else:
            detached_keys = detach_array_leaf(keys, mode=normalized)
            detached_values = detach_array_leaf(values, mode=normalized)
            entry.keys = detached_keys
            entry.values = detached_values
        stats["entries"] += 1
        stats["arrays"] += 2
        stats["bytes"] += int(detached_keys.nbytes) + int(detached_values.nbytes)
    return stats


def detach_cache_state(
    cache: list[Any],
    *,
    components: set[str],
    mode: str,
) -> dict[str, int]:
    """Detach requested cache groups and combine accounting stats."""
    requested = {item.strip().lower().replace("-", "_") for item in components if item}
    stats = detach_recurrent_cache_state(
        cache,
        components=requested & {"gdn", "conv"},
        mode=mode,
    )
    if "attn" in requested or "attn_tail" in requested:
        attn_stats = detach_attention_cache_state(cache, mode=mode)
        for key, value in attn_stats.items():
            stats[key] = int(stats.get(key, 0)) + int(value)
    return stats


def _is_trimmable(entry: Any) -> bool:
    try:
        return bool(entry.is_trimmable())
    except Exception:
        return False
