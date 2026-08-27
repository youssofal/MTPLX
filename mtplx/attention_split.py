"""Split-SDPA hooks for long-context full-attention diagnostics."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# F23b (2026-08-16): packed-GQA route declines. Counted only when the lane
# is enabled AND the call is a verify-shaped dense-cache window (q_len 2..4,
# cache present, blockwise/paged lanes not owning attention) yet the route
# still fell back to fused SDPA. By-design non-applicability (q_len 1
# decode, prefill, paged-cache calls) is never counted. Import-stable
# surface for /health; increments happen only on the declined path —
# engaged calls skip the block via the same gate bool the router uses.
# Kernel-level contract bails have their own precise counters in
# mtplx.kernels.sdpa_gqa_packed.gqa_packed_bail_counts. Inside a compiled
# verify graph this python body runs at trace time only, so traced-path
# declines count once per trace, not once per replay.
gqa_packed_route_bail_counts: dict[str, int] = {}


def _count_gqa_packed_route_bail(reason: str) -> None:
    gqa_packed_route_bail_counts[reason] = (
        gqa_packed_route_bail_counts.get(reason, 0) + 1
    )


def _env_index_set(name: str) -> set[int]:
    raw = os.environ.get(name, "")
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _cache_offset_value(cache: Any) -> int | mx.array:
    return getattr(cache, "offset", 0) if cache is not None else 0


def _cache_offset_static_int(cache: Any) -> int | None:
    offset = _cache_offset_value(cache)
    if isinstance(offset, mx.array):
        return None
    return int(offset or 0)


def _cache_offset_int(cache: Any) -> int:
    offset = _cache_offset_value(cache)
    if isinstance(offset, mx.array):
        if offset.size != 1:
            return int(mx.max(offset).item())
        return int(offset.item())
    return int(offset or 0)


def split_sdpa_mask(
    mask: Any | None,
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Any | None:
    """Slice an SDPA mask for a query chunk without changing causal semantics."""

    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]


def split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Any | None,
    cache: Any | None,
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    """Run full-precision SDPA in query chunks.

    Each query row is independent mathematically, but MLX's fused SDPA kernels
    can use shape-dependent reduction paths. Treat this as a diagnostic or a
    candidate that still needs the normal acceptance-decision parity gates.
    """

    from mlx_lm.models.base import scaled_dot_product_attention

    q_len = int(queries.shape[2])
    chunk = max(1, int(chunk_size))
    if q_len <= chunk:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk):
        end = min(start + chunk, q_len)
        key_end = min(int(keys.shape[2]), int(cached_prefix_len) + end)
        chunk_mask = split_sdpa_mask(
            mask,
            query_start=start,
            query_end=end,
            key_end=key_end,
        )
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    if q_proj is None or q_norm is None:
        return False
    weight = getattr(q_proj, "weight", None)
    norm_weight = getattr(q_norm, "weight", None)
    if weight is None or norm_weight is None:
        return False
    num_heads = int(getattr(attn, "num_attention_heads", getattr(attn, "n_heads", 0)))
    expected = 2 * num_heads * int(norm_weight.shape[0])
    return int(weight.shape[0]) == expected


def _install_split_attention_hook(attn: Any) -> bool:
    cls = type(attn)
    if getattr(cls, "_mtplx_split_full_attention_installed", False):
        return False

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if not getattr(self, "_mtplx_split_full_attention_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        from mlx_lm.models.base import scaled_dot_product_attention

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0,
            2,
            1,
            3,
        )

        cached_prefix_offset = _cache_offset_value(cache)
        cached_prefix_len = _cache_offset_static_int(cache)
        blockwise_threshold = int(
            getattr(self, "_mtplx_blockwise_full_attention_threshold", 1024)
        )
        can_slice_mask = (
            mask is None
            or (isinstance(mask, str) and mask == "causal")
            or isinstance(mask, mx.array)
        )
        blockwise_enabled = bool(
            cache is not None
            and getattr(self, "_mtplx_blockwise_full_attention_enabled", False)
            and cached_prefix_len is not None
            and cached_prefix_len >= blockwise_threshold
            and hasattr(cache, "update_without_fetch")
            and hasattr(cache, "active_block_slices")
            and can_slice_mask
        )
        vllm_metal_paged_enabled = bool(
            cache is not None
            and getattr(self, "_mtplx_vllm_metal_paged_enabled", False)
            and hasattr(cache, "update_without_fetch")
            and hasattr(cache, "paged_attention")
            and int(B) == 1
            and can_slice_mask
        )
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_offset)
            keys = self.rope(keys, offset=cached_prefix_offset)
            if blockwise_enabled or vllm_metal_paged_enabled:
                cache.update_without_fetch(keys, values)
            else:
                keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Ragged-batch 2pass verify attention (dense batched-MTP lane): the
        # fused vector path is correct-but-slow at this shape (~180 GB/s on
        # per-row 24k KV streams, measured), and the array-offset ragged lane
        # fails every other custom path closed. This branch streams K/V
        # flash-style with PER-ROW causal bounds taken from the ragged cache's
        # pre-write offsets, no mask tensor is ever materialized. Opt-in via
        # the instance attr (set by the batched driver); env kill-switch
        # MTPLX_RAGGED_2PASS=0.
        if (
            getattr(self, "_mtplx_ragged_2pass_enabled", False)
            and cache is not None
            and not blockwise_enabled
            and not vllm_metal_paged_enabled
            and cached_prefix_len is None  # array offsets = the ragged lane
            and hasattr(cached_prefix_offset, "ndim")
            and os.environ.get("MTPLX_RAGGED_2PASS", "1").strip().lower()
            not in {"0", "false", "off", "no"}
        ):
            from .kernels.sdpa_2pass import sdpa_2pass_tail_rowoffs_kvshared

            ragged_out = sdpa_2pass_tail_rowoffs_kvshared(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                offsets=cached_prefix_offset,
            )
            if ragged_out is not None:
                output = ragged_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output * mx.sigmoid(gate))

        chunk_size = int(getattr(self, "_mtplx_split_full_attention_chunk_size", 1))
        threshold = int(getattr(self, "_mtplx_split_full_attention_threshold", 1024))
        sdpa_2pass_enabled = bool(getattr(self, "_mtplx_sdpa_2pass_enabled", False))
        sdpa_2pass_threshold = int(getattr(self, "_mtplx_sdpa_2pass_threshold", 1024))
        sdpa_2pass_max_q = int(getattr(self, "_mtplx_sdpa_2pass_max_q", 16))
        should_use_2pass = (
            sdpa_2pass_enabled
            and cache is not None
            and cached_prefix_len is not None
            and cached_prefix_len >= sdpa_2pass_threshold
            and 0 < int(queries.shape[2]) <= sdpa_2pass_max_q
            and can_slice_mask
        )
        # Packed-row GQA verify kernel (speed-war Lane A, 2026-07-05): the
        # decode-verify q=2..4 window over a long dense KV. Uses the cache's
        # full capacity buffers + offset, so it works identically on the
        # eager stock KVCache (python offset) and inside the compiled verify
        # graph (TensorOffsetKVCache array offset). Threshold checks the
        # STATIC buffer capacity because the offset may be a traced array.
        gqa_packed_enabled = bool(
            getattr(self, "_mtplx_gqa_packed_sdpa_enabled", False)
        )
        gqa_packed_threshold = int(
            getattr(self, "_mtplx_gqa_packed_sdpa_threshold", 8192)
        )
        if gqa_packed_enabled:
            from .kernel_selfcheck import lane_disabled

            gqa_packed_enabled = not lane_disabled("gqa_packed_sdpa")
        should_use_gqa_packed = (
            gqa_packed_enabled
            and cache is not None
            and not blockwise_enabled
            and not vllm_metal_paged_enabled
            and 2 <= int(queries.shape[2]) <= 4
            and can_slice_mask
            and getattr(cache, "keys", None) is not None
            and getattr(cache, "values", None) is not None
            and int(cache.keys.shape[2]) >= gqa_packed_threshold
        )
        if should_use_gqa_packed and isinstance(mask, mx.array):
            # Only the capacity-wide tail-causal bool mask our cache
            # adapters emit is equivalent to the kernel's built-in
            # semantics; anything else falls back to stock.
            should_use_gqa_packed = (
                mask.dtype == mx.bool_
                and int(mask.shape[-2]) == int(queries.shape[2])
                and int(mask.shape[-1]) == int(cache.keys.shape[2])
            )
        if (
            gqa_packed_enabled
            and not should_use_gqa_packed
            and cache is not None
            and not blockwise_enabled
            and not vllm_metal_paged_enabled
            and 2 <= int(queries.shape[2]) <= 4
        ):
            # F23b: enabled verify-shaped dense-cache window that the packed
            # route declined — record why (bail path only).
            if (
                getattr(cache, "keys", None) is None
                or getattr(cache, "values", None) is None
            ):
                _count_gqa_packed_route_bail("kv_buffers_none")
            elif int(cache.keys.shape[2]) < gqa_packed_threshold:
                _count_gqa_packed_route_bail("capacity_below_threshold")
            elif not can_slice_mask:
                _count_gqa_packed_route_bail("mask_type_unsupported")
            else:
                _count_gqa_packed_route_bail("mask_shape_mismatch")
        should_use_vllm_metal_paged = (
            vllm_metal_paged_enabled
            and cache is not None
            and hasattr(cache, "paged_attention")
            and can_slice_mask
        )
        should_split = (
            cache is not None
            and getattr(self, "_mtplx_split_full_attention_explicit_enabled", False)
            and cached_prefix_len is not None
            and cached_prefix_len >= threshold
            and int(queries.shape[2]) > max(1, chunk_size)
            and can_slice_mask
        )
        if should_use_vllm_metal_paged:
            impl_override = (
                "fast_sdpa_gather"
                if getattr(self, "_mtplx_vllm_metal_exact_gather_layer", False)
                else None
            )
            output = cache.paged_attention(
                queries,
                scale=self.scale,
                mask=mask,
                impl_override=impl_override,
            )
            if output is None:
                if hasattr(cache, "record_dense_fallback"):
                    cache.record_dense_fallback()
                elif hasattr(cache, "dense_fallback_calls"):
                    cache.dense_fallback_calls += 1
                if (
                    hasattr(cache, "long_context_dense_fallback_forbidden")
                    and cache.long_context_dense_fallback_forbidden()
                ):
                    raise RuntimeError(
                        "Sustained long-context paged attention attempted dense "
                        "cache.state fallback after the partition threshold"
                    )
                keys, values = cache.state
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )
        elif should_use_gqa_packed:
            from .kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail

            output = sdpa_gqa_packed_tail(
                queries=queries,
                keys=cache.keys,
                values=cache.values,
                offset=cache.offset,
                scale=self.scale,
            )
            if output is not None:
                self._mtplx_gqa_packed_sdpa_calls = (
                    int(getattr(self, "_mtplx_gqa_packed_sdpa_calls", 0)) + 1
                )
                if _env_enabled("MTPLX_GQA_PACKED_SDPA_TRACE") and (
                    self._mtplx_gqa_packed_sdpa_calls <= 2
                ):
                    import sys as _sys

                    print(
                        "mtplx_gqa_packed_route engaged "
                        f"layer={getattr(self, '_mtplx_full_attention_index', -1)} "
                        f"q_len={int(queries.shape[2])} "
                        f"capacity={int(cache.keys.shape[2])}",
                        file=_sys.stderr,
                        flush=True,
                    )
            else:
                if _env_enabled("MTPLX_GQA_PACKED_SDPA_TRACE"):
                    import sys as _sys

                    print(
                        "mtplx_gqa_packed_route bailed_to_fused "
                        f"layer={getattr(self, '_mtplx_full_attention_index', -1)} "
                        f"q_len={int(queries.shape[2])}",
                        file=_sys.stderr,
                        flush=True,
                    )
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )
        elif should_use_2pass:
            from .kernels.sdpa_2pass import sdpa_2pass_tail

            output = sdpa_2pass_tail(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
                max_q_len=sdpa_2pass_max_q,
            )
            if output is None:
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )
        elif blockwise_enabled and can_slice_mask:
            from .block_attention import blockwise_attention

            output = blockwise_attention(
                queries=queries,
                cache=cache,
                scale=self.scale,
                cached_prefix_len=cached_prefix_len,
            )
        elif should_split:
            self._mtplx_split_full_attention_calls = int(
                getattr(self, "_mtplx_split_full_attention_calls", 0)
            ) + 1
            output = split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=chunk_size,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    cls.__call__ = split_call
    cls._mtplx_split_full_attention_installed = True
    return True


def _full_attention_layers(model: Any):
    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    for layer in getattr(inner, "layers", []):
        if getattr(layer, "is_linear", False):
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            yield attn


def configure_split_full_attention(
    model: Any,
    *,
    enabled: bool | None = None,
    chunk_size: int | None = None,
    threshold: int | None = None,
) -> dict[str, int | bool]:
    """Configure query-chunked SDPA for Qwen3Next full-attention layers."""

    active = _env_enabled("MTPLX_SPLIT_FULL_ATTN", default=False) if enabled is None else bool(enabled)
    blockwise = _env_enabled("MTPLX_BLOCKWISE_ATTN", default=False)
    sdpa_2pass = _env_enabled("MTPLX_SDPA_2PASS", default=False)
    vllm_metal_paged = _env_enabled("MTPLX_VLLM_METAL_PAGED_ATTN", default=False)
    gqa_packed = _env_enabled("MTPLX_GQA_PACKED_SDPA", default=False)
    blockwise_threshold = int(os.environ.get("MTPLX_BLOCKWISE_ATTN_THRESHOLD", "1024"))
    sdpa_2pass_threshold = int(os.environ.get("MTPLX_SDPA_2PASS_THRESHOLD", "1024"))
    sdpa_2pass_max_q = int(os.environ.get("MTPLX_SDPA_2PASS_MAX_Q", "16"))
    gqa_packed_threshold = int(
        os.environ.get("MTPLX_GQA_PACKED_SDPA_THRESHOLD", "8192")
    )
    exact_gather_last_n = int(
        os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N", "0")
        or "0"
    )
    exact_gather_indices = _env_index_set(
        "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES"
    )
    chunk_was_explicit = chunk_size is not None or "MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE" in os.environ
    chunk = int(chunk_size if chunk_size is not None else os.environ.get("MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE", "1"))
    chunk_defaulted = False
    if active and chunk <= 1:
        chunk = 2048
        chunk_defaulted = True
    min_prefix = int(threshold if threshold is not None else os.environ.get("MTPLX_SPLIT_FULL_ATTN_THRESHOLD", "1024"))
    stats = {
        "enabled": bool(active or sdpa_2pass or vllm_metal_paged or gqa_packed),
        "split_full_attn_enabled": bool(active),
        "split_full_attn_chunk_size": int(chunk),
        "split_full_attn_chunk_size_was_explicit": bool(chunk_was_explicit),
        "split_full_attn_chunk_size_defaulted": bool(chunk_defaulted),
        "split_full_attn_calls": 0,
        "blockwise_enabled": bool(blockwise),
        "blockwise_threshold": int(blockwise_threshold),
        "sdpa_2pass_enabled": bool(sdpa_2pass),
        "sdpa_2pass_threshold": int(sdpa_2pass_threshold),
        "sdpa_2pass_max_q": int(sdpa_2pass_max_q),
        "gqa_packed_sdpa_enabled": bool(gqa_packed),
        "gqa_packed_sdpa_threshold": int(gqa_packed_threshold),
        "vllm_metal_paged_enabled": bool(vllm_metal_paged),
        "vllm_metal_exact_gather_last_n": int(exact_gather_last_n),
        "vllm_metal_exact_gather_indices": sorted(exact_gather_indices),
        "layers": 0,
        "installed": 0,
        "exact_gather_layers": 0,
        "chunk_size": int(chunk),
        "threshold": int(min_prefix),
    }
    full_layers = list(_full_attention_layers(model))
    full_layer_count = len(full_layers)
    for full_idx, attn in enumerate(full_layers):
        exact_gather_layer = bool(
            vllm_metal_paged
            and (
                full_idx in exact_gather_indices
                or (
                    exact_gather_last_n > 0
                    and full_idx >= max(0, full_layer_count - exact_gather_last_n)
                )
            )
        )
        stats["installed"] += int(_install_split_attention_hook(attn))
        attn._mtplx_split_full_attention_enabled = bool(
            active or sdpa_2pass or vllm_metal_paged or gqa_packed
        )
        attn._mtplx_split_full_attention_explicit_enabled = bool(active)
        attn._mtplx_blockwise_full_attention_enabled = bool(blockwise)
        attn._mtplx_blockwise_full_attention_threshold = int(blockwise_threshold)
        attn._mtplx_sdpa_2pass_enabled = bool(sdpa_2pass)
        attn._mtplx_sdpa_2pass_threshold = int(sdpa_2pass_threshold)
        attn._mtplx_sdpa_2pass_max_q = int(sdpa_2pass_max_q)
        attn._mtplx_gqa_packed_sdpa_enabled = bool(gqa_packed)
        attn._mtplx_gqa_packed_sdpa_threshold = int(gqa_packed_threshold)
        attn._mtplx_gqa_packed_sdpa_calls = 0
        attn._mtplx_vllm_metal_paged_enabled = bool(vllm_metal_paged)
        attn._mtplx_vllm_metal_exact_gather_layer = exact_gather_layer
        attn._mtplx_full_attention_index = int(full_idx)
        attn._mtplx_full_attention_count = int(full_layer_count)
        attn._mtplx_split_full_attention_chunk_size = int(chunk)
        attn._mtplx_split_full_attention_threshold = int(min_prefix)
        attn._mtplx_split_full_attention_calls = 0
        stats["layers"] += 1
        stats["exact_gather_layers"] += int(exact_gather_layer)
    return stats


def configure_ragged_2pass_attention(model: Any, *, enabled: bool = True) -> int:
    """Enable the ragged-batch 2pass verify attention on every full-attention
    trunk layer (dense batched-MTP lane). Installs the split hook if needed and
    sets both the hook-enable and the ragged-2pass instance flags. Returns the
    number of attention modules configured."""
    count = 0
    for attn in _full_attention_layers(model):
        _install_split_attention_hook(attn)
        attn._mtplx_split_full_attention_enabled = True
        attn._mtplx_ragged_2pass_enabled = bool(enabled)
        count += 1
    return count
