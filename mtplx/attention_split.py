"""Split-SDPA hooks for long-context full-attention diagnostics."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from .attention_context import (
    current_attention_phase,
    format_kv_attention_record,
    note_kv_attention_record,
)
from .attention_math import attention_gate
from .compile_state import in_compiled_step_body, is_compile_trace_error
from .rope_origin import (
    cache_owns_rotary_origin,
    note_unowned_rotary_origin,
    rope_offset_of,
)


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# KV attention diagnostic (issue #526: all-NaN logits under q4/q8 KV with no
# way to tell from the log which attention route produced them).
#
#   MTPLX_KV_ATTENTION_TRACE=1          one line per full-attention call here
#   MTPLX_KV_ATTENTION_TRACE=nonfinite  every call's output is checked; only
#                                       non-finite ones print
#   unset                               no device read and no sync
#
# In every mode each trunk layer's latest call is recorded, host facts only,
# in the current request's record (attention_context), and a request that
# fails with non-finite logits reports the first layer's record. Checking an
# output costs a sync per layer, which is why it is opt-in. A compiled step
# runs this Python body at trace time only: its record is kept as a trace
# and its printed line says offset=traced and finite=traced.


def _kv_attention_trace_mode() -> str:
    raw = (os.environ.get("MTPLX_KV_ATTENTION_TRACE") or "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return ""
    if raw in ("nonfinite", "non_finite", "non-finite"):
        return "nonfinite"
    return "all"


def _kv_cache_bits(cache: Any) -> int | str:
    if cache is None:
        return "-"
    if getattr(cache, "turboquant", False):
        return "turboquant"
    bits = getattr(cache, "kv_bits", None)  # promoted quantized adapter
    if bits is None and getattr(cache, "kv_quant", False):
        bits = cache.kv_quant_config.bits  # eager quantized pages
    return 16 if bits is None else int(bits)


def _kv_cache_capacity(cache: Any) -> int | None:
    if cache is None:
        return None
    capacity = getattr(cache, "capacity", None)  # paged caches and adapters
    if isinstance(capacity, int):
        return capacity
    # Plain instance attributes only: on some containers ``keys`` is a
    # property that dequantizes or concatenates the whole cache.
    fields = getattr(cache, "__dict__", {})
    keys = fields.get("keys")
    if keys is None and isinstance(fields.get("cache"), list) and fields["cache"]:
        keys = fields["cache"][0]  # tensor-offset adapters: [keys, values, offset]
    shape = getattr(keys, "shape", None)
    return int(shape[2]) if shape is not None and len(shape) == 4 else None


def _kv_mask_kind(mask: Any) -> str:
    if mask is None:
        return "none"
    if isinstance(mask, str):
        return mask
    return "array_" + str(getattr(mask, "dtype", "?")).removeprefix("mlx.core.")


def _host_value(value: Any) -> Any:
    """``value.item()`` eagerly; None while ``mx.compile`` traces it."""
    try:
        return value.item()
    except ValueError as exc:
        if is_compile_trace_error(exc):
            return None
        raise


def _note_kv_attention(
    attn: Any,
    cache: Any,
    queries: Any,
    output: Any,
    mask: Any,
    route: str,
    fallback: str,
) -> None:
    layer = int(getattr(attn, "_mtplx_full_attention_index", -1))
    if layer < 0:
        # Not a trunk full-attention layer (a draft head sharing the class):
        # its call never feeds the target logits a failure reports.
        return
    offset = getattr(cache, "offset", None) if cache is not None else None
    shown_offset: Any = (
        offset if isinstance(offset, int) else ("array" if isinstance(offset, mx.array) else "-")
    )
    traced = in_compiled_step_body()
    finite = "unchecked"
    mode = _kv_attention_trace_mode()
    if mode:
        verdict = _host_value(mx.all(mx.isfinite(output)))
        if verdict is None:
            traced = True
        finite = "traced" if verdict is None else str(int(bool(verdict)))
        if isinstance(offset, mx.array):
            value = _host_value(offset)
            shown_offset = "traced" if value is None else int(value)
    record = (
        layer,
        current_attention_phase(),
        "none" if cache is None else type(cache).__name__,
        _kv_cache_bits(cache),
        route,
        queries.dtype,
        shown_offset,
        _kv_cache_capacity(cache),
        int(queries.shape[2]),
        _kv_mask_kind(mask),
        fallback,
        finite,
    )
    note_kv_attention_record(layer, record, traced=traced)
    if mode == "all" or (mode == "nonfinite" and finite == "0"):
        import sys as _sys

        print(format_kv_attention_record(record), file=_sys.stderr, flush=True)


def _paged_route_label(cache: Any) -> str:
    """Route label for a call the cache's own ``paged_attention`` served."""
    if getattr(cache, "kv_quant", False):
        # Eager quantized pages latch one numerics route per request.
        return "paged_kv_quant_" + str(getattr(cache, "_kv_quant_route", None) or "unlatched")
    return "paged_kernel"


def _paged_decline_label(cache: Any, mask: Any) -> str:
    """Why ``paged_attention`` returned None (host data only)."""
    if isinstance(mask, mx.array) and hasattr(cache, "rollback_state"):
        # Tensor-offset paged adapters build capacity-wide array masks and
        # their kernels take only causal/None masks.
        return "array_mask"
    bailout = getattr(cache, "paged_attention_last_bailout", None)
    if isinstance(bailout, dict) and bailout.get("reason"):
        return "last_bailout_" + str(bailout["reason"])
    return "paged_declined"


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


#: The positive receipt next to the bail counts above (issue #506): how often
#: the packed route ACCEPTED a verify window, keyed ``q<rows>_cap<bucket>``.
#: Read both the same way. They are cumulative since boot, and they tick in
#: Python, so inside the compiled verifier they count graph TRACES, not calls:
#: one trace per (window, capacity bucket), and the compiled graph then runs
#: the kernel on every call without coming back here. Warm-up traces small
#: capacities, so ``capacity_below_threshold`` on a fresh daemon is warm-up,
#: not the first request. Above the compiled-verify context fence the verifier
#: runs eagerly and both counters count calls.
gqa_packed_route_engaged_counts: dict[str, int] = {}


def _count_gqa_packed_route_engaged(q_len: int, capacity: int) -> None:
    # An eager cache grows 256 rows at a time; a power-of-two bucket keeps the
    # key set small over a long session.
    bucket = 1 << max(0, int(capacity).bit_length() - 1)
    key = f"q{int(q_len)}_cap{bucket}"
    gqa_packed_route_engaged_counts[key] = (
        gqa_packed_route_engaged_counts.get(key, 0) + 1
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


def _cache_rope_offset(cache: Any) -> int | mx.array:
    """Where the rope rotates the next row: the cache's own rotary origin.

    A stock cache has none, so this is its host offset. A tensor-offset
    cache of the compiled routes owns one (``rope_origin.RotaryOrigin``): the
    plain offset array for a text request and ``offset + rope_delta`` for an
    image request, so the traced graph never reads the request context.
    """

    return rope_offset_of(cache) if cache is not None else 0


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
        route_off = not getattr(self, "_mtplx_split_full_attention_enabled", False)
        if route_off and not cache_owns_rotary_origin(cache):
            # No MTPLX attention route asked for, and the cache has no rotary
            # origin of its own: the stock forward, untouched. A cache that
            # owns one (an image request on a compiled route) rotates at it
            # below; the body under it is the stock forward's, row for row.
            # The canary first: a tensor-offset cache that SHOULD have owned
            # one is counted here, before the stock forward ropes it plainly.
            if current_attention_phase() != "decode_verify" or not (
                _attention_has_gated_q_proj(self)
            ):
                note_unowned_rotary_origin(cache)
                return original_call(self, x, mask=mask, cache=cache)
            # A gated verify forward takes the body as well (its rope reads
            # the offset through rope_offset_of, which runs the same canary).
            # With no route asked for, every lane below stays off (blockwise
            # included, which has a switch of its own) and the body is the
            # stock forward row for row, except that its gate goes through
            # attention_gate: mlx-lm's ``output * mx.sigmoid(gate)`` runs a
            # standalone sigmoid eagerly and a fused one inside the compiled
            # verifier's trace, which differ in the last bit of some bfloat16
            # and float32 values (mtplx/attention_math.py).
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
            and not (route_off and not cache_owns_rotary_origin(cache))
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
            rope_offset = _cache_rope_offset(cache)
            queries = self.rope(queries, offset=rope_offset)
            keys = self.rope(keys, offset=rope_offset)
            if blockwise_enabled or vllm_metal_paged_enabled:
                cache.update_without_fetch(keys, values)
            else:
                keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

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
        # MTPLX_GQA_PACKED_WIDE (2026-08-25 flat-decode): route q_len 5-16
        # to the query-group kernel instead of the second-bank path the QL
        # sweep measured as the depth cliff. Off = shipping behavior.
        gqa_packed_wide = _env_enabled("MTPLX_GQA_PACKED_WIDE")
        nax_flash_wide = _env_enabled("MTPLX_NAX_FLASH_WIDE") and _env_enabled(
            "MTPLX_NAX_FLASH_ROUTE"
        )
        should_use_gqa_packed = (
            gqa_packed_enabled
            and cache is not None
            and not blockwise_enabled
            and not vllm_metal_paged_enabled
            # 8 rows since 2026-07-21 (second float4 bank): depth 4's
            # verify window is q_len 5; QL <= 4 compiles identically.
            and 2 <= int(queries.shape[2]) <= max(
                16 if gqa_packed_wide else 8,
                # MTPLX_NAX_FLASH_WIDE: the flash-decoding kernel also takes the
                # 9..32-row verify windows (context-copy blocks); the scalar
                # packed kernels still stop at their own limits and bail.
                32 if nax_flash_wide else 0,
            )
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
            and 2 <= int(queries.shape[2]) <= 8
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
        if _env_enabled("MTPLX_ROUTE_DEBUG"):
            # One line per layer for the first 2 decode-shaped calls: which
            # branch the ladder takes and why — the 147.4k lane went dark
            # because every fast branch declined SILENTLY (2026-08-26).
            dbg_count = int(getattr(self, "_mtplx_route_debug_calls", 0))
            if dbg_count < 8 and int(queries.shape[2]) <= 16:
                self._mtplx_route_debug_calls = dbg_count + 1
                import sys as _sys

                print(
                    "mtplx_route_debug "
                    f"layer={getattr(self, '_mtplx_full_attention_index', -1)} "
                    f"q_len={int(queries.shape[2])} "
                    f"cache={type(cache).__name__} "
                    f"blockwise={blockwise_enabled} "
                    f"vllm_flag={bool(getattr(self, '_mtplx_vllm_metal_paged_enabled', False))} "
                    f"vllm_enabled={vllm_metal_paged_enabled} "
                    f"vllm_should={should_use_vllm_metal_paged} "
                    f"packed_enabled={gqa_packed_enabled} "
                    f"packed_should={should_use_gqa_packed} "
                    f"has_uwf={hasattr(cache, 'update_without_fetch')} "
                    f"has_pa={hasattr(cache, 'paged_attention')} "
                    f"keys_none={getattr(cache, 'keys', None) is None} "
                    f"cap={0 if getattr(cache, 'keys', None) is None else int(cache.keys.shape[2])} "
                    f"mask={type(mask).__name__} "
                    f"can_slice={can_slice_mask} "
                    f"twopass={should_use_2pass}",
                    file=_sys.stderr,
                    flush=True,
                )
        should_split = (
            cache is not None
            and getattr(self, "_mtplx_split_full_attention_explicit_enabled", False)
            and cached_prefix_len is not None
            and cached_prefix_len >= threshold
            and int(queries.shape[2]) > max(1, chunk_size)
            and can_slice_mask
        )
        route = "sdpa"
        fallback = ""
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
            route = _paged_route_label(cache)
            if output is None:
                route = "dense_state_sdpa"
                fallback = _paged_decline_label(cache, mask)
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
            from .kernels.sdpa_gqa_packed import (
                sdpa_gqa_packed_tail,
                sdpa_gqa_packed_tail_grouped,
            )

            output = None
            # MTPLX_NAX_FLASH_ROUTE (2026-09-01 hyper K2): TensorOps
            # flash-decoding kernel (in-threadgroup key split, no V staging)
            # for every packed-eligible window. Walk bench 72.7k QL4:
            # 1.08 ms/layer vs packed 1.42 (-24%) at half the power. Bails
            # fall through to the scalar routes unchanged.
            if _env_enabled("MTPLX_NAX_FLASH_ROUTE") and int(queries.shape[2]) >= 2:
                from .kernels.sdpa_nax_flash import sdpa_nax_flash
                from .kernels.sdpa_nax_flash_dsplit import (
                    sdpa_nax_flash_dsplit,
                    sdpa_nax_flash_dsplit_wide,
                )

                # Variant B (head-dim split, 64 accumulators/thread, no spills) owns the
                # M<=32 windows (QL<=5 at GQA 6): 0.917 vs 1.015 ms/layer at 72.7k, 0.257 vs
                # 0.306 at 16k. Variant A covers the wider windows (QL 6-10). Both bail to
                # the scalar routes on any contract miss.
                if not lane_disabled("nax_flash_dsplit_sdpa"):
                    output = sdpa_nax_flash_dsplit(
                        queries=queries,
                        keys=cache.keys,
                        values=cache.values,
                        offset=cache.offset,
                        scale=self.scale,
                    )
                # MTPLX_NAX_FLASH_WIDE: q_len 9..32 (54..192 rows per KV head) in one
                # KV pass, the query rows tiled in groups inside the dsplit kernel.
                # q_len 6..8 stay on variant A, which is as fast there.
                if (
                    output is None
                    and nax_flash_wide
                    and int(queries.shape[2]) >= 9
                    and not lane_disabled("nax_flash_dsplit_wide_sdpa")
                ):
                    output = sdpa_nax_flash_dsplit_wide(
                        queries=queries,
                        keys=cache.keys,
                        values=cache.values,
                        offset=cache.offset,
                        scale=self.scale,
                    )
                if output is None and not lane_disabled("nax_flash_sdpa"):
                    output = sdpa_nax_flash(
                        queries=queries,
                        keys=cache.keys,
                        values=cache.values,
                        offset=cache.offset,
                        scale=self.scale,
                    )
                if output is not None:
                    route = "nax_flash"
                    self._mtplx_nax_flash_calls = (
                        int(getattr(self, "_mtplx_nax_flash_calls", 0)) + 1
                    )
                    if _env_enabled("MTPLX_GQA_PACKED_SDPA_TRACE") and (
                        self._mtplx_nax_flash_calls <= 2
                    ):
                        import sys as _sys

                        print(
                            "mtplx_nax_flash_route engaged "
                            f"layer={getattr(self, '_mtplx_full_attention_index', -1)} "
                            f"q_len={int(queries.shape[2])} "
                            f"capacity={int(cache.keys.shape[2])}",
                            file=_sys.stderr,
                            flush=True,
                        )
            if output is not None:
                pass
            # MTPLX_NAX_TILE_ROUTE (2026-08-26 hyper): TensorOps wide-M tile
            # kernel for q_len >= 6 — the M-curve regime where the scalar
            # kernels pay (Battery A + spot receipts: QL9 +34%/+45% at
            # 71k/128k). Bails fall through to the scalar routes unchanged.
            elif (
                _env_enabled("MTPLX_NAX_TILE_ROUTE")
                and int(queries.shape[2]) >= 6
                and not lane_disabled("nax_tile_sdpa")
            ):
                from .kernels.sdpa_nax_tile import sdpa_nax_tile

                output = sdpa_nax_tile(
                    queries=queries,
                    keys=cache.keys,
                    values=cache.values,
                    offset=cache.offset,
                    scale=self.scale,
                )
                if output is not None:
                    route = "nax_tile"
                    self._mtplx_nax_tile_calls = (
                        int(getattr(self, "_mtplx_nax_tile_calls", 0)) + 1
                    )
            if output is not None:
                pass
            # Grouped wins past the second-bank register cliff: 2026-08-25
            # three-way sweep at 71k — bank2 ahead at QL5-6; grouped ahead
            # from QL7 (QL8 68.6 vs stock 83.7; QL9 with the mixed 4+5 v3
            # tail 61.1 vs stock 83.6, -27%).
            elif gqa_packed_wide and int(queries.shape[2]) >= 7:
                route = "gqa_packed_grouped"
                output = sdpa_gqa_packed_tail_grouped(
                    queries=queries,
                    keys=cache.keys,
                    values=cache.values,
                    offset=cache.offset,
                    scale=self.scale,
                )
            else:
                route = "gqa_packed"
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
                _count_gqa_packed_route_engaged(
                    int(queries.shape[2]), int(cache.keys.shape[2])
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
                route = "sdpa"
                fallback = "gqa_packed_declined"
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
            route = "sdpa_2pass"
            if output is None:
                route = "sdpa"
                fallback = "sdpa_2pass_declined"
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

            route = "blockwise"
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
            route = "split_sdpa"
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
        _note_kv_attention(self, cache, queries, output, mask, route, fallback)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attention_gate(output, gate))

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
        if attn is None:
            continue
        # Modules whose attention semantics are not plain dense SDPA (e.g.
        # the qwen4_exp QSA indexer mask) opt out class-side; hooking them
        # would replace their __call__ with a rewrite that drops those
        # semantics.
        if getattr(attn, "_mtplx_generic_sdpa_rewrites_unsupported", False):
            continue
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
