"""Construction-bound physical-M6 target replay for the Mia DSpark model.

Only the cache-free regions are compiled.  Each native attention invocation is
kept between those regions so its current RoPE epoch, cache pages, compressor
frontier, and index records remain ordinary eager runtime state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mtplx.attention_context import current_attention_phase


_MIA_TARGET_LAYERS = 43
_MIA_TAP_START = 40
_PHYSICAL_WIDTH = 6


class MiaDFlashTargetPhaseRoute:
    """Construction-owned phase table for prefill and physical-M6 verify."""

    def __init__(self, *, prefill, decode_verify) -> None:
        self._routes = {
            "prefill": prefill,
            "decode_verify": decode_verify,
        }

    def __call__(self, input_ids, cache):
        return self._routes[current_attention_phase()](input_ids, cache)


class MiaPhysicalM6PiecewiseTargetRoute:
    """Fixed-width target body with compiled pure regions around eager attention."""

    def __init__(
        self,
        model,
        *,
        physical_width: int,
        compile_fn: Callable[[Callable[..., Any]], Callable[..., Any]] = mx.compile,
    ) -> None:
        if int(physical_width) != _PHYSICAL_WIDTH:
            raise ValueError("the Mia target route requires physical width 6")
        layers = tuple(model.layers)
        if len(layers) != _MIA_TARGET_LAYERS:
            raise ValueError("the Mia target route requires exactly 43 target layers")

        self.physical_width = _PHYSICAL_WIDTH
        self._hidden_size = int(model.args.hidden_size)
        self._hc_mult = int(model.args.hc_mult)
        self._layers = layers
        self._attentions = tuple(
            layer.attn._mia_m6_forward_impl for layer in layers
        )
        self._prefix = compile_fn(self._make_prefix(model, layers[0]))
        self._regions = tuple(
            compile_fn(
                self._make_region(
                    model,
                    layer_id,
                    layer,
                    layers[layer_id + 1] if layer_id + 1 < _MIA_TARGET_LAYERS else None,
                )
            )
            for layer_id, layer in enumerate(layers)
        )

    @staticmethod
    def _make_prefix(model, first_layer):
        embed_tokens = model.embed_tokens
        mhc = model._mia_mhc
        attn_hc = first_layer.attn_hc
        attn_norm = first_layer.attn_norm

        def prefix(input_ids):
            return mhc.pre_broadcast(
                embed_tokens(input_ids),
                attn_hc,
                attn_norm,
            )

        return prefix

    def _make_region(self, model, layer_id, layer, next_layer):
        mhc = model._mia_mhc
        ffn_hc = layer.ffn_hc
        ffn_norm = layer.ffn_norm
        ffn = self._make_fixed_m6_ffn(layer.ffn)
        hidden_size = self._hidden_size
        hc_mult = self._hc_mult
        lead = (1, self.physical_width)

        if next_layer is None:

            def final_region(value, residual, post, comb, input_ids):
                residual, post, comb, value = mhc.post_pre_ffn(
                    value,
                    residual,
                    post,
                    comb,
                    ffn_hc,
                    ffn_norm,
                )
                value = ffn(value, input_ids)
                reconstructed = mhc.post(value, residual, post, comb)
                hidden = reconstructed.reshape(*lead, hc_mult, hidden_size)
                tap = mx.mean(hidden, axis=-2)
                return hidden, tap

            return final_region

        next_attn_hc = next_layer.attn_hc
        next_attn_norm = next_layer.attn_norm

        if layer_id >= _MIA_TAP_START:

            def tail_region(value, residual, post, comb, input_ids):
                residual, post, comb, value = mhc.post_pre_ffn(
                    value,
                    residual,
                    post,
                    comb,
                    ffn_hc,
                    ffn_norm,
                )
                value = ffn(value, input_ids)
                reconstructed = mhc.post(value, residual, post, comb)
                tap = mx.mean(
                    reconstructed.reshape(*lead, hc_mult, hidden_size),
                    axis=-2,
                )
                residual, post, comb, value = mhc.post_pre_attn(
                    value,
                    residual,
                    post,
                    comb,
                    next_attn_hc,
                    next_attn_norm,
                )
                return residual, post, comb, value, tap

            return tail_region

        def middle_region(value, residual, post, comb, input_ids):
            residual, post, comb, value = mhc.post_pre_ffn(
                value,
                residual,
                post,
                comb,
                ffn_hc,
                ffn_norm,
            )
            value = ffn(value, input_ids)
            return mhc.post_pre_attn(
                value,
                residual,
                post,
                comb,
                next_attn_hc,
                next_attn_norm,
            )

        return middle_region

    def _make_fixed_m6_ffn(self, ffn):
        input_rows = ffn._input_rows_impl
        gate = ffn.gate
        shared_experts = ffn.shared_experts
        routed_with_shared = ffn._mia_exl3_m6_fused
        physical_width = self.physical_width
        hidden_size = self._hidden_size
        lead = (1, physical_width)

        def fixed_m6_ffn(value, input_ids):
            rows = value.reshape(physical_width, hidden_size)
            ids = input_rows(input_ids)
            shared = shared_experts(rows)
            indices, weights = gate(rows, ids)
            return routed_with_shared(
                rows,
                indices,
                weights,
                shared,
            ).reshape(*lead, hidden_size)

        return fixed_m6_ffn

    def __call__(self, input_ids, cache):
        cycle = cache[0]._mia_m6_schedule.current_cycle
        residual, post, comb, value = self._prefix(input_ids)
        lead = (1, self.physical_width)

        for layer_id in range(_MIA_TAP_START):
            value = self._attentions[layer_id](
                value.reshape(*lead, self._hidden_size),
                cache=cache[layer_id],
                schedule=cycle.by_layer[layer_id],
            )
            residual, post, comb, value = self._regions[layer_id](
                value,
                residual,
                post,
                comb,
                input_ids,
            )

        taps = []
        for layer_id in range(_MIA_TAP_START, _MIA_TARGET_LAYERS - 1):
            value = self._attentions[layer_id](
                value.reshape(*lead, self._hidden_size),
                cache=cache[layer_id],
                schedule=cycle.by_layer[layer_id],
            )
            residual, post, comb, value, tap = self._regions[layer_id](
                value,
                residual,
                post,
                comb,
                input_ids,
            )
            taps.append(tap)

        final_layer_id = _MIA_TARGET_LAYERS - 1
        value = self._attentions[final_layer_id](
            value.reshape(*lead, self._hidden_size),
            cache=cache[final_layer_id],
            schedule=cycle.by_layer[final_layer_id],
        )
        hidden, final_tap = self._regions[final_layer_id](
            value,
            residual,
            post,
            comb,
            input_ids,
        )
        return hidden, (*taps, final_tap)


def install_mia_physical_m6_piecewise_target(model):
    """Validate the fixed Mia body once and bind its direct target route."""

    if (
        int(model.args.hidden_size) != 4096
        or int(model.args.hc_mult) != 4
        or len(tuple(model.layers)) != _MIA_TARGET_LAYERS
    ):
        raise ValueError("Mia piecewise target requires hidden=4096, hc=4, layers=43")
    route = MiaPhysicalM6PiecewiseTargetRoute(
        model,
        physical_width=_PHYSICAL_WIDTH,
    )
    model._mia_piecewise_target_route = route
    return route
