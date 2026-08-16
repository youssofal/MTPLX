"""Construction-bound physical-M3 target traversal for Flash-0731 DSpark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx


_ROWS = 3
_LAYERS = 43
_HIDDEN = 4096
_HEADS = 64
_HEAD_DIM = 512
_HC = 4
_TAP_LAYERS = (40, 41, 42)


class M3TargetContractError(ValueError):
    """The loaded owner cannot bind the fixed physical-M3 route."""


def _contract_error(detail: str) -> M3TargetContractError:
    return M3TargetContractError(
        f"DeepSeek-V4 0731 physical-M3 contract failed: {detail}"
    )


@dataclass(frozen=True, slots=True)
class M3TargetContract:
    layers: int
    hidden_size: int
    hc_mult: int
    target_layer_ids: tuple[int, int, int]


def validate_0731_m3_target(model: Any) -> M3TargetContract:
    """Validate full-artifact ownership once before route construction."""

    args = getattr(model, "args", None)
    body = getattr(model, "model", None)
    dspark = getattr(model, "_dspark", None)
    if args is None or body is None or dspark is None:
        raise _contract_error("requires a loaded full 0731 DSpark owner")
    expected = {
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": 1,
        "head_dim": _HEAD_DIM,
    }
    for name, expected_value in expected.items():
        if int(getattr(args, name, -1)) != expected_value:
            raise _contract_error(f"{name} is not {expected_value}")
    layers = tuple(getattr(body, "layers", ()))
    if len(layers) != _LAYERS:
        raise _contract_error(f"body owns {len(layers)} layers, expected {_LAYERS}")
    if int(getattr(body, "hc_mult", -1)) != _HC:
        raise _contract_error(f"body hc_mult is not {_HC}")
    taps = tuple(int(value) for value in getattr(dspark, "target_layer_ids", ()))
    if taps != _TAP_LAYERS:
        raise _contract_error("DSpark target taps are not (40, 41, 42)")
    if len(tuple(getattr(dspark, "stages", ()))) != 3:
        raise _contract_error("DSpark does not own exactly three stages")
    if not callable(getattr(body, "embed_tokens", None)):
        raise _contract_error("body embedding route is absent")
    if not callable(getattr(model, "logits_from_hc_hidden", None)):
        raise _contract_error("target logit route is absent")
    return M3TargetContract(
        layers=_LAYERS,
        hidden_size=_HIDDEN,
        hc_mult=_HC,
        target_layer_ids=_TAP_LAYERS,
    )


class _M3TargetBody:
    """Exact three-row trunk traversal through 43 prebound layer routes."""

    __slots__ = ("_body", "_layer_routes", "_tap_layers")

    def __init__(
        self,
        body: Any,
        layer_routes: Sequence[Callable],
        tap_layers: tuple[int, int, int],
    ) -> None:
        self._body = body
        self._layer_routes = tuple(layer_routes)
        self._tap_layers = tap_layers

    def __call__(self, input_ids: mx.array, cache=None) -> tuple[mx.array, mx.array]:
        hidden = self._body.embed_tokens(input_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], self._body.hc_mult, hidden.shape[-1]),
        )
        entries = (None,) * len(self._layer_routes) if cache is None else cache
        taps = []
        for layer_id, (route, entry) in enumerate(zip(self._layer_routes, entries)):
            hidden = route(hidden, input_ids, entry)
            if layer_id in self._tap_layers:
                taps.append(mx.mean(hidden, axis=2))
        return hidden, mx.concatenate(taps, axis=-1)


class M3CompiledTailLayer:
    """One native-width HC/attention boundary and one compiled M3 tail."""

    __slots__ = ("_attention", "_attn_hc_pre", "_attn_norm", "_m3_tail")

    def __init__(self, layer: Any, m3_tail: Callable) -> None:
        attention = getattr(layer, "attn", None)
        hc = getattr(layer, "attn_hc", None)
        norm = getattr(layer, "attn_norm", None)
        pre = getattr(hc, "pre", None)
        if not callable(attention) or not callable(pre) or not callable(norm):
            raise M3TargetContractError(
                "compiled M3 layer is missing its stock attention boundary"
            )
        if not callable(m3_tail):
            raise M3TargetContractError(
                "compiled M3 layer requires a callable width-three tail"
            )
        self._attention = attention
        self._attn_hc_pre = pre
        self._attn_norm = norm
        self._m3_tail = m3_tail

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache) -> mx.array:
        attention_in, post, comb = self._attn_hc_pre(hidden)
        attention_in = self._attn_norm(attention_in)
        attention_out = self._attention(attention_in, mask=None, cache=cache)
        return self._m3_tail(
            attention_out,
            hidden,
            post,
            comb,
            input_ids,
        )


def build_m3_compiled_tail_layer(
    layer: Any,
    m3_tail: Callable,
) -> M3CompiledTailLayer:
    return M3CompiledTailLayer(layer, m3_tail)


@dataclass(frozen=True, slots=True)
class FixedM3TargetRoute:
    """Explicit width table: physical M3 or the prebound native base route."""

    base: Callable
    m3_body: _M3TargetBody
    logits_from_hc_hidden: Callable

    def __call__(self, owner: Any, input_ids: mx.array, cache=None):
        if int(input_ids.shape[1]) == _ROWS:
            return self.m3_body(input_ids, cache)
        return self.base(owner, input_ids, cache)

    def forward(
        self,
        input_ids: mx.array,
        cache=None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        hidden, taps = self.m3_body(input_ids, cache)
        return self.logits_from_hc_hidden(hidden), hidden, taps


def build_0731_m3_target_route(
    model: Any,
    *,
    full_layer_routes: Sequence[Callable],
    base_route: Callable,
) -> FixedM3TargetRoute:
    """Build the sole fixed-M3 route; nothing is attached here."""

    contract = validate_0731_m3_target(model)
    routes = tuple(full_layer_routes)
    if len(routes) != contract.layers or not all(callable(route) for route in routes):
        raise _contract_error("requires exactly 43 prebound full-layer routes")
    if not callable(base_route):
        raise _contract_error("prebound native base target route is absent")
    return FixedM3TargetRoute(
        base=base_route,
        m3_body=_M3TargetBody(model.model, routes, contract.target_layer_ids),
        logits_from_hc_hidden=model.logits_from_hc_hidden,
    )
