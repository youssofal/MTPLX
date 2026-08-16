"""Fixed-shape post-attention verifier islands for DeepSeek-V4-Flash.

Attention and its cache stay on the ordinary eager path with the exact logical
compressed slice.  Only the dependency chain after attention is compiled:
attention HC-post, FFN HC-pre/RMSNorm, router, stock affine gather-QMM MoE,
route reduction/shared add, and FFN HC-post.

The production checkpoint has three structural layer layouts (hash/gs32,
score/gs32, score/gs64).  Combined with physical M2/M3/M4 this produces nine
module-level tapes.  Layer weights are array inputs, so 43 layers do not create
129 layer-owned compiler functions or close checkpoint arrays into a tape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from .attention_context import current_attention_phase, current_model_forward_kind
from .deepseek_v4_0731_moe import DeepseekV40731PackedQ2SwitchGLU
from .moe_packed_projections import PackedSwitchGLU
from .models import deepseek_v4 as D


_WIDTHS = (2, 3, 4)


def deepseek_v4_attention_island_enabled() -> bool:
    """Read the opt-in once while constructing the loaded runtime."""

    return os.environ.get("MTPLX_DSV4_ATTENTION_ISLAND", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class AttentionIslandError(RuntimeError):
    """The exact post-attention island could not be installed safely."""


@dataclass(frozen=True, slots=True)
class _Projection:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int
    output_dim: int
    input_dim: int


def _projection_contract(module: Any, label: str, *, expected_bits: int) -> _Projection:
    """Validate one stock affine projection once and bind its array leaves."""

    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    bits = int(getattr(module, "bits", -1))
    group_size = int(getattr(module, "group_size", -1))
    mode = str(getattr(module, "mode", "")).lower()
    if bits != expected_bits or mode != "affine":
        raise AttentionIslandError(
            f"{label} must be {expected_bits}-bit affine; got bits={bits}, "
            f"mode={mode!r}"
        )
    if group_size not in {32, 64, 128}:
        raise AttentionIslandError(
            f"{label} has unsupported affine group_size={group_size}"
        )
    if (
        getattr(weight, "dtype", None) != mx.uint32
        or getattr(scales, "dtype", None) not in {mx.bfloat16, mx.float32}
        or getattr(biases, "dtype", None) not in {mx.bfloat16, mx.float32}
    ):
        raise AttentionIslandError(
            f"{label} requires U32 weights and floating affine scale/bias leaves"
        )
    if (
        getattr(weight, "ndim", -1) not in {2, 3}
        or getattr(scales, "ndim", -1) != weight.ndim
        or getattr(biases, "ndim", -1) != weight.ndim
        or tuple(weight.shape[:-1]) != tuple(scales.shape[:-1])
        or tuple(scales.shape) != tuple(biases.shape)
    ):
        raise AttentionIslandError(f"{label} affine leaf geometry is invalid")
    input_dim = int(weight.shape[-1]) * (32 // bits)
    if int(scales.shape[-1]) * group_size != input_dim:
        raise AttentionIslandError(f"{label} packed input geometry is invalid")
    return _Projection(
        weight=weight,
        scales=scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        output_dim=int(weight.shape[-2]),
        input_dim=input_dim,
    )


def _qmm(x: mx.array, projection: _Projection) -> mx.array:
    return mx.quantized_matmul(
        x,
        projection.weight,
        scales=projection.scales,
        biases=projection.biases,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode="affine",
    )


def _gather_qmm(x: mx.array, indices: mx.array, projection: _Projection) -> mx.array:
    return mx.gather_qmm(
        x,
        projection.weight,
        scales=projection.scales,
        biases=projection.biases,
        rhs_indices=indices,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode="affine",
        sorted_indices=False,
    )


def _route(
    x: mx.array,
    input_ids: mx.array,
    weight: mx.array,
    auxiliary: mx.array,
    *,
    hash_router: bool,
    topk: int,
    score_func: str,
    route_scale: float,
) -> tuple[mx.array, mx.array]:
    """Exact :class:`MoEGate` order with topology fixed in the tape."""

    scores = x.astype(mx.float32) @ weight.astype(mx.float32).T
    if score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:
        scores = mx.sqrt(D.nn.softplus(scores))
    if hash_router:
        indices = auxiliary[input_ids.reshape(-1)]
    else:
        biased = scores + auxiliary
        indices = mx.argpartition(-biased, kth=topk - 1, axis=-1)[..., :topk]
    route_weights = mx.take_along_axis(scores, indices, axis=-1)
    if score_func != "softmax":
        route_weights = route_weights / mx.sum(route_weights, axis=-1, keepdims=True)
    return indices, route_weights * route_scale


def _moe(
    x: mx.array,
    indices: mx.array,
    route_weights: mx.array,
    routed_gate: _Projection | None,
    routed_up: _Projection | None,
    routed_down: _Projection,
    shared_gate: _Projection,
    shared_up: _Projection,
    shared_down: _Projection,
    *,
    routed_gate_up: _Projection | None,
    routed_combine: Callable | None,
    routed_limit: float,
    shared_limit: float,
) -> mx.array:
    """Stock unsorted Q2/Q4 arithmetic for the production top-6 tiny-M shape."""

    gathered_x = mx.expand_dims(x, (-2, -3))
    if routed_gate_up is None:
        up = _gather_qmm(gathered_x, indices, routed_up)  # type: ignore[arg-type]
        gate = _gather_qmm(gathered_x, indices, routed_gate)  # type: ignore[arg-type]
    else:
        packed = _gather_qmm(gathered_x, indices, routed_gate_up)
        gate, up = mx.split(packed, [routed_gate_up.output_dim // 2], axis=-1)
    if routed_limit > 0:
        up = mx.clip(up, -routed_limit, routed_limit)
        gate = mx.minimum(gate, routed_limit)
    routed = _gather_qmm(D.nn.silu(gate) * up, indices, routed_down)
    routed = routed.squeeze(-2)
    if routed_combine is None:
        routed = (routed * route_weights[..., None].astype(routed.dtype)).sum(axis=-2)
    else:
        routed = routed_combine(routed, route_weights)

    shared_gate_out = _qmm(x, shared_gate)
    shared_up_out = _qmm(x, shared_up)
    if shared_limit > 0:
        shared_up_out = mx.clip(shared_up_out, -shared_limit, shared_limit)
        shared_gate_out = mx.minimum(shared_gate_out, shared_limit)
    shared = _qmm(D.nn.silu(shared_gate_out) * shared_up_out, shared_down)
    return routed + shared


def _island_impl(
    attn_out: mx.array,
    attn_residual: mx.array,
    attn_post: mx.array,
    attn_comb: mx.array,
    input_ids: mx.array,
    ffn_fn_t: mx.array,
    ffn_base: mx.array,
    ffn_scale_vec: mx.array,
    norm_weight: mx.array,
    router_weight: mx.array,
    router_auxiliary: mx.array,
    routed_gate: _Projection | None,
    routed_up: _Projection | None,
    routed_down: _Projection,
    shared_gate: _Projection,
    shared_up: _Projection,
    shared_down: _Projection,
    *,
    routed_gate_up: _Projection | None,
    routed_combine: Callable | None,
    hc: int,
    iters: int,
    hc_eps: float,
    norm_eps: float,
    sinkhorn_kernel: bool,
    hash_router: bool,
    topk: int,
    score_func: str,
    route_scale: float,
    routed_limit: float,
    shared_limit: float,
) -> mx.array:
    h = D._hc_post_impl(attn_out, attn_residual, attn_post, attn_comb)
    ffn_residual = h

    if sinkhorn_kernel:

        def normalise(comb):
            return D._sinkhorn_kernel_apply(comb, hc, iters, hc_eps)

    else:

        def normalise(comb):
            return D._sinkhorn_ops(comb, iters, hc_eps)

    x, ffn_post, ffn_comb = D._hc_pre_impl(
        h,
        ffn_fn_t,
        ffn_base,
        ffn_scale_vec,
        hc,
        iters,
        hc_eps,
        normalise,
    )
    x = mx.fast.rms_norm(x, norm_weight, norm_eps)
    shape = x.shape
    xf = x.reshape(-1, shape[-1])
    indices, route_weights = _route(
        xf,
        input_ids,
        router_weight,
        router_auxiliary,
        hash_router=hash_router,
        topk=topk,
        score_func=score_func,
        route_scale=route_scale,
    )
    y = _moe(
        xf,
        indices,
        route_weights,
        routed_gate,
        routed_up,
        routed_down,
        shared_gate,
        shared_up,
        shared_down,
        routed_gate_up=routed_gate_up,
        routed_combine=routed_combine,
        routed_limit=routed_limit,
        shared_limit=shared_limit,
    ).reshape(shape)
    return D._hc_post_impl(y, ffn_residual, ffn_post, ffn_comb)


_TAPES: dict[tuple[Any, ...], Callable] = {}


def _attention_island_tape(
    *,
    width: int,
    hash_router: bool,
    routed_gate: _Projection | None,
    routed_up: _Projection | None,
    routed_gate_up: _Projection | None,
    routed_combine: Callable | None,
    routed_down: _Projection,
    shared_gate: _Projection,
    shared_up: _Projection,
    shared_down: _Projection,
    hc: int,
    iters: int,
    hc_eps: float,
    norm_eps: float,
    sinkhorn_kernel: bool,
    topk: int,
    score_func: str,
    route_scale: float,
    routed_limit: float,
    shared_limit: float,
) -> Callable:
    paired = routed_gate_up is not None
    if paired:
        effective_gate = effective_up = effective_pair = routed_gate_up
    else:
        assert routed_gate is not None and routed_up is not None
        effective_gate = routed_gate
        effective_up = routed_up
        # Preserve one compiled signature; this final leaf is unused when stock.
        effective_pair = routed_gate
    projections = (
        effective_gate,
        effective_up,
        routed_down,
        shared_gate,
        shared_up,
        shared_down,
        effective_pair,
    )
    key = (
        int(width),
        bool(hash_router),
        paired,
        routed_combine is not None,
        tuple((p.bits, p.group_size) for p in projections),
        int(hc),
        int(iters),
        float(hc_eps),
        float(norm_eps),
        bool(sinkhorn_kernel),
        int(topk),
        str(score_func),
        float(route_scale),
        float(routed_limit),
        float(shared_limit),
    )
    tape = _TAPES.get(key)
    if tape is not None:
        return tape

    specs = tuple(
        (p.bits, p.group_size, p.output_dim, p.input_dim) for p in projections
    )

    def impl(
        attn_out,
        attn_residual,
        attn_post,
        attn_comb,
        input_ids,
        ffn_fn_t,
        ffn_base,
        ffn_scale_vec,
        norm_weight,
        router_weight,
        router_auxiliary,
        rg_weight,
        rg_scales,
        rg_biases,
        ru_weight,
        ru_scales,
        ru_biases,
        rd_weight,
        rd_scales,
        rd_biases,
        sg_weight,
        sg_scales,
        sg_biases,
        su_weight,
        su_scales,
        su_biases,
        sd_weight,
        sd_scales,
        sd_biases,
        rgu_weight,
        rgu_scales,
        rgu_biases,
    ):
        arrays = (
            (rg_weight, rg_scales, rg_biases),
            (ru_weight, ru_scales, ru_biases),
            (rd_weight, rd_scales, rd_biases),
            (sg_weight, sg_scales, sg_biases),
            (su_weight, su_scales, su_biases),
            (sd_weight, sd_scales, sd_biases),
            (rgu_weight, rgu_scales, rgu_biases),
        )
        bound = tuple(
            _Projection(*leaves, *spec)
            for leaves, spec in zip(arrays, specs, strict=True)
        )
        brg, bru, brd, bsg, bsu, bsd, brgu = bound
        return _island_impl(
            attn_out,
            attn_residual,
            attn_post,
            attn_comb,
            input_ids,
            ffn_fn_t,
            ffn_base,
            ffn_scale_vec,
            norm_weight,
            router_weight,
            router_auxiliary,
            None if paired else brg,
            None if paired else bru,
            brd,
            bsg,
            bsu,
            bsd,
            routed_gate_up=brgu if paired else None,
            routed_combine=routed_combine,
            hc=hc,
            iters=iters,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            sinkhorn_kernel=sinkhorn_kernel,
            hash_router=hash_router,
            topk=topk,
            score_func=score_func,
            route_scale=route_scale,
            routed_limit=routed_limit,
            shared_limit=shared_limit,
        )

    tape = mx.compile(impl)
    _TAPES[key] = tape
    return tape


class _BoundAttentionIslandLayer:
    __slots__ = ("_tape", "_leaves", "width")

    def __init__(self, tape: Callable, leaves: tuple[mx.array, ...], width: int):
        self._tape = tape
        self._leaves = leaves
        self.width = int(width)

    def __call__(self, attn_out, attn_residual, attn_post, attn_comb, input_ids):
        return self._tape(
            attn_out,
            attn_residual,
            attn_post,
            attn_comb,
            input_ids,
            *self._leaves,
        )


def _bind_attention_island_layer(
    layer: D.DeepseekV4DecoderLayer,
    *,
    width: int,
    allowed_widths: tuple[int, ...] = _WIDTHS,
    shared_bits: int = 4,
    routed_pair: bool = False,
    routed_combine: Callable | None = None,
    routed_switch: Any | None = None,
) -> _BoundAttentionIslandLayer:
    """Validate and bind one layer; its hot call performs no discovery."""

    if int(width) not in allowed_widths:
        raise AttentionIslandError(f"unsupported verifier width {width}")
    if int(shared_bits) not in {4, 8}:
        raise AttentionIslandError(f"unsupported shared-expert bits {shared_bits}")
    if type(layer) is not D.DeepseekV4DecoderLayer:
        raise AttentionIslandError("requires an exact DeepseekV4DecoderLayer")
    ffn = layer.ffn
    if type(ffn) is not D.DeepseekV4MoE:
        raise AttentionIslandError("requires the stock DeepSeek-V4 MoE topology")
    switch = ffn.switch_mlp if routed_switch is None else routed_switch
    shared = ffn.shared_experts
    if routed_pair:
        if not isinstance(
            switch,
            (DeepseekV40731PackedQ2SwitchGLU, PackedSwitchGLU),
        ):
            raise AttentionIslandError("requires the packed routed Q2 topology")
    elif isinstance(switch, PackedSwitchGLU):
        raise AttentionIslandError("requires the stock routed Q2 topology")
    if type(switch.activation) is not D.ClampedSwiGLU:
        raise AttentionIslandError("requires the exact clamped SwiGLU activation")
    if routed_pair:
        rgu = _projection_contract(
            switch.gate_up_proj,
            "packed routed gate/up",
            expected_bits=2,
        )
        rd = _projection_contract(switch.down_proj, "routed down", expected_bits=2)
        if int(switch._split_at) * 2 != rgu.output_dim:
            raise AttentionIslandError("packed routed gate/up split is inconsistent")
        rg = ru = None
        input_dim = rgu.input_dim
        intermediate = int(switch._split_at)
        effective_routed = (rgu, rgu, rd, rgu)
    else:
        rg, ru, rd = tuple(
            _projection_contract(module, label, expected_bits=2)
            for module, label in (
                (switch.gate_proj, "routed gate"),
                (switch.up_proj, "routed up"),
                (switch.down_proj, "routed down"),
            )
        )
        rgu = None
        input_dim = rg.input_dim
        intermediate = rg.output_dim
        effective_routed = (rg, ru, rd, rg)
    sg, su, sd = tuple(
        _projection_contract(module, label, expected_bits=int(shared_bits))
        for module, label in (
            (shared.gate_proj, "shared gate"),
            (shared.up_proj, "shared up"),
            (shared.down_proj, "shared down"),
        )
    )
    if (
        (
            not routed_pair
            and (rg.input_dim != ru.input_dim or rg.output_dim != ru.output_dim)
        )
        or rd.input_dim != intermediate
        or rd.output_dim != input_dim
        or sg.input_dim != input_dim
        or sg.output_dim != su.output_dim
        or su.input_dim != input_dim
        or sd.input_dim != sg.output_dim
        or sd.output_dim != input_dim
    ):
        raise AttentionIslandError("routed/shared projection geometry is inconsistent")
    hc = layer.ffn_hc
    fn_t, base, scale_vec = hc._static()
    router = ffn.gate
    if routed_combine is not None and (
        int(width) != 1 or int(router.topk) != 6 or rd.output_dim != 4096
    ):
        raise AttentionIslandError(
            "row-owned combine requires the exact AR M1/top-6/hidden-4096 route"
        )
    auxiliary = router.tid2eid if router.hash else router.e_score_correction_bias
    tape = _attention_island_tape(
        width=width,
        hash_router=bool(router.hash),
        routed_gate=rg,
        routed_up=ru,
        routed_gate_up=rgu,
        routed_combine=routed_combine,
        routed_down=rd,
        shared_gate=sg,
        shared_up=su,
        shared_down=sd,
        hc=int(hc.hc),
        iters=int(hc._iters),
        hc_eps=float(hc.eps),
        norm_eps=float(layer.ffn_norm.eps),
        sinkhorn_kernel=bool(hc._sinkhorn_kernel),
        topk=int(router.topk),
        score_func=str(router.score_func),
        route_scale=float(router.route_scale),
        routed_limit=float(switch.activation.limit),
        shared_limit=float(shared.limit),
    )
    leaves = (
        fn_t,
        base,
        scale_vec,
        layer.ffn_norm.weight,
        router.weight,
        auxiliary,
        *(
            leaf
            for p in effective_routed[:3] + (sg, su, sd, effective_routed[3])
            for leaf in (p.weight, p.scales, p.biases)
        ),
    )
    return _BoundAttentionIslandLayer(tape, leaves, width)


class _BoundWidthBody:
    """One exact-width traversal: attention remains eager, islands are direct."""

    __slots__ = ("_body", "_layers", "width")

    def __init__(self, body: D.DeepseekV4Model, layers, width: int):
        self._body = body
        self._layers = tuple(layers)
        self.width = int(width)

    def __call__(self, input_ids: mx.array, cache=None):
        h = self._body.embed_tokens(input_ids)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (*h.shape[:2], self._body.hc_mult, h.shape[-1]),
        )
        if cache is None:
            cache = (None,) * len(self._layers)
        for (layer, island), entry in zip(self._layers, cache, strict=True):
            residual = h
            x, post, comb = layer.attn_hc.pre(h)
            x = layer.attn_norm(x)
            # This is deliberately eager and uses the real cache/logical slice.
            x = layer.attn(x, mask=None, cache=entry)
            h = island(x, residual, post, comb, input_ids)
        return h


@dataclass(frozen=True, slots=True)
class _AttentionIslandTargetRoute:
    """Installed target route; only real phase and logical M remain dynamic."""

    stock: Callable
    widths: dict[int, Callable]

    def __call__(self, input_ids: mx.array, cache=None):
        shape = tuple(int(dimension) for dimension in input_ids.shape)
        width = shape[1] if len(shape) == 2 and shape[0] == 1 else -1
        if (
            current_attention_phase() == "decode_verify"
            and current_model_forward_kind() == "target_verify"
            and width in _WIDTHS
        ):
            return self.widths[width](input_ids, cache)
        return self.stock(input_ids, cache)


class _AttentionIslandArmSelector:
    """Between-generation selector used by the one-load performance bracket."""

    __slots__ = ("_model", "stock", "candidate", "candidate_selected")

    def __init__(self, model: Any, stock: Callable, candidate: Callable):
        self._model = model
        self.stock = stock
        self.candidate = candidate
        self.candidate_selected = True
        model._target_hc_hidden_route = candidate

    def select(self, enabled: bool) -> None:
        self.candidate_selected = bool(enabled)
        self._model._target_hc_hidden_route = (
            self.candidate if self.candidate_selected else self.stock
        )


def select_deepseek_v4_attention_island_arm(model: Any, enabled: bool) -> None:
    """Select a preinstalled bracket arm outside measured generation."""

    selector = getattr(model, "_mtplx_dsv4_attention_island_selector", None)
    if (
        type(selector) is not _AttentionIslandArmSelector
        or selector._model is not model
    ):
        raise AttentionIslandError("attention-island arm selector is not installed")
    selector.select(enabled)


def _validate_model(model: D.Model, config: dict) -> None:
    if type(model) is not D.Model or type(model.model) is not D.DeepseekV4Model:
        raise AttentionIslandError("requires the native DeepSeek-V4 target model")
    if bool(getattr(D, "_FP32_ACTIVATIONS", False)):
        raise AttentionIslandError("requires BF16 activation storage")
    try:
        D._validate_loaded_moe_tail_contract(model, config)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AttentionIslandError(str(exc)) from exc
    for index, layer in enumerate(model.layers):
        for hc_name in ("attn_hc", "ffn_hc"):
            hc = getattr(layer, hc_name)
            if (
                int(hc.dim) != 4096
                or int(hc.hc) != 4
                or int(hc._iters) != 20
                or float(hc.eps) != 1e-6
                or tuple(hc.fn.shape) != (24, 16384)
                or tuple(hc.base.shape) != (24,)
                or tuple(hc.scale.shape) != (3,)
            ):
                raise AttentionIslandError(
                    f"layer {index} {hc_name} geometry is not canonical"
                )
        if (
            tuple(layer.ffn_norm.weight.shape) != (4096,)
            or layer.ffn_norm.weight.dtype != mx.bfloat16
            or float(layer.ffn_norm.eps) != 1e-6
        ):
            raise AttentionIslandError(
                f"layer {index} FFN RMSNorm storage is not canonical"
            )


def install_deepseek_v4_attention_island(
    model: D.Model, config: dict
) -> dict[str, Any]:
    """Validate the canonical checkpoint, prebind nine tapes, and install."""

    if getattr(model, "_dspark", None) is not None:
        raise AttentionIslandError(
            "DSpark requires tap-aware attention-island support; refusing to "
            "install a target route that its tap collector would bypass"
        )
    _validate_model(model, config)
    stock = getattr(model, "_target_hc_hidden_route", model.model.hc_hidden)
    bound_by_width: dict[int, _BoundWidthBody] = {}
    all_bound = []
    for width in _WIDTHS:
        bound_layers = tuple(
            (layer, _bind_attention_island_layer(layer, width=width))
            for layer in model.layers
        )
        all_bound.extend(island for _, island in bound_layers)
        bound_by_width[width] = _BoundWidthBody(model.model, bound_layers, width)
    route = _AttentionIslandTargetRoute(stock=stock, widths=bound_by_width)
    selector = _AttentionIslandArmSelector(model, stock, route)
    model._mtplx_dsv4_attention_island_selector = selector
    return {
        "installed": True,
        "widths": list(_WIDTHS),
        "body_layers": len(model.layers),
        "bound_layer_routes": len(all_bound),
        "shared_tapes": len({id(bound._tape) for bound in all_bound}),
        "expected_shared_tapes": 9,
        "attention": "eager_exact_logical_cache",
        "weight_binding": "explicit_array_inputs",
        "runtime_fallback": False,
        "hot_environment_reads": False,
        "hot_counters": False,
    }
