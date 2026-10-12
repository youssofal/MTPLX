"""Construction-bound physical-M4 stock-QMM combine route for Flash-Next."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .kernels.qwen4_m4_stage3 import bind
from .kernels.qwen4_m4_routed_down import (
    bind as bind_routed_down_reduce,
    bind_residual_tail,
)
from .kernels.qwen4_m4_routed_glu import bind as bind_routed_glu
from .kernels.qwen4_m4_routed_down_q2 import (
    bind as bind_routed_down_reduce_q2g64,
    bind_residual_tail as bind_residual_tail_q2g64,
)
from .kernels.qwen4_m4_routed_glu_q2 import bind as bind_routed_glu_q2g64
from .kernels import qwen4_m4_route as _route_kernel
from .models.qwen4_exp import (
    DecoderLayer,
    SparseMoeBlock,
    _FusedGateUpMLP,
    _FusedGateUpSwitchGLU,
)
from .runtime_options import env_bool


def qwen4_m4_stage3_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_STAGE3", default=False)


#: Replace the ten-dispatch routing head (router q8 GEMV, precise softmax,
#: argpartition top-10, take_along_axis, bf16 renormalise, shared-gate GEMV and
#: its sigmoid) with the two kernels in ``kernels/qwen4_m4_route``.  Off by
#: default.  Bit-exactness is not asserted by argument: every layer's whole
#: ``(expert_ids, route_scores, shared_factor)`` tuple is compared against the
#: stock scaffold bit for bit at install (``_same_bits``), and a mismatch raises.
#: A flipped near-tie changes WHICH experts run, so there is no tolerance to
#: fall back on and no silent fallback path.
QWEN4_ROUTE_KERNEL_ENV = "MTPLX_QWEN4_ROUTE_KERNEL"

#: Threads per output row in the router GEMV: ``1`` reproduces MLX's own
#: ``qmv_wide`` thread layout (4,160 threads), ``4`` gives each verifier vector
#: its own lane octet (16,640 threads).  Both run the identical per-vector
#: accumulation, so this is a scheduling knob only.
QWEN4_ROUTE_KERNEL_VEC_LANES_ENV = "MTPLX_QWEN4_ROUTE_KERNEL_VEC_LANES"

_ROUTE_KERNEL_CACHE: bool | None = None
_ROUTE_KERNEL_VEC_LANES_CACHE: int | None = None


def qwen4_route_kernel_enabled() -> bool:
    """Return the ``MTPLX_QWEN4_ROUTE_KERNEL`` gate; read once, default off."""

    global _ROUTE_KERNEL_CACHE
    if _ROUTE_KERNEL_CACHE is None:
        _ROUTE_KERNEL_CACHE = env_bool(QWEN4_ROUTE_KERNEL_ENV, default=False)
    return _ROUTE_KERNEL_CACHE


def qwen4_route_kernel_vec_lanes() -> int:
    """Return the route GEMV's vector-lane count; read once, default 4.

    Raises on anything the kernel is not built for rather than clamping: a
    typo'd sweep value must not silently measure the default arm.
    """

    global _ROUTE_KERNEL_VEC_LANES_CACHE
    if _ROUTE_KERNEL_VEC_LANES_CACHE is None:
        raw = os.environ.get(QWEN4_ROUTE_KERNEL_VEC_LANES_ENV)
        if raw is None or raw.strip() == "":
            value = _route_kernel.DEFAULT_VEC_LANES
        else:
            try:
                value = int(raw.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{QWEN4_ROUTE_KERNEL_VEC_LANES_ENV}={raw!r} is not an "
                    "integer"
                ) from exc
            if value not in _route_kernel.VEC_LANES_CHOICES:
                raise ValueError(
                    f"{QWEN4_ROUTE_KERNEL_VEC_LANES_ENV}={value} is not one of "
                    f"{_route_kernel.VEC_LANES_CHOICES}"
                )
        _ROUTE_KERNEL_VEC_LANES_CACHE = value
    return _ROUTE_KERNEL_VEC_LANES_CACHE


def reset_qwen4_route_kernel_cache() -> None:
    """Drop the memoized route-kernel gates.  Test-support only."""

    global _ROUTE_KERNEL_CACHE, _ROUTE_KERNEL_VEC_LANES_CACHE
    _ROUTE_KERNEL_CACHE = None
    _ROUTE_KERNEL_VEC_LANES_CACHE = None


def qwen4_m4_routed_down_reduce_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", default=False)


def qwen4_m4_routed_down_residual_tail_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", default=False)


def qwen4_m4_routed_glu_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_GLU", default=False)


def qwen4_m4_stage3_flags() -> tuple[bool, bool, bool, bool]:
    """Capture and validate the complete construction-time feature route."""

    stage3_enabled = qwen4_m4_stage3_enabled()
    routed_down_reduce_enabled = qwen4_m4_routed_down_reduce_enabled()
    routed_down_residual_tail_enabled = (
        qwen4_m4_routed_down_residual_tail_enabled()
    )
    routed_glu_enabled = qwen4_m4_routed_glu_enabled()
    route_kernel_enabled = qwen4_route_kernel_enabled()
    if not stage3_enabled and (
        routed_down_reduce_enabled
        or routed_down_residual_tail_enabled
        or routed_glu_enabled
        or route_kernel_enabled
    ):
        raise ValueError("qwen4 M4 child routes require M4 stage3")
    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        route_kernel_enabled=route_kernel_enabled,
    )
    if route_kernel_enabled:
        # Read now so a typo'd sweep value fails at flag capture, before any
        # weight is touched.
        qwen4_route_kernel_vec_lanes()
    return (
        stage3_enabled,
        routed_down_reduce_enabled,
        routed_down_residual_tail_enabled,
        routed_glu_enabled,
    )


def _validate_feature_combination(
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    route_kernel_enabled: bool = False,
) -> None:
    if routed_down_residual_tail_enabled and not routed_down_reduce_enabled:
        raise ValueError(
            "qwen4 M4 routed-down residual tail requires routed-down reduction"
        )
    if routed_glu_enabled and not routed_down_residual_tail_enabled:
        raise ValueError("qwen4 M4 routed GLU requires routed residual tail")
    if route_kernel_enabled and not routed_glu_enabled:
        raise ValueError(
            f"{QWEN4_ROUTE_KERNEL_ENV} replaces the routing head of the paired "
            "routed-GLU lane and requires MTPLX_QWEN4_M4_ROUTED_GLU"
        )


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _m4_forward(block: SparseMoeBlock, x: mx.array, stage3) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input = mx.expand_dims(x, (-2, -3))
    routed_gate, routed_up = routed._gu(
        routed_input,
        expert_ids,
        sorted_indices=False,
    )
    routed_h = nn.silu(routed_gate) * routed_up
    routed_down = routed.down_proj(
        routed_h,
        expert_ids,
        sorted_indices=False,
    ).squeeze(-2).reshape(4, 10, 2560)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = stage3(
        routed_down,
        shared_down,
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


def _m4_routed_down_reduce_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_reduce,
) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input = mx.expand_dims(x, (-2, -3))
    routed_gate, routed_up = routed._gu(
        routed_input,
        expert_ids,
        sorted_indices=False,
    )
    routed_h = (nn.silu(routed_gate) * routed_up).reshape(4, 10, 640)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = routed_down_reduce(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


def _m4_routed_down_residual_tail_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_residual_tail,
    hyper: mx.array,
    inject: mx.array,
) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input = mx.expand_dims(x, (-2, -3))
    routed_gate, routed_up = routed._gu(
        routed_input,
        expert_ids,
        sorted_indices=False,
    )
    routed_h = (nn.silu(routed_gate) * routed_up).reshape(4, 10, 640)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    return routed_down_residual_tail(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
        hyper,
        inject,
    )


def _m4_paired_routed_glu_residual_tail_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_gu_activation,
    routed_down_residual_tail,
    hyper: mx.array,
    inject: mx.array,
    route=None,
) -> mx.array:
    """Paired routed GLU producer plus the fused residual tail.

    Both routed gathers on this lane happen inside Metal kernels (the paired
    GLU kernel resolves ``expert = expert_ids[selected]`` and walks the fused
    pack itself; the routed-down kernel does the same over its fixed
    ``SLOT_ORDER`` reduction tree), so the expert ids handed to them must keep
    the order the kernels were validated against.
    """

    # One metadata-only view, shared by the route head and the routed GLU.
    rows = x.reshape(4, 2560)
    if route is None:
        gates = mx.softmax(block.gate(x), axis=-1, precise=True)
        expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
            ..., -block.top_k :
        ]
        route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
        if block.norm_topk_prob:
            route_scores = route_scores / route_scores.sum(
                axis=-1, keepdims=True
            )
        expert_ids = expert_ids.reshape(4, 10)
        route_scores = route_scores.reshape(4, 10)
        shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)
    else:
        # MTPLX_QWEN4_ROUTE_KERNEL: the same ten dispatches in two, emitting
        # the tuple below directly. Validated bit-exact per layer at install.
        expert_ids, route_scores, shared_factor = route(
            rows,
            block.gate.weight,
            block.gate.scales,
            block.gate.biases,
            block.shared_expert_gate.weight,
            block.shared_expert_gate.scales,
            block.shared_expert_gate.biases,
        )

    routed = block.switch_mlp
    routed_h = routed_gu_activation(
        rows,
        routed.gu_weight,
        routed.gu_scales,
        routed.gu_biases,
        expert_ids,
    )

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)

    return routed_down_residual_tail(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids,
        route_scores.astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
        hyper,
        inject,
    )


def _m4_routed_down_residual_tail_layer_forward(
    layer: DecoderLayer,
    hidden: mx.array,
    *,
    input_ids: mx.array,
    ssm_mask: mx.array,
    cache: Any,
) -> mx.array:
    if "ple" in layer:
        hidden = hidden + layer.ple(hidden, input_ids, cache)

    mixed, hyper, inject = layer.attn_hyper_connection(hidden)
    if layer.is_linear:
        block_out = layer.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = layer.self_attn(mixed, cache)
    # The attention write rides into the MLP read (see DecoderLayer).
    mixed, hyper, inject = layer.mlp_hyper_connection(
        hyper, pending=(block_out, inject)
    )
    return _m4_routed_down_residual_tail_forward(
        layer.mlp,
        mixed,
        layer._mtplx_m4_routed_down_residual_tail,
        hyper,
        inject,
    )


def _m4_paired_routed_glu_residual_tail_layer_forward(
    layer: DecoderLayer,
    hidden: mx.array,
    *,
    input_ids: mx.array,
    ssm_mask: mx.array,
    cache: Any,
) -> mx.array:
    if "ple" in layer:
        hidden = hidden + layer.ple(hidden, input_ids, cache)

    mixed, hyper, inject = layer.attn_hyper_connection(hidden)
    if layer.is_linear:
        block_out = layer.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = layer.self_attn(mixed, cache)
    # The attention write rides into the MLP read (see DecoderLayer).
    mixed, hyper, inject = layer.mlp_hyper_connection(
        hyper, pending=(block_out, inject)
    )
    return _m4_paired_routed_glu_residual_tail_forward(
        layer.mlp,
        mixed,
        layer._mtplx_m4_routed_glu,
        layer._mtplx_m4_routed_down_residual_tail,
        hyper,
        inject,
        route=getattr(layer, "_mtplx_m4_route", None),
    )


class _M4Stage3SparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_forward(self, x, self._mtplx_m4_stage3)
        return super().__call__(x)


class _M4RoutedDownReduceSparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_routed_down_reduce_forward(
                self,
                x,
                self._mtplx_m4_routed_down_reduce,
            )
        return super().__call__(x)


class _M4RoutedDownResidualTailDecoderLayer(DecoderLayer):
    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        rows = int(hidden.size // hidden.shape[-1])
        if rows == 4:
            return _m4_routed_down_residual_tail_layer_forward(
                self,
                hidden,
                input_ids=input_ids,
                ssm_mask=ssm_mask,
                cache=cache,
            )
        return super().__call__(
            hidden,
            input_ids=input_ids,
            ssm_mask=ssm_mask,
            cache=cache,
        )


class _M4PairedRoutedGluResidualTailDecoderLayer(DecoderLayer):
    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        rows = int(hidden.size // hidden.shape[-1])
        if rows == 4:
            return _m4_paired_routed_glu_residual_tail_layer_forward(
                self,
                hidden,
                input_ids=input_ids,
                ssm_mask=ssm_mask,
                cache=cache,
            )
        return super().__call__(
            hidden,
            input_ids=input_ids,
            ssm_mask=ssm_mask,
            cache=cache,
        )


def _projection_contract(projection: Any) -> tuple[Any, ...]:
    return (
        int(projection.bits),
        int(projection.group_size),
        str(projection.mode),
        projection.weight.dtype,
        projection.scales.dtype,
        projection.biases.dtype,
        tuple(projection.weight.shape),
        tuple(projection.scales.shape),
        tuple(projection.biases.shape),
    )


def _fused_gu_contract(owner: Any) -> tuple[Any, ...]:
    return (
        int(owner.bits),
        int(owner.group_size),
        str(owner.mode),
        owner.gu_weight.dtype,
        owner.gu_scales.dtype,
        owner.gu_biases.dtype,
        tuple(owner.gu_weight.shape),
        tuple(owner.gu_scales.shape),
        tuple(owner.gu_biases.shape),
    )


_ROUTER_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 640),
    (512, 40),
    (512, 40),
)
_SHARED_GATE_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1, 640),
    (1, 40),
    (1, 40),
)
_ROUTED_GU_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 1280, 320),
    (512, 1280, 80),
    (512, 1280, 80),
)
_SHARED_GU_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1280, 640),
    (1280, 40),
    (1280, 40),
)
_ROUTED_DOWN_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 2560, 80),
    (512, 2560, 20),
    (512, 2560, 20),
)
_SHARED_DOWN_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (2560, 160),
    (2560, 10),
    (2560, 10),
)
#: Routed experts in MLX affine 2-bit, group 64: the layout of a ggml Q2_0
#: expert pack repacked without loss (scale = d, bias = -d, codes reused).
#: Served by the q2/g64 twins of the routed GLU and routed-down kernels.
_ROUTED_GU_CONTRACT_Q2G64 = (
    2,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 1280, 160),
    (512, 1280, 40),
    (512, 1280, 40),
)
_ROUTED_DOWN_CONTRACT_Q2G64 = (
    2,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 2560, 40),
    (512, 2560, 10),
    (512, 2560, 10),
)

ROUTED_Q4G32 = (4, 32)
ROUTED_Q2G64 = (2, 64)
#: Routed-expert geometry -> (fused gate/up contract, down contract).
_ROUTED_CONTRACTS = {
    ROUTED_Q4G32: (_ROUTED_GU_CONTRACT, _ROUTED_DOWN_CONTRACT),
    ROUTED_Q2G64: (_ROUTED_GU_CONTRACT_Q2G64, _ROUTED_DOWN_CONTRACT_Q2G64),
}


def _block_routed_geometry(block: Any) -> tuple[int, int]:
    """(bits, group_size) of a block's routed fused gate/up owner.

    An owner that passed ``_validate_block_contract`` carries both; anything
    else keeps the q4/g32 geometry the lane shipped with."""

    routed = getattr(block, "switch_mlp", None)
    return (
        int(getattr(routed, "bits", ROUTED_Q4G32[0])),
        int(getattr(routed, "group_size", ROUTED_Q4G32[1])),
    )


def _routed_binders(geometry: tuple[int, int]):
    """(routed GLU, routed-down reduce, routed-down residual tail) binders.

    Resolved through the module globals at call time, so a test that
    monkeypatches one binder sees it used."""

    if geometry == ROUTED_Q2G64:
        return (
            bind_routed_glu_q2g64,
            bind_routed_down_reduce_q2g64,
            bind_residual_tail_q2g64,
        )
    return bind_routed_glu, bind_routed_down_reduce, bind_residual_tail


def _validate_layer_owner(layer: Any, *, index: int) -> None:
    if type(layer) is not DecoderLayer:
        raise ValueError(f"qwen4 M4 stage3 layer {index} has wrong decoder owner")


def _validate_block_contract(block: Any, *, index: int) -> None:
    label = f"qwen4 M4 stage3 layer {index}"
    if type(block) is not SparseMoeBlock:
        raise ValueError(f"{label} has wrong MoE owner")
    if int(block.num_experts) != 512:
        raise ValueError(f"{label} requires 512 experts")
    if int(block.top_k) != 10:
        raise ValueError(f"{label} requires exact top-k 10")
    if not bool(block.norm_topk_prob):
        raise ValueError(f"{label} requires top-k probability normalization")
    if block.sharding_group is not None:
        raise ValueError(f"{label} does not support sharding")
    if type(block.switch_mlp) is not _FusedGateUpSwitchGLU:
        raise ValueError(f"{label} lacks exact routed fused GU owner")
    if type(block.shared_expert) is not _FusedGateUpMLP:
        raise ValueError(f"{label} lacks exact shared fused GU owner")
    if _projection_contract(block.gate) != _ROUTER_CONTRACT:
        raise ValueError(f"{label} router mismatch")
    if _projection_contract(block.shared_expert_gate) != _SHARED_GATE_CONTRACT:
        raise ValueError(f"{label} shared gate mismatch")
    routed_contracts = _ROUTED_CONTRACTS.get(_block_routed_geometry(block))
    if (
        routed_contracts is None
        or _fused_gu_contract(block.switch_mlp) != routed_contracts[0]
    ):
        raise ValueError(f"{label} routed fused GU mismatch")
    if _fused_gu_contract(block.shared_expert) != _SHARED_GU_CONTRACT:
        raise ValueError(f"{label} shared fused GU mismatch")
    if _projection_contract(block.switch_mlp.down_proj) != routed_contracts[1]:
        raise ValueError(f"{label} routed down mismatch")
    if _projection_contract(block.shared_expert.down_proj) != _SHARED_DOWN_CONTRACT:
        raise ValueError(f"{label} shared down mismatch")


def _validate_input_contract(layer: Any, *, index: int) -> None:
    norm = getattr(getattr(layer, "mlp_hyper_connection", None), "hc_norm", None)
    weight = getattr(norm, "weight", None)
    if (
        weight is None
        or tuple(weight.shape) != (4 * 2560,)
        or weight.dtype != mx.bfloat16
    ):
        raise ValueError(
            f"qwen4 M4 stage3 layer {index} requires BF16 hidden input ownership"
        )


def _validate_residual_tail_contract(layer: Any, *, index: int) -> None:
    """Admit only the exact BF16 hyper geometry consumed by the fused store."""

    label = f"qwen4 M4 residual tail layer {index}"
    connection = getattr(layer, "mlp_hyper_connection", None)
    if (
        connection is None
        or int(getattr(connection, "hc_count", -1)) != 4
        or int(getattr(connection, "hidden_size", -1)) != 2560
    ):
        raise ValueError(f"{label} requires exactly four hyper streams")
    inject_projection = getattr(connection, "block_inject_weight", None)
    inject_weight = getattr(inject_projection, "weight", None)
    if (
        inject_weight is None
        or tuple(inject_weight.shape) != (4, 4 * 2560)
        or inject_weight.dtype != mx.bfloat16
    ):
        raise ValueError(f"{label} requires BF16 inject ownership")


def bind_qwen4_m4_residual_tail(layer: Any, *, index: int):
    """Validate and bind the optional M4 residual tail at construction."""

    _validate_input_contract(layer, index=index)
    _validate_residual_tail_contract(layer, index=index)
    _validate_block_contract(layer.mlp, index=index)
    _, _, bind_tail = _routed_binders(_block_routed_geometry(layer.mlp))
    return bind_tail()


def _build_install_plans(
    layers,
    *,
    stage3,
    routed_down_reduce,
    routed_down_residual_tail_enabled: bool,
):
    plans = []
    for index, layer in enumerate(layers):
        block = layer.mlp
        if routed_down_residual_tail_enabled:
            _validate_layer_owner(layer, index=index)
        _validate_input_contract(layer, index=index)
        _validate_block_contract(block, index=index)
        if routed_down_residual_tail_enabled:
            _validate_residual_tail_contract(layer, index=index)
        plans.append((layer, block, stage3, routed_down_reduce))
    return tuple(plans)


def _installation_report(
    *,
    layer_count: int,
    max_delta: float,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    route_kernel_enabled: bool = False,
    route_kernel_vec_lanes: int | None = None,
    routed_geometry: tuple[int, int] = ROUTED_Q4G32,
) -> dict[str, Any]:
    bits, group_size = routed_geometry
    routed_tag = f"q{bits}g{group_size}"
    return {
        "installed": True,
        "layers": layer_count,
        "rows": 4,
        "max_abs_diff": max_delta,
        "boundary": (
            f"paired_routed_{routed_tag}_glu_reduce_shared_add_mlp_residual"
            if routed_glu_enabled
            else f"routed_{routed_tag}_reduce_shared_add_mlp_residual"
            if routed_down_residual_tail_enabled
            else "stock_qmm_combine_tail"
        ),
        "reference_boundary": (
            "retained_m4_routed_down_then_stock_residual"
            if routed_down_residual_tail_enabled
            else "stock_sparse_moe_block"
        ),
        "routed": f"stock_q{bits}/g{group_size}",
        "shared": "stock_q8/g64",
        "routed_down_reduce": routed_down_reduce_enabled,
        "routed_down_residual_tail": routed_down_residual_tail_enabled,
        "paired_routed_glu": routed_glu_enabled,
        "paired_routed_glu_layers": layer_count if routed_glu_enabled else 0,
        "route_kernel": {
            "installed": bool(route_kernel_enabled),
            "layers": layer_count if route_kernel_enabled else 0,
            "vec_lanes": route_kernel_vec_lanes,
            "dispatches_per_layer": (
                _route_kernel.FUSED_DISPATCHES_PER_LAYER
                if route_kernel_enabled
                else _route_kernel.EAGER_DISPATCHES_PER_LAYER
            ),
        },
        "exact_layers": layer_count,
        "combined_residual_tail_layers": (
            layer_count if routed_down_residual_tail_enabled else 0
        ),
    }


def _install_validated_plans(
    plans,
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    routed_glu=None,
    route=None,
) -> None:
    """Mutate only the complete plan set after validation and exact self-checks."""

    for layer, block, stage3, routed_down_reduce in plans:
        block._mtplx_m4_stage3 = stage3
        if routed_glu_enabled:
            layer._mtplx_m4_routed_glu = routed_glu
            layer._mtplx_m4_routed_down_residual_tail = routed_down_reduce
            # Plain callable or None; Module.__setattr__ keeps it off the
            # parameter dict, and the forward reads it with getattr so an
            # un-armed layer never pays a branch on a missing attribute.
            layer._mtplx_m4_route = route
            layer.__class__ = _M4PairedRoutedGluResidualTailDecoderLayer
        elif routed_down_residual_tail_enabled:
            layer._mtplx_m4_routed_down_residual_tail = routed_down_reduce
            layer.__class__ = _M4RoutedDownResidualTailDecoderLayer
        elif routed_down_reduce_enabled:
            block._mtplx_m4_routed_down_reduce = routed_down_reduce
            block.__class__ = _M4RoutedDownReduceSparseMoeBlock
        else:
            block.__class__ = _M4Stage3SparseMoeBlock


def _same_bits(want: mx.array, got: mx.array) -> mx.array:
    """``array_equal`` on bit patterns: -0 and +0 differ, as they do in a
    later sum, a sign test or a sort. Integer arrays compare as values."""

    if want.dtype != got.dtype or want.shape != got.shape:
        return mx.array(False)
    if mx.issubdtype(want.dtype, mx.floating):
        view = {2: mx.uint16, 4: mx.uint32}[want.dtype.size]
        return mx.array_equal(mx.view(want, view), mx.view(got, view))
    return mx.array_equal(want, got)


def install_qwen4_m4_stage3(
    runtime: Any,
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
) -> dict[str, Any]:
    """Validate every owner, self-check the kernel, then install M4 directly."""

    text = _text_model(runtime)
    layers = tuple(text.model.layers)
    if len(layers) != 48:
        raise ValueError(f"qwen4 M4 stage3 requires 48 layers, got {len(layers)}")

    route_kernel_enabled = qwen4_route_kernel_enabled()
    route_kernel_vec_lanes = (
        qwen4_route_kernel_vec_lanes() if route_kernel_enabled else None
    )
    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        route_kernel_enabled=route_kernel_enabled,
    )
    validated_plans = _build_install_plans(
        layers,
        stage3=None,
        routed_down_reduce=None,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    if route_kernel_enabled:
        # Construction-bound: every pack the two kernels hardcode is checked
        # before anything is bound, with the offending field named. Runs after
        # _build_install_plans so an owner-type problem reports as one.
        for index, (_, block, _, _) in enumerate(validated_plans):
            _route_kernel.check_contract(block, index=index)
    # Every layer passed a supported routed contract; the kernels are bound
    # once for the whole stack, so the stack must share one geometry.
    geometries = {
        _block_routed_geometry(block) for _, block, _, _ in validated_plans
    }
    if len(geometries) != 1:
        raise ValueError(
            "qwen4 M4 stage3 requires one routed expert geometry across layers, "
            f"got {sorted(geometries)}"
        )
    (geometry,) = geometries
    bind_glu, bind_reduce, bind_tail = _routed_binders(geometry)
    stage3 = bind()
    routed_down_reduce = (
        bind_tail()
        if routed_down_residual_tail_enabled
        else bind_reduce()
        if routed_down_reduce_enabled
        else None
    )
    routed_glu = bind_glu() if routed_glu_enabled else None
    route = (
        _route_kernel.bind(vec_lanes=route_kernel_vec_lanes)
        if route_kernel_enabled
        else None
    )
    plans = tuple(
        (layer, block, stage3, routed_down_reduce)
        for layer, block, _, _ in validated_plans
    )

    sample = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.0009765625)
    sample = sample.reshape(1, 4, 2560).astype(mx.bfloat16)
    max_delta = 0.0
    for index, (layer, block, stage3, routed_down_reduce) in enumerate(plans):
        if route is not None:
            # The route head decides WHICH experts run. A near-tie that flips
            # is not a rounding-class difference, so this gate compares all
            # three emitted tensors bit for bit, no tolerance, no fallback.
            _route_kernel.check_input(sample.reshape(4, 2560))
            reference_route = _route_kernel.stock_route(block, sample)
            candidate_route = route(
                sample.reshape(4, 2560),
                block.gate.weight,
                block.gate.scales,
                block.gate.biases,
                block.shared_expert_gate.weight,
                block.shared_expert_gate.scales,
                block.shared_expert_gate.biases,
            )
            names = ("expert_ids", "route_scores", "shared_factor")
            checks = tuple(
                _same_bits(want, got)
                for want, got in zip(reference_route, candidate_route)
            )
            mx.eval(checks)
            for name, ok in zip(names, checks):
                if not bool(ok.item()):
                    raise ValueError(
                        f"{QWEN4_ROUTE_KERNEL_ENV} layer {index}: {name} is "
                        "not bit-exact with the stock routing scaffold"
                    )
        stock = None if routed_down_residual_tail_enabled else block(sample)
        reference = _m4_forward(block, sample, stage3)
        if routed_down_residual_tail_enabled:
            hyper = mx.sin(
                mx.arange(4 * 4 * 2560, dtype=mx.float32) * 0.000244140625
            ).reshape(1, 4, 4 * 2560).astype(mx.bfloat16)
            inject = mx.cos(
                mx.arange(4 * 4, dtype=mx.float32) * 0.0625
            ).reshape(1, 4, 4).astype(mx.bfloat16)
            candidate = (
                _m4_paired_routed_glu_residual_tail_forward(
                    block,
                    sample,
                    routed_glu,
                    routed_down_reduce,
                    hyper,
                    inject,
                    route=route,
                )
                if routed_glu_enabled
                else _m4_routed_down_residual_tail_forward(
                    block,
                    sample,
                    routed_down_reduce,
                    hyper,
                    inject,
                )
            )
            reference = hyper + (
                reference[..., None, :] * inject[..., :, None]
            ).reshape(*hyper.shape)
        elif routed_down_reduce_enabled:
            candidate = _m4_routed_down_reduce_forward(
                block, sample, routed_down_reduce
            )
        else:
            candidate = reference
        if routed_down_reduce_enabled:
            exact = _same_bits(reference, candidate)
            mx.eval(exact)
            if not bool(exact.item()):
                raise ValueError(
                    f"qwen4 M4 stage3 layer {index} routed-down "
                    "reduction self-check failed"
                )
        comparison = reference if routed_down_residual_tail_enabled else stock
        delta = mx.max(
            mx.abs(comparison.astype(mx.float32) - candidate.astype(mx.float32))
        )
        finite = mx.all(mx.isfinite(candidate))
        mx.eval(delta, finite)
        observed = float(delta.item())
        if not bool(finite.item()) or observed > 0.001953125:
            raise ValueError(
                f"qwen4 M4 stage3 layer {index} self-check failed: dmax={observed}"
            )
        max_delta = max(max_delta, observed)

    _install_validated_plans(
        plans,
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        routed_glu=routed_glu,
        route=route,
    )
    report = _installation_report(
        layer_count=len(plans),
        max_delta=max_delta,
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        route_kernel_enabled=route_kernel_enabled,
        route_kernel_vec_lanes=route_kernel_vec_lanes,
        routed_geometry=geometry,
    )
    runtime.qwen4_m4_stage3_report = report
    return report


__all__ = [
    "QWEN4_ROUTE_KERNEL_ENV",
    "QWEN4_ROUTE_KERNEL_VEC_LANES_ENV",
    "qwen4_route_kernel_enabled",
    "qwen4_route_kernel_vec_lanes",
    "install_qwen4_m4_stage3",
    "qwen4_m4_routed_down_reduce_enabled",
    "qwen4_m4_routed_down_residual_tail_enabled",
    "qwen4_m4_routed_glu_enabled",
    "qwen4_m4_stage3_enabled",
    "qwen4_m4_stage3_flags",
    "reset_qwen4_route_kernel_cache",
]
