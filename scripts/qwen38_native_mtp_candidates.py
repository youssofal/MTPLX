#!/usr/bin/env python3
"""Immutable, CPU-only ownership registry for the Qwen 3.8 native-MTP campaign.

The registry describes construction-time route alternatives.  It deliberately
does not import MLX, inspect a model, read environment variables, or participate
in generation.  Runtime installation validates the corresponding artifact and
prebinds the selected callable before measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

NATIVE_MTP_DEPTH_MAX = 3
NATIVE_D3_VERIFY_LOGICAL_M = 4
NATIVE_SAMPLER = MappingProxyType(
    {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
)
QWEN38_AFFINE_PACKING = "mlx_affine_u32_le"


class NativeMTPRouteError(ValueError):
    """A control/candidate bracket does not isolate one native-MTP surface."""


@dataclass(frozen=True)
class CallsiteRegime:
    route: str
    phase: str
    logical_m: tuple[int, ...]
    dynamic_logical_m: str | None
    native_depths: tuple[int, ...]
    callsite_shape: str
    active_when: str
    bypassed_by: str | None = None
    active_candidate_surfaces: tuple[str, ...] = ()
    bypassed_candidate_surfaces: tuple[str, ...] = ()
    read_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParentRule:
    required_control_features: frozenset[str] = frozenset()
    replacement_only: bool = False
    replaces: frozenset[str] = frozenset()
    implicit_replaces: str | None = None


@dataclass(frozen=True)
class NativeMTPCandidate:
    row: int
    feature: str
    pr_number: int
    source_commit: str
    phase: str
    ownership: str
    owned_surfaces: tuple[str, ...]
    logical_m: tuple[int, ...]
    dynamic_logical_m: str | None
    native_depths: tuple[int, ...]
    callsite_shape: str
    callsite_regimes: tuple[CallsiteRegime, ...]
    dtypes: tuple[str, ...]
    quant_bits: int | None
    group_size: int | None
    packing: str | None
    parent_rule: ParentRule = ParentRule()
    incompatible: frozenset[str] = frozenset()
    unchanged_dependencies: tuple[str, ...] = ()
    min_context_tokens: int = 0
    artifact_variant: str | None = None
    artifact_manifest_sha256: str | None = None
    artifact_file_sha256: str | None = None
    artifact_bytes: int | None = None
    bypassed_candidate_callable_features: frozenset[str] = frozenset()
    partially_used_artifact_features: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FrozenSubstrate:
    row: int
    feature: str
    pr_number: int
    source_commit: str
    phase: str
    owned_surface: str


@dataclass(frozen=True)
class ExcludedRoute:
    feature: str
    classification: str
    reason: str
    row: int | None = None
    logical_m: int | None = None


@dataclass(frozen=True)
class NativeMTPRouteDelta:
    control_features: frozenset[str]
    candidate_features: frozenset[str]
    candidate_feature: str
    added: frozenset[str]
    removed: frozenset[str]
    replacement: bool
    implicit_replaced_surface: str | None


_MTP_Q4_LINEAR_MODULES = (
    "fc",
    "layers.0.mlp.down_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.v_proj",
)
_MTP_NORMS = (
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.q_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def _mtp_block_surfaces(*, precision_islands: bool) -> tuple[str, ...]:
    surfaces = tuple(
        f"language_model.mtp.{module}.{tensor}"
        for module in _MTP_Q4_LINEAR_MODULES
        for tensor in ("weight", "scales", "biases")
    ) + tuple(f"language_model.mtp.{name}" for name in _MTP_NORMS)
    if precision_islands:
        surfaces += tuple(
            f"language_model.mtp.precision_islands.{projection}.{tensor}"
            for projection in ("q", "k", "v")
            for tensor in ("weight", "indices")
        )
    return surfaces


_BLOCK_FEATURES = frozenset(
    {"r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands"}
)
_INPUT_FEATURES = frozenset(
    {"r61_dual_norm_concat", "r63_q8_embedding_dual_norm"}
)
MUTUALLY_EXCLUSIVE_CANDIDATE_SETS = (_BLOCK_FEATURES, _INPUT_FEATURES)

_STOCK_HISTORY_ACTIVE_WHEN = (
    "stock_history selected: prompt_tokens<16384 or row20 not installed"
)
_STOCK_HISTORY_BYPASSED_BY = (
    "r20_kv_only_history selected at prompt_tokens>=16384"
)


def _decode_regime(phase: str, shape: str) -> CallsiteRegime:
    return CallsiteRegime(
        route="native_proposal",
        phase=phase,
        logical_m=(1,),
        dynamic_logical_m=None,
        native_depths=(1, 2, 3),
        callsite_shape=shape,
        active_when="native proposal route D1-D3",
    )


def _stock_history_regime(shape: str) -> CallsiteRegime:
    return CallsiteRegime(
        route="stock_history",
        phase="mtp_history/prefill",
        logical_m=(),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(),
        callsite_shape=shape,
        active_when=_STOCK_HISTORY_ACTIVE_WHEN,
        bypassed_by=_STOCK_HISTORY_BYPASSED_BY,
    )


_BLOCK_COMMON_ROW20_SURFACES = (
    "language_model.mtp.pre_fc_norm_embedding.weight",
    "language_model.mtp.pre_fc_norm_hidden.weight",
    "language_model.mtp.fc.weight",
    "language_model.mtp.fc.scales",
    "language_model.mtp.fc.biases",
    "language_model.mtp.layers.0.input_layernorm.weight",
    "language_model.mtp.layers.0.self_attn.k_norm.weight",
)
_BLOCK_ROW20_READ_DEPENDENCIES = (
    "language_model.model.embed_tokens",
    "language_model.mtp.pre_fc_norm_embedding",
    "language_model.mtp.pre_fc_norm_hidden",
    "language_model.mtp.fc",
    "language_model.mtp.layers.0.input_layernorm",
    "language_model.mtp.layers.0.self_attn.k_proj._mtplx_pack_linear",
    "language_model.mtp.layers.0.self_attn.v_proj._mtplx_pack_linear",
    "language_model.mtp.layers.0.self_attn.k_norm",
    "language_model.mtp.layers.0.self_attn.rope",
)
_ROW61_OWNED_SURFACES = (
    "language_model._mtplx_prepare_mtp_inputs_dual route binding",
    "mtplx.qwen38_challenge_kernels.qwen38_dual_rms_norm_concat",
)
_ROW61_UNCHANGED_DEPENDENCIES = (
    "language_model.model.embed_tokens",
    "language_model.mtp.pre_fc_norm_embedding.weight",
    "language_model.mtp.pre_fc_norm_hidden.weight",
)
_ROW63_OWNED_SURFACES = (
    "language_model._mtplx_prepare_mtp_inputs_row63 route binding",
    "mtplx.qwen38_challenge_kernels.qwen38_q8_embedding_dual_rms_norm_concat",
)
_ROW63_UNCHANGED_DEPENDENCIES = (
    "language_model.model.embed_tokens Q8/group-64 weights/scales/biases",
    "language_model.mtp.pre_fc_norm_embedding.weight",
    "language_model.mtp.pre_fc_norm_hidden.weight",
)


def _row20_block_regime(
    owned_surfaces: tuple[str, ...],
    *,
    precision_islands: bool,
) -> CallsiteRegime:
    if precision_islands:
        projection_surfaces = (
            "language_model.mtp.layers.0.self_attn.k_proj._mtplx_pack_linear.weight <- precision_islands.k.weight sorted by precision_islands.k.indices",
            "language_model.mtp.layers.0.self_attn.v_proj._mtplx_pack_linear.weight <- precision_islands.v.weight sorted by precision_islands.v.indices",
        )
    else:
        projection_surfaces = tuple(
            f"language_model.mtp.layers.0.self_attn.{projection}.{tensor}"
            for projection in ("k_proj", "v_proj")
            for tensor in ("weight", "scales", "biases")
        )
    active = _BLOCK_COMMON_ROW20_SURFACES + projection_surfaces
    return CallsiteRegime(
        route="row20_kv_only_history",
        phase="mtp_history/prefill",
        logical_m=(),
        dynamic_logical_m="row20 K/V-only chunk length L>=1",
        native_depths=(),
        callsite_shape="B=1 L>=1 row20 K/V-only history append",
        active_when="row20 installed and prompt_tokens>=16384",
        active_candidate_surfaces=active,
        bypassed_candidate_surfaces=tuple(
            surface for surface in owned_surfaces if surface not in active
        ),
        read_dependencies=_BLOCK_ROW20_READ_DEPENDENCIES,
    )


def _row20_input_bypass_regime(
    owned_surfaces: tuple[str, ...],
    unchanged_dependencies: tuple[str, ...],
) -> CallsiteRegime:
    return CallsiteRegime(
        route="row20_kv_only_history",
        phase="mtp_history/prefill",
        logical_m=(),
        dynamic_logical_m="row20 K/V-only chunk length L>=1",
        native_depths=(),
        callsite_shape="B=1 L>=1 row20 direct embedding/norm/fc path",
        active_when="row20 installed and prompt_tokens>=16384",
        active_candidate_surfaces=(),
        bypassed_candidate_surfaces=owned_surfaces,
        read_dependencies=unchanged_dependencies,
    )


_R17_BLOCK_SURFACES = _mtp_block_surfaces(precision_islands=False)
_R36_BLOCK_SURFACES = _mtp_block_surfaces(precision_islands=True) + (
    "language_model.mtp.layers.0.self_attn.k_proj._mtplx_pack_linear.weight <- precision_islands.k.weight sorted by precision_islands.k.indices",
    "language_model.mtp.layers.0.self_attn.v_proj._mtplx_pack_linear.weight <- precision_islands.v.weight sorted by precision_islands.v.indices",
)

_CANDIDATES = {
    "r08_device_draft": NativeMTPCandidate(
        row=8,
        feature="r08_device_draft",
        pr_number=41,
        source_commit="11670086c1b9c3bd2d3d0323f9f6c346b65770f6",
        phase="mtp_decode/proposal",
        ownership="code",
        owned_surfaces=(
            "mtplx.generation._device_draft_route_depths",
            "mtplx.generation._make_device_draft_core",
            "request-bound D1/D2/D3 compiled native proposal callables",
        ),
        logical_m=(1,),
        dynamic_logical_m=None,
        native_depths=(1, 2, 3),
        callsite_shape="B=1 sequential proposal head calls: hidden [1,1,5120], token_ids [1,1]",
        callsite_regimes=(
            _decode_regime(
                "mtp_decode/proposal",
                "B=1 hidden [1,1,5120], token_ids [1,1]",
            ),
        ),
        dtypes=("bfloat16", "int32"),
        quant_bits=None,
        group_size=None,
        packing=None,
    ),
    "r10_compact_vocab": NativeMTPCandidate(
        row=10,
        feature="r10_compact_vocab",
        pr_number=59,
        source_commit="61936f26547df47fcdf48957115a2b8e9c4d1a37",
        phase="mtp_decode/head",
        ownership="code",
        owned_surfaces=(
            "language_model._mtplx_draft_lm_head",
            "language_model._mtplx_draft_token_id_map",
            "draft Q4 output rows [0:98304] + [248044:248070]",
        ),
        logical_m=(1,),
        dynamic_logical_m=None,
        native_depths=(1, 2, 3),
        callsite_shape="[1,1,5120] -> 98,330 reachable logits padded to 98,336 rows",
        callsite_regimes=(
            _decode_regime(
                "mtp_decode/head",
                "B=1 hidden [1,1,5120] -> 98,330 reachable proposal logits",
            ),
        ),
        dtypes=("uint32", "bfloat16", "int32"),
        quant_bits=4,
        group_size=64,
        packing=QWEN38_AFFINE_PACKING,
        parent_rule=ParentRule(
            required_control_features=frozenset({"r08_device_draft"}),
        ),
    ),
    "r11_position_ema": NativeMTPCandidate(
        row=11,
        feature="r11_position_ema",
        pr_number=63,
        source_commit="62174dbbca88380f4f9a7199abd0397d188478c6",
        phase="mtp_decode/adaptive",
        ownership="code",
        owned_surfaces=(
            "mtplx.adaptive.PositionEMADepthPolicy",
            "request-local native D0/D1/D2/D3 route selection",
            "deferred committed-history backlog for D0",
        ),
        logical_m=(1, 2, 3, 4),
        dynamic_logical_m=None,
        native_depths=(0, 1, 2, 3),
        callsite_shape="D0 target-only or D1-D3 proposals; target verify logical M=depth+1 <=4",
        callsite_regimes=(
            CallsiteRegime(
                route="adaptive_native",
                phase="mtp_decode/adaptive",
                logical_m=(1, 2, 3, 4),
                dynamic_logical_m=None,
                native_depths=(0, 1, 2, 3),
                callsite_shape="D0 target-only or D1-D3 proposals; target verify M=depth+1 <=4",
                active_when="position_ema policy selected for the request",
            ),
        ),
        dtypes=(),
        quant_bits=None,
        group_size=None,
        packing=None,
    ),
    "r17_q4_mtp_block": NativeMTPCandidate(
        row=17,
        feature="r17_q4_mtp_block",
        pr_number=126,
        source_commit="deb63ad0d1701d9d14cacd34d901ae7c0588c432",
        phase="mtp_history+decode/block",
        ownership="artifact",
        owned_surfaces=_R17_BLOCK_SURFACES,
        logical_m=(1,),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(1, 2, 3),
        callsite_shape=(
            "complete one-layer MTP block: proposal B=1 L=1 [1,1,10240]; "
            "stock history B=1 L>=1 [1,L,10240]"
        ),
        callsite_regimes=(
            _decode_regime("mtp_decode", "B=1 L=1 proposal input [1,1,10240]"),
            _stock_history_regime("B=1 L>=1 stock-history input [1,L,10240]"),
            _row20_block_regime(
                _R17_BLOCK_SURFACES,
                precision_islands=False,
            ),
        ),
        dtypes=("uint32", "bfloat16"),
        quant_bits=4,
        group_size=64,
        packing=QWEN38_AFFINE_PACKING,
        parent_rule=ParentRule(
            replacement_only=True,
            implicit_replaces="stock_mtp_block",
        ),
        incompatible=_BLOCK_FEATURES - {"r17_q4_mtp_block"},
        artifact_variant="r17",
        artifact_manifest_sha256="cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d",
        artifact_file_sha256="0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815",
        artifact_bytes=238_934_093,
    ),
    "r20_kv_only_history": NativeMTPCandidate(
        row=20,
        feature="r20_kv_only_history",
        pr_number=180,
        source_commit="cf350293feb435f17a110e5ab5324dd893fd523c",
        phase="mtp_history/prefill",
        ownership="code",
        owned_surfaces=(
            "language_model._mtplx_qwen38_kv_only_history_impl route binding",
            "prebound concatenated K/V buffers derived from installed projections",
            "mtplx.mtp_patch.install_qwen38_kv_only_history_append",
        ),
        logical_m=(),
        dynamic_logical_m="prefill chunk length L>=1",
        native_depths=(),
        callsite_shape=(
            "B=1, L=len(token_ids), hidden/input [1,L,5120], K/V [1,4,L,256]; "
            "prompt_tokens<16384 selects stock_history"
        ),
        callsite_regimes=(
            CallsiteRegime(
                route="row20_kv_only_history",
                phase="mtp_history/prefill",
                logical_m=(),
                dynamic_logical_m="prefill chunk length L>=1",
                native_depths=(),
                callsite_shape="B=1 L>=1 hidden/input [1,L,5120], K/V [1,4,L,256]",
                active_when=(
                    "row20 installed and prompt_tokens>=16384 selects kv_only_history"
                ),
            ),
        ),
        dtypes=("uint32", "bfloat16"),
        quant_bits=None,
        group_size=None,
        packing=None,
        unchanged_dependencies=_BLOCK_ROW20_READ_DEPENDENCIES,
        min_context_tokens=16_384,
        bypassed_candidate_callable_features=_INPUT_FEATURES,
        partially_used_artifact_features=_BLOCK_FEATURES,
    ),
    "r28_q4_mtp_block": NativeMTPCandidate(
        row=28,
        feature="r28_q4_mtp_block",
        pr_number=304,
        source_commit="6209702fba83a744eb3deb598905d59978f9e5e7",
        phase="mtp_history+decode/block",
        ownership="artifact",
        owned_surfaces=_R17_BLOCK_SURFACES,
        logical_m=(1,),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(1, 2, 3),
        callsite_shape=(
            "alternate complete one-layer MTP block: proposal B=1 L=1 [1,1,10240]; "
            "stock history B=1 L>=1 [1,L,10240]"
        ),
        callsite_regimes=(
            _decode_regime("mtp_decode", "B=1 L=1 proposal input [1,1,10240]"),
            _stock_history_regime("B=1 L>=1 stock-history input [1,L,10240]"),
            _row20_block_regime(
                _R17_BLOCK_SURFACES,
                precision_islands=False,
            ),
        ),
        dtypes=("uint32", "bfloat16"),
        quant_bits=4,
        group_size=64,
        packing=QWEN38_AFFINE_PACKING,
        parent_rule=ParentRule(
            replacement_only=True,
            replaces=frozenset({"r17_q4_mtp_block"}),
            implicit_replaces="stock_mtp_block",
        ),
        incompatible=_BLOCK_FEATURES - {"r28_q4_mtp_block"},
        artifact_variant="r28",
        artifact_manifest_sha256="7d62702795865b9036afe4bddcd16a2a8eb973c0caced15e5243139dda067f47",
        artifact_file_sha256="c934b40f1254858425cc0b5fdfe62b6ae13d1a4aff74da9d81606e92fdcf41ee",
        artifact_bytes=238_934_129,
    ),
    "r36_qkv_islands": NativeMTPCandidate(
        row=36,
        feature="r36_qkv_islands",
        pr_number=423,
        source_commit="ed4dfd6b0e95bb1cafb26c694bc247f551d550fe",
        phase="mtp_history+decode/block",
        ownership="artifact",
        owned_surfaces=_R36_BLOCK_SURFACES,
        logical_m=(1,),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(1, 2, 3),
        callsite_shape=(
            "complete Q4 MTP block plus 1,024-row BF16 Q/K/V corrections: "
            "proposal B=1 L=1; stock history B=1 L>=1"
        ),
        callsite_regimes=(
            _decode_regime("mtp_decode", "B=1 L=1 proposal input [1,1,10240]"),
            _stock_history_regime("B=1 L>=1 stock-history input [1,L,10240]"),
            _row20_block_regime(
                _R36_BLOCK_SURFACES,
                precision_islands=True,
            ),
        ),
        dtypes=("uint32", "bfloat16", "int32"),
        quant_bits=4,
        group_size=64,
        packing=QWEN38_AFFINE_PACKING,
        parent_rule=ParentRule(
            replacement_only=True,
            replaces=frozenset(
                {"r17_q4_mtp_block", "r28_q4_mtp_block"}
            ),
            implicit_replaces="stock_mtp_block",
        ),
        incompatible=_BLOCK_FEATURES - {"r36_qkv_islands"},
        artifact_variant="r36",
        artifact_manifest_sha256="477ba7266c6f726fafca7f7646e894fd962fa1a93c7672e04282f6243163549a",
        artifact_file_sha256="517bb133d7ca6e228a5129710b3cb2c25aa9944753b9f9a225fa1e8135df5e65",
        artifact_bytes=270_404_624,
    ),
    "r61_dual_norm_concat": NativeMTPCandidate(
        row=61,
        feature="r61_dual_norm_concat",
        pr_number=866,
        source_commit="8b54ff11c6d686628f6534d7127a261115782757",
        phase="mtp_history+decode/input",
        ownership="code",
        owned_surfaces=_ROW61_OWNED_SURFACES,
        logical_m=(1,),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(1, 2, 3),
        callsite_shape=(
            "two BF16 inputs: proposal [1,1,5120] -> [1,1,10240]; "
            "stock history [1,L,5120] -> [1,L,10240]"
        ),
        callsite_regimes=(
            _decode_regime(
                "mtp_decode", "two BF16 proposal inputs [1,1,5120] -> [1,1,10240]"
            ),
            _stock_history_regime(
                "B=1 L>=1 two BF16 stock-history inputs [1,L,5120] -> [1,L,10240]"
            ),
            _row20_input_bypass_regime(
                _ROW61_OWNED_SURFACES,
                _ROW61_UNCHANGED_DEPENDENCIES,
            ),
        ),
        dtypes=("bfloat16",),
        quant_bits=None,
        group_size=None,
        packing=None,
        unchanged_dependencies=_ROW61_UNCHANGED_DEPENDENCIES,
        incompatible=_INPUT_FEATURES - {"r61_dual_norm_concat"},
    ),
    "r63_q8_embedding_dual_norm": NativeMTPCandidate(
        row=63,
        feature="r63_q8_embedding_dual_norm",
        pr_number=911,
        source_commit="61612aa89dc65ecff2a7baf92940f6e8f36af4a8",
        phase="mtp_history+decode/input",
        ownership="code",
        owned_surfaces=_ROW63_OWNED_SURFACES,
        logical_m=(1,),
        dynamic_logical_m="stock-history chunk length L>=1",
        native_depths=(1, 2, 3),
        callsite_shape=(
            "Q8 embedding + BF16 hidden: proposal token_ids [1,1] -> [1,1,10240]; "
            "stock history token_ids [1,L] -> [1,L,10240]"
        ),
        callsite_regimes=(
            _decode_regime(
                "mtp_decode", "Q8 embedding token_ids [1,1] + BF16 hidden [1,1,5120]"
            ),
            _stock_history_regime(
                "B=1 L>=1 Q8 embedding token_ids [1,L] + BF16 hidden [1,L,5120]"
            ),
            _row20_input_bypass_regime(
                _ROW63_OWNED_SURFACES,
                _ROW63_UNCHANGED_DEPENDENCIES,
            ),
        ),
        dtypes=("uint32", "bfloat16", "int32"),
        quant_bits=8,
        group_size=64,
        packing=QWEN38_AFFINE_PACKING,
        parent_rule=ParentRule(
            replacement_only=True,
            replaces=frozenset({"r61_dual_norm_concat"}),
        ),
        unchanged_dependencies=_ROW63_UNCHANGED_DEPENDENCIES,
        incompatible=_INPUT_FEATURES - {"r63_q8_embedding_dual_norm"},
    ),
}

NATIVE_MTP_CANDIDATES: Mapping[str, NativeMTPCandidate] = MappingProxyType(
    _CANDIDATES
)
CANDIDATE_ROWS = frozenset(spec.row for spec in NATIVE_MTP_CANDIDATES.values())
del _CANDIDATES

FROZEN_TARGET_SUBSTRATE: Mapping[int, FrozenSubstrate] = MappingProxyType(
    {
        18: FrozenSubstrate(18, "r18_gdn_decay_memo", 135, "b6ce964b16bbb7836480a29e9f5e436bb99a35dd", "target/general", "48 linear-attention GDN decay memos"),
        21: FrozenSubstrate(21, "r21_qk_rms_rope", 186, "4eb54489fb518b2040aa31de5e7344ed908a81d6", "target/general", "16 target full-attention Q/K RMSNorm plus RoPE modules"),
        24: FrozenSubstrate(24, "r24_eval_ladder", 234, "7351e62674bc600f0ca148d3a1b0604716a09db6", "target/general", "target layer evaluation ladder and Q/K L<=16 bound"),
        26: FrozenSubstrate(26, "r26_prefill_ladder_3", 276, "033f622755ac407c31b75f613ad590dbc0976b96", "target/general", "target prefill evaluation cadence and Q/K L<=32 bound"),
        48: FrozenSubstrate(48, "r48_boundary_fused", 543, "86fb1f020fc1fddc7e55aceac4761e5054b71dd6", "target/general", "64-layer target residual/RMSNorm boundary capture"),
        50: FrozenSubstrate(50, "r50_wired_residency", 572, "c0e34afd857e9a3db5f9d3eb2430ddb969bfa97b", "target/general", "process active-footprint wired limit"),
        53: FrozenSubstrate(53, "r53_command_buffers", 600, "0c90733d383f6b987a29682bf9eb9458a6172bfa", "target/general", "process-latched 512 MiB and 50-op command-buffer profile"),
    }
)
_FROZEN_BY_FEATURE = MappingProxyType(
    {spec.feature: spec for spec in FROZEN_TARGET_SUBSTRATE.values()}
)

UNREACHABLE_NATIVE_ROUTES: Mapping[str, ExcludedRoute] = MappingProxyType(
    {
        "native_d4_plus": ExcludedRoute("native_d4_plus", "unreachable", "native manifest mtp_depth_max=3; D4 starts at verify logical M5", logical_m=5),
        "dflash_m5": ExcludedRoute("dflash_m5", "unreachable", "DFlash M5 has no native D3 call site; native verify is M4", logical_m=5),
        "dflash_m6": ExcludedRoute("dflash_m6", "unreachable", "DFlash M6 has no native D3 call site; native verify is M4", logical_m=6),
        "dflash_m7": ExcludedRoute("dflash_m7", "unreachable", "DFlash M7 has no native D3 call site; native verify is M4", logical_m=7),
        "dflash_m8": ExcludedRoute("dflash_m8", "unreachable", "DFlash M8 has no native D3 call site; native verify is M4", logical_m=8),
    }
)

ARGMAX_ONLY_INELIGIBLE: Mapping[str, ExcludedRoute] = MappingProxyType(
    {
        f"r{row:02d}_argmax_shortlist": ExcludedRoute(
            f"r{row:02d}_argmax_shortlist",
            "correctness-ineligible",
            "temperature1/top-p.95/top-k20 requires full proposal distribution",
            row=row,
        )
        for row in (19, 42, 47, 67, 69)
    }
)

REMOVED_FAMILIES: Mapping[str, ExcludedRoute] = MappingProxyType(
    {
        "r70_qmv_sumtable": ExcludedRoute("r70_qmv_sumtable", "removed", "fixed-D3 lazy-history QMV family was rejected as incompatible", row=70),
        "r78_qmv_active_groups": ExcludedRoute("r78_qmv_active_groups", "removed", "depends on removed row-70 QMV family", row=78),
        "r80_qmv_m2": ExcludedRoute("r80_qmv_m2", "removed", "depends on removed row-70/78 QMV family", row=80),
        "source_proposal": ExcludedRoute("source_proposal", "removed", "superseded source-proposal branch is not a native route candidate"),
    }
)

_ALIASES = MappingProxyType(
    {
        "kv_only_history": "r20_kv_only_history",
        "dual_norm": "r61_dual_norm_concat",
    }
)


def _reject_incompatible(features: frozenset[str]) -> None:
    for feature in features & NATIVE_MTP_CANDIDATES.keys():
        conflicts = NATIVE_MTP_CANDIDATES[feature].incompatible & features
        if conflicts:
            raise NativeMTPRouteError(
                "incompatible native-MTP alternatives: "
                f"{feature} and {sorted(conflicts)[0]}"
            )


def canonicalize_native_mtp_route(route_id: str) -> frozenset[str]:
    """Parse one route ID and return its unique canonical feature set."""

    raw_tokens = str(route_id).split("+")
    if not raw_tokens or any(not token for token in raw_tokens):
        raise NativeMTPRouteError("route is empty or contains an empty feature")
    if len(raw_tokens) != len(set(raw_tokens)):
        raise NativeMTPRouteError("duplicate raw route feature")
    if "control" in raw_tokens:
        if len(raw_tokens) != 1:
            raise NativeMTPRouteError("control cannot be combined with route features")
        return frozenset({"control"})

    canonical_tokens = [_ALIASES.get(feature, feature) for feature in raw_tokens]
    if len(canonical_tokens) != len(set(canonical_tokens)):
        raise NativeMTPRouteError("duplicate canonical route feature")

    for feature in canonical_tokens:
        if feature in UNREACHABLE_NATIVE_ROUTES:
            raise NativeMTPRouteError(f"unreachable native-MTP route: {feature}")
        if feature in ARGMAX_ONLY_INELIGIBLE:
            raise NativeMTPRouteError(f"correctness-ineligible argmax route: {feature}")
        if feature in REMOVED_FAMILIES:
            raise NativeMTPRouteError(f"removed native-MTP family: {feature}")
        if feature not in NATIVE_MTP_CANDIDATES and feature not in _FROZEN_BY_FEATURE:
            raise NativeMTPRouteError(f"unknown route feature: {feature}")
    canonical = frozenset(canonical_tokens)
    _reject_incompatible(canonical)
    return canonical


def validate_native_mtp_route_delta(
    control_route: str,
    candidate_route: str,
    *,
    allow_frozen_candidate: bool = False,
) -> NativeMTPRouteDelta:
    """Require one eligible native-MTP addition or one explicit replacement.

    Frozen target/general features may be present, but their set must be byte-
    for-byte route-identical across the two arms.  This function is intended
    for bracket construction; it is never called from measured generation.
    """

    control = canonicalize_native_mtp_route(control_route) - {"control"}
    candidate = canonicalize_native_mtp_route(candidate_route) - {"control"}

    control_frozen = control & _FROZEN_BY_FEATURE.keys()
    candidate_frozen = candidate & _FROZEN_BY_FEATURE.keys()
    if control_frozen != candidate_frozen and not allow_frozen_candidate:
        raise NativeMTPRouteError(
            "frozen substrate must be identical between control and candidate"
        )

    added = frozenset(candidate - control)
    removed = frozenset(control - candidate)
    if len(added) != 1:
        raise NativeMTPRouteError(
            "control/candidate routes must differ by exactly one native-MTP candidate"
        )
    candidate_feature = next(iter(added))
    if candidate_feature in _FROZEN_BY_FEATURE:
        if not allow_frozen_candidate:
            raise NativeMTPRouteError(
                "candidate delta is not an eligible native-MTP surface: "
                f"{candidate_feature}"
            )
        if removed:
            raise NativeMTPRouteError(
                "frozen optimized feature must be an isolated addition"
            )
        return NativeMTPRouteDelta(
            control_features=control,
            candidate_features=candidate,
            candidate_feature=candidate_feature,
            added=added,
            removed=removed,
            replacement=False,
            implicit_replaced_surface=None,
        )
    if candidate_feature not in NATIVE_MTP_CANDIDATES:
        raise NativeMTPRouteError(
            f"candidate delta is not an eligible native-MTP surface: {candidate_feature}"
        )

    candidate_spec = NATIVE_MTP_CANDIDATES[candidate_feature]
    parent_rule = candidate_spec.parent_rule
    missing_parent = parent_rule.required_control_features - control
    if missing_parent:
        raise NativeMTPRouteError(
            f"{candidate_feature} requires control feature {sorted(missing_parent)[0]}"
        )

    replacement = bool(removed)
    implicit_replaced_surface: str | None = None
    if parent_rule.replacement_only:
        if not replacement:
            if parent_rule.implicit_replaces is None:
                raise NativeMTPRouteError(
                    f"{candidate_feature} is replacement-only"
                )
            replacement = True
            implicit_replaced_surface = parent_rule.implicit_replaces
        else:
            if len(removed) != 1:
                raise NativeMTPRouteError(
                    "explicit replacement must remove exactly one alternative"
                )
            removed_feature = next(iter(removed))
            if removed_feature not in parent_rule.replaces:
                raise NativeMTPRouteError(
                    f"{candidate_feature} must replace one of {sorted(parent_rule.replaces)}"
                )
    elif replacement:
        if len(removed) != 1:
            raise NativeMTPRouteError(
                "explicit replacement must remove exactly one alternative"
            )
        removed_feature = next(iter(removed))
        if removed_feature not in candidate_spec.incompatible:
            raise NativeMTPRouteError(
                "arbitrary route difference is not an explicit native-MTP replacement"
            )
        raise NativeMTPRouteError(
            f"{candidate_feature} is addition-only and cannot replace {removed_feature}"
        )

    return NativeMTPRouteDelta(
        control_features=control,
        candidate_features=candidate,
        candidate_feature=candidate_feature,
        added=added,
        removed=removed,
        replacement=replacement,
        implicit_replaced_surface=implicit_replaced_surface,
    )


__all__ = (
    "ARGMAX_ONLY_INELIGIBLE",
    "CANDIDATE_ROWS",
    "CallsiteRegime",
    "FROZEN_TARGET_SUBSTRATE",
    "MUTUALLY_EXCLUSIVE_CANDIDATE_SETS",
    "NATIVE_D3_VERIFY_LOGICAL_M",
    "NATIVE_MTP_CANDIDATES",
    "NATIVE_MTP_DEPTH_MAX",
    "NATIVE_SAMPLER",
    "ParentRule",
    "REMOVED_FAMILIES",
    "UNREACHABLE_NATIVE_ROUTES",
    "NativeMTPCandidate",
    "NativeMTPRouteDelta",
    "NativeMTPRouteError",
    "canonicalize_native_mtp_route",
    "validate_native_mtp_route_delta",
)
