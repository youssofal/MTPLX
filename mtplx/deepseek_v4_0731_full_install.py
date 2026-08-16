"""Atomic construction installer for the measured Flash-0731 target stack.

There is one supported configuration: packed affine-Q2/group-128 routed
gate/up, row-owned M1 reduction, compiled packed-Q2 physical M3 tails, fused
WQB-qhead, and fused WOB. Pinned config/index metadata, loaded ownership,
storage, and every 43-layer self-check complete before any route is published.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from . import deepseek_v4_attention_island as AI
from .deepseek_v4_0731_m3_target import (
    build_0731_m3_target_route,
    build_m3_compiled_tail_layer,
)
from .deepseek_v4_0731_moe import (
    build_routed_q2_pair,
    build_row_owned_combine_m1,
    exact_selfcheck_row_owned_combine_m1,
    validate_routed_q2_pair,
)


EXPECTED_FULL_CONFIG_SHA256 = (
    "44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f"
)
EXPECTED_FULL_INDEX_SHA256 = (
    "f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861"
)
RECORDED_ARTIFACT_LABEL = "mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed"
EXPECTED_SOURCE_REVISION = "10001e0065f8394e03e968e652cbbe7cd2ca122c"

_EXPECTED_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "swiglu_limit": 10.0,
    "num_nextn_predict_layers": 1,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}
_TARGET_LAYER_IDS = (40, 41, 42)
_LAYERS = 43
_STAGE_COUNT = 3
_M3_WQB_SHAPE = (1, 3, 1024)
_M3_WOB_SHAPE = (1, 3, 8192)


@dataclass(frozen=True, slots=True)
class Full0731DSparkContract:
    layers: int
    target_layer_ids: tuple[int, int, int]
    stage_count: int
    source_revision: str
    config_sha256: str
    index_sha256: str


def _contract_error(detail: str) -> ValueError:
    return ValueError(f"DeepSeek-V4 0731 full DSpark contract failed: {detail}")


def _metadata_revision(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise _contract_error(f"artifact metadata is unreadable: {exc}") from exc


def validate_full_0731_dspark_artifact(
    model_path: str | Path,
    config: dict[str, Any],
) -> Full0731DSparkContract:
    """Require the exact measured config/index and recorded metadata revision."""

    root = Path(model_path)
    try:
        config_bytes = (root / "config.json").read_bytes()
        index_bytes = (root / "model.safetensors.index.json").read_bytes()
    except OSError as exc:
        raise _contract_error(f"artifact identity is unreadable: {exc}") from exc
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    if config_sha != EXPECTED_FULL_CONFIG_SHA256:
        raise _contract_error("config SHA-256 does not match the pinned artifact")
    if index_sha != EXPECTED_FULL_INDEX_SHA256:
        raise _contract_error("index SHA-256 does not match the pinned artifact")
    for name, expected in _EXPECTED_CONFIG.items():
        if config.get(name) != expected:
            raise _contract_error(
                f"config {name}={config.get(name)!r}, expected {expected!r}"
            )
    metadata_root = root / ".cache/huggingface/download"
    revisions = (
        _metadata_revision(metadata_root / "config.json.metadata"),
        _metadata_revision(metadata_root / "model.safetensors.index.json.metadata"),
    )
    if revisions != (EXPECTED_SOURCE_REVISION, EXPECTED_SOURCE_REVISION):
        raise _contract_error("recorded metadata revision does not match")
    return Full0731DSparkContract(
        layers=_LAYERS,
        target_layer_ids=_TARGET_LAYER_IDS,
        stage_count=_STAGE_COUNT,
        source_revision=EXPECTED_SOURCE_REVISION,
        config_sha256=config_sha,
        index_sha256=index_sha,
    )


def _validate_loaded_dspark_owner(
    model: Any,
    contract: Full0731DSparkContract,
) -> tuple[tuple[Any, ...], Callable]:
    try:
        layers = tuple(model.model.layers)
        dspark = model._dspark
        stages = tuple(dspark.stages)
        published_stages = tuple(getattr(model.mtp, "layers", model.mtp))
        native_route = model._target_hidden_route
    except AttributeError as exc:
        raise _contract_error(
            f"loaded model lacks native DSpark ownership: {exc}"
        ) from exc
    if len(layers) != contract.layers:
        raise _contract_error(f"loaded layer count is {len(layers)}, expected 43")
    if dspark is None:
        raise _contract_error("loaded model does not own native DSpark stages")
    if tuple(getattr(dspark, "target_layer_ids", ())) != contract.target_layer_ids:
        raise _contract_error("loaded DSpark target taps are not (40, 41, 42)")
    if len(stages) != contract.stage_count:
        raise _contract_error("loaded DSpark does not own exactly three stages")
    if len(published_stages) != len(stages) or any(
        published is not owned
        for published, owned in zip(published_stages, stages, strict=True)
    ):
        raise _contract_error("DSpark stage ownership is not preserved by model.mtp")
    if not callable(native_route):
        raise _contract_error("native DSpark target route is absent")
    return layers, native_route


class _BoundDSparkM1Body:
    """Prebound M1 trunk retaining post-layer taps 40, 41, and 42."""

    __slots__ = ("_body", "_layers", "_target_layer_ids")

    def __init__(self, body: Any, layers, target_layer_ids: tuple[int, int, int]):
        self._body = body
        self._layers = tuple(layers)
        self._target_layer_ids = target_layer_ids

    def __call__(self, input_ids: mx.array, cache=None):
        hidden = self._body.embed_tokens(input_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], self._body.hc_mult, hidden.shape[-1]),
        )
        entries = (None,) * len(self._layers) if cache is None else cache
        taps = []
        for layer_id, ((layer, tail), entry) in enumerate(zip(self._layers, entries)):
            residual = hidden
            attention_in, post, comb = layer.attn_hc.pre(hidden)
            attention_in = layer.attn_norm(attention_in)
            attention_out = layer.attn(attention_in, mask=None, cache=entry)
            hidden = tail(
                attention_out,
                residual,
                post,
                comb,
                input_ids,
            )
            if layer_id in self._target_layer_ids:
                taps.append(mx.mean(hidden, axis=2))
        return hidden, mx.concatenate(taps, axis=-1)


@dataclass(frozen=True, slots=True)
class _FullDSparkTargetRoute:
    native: Callable
    m1: Callable

    def __call__(self, owner: Any, input_ids: mx.array, cache=None):
        if int(input_ids.shape[1]) == 1:
            return self.m1(input_ids, cache)
        return self.native(owner, input_ids, cache)


def _require_wqb_receipt(receipt: Any) -> dict[str, Any]:
    published = tuple(getattr(receipt, "published_routes", ()))
    if (
        len(published) != _LAYERS
        or int(getattr(receipt, "q6_count", -1)) != _LAYERS
        or int(getattr(receipt, "exact_selfchecked", -1)) != _LAYERS
        or not callable(getattr(receipt, "publish", None))
        or not callable(getattr(receipt, "restore", None))
    ):
        raise _contract_error("fused WQB-qhead preparation is not 43/43 exact")
    return {
        "candidate": "official-wheel-custom-fixed-m3-wqb-qhead-fused",
        "layers_installed": _LAYERS,
        "q6_g128_layers": _LAYERS,
        "exact_selfchecked_layers": _LAYERS,
        "shape": [1, 3, 1024],
        "output_shape": [1, 3, 64, 512],
    }


def _require_wob_receipt(receipt: Any) -> dict[str, Any]:
    published = tuple(getattr(receipt, "published_routes", ()))
    if (
        len(published) != _LAYERS
        or int(getattr(receipt, "q6_count", -1)) != _LAYERS
        or int(getattr(receipt, "exact_selfchecked", -1)) != _LAYERS
        or int(getattr(receipt, "o_lora_sink_count", -1)) != _LAYERS
        or not callable(getattr(receipt, "publish", None))
        or not callable(getattr(receipt, "restore", None))
    ):
        raise _contract_error("fused WOB preparation is not 43/43 exact")
    return {
        "candidate": "official-wheel-custom-fixed-m3-affine-qmv",
        "layers_installed": _LAYERS,
        "q6_g128_layers": _LAYERS,
        "exact_selfchecked_layers": _LAYERS,
        "active_o_lora_sinks_installed": _LAYERS,
        "shape": [1, 3, 8192],
        "output_size": 4096,
    }


def _m3_wqb_qhead_exact_selfcheck():
    """Build the construction-only fused M3-versus-three-M1 q-head gate."""

    key = mx.random.key(73_103_1025)
    qr_key, phase_key = mx.random.split(key)
    qr = mx.random.normal(_M3_WQB_SHAPE, key=qr_key).astype(mx.bfloat16)
    phase = mx.random.normal((3, 32), key=phase_key)
    cos, sin = mx.cos(phase), mx.sin(phase)

    def check(stock: Callable, candidate: Callable, _layer_index: int) -> bool:
        actual = candidate(qr, cos, sin)
        expected = mx.concatenate(
            tuple(
                stock(qr[:, row : row + 1], cos[row : row + 1], sin[row : row + 1])
                for row in range(3)
            ),
            axis=1,
        )
        mx.eval(actual, expected)
        return bool(mx.array_equal(actual, expected))

    return check


def _m3_wob_exact_selfcheck():
    """Build the construction-only real-weight M3-versus-three-M1 gate."""

    key = mx.random.key(73_103_8192)
    value = mx.random.normal(_M3_WOB_SHAPE, key=key).astype(mx.bfloat16)

    def check(stock: Callable, candidate: Callable, _layer_index: int) -> bool:
        actual = candidate(value)
        expected = mx.concatenate(
            tuple(stock(value[:, row : row + 1]) for row in range(3)),
            axis=1,
        )
        mx.eval(actual, expected)
        return bool(mx.array_equal(actual, expected))

    return check


@dataclass(frozen=True, slots=True)
class PreparedFull0731DSparkTargetStack:
    """Fully checked target stack whose publication is one reversible step."""

    model: Any
    native_route: Callable
    target_route: Callable
    layers: tuple[Any, ...]
    stock_switches: tuple[Any, ...]
    replacement_switches: tuple[Any, ...]
    wqb: Any
    wob: Any
    receipt: dict[str, Any]

    def publish(self) -> None:
        """Publish every prepared route, rolling the whole option back on error."""

        try:
            for layer, replacement in zip(self.layers, self.replacement_switches):
                layer.ffn.switch_mlp = replacement
            self.wqb.publish()
            self.wob.publish()
            self.model._target_hidden_route = self.target_route
        except Exception as publication_error:
            try:
                self.restore()
            except Exception as restoration_error:
                publication_error.add_note(
                    f"target-stack rollback also raised: {restoration_error}"
                )
            raise

    def restore(self) -> None:
        """Restore in reverse publication order."""

        errors = []
        try:
            self.model._target_hidden_route = self.native_route
        except Exception as exc:
            errors.append(exc)
        try:
            self.wob.restore()
        except Exception as exc:
            errors.append(exc)
        try:
            self.wqb.restore()
        except Exception as exc:
            errors.append(exc)
        for layer, switch in zip(self.layers, self.stock_switches):
            try:
                layer.ffn.switch_mlp = switch
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("target-stack route restoration failed", errors)


def prepare_full_0731_dspark_compiled_tail_q2_pair(
    model: Any,
    config: dict[str, Any],
    model_path: str | Path,
    *,
    prepare_wqb_qhead: Callable[[tuple[Any, ...]], Any],
    prepare_wob: Callable[[tuple[Any, ...]], Any],
) -> PreparedFull0731DSparkTargetStack:
    """Build and check the measured target stack without publishing any route."""

    if not callable(prepare_wqb_qhead) or not callable(prepare_wob):
        raise _contract_error("both fused projection preparers are required")
    contract = validate_full_0731_dspark_artifact(model_path, config)
    layers, native_route = _validate_loaded_dspark_owner(model, contract)
    try:
        for layer in layers:
            switch = layer.ffn.switch_mlp
            validate_routed_q2_pair(
                switch.gate_proj,
                switch.up_proj,
                hidden_size=4096,
                width=2048,
                experts=256,
            )
    except AttributeError as exc:
        raise _contract_error(f"trunk MoE topology is incomplete: {exc}") from exc

    row_owned = build_row_owned_combine_m1(hidden_size=4096, top_k=6)
    exact_selfcheck_row_owned_combine_m1(row_owned)
    staged = []
    m3_layer_routes = []
    for layer in layers:
        original = layer.ffn.switch_mlp
        replacement = build_routed_q2_pair(
            original,
            hidden_size=4096,
            width=2048,
            experts=256,
        )
        m1_tail = AI._bind_attention_island_layer(
            layer,
            width=1,
            allowed_widths=(1,),
            shared_bits=8,
            routed_pair=True,
            routed_combine=row_owned,
            routed_switch=replacement,
        )
        m3_tail = AI._bind_attention_island_layer(
            layer,
            width=3,
            allowed_widths=(3,),
            shared_bits=8,
            routed_pair=True,
            routed_switch=replacement,
        )
        m3_layer = build_m3_compiled_tail_layer(layer, m3_tail)
        staged.append((layer, original, replacement, m1_tail))
        m3_layer_routes.append(m3_layer)

    m1_body = _BoundDSparkM1Body(
        model.model,
        tuple((layer, m1_tail) for layer, _, _, m1_tail in staged),
        contract.target_layer_ids,
    )
    m1_route = _FullDSparkTargetRoute(native=native_route, m1=m1_body)
    target_route = build_0731_m3_target_route(
        model,
        full_layer_routes=tuple(m3_layer_routes),
        base_route=m1_route,
    )

    # Projection preparation is also staging-only. Until both prepared objects
    # and their exact receipts are accepted, every model route remains stock.
    wqb_prepared = prepare_wqb_qhead(
        layers,
        exact_selfcheck=_m3_wqb_qhead_exact_selfcheck(),
    )
    wqb_receipt = _require_wqb_receipt(wqb_prepared)
    wob_prepared = prepare_wob(
        layers,
        exact_selfcheck=_m3_wob_exact_selfcheck(),
    )
    wob_receipt = _require_wob_receipt(wob_prepared)

    receipt = {
        "candidate": "mtplx-full-dspark-compiled-tail-packed-q2-pair-m1-m3",
        "artifact_label": RECORDED_ARTIFACT_LABEL,
        "validated_config_sha256": contract.config_sha256,
        "validated_index_sha256": contract.index_sha256,
        "validated_metadata_revision": contract.source_revision,
        "layers_installed": len(staged),
        "decode_m": 1,
        "fixed_k": 2,
        "physical_target_rows": 3,
        "m3_tail": "fixed-width3-compiled-tail",
        "m3_wqb": wqb_receipt,
        "m3_wob": wob_receipt,
        "row_owned_combine": True,
        "non_m1_m3_route": "native-dspark",
        "routed_bits": 2,
        "routed_group_size": 128,
        "routed_gate_up_paired": True,
        "shared_bits": 8,
        "target_taps": contract.target_layer_ids,
        "dspark_stages": contract.stage_count,
        "stage_ownership": "native",
    }
    return PreparedFull0731DSparkTargetStack(
        model=model,
        native_route=native_route,
        target_route=target_route,
        layers=layers,
        stock_switches=tuple(original for _, original, _, _ in staged),
        replacement_switches=tuple(replacement for _, _, replacement, _ in staged),
        wqb=wqb_prepared,
        wob=wob_prepared,
        receipt=receipt,
    )


def install_full_0731_dspark_compiled_tail_q2_pair(
    model: Any,
    config: dict[str, Any],
    model_path: str | Path,
    *,
    prepare_wqb_qhead: Callable[[tuple[Any, ...]], Any],
    prepare_wob: Callable[[tuple[Any, ...]], Any],
) -> dict[str, Any]:
    """Prepare and atomically publish the sole measured target stack."""

    prepared = prepare_full_0731_dspark_compiled_tail_q2_pair(
        model,
        config,
        model_path,
        prepare_wqb_qhead=prepare_wqb_qhead,
        prepare_wob=prepare_wob,
    )
    prepared.publish()
    return prepared.receipt
