"""GDN state-capture verify helpers for Qwen3.5/Qwen3.6."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# One-time warning latch for an explicitly requested but unavailable
# headquarter tape kernel (PR #209 review edit).
_HEADQUARTER_IMPORT_WARNED = False


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_GDN_POSTCONV_STATS: dict[str, Any] = {
    "enabled": False,
    "installed": False,
    "installation_status": "disabled",
    "installation_error": None,
    "gdn_layers": 0,
    "validated_contract": None,
    "implementation": "inline_g",
}
_A3B_GDN_POSTCONV_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention" for index in range(40)
)


class A3BGDNPostconvConfigError(RuntimeError):
    """The exact A3B GDN post-conv lane could not be installed."""


@dataclass(frozen=True)
class A3BGDNPostconvInstallPlan:
    """Externally validated A3B GDN ownership awaiting its self-check."""

    gdns: tuple[Any, ...]


@dataclass(frozen=True)
class A3BGDNPostconvFactory:
    """Selfchecked, order-stable callables for the exact M1/M2/M3 traces.

    ``m3_implementations`` is the k=2 (3-row) verify recurrence; it defaults to
    empty so K1-only construction paths are unchanged and is populated whenever
    the postconv is installed.
    """

    m1_implementations: tuple[Callable[..., Any], ...]
    m2_implementations: tuple[Callable[..., Any], ...]
    m3_implementations: tuple[Callable[..., Any], ...] = ()
    b8_t2_implementations: tuple[Callable[..., Any], ...] = ()
    # Native three-row cohort verify (B3/T2). The kernel source is byte-shared
    # with the B8/T2 launch; only the grid z extent (rows*Hv) and output batch
    # extent differ, and each row's arithmetic is independent of grid size.
    b3_t2_implementations: tuple[Callable[..., Any], ...] = ()


def _a3b_gdn_postconv_contract() -> dict[str, Any]:
    return {
        "batch": 1,
        "logical_m": [1, 2, 3],
        "routes": {
            "m1_correction": {
                "conv_shape": [1, 1, 8192],
                "gate_shapes": {"a": [1, 1, 32], "b": [1, 1, 32]},
                "output_shape": [1, 1, 32, 128],
                "captured_states_shape": [1, 1, 32, 128, 128],
            },
            "m2_verify": {
                "conv_shape": [1, 2, 8192],
                "gate_shapes": {"a": [1, 2, 32], "b": [1, 2, 32]},
                "output_shape": [1, 2, 32, 128],
                "captured_states_shape": [1, 2, 32, 128, 128],
            },
            "m3_verify": {
                "conv_shape": [1, 3, 8192],
                "gate_shapes": {"a": [1, 3, 32], "b": [1, 3, 32]},
                "output_shape": [1, 3, 32, 128],
                "captured_states_shape": [1, 3, 32, 128, 128],
            },
        },
        "state_shape": [1, 32, 128, 128],
        "input_dtype": "bfloat16",
        "state_dtype": "float32",
        "key_heads": 16,
        "value_heads": 32,
        "key_axis": 128,
        "value_axis": 128,
        "threadgroup": [32, 4, 1],
    }


def a3b_gdn_postconv_enabled() -> bool:
    return _env_enabled("MTPLX_FUSE_GDN_POST_CONV")


def _fail_a3b_gdn_postconv_configuration(message: str) -> None:
    _GDN_POSTCONV_STATS["installed"] = False
    _GDN_POSTCONV_STATS["installation_status"] = "configuration_error"
    _GDN_POSTCONV_STATS["installation_error"] = str(message)
    raise A3BGDNPostconvConfigError(message)


# Post-conv recurrence implementation selection.  ``inline_g`` (default) is the
# accepted TGY4 route; ``headquarter`` is the C1 redesigned-execution kernel.
_A3B_GDN_POSTCONV_IMPL_ENV = "MTPLX_A3B_GDN_POSTCONV_IMPL"
_A3B_GDN_POSTCONV_IMPL_DEFAULT = "inline_g"
_A3B_GDN_POSTCONV_IMPLS = ("inline_g", "headquarter")


def _a3b_gdn_postconv_impl_selection() -> str:
    """Resolve the requested post-conv implementation, fail-closed on unknown.

    Unset/empty selects the default ``inline_g`` route so the installed stack is
    byte-identical to the accepted baseline; any other value than the exact
    supported names hard-fails through the postconv configuration convention.
    """
    raw = os.environ.get(_A3B_GDN_POSTCONV_IMPL_ENV)
    value = (raw or "").strip().lower()
    if value == "":
        return _A3B_GDN_POSTCONV_IMPL_DEFAULT
    if value not in _A3B_GDN_POSTCONV_IMPLS:
        _fail_a3b_gdn_postconv_configuration(
            f"A3B GDN postconv {_A3B_GDN_POSTCONV_IMPL_ENV} must be one of "
            "'inline_g' or 'headquarter' (unset defaults to 'inline_g'); "
            f"got {raw!r}"
        )
    return value


def _a3b_gdn_postconv_headquarter_requested() -> bool:
    """Non-raising probe of whether the headquarter route is explicitly requested."""
    raw = os.environ.get(_A3B_GDN_POSTCONV_IMPL_ENV)
    return (raw or "").strip().lower() == "headquarter"


def _validate_a3b_quant_projection(
    gdn: Any,
    name: str,
    scales_shape: tuple[int, ...],
    layer_index: int,
) -> None:
    projection = getattr(gdn, name, None)
    scales = getattr(projection, "scales", None)
    if (
        int(getattr(projection, "bits", -1)) != 4
        or int(getattr(projection, "group_size", -1)) != 64
        or getattr(projection, "mode", None) != "affine"
        or tuple(getattr(scales, "shape", ())) != scales_shape
        or getattr(scales, "dtype", None) != mx.bfloat16
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv projection_quantization mismatch for "
            f"{name} at GDN layer {layer_index}"
        )


def prepare_a3b_gdn_postconv(
    model: Any,
    *,
    config: dict[str, Any],
) -> A3BGDNPostconvInstallPlan | None:
    """Validate checkpoint/model facts once for the exact A3B M1/M2 lanes."""
    _reset_gdn_postconv_stats_for_tests()
    if not a3b_gdn_postconv_enabled():
        return None
    _GDN_POSTCONV_STATS["enabled"] = True
    if not _env_enabled("MTPLX_COMPILED_TARGET_PREFIX"):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv compiled_target_prefix_flag must be enabled"
        )
    if _env_enabled("MTPLX_NATIVE_GDN_TAIL"):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology conflicts with MTPLX_NATIVE_GDN_TAIL"
        )

    text_config = config.get("text_config")
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or not isinstance(text_config, dict)
        or text_config.get("model_type") != "qwen3_5_moe_text"
        or int(text_config.get("hidden_size", -1)) != 2048
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires the exact A3B model"
        )
    if text_config.get("dtype") != "bfloat16":
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv config_dtype requires bfloat16"
        )

    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    layers = list(getattr(inner, "layers", ()) or ())
    if len(layers) != 40 or int(text_config.get("num_hidden_layers", -1)) != 40:
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv layer_count requires exactly 40 layers"
        )
    actual_linear = [bool(getattr(layer, "is_linear", False)) for layer in layers]
    configured_types = tuple(text_config.get("layer_types", ()))
    expected_linear = [
        kind == "linear_attention" for kind in _A3B_GDN_POSTCONV_LAYER_TYPES
    ]
    if (
        actual_linear != expected_linear
        or configured_types != _A3B_GDN_POSTCONV_LAYER_TYPES
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires exact 30-layer ownership"
        )
    gdns = [
        getattr(layer, "linear_attn", None)
        for layer, is_linear in zip(layers, actual_linear)
        if is_linear
    ]
    if len(gdns) != 30 or any(gdn is None for gdn in gdns):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires all 30 GDN modules"
        )

    config_geometry = {
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if (
        any(
            int(text_config.get(name, -1)) != expected
            for name, expected in config_geometry.items()
        )
        or float(text_config.get("rms_norm_eps", -1.0)) != 1e-6
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv head_geometry mismatch in model config"
        )

    for index, gdn in enumerate(gdns):
        if getattr(gdn, "sharding_group", None) is not None:
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv sharding is forbidden at GDN layer {index}"
            )
        if (
            int(getattr(gdn, "conv_dim", -1)) != 8192
            or int(getattr(gdn, "key_dim", -1)) != 2048
            or int(getattr(gdn, "conv_kernel_size", -1)) != 4
        ):
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv conv_geometry mismatch at GDN layer {index}"
            )
        if (
            int(getattr(gdn, "num_k_heads", -1)) != 16
            or int(getattr(gdn, "num_v_heads", -1)) != 32
            or int(getattr(gdn, "head_k_dim", -1)) != 128
            or int(getattr(gdn, "head_v_dim", -1)) != 128
        ):
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv head_geometry mismatch at GDN layer {index}"
            )
        parameters = (
            ("A_log", (32,)),
            ("dt_bias", (32,)),
            ("conv1d.weight", (8192, 4, 1)),
        )
        for parameter_name, expected_shape in parameters:
            node = gdn
            for part in parameter_name.split("."):
                node = getattr(node, part, None)
            if tuple(getattr(node, "shape", ())) != expected_shape:
                _fail_a3b_gdn_postconv_configuration(
                    "A3B GDN postconv parameter_shape mismatch for "
                    f"{parameter_name} at GDN layer {index}"
                )
            if getattr(node, "dtype", None) != mx.bfloat16:
                _fail_a3b_gdn_postconv_configuration(
                    "A3B GDN postconv parameter_dtype requires BF16 for "
                    f"{parameter_name} at GDN layer {index}"
                )
        _validate_a3b_quant_projection(gdn, "in_proj_qkv", (8192, 32), index)
        _validate_a3b_quant_projection(gdn, "in_proj_a", (32, 32), index)
        _validate_a3b_quant_projection(gdn, "in_proj_b", (32, 32), index)

    _GDN_POSTCONV_STATS.update(
        {
            "installation_status": "awaiting_selfcheck",
            "installation_error": None,
            "gdn_layers": 30,
            "validated_contract": _a3b_gdn_postconv_contract(),
        }
    )
    return A3BGDNPostconvInstallPlan(gdns=tuple(gdns))


def install_a3b_gdn_postconv(
    plan: A3BGDNPostconvInstallPlan,
    selfcheck_report: dict[str, Any] | None,
) -> A3BGDNPostconvFactory:
    """Install the exact M1/M2 callables only after their combined self-check."""
    lanes = {} if selfcheck_report is None else selfcheck_report.get("lanes", {})
    implementation = _a3b_gdn_postconv_impl_selection()
    if implementation == "headquarter":
        required_lane = "gdn_postconv_headquarter"
        m1_apply = _apply_enabled_a3b_gdn_postconv_m1_headquarter
        m2_apply = _apply_enabled_a3b_gdn_postconv_m2_headquarter
        m3_apply = _apply_enabled_a3b_gdn_postconv_m3_headquarter
        b8_t2_apply = _apply_enabled_a3b_gdn_postconv_b8_t2_headquarter
        b3_t2_apply = _apply_enabled_a3b_gdn_postconv_b3_t2_headquarter
    else:
        required_lane = "gdn_postconv_inline_g"
        m1_apply = _apply_enabled_a3b_gdn_postconv_m1_tgy4
        m2_apply = _apply_enabled_a3b_gdn_postconv_m2_tgy4
        m3_apply = _apply_enabled_a3b_gdn_postconv_m3_tgy4
        b8_t2_apply = _apply_enabled_a3b_gdn_postconv_b8_t2_tgy4
        b3_t2_apply = _apply_enabled_a3b_gdn_postconv_b3_t2_tgy4
    if lanes.get(required_lane) != "ok":
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv selfcheck did not validate the exact M1/M2 kernels"
            + (
                ""
                if implementation == _A3B_GDN_POSTCONV_IMPL_DEFAULT
                else f" for the {implementation} route"
            )
        )
    factory = A3BGDNPostconvFactory(
        m1_implementations=tuple(
            partial(
                m1_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        m2_implementations=tuple(
            partial(
                m2_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        m3_implementations=tuple(
            partial(
                m3_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        b8_t2_implementations=tuple(
            partial(
                b8_t2_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        b3_t2_implementations=tuple(
            partial(
                b3_t2_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
    )
    _GDN_POSTCONV_STATS["installed"] = True
    _GDN_POSTCONV_STATS["installation_status"] = "installed"
    _GDN_POSTCONV_STATS["implementation"] = implementation
    return factory


def gdn_postconv_stats() -> dict[str, Any]:
    """Report the immutable installation contract, never hot-path counters."""
    report = dict(_GDN_POSTCONV_STATS)
    contract = report.get("validated_contract")
    report["validated_contract"] = (
        dict(contract) if isinstance(contract, dict) else None
    )
    return report


def _reset_gdn_postconv_stats_for_tests() -> None:
    _GDN_POSTCONV_STATS.update(
        {
            "enabled": False,
            "installed": False,
            "installation_status": "disabled",
            "installation_error": None,
            "gdn_layers": 0,
            "validated_contract": None,
            "implementation": "inline_g",
        }
    )


def _cache_context_len(cache: Any) -> int:
    if cache is None:
        return 0
    best = 0
    for entry in cache:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, mx.array):
            continue
        if offset is not None:
            best = max(best, int(offset or 0))
            continue
        size = getattr(entry, "size", None)
        if callable(size):
            try:
                best = max(best, int(size() or 0))
            except Exception:
                pass
    return best


def _target_layer_eval_every(context_len: int) -> int:
    schedule = os.environ.get("MTPLX_TARGET_LAYER_EVAL_SCHEDULE", "").strip()
    selected = 0
    if schedule:
        for part in schedule.replace(";", ",").split(","):
            item = part.strip()
            if not item:
                continue
            try:
                threshold_text, every_text = item.split(":", 1)
                threshold = int(threshold_text)
                every = int(every_text)
            except ValueError:
                continue
            if int(context_len) >= threshold:
                selected = max(0, every)
        return selected
    return int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY", "0") or "0")


def _make_linear_conv1d_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto c_idx = thread_position_in_grid.x;
        auto b_idx = thread_position_in_grid.y;

        if (c_idx >= ConvDim) {
          return;
        }

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          float acc = 0.0f;
          for (int k = 0; k < Keep; ++k) {
            float x;
            if (parent_idx < 0) {
              x = static_cast<float>(
                base_conv_state[(b_idx * Keep + k) * ConvDim + c_idx]
              );
            } else {
              x = static_cast<float>(
                conv_states[
                  (((b_idx * T + parent_idx) * Keep + k) * ConvDim) + c_idx
                ]
              );
            }
            auto w = static_cast<float>(conv_weight[c_idx * (Keep + 1) + k]);
            acc += x * w;
          }

          auto qkv_t = qkv + (b_idx * T + t) * ConvDim;
          acc += static_cast<float>(qkv_t[c_idx])
            * static_cast<float>(conv_weight[c_idx * (Keep + 1) + Keep]);

          conv_out[(b_idx * T + t) * ConvDim + c_idx] =
            static_cast<InT>(acc);

          for (int k = 0; k < Keep; ++k) {
            InT value;
            if (k + 1 < Keep) {
              if (parent_idx < 0) {
                value = base_conv_state[(b_idx * Keep + k + 1) * ConvDim + c_idx];
              } else {
                value = conv_states[
                  (((b_idx * T + parent_idx) * Keep + k + 1) * ConvDim) + c_idx
                ];
              }
            } else {
              value = qkv_t[c_idx];
            }
            conv_states[
              (((b_idx * T + t) * Keep + k) * ConvDim) + c_idx
            ] = value;
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_conv1d_capture",
        input_names=["qkv", "base_conv_state", "conv_weight", "T"],
        output_names=["conv_out", "conv_states"],
        source=source,
    )


def _make_linear_gated_delta_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_capture_v2",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_final_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }
        }

        auto state_out_ptr = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_out_ptr[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_final_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_stream_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          int capture_t = t - CaptureStart;
          if (capture_t >= 0) {
            auto state_t = states
              + (((b_idx * CaptureT + capture_t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto rounded = static_cast<StT>(state[i]);
              state_t[s_idx] = rounded;
              state[i] = static_cast<float>(rounded);
            }
          } else {
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(static_cast<StT>(state[i]));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_stream_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
            tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx] = delta;
          }

          for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = final_state + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "final_state", "tape"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_replay_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < Steps; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto g_t = g + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float k_sum = 0.0f;
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              k_sum += k_raw[i] * k_raw[i];
            }
            k_sum = simd_sum(k_sum);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          auto delta = tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            state[i] = state[i] + k_shared[s_idx] * delta;
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_replay_v1",
        input_names=["tape", "conv_out", "g", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_inline_g_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];
        threadgroup float g_shared;
        threadgroup float beta_shared;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto a_t = a + (b_idx * T + t) * Hv;
          auto b_t = b + (b_idx * T + t) * Hv;

          if (dk_idx == 0 && local_dv_idx == 0) {
            InT b_val = b_t[hv_idx];
            auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
            InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

            InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
            constexpr InT inf = metal::numeric_limits<InT>::infinity();
            InT maxval = metal::max(a_val, InT(0));
            InT minval = metal::min(a_val, InT(0));
            InT softplus_val = (minval == -inf || maxval == inf)
              ? maxval
              : (maxval + log1p(metal::exp(minval - maxval)));
            float decay_a = metal::exp(float(A_log[hv_idx]));
            beta_shared = static_cast<float>(beta_val);
            g_shared = metal::exp(-decay_a * float(softplus_val));
          }

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_shared;
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem)
            * beta_shared;

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_inline_g_v1",
        input_names=["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_headquarter_kernel():
    # C1 "headquarter" redesigned execution: one threadgroup per (head, Dv-quarter)
    # => grid (SIMDS*32, QUARTERS, B*Hv), threadgroup (SIMDS*32, 1, 1) = 8 simdgroups.
    # simd 0 computes the head's q/k rms-norm+scale + g/beta once into threadgroup
    # memory (redundancy 32x -> 4x), one producer->consumer barrier, then each
    # simdgroup drives RPS=(Dv/QUARTERS)/SIMDS=4 dv rows with fp32 state resident in
    # registers across the T loop.  Source verbatim from the G3a C1 bench candidate
    # (bit-exact vs inline_g: parity 0.0 on y and states at m1 and m2).
    if not mx.metal.is_available():
        return None

    source = """
    // --- geometry -----------------------------------------------------------
    auto n = thread_position_in_grid.z;          // b_idx*Hv + hv_idx
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    auto quarter = thread_position_in_grid.y;     // 0..QUARTERS-1
    uint tptg = thread_position_in_threadgroup.x; // 0..(SIMDS*32-1)
    uint simd_id = tptg / 32u;                     // 0..SIMDS-1
    uint dk_idx = thread_index_in_simdgroup;       // 0..31
    constexpr int n_per_t = Dk / 32;               // 4  (float4 per lane)
    constexpr int QSIZE = Dv / Quarters;           // dv rows per quarter (32)
    constexpr int RPS = QSIZE / Simds;             // dv rows per simdgroup (4)
    int base_dv = int(quarter) * QSIZE + int(simd_id) * RPS;

    float inv_scale = 1.0f / metal::sqrt(float(Dk));
    float q_scale = inv_scale * inv_scale;
    float k_scale = static_cast<float>(static_cast<InT>(inv_scale));

    threadgroup float q_shared[Dk];
    threadgroup float k_shared[Dk];
    threadgroup float g_shared;
    threadgroup float beta_shared;

    // running fp32 state for this simdgroup's RPS rows, resident in registers
    float S[RPS][n_per_t];
    for (int r = 0; r < RPS; ++r) {
      const device float4* s4 = reinterpret_cast<const device float4*>(
        state_in + (n * Dv + (base_dv + r)) * Dk);
      float4 sv = s4[dk_idx];
      S[r][0] = sv.x; S[r][1] = sv.y; S[r][2] = sv.z; S[r][3] = sv.w;
    }

    for (int t = 0; t < T; ++t) {
      auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
      auto q_t = conv_t + hk_idx * Dk;
      auto k_t = conv_t + KeyDim + hk_idx * Dk;
      auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
      auto a_t = a + (b_idx * T + t) * Hv;
      auto b_t = b + (b_idx * T + t) * Hv;

      // --- producer: simd 0 computes shared q/k (+ g/beta) once -------------
      if (simd_id == 0u) {
        if (dk_idx == 0u) {
          InT b_val = b_t[hv_idx];
          auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
          InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

          InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
          constexpr InT inf = metal::numeric_limits<InT>::infinity();
          InT maxval = metal::max(a_val, InT(0));
          InT minval = metal::min(a_val, InT(0));
          InT softplus_val = (minval == -inf || maxval == inf)
            ? maxval
            : (maxval + log1p(metal::exp(minval - maxval)));
          float decay_a = metal::exp(float(A_log[hv_idx]));
          beta_shared = static_cast<float>(beta_val);
          g_shared = metal::exp(-decay_a * float(softplus_val));
        }

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float q_raw[n_per_t];
        float k_raw[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          q_raw[i] = static_cast<float>(q_t[s_idx]);
          k_raw[i] = static_cast<float>(k_t[s_idx]);
          q_sum += q_raw[i] * q_raw[i];
          k_sum += k_raw[i] * k_raw[i];
        }
        q_sum = simd_sum(q_sum);
        k_sum = simd_sum(k_sum);
        float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
        float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
          auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
          q_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
          k_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);   // BARRIER 1 (producer->consumer)

      // --- consumer: each simdgroup drives its RPS rows --------------------
      float g_local = g_shared;
      float beta_local = beta_shared;
      float qloc[n_per_t];
      float kloc[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        qloc[i] = q_shared[s_idx];
        kloc[i] = k_shared[s_idx];
      }

      float kv[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] * g_local;
          acc += S[r][i] * kloc[i];
        }
        kv[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { kv[r] = simd_sum(kv[r]); }

      float delta[RPS];
      for (int r = 0; r < RPS; ++r) {
        delta[r] = (static_cast<float>(v_t[base_dv + r]) - kv[r]) * beta_local;
      }

      float out[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] + kloc[i] * delta[r];
          acc += S[r][i] * qloc[i];
        }
        out[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { out[r] = simd_sum(out[r]); }

      auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
      for (int r = 0; r < RPS; ++r) {
        int dv = base_dv + r;
        if (dk_idx == 0u) {
          y_t[dv] = static_cast<InT>(out[r]);
        }
        device float4* o4 = reinterpret_cast<device float4*>(
          states + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv) * Dk);
        o4[dk_idx] = float4(S[r][0], S[r][1], S[r][2], S[r][3]);
      }

      if (t + 1 < T) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // BARRIER 2 (WAR guard, T>1 only)
      }
    }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_headquarter_v1",
        input_names=["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


_linear_conv1d_kernel = _make_linear_conv1d_kernel()
_linear_gated_delta_kernel = _make_linear_gated_delta_kernel()
_linear_gated_delta_final_kernel = _make_linear_gated_delta_final_kernel()
_linear_gated_delta_from_conv_kernel = _make_linear_gated_delta_from_conv_kernel()
_linear_gated_delta_from_conv_stream_kernel = (
    _make_linear_gated_delta_from_conv_stream_kernel()
)
_linear_gated_delta_from_conv_tape_kernel = (
    _make_linear_gated_delta_from_conv_tape_kernel()
)
_linear_gated_delta_from_conv_tape_replay_kernel = (
    _make_linear_gated_delta_from_conv_tape_replay_kernel()
)
_linear_gated_delta_from_conv_inline_g_kernel = (
    _make_linear_gated_delta_from_conv_inline_g_kernel()
)
_linear_gated_delta_from_conv_headquarter_kernel = (
    _make_linear_gated_delta_from_conv_headquarter_kernel()
)

_LINEAR_GDN_ALIASES = {"linear_gdn", "linear_gdn_len5"}
_LINEAR_GDN_FROM_CONV_ALIASES = {
    "linear_gdn_from_conv",
    "linear_gdn_from_conv_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_ALIASES = {
    "linear_gdn_from_conv_stream",
    "linear_gdn_from_conv_stream_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES = {
    "linear_gdn_from_conv_stream_skip0",
    "linear_gdn_from_conv_stream_skip0_len5",
}
_LINEAR_GDN_FROM_CONV_TAPE_ALIASES = {
    "linear_gdn_from_conv_tape",
    "linear_gdn_from_conv_tape_len5",
}
_LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES = {
    "linear_gdn_from_conv_inline_g",
    "linear_gdn_from_conv_inline_g_len5",
}
_LINEAR_GDN_FINAL_ALIASES = {"linear_gdn_final", "linear_gdn_final_len5"}
_DEMOTED_GDN_ALIASES = {
    "linear_gdn_conv",
    "linear_gdn_len6",
    "linear_gdn_mlp_gateup",
}


def _contiguous_recurrent_leaf(value: mx.array) -> mx.array:
    # Mirrors mlx-lm #1077's cache ownership fix: the authoritative recurrent
    # leaf must not retain the larger per-position capture buffer.
    return mx.contiguous(value)


def _maybe_contiguous_authoritative_gdn_leaf(value: mx.array) -> mx.array:
    if not _env_enabled("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE"):
        return value
    return _contiguous_recurrent_leaf(value)


def _gdn_tape_meta(gdn: Any) -> dict[str, int]:
    return {
        "conv_dim": int(gdn.conv_dim),
        "head_k_dim": int(gdn.head_k_dim),
        "head_v_dim": int(gdn.head_v_dim),
        "num_k_heads": int(gdn.num_k_heads),
        "num_v_heads": int(gdn.num_v_heads),
        "key_dim": int(gdn.key_dim),
    }


def _gdn_meta_int(meta: Any, name: str) -> int:
    if isinstance(meta, dict):
        return int(meta[name])
    return int(getattr(meta, name))


def resolve_gdn_capture_backend(backend: str | None = None) -> str:
    """Resolve the GDN capture backend with backwards-compatible env support."""
    if backend is None:
        env_value = os.environ.get("MTPLX_CAPTURE_CUSTOM_KERNEL")
        if env_value is None:
            return "stock"
        normalized_env = env_value.lower().replace("-", "_")
        if normalized_env in {"1", "true", "yes", "on"} | _LINEAR_GDN_ALIASES:
            return "linear_gdn"
        if normalized_env in _LINEAR_GDN_FROM_CONV_ALIASES:
            return "linear_gdn_from_conv"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
            return "linear_gdn_from_conv_stream"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
            return "linear_gdn_from_conv_stream_skip0"
        if normalized_env in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
            return "linear_gdn_from_conv_tape"
        if normalized_env in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
            return "linear_gdn_from_conv_inline_g"
        if normalized_env in _LINEAR_GDN_FINAL_ALIASES:
            return "linear_gdn_final"
        if normalized_env in {"0", "false", "no", "off", "stock"}:
            return "stock"
        if normalized_env in _DEMOTED_GDN_ALIASES:
            raise ValueError(
                f"MTPLX_CAPTURE_CUSTOM_KERNEL backend {env_value!r} is not promoted; "
                "use 'stock', 'linear-gdn', 'linear-gdn-len5', or "
                "'linear-gdn-from-conv'"
            )
        raise ValueError(
            "MTPLX_CAPTURE_CUSTOM_KERNEL must be one of 1/0, true/false, "
            "'linear-gdn', 'linear-gdn-len5', 'linear-gdn-from-conv', or 'stock'"
        )
    normalized = backend.replace("-", "_")
    if normalized == "stock":
        return "stock"
    if normalized in _LINEAR_GDN_ALIASES:
        return "linear_gdn"
    if normalized in _LINEAR_GDN_FROM_CONV_ALIASES:
        return "linear_gdn_from_conv"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
        return "linear_gdn_from_conv_stream"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
        return "linear_gdn_from_conv_stream_skip0"
    if normalized in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
        return "linear_gdn_from_conv_tape"
    if normalized in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
        return "linear_gdn_from_conv_inline_g"
    if normalized in _LINEAR_GDN_FINAL_ALIASES:
        return "linear_gdn_final"
    if normalized in _DEMOTED_GDN_ALIASES:
        raise ValueError(
            f"GDN capture backend {backend!r} is not promoted; use 'stock' or "
            "'linear-gdn-len5'"
        )
    raise ValueError(
        "GDN capture backend must be 'stock', 'linear-gdn', 'linear-gdn-len5', "
        "'linear-gdn-from-conv', or diagnostic 'linear-gdn-final'"
    )


def _linear_conv1d_capture(
    qkv: mx.array, base_conv_state: mx.array, conv_weight: mx.array
):
    if _linear_conv1d_kernel is None:
        return None
    B, T, conv_dim = qkv.shape
    keep = int(base_conv_state.shape[1])
    if (
        len(conv_weight.shape) != 3
        or int(conv_weight.shape[0]) != conv_dim
        or int(conv_weight.shape[1]) != keep + 1
        or int(conv_weight.shape[2]) != 1
    ):
        return None
    input_type = qkv.dtype
    raw_conv, conv_states = _linear_conv1d_kernel(
        inputs=[qkv, base_conv_state, conv_weight, T],
        template=[("InT", input_type), ("Keep", keep), ("ConvDim", conv_dim)],
        grid=(conv_dim, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, conv_dim), (B, T, keep, conv_dim)],
        output_dtypes=[input_type, input_type],
    )
    return nn.silu(raw_conv), conv_states


def _matching_quantized_linears(left: Any, right: Any) -> bool:
    if not isinstance(left, nn.QuantizedLinear) or not isinstance(
        right, nn.QuantizedLinear
    ):
        return False
    if "bias" in left or "bias" in right:
        return False
    return (
        int(left.bits) == int(right.bits)
        and int(left.group_size) == int(right.group_size)
        and str(left.mode) == str(right.mode)
        and tuple(left.weight.shape[1:]) == tuple(right.weight.shape[1:])
        and tuple(left.scales.shape[1:]) == tuple(right.scales.shape[1:])
        and tuple(left.biases.shape[1:]) == tuple(right.biases.shape[1:])
    )


def _fused_quantized_pair(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    left: nn.QuantizedLinear,
    right: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array] | None:
    if not _matching_quantized_linears(left, right):
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([left.weight, right.weight], axis=0)
        scales = mx.concatenate([left.scales, right.scales], axis=0)
        biases = mx.concatenate([left.biases, right.biases], axis=0)
        mx.eval(weight, scales, biases)
        cached = (weight, scales, biases, int(left.weight.shape[0]))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_at = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(left.group_size),
        bits=int(left.bits),
        mode=str(left.mode),
    )
    left_out, right_out = mx.split(out, [int(split_at)], axis=-1)
    return left_out, right_out


def _fused_quantized_many(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    modules: tuple[nn.QuantizedLinear, ...],
) -> tuple[mx.array, ...] | None:
    if not modules:
        return None
    first = modules[0]
    if any(not _matching_quantized_linears(first, module) for module in modules[1:]):
        return None
    if "bias" in first:
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([module.weight for module in modules], axis=0)
        scales = mx.concatenate([module.scales for module in modules], axis=0)
        biases = mx.concatenate([module.biases for module in modules], axis=0)
        mx.eval(weight, scales, biases)
        sizes = [int(module.weight.shape[0]) for module in modules]
        split_points = []
        running = 0
        for size in sizes[:-1]:
            running += size
            split_points.append(running)
        cached = (weight, scales, biases, tuple(split_points))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_points = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(first.group_size),
        bits=int(first.bits),
        mode=str(first.mode),
    )
    return tuple(mx.split(out, list(split_points), axis=-1))


def _gdn_input_projections(
    gdn: Any, inputs: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    fuse_mode = os.environ.get("MTPLX_FUSE_GDN_PROJECTIONS", "").lower()
    if fuse_mode in {"all", "4to1", "one"}:
        fused = _fused_quantized_many(
            gdn,
            "_mtplx_fused_qkvzba",
            inputs,
            (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
        )
        if fused is not None:
            qkv, z, b, a = fused
            return qkv, z, b, a
    if fuse_mode in {"1", "true", "yes", "on"}:
        qkvz = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_qkvz",
            inputs,
            gdn.in_proj_qkv,
            gdn.in_proj_z,
        )
        ba = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_ba",
            inputs,
            gdn.in_proj_b,
            gdn.in_proj_a,
        )
        if qkvz is not None and ba is not None:
            qkv, z = qkvz
            b, a = ba
            return qkv, z, b, a
    return (
        gdn.in_proj_qkv(inputs),
        gdn.in_proj_z(inputs),
        gdn.in_proj_b(inputs),
        gdn.in_proj_a(inputs),
    )


def configure_qwen38_row18_gdn_decay_memo(
    model: Any,
    *,
    active: bool,
) -> dict[str, int]:
    """Materialize row 18's per-layer ``-exp(A_log)`` outside generation."""
    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    layers = getattr(inner, "layers", None) or []
    configured = 0
    active_modules = 0
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None or not hasattr(gdn, "A_log") or not hasattr(gdn, "dt_bias"):
            continue
        configured += 1
        memo = -mx.exp(gdn.A_log.astype(mx.float32)) if active else None
        if memo is not None:
            mx.eval(memo)
            active_modules += 1
        if memo is None:
            from mlx_lm.models.gated_delta import compute_g

            gdn._mtplx_compute_g = partial(
                compute_g,
                gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
        else:
            gdn._mtplx_compute_g = lambda a, memo=memo, bias=gdn.dt_bias: mx.exp(
                memo * nn.softplus(a + bias)
            )
    return {
        "configured_modules": configured,
        "active_modules": active_modules,
    }


def bind_stock_gdn_compute_g(model: Any) -> int:
    """Install the stock decay callable once on every constructed GDN module."""

    from mlx_lm.models.gated_delta import compute_g

    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    bound = 0
    for layer in getattr(inner, "layers", None) or []:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None or not hasattr(gdn, "A_log") or not hasattr(gdn, "dt_bias"):
            continue
        if not hasattr(gdn, "_mtplx_compute_g"):
            gdn._mtplx_compute_g = partial(
                compute_g,
                gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
        bound += 1
    return bound


def _stock_conv1d_capture(qkv: mx.array, base_conv_state: mx.array, gdn: Any):
    """Run the exact MLX Conv1d path and capture each linear-prefix state."""
    B, T, _ = qkv.shape
    keep = int(base_conv_state.shape[1])
    conv_input = mx.concatenate([base_conv_state, qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_input))
    conv_states = mx.stack(
        [conv_input[:, i + 1 : i + 1 + keep, :] for i in range(T)],
        axis=1,
    )
    return conv_out, conv_states


def _linear_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_final(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_final_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_final_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_stream_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    capture_start: int = 0,
):
    if _linear_gated_delta_from_conv_stream_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    capture_start = int(capture_start)
    if capture_start < 0 or capture_start >= int(T):
        return None
    capture_t = int(T) - capture_start
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    default_tgy = "8" if capture_start else "32"
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", default_tgy))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_stream_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
            ("CaptureStart", capture_start),
            ("CaptureT", capture_t),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, capture_t, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_tape_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    # Alternative execution layout for the same contract (A3B C1 lineage).
    # Fail-closed: any ineligibility returns None from the wrapper and we
    # fall through to the incumbent TGY kernel below.
    if (
        os.environ.get("MTPLX_LINEAR_GDN_TAPE_IMPL", "").strip().lower()
        == "headquarter"
    ):
        try:
            from .kernels.gdn_tape_headquarter import headquarter_tape_capture
        except ImportError as exc:
            # The user explicitly opted in; falling back must be loud, not
            # silent, or the incumbent masquerades as the requested kernel.
            headquarter_tape_capture = None
            global _HEADQUARTER_IMPORT_WARNED
            if not _HEADQUARTER_IMPORT_WARNED:
                _HEADQUARTER_IMPORT_WARNED = True
                logger.warning(
                    "MTPLX_LINEAR_GDN_TAPE_IMPL=headquarter requested but the "
                    "kernel module is unavailable (%s); using the incumbent "
                    "tape kernel",
                    exc,
                )
        if headquarter_tape_capture is not None:
            result = headquarter_tape_capture(conv_out, g, beta, state, gdn)
            if result is not None:
                return result
    if _linear_gated_delta_from_conv_tape_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
    except ValueError:
        tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_tape_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _linear_gated_delta_from_conv_tape_replay(
    tape: mx.array,
    conv_out: mx.array,
    g: mx.array,
    state: mx.array,
    gdn_meta: Any,
    *,
    steps: int,
):
    if _linear_gated_delta_from_conv_tape_replay_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != _gdn_meta_int(gdn_meta, "conv_dim"):
        return None
    steps = int(steps)
    if steps <= 0 or steps > int(T):
        return None
    Dk = _gdn_meta_int(gdn_meta, "head_k_dim")
    Dv = _gdn_meta_int(gdn_meta, "head_v_dim")
    Hk = _gdn_meta_int(gdn_meta, "num_k_heads")
    Hv = _gdn_meta_int(gdn_meta, "num_v_heads")
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
    except ValueError:
        tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    (state_out,) = _linear_gated_delta_from_conv_tape_replay_kernel(
        inputs=[tape, conv_out, g, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", _gdn_meta_int(gdn_meta, "key_dim")),
            ("ConvDim", _gdn_meta_int(gdn_meta, "conv_dim")),
            ("Steps", steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[state.shape],
        output_dtypes=[state_type],
    )
    return state_out


def _linear_gated_delta_from_conv_inline_g_capture(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_inline_g_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, gdn.A_log, gdn.dt_bias, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _a3b_compiled_target_gdn_postconv_m1_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M1 recurrence with TGY4."""
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 1],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 1, 32, 128), (1, 1, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M2 recurrence with TGY4."""
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_b8_t2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed eight-row A3B M2 recurrence with TGY4."""
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 256),
        threadgroup=(32, 4, 1),
        output_shapes=[(8, 2, 32, 128), (8, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_b3_t2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed three-row A3B M2 recurrence with TGY4.

    Identical arithmetic to the eight-row launch: the inline_g source derives
    ``b_idx = grid.z / Hv`` so the batch extent lives only in the grid z size
    (rows * Hv = 3 * 32 = 96) and the output batch dimension.
    """
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 96),
        threadgroup=(32, 4, 1),
        output_shapes=[(3, 2, 32, 128), (3, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m1_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M1/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_m1_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M2/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_m2_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_b8_t2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed eight-row A3B M2/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_b8_t2_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_b3_t2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed three-row A3B M2/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_b3_t2_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _a3b_compiled_target_gdn_postconv_m1_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M1 recurrence with the C1 headquarter kernel."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 1],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, 32, 128), (1, 1, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M2 recurrence with the C1 headquarter kernel."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_b8_t2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed eight-row A3B M2 recurrence with headquarter."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 256),
        threadgroup=(256, 1, 1),
        output_shapes=[(8, 2, 32, 128), (8, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_b3_t2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed three-row A3B M2 recurrence with headquarter.

    Same source as the eight-row launch; the grid z extent carries the batch
    (rows * Hv = 96) and the outputs carry three rows.
    """
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 96),
        threadgroup=(256, 1, 1),
        output_shapes=[(3, 2, 32, 128), (3, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m1_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M1 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_m1_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M2 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_m2_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_b8_t2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed eight-row A3B M2 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_b8_t2_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_b3_t2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed three-row A3B M2 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_b3_t2_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _a3b_compiled_target_gdn_postconv_m3_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the A3B compiled-target M3 (k=2, 3-row) recurrence with TGY4.

    Identical to the M2 launch except the logical sequence length is 3 -- the
    inline_g kernel scans ``logical_m`` positions, so the k=2 verify
    ``[primary, d1, d2]`` recurrence reuses the exact M1/M2 arithmetic per row.
    """
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 3],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 3, 32, 128), (1, 3, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m3_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the A3B compiled-target M3 (k=2, 3-row) recurrence with headquarter."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 3],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 3, 32, 128), (1, 3, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m3_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M3/TGY4 route (k=2)."""
    return _a3b_compiled_target_gdn_postconv_m3_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m3_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M3 headquarter route (k=2)."""
    return _a3b_compiled_target_gdn_postconv_m3_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _stock_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    mask: Any,
    gdn: Any,
):
    """Capture per-position recurrent state through stock MLX single-token steps."""
    from mlx_lm.models.gated_delta import gated_delta_update

    T = int(q.shape[1])
    outs = []
    states = []
    current = state
    for idx in range(T):
        step_mask = None
        if mask is not None and not isinstance(mask, str):
            step_mask = mask[:, idx : idx + 1]
        out, current = gated_delta_update(
            q[:, idx : idx + 1, :, :],
            k[:, idx : idx + 1, :, :],
            v[:, idx : idx + 1, :, :],
            a[:, idx : idx + 1, :],
            b[:, idx : idx + 1, :],
            gdn.A_log,
            gdn.dt_bias,
            current,
            step_mask,
            use_kernel=not gdn.training,
        )
        outs.append(out)
        states.append(current)
    return mx.concatenate(outs, axis=1), mx.stack(states, axis=1)


def gdn_forward_with_capture(
    gdn: Any,
    inputs: mx.array,
    mask: Any = None,
    cache: Any = None,
    *,
    capture_backend: str | None = None,
):
    if getattr(gdn, "sharding_group", None) is not None:
        return gdn(inputs, mask=mask, cache=cache), None

    B, S, _ = inputs.shape
    qkv, z, b, a = _gdn_input_projections(gdn, inputs)
    z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=inputs.dtype,
        )

    conv_capture = None
    if _env_enabled("MTPLX_LINEAR_CONV1D_CAPTURE"):
        conv_capture = _linear_conv1d_capture(qkv, conv_state, gdn.conv1d.weight)
    if conv_capture is None:
        conv_capture = _stock_conv1d_capture(qkv, conv_state, gdn)
    conv_out, conv_states = conv_capture
    backend = resolve_gdn_capture_backend(capture_backend)

    state = cache[1] if cache and cache[1] is not None else None
    if state is None:
        state = mx.zeros(
            (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32
        )

    final_only_capture = False
    capture_start = 0
    if backend == "linear_gdn_from_conv_inline_g":
        delta_result = _linear_gated_delta_from_conv_inline_g_capture(
            conv_out,
            a,
            b,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_from_conv_tape":
        beta = mx.sigmoid(b)
        g = gdn._mtplx_compute_g(a)
        delta_result = _linear_gated_delta_from_conv_tape_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state, tape = delta_result
        states = final_state[:, None, :, :, :]
    elif backend in {
        "linear_gdn_from_conv_stream",
        "linear_gdn_from_conv_stream_skip0",
    }:
        beta = mx.sigmoid(b)
        g = gdn._mtplx_compute_g(a)
        capture_start = 1 if backend == "linear_gdn_from_conv_stream_skip0" else 0
        delta_result = _linear_gated_delta_from_conv_stream_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
            capture_start=capture_start,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend in {"linear_gdn", "linear_gdn_from_conv"}:
        use_from_conv = backend == "linear_gdn_from_conv" or os.environ.get(
            "MTPLX_LINEAR_GDN_FROM_CONV", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        beta = mx.sigmoid(b)
        g = gdn._mtplx_compute_g(a)
        if use_from_conv:
            delta_result = _linear_gated_delta_from_conv_capture(
                conv_out, g, beta, state, gdn
            )
        else:
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                    [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                    [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
                )
            ]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            delta_result = _linear_gated_delta_capture(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_final":
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = gdn._mtplx_compute_g(a)
        delta_result = _linear_gated_delta_final(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state = delta_result
        states = final_state[:, None, :, :, :]
        final_only_capture = True
    else:
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, states = _stock_gated_delta_capture(q, k, v, a, b, state, mask, gdn)

    if cache is not None:
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = _maybe_contiguous_authoritative_gdn_leaf(states[:, -1, :, :, :])
        cache.advance(S)

    tail_projected = False
    if os.environ.get("MTPLX_NATIVE_GDN_TAIL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernels.native_gdn_tail import native_gdn_norm_gate_out_qmv8

        out = native_gdn_norm_gate_out_qmv8(
            out,
            z,
            gdn.norm.weight,
            gdn.norm.eps,
            gdn.out_proj,
            num_simdgroups=int(os.environ.get("MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS") or 2),
        )
        tail_projected = True
    elif os.environ.get("MTPLX_FUSE_GDN_NORM_GATE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernel_selfcheck import lane_disabled
        from .kernels.fused_norm import fused_gdn_norm_gate

        if lane_disabled("fused_gdn_norm_gate"):
            out = gdn.norm(out, z)
        else:
            out = fused_gdn_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps)
    else:
        out = gdn.norm(out, z)
    if not tail_projected:
        out = out.reshape(B, S, -1)
        if os.environ.get("MTPLX_GDN_OUT_QMV8", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .verify_qmv import stocklike_qmv8_matmul

            out = stocklike_qmv8_matmul(out, gdn.out_proj)
        else:
            out = gdn.out_proj(out)
    if final_only_capture:
        return out, {"final_only": True}
    if backend == "linear_gdn_from_conv_tape":
        return out, {
            "conv_states": conv_states,
            "conv_out": conv_out,
            "g": g,
            "state_in": state,
            "tape": tape,
            "gdn_meta": _gdn_tape_meta(gdn),
        }
    if capture_start:
        return out, {
            "conv_states": conv_states[:, capture_start:, :, :],
            "states": states,
            "capture_start": capture_start,
        }
    return out, {"conv_states": conv_states, "states": states}


def _a3b_gdn_forward_with_fixed_postconv(
    gdn: Any,
    inputs: mx.array,
    cache: Any,
    postconv_implementation: Callable[..., Any],
):
    """Build the unchecked exact A3B GDN graph with stock surroundings."""
    B, S, _ = inputs.shape
    qkv = gdn.in_proj_qkv(inputs)
    z = gdn.in_proj_z(inputs).reshape(B, S, 32, 128)
    b = gdn.in_proj_b(inputs)
    a = gdn.in_proj_a(inputs)
    conv_state = cache[0]
    conv_out, conv_states = _stock_conv1d_capture(qkv, conv_state, gdn)
    out, states = postconv_implementation(conv_out, a, b, cache[1])
    cache[0] = mx.contiguous(conv_states[:, -1, :, :])
    cache[1] = states[:, -1, :, :, :]
    out = gdn.norm(out, z)
    out = gdn.out_proj(out.reshape(B, S, -1))
    return out, {"conv_states": conv_states, "states": states}


def _b8_t2_rowwise_b1_qlinear(
    inputs: mx.array,
    implementation: Callable[[mx.array], mx.array],
) -> mx.array:
    """Run the fixed B8/T2 input as eight unchanged B1/T2 projections."""

    return mx.concatenate(
        tuple(implementation(inputs[row : row + 1]) for row in range(8)),
        axis=0,
    )


def _a3b_gdn_forward_with_fixed_postconv_bound_projections(
    gdn: Any,
    inputs: mx.array,
    cache: Any,
    postconv_implementation: Callable[..., Any],
    b1_qkv_implementation: Callable[[mx.array], mx.array],
    z_implementation: Callable[[mx.array], mx.array],
    b_implementation: Callable[[mx.array], mx.array],
    a_implementation: Callable[[mx.array], mx.array],
):
    """Build the balanced B8 graph with construction-bound projections."""

    B, S, _ = inputs.shape
    qkv = b1_qkv_implementation(inputs)
    z = z_implementation(inputs).reshape(B, S, 32, 128)
    b = b_implementation(inputs)
    a = a_implementation(inputs)
    conv_state = cache[0]
    conv_out, conv_states = _stock_conv1d_capture(qkv, conv_state, gdn)
    out, states = postconv_implementation(conv_out, a, b, cache[1])
    cache[0] = mx.contiguous(conv_states[:, -1, :, :])
    cache[1] = states[:, -1, :, :, :]
    out = gdn.norm(out, z)
    out = gdn.out_proj(out.reshape(B, S, -1))
    return out, {"conv_states": conv_states, "states": states}


def forward_with_a3b_gdn_postconv_capture(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    hidden_variant: str | None,
    postconv_implementations: tuple[Callable[..., Any], ...],
):
    """Build the unchecked exact 40-layer A3B target trace."""
    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)

    from mlx_lm.models.base import create_attention_mask

    attention_mask = create_attention_mask(hidden_states, cache[3])
    captures: dict[int, dict[str, mx.array]] = {}
    implementation_iter = iter(postconv_implementations)
    for layer_idx, (layer, layer_cache, kind) in enumerate(
        zip(inner.layers, cache, _A3B_GDN_POSTCONV_LAYER_TYPES)
    ):
        normed = layer.input_layernorm(hidden_states)
        if kind == "linear_attention":
            r, capture = _a3b_gdn_forward_with_fixed_postconv(
                layer.linear_attn,
                normed,
                layer_cache,
                next(implementation_iter),
            )
            captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=attention_mask, cache=layer_cache)
        h = hidden_states + r
        mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
    return logits, hidden, captures


def forward_with_a3b_gdn_postconv_capture_bound_projections(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    hidden_variant: str | None,
    postconv_implementations: tuple[Callable[..., Any], ...],
    qkv_implementations: tuple[Callable[[mx.array], mx.array], ...],
    z_implementations: tuple[Callable[[mx.array], mx.array], ...],
    b_implementations: tuple[Callable[[mx.array], mx.array], ...],
    a_implementations: tuple[Callable[[mx.array], mx.array], ...],
):
    """Build the unchecked layer-zero-B1-QKV/Z/B balanced B8/T2 trace."""

    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)

    from mlx_lm.models.base import create_attention_mask

    attention_mask = create_attention_mask(hidden_states, cache[3])
    captures: dict[int, dict[str, mx.array]] = {}
    postconv_iter = iter(postconv_implementations)
    qkv_iter = iter(qkv_implementations)
    z_iter = iter(z_implementations)
    b_iter = iter(b_implementations)
    a_iter = iter(a_implementations)
    for layer_idx, (layer, layer_cache, kind) in enumerate(
        zip(inner.layers, cache, _A3B_GDN_POSTCONV_LAYER_TYPES)
    ):
        normed = layer.input_layernorm(hidden_states)
        if kind == "linear_attention":
            r, capture = _a3b_gdn_forward_with_fixed_postconv_bound_projections(
                layer.linear_attn,
                normed,
                layer_cache,
                next(postconv_iter),
                next(qkv_iter),
                next(z_iter),
                next(b_iter),
                next(a_iter),
            )
            captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=attention_mask, cache=layer_cache)
        h = hidden_states + r
        mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
    return logits, hidden, captures


def _fused_post_norm_tg_override() -> int | None:
    """Threadgroup override for the fused post-norm residual lane.

    None (the default) lets fused_add_rmsnorm mirror mx.fast.rms_norm's own
    exact-fit/looped dispatch, which is bitwise-identical to the unfused
    reference at every probed axis/row/dtype. A fixed value forces the looped
    kernel at that lane count and changes the fp32 partial-sum partition — the
    shipped 512 produced one-ULP fp16 flips at axes 3072/5120 from 64 rows up
    (#319). Env knob exists for A/B archaeology only.
    """
    raw = os.environ.get("MTPLX_FUSE_POST_NORM_RESIDUAL_TG", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class _StockBoundaryRoute:
    def prepare(self, hidden_states, _base, _delta, layer):
        return hidden_states, layer.input_layernorm(hidden_states)

    def finish(self, hidden_in, residual, hidden_states, layer):
        if os.environ.get("MTPLX_FUSE_POST_NORM_RESIDUAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .kernel_selfcheck import lane_disabled
            from .kernels.fused_norm import fused_add_rmsnorm

            if lane_disabled("fused_add_rmsnorm"):
                h = hidden_states + residual
                mlp_input = layer.post_attention_layernorm(h)
            else:
                h, mlp_input = fused_add_rmsnorm(
                    hidden_states,
                    residual,
                    layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.eps,
                    threadgroup_size=(
                        override
                        if (override := _fused_post_norm_tg_override()) is not None
                        else (512 if hidden_states.dtype == mx.bfloat16 else None)
                    ),
                )
        else:
            h = hidden_states + residual
            mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)
        return hidden_states, hidden_states, None

    def eval_roots(self, hidden_states, _base, _delta):
        return (hidden_states,)

    def finalize(self, hidden_states, _base, _delta):
        return hidden_states


class _Row48BoundaryRoute:
    def prepare(self, _hidden_states, base, delta, layer):
        if delta is None:
            return base, layer.input_layernorm(base)
        from .kernels.fused_norm import fused_add_rmsnorm

        return fused_add_rmsnorm(
            base,
            delta,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            threadgroup_size=1024,
        )

    def finish(self, hidden_in, residual, _hidden_states, layer):
        from .kernels.fused_norm import fused_add_rmsnorm

        h, mlp_input = fused_add_rmsnorm(
            hidden_in,
            residual,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            threadgroup_size=1024,
        )
        delta = layer.mlp(mlp_input)
        return h, h, delta

    def eval_roots(self, _hidden_states, base, delta):
        return (base, delta)

    def finalize(self, _hidden_states, base, delta):
        return base if delta is None else base + delta


_STOCK_BOUNDARY_ROUTE = _StockBoundaryRoute()
_ROW48_BOUNDARY_ROUTE = _Row48BoundaryRoute()


def forward_with_gdn_capture(
    model: Any,
    inputs: mx.array,
    cache=None,
    return_hidden: bool = False,
    *,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
    boundary_route: Any = _STOCK_BOUNDARY_ROUTE,
):
    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(inner.layers)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
    captures: dict[int, dict[str, mx.array]] = {}
    backend = resolve_gdn_capture_backend(capture_backend)
    context_len = _cache_context_len(cache)
    layer_eval_every = _target_layer_eval_every(context_len)
    layer_eval_threshold = int(
        os.environ.get("MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD", "0") or "0"
    )
    layer_eval_max_q = int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q", "8") or "8")
    layer_eval_enabled = (
        layer_eval_every > 0
        and int(inputs.shape[1]) <= max(1, layer_eval_max_q)
        and context_len >= max(0, layer_eval_threshold)
    )

    boundary_base = hidden_states
    boundary_delta = None
    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_in, normed = boundary_route.prepare(
            hidden_states,
            boundary_base,
            boundary_delta,
            layer,
        )
        if layer.is_linear:
            if os.environ.get("MTPLX_ABLATE_LINEAR_ATTN", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                r = mx.zeros_like(normed)
            else:
                r, capture = gdn_forward_with_capture(
                    layer.linear_attn,
                    normed,
                    mask=mask,
                    cache=layer_cache,
                    capture_backend=backend,
                )
                if capture is not None:
                    if capture.get("final_only"):
                        captures["__final_only__"] = True
                    else:
                        captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=mask, cache=layer_cache)
        hidden_states, boundary_base, boundary_delta = boundary_route.finish(
            hidden_in,
            r,
            hidden_states,
            layer,
        )
        if layer_eval_enabled and (layer_idx + 1) % layer_eval_every == 0:
            mx.eval(
                *boundary_route.eval_roots(
                    hidden_states,
                    boundary_base,
                    boundary_delta,
                )
            )

    hidden_states = boundary_route.finalize(
        hidden_states,
        boundary_base,
        boundary_delta,
    )

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    if return_hidden:
        hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
        return logits, hidden, captures
    return logits, captures


def configure_qwen38_row48_capture(runtime: Any, *, active: bool) -> dict[str, int]:
    """Prebind the stock or fused-boundary target capture entrypoint."""

    route = _ROW48_BOUNDARY_ROUTE if active else _STOCK_BOUNDARY_ROUTE
    runtime._forward_ar_capture_gdn = partial(
        forward_with_gdn_capture,
        boundary_route=route,
    )
    return {"active": int(active), "construction_bound": 1}


def commit_captured_prefix(
    cache: list[Any],
    captures: dict[int, dict[str, mx.array]],
    keep_tokens: int,
    verified_tokens: int,
    *,
    detach_components: set[str] | None = None,
    detach_mode: str = "selected_slice_contiguous_eval",
    detach_stats: dict[str, int] | None = None,
) -> bool:
    if keep_tokens <= 0 or keep_tokens > verified_tokens:
        return False
    if captures.get("__final_only__"):
        return False
    detach_requested = {
        item.strip().lower().replace("-", "_")
        for item in (detach_components or set())
        if item
    }
    trim_tokens = verified_tokens - keep_tokens
    capture_index = keep_tokens - 1
    for capture in captures.values():
        if isinstance(capture, dict):
            capture_start = int(capture.get("capture_start", 0))
            if capture_index - capture_start < 0:
                return False
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            capture_start = int(capture.get("capture_start", 0))
            adjusted_index = capture_index - capture_start
            conv_state = mx.contiguous(capture["conv_states"][:, adjusted_index, :, :])
            if "conv" in detach_requested:
                from .cache_state import detach_array_leaf

                conv_state = detach_array_leaf(conv_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(
                        conv_state.nbytes
                    )
            if "tape" in capture:
                replayed_state = _linear_gated_delta_from_conv_tape_replay(
                    capture["tape"],
                    capture["conv_out"],
                    capture["g"],
                    capture["state_in"],
                    capture.get("gdn_meta", capture.get("gdn")),
                    steps=capture_index + 1,
                )
                if replayed_state is None:
                    return False
                gdn_state = _maybe_contiguous_authoritative_gdn_leaf(replayed_state)
            else:
                gdn_state = _contiguous_recurrent_leaf(
                    capture["states"][:, adjusted_index, :, :, :]
                )
            if "gdn" in detach_requested:
                from .cache_state import detach_array_leaf

                gdn_state = detach_array_leaf(gdn_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(
                        gdn_state.nbytes
                    )
            from .cache_state import replace_recurrent_cache_state

            replace_recurrent_cache_state(entry, [conv_state, gdn_state])
        elif trim_tokens and hasattr(entry, "is_trimmable") and entry.is_trimmable():
            entry.trim(trim_tokens)
    return True


def _select_captured_rows(value: mx.array, indices: list[int]) -> mx.array:
    """Select one captured time position per batch row without a host round trip."""
    batch = int(value.shape[0])
    if batch != len(indices):
        raise ValueError(
            f"capture batch has {batch} rows, but {len(indices)} positions were given"
        )
    selector = mx.array(indices, dtype=mx.int32).reshape(
        (batch, 1) + (1,) * (int(value.ndim) - 2)
    )
    selector = mx.broadcast_to(selector, (batch, 1) + tuple(value.shape[2:]))
    return mx.contiguous(mx.take_along_axis(value, selector, axis=1)[:, 0])


def commit_captured_rows(
    cache: list[Any],
    captures: dict[int, dict[str, mx.array]],
    keep_tokens_by_row: list[int] | tuple[int, ...],
    verified_tokens: int,
) -> bool:
    """Commit a different verified prefix length for every fixed cohort row.

    This is the Qwen 35B A3B ``[B, 2]`` MTP commit boundary.  Full-attention
    entries must already be :class:`RaggedBatchKVCache` instances so their
    logical offsets can move independently.  Recurrent entries are rebound to
    the captured state at each row's authoritative position.  The installed
    post-conv capture path supplies both states directly; tape replay is not a
    supported hot-path fallback.
    """
    verified = int(verified_tokens)
    keeps = [int(value) for value in keep_tokens_by_row]
    if not keeps or any(value <= 0 or value > verified for value in keeps):
        return False
    if captures.get("__final_only__"):
        return False

    from .cache_state import _is_trimmable
    from .ragged_kv_cache import RaggedBatchKVCache

    adjusted_by_layer: dict[int, list[int]] = {}
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None:
            if "tape" in capture:
                return False
            if "conv_states" not in capture or "states" not in capture:
                return False
            capture_start = int(capture.get("capture_start", 0))
            adjusted = [value - 1 - capture_start for value in keeps]
            if any(value < 0 for value in adjusted):
                return False
            if len(keeps) != int(capture["conv_states"].shape[0]):
                return False
            adjusted_by_layer[layer_idx] = adjusted
        elif _is_trimmable(entry):
            if isinstance(entry, RaggedBatchKVCache):
                if entry.offsets is not None and int(entry.offsets.size) != len(keeps):
                    return False
            elif len(set(keeps)) != 1:
                return False
        elif entry is not None and hasattr(entry, "state"):
            # The installed A3B layout has exactly 30 recurrent entries and
            # every one must have a post-conv capture.  Missing ownership is a
            # cohort failure, never permission to keep speculative final state.
            return False

    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None:
            adjusted = adjusted_by_layer[layer_idx]
            conv_state = _select_captured_rows(capture["conv_states"], adjusted)
            gdn_state = _select_captured_rows(capture["states"], adjusted)
            # Rebind the two leaves directly.  OwnedRecurrentStateCache's
            # item assignment is deliberately lazy; replace_state would add a
            # per-cycle synchronization and copy to this enabled hot path.
            if hasattr(entry, "__setitem__"):
                entry[0] = conv_state
                entry[1] = gdn_state
            else:
                entry.state = [conv_state, gdn_state]
        elif isinstance(entry, RaggedBatchKVCache):
            entry.offsets = (
                entry.offsets - verified + mx.array(keeps, dtype=mx.int32)
            ).astype(mx.int32)
        elif _is_trimmable(entry):
            trim = verified - keeps[0]
            if trim:
                entry.trim(trim)
    return True
