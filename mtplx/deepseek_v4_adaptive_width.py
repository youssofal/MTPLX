"""Construction contract for the preregistered DeepSeek-V4 max-K3 policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any

from .sampling import SamplerConfig


D1_MARGIN_THRESHOLD = 0.25
D2_MARGIN_THRESHOLD = 10.0
MAX_SPECULATIVE_DEPTH = 3
_FACTORY_SEAL = object()
_CANONICAL_TARGET_ROWS = (2, 3, 4)
_CANONICAL_O_LORA_FINGERPRINT = (
    "gather_qmm",
    44,
    43,
    1,
    43,
    1,
    True,
    True,
    43,
    "gather_qmm_m4_wide_direct",
    "_DirectGatherOLoraWideM4",
    1,
    "dense_bf16_stock_direct",
    "_DirectDenseMTPOLora",
    44,
    44,
    True,
)


def _o_lora_fingerprint(report: Any) -> tuple[Any, ...] | None:
    if not isinstance(report, dict):
        return None
    census = report.get("callable_census")
    if not isinstance(census, dict):
        return None
    return (
        report.get("mode"),
        report.get("module_count"),
        report.get("trunk_module_count"),
        report.get("mtp_module_count"),
        report.get("body_direct"),
        report.get("mtp_stock"),
        report.get("body_all_mode_matches"),
        report.get("route_plan_matches"),
        census.get("body_route_objects"),
        census.get("body_route_kind"),
        census.get("body_callable_class"),
        census.get("mtp_route_objects"),
        census.get("mtp_route_kind"),
        census.get("mtp_callable_class"),
        census.get("total_route_objects"),
        census.get("unique_route_objects"),
        census.get("mtp_distinct_type"),
    )


def _serialized_report(report: Any) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True, init=False)
class _DeepSeekV4TargetWidthRoute:
    """One prebound target-forward surface for an exact verify width."""

    expected_physical_rows: int
    forward: Callable[..., Any]
    _installation_seal: object = field(repr=False)

    def __init__(
        self,
        *,
        factory_seal: object,
        target_rows: int,
        forward: Callable[..., Any],
    ) -> None:
        if factory_seal is not _FACTORY_SEAL:
            raise TypeError("adaptive width target routes are factory-only")
        object.__setattr__(self, "expected_physical_rows", int(target_rows))
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "_installation_seal", factory_seal)

    @property
    def target_rows(self) -> int:
        return self.expected_physical_rows

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        return self.forward(input_ids, **kwargs)


@dataclass(frozen=True, slots=True, init=False)
class _DeepSeekV4AdaptiveWidthPolicy:
    """The single preregistered policy and its construction-validated surfaces."""

    runtime_object_id: int
    _runtime: Any = field(repr=False)
    _model: Any = field(repr=False)
    _capture_forward: Callable[..., Any] = field(repr=False)
    _capture_forward_function: Callable[..., Any] = field(repr=False)
    _o_lora_report_json: str = field(repr=False)
    _installation_seal: object = field(repr=False)
    target_routes: tuple[
        _DeepSeekV4TargetWidthRoute,
        _DeepSeekV4TargetWidthRoute,
        _DeepSeekV4TargetWidthRoute,
    ]
    d1_margin_threshold: float = field(default=D1_MARGIN_THRESHOLD, init=False)
    d2_margin_threshold: float = field(default=D2_MARGIN_THRESHOLD, init=False)
    max_speculative_depth: int = field(default=MAX_SPECULATIVE_DEPTH, init=False)
    verify_strategy: str = field(default="capture_commit", init=False)
    verify_core: str = field(default="stock", init=False)
    mtp_history_policy: str = field(default="committed", init=False)

    def __init__(
        self,
        *,
        factory_seal: object,
        runtime: Any,
        capture_forward: Callable[..., Any],
        capture_forward_function: Callable[..., Any],
        o_lora_report_json: str,
        target_routes: tuple[
            _DeepSeekV4TargetWidthRoute,
            _DeepSeekV4TargetWidthRoute,
            _DeepSeekV4TargetWidthRoute,
        ],
    ) -> None:
        if factory_seal is not _FACTORY_SEAL:
            raise TypeError("adaptive width policies are factory-only")
        object.__setattr__(self, "runtime_object_id", id(runtime))
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_model", runtime.model)
        object.__setattr__(self, "_capture_forward", capture_forward)
        object.__setattr__(
            self, "_capture_forward_function", capture_forward_function
        )
        object.__setattr__(self, "_o_lora_report_json", o_lora_report_json)
        object.__setattr__(self, "_installation_seal", factory_seal)
        object.__setattr__(self, "target_routes", target_routes)
        object.__setattr__(self, "d1_margin_threshold", D1_MARGIN_THRESHOLD)
        object.__setattr__(self, "d2_margin_threshold", D2_MARGIN_THRESHOLD)
        object.__setattr__(self, "max_speculative_depth", MAX_SPECULATIVE_DEPTH)
        object.__setattr__(self, "verify_strategy", "capture_commit")
        object.__setattr__(self, "verify_core", "stock")
        object.__setattr__(self, "mtp_history_policy", "committed")

    def stop_after_d1(self, margin: float) -> bool:
        return float(margin) < self.d1_margin_threshold

    def stop_after_d2(self, margin: float) -> bool:
        return float(margin) < self.d2_margin_threshold

    def validate_request(
        self,
        rt: Any,
        *,
        sampler: SamplerConfig,
        draft_sampler: SamplerConfig,
        speculative_depth: int,
        verify_strategy: str,
        verify_core: str,
        mtp_history_policy: str,
    ) -> None:
        """Reject a launch that differs from the installed policy contract."""

        if self._installation_seal is not _FACTORY_SEAL:
            raise ValueError("adaptive width policy installation seal is invalid")
        if self._runtime is not rt or id(rt) != self.runtime_object_id:
            raise ValueError("adaptive width policy belongs to a different runtime")
        if getattr(rt, "model", None) is not self._model:
            raise ValueError("adaptive width policy model authority changed")
        model_type = str(getattr(self._model, "model_type", "") or "").lower()
        if model_type != "deepseek_v4":
            raise ValueError("adaptive width policy DeepSeek-V4 authority changed")
        report = getattr(rt, "deepseek_v4_o_lora_report", None)
        if (
            _o_lora_fingerprint(report) != _CANONICAL_O_LORA_FINGERPRINT
            or _serialized_report(report) != self._o_lora_report_json
        ):
            raise ValueError("adaptive width policy canonical o-LoRA report changed")
        current_forward = getattr(rt, "forward_ar_capture", None)
        if (
            not callable(current_forward)
            or getattr(current_forward, "__self__", None) is not rt
            or getattr(current_forward, "__func__", None)
            is not self._capture_forward_function
            or getattr(type(rt), "forward_ar_capture", None)
            is not self._capture_forward_function
        ):
            raise ValueError("adaptive width policy capture-forward authority changed")
        if (
            not isinstance(self.target_routes, tuple)
            or len(self.target_routes) != 3
            or tuple(route.expected_physical_rows for route in self.target_routes)
            != _CANONICAL_TARGET_ROWS
            or any(
                type(route) is not _DeepSeekV4TargetWidthRoute
                or route._installation_seal is not _FACTORY_SEAL
                or route.forward is not self._capture_forward
                or getattr(route.forward, "__self__", None) is not rt
                or getattr(route.forward, "__func__", None)
                is not self._capture_forward_function
                for route in self.target_routes
            )
        ):
            raise ValueError("adaptive width policy target route authority changed")
        _validate_launch(
            sampler=sampler,
            draft_sampler=draft_sampler,
            speculative_depth=speculative_depth,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            mtp_history_policy=mtp_history_policy,
        )


def validate_installed_deepseek_v4_adaptive_width_policy(
    policy: Any,
    rt: Any,
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> None:
    """Authenticate the installed private type before trusting its methods."""

    if (
        type(policy) is not _DeepSeekV4AdaptiveWidthPolicy
        or getattr(policy, "_installation_seal", None) is not _FACTORY_SEAL
    ):
        raise ValueError("adaptive width policy must be factory-installed")
    policy.validate_request(
        rt,
        sampler=sampler,
        draft_sampler=draft_sampler,
        speculative_depth=speculative_depth,
        verify_strategy=verify_strategy,
        verify_core=verify_core,
        mtp_history_policy=mtp_history_policy,
    )


def _validate_launch(
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> None:
    if float(sampler.temperature) > 0.0:
        raise ValueError("adaptive width policy requires a greedy target sampler")
    if float(draft_sampler.temperature) > 0.0:
        raise ValueError("adaptive width policy requires a greedy draft sampler")
    if int(speculative_depth) != MAX_SPECULATIVE_DEPTH:
        raise ValueError("adaptive width policy requires fixed planned max-K3")
    if verify_strategy != "capture_commit":
        raise ValueError("adaptive width policy requires capture_commit verification")
    if verify_core != "stock":
        raise ValueError("adaptive width policy requires the stock verify core")
    if mtp_history_policy != "committed":
        raise ValueError("adaptive width policy requires committed MTP history")


def _validate_runtime(rt: Any) -> tuple[Callable[..., Any], Callable[..., Any], str]:
    if not bool(getattr(rt, "mtp_enabled", False)):
        raise ValueError("adaptive width policy requires an MTP-enabled runtime")
    model = getattr(rt, "model", None)
    model_type = str(getattr(model, "model_type", "") or "").lower()
    if model_type != "deepseek_v4":
        raise ValueError("adaptive width policy is only valid for DeepSeek-V4")

    report = getattr(rt, "deepseek_v4_o_lora_report", None)
    if _o_lora_fingerprint(report) != _CANONICAL_O_LORA_FINGERPRINT:
        raise ValueError("adaptive width policy requires the canonical o-LoRA route")

    forward = getattr(rt, "forward_ar_capture", None)
    forward_function = getattr(forward, "__func__", None)
    if (
        not callable(forward)
        or getattr(forward, "__self__", None) is not rt
        or not callable(forward_function)
        or getattr(type(rt), "forward_ar_capture", None) is not forward_function
    ):
        raise ValueError(
            "adaptive width policy requires the canonical owned capture target forward"
        )
    return forward, forward_function, _serialized_report(report)


def install_deepseek_v4_adaptive_width_policy(
    rt: Any,
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig | None,
    speculative_depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> _DeepSeekV4AdaptiveWidthPolicy:
    """Validate and bind the only supported adaptive-width configuration."""

    resolved_draft_sampler = sampler if draft_sampler is None else draft_sampler
    _validate_launch(
        sampler=sampler,
        draft_sampler=resolved_draft_sampler,
        speculative_depth=speculative_depth,
        verify_strategy=verify_strategy,
        verify_core=verify_core,
        mtp_history_policy=mtp_history_policy,
    )
    target_forward, target_forward_function, report_json = _validate_runtime(rt)
    target_routes = tuple(
        _DeepSeekV4TargetWidthRoute(
            factory_seal=_FACTORY_SEAL,
            target_rows=rows,
            forward=target_forward,
        )
        for rows in _CANONICAL_TARGET_ROWS
    )
    return _DeepSeekV4AdaptiveWidthPolicy(
        factory_seal=_FACTORY_SEAL,
        runtime=rt,
        capture_forward=target_forward,
        capture_forward_function=target_forward_function,
        o_lora_report_json=report_json,
        target_routes=target_routes,  # type: ignore[arg-type]
    )
