"""Install-time validation for the DENSE batched-MTP serving lane.

This is the dense counterpart of ``install_a3b_mtp_batch_lane``
(``mtplx/a3b_mtp_batch.py``), and it deliberately gates on something
different.

The A3B lane compiles a fixed-width capture graph against one exact MoE
topology, so its ``_validate_config`` is an exact-match allowlist: model type
``qwen3_5_moe``, 256 experts, 8 experts per token, a specific hidden size and
layer-type string, specific quantization bits and group sizes. Anything else is
refused, because anything else would compile a graph that is silently wrong.

``generate_dense_mtp_batch`` compiles no such graph. It builds its shapes from
the runtime at call time and it shipped benched against two different public
dense models. So the honest dense gate is a CAPABILITY gate, not a config
allowlist:

* the model must be DENSE, not MoE (a MoE config belongs to the A3B lane, and
  routing it here would run the dense driver over a router topology it does not
  model);
* MTP must be enabled, because the driver's whole cycle is draft-then-verify;
* the runtime must expose the five entry points the driver calls;
* the capture backend must materialize per-step GDN states, because per-row
  commit selects each row's state at its own accept length.

Refusing a MoE model here is the "bypassing the MoE router-receipt gate" half
of the work: the dense lane does not consult the router receipt at all, and it
declines the models for which that receipt exists.

Install runs NO forward pass. The A3B lane can afford a numerical self-check at
startup because its graph is fixed and its cost is bounded; a dense self-check
would mean a real prefill of a 27B model on the startup path. The self-check
recorded here is structural, and it says so in its own payload rather than
claiming a numerical result it did not compute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mtplx.artifacts import load_config
from mtplx.gdn_capture import resolve_gdn_capture_backend

__all__ = [
    "DenseMTPBatchGeometry",
    "DenseMTPBatchInstallError",
    "InstalledDenseMTPBatchLane",
    "install_dense_mtp_batch_lane",
    "model_is_dense_mtp_batch_capable",
]

# Capture backends that do NOT materialize per-step GDN states. Mirrors the
# driver's own refusal exactly (``generate_dense_mtp_batch`` rejects these two
# by resolved name) rather than keeping a separate allowlist that could drift
# and over-refuse a backend the driver accepts. The point of repeating it here
# is WHEN, not what: at install, on the startup path, instead of on the first
# cohort a real request lands in.
_NON_STATE_MATERIALIZING_BACKENDS = {
    "linear_gdn_from_conv_tape",
    "linear_gdn_final",
}

# Runtime entry points ``generate_dense_mtp_batch`` calls. ``update_mtp_cache``
# is required only for head_history='committed', which is the lane default and
# the measured-better policy, so it is required here too.
_REQUIRED_RUNTIME_METHODS = (
    "forward_ar",
    "forward_ar_capture",
    "draft_mtp",
    "make_cache",
    "make_mtp_cache",
    "update_mtp_cache",
)


class DenseMTPBatchInstallError(RuntimeError):
    """The dense batched-MTP lane cannot be installed against this runtime."""


@dataclass(frozen=True)
class DenseMTPBatchGeometry:
    """Admission bounds for one installed dense lane.

    ``cohort_slots`` is the widest cohort the service will seal. It is an
    admission cap, not a compiled shape: unlike the A3B lane there is no
    fixed-width graph behind it, so changing it costs nothing at install.

    ``max_context_tokens`` bounds ``prompt_tokens + max_tokens`` for one row.
    The driver freezes KV capacity per run from those two numbers, so this is
    what keeps a single very long request from sizing the whole cohort's cache.
    """

    cohort_slots: int = 8
    max_context_tokens: int = 32768
    depth: int = 3


@dataclass(frozen=True)
class InstalledDenseMTPBatchLane:
    """A validated dense runtime plus the decode settings the service uses.

    Mirrors the attribute surface the server already reads off the A3B lane
    (``route_id``, ``geometry``, ``config_fingerprint``, ``numerics_profile``,
    ``selfcheck``) so the observability and startup-banner paths need no
    special-casing for dense.
    """

    runtime: Any
    geometry: DenseMTPBatchGeometry
    route_id: str
    config_fingerprint: str
    model_type: str
    numerics_profile: str = "dense-throughput"
    capture_backend: str = "stock"
    head_history: str = "committed"
    loop_mode: str = "pipelined"
    draft_core: str = "eager"
    history_window: int = 8192
    prefill_chunk: int = 2048
    pad_id: int = 0
    selfcheck: dict[str, Any] = field(default_factory=dict)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else {}


def model_is_dense_mtp_batch_capable(config: dict[str, Any]) -> bool:
    """True when ``config`` is a DENSE qwen3_5, i.e. this lane's model.

    Used by the server to choose between the A3B lane and this one without
    catching an exception from either install. A MoE config returns False here
    and is the A3B lane's business; the two are mutually exclusive by
    construction, so a model can never install both.
    """

    text = _text_config(config)
    model_type = str(config.get("model_type") or "")
    text_type = str(text.get("model_type") or "")
    if "moe" in model_type or "moe" in text_type:
        return False
    if text.get("num_experts") is not None:
        return False
    return model_type == "qwen3_5"


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Stable short hash over the config fields the lane's behaviour depends on.

    Deliberately NOT a hash of the whole config: the point is to notice that
    the served topology or quantization changed under a cached route id, not to
    change identity when an unrelated key moves.
    """

    text = _text_config(config)
    material = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "layer_types": text.get("layer_types"),
        "vocab_size": text.get("vocab_size"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "quantization": config.get("quantization"),
        "mtplx_mtp_quantization": config.get("mtplx_mtp_quantization"),
        "mtplx_mtp_contract": config.get("mtplx_mtp_contract"),
    }
    blob = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def install_dense_mtp_batch_lane(
    runtime: Any,
    *,
    cohort_slots: int = 8,
    depth: int = 3,
    max_context_tokens: int = 32768,
    capture_backend: str = "stock",
    head_history: str = "committed",
    loop_mode: str = "pipelined",
    draft_core: str = "eager",
    history_window: int = 8192,
    prefill_chunk: int = 2048,
    pad_id: int = 0,
) -> InstalledDenseMTPBatchLane:
    """Validate a dense runtime for batched-MTP serving and freeze its settings.

    Raises :class:`DenseMTPBatchInstallError` on anything that would make the
    driver wrong or make it fail on a user's first request. Fails closed, in
    the A3B lane's discipline: there is no degraded install and no silent
    fallback to serial decode.
    """

    if int(cohort_slots) < 2:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch cohort_slots must be >= 2; got {cohort_slots}"
        )
    if int(depth) < 1:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch depth must be >= 1; got {depth}"
        )
    if int(max_context_tokens) < 1:
        raise DenseMTPBatchInstallError(
            "dense mtp_batch max_context_tokens must be >= 1; "
            f"got {max_context_tokens}"
        )

    model_path = getattr(runtime, "model_path", None)
    if model_path is None:
        raise DenseMTPBatchInstallError(
            "dense mtp_batch requires a runtime with model_path"
        )
    config = load_config(model_path)
    text = _text_config(config)
    if not isinstance(text, dict) or not text:
        raise DenseMTPBatchInstallError("dense mtp_batch requires text_config")

    model_type = str(config.get("model_type") or "")
    text_type = str(text.get("model_type") or "")
    if "moe" in model_type or "moe" in text_type or text.get("num_experts") is not None:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch refuses the MoE topology ({model_type or '?'} / "
            f"{text_type or '?'}); that model installs the A3B lane instead"
        )
    if not model_is_dense_mtp_batch_capable(config):
        raise DenseMTPBatchInstallError(
            "dense mtp_batch requires model_type 'qwen3_5'; "
            f"got {model_type or '(missing)'}"
        )

    if not bool(getattr(runtime, "mtp_enabled", False)):
        raise DenseMTPBatchInstallError(
            "dense mtp_batch requires an MTP-enabled runtime; the lane is a "
            "draft-then-verify loop and has no target-only path"
        )
    missing = [
        name for name in _REQUIRED_RUNTIME_METHODS if not callable(getattr(runtime, name, None))
    ]
    if missing:
        raise DenseMTPBatchInstallError(
            "dense mtp_batch runtime is missing required entry points: "
            + ", ".join(missing)
        )

    try:
        resolved_backend = resolve_gdn_capture_backend(capture_backend)
    except ValueError as exc:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch capture backend {capture_backend!r} is not a "
            f"capture backend at all: {exc}"
        ) from exc
    if resolved_backend in _NON_STATE_MATERIALIZING_BACKENDS:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch capture backend {capture_backend!r} does not "
            "materialize per-step GDN states, so per-row commit cannot select "
            "a row's state at its own accept length"
        )

    head_history = str(head_history).strip().lower()
    if head_history not in {"cycle", "committed"}:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch head_history must be 'cycle' or 'committed'; "
            f"got {head_history!r}"
        )
    loop_mode = str(loop_mode).strip().lower()
    if loop_mode not in {"pipelined", "serial"}:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch loop_mode must be 'pipelined' or 'serial'; "
            f"got {loop_mode!r}"
        )
    draft_core = str(draft_core).strip().lower()
    if draft_core not in {"eager", "compiled"}:
        raise DenseMTPBatchInstallError(
            f"dense mtp_batch draft_core must be 'eager' or 'compiled'; "
            f"got {draft_core!r}"
        )
    if draft_core == "compiled":
        # Item 3 makes sampling per row, and exact speculative sampling needs
        # the eager draft chain. Installing 'compiled' would mean every
        # temperature > 0 request failed at cohort time instead of at startup.
        raise DenseMTPBatchInstallError(
            "dense mtp_batch installs draft_core='eager' only; the compiled "
            "draft chain cannot run speculative sampling, so a compiled lane "
            "would refuse every temperature > 0 request at cohort time"
        )

    fingerprint = _config_fingerprint(config)
    route_id = (
        f"dense_mtp_batch/{model_type}/d{int(depth)}/"
        f"{resolved_backend}/{head_history}/{loop_mode}/{fingerprint}"
    )
    selfcheck = {
        "ok": True,
        "mode": "structural",
        "ran_forward": False,
        "why": (
            "the dense driver compiles no fixed-width graph, so there is no "
            "captured route to check numerically at startup; correctness is "
            "gated by the per-stream sha parity test instead"
        ),
        "model_type": model_type,
        "text_model_type": text_type,
        "capture_backend": resolved_backend,
    }
    return InstalledDenseMTPBatchLane(
        runtime=runtime,
        geometry=DenseMTPBatchGeometry(
            cohort_slots=int(cohort_slots),
            max_context_tokens=int(max_context_tokens),
            depth=int(depth),
        ),
        route_id=route_id,
        config_fingerprint=fingerprint,
        model_type=model_type,
        capture_backend=resolved_backend,
        head_history=head_history,
        loop_mode=loop_mode,
        draft_core=draft_core,
        history_window=int(history_window),
        prefill_chunk=int(prefill_chunk),
        pad_id=int(pad_id),
        selfcheck=selfcheck,
    )
