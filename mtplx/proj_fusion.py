"""Load-time projection fusion for Qwen3.5 hybrid models.

Concatenates projections that share one input along the output axis and replaces
the source modules, so their weights are freed: the GDN ``in_proj`` group into one
N=16480 matmul, attention q/k/v into N=14336, MLP gate/up into N=2*ffn. Affine
quantization groups run along the input axis, so the concatenation needs no
requantization. The fused module is an ``nn.QuantizedLinear``, so the verify-kernel
patches on ``nn.QuantizedLinear.__call__`` still route it. Members fall back to the
unfused computation outside the row window where fusion is bitwise exact.

Default off. ``MTPLX_FUSE_PROJ`` selects families: ``gdn``, ``attn``, ``mlp``,
``1``/``on``/``yes`` == ``gdn,attn``, ``all`` == ``gdn,attn,mlp``.
``MTPLX_FUSE_PROJ_MAX_ROWS`` overrides the row window ceiling.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import mlx.core as mx
import mlx.nn as nn

FUSE_ENV = "MTPLX_FUSE_PROJ"
MAX_ROWS_ENV = "MTPLX_FUSE_PROJ_MAX_ROWS"

_GDN_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_ATTN_NAMES = ("q_proj", "k_proj", "v_proj")
_MLP_NAMES = ("gate_proj", "up_proj")

_STATS: dict[str, Any] = {
    "enabled": False,
    "groups": "",
    "gdn": 0,
    "attn": 0,
    "mlp": 0,
    "skipped": 0,
    "skip_reasons": [],
    "max_fused_rows": 0,
    "fused_dispatches": 0,
    "member_calls": 0,
    "fused_lane_calls": 0,
    "unfused_lane_calls": 0,
    "freed_bytes": 0,
}


def requested_groups() -> set[str]:
    """Which projection families the environment asks to fuse."""

    raw = os.environ.get(FUSE_ENV, "").strip().lower()
    if raw in {"", "0", "off", "false", "no"}:
        return set()
    if raw in {"1", "true", "yes", "on"}:
        return {"gdn", "attn"}
    if raw == "all":
        return {"gdn", "attn", "mlp"}
    parts = {p.strip() for p in raw.replace(";", ",").split(",")}
    return {p for p in parts if p in {"gdn", "attn", "mlp"}}


def fuse_projections_enabled() -> bool:
    return bool(requested_groups())


def fused_projection_stats() -> dict[str, Any]:
    stats = dict(_STATS)
    stats["skip_reasons"] = list(_STATS["skip_reasons"])
    return stats


def reset_fused_projection_counters() -> None:
    _STATS["fused_dispatches"] = 0
    _STATS["member_calls"] = 0
    _STATS["fused_lane_calls"] = 0
    _STATS["unfused_lane_calls"] = 0


def _current_attention_phase() -> str | None:
    try:
        from .attention_context import current_attention_phase
    except Exception:  # pragma: no cover - standalone use outside the package
        return None
    return current_attention_phase()


class _FusionHub:
    """Runs one fused matmul per distinct input (identity-keyed) and serves each
    member a slice, dropping the memo once every member has been served."""

    __slots__ = ("fused", "split_points", "n_parts", "_full_mask", "_key", "_outs", "_served")

    def __init__(self, fused: nn.QuantizedLinear, split_points: list[int], n_parts: int):
        self.fused = fused
        self.split_points = list(split_points)
        self.n_parts = int(n_parts)
        self._full_mask = (1 << int(n_parts)) - 1
        self._key: Any = None
        self._outs: tuple[mx.array, ...] | None = None
        self._served = 0

    def part(self, x: mx.array, index: int) -> mx.array:
        _STATS["fused_lane_calls"] += 1
        if self._outs is None or self._key is not x:
            # Publish the key only after the matmul succeeds, or a retry after a
            # raise would be served the previous input's slices.
            self._key = None
            self._outs = None
            self._served = 0
            _outs = tuple(mx.split(self.fused(x), self.split_points, axis=-1))
            self._key = x
            self._outs = _outs
            _STATS["fused_dispatches"] += 1
        out = self._outs[index]
        self._served |= 1 << index
        if self._served == self._full_mask:
            self._key = None
            self._outs = None
            self._served = 0
        return out


class FusedProjectionMember(nn.QuantizedLinear):
    """One member of a fused projection group.

    ``weight`` / ``scales`` / ``biases`` are zero-copy row views into the fused
    arrays, so the member is a valid ``nn.QuantizedLinear`` owning no storage.
    ``__call__`` takes the fused lane only inside the row window where the fused
    matmul is bitwise identical to the separate ones, and otherwise defers to
    ``nn.QuantizedLinear.__call__`` over its own view. MLX picks kernel geometry
    from N, so the reduction order is N-dependent: on the stock lane fusing measured
    bitwise identical at M<=4 and differed by up to 2.5e-1 at M=6,7,8,17,32,64,128,
    512,1024. Serve is M=4 verify and M=1 draft; prefill stays on the unfused
    arithmetic.
    """

    def __init__(
        self,
        hub: _FusionHub,
        index: int,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array | None,
        *,
        group_size: int,
        bits: int,
        mode: str,
        max_rows: int,
    ):
        nn.Module.__init__(self)
        self._hub = hub
        self._index = int(index)
        self._max_rows = int(max_rows)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = str(mode)
        self.weight = weight
        self.scales = scales
        if biases is not None:
            self.biases = biases
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        _STATS["member_calls"] += 1
        rows = 1
        for d in x.shape[:-1]:
            rows *= int(d)
        # Rows above 4 are N-invariant only on lanes with row-invariant kernels, so a
        # raised ceiling also requires fp16 activations outside prefill.
        if rows <= 4 or (
            rows <= self._max_rows
            and x.dtype == mx.float16
            and _current_attention_phase() != "prefill"
        ):
            return self._hub.part(x, self._index)
        _STATS["unfused_lane_calls"] += 1
        return nn.QuantizedLinear.__call__(self, x)

    def _extra_repr(self) -> str:  # pragma: no cover - debug only
        return (
            f"fused_member index={self._index} output_dims={self['weight'].shape[0]} "
            f"max_rows={self._max_rows}, group_size={self.group_size}, "
            f"bits={self.bits}, mode={self.mode}"
        )


def _make_quantized_linear(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> nn.QuantizedLinear:
    """Build an ``nn.QuantizedLinear`` around ready-made arrays, bypassing
    ``__init__`` so no random ``[out, in]`` float matrix is allocated."""

    ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(ql)
    ql.group_size = int(group_size)
    ql.bits = int(bits)
    ql.mode = str(mode)
    ql.weight = weight
    ql.scales = scales
    if biases is not None:
        ql.biases = biases
    ql.freeze()
    return ql


def _quant_signature(module: Any) -> tuple[int, int, str] | None:
    if not isinstance(module, nn.QuantizedLinear):
        return None
    if "scales" not in module or "weight" not in module:
        return None
    return (
        int(getattr(module, "group_size", 0) or 0),
        int(getattr(module, "bits", 0) or 0),
        str(getattr(module, "mode", "affine")),
    )


def _why_not_fusable(modules: tuple[Any, ...]) -> str | None:
    first = modules[0]
    first_sig = _quant_signature(first)
    if first_sig is None:
        return "not an affine-quantized nn.QuantizedLinear"
    for module in modules:
        if isinstance(module, FusedProjectionMember):
            return "already fused"
        if _quant_signature(module) != first_sig:
            return "quantization config differs across the group"
        if "bias" in module:
            return "projection carries an additive bias"
        if module["weight"].ndim != 2:
            return "unexpected weight rank"
        if module["weight"].shape[1] != first["weight"].shape[1]:
            return "input widths differ"
        if module["weight"].dtype != first["weight"].dtype:
            return "weight dtypes differ"
        if module["scales"].shape[1] != first["scales"].shape[1]:
            return "scale group counts differ"
        if (module.get("biases") is None) != (first.get("biases") is None):
            return "quantization-bias presence differs"
    return None


def _array_bytes(array: mx.array | None) -> int:
    if array is None:
        return 0
    return int(array.size) * int(array.dtype.size)


def _default_max_rows() -> int:
    """Row ceiling for the fused lane. Defaults to the M<=4 qmv regime, which is
    N-invariant on every lane. Raise with ``MTPLX_FUSE_PROJ_MAX_ROWS`` only after
    measuring the exact window on the target backend."""

    raw = os.environ.get(MAX_ROWS_ENV, "").strip()
    if raw:
        return int(raw)
    return 4


def _fuse_group(owner: Any, names: tuple[str, ...], fused_attr: str, max_rows: int) -> str | None:
    """Replace ``names`` on ``owner`` with one fused projection. Returns a skip reason."""

    modules = tuple(getattr(owner, name, None) for name in names)
    if any(module is None for module in modules):
        return "member missing"
    reason = _why_not_fusable(modules)
    if reason is not None:
        return reason

    first = modules[0]
    group_size = int(first.group_size)
    bits = int(first.bits)
    mode = str(getattr(first, "mode", "affine"))
    has_biases = first.get("biases") is not None

    freed = 0
    for module in modules:
        freed += _array_bytes(module["weight"])
        freed += _array_bytes(module["scales"])
        freed += _array_bytes(module.get("biases"))

    weight = mx.concatenate([m["weight"] for m in modules], axis=0)
    scales = mx.concatenate([m["scales"] for m in modules], axis=0)
    biases = mx.concatenate([m["biases"] for m in modules], axis=0) if has_biases else None
    if biases is None:
        mx.eval(weight, scales)
    else:
        mx.eval(weight, scales, biases)

    fused = _make_quantized_linear(
        weight, scales, biases, group_size=group_size, bits=bits, mode=mode
    )

    rows = [int(m["weight"].shape[0]) for m in modules]
    scale_rows = [int(m["scales"].shape[0]) for m in modules]
    split_points, running = [], 0
    for n in rows[:-1]:
        running += n
        split_points.append(running)

    hub = _FusionHub(fused, split_points, len(modules))
    # Underscore-prefixed so Module.valid_parameter_filter skips it: as a registered
    # child it would double-count with the member views and materialise copies.
    setattr(owner, fused_attr, fused)

    w_at = s_at = 0
    for index, name in enumerate(names):
        w_view = weight[w_at : w_at + rows[index]]
        s_view = scales[s_at : s_at + scale_rows[index]]
        b_view = biases[s_at : s_at + scale_rows[index]] if biases is not None else None
        w_at += rows[index]
        s_at += scale_rows[index]
        setattr(
            owner,
            name,
            FusedProjectionMember(
                hub,
                index,
                w_view,
                s_view,
                b_view,
                group_size=group_size,
                bits=bits,
                mode=mode,
                max_rows=max_rows,
            ),
        )

    _STATS["freed_bytes"] += int(freed)
    return None


def _note_skip(kind: str, reason: str) -> None:
    _STATS["skipped"] += 1
    text = f"{kind}: {reason}"
    if text not in _STATS["skip_reasons"]:
        _STATS["skip_reasons"].append(text)


def configure_fused_projections(model: Any | None = None) -> dict[str, Any]:
    """Fuse the requested projection families of ``model`` in place. No-op unless
    ``MTPLX_FUSE_PROJ`` selects a family. Idempotent."""

    groups = requested_groups()
    _STATS["enabled"] = bool(groups)
    _STATS["groups"] = ",".join(sorted(groups))
    _STATS["gdn"] = 0
    _STATS["attn"] = 0
    _STATS["mlp"] = 0
    _STATS["skipped"] = 0
    _STATS["skip_reasons"] = []
    _STATS["freed_bytes"] = 0
    reset_fused_projection_counters()

    if not groups or model is None:
        return fused_projection_stats()

    max_rows = _default_max_rows()
    _STATS["max_fused_rows"] = int(max_rows)

    # MLX keeps freed blocks cached, so release periodically or the construction
    # transient is the full ~10 GiB of replaced weights.
    since_release = 0

    def _maybe_release(n: int) -> int:
        if n >= 8:
            mx.clear_cache()
            return 0
        return n

    for _, module in model.named_modules():
        if "gdn" in groups and all(hasattr(module, n) for n in _GDN_NAMES):
            reason = _fuse_group(module, _GDN_NAMES, "_mtplx_fused_in_proj", max_rows)
            if reason is None:
                _STATS["gdn"] += 1
                since_release = _maybe_release(since_release + 1)
            else:
                _note_skip("gdn", reason)
        if "attn" in groups and all(hasattr(module, n) for n in _ATTN_NAMES):
            reason = _fuse_group(module, _ATTN_NAMES, "_mtplx_fused_qkv_proj", max_rows)
            if reason is None:
                _STATS["attn"] += 1
                since_release = _maybe_release(since_release + 1)
            else:
                _note_skip("attn", reason)
        if "mlp" in groups and all(hasattr(module, n) for n in _MLP_NAMES):
            reason = _fuse_group(module, _MLP_NAMES, "_mtplx_fused_gate_up_proj", max_rows)
            if reason is None:
                _STATS["mlp"] += 1
                since_release = _maybe_release(since_release + 1)
            else:
                _note_skip("mlp", reason)

    if _STATS["gdn"] or _STATS["attn"] or _STATS["mlp"]:
        mx.clear_cache()
    reset_fused_projection_counters()
    # logger.info does not reach the serve console.
    print(
        f"[proj-fusion] groups={_STATS['groups']} gdn={_STATS['gdn']} "
        f"attn={_STATS['attn']} mlp={_STATS['mlp']} skipped={_STATS['skipped']} "
        f"max_fused_rows={_STATS['max_fused_rows']} "
        f"freed={_STATS['freed_bytes'] / 2 ** 30:.2f}GiB "
        f"reasons={_STATS['skip_reasons']}",
        file=sys.stderr,
        flush=True,
    )
    return fused_projection_stats()
