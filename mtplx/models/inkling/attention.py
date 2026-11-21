"""Relative attention for a query-dependent sink and chronological keys."""

import hashlib
import logging
import os
import platform
from collections import Counter
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import RLock

import mlx.core as mx

from .attention_metal import grouped_sink_attention, streamed_attention

_log = logging.getLogger(__name__)
_lock = RLock()
_counts = Counter()
_proofs = {}
_verify_proofs = {}
_request = ContextVar("inkling_attention_request", default=None)
_ROUTES = {"streamed": streamed_attention, "grouped_sink": grouped_sink_attention}
_PREFILL_TRIAL_DEVICE = ("Apple M5 Max", "applegpu_g17s", "27.0.1", "0.32.2")
# Exact priced widths. Interpolate key counts only between wins at that
# same width and window. These ranges authorize product experiments only.
_PREFILL_TRIAL_RANGES = {
    0: ((512, 512, 512, 512), (512, 512, 4096, 65536),
        (2048, 2048, 2048, 24576), (128, 128, 24576, 24576)),
    1024: ((2048, 2048, 2048, 3071),),
}
# Product-qualified ranges for the native 24:8, width-128, extent-1024 model.
# Key: (device, architecture, macOS, MLX, pack fingerprint, dtype, route, window).
# Value: (min rows, max rows, min chronological keys, max chronological keys) ranges.
# Admit only after pack-specific product A/B, 4K/64K KL, Pi and class sign-off.
_MEASURED_RANGES = {}


def attention_route_counts():
    """Process totals of graphs selected, including stock and its reason."""
    with _lock:
        return dict(_counts)


def attention_health():
    with _lock:
        return {"mode": os.environ.get("MTPLX_INKLING_ATTENTION", "stock"),
                "verify_mode": os.environ.get("MTPLX_INKLING_VERIFY_ATTENTION", "chronological"),
                "process_graph_selections": dict(_counts),
                "verify_first_use": {"passed": sum(v is None for v in _verify_proofs.values()),
                                     "declined": sum(v is not None for v in _verify_proofs.values())}}


def _pack_fingerprint(directory):
    # Bind the release gate to the qualified pack and recipe without reading
    # weights on the generation path. Anonymous or unstamped packs stay stock.
    try:
        root = Path(directory)
        return hashlib.sha256((root / "mtplx-pack.json").read_bytes() + b"\0" +
                              (root / "config.json").read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def observe_attention_request(fn, rt, *args, **kwargs):
    if _request.get() is not None:
        return fn(rt, *args, **kwargs)
    mode = os.environ.get("MTPLX_INKLING_ATTENTION", "stock")
    pack = _pack_fingerprint(rt.model_path) if mode == "auto" and _MEASURED_RANGES else None
    receipt = {"mode": mode, "verify_mode": os.environ.get("MTPLX_INKLING_VERIFY_ATTENTION", "chronological"),
               "graph_selections": {}, "prefill_shapes": {}, "unlisted_prefill_graphs": 0,
               "verify_widths": {}, "verify_first_use": {}}
    token = _request.set((receipt, pack))
    try:
        output = fn(rt, *args, **kwargs)
        output.stats.inkling_attention = receipt
        return output
    finally:
        _request.reset(token)


@lru_cache(maxsize=1)
def _metal_identity():
    info = mx.device_info(mx.gpu)
    return (
        info.get("device_name"), info.get("architecture"),
        platform.mac_ver()[0], version("mlx"),
    )


@lru_cache(maxsize=1)
def _bias_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_relative_bias",
        input_names=["profiles", "sink", "geometry"],
        output_names=["bias"],
        source="""
            const uint i = thread_position_in_grid.x;
            const int keys = geometry[0], first = geometry[1], window = geometry[2];
            const int heads = profiles_shape[2], rows = profiles_shape[1], extent = profiles_shape[3];
            const int col = i % (keys + 1);
            uint plane = i / (keys + 1);
            const int row = plane % rows;
            plane /= rows;
            const int head = plane % heads, batch = plane / heads;
            const uint p = (batch * rows + row) * heads + head;
            const int distance = first + row - col;
            if (col == keys) bias[i] = sink[p];
            else if (distance < 0 || (window && distance >= window)) bias[i] = T(-INFINITY);
            else if (distance >= extent) bias[i] = T(0);
            else bias[i] = profiles[p * extent + distance];
        """,
    )


def relative_bias(profiles, sink, keys, first, window, *, metal=True):
    batch, rows, heads, extent = profiles.shape
    shape = (batch, heads, rows, keys + 1)
    if metal and mx.default_device() == mx.gpu and mx.metal.is_available():
        return _bias_kernel()(
            inputs=[profiles, sink, mx.array([keys, first, window], mx.int32)],
            template=[("T", profiles.dtype)],
            grid=(batch * heads * rows * (keys + 1), 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[shape],
            output_dtypes=[profiles.dtype],
        )[0]
    distance = first + mx.arange(rows)[:, None] - mx.arange(keys)[None, :]
    values = mx.take_along_axis(
        profiles.transpose(0, 2, 1, 3),
        mx.clip(distance, 0, extent - 1)[None, None],
        axis=-1,
    )
    values = mx.where(distance >= extent, mx.zeros((), profiles.dtype), values)
    valid = distance >= 0 if not window else (distance >= 0) & (distance < window)
    return mx.concatenate(
        [
            mx.where(valid, values, mx.array(-float("inf"), profiles.dtype)),
            sink.transpose(0, 2, 1)[..., None],
        ],
        axis=-1,
    )


def stock_attention(query, key, value, profiles, sink, window):
    """Tile the query axis so a long prompt never builds its full quadratic bias tensor."""
    batch, heads, rows, width = query.shape
    keys = key.shape[2]
    zero = mx.zeros((batch, key.shape[1], 1, width), key.dtype)
    key = mx.concatenate([key, zero], axis=2)
    value = mx.concatenate([value, zero], axis=2)
    step = max(1, (4 * 1024 * 1024) // (batch * heads * (keys + 1)))
    parts = []
    for start in range(0, rows, step):
        end = min(rows, start + step)
        bias = relative_bias(
            profiles[:, start:end],
            sink[:, start:end],
            keys,
            keys - rows + start,
            window,
        )
        part = mx.fast.scaled_dot_product_attention(
            query[:, :, start:end], key, value, scale=width**-0.5, mask=bias
        )
        if rows > step:
            mx.eval(part)
        parts.append(part)
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def _in_ranges(rows, keys, ranges):
    return any(lo <= rows <= hi and begin <= keys <= end for lo, hi, begin, end in ranges)


def _decline_reason(route, mode, operands, window):
    if mode == "stock":
        return "forced_stock"
    if mode not in ("auto", "candidate", "prefill", *_ROUTES):
        return "unknown_mode"
    if (mode == "prefill" and route != "streamed") or mode not in ("auto", "candidate", "prefill", route):
        return "other_route"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    query, key, value, profiles, sink = operands
    batch, heads, rows, width = query.shape
    kv_heads, keys = key.shape[1:3]
    if query.dtype not in (mx.bfloat16, mx.float16) or any(x.dtype != query.dtype for x in operands[1:]):
        return "dtype"
    if (width != 128 or key.shape != value.shape or key.shape[0] != batch or key.shape[3] != width
            or not kv_heads or heads % kv_heads or not 0 < rows <= keys or window < 0
            or profiles.ndim != 4 or profiles.shape[:3] != (batch, rows, heads) or profiles.shape[3] < 1
            or sink.shape != (batch, rows, heads) or (route == "grouped_sink" and batch != 1)):
        return "geometry"
    if mode == "prefill":
        if query.dtype != mx.bfloat16 or (batch, heads, kv_heads, profiles.shape[3]) != (1, 24, 8, 1024):
            return "outside_prefill_trial"
        if not _in_ranges(rows, keys, _PREFILL_TRIAL_RANGES.get(window, ())):
            return "outside_prefill_trial"
        if _metal_identity() != _PREFILL_TRIAL_DEVICE:
            return "unmeasured_device"
    if mode == "auto":
        if not _MEASURED_RANGES or (batch, heads, kv_heads, profiles.shape[3]) != (1, 24, 8, 1024):
            return "unqualified"
        active = _request.get()
        if active is None or active[1] is None:
            return "unqualified_pack"
        dtype = "bfloat16" if query.dtype == mx.bfloat16 else "float16"
        cell = (*_metal_identity(), active[1], dtype, route, window)
        if not _in_ranges(rows, keys, _MEASURED_RANGES.get(cell, ())):
            return "unqualified"
    return None


def _first_use(route, operands, window):
    """Check a small real suffix once per specialization. Keep only the verdict."""
    query, key, value, profiles, sink = operands
    group = query.shape[1] // key.shape[1]
    rows = min(query.shape[2], 17)
    spec = (route, query.dtype, group, window, rows)
    with _lock:
        if spec in _proofs:
            return _proofs[spec]
        # Preserve the strided cache and input layout. Exercise an expired
        # profile as well as a partial matrix tile when the history permits.
        short = (
            query[:1, :group, -rows:], key[:1, :1, -65:], value[:1, :1, -65:],
            profiles[:1, -rows:, :group, :19], sink[:1, -rows:, :group],
        )
        ref = stock_attention(*short, window).astype(mx.float32)
        mx.eval(ref)
        try:
            got = _ROUTES[route](*short, window).astype(mx.float32)
            atol, rtol = (0.008, 0.015625) if query.dtype == mx.bfloat16 else (0.001, 0.002)
            ok = mx.all(mx.isfinite(got) & mx.isfinite(ref) & (mx.abs(got - ref) <= atol + rtol * mx.abs(ref)))
            reason = None if bool(ok.item()) else "numeric_mismatch"
            if reason:
                _log.warning("Inkling %s first-use error %g", route, mx.max(mx.abs(got - ref)).item())
        except (RuntimeError, ValueError, TypeError) as exc:
            _log.warning("Inkling %s first-use check failed: %s", route, exc)
            reason = "first_use_error"
        _proofs[spec] = reason
        return reason


def _chronological_verify(operands, window):
    """Submit the stock width-one arithmetic on each query's actual history."""
    q, k, v, profiles, sink = operands
    rows, keys = q.shape[2], k.shape[2]
    # Check again as the local ring fills/wraps, and at growing full-cache tiers.
    tier = min(rows, max(0, keys - window + 1)) if window else keys.bit_length()
    spec = (q.dtype, q.shape[1], k.shape[1], rows, window, tier)
    with _lock:
        prior = _verify_proofs.get(spec)
    if prior is not None:
        return None, prior, "cached_decline"

    pieces = []
    for row in range(rows):
        stop = keys - rows + row + 1
        begin = max(0, stop - window) if window else 0
        pieces.append(stock_attention(q[:, :, row:row + 1], k[:, :, begin:stop], v[:, :, begin:stop],
                                      profiles[:, row:row + 1], sink[:, row:row + 1], window))
    out = mx.concatenate(pieces, axis=2)
    with _lock:
        if spec in _verify_proofs:
            reason = _verify_proofs[spec]
            return (out if reason is None else None), reason, ("cached_pass" if reason is None else "cached_decline")
        try:
            flags = []
            # Form the endpoint histories independently by dropping future
            # tokens first, then retaining the last window of chronological KV.
            for row in (0, rows - 1):
                after = rows - row - 1
                pk, pv = (k[:, :, :-after], v[:, :, :-after]) if after else (k, v)
                if window:
                    pk, pv = pk[:, :, -window:], pv[:, :, -window:]
                wanted = stock_attention(q[:, :, row:row + 1], pk, pv,
                                         profiles[:, row:row + 1], sink[:, row:row + 1], window)
                actual = out[:, :, row:row + 1]
                flags.append(mx.all(mx.isfinite(actual) & (actual.view(mx.uint16) == wanted.view(mx.uint16))))
            reason = None if bool(mx.all(mx.stack(flags)).item()) else "verify_bit_mismatch"
        except (RuntimeError, ValueError, TypeError) as exc:
            _log.warning("Inkling chronological verify first-use check failed: %s", exc)
            reason = "verify_first_use_error"
        _verify_proofs[spec] = reason
    return (out if reason is None else None), reason, ("passed" if reason is None else "declined")


def attend(query, key, value, profiles, sink, window, *, verify=False):
    operands = (query, key, value, profiles, sink)
    route = "grouped_sink" if query.shape[2] <= 9 else "streamed"
    active = _request.get()
    mode = active[0]["mode"] if active is not None else os.environ.get("MTPLX_INKLING_ATTENTION", "stock")
    verify_mode = (active[0]["verify_mode"] if active is not None else
                   os.environ.get("MTPLX_INKLING_VERIFY_ATTENTION", "chronological")) if verify else "auto"
    rows = query.shape[2]
    use_chronological = verify and 2 <= rows <= 8 and (
        verify_mode == "chronological" or
        (verify_mode == mode == "auto" and bool(_MEASURED_RANGES) and
         _decline_reason("chronological_verify", "auto", operands, window) is None)
    )
    proof = None
    if use_chronological:
        route = "chronological_verify"
        reason = _decline_reason("grouped_sink", "candidate", operands, window)
        if reason is None:
            result, reason, proof = _chronological_verify(operands, window)
    else:
        if verify_mode in ("chronological", "stock"):
            mode = "stock"
        reason = _decline_reason(route, mode, operands, window) if verify_mode in ("auto", "stock", "chronological") else "unknown_verify_mode"
        if reason is None:
            reason = _first_use(route, operands, window)
        if reason is None:
            result = _ROUTES[route](*operands, window)
    if reason is None:
        outcome = "selected"
    else:
        result = stock_attention(*operands, window)
        outcome = "stock." + reason
    scope = "window" if window else "full"
    counter = ".".join((route, scope, outcome))
    with _lock:
        first = counter not in _counts
        _counts[counter] += 1
        if active is not None:
            receipt = active[0]
            counts = receipt["graph_selections"]
            counts[counter] = counts.get(counter, 0) + 1
            if verify:
                widths = receipt["verify_widths"].setdefault(f"{scope}.q{rows}", {})
                widths[counter] = widths.get(counter, 0) + 1
                if proof is not None:
                    checked = receipt["verify_first_use"]
                    checked[proof] = checked.get(proof, 0) + 1
            elif route == "streamed":
                shape = f"{scope}.q{query.shape[2]}.k{key.shape[2]}"
                shapes = receipt["prefill_shapes"]
                if shape in shapes or len(shapes) < 64:
                    outcomes = shapes.setdefault(shape, {})
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                else:
                    receipt["unlisted_prefill_graphs"] += 1
    if first:
        _log.info("Inkling attention %s", counter)
    return result
