"""Read the native 16-bit vocabulary matrix while keeping FP32 output logits."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
import os
from threading import RLock
from time import perf_counter

import mlx.core as mx
import numpy as np
from mtplx.kernels.first_use import GpuRace, Incomplete, Mismatch, ProofCache


_results, _calls, _lock = {}, Counter(), RLock()
_request_projections = ContextVar("inkling_request_projections", default=None)
_log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _working_set(device):
    info = mx.device_info(mx.default_device())
    memory = info.get("memory_size") or info.get("total_memory") or 0
    return int(info.get("max_recommended_working_set_size") or 0.7 * int(memory))


def fits_route(extra):
    return mx.get_active_memory() + extra + 128 * 1024**2 < _working_set(str(mx.default_device()))


def pack_proofs(path):
    @lru_cache(maxsize=4)
    def resolve(device):
        cache = ProofCache.for_pack(path)
        if cache.identity is None:
            return cache, None
        # The common identity owns the build, hardware, index and weight enumeration.
        # Include Inkling's recipe files and local replacements with preserved modification time.
        digest = hashlib.sha256(str(path.resolve()).encode())
        try:
            for name in ("mtplx-pack.json", "hf_quant_config.json"):
                f = path / name
                if f.is_file():
                    digest.update(name.encode() + b"\0" + f.read_bytes())
            for record in cache.identity["files"]:
                if len(record) == 3:  # Content-addressed shards already carry their immutable blob name.
                    st = (path / record[0]).stat()
                    digest.update(json.dumps([record[0], st.st_dev, st.st_ino, st.st_ctime_ns]).encode())
        except OSError as exc:
            _log.warning("Using in-memory Inkling proofs for %s: %s", path, exc)
            return ProofCache(None, None, pack_path=path, identity_error=str(exc)), None
        return cache, digest.hexdigest()
    return resolve


@lru_cache(maxsize=4)
def _anonymous(device):
    return ProofCache(None, None), None


@lru_cache(maxsize=8)
def _lookup_only(cache):
    # A cold no-fit refusal on this view cannot memoize the live proof out for the rest of a load.
    return ProofCache(cache.identity, cache.directory.parent, pack_path=cache.pack_path)


def note_route(name, reason):
    if name == "projection":
        request = _request_projections.get()
        if request is not None:
            request["panel" if reason in ("selected", "explicit_trial") else reason] += 1
    key = name + "." + reason
    with _lock:
        if not _calls[key]:
            _log.info("Inkling route %s", key)
        _calls[key] += 1


@contextmanager
def observe_projection_routes():
    """Count each eligible panel and its exact fallback in the serving request."""
    counts = Counter()
    previous = _request_projections.set(counts)
    try:
        yield counts
    finally:
        _request_projections.reset(previous)


def inkling_first_use_stats():
    with _lock:
        return {"calls": dict(_calls), "decisions": list(_results.values())}


def _bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item()


@dataclass(frozen=True)
class _WallRace(GpuRace):
    arm_ms: tuple = ((), ())
    first_use_ms: float = 0.

    def priced_arms(self):
        # Preserve the verified margin against the slowest candidate and the fastest stock sample.
        return (min(self.arm_ms[0]) * 1e6,), (max(self.arm_ms[1]) * 1e6,)

    def receipt(self):
        return {**super().receipt(), "price_metric": "synchronized_wall", "first_use_ms": self.first_use_ms,
                "comparison": "slowest_candidate_vs_fastest_stock",
                "arm_ms": self.arm_ms}


def measured_route(name, shape, stock, candidate, *, work_bytes, extra_bytes=0, proof=_bits_equal, trial=False, pack=None):
    if extra_bytes and not fits_route(extra_bytes):
        note_route(name, "run_no_fit")
        return False
    device = str(mx.default_device())
    owner, recipe = (pack or _anonymous)(device)
    cell = (owner.directory or owner, recipe, name, tuple(shape))
    with _lock:
        if cell not in _results:
            cache = owner
            may_prove = fits_route(work_bytes)
            if not may_prove:
                if cache.directory is None:
                    note_route(name, "probe_no_fit")
                    return False
                cache = _lookup_only(owner)
            started = perf_counter()
            def prove():
                if not may_prove:
                    return None, Incomplete("probe_no_fit")
                reference = stock()
                mx.eval(reference)
                proposed = candidate()
                mx.eval(proposed)
                return (True, None) if proof(reference, proposed) else (None, Mismatch("unequal"))

            def price(_):
                times = [[], []]
                for arm in (0, 1, 1, 0):
                    start = perf_counter()
                    value = (candidate if arm else stock)()
                    mx.eval(value)
                    mx.synchronize()
                    times[arm].append(1000 * (perf_counter() - start))
                    del value
                ticks = [tuple(max(1, round(ms * 1e6)) for ms in arm) for arm in times]
                return _WallRace(*ticks, 1, arm_ms=tuple(map(tuple, times)), first_use_ms=1000 * (perf_counter() - started))

            key = [recipe, device, *shape]
            prior_hits = cache.hits
            plan, refusal = cache.decide(name, key, prove, race=price)
            if refusal == "probe_no_fit":
                note_route(name, refusal)
                return False
            timing = cache.timing_for(name, key) or {}
            if timing and not timing.get("retry_error"):
                reason = "selected" if plan is not None else "no_gain"
            else:
                reason = "unequal" if refusal == "unequal" else "probe_error"
            _results[cell] = {"install": (cache.identity or {}).get("build"), "pack": recipe, "route": name,
                             "shape": list(shape), "reason": reason, "origin": "disk" if cache.hits > prior_hits else "measured",
                             "arm_ms": timing.get("arm_ms", [[], []]),
                             "first_use_ms": timing.get("first_use_ms", 1000 * (perf_counter() - started))}
            _log.info("Inkling first use %s", _results[cell])
        reason = _results[cell]["reason"]
        use = reason == "selected" or (trial and reason == "no_gain")
        note_route(name, "explicit_trial" if use and trial else reason)
        return use


@lru_cache(maxsize=1)
def _project():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_vocabulary",
        input_names=["states", "weight"],
        output_names=["scores"],
        source="""
            uint lane = thread_index_in_simdgroup;
            uint row = thread_position_in_grid.x / 32;
            uint first = thread_position_in_grid.y * PANEL_ROWS;
            float sums[PANEL_ROWS] = {};
            for (uint column = lane; column < WIDTH; column += 32) {
                float coefficient = float(weight[row * WIDTH + column]);
                for (uint t = 0; t < PANEL_ROWS; ++t)
                    sums[t] = fma(float(states[(first + t) * WIDTH + column]), coefficient, sums[t]);
            }
            for (uint t = 0; t < PANEL_ROWS; ++t) {
                float score = simd_sum(sums[t]);
                if (lane == 0) scores[(first + t) * VOCAB + row] = score;
            }
        """,
    )


def _native(h, weight, vocab, panel):
    batch, rows, width = h.shape
    return _project()(
        inputs=[h, weight],
        template=[("WIDTH", width), ("VOCAB", vocab), ("PANEL_ROWS", panel)],
        grid=(32 * vocab, batch * rows // panel, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, rows, vocab)],
        output_dtypes=[mx.float32],
    )[0]


def _stock_tiles(h, weight, vocab, *, tile_bytes=64 * 1024**2):
    # Each dot retains its complete input width. Overlap the last slab so an
    # unusually short output tail cannot select a different narrow matrix layout.
    capacity = max(1, tile_bytes // (4 * weight.shape[-1]))
    step = min(vocab, 1 << (capacity.bit_length() - 1))
    h = h.astype(mx.float32)
    parts = []
    for start in range(0, vocab, step):
        end = min(start + step, vocab)
        first = min(start, vocab - step)
        scores = h @ weight[first : first + step].astype(mx.float32).T
        tail = scores[..., start - first : end - first]
        mx.eval(tail)
        parts.append(tail)
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)


def _singleton_proof(h, weight, vocab, actual):
    # Products of two FP16/BF16 operands are exact in FP32. Replay the published
    # 32-lane dot on actual vocabulary rows. Require one reduction order across all rows.
    indices = np.unique(np.linspace(0, vocab - 1, min(vocab, 64), dtype=np.int32))
    x = np.asarray(h[0, 0].astype(mx.float32))
    w = np.asarray(weight[mx.array(indices)].astype(mx.float32))
    got = np.asarray(actual[0, 0, mx.array(indices)]).view(np.uint32)
    partial = np.zeros((len(indices), 32), np.float32)
    for start in range(0, len(x), 32):
        v = x[start : start + 32]
        partial[:, : len(v)] += w[:, start : start + len(v)] * v
    if not np.isfinite(partial).all() or not np.any(partial):
        return False
    across, adjacent = partial, partial
    while across.shape[1] > 1:
        half = across.shape[1] // 2
        across = across[:, :half] + across[:, half:]
        adjacent = adjacent[:, ::2] + adjacent[:, 1::2]
    return any(np.array_equal(got, sums[:, 0].view(np.uint32)) for sums in (across, adjacent))


def vocabulary_logits(h, weight, vocab, *, metal=True, pack=None):
    from .precision import AffineWeight

    if isinstance(weight, AffineWeight):
        note_route("head", "affine")
        return weight.product(h.astype(mx.float32))[..., :vocab]
    batch, rows, width = h.shape

    def stock(x):
        if (
            mx.default_device() == mx.gpu
            and width
            and vocab
            and weight.dtype != mx.float32
            and not fits_route(4 * vocab * (width + x.size // width))
        ):
            note_route("head", "bounded_stock")
            return _stock_tiles(x, weight, vocab)
        return x.astype(mx.float32) @ weight[:vocab].astype(mx.float32).T

    if (
        not metal
        or not batch
        or not width
        or not 1 <= rows <= 8
        or weight.ndim != 2
        or not 0 < vocab <= weight.shape[0]
        or width != weight.shape[1]
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or h.dtype not in (mx.float16, mx.bfloat16)
        or weight.dtype != h.dtype
    ):
        note_route("head", "stock_shape_or_device")
        return stock(h)
    one = h[:1, :1]
    ok = measured_route(
        "head.singleton",
        (str(h.dtype), width, vocab),
        lambda: stock(one),
        lambda: _native(one, weight, vocab, 1),
        work_bytes=4 * vocab * (width + 3),
        proof=lambda _, v: _singleton_proof(one, weight, vocab, v),
        pack=pack,
    )
    if not ok:
        return stock(h)

    def original():
        return _native(h, weight, vocab, 1)

    setting = os.environ.get("MTPLX_INKLING_HEAD_PANEL", "auto").lower()
    if rows == 1 or setting not in ("auto", "1", "on"):
        note_route("head.panel", "singleton")
        return original()
    shared = measured_route(
        "head.panel",
        (str(h.dtype), batch, rows, width, vocab),
        original,
        lambda: _native(h, weight, vocab, rows),
        work_bytes=16 * batch * rows * vocab,
        trial=setting in ("1", "on"),
        pack=pack,
    )
    return _native(h, weight, vocab, rows) if shared else original()
