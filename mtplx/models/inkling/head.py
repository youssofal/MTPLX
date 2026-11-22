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
from . import launch_limits
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


@lru_cache(maxsize=8)
def _memory_size(device):
    info = mx.device_info(mx.default_device())
    return int(info.get("memory_size") or info.get("total_memory") or 0)


#: An engine limit above this share of RAM is not one the server set (MLX's own default is 1.5 working sets).
_SET_LIMIT_SHARE = 0.8


def room_line():
    """What MLX may hold before a route's extra memory is declined: the GPU's recommended working set, or the
    engine's memory limit where the server set it higher. macOS recommends two thirds of RAM to the GPU up to
    36 GB and the memory plan gives the engine three quarters, so on a 24 GB Mac with Bare Speed (15.5 GiB
    resident at a read) the working set left no room for the one-launch prompt attention past a read's first
    chunk, though the plan, the admission and the per-chunk check all hold the engine to its 18 GiB (a cold
    8K read on a simulated 24 GB Mac served 36 of 144 calls on it, 16K and 24K 16 and 14 of 48 in the first
    chunk and none after; this Mac as itself, every call). The plan's own reads go above the working set by
    design (the 2.13.1 24 GB read peaks at 16.8 GiB), so the line is the limit the engine is held to."""
    device = str(mx.default_device())
    line = _working_set(device)
    try:
        limit = int(mx.get_memory_limit())
    except Exception:  # noqa: BLE001 - an MLX without the call keeps the working set
        return line
    memory = _memory_size(device)
    if memory and limit <= _SET_LIMIT_SHARE * memory:
        return max(line, limit)
    return line


def fits_route(extra):
    return mx.get_active_memory() + extra + 128 * 1024**2 < room_line()


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


_PROJECT = """
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
        """


@lru_cache(maxsize=1)
def _project():
    return launch_limits.kernel(
        name="mtplx_inkling_vocabulary",
        input_names=["states", "weight"],
        output_names=["scores"],
        source=_PROJECT,
    )


@lru_cache(maxsize=1)
def _project_capped():
    # The same launch with the soft cap applied to each finished score before it is written: the
    # library's own Divide, Tanh and Multiply, read from the installed headers (fast/library_gemm.py),
    # so a score is tanh(score / cap) * cap as MLX's three elementwise launches give it.
    from .fast import library_gemm

    ops, why = library_gemm.library("elementwise")
    if ops is None:
        raise RuntimeError(f"no soft-capped head here: {why}")
    finish = ("#define FINISH(v) Multiply()(Tanh()(Divide()((v), as_type<float>(uint(CAP)))), "
              "as_type<float>(uint(CAP)))\n")
    return launch_limits.kernel(
        name="mtplx_inkling_vocabulary_capped",
        input_names=["states", "weight"],
        output_names=["scores"],
        source=_PROJECT.replace("= score;", "= FINISH(score);"),
        header=ops + finish,
    )


def _native(h, weight, vocab, panel, cap=None):
    batch, rows, width = h.shape
    template = [("WIDTH", width), ("VOCAB", vocab), ("PANEL_ROWS", panel)]
    if cap:
        template.append(("CAP", int(np.array(cap, np.float32).view(np.uint32))))
    return (_project_capped() if cap else _project())(
        inputs=[h, weight],
        template=template,
        grid=(32 * vocab, batch * rows // panel, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, rows, vocab)],
        output_dtypes=[mx.float32],
    )[0]


_capped_verdicts: dict = {}


def _capped(h, weight, vocab, panel, cap):
    """The native launch with the soft cap inside it, or None where it refuses: its first use per shape
    on this Mac must equal, value for value, the same launch uncapped and capped by MLX's operations
    (``tanh(logits / cap) * cap``, as ``Model.head`` caps). A proof of its own, apart from the uncapped
    route's, so a refusal here leaves that route as it was."""
    try:
        out = _native(h, weight, vocab, panel, cap)
    except RuntimeError:
        note_route("head.capped", "unavailable")
        return None
    key = (str(h.dtype), tuple(h.shape), tuple(weight.shape), vocab, panel, float(cap))
    verdict = _capped_verdicts.get(key)
    if verdict is None:
        want = mx.tanh(_native(h, weight, vocab, panel) / cap) * cap
        mx.eval(out, want)
        verdict = _bits_equal(out, want)
        _capped_verdicts[key] = verdict
        note_route("head.capped", "proved" if verdict else "unequal")
    if not verdict:
        return None
    note_route("head.capped", "served")
    return out


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
    return _logits(h, weight, vocab, metal, pack, None)[0]


def head_logits(h, weight, vocab, cap, *, pack=None):
    """``Model.head``'s logits: ``vocabulary_logits`` and then the soft cap ``tanh(logits / cap) * cap``
    (``cap`` 0 or None: none). On the native route the cap is applied inside the vocabulary launch
    (one launch where the head takes four), every value equal to the launch capped by MLX's
    operations (proven per shape on this Mac's first use); elsewhere MLX's operations cap.
    Returns ``(logits, route)``: route "native" (the launch served, cap included), "capped_by_mlx"
    (the launch served, the capped launch refused) or "stock" (MLX's product)."""
    logits, route = _logits(h, weight, vocab, True, pack, cap or None)
    if cap and route != "native":
        logits = mx.tanh(logits / cap) * cap
        route = "capped_by_mlx" if route == "uncapped" else route
    return logits, route


def _logits(h, weight, vocab, metal, pack, cap):
    """``(logits, route)``: route "native" (the vocabulary launch; capped inside it when ``cap``),
    "uncapped" (the launch served, its capped form refused: the caller caps) or "stock"."""
    from .precision import AffineWeight

    if isinstance(weight, AffineWeight):
        note_route("head", "affine")
        return weight.product(h.astype(mx.float32))[..., :vocab], "stock"
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
        return stock(h), "stock"
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
        return stock(h), "stock"

    def original():
        return _native(h, weight, vocab, 1)

    setting = os.environ.get("MTPLX_INKLING_HEAD_PANEL", "auto").lower()
    if rows == 1 or setting not in ("auto", "1", "on"):
        note_route("head.panel", "singleton")
        panel = 1
    else:
        shared = measured_route(
            "head.panel",
            (str(h.dtype), batch, rows, width, vocab),
            original,
            lambda: _native(h, weight, vocab, rows),
            work_bytes=16 * batch * rows * vocab,
            trial=setting in ("1", "on"),
            pack=pack,
        )
        panel = rows if shared else 1
    if cap:
        capped = _capped(h, weight, vocab, panel, cap)
        if capped is not None:
            return capped, "native"
        return _native(h, weight, vocab, panel), "uncapped"
    return _native(h, weight, vocab, panel), "native"
