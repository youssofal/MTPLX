"""Prove a kernel once on the first shape that reaches it, and reuse the proof where it holds.

A Metal kernel replaces part of a model's graph only after it has been compared with the graph it
replaces, on the Mac, the build and the weights in front of it.  That comparison costs a little at
first use; ``ProofCache`` keeps its result on disk, so the same pack on the same Mac and build pays
for it once.

    proofs = ProofCache.for_pack(pack_folder)
    plan, reason = proofs.decide("my_kernel", shape, prove)

``prove()`` returns ``(plan, None)`` when the kernel may serve this shape and ``(None, reason)`` when
it may not.  A plan is anything JSON can hold.  What is kept:

* a plan, and a refusal whose reason is a ``Mismatch`` (the comparison finished and the kernel was
  wrong), are saved and reused on every later load;
* any other refusal, an ``Incomplete`` comparison, or an exception inside ``prove`` lasts one model
  load and is checked again on the next.

A record is bound to an identity: this package's sources and the MLX Metal headers they build
against, the mtplx, MLX and mlx-lm versions, the OS, the GPU and its architecture (with any
override), and every weight file of the pack.  Change one and the proof is taken again.  Records
live under ``~/.mtplx/cache/kernel-proofs`` (``MTPLX_KERNEL_PROOF_DIR`` moves them) and expire
thirty days after their build was last used; when the folder cannot be written the proofs are kept
in memory and every kernel still gets its live check.

``decide(..., race=timer)`` also prices the kernel: ``timer(plan)`` returns the GPU time in
nanoseconds of the stock graph and of the kernel (two integers, or a ``GpuRace`` of several
samples each), and the kernel serves only when it is faster by a margin.  This module ships no
timer of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from threading import RLock

from mtplx import __version__

log = logging.getLogger(__name__)


# A busy GPU may vary both arms while still pricing every replay of one below the other.
_RACE_MARGIN_PERCENT = 5
_OVERLAP_RETRY_SECONDS = 24 * 60 * 60


def _race_decision(stock, candidate):
    """A separated sub-margin price is a tie. Touching or crossing ranges need another visit."""
    stock_min, candidate_min = min(stock), min(candidate)
    if max(candidate) < stock_min:
        return "win" if candidate_min * 100 <= stock_min * (100 - _RACE_MARGIN_PERCENT) else "tie"
    if max(stock) < candidate_min:
        return "loss" if candidate_min * 100 >= stock_min * (100 + _RACE_MARGIN_PERCENT) else "tie"
    return "inconclusive"


@dataclass(frozen=True)
class GpuRace:
    stock: tuple[int, ...]
    candidate: tuple[int, ...]
    repeats: int
    stock_encode: tuple[int, ...] = ()
    candidate_encode: tuple[int, ...] = ()
    launches: tuple[int, int] | None = None

    def priced_arms(self):
        if not self.stock_encode:
            return self.stock, self.candidate
        return (tuple(gpu + cpu for gpu, cpu in zip(self.stock, self.stock_encode)),
                tuple(gpu + cpu for gpu, cpu in zip(self.candidate, self.candidate_encode)))

    def __iter__(self):
        return iter((min(self.stock), min(self.candidate)))

    def receipt(self):
        fields = {"rounds": len(self.stock), "dispatch_repeats": self.repeats, "price_metric": "gpu",
                  "stock_gpu_max_ns": max(self.stock), "candidate_gpu_max_ns": max(self.candidate)}
        if self.launches is not None:
            fields["stock_launches"], fields["candidate_launches"] = self.launches
        if self.stock_encode:
            stock, candidate = self.priced_arms()
            fields.update(price_metric="encode_plus_gpu", stock_encode_gpu_ns=min(stock),
                          candidate_encode_gpu_ns=min(candidate), stock_encode_gpu_max_ns=max(stock),
                          candidate_encode_gpu_max_ns=max(candidate))
        return fields


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_digest(*roots: Path) -> str:
    """Hash executable package sources, native kernels and the Metal headers they include."""
    digest = hashlib.sha256()
    extensions = {".py", ".metal", ".h", ".hpp", ".cpp", ".mm", ".so", ".metallib", ".dylib"}
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in extensions)
        for source in files:
            digest.update(str(source.relative_to(root.parent)).encode() + b"\0")
            digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


class Mismatch(str):
    """A completed comparison refuted the cut. Other refusals last one model load."""


class Incomplete(str):
    """A comparison could not finish, including inside a multi-stage proof."""


def _pack_file(path: Path, root: Path):
    name = str(path.relative_to(root))
    if path.is_symlink():
        blob = path.resolve(strict=True)
        if blob.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", blob.name):
            return (name, blob.name)
    stat = path.stat()
    return (name, stat.st_size, stat.st_mtime_ns)


def _write_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as tmp:
        try:
            json.dump(value, tmp, sort_keys=True)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def _prune(root, active):
    """Expire self-identifying directories and checksummed records from older builds."""
    cutoff = time.time() - 30 * 86400
    for folder in root.iterdir():
        if folder == active or folder.is_symlink():
            continue
        try:
            if re.fullmatch(r"[0-9a-f]{64}\.json", folder.name):
                if not folder.is_file() or folder.stat().st_mtime >= cutoff:
                    continue
                saved = json.loads(folder.read_bytes())
                if set(saved) not in ({"result", "checksum"}, {"result", "race", "checksum"}):
                    continue
                inputs = [folder.stem, saved["result"]]
                if "race" in saved:
                    inputs.append(saved["race"])
                if _digest(inputs) == saved["checksum"]:
                    folder.unlink(missing_ok=True)
                continue
            if not re.fullmatch(r"[0-9a-f]{64}", folder.name) or not folder.is_dir():
                continue
            marker = folder / "identity.json"
            if marker.is_symlink() or marker.stat().st_mtime >= cutoff:
                continue
            if _digest(json.loads(marker.read_bytes())["identity"]) != folder.name:
                continue
            files = list(folder.iterdir())
            if any(f.is_symlink() or not f.is_file() or
                   (f != marker and not re.fullmatch(r"[A-Za-z0-9_.-]+-[0-9a-f]{64}\.json", f.name))
                   for f in files):
                continue
            for f in files:
                f.unlink(missing_ok=True)
            folder.rmdir()
        except (OSError, KeyError, ValueError, TypeError):
            continue  # another boot may already have collected the directory


class ProofCache:
    """One cache per loaded model. Disk records identify the checked code, operands and GPU."""

    def __init__(self, identity, directory, *, pack_path=None, identity_error=None):
        self.identity = identity
        self.directory = Path(directory) / _digest(identity) if directory is not None else None
        self.pack_path = str(pack_path) if pack_path is not None else None
        self.identity_error = identity_error
        self._memo = {}
        self._races = {}
        self._lock = RLock()
        self._prepared = False
        self.hits = 0
        self.checks = 0

    @classmethod
    def for_pack(cls, path: Path):
        """Local files survive a remount and content-addressed shards survive a cache move."""
        try:
            import mlx.core as mx
            from mtplx.artifacts import expected_mtp_file
            from mtplx.nax_detect import gpu_family_fallback_forced

            info = mx.device_info(mx.gpu)
            weights = set(path.rglob("*.safetensors"))
            sidecar = expected_mtp_file(path)  # rglob does not descend through a linked sidecar directory
            if sidecar.is_file():
                weights.add(sidecar)
            if not weights or not info.get("architecture") or not info.get("device_name"):
                raise ValueError("missing GPU identity or weight shards")
            metadata = [path / name for name in ("config.json", "model.safetensors.index.json", "mtplx_runtime.json")]
            files = [_pack_file(p, path) for p in [*sorted(weights), *(p for p in metadata if p.is_file())]]
            package = Path(__file__).resolve().parents[1]
            roots = [package, Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels"]
            if (package.parent / "pyproject.toml").is_file():
                roots.append(package.parent / "native_extensions")
            identity = {
                "schema": 3, "build": _source_digest(*roots), "mtplx": __version__,
                "mlx": mx.__version__, "mlx_lm": version("mlx-lm"), "os": platform.release(),
                "device": info["device_name"], "architecture": info["architecture"],
                "architecture_override": os.getenv("MLX_METAL_GPU_ARCH"),
                "rehearsal": gpu_family_fallback_forced(), "files": files,
            }
            root = os.getenv("MTPLX_KERNEL_PROOF_DIR") or Path.home() / ".mtplx/cache/kernel-proofs"
            return cls(identity, root, pack_path=path)
        except Exception as exc:  # identity metadata is optional; every shape still gets its live proof
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("Using in-memory kernel proofs for %s: %s", path, reason)
            return cls(None, None, pack_path=path, identity_error=reason)

    def _prepare(self):
        if self._prepared or self.directory is None:
            return
        self._prepared = True
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _write_json(self.directory / "identity.json", {"identity": self.identity, "pack_path": self.pack_path})
            _prune(self.directory.parent, self.directory)
        except OSError as exc:
            self.identity_error = str(exc)
            log.warning("Using in-memory kernel proofs for %s: %s", self.pack_path, exc)
            self.directory = None

    def race_report(self, name):
        """Per-shape GPU minima and their decision, including results reused from disk."""
        return list(self._races.get(name, {}).values())

    def timing_for(self, name, shape):
        return self._races.get(name, {}).get(_digest([name, shape, True]))

    def decide(self, name, shape, prove, *, race=None):
        """Return (plan, reason). prove() returns that pair. Optional race(plan) returns two GPU minima in ns.

        Completed prices survive a reload. Overlapping samples can be retried once a day.
        A first probe or timer that fails without a measurement is retried on the next install.
        """
        with self._lock:
            return self._decide(name, shape, prove, race)

    def _decide(self, name, shape, prove, race):
        key = _digest([name, shape, bool(race)])
        if key in self._memo:
            return self._memo[key]
        self._prepare()
        path = self.directory / f"{name}-{key}.json" if self.directory and re.fullmatch(r"[A-Za-z0-9_.-]+", name) else None
        expired_race = None
        if path is not None:
            try:
                saved = json.loads(path.read_bytes())
                result, timing = saved["result"], saved["race"]
                if (saved["checksum"] != _digest([self.identity, key, result, timing]) or len(result) != 2 or
                        (timing is not None and not isinstance(timing, dict))):
                    raise ValueError("proof checksum or result layout differs")
                retry_after = timing.get("retry_after") if timing else None
                if retry_after is None or time.time() < retry_after:
                    if timing is not None:
                        self._races.setdefault(name, {})[key] = {**timing, "from_cache": True}
                    self.hits += 1
                    self._memo[key] = tuple(result)
                    return self._memo[key]
                expired_race = timing
            except FileNotFoundError:
                pass
            except (OSError, KeyError, TypeError, ValueError) as exc:
                log.warning("Checking %s again after reading %s: %s", name, path, exc)

        self.checks += 1
        timing = None
        try:
            plan, reason = prove()
            if (plan is None) == (reason is None):
                raise ValueError("a proof must return either a plan or a decline reason")
            persist = plan is not None or isinstance(reason, Mismatch)
            if plan is not None and race is not None:
                started = time.perf_counter()
                samples = race(plan)
                elapsed = time.perf_counter() - started
                stock, candidate = samples
                if any(type(t) is not int or t <= 0 for t in (stock, candidate)):
                    raise ValueError("missing Metal command-buffer timestamps")
                arms = ((stock,), (candidate,))
                details = {}
                if isinstance(samples, GpuRace):
                    arms = samples.priced_arms()
                    details = samples.receipt()
                decision = _race_decision(*arms)
                timing = {"shape": shape, "stock_gpu_ns": stock, "candidate_gpu_ns": candidate,
                          "decision": decision, "confirmed": decision != "inconclusive", "measured_s": elapsed, **details}
                if decision == "inconclusive":
                    timing["retry_after"] = time.time() + _OVERLAP_RETRY_SECONDS
                self._races.setdefault(name, {})[key] = {**timing, "from_cache": False}
                if decision != "win":
                    stock_price, candidate_price = map(min, arms)
                    metric = "encode plus GPU" if details.get("price_metric") == "encode_plus_gpu" else "GPU"
                    plan, reason = None, f"{metric} minimum {candidate_price} ns vs stock {stock_price} ns ({decision})"
        except Exception as exc:  # GPU or compiler failure is a verdict for this load only
            plan, reason = None, f"{name} first-use check raised {type(exc).__name__}: {exc}"
            persist = False

        if expired_race is not None and not persist:
            # A failed refresh must not charge the next launch for today's retry again.
            timing = {**expired_race, "retry_after": time.time() + _OVERLAP_RETRY_SECONDS, "retry_error": str(reason)}
            self._races.setdefault(name, {})[key] = {**timing, "from_cache": True}
            persist = True
        self._memo[key] = (plan, reason)
        if persist and path is not None:
            result = [plan, reason]
            try:
                _write_json(path, {"cut": name, "shape": shape, "result": result, "race": timing,
                                   "checksum": _digest([self.identity, key, result, timing])})
            except OSError as exc:
                log.warning("Could not save %s proof at %s: %s", name, path, exc)
        return self._memo[key]
