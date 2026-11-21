"""The installed MLX's own dense product code (matrix-unit tile loop, mat-vec), read when a launch of
ours is built, so the products inside that launch give the library's bits.

A launch of ours that replaces several of the library's launches (a product, a reduction, another
product) must leave every value as the library's launches would. The arithmetic of the products is
therefore the library's own: this module reads its headers from the folder MLX installs beside
itself, joins each one with the headers it includes (once each, in include order), and hands the text
to ``mx.fast.metal_kernel`` as the launch's header, where the compiler builds it together with the
launch on this Mac. Nothing of that text lives in this tree, and a new MLX brings its own arithmetic.

``library(*parts)`` returns ``(text, None)`` or ``(None, why)``: the named parts' headers and the
headers they include, each once (two parts that include the same header share it), in include order.
A header that is missing or does not hold the functions a caller needs makes the caller keep the
library's own launches, counted. The headers every custom kernel already carries (MLX's ``utils.h``
and what it includes) are left out. Parts:

* ``"matrix_units"``: ``mlx::steel::gemm_loop`` (one simdgroup's tile of a product on the matrix
  units, the K axis in order), ``NAXTile`` and ``dispatch_bool``: what the library's split-K and
  regular products run inside each simdgroup on Macs with matrix units.
* ``"matvec"``: ``GEMVKernel`` (the library's mat-vec, ``run`` a static device function).
* ``"elementwise"``: the library's elementwise operations (``Divide``, ``Multiply``, ``Tanh``, ...),
  as its unary and binary launches apply them.

``product_tiles(m, n, k)`` states the library's choice of kernel and tiles for a float32 product of an
``(m, k)`` matrix with an ``(n, k)`` matrix transposed on this Mac's GPU family, as its dispatch makes
it for the small shapes the callers use; first-use proofs on real operands confirm every shape.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import mlx.core as mx

_BASE = "mlx/backend/metal/kernels/"
_QUOTED = re.compile(r'^[ \t]*#include[ \t]+"(mlx/backend/metal/kernels/[^"]+)"[ \t]*$', re.M)
# Every mx.fast.metal_kernel source already carries these (MLX's kernels/utils.h and its includes).
_CARRIED = frozenset(_BASE + n for n in ("utils.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h", "logging.h"))


def _root() -> Path:
    return Path(mx.__file__).resolve().parent / "include"


_PARTS = {
    "matrix_units": (("steel/gemm/gemm_nax.h",),
                     ("auto gemm_loop(", "struct NAXTile", "dispatch_bool", "MetalPerformancePrimitives")),
    "matvec": (("gemv.h",), ("struct GEMVKernel", "static METAL_FUNC void run(")),
    "elementwise": (("binary_ops.h", "unary_ops.h"), ("struct Divide", "struct Multiply", "struct Tanh")),
}


def _joined(entries) -> str:
    """The entries and the library headers they include, each once, includes before includers."""
    done: set[str] = set()
    parts: list[str] = []

    def visit(rel: str) -> None:
        if rel in done or rel in _CARRIED:
            return
        done.add(rel)
        text = (_root() / rel).read_text()
        for match in _QUOTED.finditer(text):
            visit(match.group(1))
        parts.append(f"// {rel} (installed MLX {mx.__version__})\n" + _QUOTED.sub("", text).replace("#pragma once", ""))

    for entry in entries:
        visit(entry)
    return "\n".join(parts)


@lru_cache(maxsize=None)
def library(*names: str) -> tuple[Optional[str], Optional[str]]:
    """The joined text of the named parts (``_PARTS``), or the reason it cannot serve."""
    try:
        entries = [_BASE + entry for name in names for entry in _PARTS[name][0]]
        text = _joined(entries)
    except KeyError as exc:
        return None, f"no library part {exc}"
    except (OSError, ValueError) as exc:
        return None, f"the installed MLX has no readable kernel headers ({type(exc).__name__}: {exc})"
    for name in names:
        for needle in _PARTS[name][1]:
            if needle not in text:
                return None, f"the installed MLX's {name} part has no {needle!r}"
    return text, None


class Tiles(NamedTuple):
    """The library's kernel for a small float32 product ``(m, k) x (n, k)^T`` on this Mac."""

    #: "matvec" (one of m, n is 1), "split_k" (matrix units, the K axis cut into partitions).
    kind: str
    #: split_k: K values a partition (the last takes the rest), partitions, K block of the tile loop.
    partition: int = 0
    partitions: int = 0
    block_k: int = 0


def gpu_generation() -> int:
    """The Apple GPU generation of this Mac (``applegpu_g17s`` -> 17), 0 when unknown."""
    try:
        arch = str(mx.device_info().get("architecture", ""))
    except (AttributeError, RuntimeError, ValueError):   # an unknown device takes the library's launches
        arch = ""
    match = re.search(r"applegpu_g(\d+)", arch)
    return int(match.group(1)) if match else 0


def product_tiles(m: int, n: int, k: int) -> tuple[Optional[Tiles], Optional[str]]:
    """``(tiles, None)`` for the shapes the depth reads use, else ``(None, why)``."""
    if min(m, n) == 1:
        return Tiles("matvec"), None
    if gpu_generation() < 17:
        return None, "no matrix units: the library's product here is its simdgroup-matrix form (not twinned yet)"
    if m > 32 or n > 64 or k < 3 * max(m, n) or k > 4096:
        return None, f"shape {m}x{n}x{k} is outside the small split-K products twinned here"
    # Small split-K tiles: 64 x 64 a threadgroup (2 x 2 simdgroups of 32 x 32), K blocks of 256,
    # partitions of K/2 (K <= 1024), 1,024 (K <= 2,048) or 2,048 values (K <= 4,096).
    part = k // 2 if k <= 1024 else (1024 if k <= 2048 else 2048)
    return Tiles("split_k", part, -(-k // part), 256), None


__all__ = ["Tiles", "gpu_generation", "library", "product_tiles"]
