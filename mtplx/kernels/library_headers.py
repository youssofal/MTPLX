"""Kernel header text of the installed MLX, read when a launch of ours is first built.

A launch of ours that stands in for several of the library's must give the library's bits, so the
products and reductions inside it are the library's own code: this module reads the headers MLX installs
beside itself and returns their text for ``mx.fast.metal_kernel`` to compile with the rest of the launch on
this Mac. Nothing of that text lives in this tree, and a new MLX brings its own arithmetic with it.

``text(*parts)`` returns ``(text, None)`` or ``(None, why)``. A part names the headers a caller needs and
the declarations it relies on in them (``PARTS``); the text holds every header the parts reach through
their quoted includes once, includes before includers, in the order the parts are named, without the
headers every custom kernel already starts with (MLX's ``kernels/utils.h`` and what it includes). A part
whose header is missing or does not hold what the part relies on is refused with the reason, counted in
``refusals()``, and the caller keeps the library's own launches.

``generation()`` is the Apple GPU generation of this Mac (``applegpu_g17s`` -> 17; 0 when unknown):
tensor-unit parts serve generation 17 and later.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock

import mlx.core as mx

_KERNELS = "mlx/backend/metal/kernels/"
#: Every custom kernel's source already starts with this header and what it includes.
_PRELUDE = _KERNELS + "utils.h"
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")

#: name -> (headers under the kernels folder, declarations the callers rely on).
PARTS = {
    # one simdgroup's tensor-unit fragments, tiles and tile multiply-accumulate
    "tensor_tiles": (("steel/gemm/nax.h",),
                     ("struct BaseNAXFrag", "struct NAXTile", "tile_matmad_nax", "get_coord", "matmul2d_descriptor")),
    # the tensor-unit GEMM loop over the reduction (each simdgroup's tile of a product, K in order)
    "tensor_gemm": (("steel/gemm/gemm_nax.h",), ("gemm_loop(", "struct NAXTile", "dispatch_bool")),
    # the matrix-vector product
    "matvec": (("gemv.h",), ("struct GEMVKernel", "static METAL_FUNC void run(")),
    # the elementwise operations the unary and binary launches apply
    "elementwise": (("unary_ops.h", "binary_ops.h"), ("struct Divide", "struct Multiply", "struct Tanh")),
}

_lock = Lock()
_refusals: Counter = Counter()


def refusals():
    """Refusals so far, by reason."""
    with _lock:
        return dict(_refusals)


def _root() -> Path:
    return Path(mx.__file__).resolve().parent / "include"


def _walk(root, name, done, out):
    """``name``'s lines with every quoted include it reaches expanded once, in order (``out`` None: only mark
    the headers as present)."""
    if name in done:
        return
    done.add(name)
    for line in (root / name).read_text().splitlines():
        found = _QUOTED_INCLUDE.match(line)
        if found:
            _walk(root, found.group(1), done, out)
        elif out is not None and not _ONCE.match(line):
            out.append(line)


def read(parts, root):
    """``(text, None)`` or ``(None, why)`` for ``parts`` read under ``root`` (uncached, uncounted)."""
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        return None, f"unknown_part:{unknown[0]}"
    for name in (_PRELUDE, *(_KERNELS + h for p in parts for h in PARTS[p][0])):
        if not (root / name).is_file():
            return None, f"missing:{name[len(_KERNELS):]}"
    done, chunks = set(), []
    try:
        _walk(root, _PRELUDE, done, None)
        for part in parts:
            for header in PARTS[part][0]:
                lines = []
                _walk(root, _KERNELS + header, done, lines)
                if lines:
                    chunks.append(f"// {header} and its includes (installed MLX {mx.__version__})\n" + "\n".join(lines))
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    body = "\n".join(chunks) + "\n"
    for part in parts:
        for needle in PARTS[part][1]:
            if needle not in body:
                return None, f"changed:{part}:{needle}"
    return body, None


@lru_cache(maxsize=None)
def text(*parts):
    """The installed headers of ``parts`` as one text, ``(text, None)``; or ``(None, why)``, counted."""
    body, why = read(parts, _root())
    if why is not None:
        with _lock:
            _refusals[why] += 1
    return body, why


@lru_cache(maxsize=1)
def generation():
    """The Apple GPU generation of this Mac (``applegpu_g17s`` -> 17), 0 when unknown."""
    try:
        arch = str(mx.device_info().get("architecture", ""))
    except (AttributeError, RuntimeError, ValueError):
        return 0
    found = re.search(r"applegpu_g(\d+)", arch)
    return int(found.group(1)) if found else 0


__all__ = ["PARTS", "generation", "read", "refusals", "text"]
