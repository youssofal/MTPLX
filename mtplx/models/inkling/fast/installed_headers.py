"""Kernel header text of the installed MLX, read when one of our launches is first built.

Some of our prompt-width launches multiply on the tensor units with MLX's own tile, fragment and GEMM-loop
templates, so that every product they compute has the sums MLX's own kernel gives it. Those templates are read
here from the headers MLX installs beside itself and handed to ``mx.fast.metal_kernel`` with the rest of the
launch: nothing of their text lives in this tree, and a new MLX brings its own arithmetic with it. A header
this reader cannot find or does not recognise makes the caller keep MLX's own launches (``text`` returns the
reason instead).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import mlx.core as mx

_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")
#: Every custom kernel's source already starts with this header (and what it includes).
_KERNEL_PRELUDE = "mlx/backend/metal/kernels/utils.h"


def _lines(root, name, done, out):
    """``name``'s lines with each quoted include it reaches expanded once, in order; ``done`` holds the headers
    already present (they and their includes are skipped)."""
    if name in done:
        return
    done.add(name)
    for line in (root / name).read_text().splitlines():
        found = _QUOTED_INCLUDE.match(line)
        if found:
            _lines(root, found.group(1), done, out)
        elif out is not None and not _ONCE.match(line):
            out.append(line)


@lru_cache(maxsize=None)
def text(names, needles=()):
    """(text, None): the installed headers ``names`` (a tuple, in order) with their includes expanded once and
    ``using namespace mlx::steel`` after them; or (None, why) when one is missing or lacks a ``needle``."""
    try:
        root = Path(mx.__file__).resolve().parent / "include"
        for name in (_KERNEL_PRELUDE, *names):
            if not (root / name).is_file():
                return None, f"the installed MLX ships no {name}"
        done = set()
        _lines(root, _KERNEL_PRELUDE, done, None)  # present already: only mark it and its includes
        out = []
        for name in names:
            _lines(root, name, done, out)
    except Exception as exc:  # noqa: BLE001 - an unreadable install keeps MLX's own launches
        return None, f"{type(exc).__name__}: {exc}"
    body = "\n".join(out)
    for needle in needles:
        if needle not in body:
            return None, f"the installed headers have no {needle}"
    return body + "\nusing namespace mlx::steel;\n", None
