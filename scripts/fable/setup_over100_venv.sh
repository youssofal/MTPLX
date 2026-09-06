#!/bin/sh
# Reproduce the .venv + native extensions for the Qwen3.8 Flash-Next over-100
# work so the served full stack (incl. the QSA sparse-decode lane and the
# stacked cached-PLE auxiliary lane) can run the 16,384-token canonical cell.
# CPU only; no GPU.
#
# Pins mlx 0.32.2 (== production) and an editable mtplx == THIS worktree, so
# `python -m mtplx.server.openai` with the worktree on cwd imports this
# branch's code. Base deps only, matching the production venv (no server extra /
# llguidance; mtplx.server does not import it).
#
# The mlx 0.32.2 wheel is built with nanobind internals v21. uv.lock still pins
# nanobind 2.12.0 (v19), which the native CMake ABI guards reject at configure
# time, so step 2 upgrades nanobind to 2.15.0 (v21) for the native builds only.
# mtplx never imports nanobind at runtime, so this does not perturb serving.
#
# The worktree root is derived from this script's own location, so it is
# correct in any checkout and carries no hard-coded home path.
set -eu

WT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WT"

# 1. venv + locked deps + editable mtplx (base only, --frozen keeps uv.lock).
nice -n 19 uv sync --frozen --no-dev

# 2. nanobind matching mlx.core's build (v21) for the native extensions.
nice -n 19 uv pip install --python .venv/bin/python 'nanobind==2.15.0'

PY="$WT/.venv/bin/python"
NB="$WT/.venv/lib/python3.12/site-packages/nanobind"

# 3. native extensions, each built into its own package dir in-place.
#    a. QSA sparse-decode (MTPLX_FABLE_QSA_SPARSE_DECODE).
#    b. CPU-stream PLE rows (the stacked ple_cached_aux lane).
for EXT in qsa_sparse_gqa:mtplx_native_qsa ple_cpu_rows:mtplx_native_ple_cpu_rows; do
  DIR="${EXT%%:*}"
  PKG="${EXT##*:}"
  SRC="$WT/native_extensions/$DIR"
  rm -rf "$SRC/build"
  nice -n 19 cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SRC/$PKG/" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DPython_EXECUTABLE="$PY" \
    -DMTPLX_NANOBIND_DIR="$NB"
  nice -n 19 cmake --build "$SRC/build" -j 8
done

# 4. verify: mlx pin, editable mtplx == this worktree, both native exts usable.
"$PY" - "$WT" <<'PYEOF'
import sys
from pathlib import Path

import mlx.core as mx
import mtplx
from mtplx.native import (
    native_qsa_available,
    _nanobind_abi_mismatch,
    native_ple_cpu_rows_available,
    _ple_cpu_rows_abi_mismatch,
)

wt = Path(sys.argv[1]).resolve()
assert mx.__version__ == "0.32.2", mx.__version__
assert Path(mtplx.__file__).resolve().parents[1] == wt, mtplx.__file__
assert native_qsa_available() and _nanobind_abi_mismatch() is None, "QSA native ext not usable"
assert (
    native_ple_cpu_rows_available() and _ple_cpu_rows_abi_mismatch() is None
), "PLE cpu-rows native ext not usable"
print(
    "over100 venv OK: mlx", mx.__version__,
    "| mtplx", mtplx.__file__,
    "| qsa + ple_cpu_rows native available",
)
PYEOF
