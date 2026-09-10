#!/bin/sh
# Reproduce the .venv + the ple_cpu_rows native extension for the Qwen3.8
# Flash-Next aux-lane substrate on upstream/main. CPU only; no GPU.
#
# NOTE (upstream/main rebase, 2026-09-07): the original over100 script built two
# extensions (qsa_sparse_gqa + ple_cpu_rows) and verified them through PR 391's
# mtplx/full_stack_env.py-based stack. On upstream main the QSA native path is
# loaded elsewhere (mtplx/kernels/qsa_prefill_direct.py, extension
# native_extensions/qsa_kernels) and there is no full_stack_env, so this adapted
# script builds only the self-contained ple_cpu_rows extension that the
# ple_cached_aux lane needs and verifies it by importing the built package
# directly. The two decode lanes ARE armed on upstream main: the server
# auto-arms MTPLX_QWEN4_PLE_CACHED_AUX / MTPLX_QSA_POOLED_ROWSEL for a served
# fixed-M4 Flash-Next pack (mtplx/server/openai.py), and the ple_cached_aux lane
# declines to stock with a printed reason when this extension is not built.
#
# Pins mlx 0.32.2 (== production) and an editable mtplx == THIS worktree.
# The mlx 0.32.2 wheel is built with nanobind internals v21. uv.lock pins
# nanobind 2.12.0 (v19), which the native CMake ABI guards reject at configure
# time, so step 2 upgrades nanobind to 2.15.0 (v21) for the native build only.
# mtplx never imports nanobind at runtime, so this does not perturb serving.
#
# The venv python is pinned to 3.12 because mlx 0.32.2 ships cp312 wheels; the
# box's base interpreter is 3.14 with mlx 0.31.2, the wrong ABI for this build.
# The worktree root is derived from this script's own location.
set -eu

WT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WT"

# 1. venv (python 3.12) + locked deps + editable mtplx (base only, --frozen).
nice -n 19 uv sync --frozen --no-dev --python 3.12

# 2. nanobind matching mlx.core's build (v21) for the native extension.
nice -n 19 uv pip install --python .venv/bin/python 'nanobind==2.15.0'

PY="$WT/.venv/bin/python"
# Derive the nanobind cmake root from the venv's actual python version.
NB="$("$PY" -c 'import nanobind,pathlib;print(pathlib.Path(nanobind.__file__).parent)')"

# 3. the ple_cpu_rows native extension, built in-place into its package dir.
#    (CPU-stream PLE rows -- the lazily-imported dep of the ple_cached_aux lane.)
SRC="$WT/native_extensions/ple_cpu_rows"
PKG="mtplx_native_ple_cpu_rows"
rm -rf "$SRC/build"
nice -n 19 cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SRC/$PKG/" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DPython_EXECUTABLE="$PY" \
  -DMTPLX_NANOBIND_DIR="$NB"
nice -n 19 cmake --build "$SRC/build" -j 8

# 4. verify: mlx pin, editable mtplx == this worktree, ple_cpu_rows importable.
"$PY" - "$WT" "$SRC" <<'PYEOF'
import sys
from pathlib import Path

import mlx.core as mx
import mtplx

wt = Path(sys.argv[1]).resolve()
src = Path(sys.argv[2]).resolve()
assert mx.__version__ == "0.32.2", mx.__version__
assert Path(mtplx.__file__).resolve().parents[1] == wt, mtplx.__file__

sys.path.insert(0, str(src))
import mtplx_native_ple_cpu_rows as ext
for sym in ("CachedSidecarProducer", "compute_cached_row_ids",
            "install_cached_sidecar_provider", "make_cpu_rows"):
    assert hasattr(ext, sym), sym
print(
    "over100 venv OK: mlx", mx.__version__,
    "| mtplx", mtplx.__file__,
    "| ple_cpu_rows native available",
)
PYEOF
