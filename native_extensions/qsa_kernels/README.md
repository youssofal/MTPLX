# mtplx_qsa_kernels — Qwen4 QSA sparse-GQA Metal kernel

Optional native extension behind `mtplx/kernels/qsa_prefill_direct.py`. It is
the only fast QSA prefill consumer an M3-class GPU can reach: the existing
`qsa_prefill_flash` kernel needs Metal 4 TensorOps (G17 / NAX), which M3 does
not have.

Vendored from oMLX PR #3244 at revision
`dc312e6e905e03d21ef0c4a86289cbfa2cf857cc`. See
`mtplx_qsa_kernels/NOTICE` for provenance, MTPLX modifications, and licenses.

Nothing in MTPLX requires this module. Without it the import stays green and
the direct lane reports unsupported.

## Build (on the target machine, in the venv that will run MTPLX)

This extension links MLX's **private C++ ABI** and compiles against the Steel
Metal headers shipped in the mlx wheel. It is not portable between wheels.
Build it in the same interpreter that serves.

```sh
# 1. Requirements: full Xcode (working `xcrun -sdk macosx metal`), CMake >= 3.27.
xcrun -sdk macosx metal --version

# 2. Exact ABI pins. nanobind MUST match the version the mlx wheel was built
#    with; a mismatch imports cleanly and then rejects every mx.array.
python -m pip install "nanobind==2.15.0"
python -c "import mlx.core as mx; print('mlx', mx.__version__)"

# 3. Build in place (puts _ext*.so and mtplx_qsa_kernels.metallib inside the
#    mtplx_qsa_kernels/ package directory, where the wrapper finds them).
cd native_extensions/qsa_kernels
python setup.py build_ext --inplace

# 4. Prove the artifact, not the symbol.
python - <<'PY'
import mtplx_qsa_kernels as ext
print(ext.BUILT_AGAINST_MLX, ext.BUILT_AGAINST_NANOBIND, ext.METAL_LIBRARY)
from mtplx.kernels.qsa_prefill_direct import (
    qsa_prefill_direct_build_info, qsa_prefill_direct_preflight)
print(qsa_prefill_direct_build_info())
print("pipeline ok:", qsa_prefill_direct_preflight())
PY
```

`otool -L mtplx_qsa_kernels/_ext*.so` must resolve `libmlx.dylib` to the mlx
wheel in this venv, not to a build-tree absolute path.

## Things that break it

* **nanobind drift.** `nanobind>=2` at the MTPLX root is a floor, not a pin.
  The exact version is what matters; `abi_probe` catches a mismatch at import
  and disables the lane with one warning.
* **mlx upgrade.** Any `pip install -U mlx`, even inside MTPLX's
  `>=0.32.2,<0.33` runtime range, invalidates this build. The wrapper detects
  it: `BUILT_AGAINST_MLX` is compared against the imported version at
  readiness, and a mismatch warns once and disables the lane, so the symptom
  is "the direct lane stopped engaging", not a crash. Rebuild here.
  `pyproject.toml` pins the BUILD to `mlx==0.32.2` so a PEP 517 isolated
  build cannot quietly pick a different 0.32.x than the serving venv holds.
* **Metallib name or location.** The C++ resolves `mtplx_qsa_kernels` from the
  directory of the loaded `.so`. Renaming or moving either one turns a symbol
  that imports fine into an `mx.eval`-time pipeline failure. That is what
  `qsa_prefill_direct_preflight()` exists to catch before traffic. A failed
  proof — from the preflight or from the first real dispatch — retires the
  lane for the whole process, so `qsa_prefill_direct_ready()` goes False, the
  M3 producer auto-gate disarms, and traffic routes to the gather tier
  instead of re-hitting the same wall every request.
* **Retuning `WM`.** It is part of the compiled kernel name and its MMA
  layout, and the M2/M3 threadgroup ceiling is 896 for some kernels, not 1024.
  Both the Metal header and the C++ carry a `static_assert(WM * 32 <= 896)`.
  It is not a Python tuning knob.

## Scope

Only `(BK, DC) = (64, 64)` is instantiated, for fp16 and bf16 — the
specialization oMLX's production glue uses and the one that was measured on
M3. The C++ `unsupported()` was narrowed to match, so the Python support
check, the C++ check, and the packaged metallib all describe the same set.
