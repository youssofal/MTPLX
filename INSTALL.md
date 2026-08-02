# Install MTPLX

MTPLX is production software for Apple Silicon Macs, distributed via pip, Homebrew, and a signed DMG.

## Requirements

- Apple Silicon Mac
- Python 3.11+
- macOS with MLX support
- Enough disk for the selected model

## Install

Recommended macOS install:

```bash
curl -fsSL https://raw.githubusercontent.com/youssofal/MTPLX/main/scripts/install_macos.sh | bash
mtplx help
```

The installer checks Homebrew Python paths directly, so it works even if a fresh
Terminal tab has not put `/opt/homebrew/bin` on PATH yet. It installs MTPLX from
PyPI into `~/.mtplx/venv` and writes a durable launcher at `~/.local/bin/mtplx`.
On Apple Silicon Homebrew installs, it also writes `/opt/homebrew/bin/mtplx` when
that directory is writable.

Python-only install:

```bash
python3 -m pip install -U mtplx
mtplx help
```

For local development:

```bash
python -m pip install -e ".[dev,server]"
```

## Runtime Dependencies

`mtplx --help`, `mtplx doctor`, `mtplx inspect`, and `mtplx init` are designed to work even before MLX is installed. Generation and serving require MLX and a verified model.

MTPLX runs on stock PyPI MLX; no fork is required for any profile (the legacy `--strict-mlx-fork-assert` flag is a deprecated no-op).

## Optional Thermal Tools

`--max` is opt-in. It is for users who need sustained throughput and accept fan noise. It is never part of the default quick start and is never used for no-fan product claims.

Check the local thermal-control state:

```bash
mtplx max --status
```

If ThermalForge or TG Pro is not present, MTPLX prints install instructions and continues without fan control for `run`, `chat`, and `serve --max`. It must not silently enable spin-loop or clock-anchor modes.

Supported public commands:

```bash
mtplx max --on       # Performance profile
mtplx max --max      # Max profile
mtplx max --off      # Silent profile
mtplx max --status   # tool/status report
```

`MTPLX_GPU_CLOCK_ANCHOR=1` is an explicit experimental diagnostic only. Do not use it for README, release, or product benchmark claims.
