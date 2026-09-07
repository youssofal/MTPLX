# Quickstart

```bash
brew install youssofal/mtplx/mtplx

mtplx help
mtplx doctor --summary
mtplx pull Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
mtplx inspect Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed --json
```

Homebrew is the recommended macOS path. Python-only installs can use PyPI:

```bash
python3 -m pip install -U mtplx
```

The GitHub release wheel remains available for reproducible installs:

```bash
gh release download --repo youssofal/mtplx --pattern '*.whl'   # latest tagged release
python3 -m pip install ./mtplx-*-py3-none-any.whl
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --no-mtp
mtplx quickstart --port 8000 --no-stats-footer
```

`--no-mtp` switches generation to target-only AR. For MTP-equipped models the
MTP runtime stays loaded, so terminal chat can use `/mtp off`, `/mtp on`, and
`/mtp status` without reloading. Native AR-only models such as
`mlx-community/Laguna-S-2.1-oQ4e` instead install an unloaded AR route at
construction because there is no MTP head to retain.

For the 128 GB DeepSeek-V4 target-only artifact, install or build `mlx-serve`
and use the external AR route:

```bash
mtplx pull philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly

MTPLX_MLX_SERVE_BIN=/path/to/mlx-serve \
mtplx serve \
  --model philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly \
  --no-mtp --yes
```

The pull pins revision `ac33e4f3ca3546e6cec104558d42161e15814e33` and admits
the exact 44-shard publication. MTPLX removes ambient `MLX_SERVE_*` settings,
uses `MLX_SERVE_WIRED=fit` and a 256 MB cache limit, and leaves the external
memory preflight on. This is not MTP or DSpark support; representative
streaming performance is unapproved.

For scheduler selection and backend-specific concurrent implementations, see
[Concurrency modes](concurrency.md).

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space, and the runtime's admission gate requires ≈85.3 GiB of unified memory
(weights plus runtime headroom and a 16 GiB system reserve) — in practice a
96 GB Mac, with 128 GB comfortable. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
