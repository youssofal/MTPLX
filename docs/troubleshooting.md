# Troubleshooting

Start with:

```bash
mtplx doctor --summary
mtplx doctor --deep --json
mtplx doctor --bundle
```

Expected production failures should be actionable, not tracebacks:

| Symptom | Repair |
|---|---|
| MLX missing | `python3 -m pip install mlx` from native arm64 Python |
| Rosetta Python | switch to native arm64 Python and rerun `mtplx doctor` |
| default model missing | `mtplx pull Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` |
| Open WebUI cannot connect | use `http://127.0.0.1:8000/v1` on the host, or `http://host.docker.internal:8000/v1` inside Docker |
| Docker daemon stopped | start Docker Desktop |
| low disk/RAM | app: choose another drive under Settings → Model Storage; CLI: change `MTPLX_MODEL_DIR`; or free storage, lower context/profile, or use a smaller model |
| huggingface.co unreachable / blocked | CLI: `HF_ENDPOINT=https://hf-mirror.com mtplx pull <repo>` (same variable for `start`/`serve`); app: Settings → Advanced → HF download mirror. Your HF token is never sent to a mirror. |

See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for the wider table.
