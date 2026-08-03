# Contributing

MTPLX is production software; changes here ship worldwide via pip, Homebrew, and the DMG. Good contributions are small, measurable, and honest about evidence.

Before opening a PR:

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Benchmarks must include hardware, model, quantization, sampler, token count, profile, fan mode, date, and commit. Do not use fan-controlled runs for product headline claims.
