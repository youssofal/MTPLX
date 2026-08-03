# Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Keep generated artifacts, model weights, and local credentials out of Git. The release repository is a product export, not a research workspace dump.

## Release

Release artifacts are published from a clean tag:

```bash
git tag -a vX.Y.Z -m "MTPLX vX.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z dist/* scripts/install_macos.sh --title "MTPLX vX.Y.Z"
```

Use GitHub CLI authentication for artifact smoke tests:

```bash
gh release download vX.Y.Z --repo youssofal/mtplx --pattern 'mtplx-X.Y.Z-py3-none-any.whl'
python3 -m pip install ./mtplx-X.Y.Z-py3-none-any.whl
mtplx help
```

PyPI publishing is wired through Trusted Publishing, not local long-lived tokens. Before enabling the upload job, configure a pending publisher on PyPI:

```text
project: mtplx
owner: youssofal
repository: MTPLX
workflow: release.yml
environment: pypi
```

The `release.yml` workflow always builds and checks artifacts for tags. PyPI
publishing is manual only:

- tag pushes build and check artifacts but do not publish
- a maintainer must run the workflow with `publish_to_pypi=true` after the tag
  exists and the local build/twine gates pass
