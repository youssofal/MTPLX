# PyPI release runbook

MTPLX publishes to PyPI through PyPI Trusted Publishing from GitHub Actions.
This avoids long-lived PyPI API tokens in GitHub secrets.

## One-time PyPI setup

`mtplx` is published through PyPI Trusted Publishing. If the publisher ever
needs to be recreated, use the exact values below.

Create the pending publisher at:

```text
https://pypi.org/manage/account/publishing/
```

Use exactly:

```text
PyPI project name: mtplx
Owner: youssofal
Repository name: MTPLX
Workflow filename: release.yml
Environment name: pypi
```

The environment name matters. PyPI checks it against the GitHub OIDC token, so
`pypi` on PyPI must match the `environment: pypi` job in
`.github/workflows/release.yml`.

## Publish a release (vX.Y.Z)

After the version bump and release tag exist, run:

```bash
gh workflow run release.yml \
  --repo youssofal/MTPLX \
  -f ref=vX.Y.Z \
  -f publish_to_pypi=true
```

Watch the run:

```bash
gh run list --repo youssofal/MTPLX --workflow release --limit 1
gh run watch --repo youssofal/MTPLX --exit-status
```

## Verify the public install

```bash
python3 -m venv /tmp/mtplx-pypi-verify
/tmp/mtplx-pypi-verify/bin/python -m pip install -U pip
/tmp/mtplx-pypi-verify/bin/python -m pip install mtplx
/tmp/mtplx-pypi-verify/bin/mtplx help
```

A stable vX.Y.Z release should install without `--pre` once it is
explicitly published.

## Release guardrails

- PyPI upload is manual only: tag pushes build artifacts but do not publish.
- Publishing requires `publish_to_pypi=true`.
- Publishing requires the `ref` input to be a version tag beginning with `v`.
- PyPI Trusted Publishing must be configured with the exact repository,
  workflow, and environment above.
