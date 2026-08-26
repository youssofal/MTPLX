from importlib.metadata import PackageNotFoundError, version
from inspect import signature
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


DFLASH_MLX_PIN = (
    "dflash-mlx @ "
    "git+https://github.com/davidtai/dflash-mlx.git@"
    "54644e991039110f30140006c892c57734b9311e"
)
DFLASH_MLX_URL = "https://github.com/davidtai/dflash-mlx.git"
DFLASH_MLX_REVISION = "54644e991039110f30140006c892c57734b9311e"


def test_competitor_extra_pins_immutable_dflash_mlx_source():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["competitors"] == [DFLASH_MLX_PIN]


def test_dflash2_runtime_api_contract():
    from dflash_mlx.draft.dflash2 import DFlash2DraftModel
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.loading import load_draft_bundle

    assert version("dflash-mlx") == "0.1.10"
    assert DFlash2DraftModel.__name__ == "DFlash2DraftModel"
    assert QwenGdnTargetOps.backend_name == "qwen_gdn"
    assert callable(stream_dflash_generate)
    assert {"prefill_step_size", "should_cancel"} <= set(
        signature(stream_dflash_generate).parameters
    )
    assert callable(load_draft_bundle)


def test_installed_dflash2_vcs_identity_matches_the_runtime_pin():
    from mtplx.dflash_identity import require_pinned_dflash_install

    identity = require_pinned_dflash_install()

    assert identity.vcs == "git"
    assert identity.url == DFLASH_MLX_URL
    assert identity.commit_id == DFLASH_MLX_REVISION
    assert identity.requested_revision == DFLASH_MLX_REVISION


def test_dflash_identity_preflight_has_no_mlx_or_dflash_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mtplx.dflash_identity; "
                "assert not any(name == 'mlx' or name.startswith('mlx.') "
                "for name in sys.modules); "
                "assert not any(name == 'dflash_mlx' or "
                "name.startswith('dflash_mlx.') for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_deepseek_context_preflight_fails_before_dflash_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from importlib.metadata import PackageNotFoundError",
                    "import mtplx.dflash_identity as identity",
                    "from mtplx.benchmarks import dflash2_runtime",
                    "def missing(name): raise PackageNotFoundError(name)",
                    "identity.distribution = missing",
                    "try:",
                    "    dflash2_runtime.build_deepseek_v4_dflash2_runtime_context()",
                    "except RuntimeError:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('missing DFlash install was accepted')",
                    "assert 'mtplx.deepseek_v4_dflash2' not in sys.modules",
                    "assert not any(name == 'dflash_mlx' or "
                    "name.startswith('dflash_mlx.') for name in sys.modules)",
                )
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class _FakeDistribution:
    def __init__(self, direct_url_text: str | None) -> None:
        self.direct_url_text = direct_url_text

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self.direct_url_text


def _vcs_receipt(
    *,
    vcs: str = "git",
    url: str = DFLASH_MLX_URL,
    commit_id: str = DFLASH_MLX_REVISION,
    requested_revision: str = DFLASH_MLX_REVISION,
) -> str:
    return json.dumps(
        {
            "url": url,
            "vcs_info": {
                "vcs": vcs,
                "commit_id": commit_id,
                "requested_revision": requested_revision,
            },
        }
    )


def test_dflash_identity_preflight_rejects_missing_package(monkeypatch):
    import mtplx.dflash_identity as identity_module

    def missing_distribution(_name: str):
        raise PackageNotFoundError("dflash-mlx")

    monkeypatch.setattr(identity_module, "distribution", missing_distribution)

    with pytest.raises(RuntimeError, match="is not installed"):
        identity_module.require_pinned_dflash_install()


def test_deepseek_runtime_loader_preflights_before_model_load(monkeypatch):
    import mtplx.dflash_identity as identity_module
    from mtplx import runtime as runtime_module
    from mtplx.benchmarks import dflash2_runtime

    load_calls = []

    def missing_distribution(_name: str):
        raise PackageNotFoundError("dflash-mlx")

    monkeypatch.setattr(identity_module, "distribution", missing_distribution)
    monkeypatch.setattr(
        runtime_module,
        "load",
        lambda *_args, **_kwargs: load_calls.append((_args, _kwargs)),
    )

    with pytest.raises(RuntimeError, match="is not installed"):
        dflash2_runtime.load_mtplx_deepseek_runtime("/models/deepseek-v4")

    assert load_calls == []


def test_dflash_identity_preflight_rejects_missing_direct_url(monkeypatch):
    import mtplx.dflash_identity as identity_module

    monkeypatch.setattr(
        identity_module,
        "distribution",
        lambda _name: _FakeDistribution(None),
    )

    with pytest.raises(RuntimeError, match="has no PEP 610 direct_url.json"):
        identity_module.require_pinned_dflash_install()


def test_dflash_identity_preflight_rejects_non_vcs_direct_url(monkeypatch):
    import mtplx.dflash_identity as identity_module

    receipt = json.dumps(
        {
            "url": DFLASH_MLX_URL,
            "archive_info": {"hash": "sha256=deadbeef"},
        }
    )
    monkeypatch.setattr(
        identity_module,
        "distribution",
        lambda _name: _FakeDistribution(receipt),
    )

    with pytest.raises(RuntimeError, match="invalid PEP 610 VCS receipt"):
        identity_module.require_pinned_dflash_install()


@pytest.mark.parametrize(
    "receipt",
    [
        pytest.param(_vcs_receipt(vcs="hg"), id="wrong-vcs"),
        pytest.param(_vcs_receipt(url="https://example.invalid/dflash.git"), id="wrong-url"),
        pytest.param(_vcs_receipt(commit_id="0" * 40), id="wrong-commit"),
        pytest.param(_vcs_receipt(requested_revision="main"), id="wrong-requested"),
    ],
)
def test_dflash_identity_preflight_rejects_mismatched_vcs_identity(
    monkeypatch,
    receipt,
):
    import mtplx.dflash_identity as identity_module

    monkeypatch.setattr(
        identity_module,
        "distribution",
        lambda _name: _FakeDistribution(receipt),
    )

    with pytest.raises(RuntimeError, match="does not match the sealed Mia runtime"):
        identity_module.require_pinned_dflash_install()
