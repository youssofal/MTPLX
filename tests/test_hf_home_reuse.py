"""Tests for reusing Hugging Face cache (HF_HOME / HF_HUB_CACHE) across CLI, catalog, and pull."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx import hf_loader
from mtplx import model_catalog
from mtplx import model_folders
from mtplx.cli import main


def _write_hf_snapshot(
    hub_root: Path,
    repo_id: str,
    sha: str = "abcdef1234567890",
    *,
    complete: bool = True,
    write_ref: bool = True,
    pack_meta: dict | None = None,
) -> Path:
    org, name = repo_id.split("/", 1)
    repo_root = hub_root / f"models--{org}--{name}"
    snap = repo_root / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}\n", encoding="utf-8")
    (snap / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    if complete:
        (snap / "model.safetensors").write_bytes(b"weights")
    if pack_meta is not None:
        (snap / "mtplx-pack.json").write_text(json.dumps(pack_meta), encoding="utf-8")
    if write_ref:
        refs = repo_root / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(sha, encoding="utf-8")
    return snap


def test_hf_hub_cache_dir_prefers_hf_hub_cache_then_hf_home(tmp_path, monkeypatch):
    hf_home = tmp_path / "custom-hf-home"
    hf_hub = tmp_path / "explicit-hub"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert hf_loader.hf_hub_cache_dir() == hf_home / "hub"

    monkeypatch.setenv("HF_HUB_CACHE", str(hf_hub))
    assert hf_loader.hf_hub_cache_dir() == hf_hub


def test_pull_model_reuses_complete_model_from_hf_home_without_downloading(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf_home"
    hub = hf_home / "hub"
    mtplx_cache = tmp_path / "mtplx_models"
    mtplx_cache.mkdir()
    repo = "someone/example-coder"
    snap = _write_hf_snapshot(hub, repo)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    monkeypatch.setattr(
        hf_loader,
        "_query_repo_snapshot",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not query/download when HF_HOME is reusable")),
    )
    monkeypatch.setattr(hf_loader, "_local_matches_remote_index", lambda *a, **kw: True)

    result = hf_loader.pull_model(repo, cache_dir=mtplx_cache)
    assert result["reused_existing"] is True
    assert Path(result["path"]).resolve() == snap.resolve()
    assert not (mtplx_cache / "someone--example-coder").exists()


def test_pull_model_ignores_incomplete_hf_home_snapshot_and_downloads(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf_home"
    hub = hf_home / "hub"
    mtplx_cache = tmp_path / "mtplx_models"
    repo = "someone/incomplete-model"
    _write_hf_snapshot(hub, repo, complete=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))

    assert hf_loader.resolve_hf_repo_snapshot(repo) is None
    with pytest.raises(FileNotFoundError):
        hf_loader.resolve_model_path(repo, cache_dir=mtplx_cache)


def test_hf_layout_inside_added_model_dirs_works_for_resolve_list_catalog_and_folders(tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    primary.mkdir()
    hf_custom = tmp_path / "external_hf_hub"
    snap = _write_hf_snapshot(
        hf_custom,
        "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed",
        pack_meta={
            "source": "thinkingmachines/Inkling-2-Mini",
            "recipe": "bare-speed",
            "display_name": "Inkling 2 Mini Bare Speed",
        },
    )
    (snap / "mtp.safetensors").write_bytes(b"mtp")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(hf_loader, "huggingface_cache_path", lambda _repo_id: None)

    resolved = hf_loader.resolve_model_path(
        "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed",
        cache_dir=primary,
        search_dirs=[hf_custom],
    )
    assert resolved.resolve() == snap.resolve()

    cached = hf_loader.list_cached_models(cache_dir=primary, search_dirs=[hf_custom])
    assert any(m.repo_id == "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed" and m.path.resolve() == snap.resolve() for m in cached)

    installed = model_catalog.scan_installed_models(cache_dir=primary, search_dirs=[hf_custom])
    assert any(
        m.catalog is not None
        and m.catalog.id == "inkling2-mini-bare-speed"
        and m.path.resolve() == snap.resolve()
        for m in installed
    )
    assert model_folders.count_packs(hf_custom) == 1


def test_hf_home_models_appear_in_list_and_scan_and_folders_command(tmp_path, monkeypatch, capsys):
    primary = tmp_path / "primary"
    primary.mkdir()
    hf_home = tmp_path / "hf_home"
    snap = _write_hf_snapshot(hf_home / "hub", "vendor/Local-HF-Model")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(primary))

    cached = hf_loader.list_cached_models()
    assert any(m.repo_id == "vendor/Local-HF-Model" and m.path.resolve() == snap.resolve() for m in cached)

    installed = model_catalog.scan_installed_models()
    assert any(m.name == "vendor/Local-HF-Model" and m.path.resolve() == snap.resolve() for m in installed)

    assert main(["folders", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["huggingface_cache"]["role"] == "huggingface_cache"
    assert payload["huggingface_cache"]["packs"] == 1
    assert payload["huggingface_cache"]["read_only"] is True
