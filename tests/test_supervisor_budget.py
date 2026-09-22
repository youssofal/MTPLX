"""Tests for mtplx.supervisor.budget: estimation and admission math."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.supervisor import budget as budget_mod
from mtplx.supervisor.budget import GIB, admit, estimate_resident_bytes, total_ram_bytes
from mtplx.supervisor.registry import EngineRegistry, EngineSpec, EngineState


def _write_safetensors(path: Path, name: str, size_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / name, "wb") as handle:
        handle.write(b"\0" * size_bytes)


def _spec(model_id: str, resident_bytes: int) -> EngineSpec:
    return EngineSpec(model_id=model_id, path=Path(f"/models/{model_id}"), resident_bytes=resident_bytes)


# ---- total_ram_bytes --------------------------------------------------


def test_total_ram_bytes_uses_detect_total_ram_bytes(monkeypatch):
    monkeypatch.setattr(budget_mod, "detect_total_ram_bytes", lambda: 32 * GIB)
    assert total_ram_bytes() == 32 * GIB


def test_total_ram_bytes_falls_back_when_detection_fails(monkeypatch):
    monkeypatch.setattr(budget_mod, "detect_total_ram_bytes", lambda: None)
    assert total_ram_bytes() > 0


# ---- estimate_resident_bytes -------------------------------------------


def test_estimate_resident_bytes_uses_catalog_peak_when_matched(monkeypatch, tmp_path):
    pack = tmp_path / "some-pack"
    pack.mkdir()
    fake = SimpleNamespace(peak_memory_gib=12.5)
    monkeypatch.setattr(budget_mod, "catalog_model_matching", lambda ref: fake if ref == "some-pack" else None)
    assert estimate_resident_bytes(pack) == int(12.5 * GIB)


def test_estimate_resident_bytes_uses_runtime_json_model_id(monkeypatch, tmp_path):
    pack = tmp_path / "unlabeled-dir"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text(json.dumps({"model_id": "catalog-id"}))
    fake = SimpleNamespace(peak_memory_gib=8.0)

    def fake_match(ref):
        return fake if ref == "catalog-id" else None

    monkeypatch.setattr(budget_mod, "catalog_model_matching", fake_match)
    assert estimate_resident_bytes(pack) == int(8.0 * GIB)


def test_estimate_resident_bytes_sums_safetensors_excluding_ngram_table(monkeypatch, tmp_path):
    pack = tmp_path / "no-catalog-match"
    monkeypatch.setattr(budget_mod, "catalog_model_matching", lambda ref: None)
    _write_safetensors(pack, "weights-00001.safetensors", 1000)
    _write_safetensors(pack, "weights-00002.safetensors", 2000)
    _write_safetensors(pack, "ngram-table.safetensors", 500_000)
    result = estimate_resident_bytes(pack)
    assert result == int(3000 * 1.15)


def test_estimate_resident_bytes_ignores_unreadable_runtime_json(monkeypatch, tmp_path):
    pack = tmp_path / "bad-json"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text("{not json")
    monkeypatch.setattr(budget_mod, "catalog_model_matching", lambda ref: None)
    _write_safetensors(pack, "weights.safetensors", 100)
    assert estimate_resident_bytes(pack) == int(100 * 1.15)


def test_estimate_resident_bytes_zero_when_no_safetensors_and_no_catalog(monkeypatch, tmp_path):
    pack = tmp_path / "empty-pack"
    pack.mkdir()
    monkeypatch.setattr(budget_mod, "catalog_model_matching", lambda ref: None)
    assert estimate_resident_bytes(pack) == 0


# ---- admit --------------------------------------------------------------


def test_admit_ok_when_budget_has_room():
    registry = EngineRegistry()
    spec = _spec("new-model", resident_bytes=10 * GIB)
    admission = admit(registry, spec, budget_bytes=64 * GIB, evict_to_fit=False)
    assert admission.ok
    assert admission.needed_bytes == 10 * GIB
    assert admission.available_bytes == 64 * GIB
    assert admission.would_free == []


def test_admit_accounts_for_ready_loading_and_draining_records():
    registry = EngineRegistry()
    registry.add(_spec("a", 20 * GIB))
    registry.set_state("a", EngineState.READY)
    registry.add(_spec("b", 10 * GIB))
    registry.set_state("b", EngineState.LOADING)
    registry.add(_spec("c", 5 * GIB))
    registry.set_state("c", EngineState.DRAINING)
    registry.add(_spec("d", 1 * GIB))  # installed, not counted

    spec = _spec("new-model", resident_bytes=20 * GIB)
    admission = admit(registry, spec, budget_bytes=64 * GIB, evict_to_fit=False)
    # 64 - (20 + 10 + 5) = 29 GiB available
    assert admission.available_bytes == 29 * GIB
    assert admission.ok


def test_admit_fails_without_evict_to_fit_and_lists_would_free():
    registry = EngineRegistry()
    registry.add(_spec("old", 20 * GIB))
    registry.set_state("old", EngineState.READY)
    registry.get("old").last_used = 1.0

    spec = _spec("qwen3.8-flash-next", resident_bytes=65 * GIB)
    admission = admit(registry, spec, budget_bytes=40.4 * GIB, evict_to_fit=False)
    assert not admission.ok
    assert admission.would_free == ["old"]
    assert "qwen3.8-flash-next" in admission.reason
    assert "GiB" in admission.reason


def test_admit_never_offers_a_pinned_engine_for_eviction():
    registry = EngineRegistry()
    registry.add(_spec("pinned", 30 * GIB))
    registry.set_state("pinned", EngineState.READY)
    registry.pin("pinned")

    spec = _spec("new-model", resident_bytes=20 * GIB)
    admission = admit(registry, spec, budget_bytes=25 * GIB, evict_to_fit=True)
    assert not admission.ok
    assert admission.would_free == []


def test_admit_evict_to_fit_frees_oldest_engines_until_it_fits():
    registry = EngineRegistry()
    registry.add(_spec("oldest", 15 * GIB))
    registry.set_state("oldest", EngineState.READY)
    registry.get("oldest").last_used = 1.0
    registry.add(_spec("newer", 15 * GIB))
    registry.set_state("newer", EngineState.READY)
    registry.get("newer").last_used = 2.0

    spec = _spec("big-model", resident_bytes=15 * GIB)
    # budget 30 GiB, both resident -> 0 available; evicting "oldest" frees 15
    admission = admit(registry, spec, budget_bytes=30 * GIB, evict_to_fit=True)
    assert admission.ok
    assert admission.would_free == ["oldest"]


def test_admit_evict_to_fit_still_fails_when_even_full_eviction_is_not_enough():
    registry = EngineRegistry()
    registry.add(_spec("only", 10 * GIB))
    registry.set_state("only", EngineState.READY)

    spec = _spec("huge-model", resident_bytes=200 * GIB)
    admission = admit(registry, spec, budget_bytes=20 * GIB, evict_to_fit=True)
    assert not admission.ok
    assert admission.would_free == ["only"]
    assert admission.needed_bytes == 200 * GIB


def test_admit_excludes_the_requested_model_itself_from_eviction_candidates():
    registry = EngineRegistry()
    registry.add(_spec("self-model", 10 * GIB))
    registry.set_state("self-model", EngineState.READY)
    registry.get("self-model").last_used = 1.0

    spec = _spec("self-model", resident_bytes=10 * GIB)
    admission = admit(registry, spec, budget_bytes=5 * GIB, evict_to_fit=True)
    assert admission.would_free == []
    assert not admission.ok
