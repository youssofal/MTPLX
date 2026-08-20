from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.model_updates as model_updates
from mtplx.hf_loader import SOURCE_MARKER_FILE
from mtplx.model_updates import (
    ENGINE_VERSION,
    STATE_CURRENT,
    STATE_ENGINE_UPDATE_REQUIRED,
    STATE_UNKNOWN,
    STATE_UPDATE_AVAILABLE,
    ModelUpdateStatus,
    _cached_pack_repo_id,
    _diff_against_remote,
    check_model_updates,
    engine_satisfies,
    fetch_models_manifest,
    models_manifest_url,
    update_cached_model,
)


def _write_pack(root: Path, name: str, marker: dict | None = None) -> Path:
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}\n", encoding="utf-8")
    if marker is not None:
        (pack / SOURCE_MARKER_FILE).write_text(
            json.dumps(marker) + "\n", encoding="utf-8"
        )
    return pack


# --- manifest fetch -------------------------------------------------------


def test_models_manifest_url_env_override(monkeypatch):
    assert models_manifest_url() == model_updates.DEFAULT_MANIFEST_URL
    monkeypatch.setenv(model_updates.MANIFEST_URL_ENV, "https://example.test/m.json")
    assert models_manifest_url() == "https://example.test/m.json"


def _fake_urlopen(payload: bytes):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    def opener(request, timeout=None):
        del request, timeout
        return _Resp(payload)

    return opener


def test_fetch_models_manifest_validates_schema(monkeypatch):
    good = {"schema": 1, "models": {"a/b": {"revision": "abc"}}}
    monkeypatch.setattr(
        model_updates.urllib.request,
        "urlopen",
        _fake_urlopen(json.dumps(good).encode()),
    )
    assert fetch_models_manifest() == good

    for bad in (b"[]", b"not json", json.dumps({"schema": 2, "models": {}}).encode(),
                json.dumps({"schema": 1, "models": []}).encode()):
        monkeypatch.setattr(
            model_updates.urllib.request, "urlopen", _fake_urlopen(bad)
        )
        assert fetch_models_manifest() is None


def test_fetch_models_manifest_offline_returns_none(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(model_updates.urllib.request, "urlopen", boom)
    assert fetch_models_manifest() is None


# --- version gate ---------------------------------------------------------


def test_engine_satisfies_min_version():
    assert engine_satisfies(None)
    assert engine_satisfies("")
    assert engine_satisfies("0.1.0")
    assert engine_satisfies(ENGINE_VERSION)
    assert not engine_satisfies("99.0.0")


# --- delta estimation -----------------------------------------------------


def test_diff_against_remote_uses_blob_ids_then_sizes():
    marker = {
        "files": {
            "mtp.safetensors": {"size": 100, "blob_id": "old"},
            "config.json": {"size": 10, "blob_id": "cfg"},
            "same-size.bin": {"size": 50, "blob_id": "aaa"},
        }
    }
    remote = {
        "mtp.safetensors": {"size": 60, "blob_id": "new"},
        "config.json": {"size": 10, "blob_id": "cfg"},
        "same-size.bin": {"size": 50, "blob_id": "bbb"},
        "added.json": {"size": 5, "blob_id": "add"},
    }
    total, changed = _diff_against_remote(marker, remote)
    assert changed == ("added.json", "mtp.safetensors", "same-size.bin")
    assert total == 60 + 50 + 5


def test_diff_against_remote_without_local_files_is_unknown():
    assert _diff_against_remote({}, {"a": {"size": 1}}) == (None, ())
    assert _diff_against_remote(None, {"a": {"size": 1}}) == (None, ())


# --- repo id resolution ---------------------------------------------------


def test_cached_pack_repo_id_prefers_marker_then_dirname_then_catalog(tmp_path):
    marked = _write_pack(tmp_path, "anything", {"repo_id": "someone/pack"})
    assert _cached_pack_repo_id(marked) == "someone/pack"

    conventional = _write_pack(tmp_path, "owner--repo-name")
    assert _cached_pack_repo_id(conventional) == "owner/repo-name"

    bare = _write_pack(tmp_path, "Qwen3.8-27B-MTPLX-Optimized-Speed")
    assert (
        _cached_pack_repo_id(bare)
        == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    )

    unknown = _write_pack(tmp_path, "some-random-local-model")
    assert _cached_pack_repo_id(unknown) is None


# --- check_model_updates --------------------------------------------------


def test_check_model_updates_states(tmp_path, monkeypatch):
    _write_pack(
        tmp_path,
        "owner--current",
        {"repo_id": "owner/current", "revision": None, "resolved_sha": "sha-cur"},
    )
    _write_pack(
        tmp_path,
        "owner--stale",
        {
            "repo_id": "owner/stale",
            "revision": None,
            "resolved_sha": "sha-old",
            "files": {"mtp.safetensors": {"size": 100, "blob_id": "old"}},
        },
    )
    _write_pack(tmp_path, "owner--untracked", {"repo_id": "owner/untracked"})
    _write_pack(
        tmp_path,
        "owner--gated",
        {"repo_id": "owner/gated", "resolved_sha": "sha-g1"},
    )

    manifest = {
        "schema": 1,
        "models": {
            "owner/current": {"revision": "sha-cur"},
            "owner/stale": {
                "revision": "sha-new",
                "note": "Quantized MTP head",
            },
            "owner/untracked": {"revision": "sha-u2"},
            "owner/gated": {"revision": "sha-g2", "min_engine_version": "99.0.0"},
        },
    }

    def fake_snapshot(repo_id, *, revision=None):
        assert repo_id == "owner/stale"
        assert revision == "sha-new"
        return "sha-new", {"mtp.safetensors": {"size": 60, "blob_id": "new"}}

    monkeypatch.setattr(model_updates, "_query_repo_snapshot", fake_snapshot)

    rows = {
        row.repo_id: row
        for row in check_model_updates(cache_dir=tmp_path, manifest=manifest)
    }
    assert rows["owner/current"].state == STATE_CURRENT
    assert rows["owner/stale"].state == STATE_UPDATE_AVAILABLE
    assert rows["owner/stale"].update_bytes == 60
    assert rows["owner/stale"].changed_files == ("mtp.safetensors",)
    assert rows["owner/stale"].note == "Quantized MTP head"
    assert rows["owner/untracked"].state == STATE_UNKNOWN
    assert rows["owner/gated"].state == STATE_ENGINE_UPDATE_REQUIRED
    assert rows["owner/gated"].min_engine_version == "99.0.0"


def test_check_model_updates_hub_fallback_for_unlisted_repo(tmp_path, monkeypatch):
    _write_pack(
        tmp_path,
        "owner--offmanifest",
        {"repo_id": "owner/offmanifest", "resolved_sha": "sha-1"},
    )
    monkeypatch.setattr(
        model_updates,
        "_query_repo_snapshot",
        lambda repo_id, *, revision=None: ("sha-2", None),
    )
    rows = check_model_updates(cache_dir=tmp_path, manifest=None)
    assert len(rows) == 1
    assert rows[0].state == STATE_UPDATE_AVAILABLE
    assert rows[0].source == "hub"


def test_check_model_updates_offline_reports_unknown_not_error(tmp_path, monkeypatch):
    _write_pack(
        tmp_path,
        "owner--offline",
        {"repo_id": "owner/offline", "resolved_sha": "sha-1"},
    )
    monkeypatch.setattr(
        model_updates,
        "_query_repo_snapshot",
        lambda repo_id, *, revision=None: (None, None),
    )
    rows = check_model_updates(cache_dir=tmp_path, manifest=None)
    assert rows[0].state == STATE_UNKNOWN
    assert rows[0].source == "none"


def test_check_model_updates_skips_symlink_overlays(tmp_path, monkeypatch):
    real = _write_pack(
        tmp_path, "owner--real", {"repo_id": "owner/real", "resolved_sha": "s1"}
    )
    (tmp_path / "overlay-exp").symlink_to(real)
    monkeypatch.setattr(
        model_updates,
        "_query_repo_snapshot",
        lambda repo_id, *, revision=None: ("s1", None),
    )
    rows = check_model_updates(cache_dir=tmp_path, manifest=None)
    assert [row.repo_id for row in rows] == ["owner/real"]


def test_check_model_updates_legacy_cache_size_diff(tmp_path, monkeypatch):
    # Pre-2.9 caches have no provenance marker at all. A size mismatch (or a
    # file missing locally — the mtp.safetensors trap) proves staleness and
    # must surface update-available; equal sizes prove nothing and must stay
    # unknown — never a false current.
    stale = _write_pack(tmp_path, "owner--legacy-stale", None)
    (stale / "mtp.safetensors").write_bytes(b"x" * 100)
    same = _write_pack(tmp_path, "owner--legacy-same", None)
    (same / "mtp.safetensors").write_bytes(b"y" * 60)
    _write_pack(tmp_path, "owner--legacy-trap", None)  # mtp.safetensors absent

    manifest = {
        "schema": 1,
        "models": {
            "owner/legacy-stale": {"revision": "sha-ls"},
            "owner/legacy-same": {"revision": "sha-sm"},
            "owner/legacy-trap": {"revision": "sha-tr"},
        },
    }
    listings = {
        "owner/legacy-stale": {
            "config.json": {"size": 3, "blob_id": "c"},
            "mtp.safetensors": {"size": 60, "blob_id": "n"},
            ".gitattributes": {"size": 1500, "blob_id": "g"},
        },
        "owner/legacy-same": {
            "config.json": {"size": 3, "blob_id": "c"},
            "mtp.safetensors": {"size": 60, "blob_id": "n"},
        },
        "owner/legacy-trap": {
            "config.json": {"size": 3, "blob_id": "c"},
            "mtp.safetensors": {"size": 60, "blob_id": "n"},
        },
    }
    monkeypatch.setattr(
        model_updates,
        "_query_repo_snapshot",
        lambda repo_id, *, revision=None: (revision, listings[repo_id]),
    )

    rows = {
        row.repo_id: row
        for row in check_model_updates(cache_dir=tmp_path, manifest=manifest)
    }

    stale_row = rows["owner/legacy-stale"]
    assert stale_row.state == STATE_UPDATE_AVAILABLE
    assert stale_row.local_revision is None
    assert stale_row.changed_files == ("mtp.safetensors",)
    assert stale_row.update_bytes == 60  # .gitattributes ignored

    assert rows["owner/legacy-same"].state == STATE_UNKNOWN

    trap_row = rows["owner/legacy-trap"]
    assert trap_row.state == STATE_UPDATE_AVAILABLE
    assert trap_row.changed_files == ("mtp.safetensors",)


def test_diff_against_local_dir_unsized_remote_still_flags(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "mtp.safetensors").write_bytes(b"x" * 10)
    update_bytes, changed = model_updates._diff_against_local_dir(
        pack, {"mtp.safetensors": {"size": None, "blob_id": "n"}}
    )
    assert changed == ("mtp.safetensors",)
    assert update_bytes is None  # unsized listing: flag the file, skip the sum


# --- update_cached_model --------------------------------------------------


def test_update_cached_model_pins_manifest_revision_and_unlinks_same_size(
    tmp_path, monkeypatch
):
    pack = _write_pack(
        tmp_path,
        "owner--pack",
        {
            "repo_id": "owner/pack",
            "resolved_sha": "sha-old",
            "files": {
                "same-size.json": {"size": 4, "blob_id": "aaa"},
                "grows.bin": {"size": 2, "blob_id": "bbb"},
            },
        },
    )
    (pack / "same-size.json").write_text("old!", encoding="utf-8")
    (pack / "grows.bin").write_bytes(b"xx")

    manifest = {
        "schema": 1,
        "models": {"owner/pack": {"revision": "sha-new"}},
    }
    monkeypatch.setattr(
        model_updates,
        "_query_repo_snapshot",
        lambda repo_id, *, revision=None: (
            "sha-new",
            {
                "same-size.json": {"size": 4, "blob_id": "ccc"},
                "grows.bin": {"size": 9, "blob_id": "ddd"},
            },
        ),
    )
    captured: dict = {}

    def fake_pull(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return {"repo_id": repo_id, "path": str(pack)}

    monkeypatch.setattr(model_updates, "pull_model", fake_pull)

    result = update_cached_model("owner/pack", cache_dir=tmp_path, manifest=manifest)

    assert result["repo_id"] == "owner/pack"
    assert captured["revision"] == "sha-new"
    assert captured["force_sync"] is True
    # the canonical dir it resolved must be the one pull_model targets
    assert captured["destination"] == pack
    # size-identical content change must be unlinked so the delta re-fetches it
    assert not (pack / "same-size.json").exists()
    # size-changing files ride the ordinary pull delta untouched
    assert (pack / "grows.bin").exists()


def test_update_cached_model_targets_bare_layout_dir(tmp_path, monkeypatch):
    # Forge-built / legacy caches use the bare pack name, not owner--name.
    # The update must be pointed at that directory — otherwise pull_model
    # recomputes the canonical path and does a full re-download into a
    # duplicate dir instead of a delta into the pack it was asked to update.
    bare = _write_pack(tmp_path, "pack", None)
    (bare / "mtp.safetensors").write_bytes(b"x" * 100)

    manifest = {"schema": 1, "models": {"owner/pack": {"revision": "sha-new"}}}
    captured: dict = {}

    def fake_pull(repo_id, **kwargs):
        captured.update(kwargs)
        return {"repo_id": repo_id, "path": str(bare)}

    monkeypatch.setattr(model_updates, "pull_model", fake_pull)

    update_cached_model("owner/pack", cache_dir=tmp_path, manifest=manifest)

    assert captured["destination"] == bare
    assert captured["force_sync"] is True


def test_update_cached_model_prefers_populated_bare_over_empty_canonical(
    tmp_path, monkeypatch
):
    # The app (and any caller) may pre-create the canonical owner--name dir.
    # An empty canonical dir must not shadow the populated legacy dir the
    # update was aimed at, or the delta becomes a full re-download.
    (tmp_path / "owner--pack").mkdir()
    bare = _write_pack(tmp_path, "pack", None)
    (bare / "mtp.safetensors").write_bytes(b"x" * 100)

    manifest = {"schema": 1, "models": {"owner/pack": {"revision": "sha-new"}}}
    captured: dict = {}

    def fake_pull(repo_id, **kwargs):
        captured.update(kwargs)
        return {"repo_id": repo_id, "path": str(bare)}

    monkeypatch.setattr(model_updates, "pull_model", fake_pull)

    update_cached_model("owner/pack", cache_dir=tmp_path, manifest=manifest)

    assert captured["destination"] == bare


def test_update_cached_model_targets_exact_stale_path_when_duplicate_is_current(
    tmp_path, monkeypatch
):
    canonical = _write_pack(tmp_path, "owner--pack", {"repo_id": "owner/pack"})
    (canonical / "mtp.safetensors").write_bytes(b"current")
    bare = _write_pack(tmp_path, "pack", None)
    (bare / "mtp.safetensors").write_bytes(b"stale")

    manifest = {"schema": 1, "models": {"owner/pack": {"revision": "sha-new"}}}
    captured: dict = {}

    def fake_pull(repo_id, **kwargs):
        captured.update(kwargs)
        return {"repo_id": repo_id, "path": str(bare)}

    monkeypatch.setattr(model_updates, "pull_model", fake_pull)
    monkeypatch.setattr(
        model_updates,
        "_cached_pack_repo_id",
        lambda path: "owner/pack" if path == bare else None,
    )

    update_cached_model(
        "owner/pack",
        cache_dir=tmp_path,
        destination_path=bare,
        manifest=manifest,
    )

    assert captured["destination"] == bare


def test_cmd_models_update_progress_json_emits_pull_schema(monkeypatch, capsys):
    from types import SimpleNamespace

    from mtplx.commands import public

    monkeypatch.setattr(
        public, "fetch_models_manifest", lambda: {"schema": 1, "models": {}},
        raising=False,
    )
    import mtplx.model_updates as mu

    monkeypatch.setattr(
        mu, "fetch_models_manifest", lambda: {"schema": 1, "models": {}}
    )
    captured: dict = {}

    def fake_update(repo, **kwargs):
        captured.update(kwargs)
        kwargs["progress_callback"](
            {
                "event": "progress",
                "repo_id": repo,
                "delta_bytes": 25,
                "size_bytes": 65,
            }
        )
        return {
            "repo_id": repo,
            "path": "/tmp/pack",
            "resolved_sha": "sha-new",
            "size_bytes": 100,
            "started_size_bytes": 40,
        }

    monkeypatch.setattr(mu, "update_cached_model", fake_update)

    args = SimpleNamespace(
        cache_dir=None,
        json=False,
        progress_json=True,
        installed_path="/tmp/pack",
    )
    rc = public._cmd_models_update(args, ["owner/pack"])
    assert rc == 0
    assert captured["destination_path"] == "/tmp/pack"

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "resolving"
    assert kinds[-1] == "result"
    assert events[-1]["resolved_sha"] == "sha-new"
    assert events[-1]["delta_bytes"] == 60
    assert events[-1]["downloaded_bytes"] == 25


def test_update_cached_model_fresh_pull_lets_pull_model_resolve(tmp_path, monkeypatch):
    # Nothing cached at all: pull_model owns destination resolution.
    manifest = {"schema": 1, "models": {"owner/pack": {"revision": "sha-new"}}}
    captured: dict = {}

    def fake_pull(repo_id, **kwargs):
        captured.update(kwargs)
        return {"repo_id": repo_id, "path": str(tmp_path / "owner--pack")}

    monkeypatch.setattr(model_updates, "pull_model", fake_pull)

    update_cached_model("owner/pack", cache_dir=tmp_path, manifest=manifest)

    assert captured["destination"] is None


def test_update_cached_model_engine_gate(tmp_path, monkeypatch):
    manifest = {
        "schema": 1,
        "models": {"owner/pack": {"revision": "x", "min_engine_version": "99.0.0"}},
    }
    monkeypatch.setattr(
        model_updates, "pull_model", lambda *a, **k: pytest.fail("must not pull")
    )
    with pytest.raises(RuntimeError, match="requires MTPLX >= 99.0.0"):
        update_cached_model("owner/pack", cache_dir=tmp_path, manifest=manifest)


def test_status_to_dict_roundtrip():
    row = ModelUpdateStatus(
        repo_id="a/b",
        path="/x",
        state=STATE_CURRENT,
        local_revision="s",
        remote_revision="s",
        source="manifest",
    )
    payload = row.to_dict()
    assert payload["repo_id"] == "a/b"
    assert payload["state"] == STATE_CURRENT
    assert payload["changed_files"] == []
