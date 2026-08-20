"""Model-pack update checks: pull provenance vs published revisions.

The Sparkle counterpart for model packs. Every ``mtplx pull`` records what it
actually downloaded (``.mtplx-source.json``: repo, resolved commit sha,
per-file blob map). This module compares those markers against the published
models manifest (``https://mtplx.com/releases/models.json``) with a Hugging
Face API fallback, so ``mtplx models --check`` and the desktop app can offer
one-click delta updates when a pack is re-published — including sidecar-only
changes (a re-quantized ``mtp.safetensors`` head) that never touch the weight
index and were previously invisible to cached installs.

Design constraints:
- Offline-first: every network failure degrades to "no update information",
  never to an error that blocks serving or listing.
- The manifest is a bless-list, not a protocol: it pins the exact commit a
  given engine version should update into (``min_engine_version`` gates packs
  that need a newer loader). Repos absent from the manifest fall back to the
  repo's current main revision on the Hub.
- Updates ride the ordinary pull path, which already skips size-identical
  files — so a head swap costs the head, not the trunk.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.hf_loader import (
    _query_repo_snapshot,
    cached_model_path,
    model_cache_dir,
    pull_model,
    read_source_marker,
    repo_id_from_model_ref,
    safe_model_name,
)
from mtplx.version import __version__ as ENGINE_VERSION

DEFAULT_MANIFEST_URL = "https://mtplx.com/releases/models.json"
MANIFEST_URL_ENV = "MTPLX_MODELS_MANIFEST_URL"
MANIFEST_TIMEOUT_S = 5.0
MANIFEST_SCHEMA = 1

STATE_CURRENT = "current"
STATE_UPDATE_AVAILABLE = "update-available"
STATE_ENGINE_UPDATE_REQUIRED = "engine-update-required"
STATE_UNKNOWN = "unknown"


def models_manifest_url() -> str:
    return os.environ.get(MANIFEST_URL_ENV) or DEFAULT_MANIFEST_URL


def fetch_models_manifest(url: str | None = None) -> dict[str, Any] | None:
    """Fetch and validate the published models manifest.

    Returns None on any failure (offline, HTTP error, malformed payload) —
    callers fall back to the Hub or report unknown.
    """

    target = url or models_manifest_url()
    try:
        request = urllib.request.Request(
            target,
            headers={"User-Agent": f"mtplx/{ENGINE_VERSION}"},
        )
        with urllib.request.urlopen(request, timeout=MANIFEST_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != MANIFEST_SCHEMA:
        return None
    models = payload.get("models")
    if not isinstance(models, dict):
        return None
    return payload


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(value).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts or [0])


def engine_satisfies(min_version: str | None) -> bool:
    if not min_version:
        return True
    return _version_tuple(ENGINE_VERSION) >= _version_tuple(min_version)


@dataclass(frozen=True)
class ModelUpdateStatus:
    repo_id: str
    path: str
    state: str
    local_revision: str | None
    remote_revision: str | None
    source: str  # "manifest" | "hub" | "none"
    note: str | None = None
    min_engine_version: str | None = None
    update_bytes: int | None = None
    changed_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "path": self.path,
            "state": self.state,
            "local_revision": self.local_revision,
            "remote_revision": self.remote_revision,
            "source": self.source,
            "note": self.note,
            "min_engine_version": self.min_engine_version,
            "update_bytes": self.update_bytes,
            "changed_files": list(self.changed_files),
        }


def _manifest_entry(manifest: dict[str, Any] | None, repo_id: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    models = manifest.get("models")
    if not isinstance(models, dict):
        return None
    for key, value in models.items():
        if isinstance(key, str) and key.casefold() == repo_id.casefold():
            return value if isinstance(value, dict) else None
    return None


def _diff_against_remote(
    marker: dict[str, Any] | None,
    remote_files: dict[str, dict[str, Any]] | None,
) -> tuple[int | None, tuple[str, ...]]:
    """Estimate the delta a sync would download (changed/new remote files)."""

    if not remote_files:
        return None, ()
    local_files = (marker or {}).get("files")
    if not isinstance(local_files, dict):
        return None, ()
    changed: list[str] = []
    total = 0
    sized = True
    for path, entry in remote_files.items():
        local = local_files.get(path)
        remote_blob = entry.get("blob_id")
        remote_size = entry.get("size")
        if isinstance(local, dict):
            local_blob = local.get("blob_id")
            local_size = local.get("size")
            if remote_blob and local_blob and remote_blob == local_blob:
                continue
            if not remote_blob and isinstance(remote_size, int) and remote_size == local_size:
                continue
        changed.append(path)
        if isinstance(remote_size, int):
            total += remote_size
        else:
            sized = False
    return (total if sized and changed else None), tuple(sorted(changed))


def _diff_against_local_dir(
    pack_dir: Path,
    remote_files: dict[str, dict[str, Any]] | None,
) -> tuple[int | None, tuple[str, ...]]:
    """Size-diff a markerless (pre-2.9) cache dir against the remote listing.

    A missing file or size mismatch PROVES the cache differs from the blessed
    revision, so ``update-available`` is safe to report. Equal sizes prove
    nothing (a size-identical content change is invisible without blob ids),
    so callers must keep reporting ``unknown`` in that case — never a false
    ``current``.
    """

    if not remote_files:
        return None, ()
    changed: list[str] = []
    total = 0
    sized = True
    for path, entry in remote_files.items():
        if path.startswith(".") or "/." in path:
            # Git plumbing (.gitattributes): present in listings, not pack
            # content, and not something a sync should trigger on.
            continue
        remote_size = entry.get("size")
        try:
            local_size: int | None = (pack_dir / path).stat().st_size
        except OSError:
            local_size = None
        if local_size is not None and isinstance(remote_size, int) and local_size == remote_size:
            continue
        changed.append(path)
        if isinstance(remote_size, int):
            total += remote_size
        else:
            sized = False
    return (total if sized and changed else None), tuple(sorted(changed))


def _cached_pack_repo_id(path: Path) -> str | None:
    """Best-effort repo id for a cache directory.

    Priority: the pull marker (authoritative), then the ``owner--name``
    directory convention, then the official catalog (matches bare pack names
    against each entry's Hub id tail).
    """

    marker = read_source_marker(path)
    repo_id = (marker or {}).get("repo_id")
    if isinstance(repo_id, str) and "/" in repo_id:
        return repo_id
    if "--" in path.name:
        candidate = path.name.replace("--", "/")
        if repo_id_from_model_ref(candidate):
            return candidate
    try:
        from mtplx.model_catalog import OFFICIAL_CATALOG

        for entry in OFFICIAL_CATALOG:
            hub_id = entry.hf_model_id
            if path.name.casefold() == hub_id.split("/", 1)[-1].casefold():
                return hub_id
    except Exception:
        pass
    return None


def check_model_updates(
    *,
    cache_dir: str | Path | None = None,
    manifest: dict[str, Any] | None | str = "auto",
    use_hub_fallback: bool = True,
    include_current: bool = True,
) -> list[ModelUpdateStatus]:
    """Compare every tracked cached pack against its published revision."""

    if manifest == "auto":
        manifest = fetch_models_manifest()
    root = model_cache_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[ModelUpdateStatus] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.is_symlink():
            continue
        repo_id = _cached_pack_repo_id(child)
        if not repo_id:
            continue
        marker = read_source_marker(child)
        local_sha = (marker or {}).get("resolved_sha")
        local_sha = local_sha if isinstance(local_sha, str) and local_sha else None

        entry = _manifest_entry(manifest if isinstance(manifest, dict) else None, repo_id)
        note = None
        min_engine = None
        remote_sha: str | None = None
        remote_files: dict[str, dict[str, Any]] | None = None
        source = "none"
        if entry:
            source = "manifest"
            revision = entry.get("revision")
            remote_sha = revision if isinstance(revision, str) and revision else None
            raw_note = entry.get("note")
            note = raw_note if isinstance(raw_note, str) else None
            raw_min = entry.get("min_engine_version")
            min_engine = raw_min if isinstance(raw_min, str) else None
        elif use_hub_fallback and local_sha:
            # Only tracked packs are worth a Hub round-trip without a
            # manifest entry: untracked ones would report unknown anyway.
            remote_sha, remote_files = _query_repo_snapshot(repo_id)
            source = "hub" if remote_sha else "none"

        update_bytes: int | None = None
        changed: tuple[str, ...] = ()
        if entry and not engine_satisfies(min_engine):
            state = STATE_ENGINE_UPDATE_REQUIRED
        elif remote_sha and local_sha:
            state = STATE_CURRENT if remote_sha == local_sha else STATE_UPDATE_AVAILABLE
            if state == STATE_UPDATE_AVAILABLE:
                if remote_files is None:
                    _, remote_files = _query_repo_snapshot(repo_id, revision=remote_sha)
                update_bytes, changed = _diff_against_remote(marker, remote_files)
        elif remote_sha and marker is None:
            # Pre-2.9 cache: no provenance marker was ever written. A size
            # diff against the blessed revision can still PROVE staleness
            # (missing or size-changed files) — that is exactly the
            # mtp.safetensors-invisible-to-the-weight-index trap. Equal
            # sizes prove nothing and stay "unknown".
            if remote_files is None:
                _, remote_files = _query_repo_snapshot(repo_id, revision=remote_sha)
            update_bytes, changed = _diff_against_local_dir(child, remote_files)
            state = STATE_UPDATE_AVAILABLE if changed else STATE_UNKNOWN
        else:
            # No remote revision (offline / unlisted) or no local provenance:
            # "unknown" is the only honest answer — never a false "current".
            state = STATE_UNKNOWN

        if state == STATE_CURRENT and not include_current:
            continue
        rows.append(
            ModelUpdateStatus(
                repo_id=repo_id,
                path=str(child),
                state=state,
                local_revision=local_sha,
                remote_revision=remote_sha,
                source=source,
                note=note,
                min_engine_version=min_engine,
                update_bytes=update_bytes,
                changed_files=changed,
            )
        )
    return rows


def update_cached_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    destination_path: str | Path | None = None,
    manifest: dict[str, Any] | None | str = "auto",
    progress_callback: Any = None,
    progress_interval_s: float = 10.0,
) -> dict[str, Any]:
    """Sync one cached pack to its published revision via the pull path.

    Rides the ordinary delta download (size-identical files are skipped).
    Size-identical files whose blob changed are unlinked first so the delta
    stays exact even when a re-published file keeps its byte length.
    """

    repo_id = repo_id_from_model_ref(model_ref) or model_ref
    if manifest == "auto":
        manifest = fetch_models_manifest()
    entry = _manifest_entry(manifest if isinstance(manifest, dict) else None, repo_id)
    if entry and not engine_satisfies(entry.get("min_engine_version")):
        raise RuntimeError(
            f"{repo_id} requires MTPLX >= {entry.get('min_engine_version')} "
            f"(this is {ENGINE_VERSION}); update MTPLX first."
        )
    target_revision: str | None = None
    if entry:
        revision = entry.get("revision")
        if isinstance(revision, str) and revision:
            target_revision = revision

    cache_root = model_cache_dir(cache_dir).expanduser().resolve()
    if destination_path is not None:
        destination = Path(destination_path).expanduser().absolute()
        if destination.parent.resolve() != cache_root:
            raise ValueError(f"update path must be a direct child of {cache_root}")
        if not destination.is_dir():
            raise FileNotFoundError(f"installed model path does not exist: {destination}")
        installed_repo = _cached_pack_repo_id(destination)
        if not installed_repo or installed_repo.casefold() != repo_id.casefold():
            raise ValueError(
                f"installed model path {destination} does not match {repo_id}"
            )
    else:
        destination = cached_model_path(repo_id, cache_dir=cache_dir)
        canonical_populated = destination.exists() and any(destination.iterdir())
        if not canonical_populated:
            # Fall back to the bare pack-name directory (forge-built or legacy
            # layouts) so an update never creates a duplicate copy of a pack.
            # An EMPTY canonical dir counts as absent: callers sometimes
            # pre-create it, and letting it win would turn a delta into a full
            # re-download while the populated legacy dir stays stale.
            bare = cache_root / safe_model_name(repo_id).split("--")[-1]
            if bare.exists():
                destination = bare
    marker = read_source_marker(destination) if destination.exists() else None
    if destination.exists() and isinstance((marker or {}).get("files"), dict):
        remote_sha, remote_files = _query_repo_snapshot(repo_id, revision=target_revision)
        if remote_files:
            _, changed = _diff_against_remote(marker, remote_files)
            for name in changed:
                stale = destination / name
                local_entry = (marker or {}).get("files", {}).get(name)
                remote_entry = remote_files.get(name) or {}
                # The pull path already re-fetches size-mismatched files;
                # only a size-identical content change needs the nudge.
                if (
                    stale.is_file()
                    and isinstance(local_entry, dict)
                    and local_entry.get("size") == remote_entry.get("size")
                ):
                    stale.unlink()

    return pull_model(
        repo_id,
        cache_dir=cache_dir,
        revision=target_revision,
        progress_callback=progress_callback,
        progress_interval_s=progress_interval_s,
        force_sync=True,
        # The resolved dir (bare legacy layouts included) — otherwise
        # pull_model recomputes the canonical owner--name path and a
        # bare-layout pack gets a full re-download into a duplicate dir
        # instead of a delta into the pack it was asked to update.
        destination=destination if destination.exists() else None,
    )
