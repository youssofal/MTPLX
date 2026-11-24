"""Model folders: where MTPLX finds packs and where downloads go.

One setting for the app and the CLI, in one file under two keys:
``model_dir`` (the download folder, which also receives Forge builds) and
``model_dirs`` (more folders searched for packs, read only), in
``~/.mtplx/config.toml`` (``MTPLX_CONFIG`` names another file). The app reads
and writes the same two keys (``SharedModelFolders`` in MTPLXAppCore), so a
folder added in either place shows in both.

A folder on a drive that is not plugged in stays in the list and is reported
as not connected; its packs show again when the drive is back. Nothing here
copies, moves or deletes a pack: removing a folder only stops searching it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mtplx import toml_lines
from mtplx.config import load_user_config, user_config_path
from mtplx.hf_loader import (
    _hf_home_env_set,
    _hf_home_reuse_disabled,
    _resolve_hf_dir_snapshot,
    hf_hub_cache_dir,
    model_cache_dir,
    model_library_roots,
)

PACK_MARKERS = ("config.json", "mtplx_pair.json")


def _canonical(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve(strict=False)


def _same(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(str(_canonical(a))) == os.path.normcase(str(_canonical(b)))


def volume_of(path: str | Path) -> str | None:
    """The /Volumes/<name> a path lives on, else None (the startup disk)."""

    parts = Path(str(path)).expanduser().parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        return f"/Volumes/{parts[2]}"
    return None


def folder_status(path: str | Path) -> dict[str, Any]:
    """Whether a folder can be read now, and if not, why in plain words."""

    raw = Path(str(path)).expanduser()
    row: dict[str, Any] = {"path": str(raw), "available": False, "reason": None, "volume": None}
    target = raw
    if raw.is_symlink() and not raw.exists():
        # The #466 layout: a link onto an external drive that is unplugged.
        target = Path(os.path.realpath(raw))
    volume = volume_of(target)
    row["volume"] = volume
    if raw.is_dir():
        row["available"] = True
        return row
    if volume and not os.path.ismount(volume):
        row["reason"] = "not_connected"
    elif raw.exists():
        row["reason"] = "not_a_folder"
    else:
        row["reason"] = "missing"
    return row


def _has_pack_marker(directory: Path) -> bool:
    return any((directory / marker).is_file() for marker in PACK_MARKERS)


def count_packs(root: str | Path) -> int:
    try:
        children = list(Path(str(root)).expanduser().iterdir())
    except OSError:
        return 0
    total = 0
    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name.startswith("models--"):
            snap = _resolve_hf_dir_snapshot(child, require_complete=False)
            if snap is not None and _has_pack_marker(snap):
                total += 1
        elif _has_pack_marker(child):
            total += 1
    return total


def hf_cache_folder_row() -> dict[str, Any] | None:
    """Read-only Hugging Face hub cache folder row when active or on disk."""
    if _hf_home_reuse_disabled():
        return None
    hf_hub = hf_hub_cache_dir()
    if not hf_hub.exists() and not _hf_home_env_set():
        return None
    status = folder_status(hf_hub)
    status["role"] = "huggingface_cache"
    status["read_only"] = True
    status["packs"] = count_packs(hf_hub) if status["available"] else None
    return status


def folders_report(
    cache_dir: str | Path | None = None,
    *,
    search_dirs: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Every folder in search order: the download folder first."""

    rows: list[dict[str, Any]] = []
    for index, root in enumerate(model_library_roots(cache_dir, search_dirs=search_dirs)):
        status = folder_status(root)
        status["role"] = "downloads" if index == 0 else "search"
        status["packs"] = count_packs(root) if status["available"] else None
        rows.append(status)
    return rows


def configured_folders(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_user_config(config_path)
    return {
        "config_path": str(config.path),
        "model_dir": config.model_dir,
        "model_dirs": list(config.model_dirs),
    }


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o644
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def save_folders(
    *,
    downloads: str | None = None,
    folders: list[str] | None = None,
    clear_downloads: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``model_dir`` and/or ``model_dirs``; every other line is kept."""

    path = user_config_path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = "# MTPLX user configuration\n"
    updates: dict[str, Any] = {}
    if downloads is not None or clear_downloads:
        updates["model_dir"] = None if clear_downloads else downloads
    if folders is not None:
        # Written even when empty: a present key is how the app tells "set to
        # no extra folders" from "never set" (which it fills from its own
        # settings once).
        updates["model_dirs"] = list(folders)
    _write_config(path, toml_lines.set_top_level_values(text, updates))
    return configured_folders(path)


def add_folders(paths: list[str], *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Add search folders. A folder must exist, or sit on a drive that is not
    plugged in now (it shows as not connected until it is)."""

    current = configured_folders(config_path)
    downloads = current["model_dir"] or str(model_cache_dir(None))
    folders = list(current["model_dirs"])
    added: list[str] = []
    for raw in paths:
        path = str(Path(raw).expanduser().absolute())
        status = folder_status(path)
        if not status["available"] and status["reason"] != "not_connected":
            raise ValueError(f"no folder at {path}")
        if _same(path, downloads) or any(_same(path, item) for item in folders):
            continue
        folders.append(path)
        added.append(path)
    result = save_folders(folders=folders, config_path=config_path) if added else current
    return {**result, "added": added}


def remove_folder(path: str, *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Stop searching a folder. Its files are not touched."""

    current = configured_folders(config_path)
    downloads = current["model_dir"] or str(model_cache_dir(None))
    if _same(path, downloads):
        raise ValueError(
            f"{path} is the download folder; choose another one first "
            "(mtplx folders downloads <folder>)"
        )
    folders = [item for item in current["model_dirs"] if not _same(item, path)]
    if len(folders) == len(current["model_dirs"]):
        raise ValueError(f"{path} is not one of the model folders")
    return {**save_folders(folders=folders, config_path=config_path), "removed": path}


def set_download_folder(path: str, *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Downloads and Forge builds go to ``path`` from now on. The previous
    download folder stays searched, so its packs keep showing; nothing moves."""

    target = Path(path).expanduser().absolute()
    status = folder_status(target)
    if not status["available"]:
        if status["reason"] == "not_connected":
            raise ValueError(f"{target} is on a drive that is not connected")
        if status["reason"] == "not_a_folder":
            raise ValueError(f"{target} is not a folder")
        if not target.parent.is_dir():
            raise ValueError(f"no folder at {target.parent}")
        target.mkdir()
    current = configured_folders(config_path)
    previous = current["model_dir"] or str(model_cache_dir(None))
    folders = [item for item in current["model_dirs"] if not _same(item, target)]
    if not _same(previous, target) and not any(_same(previous, item) for item in folders):
        folders.insert(0, previous)
    return save_folders(downloads=str(target), folders=folders, config_path=config_path)
