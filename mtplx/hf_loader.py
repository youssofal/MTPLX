"""Hugging Face model resolution and local cache helpers."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Iterator

from mtplx.artifacts import _hf_repo_id_from_ref
from mtplx.models.laguna_config import (
    LAGUNA_S_2_1_REPO_ID,
    LAGUNA_S_2_1_REPO_BYTES,
    LAGUNA_S_2_1_REQUIRED_FILES,
    LAGUNA_S_2_1_REVISION,
    laguna_s_2_1_artifact_integrity_errors,
)


DEFAULT_MODEL_CACHE = Path("~/.mtplx/models").expanduser()
DownloadProgressCallback = Callable[[dict[str, Any]], None]
REQUIRED_MTPLX_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "mtplx_runtime.json",
)
MTP_SIDECAR_FALLBACKS = (
    "mtp.safetensors",
    "mtp/weights.safetensors",
    "model-mtp.safetensors",
)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SOURCE_MARKER_FILE = ".mtplx-source.json"
#: Written for the duration of a pull: which blob each file is fetched from.
TRANSFER_MARKER_FILE = ".mtplx-transfer.json"


@dataclass(frozen=True)
class RepoFile:
    path: str
    size_bytes: int | None
    blob_id: str | None = None
    sha256: str | None = None


def _effective_model_revision(repo_id: str, revision: str | None) -> str | None:
    if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
        if revision is not None and revision != LAGUNA_S_2_1_REVISION:
            raise ValueError(
                "Laguna-S-2.1 support is pinned to revision "
                f"{LAGUNA_S_2_1_REVISION}"
            )
        return LAGUNA_S_2_1_REVISION
    return revision


def read_source_marker(path: Path) -> dict[str, Any] | None:
    """Best-effort read of the pull provenance marker (``.mtplx-source.json``).

    Written on every successful pull since 2.9.0; older caches may have no
    marker (pre-2.9 pulls) or a two-key Laguna pin marker. Callers must treat
    a missing/short marker as "provenance unknown", never as an error.
    """

    try:
        payload = json.loads((path / SOURCE_MARKER_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _source_marker_matches(
    destination: Path,
    *,
    repo_id: str,
    revision: str | None,
) -> bool:
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return True
    payload = read_source_marker(destination)
    if payload is None:
        return False
    # Subset compare: 2.9.0 markers carry provenance fields (resolved_sha,
    # pulled_at, files) on top of the original two-key pin payload.
    return payload.get("repo_id") == repo_id and payload.get("revision") == revision


def _write_source_marker(
    destination: Path,
    *,
    repo_id: str,
    revision: str | None,
    resolved_sha: str | None = None,
    files: dict[str, dict[str, Any]] | None = None,
) -> None:
    payload: dict[str, Any] = {"repo_id": repo_id, "revision": revision}
    if resolved_sha:
        payload["resolved_sha"] = resolved_sha
        payload["pulled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            from mtplx.version import __version__ as _engine_version

            payload["engine_version"] = _engine_version
        except Exception:
            pass
        if files:
            payload["files"] = files
    (destination / SOURCE_MARKER_FILE).write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _query_repo_snapshot(
    repo_id: str, *, revision: str | None = None
) -> tuple[str | None, dict[str, dict[str, Any]] | None]:
    """Resolve the remote commit sha and per-file metadata for a repo.

    One API call serves three consumers: freshness (sha compare against the
    pull marker), download pinning (every file fetched from one commit), and
    the marker's per-file blob map (exact delta detection on update, even for
    sidecars like mtp.safetensors that the weight index never lists).
    Network failures return (None, None) so offline flows keep working.
    """

    try:
        from huggingface_hub import HfApi

        info, _token = _model_info_with_anonymous_fallback(
            HfApi(), repo_id=repo_id, revision=revision
        )
    except Exception:
        return None, None
    sha = getattr(info, "sha", None)
    files: dict[str, dict[str, Any]] = {}
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not isinstance(name, str) or not name.strip():
            continue
        entry: dict[str, Any] = {}
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            entry["size"] = size
        blob_id = getattr(sibling, "blob_id", None)
        if isinstance(blob_id, str) and blob_id:
            entry["blob_id"] = blob_id
        sha256 = _sibling_lfs_sha256(sibling)
        if sha256:
            entry["sha256"] = sha256
        files[name] = entry
    return (sha if isinstance(sha, str) and sha else None), (files or None)


def _sibling_lfs_sha256(sibling: Any) -> str | None:
    """The Hub's sha256 of an LFS blob, or None for a plain git file."""

    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and value else None


def _validate_pinned_laguna_files(destination: Path, repo_id: str) -> None:
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return
    missing_or_wrong = laguna_s_2_1_artifact_integrity_errors(destination)
    if missing_or_wrong:
        raise RuntimeError(
            "pinned Laguna snapshot is incomplete or differs from revision "
            f"{LAGUNA_S_2_1_REVISION}: "
            + ", ".join(sorted(missing_or_wrong))
        )


def _pull_validation(path: Path, repo_id: str) -> dict[str, Any]:
    validation = validate_mtplx_model_files(path)
    if repo_id.casefold() != LAGUNA_S_2_1_REPO_ID.casefold():
        return validation
    _validate_pinned_laguna_files(path, repo_id)
    return {
        **validation,
        "ok": True,
        "missing_files": [],
        "contract_error": None,
        "required_files": sorted(LAGUNA_S_2_1_REQUIRED_FILES),
        "mtp_supported": False,
        "runtime_compatibility": "native-ar-only",
    }


def _require_download_disk_headroom(
    root: Path,
    *,
    total_bytes: int | None,
    started_size_bytes: int,
) -> None:
    if total_bytes is None or total_bytes <= 0:
        return
    remaining = max(0, int(total_bytes) - max(0, int(started_size_bytes)))
    headroom = 5 * 1024**3
    try:
        free = int(shutil.disk_usage(root).free)
    except OSError:
        return
    required = remaining + headroom
    if free < required:
        raise RuntimeError(
            "insufficient free disk space for model download: "
            f"need {required / 1024**3:.1f} GiB including headroom, "
            f"have {free / 1024**3:.1f} GiB"
        )


def _query_repo_files(repo_id: str, *, revision: str | None = None) -> list[RepoFile]:
    """Return downloadable files with Hub-reported sizes when available."""

    try:
        hf_hub = importlib.import_module("huggingface_hub")
        api = hf_hub.HfApi()
    except Exception:
        return []
    try:
        info, _token = _model_info_with_anonymous_fallback(
            api, repo_id=repo_id, revision=revision
        )
    except Exception:
        return []
    siblings = getattr(info, "siblings", None) or []
    files: list[RepoFile] = []
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not isinstance(name, str) or not name.strip():
            continue
        size = getattr(sibling, "size", None)
        files.append(RepoFile(path=name, size_bytes=size if isinstance(size, int) else None))
    return files


def _query_repo_total_bytes(repo_id: str, *, revision: str | None = None) -> int | None:
    """Best-effort estimate of the remote repo's total size."""

    total = 0
    for repo_file in _query_repo_files(repo_id, revision=revision):
        if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes > 0:
            total += repo_file.size_bytes
    return total or None


@contextlib.contextmanager
def _suppress_hf_hub_progress() -> Iterator[None]:
    """Suppress Hugging Face tqdm bars while MTPLX owns download progress."""

    previous_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    disabled_via_helper = False
    try:
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            disabled_via_helper = True
        except Exception:
            pass
        yield
    finally:
        if disabled_via_helper:
            try:
                from huggingface_hub.utils import enable_progress_bars

                enable_progress_bars()
            except Exception:
                pass
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_env


def model_cache_dir(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MODEL_CACHE


def model_library_roots(
    cache_dir: str | Path | None = None,
    *,
    search_dirs: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Ordered, canonical model roots with the writable cache first.

    ``cache_dir`` retains the existing explicit/config/``MTPLX_MODEL_DIR``
    primary-root contract. Additional roots are discovery-only and come from
    the caller followed by the platform path-list in ``MTPLX_MODEL_DIRS``.
    """

    if isinstance(search_dirs, (str, Path)):
        caller_dirs: Iterable[str | Path] = (search_dirs,)
    else:
        caller_dirs = search_dirs or ()
    env_dirs = os.environ.get("MTPLX_MODEL_DIRS", "").split(os.pathsep)
    candidates: Iterable[str | Path] = (
        model_cache_dir(cache_dir),
        *caller_dirs,
        *(entry for entry in env_dirs if entry.strip()),
    )
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        root = Path(text).expanduser().resolve(strict=False)
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return tuple(roots)


def ensure_model_root(root: str | Path) -> Path:
    """Create the writable model root, naming the real problem when it cannot be.

    A library kept on an external drive that is not connected resolves to a
    path under ``/Volumes`` that no user process may create, and a read-only
    location fails the same way. The bare ``PermissionError`` names whichever
    parent refused, not the setting the user has to change, so downloads and
    Forge builds say which model directory is unavailable and why.
    """

    path = Path(root).expanduser()
    if path.is_symlink() and not path.exists():
        # The #466 layout: ~/.mtplx/models is a symlink onto an external
        # drive. Unplugged, mkdir raises FileExistsError for the link itself.
        raise RuntimeError(
            f"model directory {path} is not available: it links to "
            f"{os.readlink(path)}, whose volume is not mounted. Reconnect the "
            "drive or choose another model directory."
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise RuntimeError(
            f"model directory {path} is not available: its volume is not "
            f"mounted or the location is not writable ({exc.strerror}: "
            f"{exc.filename}). Reconnect the drive or choose another model "
            "directory."
        ) from exc
    return path


def safe_model_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "--")


def repo_id_from_model_ref(value: str) -> str | None:
    return _hf_repo_id_from_ref(value)


def cached_model_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path:
    return model_cache_dir(cache_dir) / safe_model_name(repo_id)


def hf_token_for_download() -> str | bool:
    """The Hugging Face token every MTPLX Hub call sends; ``False`` is anonymous.

    One policy for pull, update checks, and inspect, and it is the library's
    own: ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` first, then the token
    stored by ``hf auth login``. ``mtplx doctor`` reports the same resolution
    through :func:`hf_token_source`, so what it says is what pull does.
    Public repos never need a token, and a stored token the Hub rejects is
    retried anonymously (see :func:`_call_hub_with_anonymous_fallback`), so a
    stale login can never break a public pull.
    """

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    try:
        from huggingface_hub import get_token
    except Exception:
        return False
    try:
        return get_token() or False
    except Exception:
        return False


def hf_token_source() -> str | None:
    """Where :func:`hf_token_for_download` found its token: ``"environment"``,
    ``"login"`` (``hf auth login``), or ``None`` when pulls are anonymous."""

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "environment"
    return "login" if hf_token_for_download() else None


def _hub_status_code(exc: BaseException) -> int | None:
    # huggingface_hub errors carry a requests response; urllib's HTTPError
    # carries the status as ``code``.
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _call_hub_with_anonymous_fallback(
    call: Callable[[str | bool], Any], token: str | bool
) -> tuple[Any, str | bool]:
    """Run ``call(token)``; if the Hub refuses a stored token, try anonymously.

    A revoked or expired login token makes the Hub answer 401 even for public
    repos, which must not turn a public pull into "access denied". Returns the
    result with the token that actually worked, so every later request of the
    same operation sends the same credential. When the anonymous attempt fails
    too the repo really is gated or private and the original refusal is what
    the user needs to see.
    """

    try:
        return call(token), token
    except Exception as exc:
        if not (token and _hub_status_code(exc) in {401, 403}):
            raise
        try:
            return call(False), False
        except Exception:
            raise exc from None


def _model_info_with_anonymous_fallback(
    api: Any, *, repo_id: str, revision: str | None
) -> tuple[Any, str | bool]:
    return _call_hub_with_anonymous_fallback(
        lambda token: api.model_info(
            repo_id=repo_id, revision=revision, files_metadata=True, token=token
        ),
        hf_token_for_download(),
    )


def _complete_indexed_weights(path: Path, index_name: str) -> bool:
    index = path / index_name
    if not index.is_file():
        return False
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict):
        return False
    filenames = {
        name
        for name in weight_map.values()
        if isinstance(name, str) and name.strip()
    }
    if not filenames:
        return False
    for name in filenames:
        shard = path / name
        try:
            if not shard.is_file() or shard.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


_SHARD_FILENAME_RE = re.compile(r"-\d+-of-\d+", re.IGNORECASE)


def _indexed_weight_files(path: Path) -> set[str] | None:
    """Relative names of the files the weight index needs, or None when the
    checkpoint has no index."""

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / index_name
        if not index.is_file():
            continue
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        names = {
            name
            for name in (weight_map.values() if isinstance(weight_map, dict) else [])
            if isinstance(name, str) and name.strip()
        }
        names.add(index_name)
        return names
    return None


def _has_incomplete_transfers(path: Path) -> bool:
    """An interrupted transfer of a file the model needs.

    Downloads stage in-flight files as ``*.incomplete``. A partial blocks
    only when its final file has not landed and the checkpoint needs it:
    markers inside the hub's ``.cache`` bookkeeping tree, partials next to
    a landed final file (an older attempt's leftover; the downloader
    replaces its partial into the final atomically and unlinks a stale
    final before refetching) and partials of files the current weight
    index does not list (an earlier revision's shard names) are not
    transfers. Treating every stray marker as "partial" kept a
    byte-complete folder on an endless Retry.
    """

    needed = _indexed_weight_files(path)
    try:
        for marker in path.rglob("*.incomplete"):
            relative = marker.relative_to(path)
            if ".cache" in relative.parts:
                continue
            final = marker.with_name(marker.name[: -len(".incomplete")])
            try:
                if final.is_file() and final.stat().st_size > 0:
                    continue
            except OSError:
                continue
            if needed is None:
                if _complete_unindexed_weights(path):
                    continue
            elif str(final.relative_to(path)) not in needed:
                continue
            return True
    except OSError:
        pass
    return False


def _complete_unindexed_weights(path: Path) -> bool:
    for pattern in ("*.safetensors", "*.bin", "*.gguf"):
        for candidate in path.glob(pattern):
            try:
                if not candidate.is_file() or candidate.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            # A shard-named file implies a weight index the download has not
            # reached yet; shard names can sort before the index (e.g.
            # "model.safetensors-00001-of-00039.safetensors" precedes
            # "model.safetensors.index.json"), so treat the copy as partial
            # rather than as a complete single-file model.
            if _SHARD_FILENAME_RE.search(candidate.name):
                return False
            return True
    return False


def cached_model_is_complete(path: Path) -> bool:
    """Return whether a Hub cache directory is ready to run.

    ``snapshot_download(local_dir=...)`` creates the destination early. An
    interrupted pull can therefore leave config/tokenizer files plus an index,
    which looks cached even though the weight shards are missing.
    """

    if not path.is_dir():
        return False
    if _has_incomplete_transfers(path):
        return False
    # Assistant-pair bundles (Gemma 4) have no top-level config.json — the
    # weights live under target/ and assistant/ with an mtplx_pair.json
    # marker. Require both halves to be complete (QA-112).
    if (path / "mtplx_pair.json").is_file():
        return _pair_bundle_is_complete(path)
    if not (path / "config.json").is_file():
        return False
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    if any((path / name).is_file() for name in index_names):
        return any(_complete_indexed_weights(path, name) for name in index_names)
    return _complete_unindexed_weights(path)


def _pair_bundle_is_complete(path: Path) -> bool:
    """Completeness for an assistant-pair bundle (target/ + assistant/).

    Resolves the half-paths from the pair marker's declared layout and
    checks each half exactly the way a single model is checked.
    """

    try:
        from mtplx.gemma4_pair import resolve_gemma4_pair_paths
    except Exception:
        return False
    resolved = resolve_gemma4_pair_paths(path)
    if not resolved:
        return False
    for key in ("target_model", "assistant_model"):
        half = resolved.get(key)
        if not half or not cached_model_is_complete(Path(half)):
            return False
    return True


def _repo_requires_qwen_mtplx_payload(repo_id: str) -> bool:
    lower = repo_id.lower()
    return lower.startswith("youssofal/qwen3.") and "mtplx" in lower


def _cached_model_ready_for_repo(path: Path, repo_id: str) -> bool:
    if not cached_model_is_complete(path):
        return False
    if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
        if not _source_marker_matches(
            path,
            repo_id=repo_id,
            revision=LAGUNA_S_2_1_REVISION,
        ):
            return False
        try:
            _validate_pinned_laguna_files(path, repo_id)
        except RuntimeError:
            return False
    if _repo_requires_qwen_mtplx_payload(repo_id) and not validate_mtplx_model_files(path).get("ok"):
        return False
    return _repo_folder_identity_problem(path, repo_id) is None


def _repo_folder_identity_problem(path: Path, repo_id: str) -> str | None:
    """Why a complete folder under ``repo_id``'s name is another build of it, or None.

    The catalog pins what some packs must say about themselves (their own
    mtplx-pack.json), because a name alone served a research build as the
    published pack. Imported here because the catalog imports this module.
    """

    from mtplx.model_catalog import repo_folder_identity_problem

    return repo_folder_identity_problem(path, repo_id)


def _hf_home_reuse_disabled() -> bool:
    return (os.environ.get("MTPLX_DISABLE_HF_HOME_REUSE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _hf_home_env_set() -> bool:
    return any(
        bool((os.environ.get(name) or "").strip())
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")
    )


def hf_hub_cache_dir() -> Path:
    """Resolve the shared Hugging Face hub cache root dynamically."""
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = (os.environ.get(env_name) or "").strip()
        if raw:
            return Path(raw).expanduser()
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    try:
        from huggingface_hub import constants

        hub_cache = getattr(constants, "HF_HUB_CACHE", None)
        if hub_cache:
            return Path(hub_cache).expanduser()
    except Exception:
        pass
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_cache_overridden() -> bool:
    if _hf_home_env_set():
        return True
    try:
        from huggingface_hub import constants

        default = str((Path.home() / ".cache" / "huggingface" / "hub").expanduser())
        current = str(Path(getattr(constants, "HF_HUB_CACHE", "")).expanduser())
        return bool(current) and os.path.normcase(current) != os.path.normcase(default)
    except Exception:
        return False


def _should_scan_hf_hub_cache(cache_dir: str | Path | None) -> bool:
    if _hf_home_reuse_disabled():
        return False
    if cache_dir is None or _hf_cache_overridden():
        return True
    try:
        return (
            Path(cache_dir).expanduser().resolve(strict=False)
            == model_cache_dir(None).expanduser().resolve(strict=False)
        )
    except OSError:
        return False


def _resolve_hf_dir_snapshot(
    repo_root: Path,
    repo_id: str | None = None,
    *,
    require_complete: bool = True,
) -> Path | None:
    if not repo_root.is_dir():
        return None
    snapshots_dir = repo_root / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates: list[Path] = []
    seen: set[str] = set()
    refs_dir = repo_root / "refs"
    ref_files: list[Path] = []
    if refs_dir.is_dir():
        main_ref = refs_dir / "main"
        if main_ref.is_file():
            ref_files.append(main_ref)
        ref_files.extend(
            p for p in sorted(refs_dir.iterdir()) if p.is_file() and p.name != "main"
        )
    for ref_file in ref_files:
        try:
            rev = ref_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not rev or "/" in rev or "\\" in rev or rev in {".", ".."}:
            continue
        snap = snapshots_dir / rev
        key = os.path.normcase(str(snap))
        if key not in seen:
            seen.add(key)
            candidates.append(snap)
    try:
        snapshot_dirs = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        snapshot_dirs = []
    for snap in snapshot_dirs:
        key = os.path.normcase(str(snap))
        if key not in seen:
            seen.add(key)
            candidates.append(snap)

    for snap in candidates:
        if not snap.is_dir():
            continue
        if not require_complete:
            return snap
        if repo_id is not None:
            if _cached_model_ready_for_repo(snap, repo_id):
                return snap
        elif cached_model_is_complete(snap):
            return snap
    return None


def resolve_hf_repo_snapshot(
    repo_id: str,
    hub_cache: str | Path | None = None,
    *,
    require_complete: bool = True,
) -> Path | None:
    """Resolve ``repo_id`` inside a Hugging Face hub cache (`models--Org--Repo/snapshots/<rev>`)."""
    root = Path(hub_cache).expanduser() if hub_cache is not None else hf_hub_cache_dir()
    repo_dir = root / f"models--{safe_model_name(repo_id)}"
    return _resolve_hf_dir_snapshot(repo_dir, repo_id, require_complete=require_complete)


def hf_repo_id_from_cache_dir(repo_root: Path, snapshot: Path | None = None) -> str | None:
    if snapshot is not None:
        marker = read_source_marker(snapshot)
        recorded = (marker or {}).get("repo_id")
        if isinstance(recorded, str) and "/" in recorded:
            return recorded
    name = repo_root.name
    if not name.startswith("models--"):
        return None
    rest = name[len("models--") :]
    if "--" not in rest:
        return None
    org, repo = rest.split("--", 1)
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def iter_hf_cache_models(
    hub_cache: str | Path | None = None,
    *,
    require_complete: bool = True,
) -> list[tuple[str, Path]]:
    """Return ``(repo_id, snapshot_path)`` pairs discovered under a Hugging Face hub cache."""
    root = Path(hub_cache).expanduser() if hub_cache is not None else hf_hub_cache_dir()
    if not root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        snap = _resolve_hf_dir_snapshot(child, require_complete=require_complete)
        if snap is None:
            continue
        repo_id = hf_repo_id_from_cache_dir(child, snap)
        if not repo_id:
            continue
        if require_complete and not _cached_model_ready_for_repo(snap, repo_id):
            continue
        found.append((repo_id, snap))
    return found


def huggingface_cache_path(repo_id: str) -> Path | None:
    """Locate ``repo_id`` in the shared Hugging Face cache, or return ``None``.

    Honors ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE``, ``HF_HOME``, and
    ``~/.cache/huggingface/hub`` dynamically without network access.
    """

    if _hf_home_reuse_disabled():
        return None
    direct = resolve_hf_repo_snapshot(repo_id)
    if direct is not None:
        return direct
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    try:
        resolved = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None
    return resolved if resolved.is_dir() else None


def resolve_model_path(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    search_dirs: Iterable[str | Path] | None = None,
) -> Path:
    local = Path(model_ref).expanduser()
    if local.exists():
        return local
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise FileNotFoundError(f"Model path is not available locally: {local}")
    for root in model_library_roots(cache_dir, search_dirs=search_dirs):
        cached = root / safe_model_name(repo_id)
        if _cached_model_ready_for_repo(cached, repo_id):
            return cached
        # Branded local builds (forge output, `mtplx models` rows) live under
        # the bare repo basename. Root precedence wins over layout preference:
        # return the first complete copy from the ordered library list.
        if "/" in repo_id:
            branded = root / repo_id.split("/", 1)[1]
            if branded != cached and _cached_model_ready_for_repo(branded, repo_id):
                return branded
        hf_in_root = resolve_hf_repo_snapshot(repo_id, hub_cache=root)
        if hf_in_root is not None:
            return hf_in_root
    # Last local step before the download hint: the machine-wide Hugging Face
    # cache. MTPLX's own library roots stay first so existing installs resolve
    # where they always did, and the shared snapshot is held to the same
    # contract gate, so a half-downloaded copy is skipped rather than served.
    shared = huggingface_cache_path(repo_id)
    if shared is not None and _cached_model_ready_for_repo(shared, repo_id):
        return shared
    # A complete folder under this name that is another build was passed over:
    # say so, so its owner knows why the download is asked for.
    others: list[str] = []
    for root in model_library_roots(cache_dir, search_dirs=search_dirs):
        names = [safe_model_name(repo_id)] + ([repo_id.split("/", 1)[1]] if "/" in repo_id else [])
        for name in dict.fromkeys(names):
            folder = root / name
            if cached_model_is_complete(folder):
                problem = _repo_folder_identity_problem(folder, repo_id)
                if problem:
                    others.append(f"{folder} is not served: {problem}.")
    raise FileNotFoundError(
        " ".join([f"Model {repo_id} is not cached.", *others, f"Run: mtplx pull {repo_id}"])
    )


def _configured_mtp_file(contract: dict[str, Any] | None, config: dict[str, Any] | None) -> str | None:
    for source in (config, contract):
        extra = source.get("mlx_lm_extra_tensors", {}) if isinstance(source, dict) else {}
        if isinstance(extra, dict) and extra.get("mtp_file"):
            return str(extra["mtp_file"])
    if isinstance(contract, dict):
        for key in ("mtp_file", "mtp_sidecar_file"):
            value = contract.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _mtp_sidecar_candidates(path: Path, contract: dict[str, Any] | None = None) -> list[str]:
    config: dict[str, Any] | None = None
    config_path = path / "config.json"
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else None
        except Exception:
            config = None
    candidates: list[str] = []
    configured = _configured_mtp_file(contract, config)
    if configured:
        candidates.append(configured)
    candidates.extend(MTP_SIDECAR_FALLBACKS)
    result: list[str] = []
    for rel in candidates:
        if rel not in result:
            result.append(rel)
    return result


def _mtp_sidecar_exists(path: Path, contract: dict[str, Any] | None = None) -> bool:
    for rel in _mtp_sidecar_candidates(path, contract):
        try:
            if (path / rel).is_file():
                return True
        except OSError:
            continue
    return False


def validate_mtplx_model_files(path: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_MTPLX_MODEL_FILES if not (path / name).exists()]
    contract: dict[str, Any] | None = None
    contract_error: str | None = None
    contract_path = path / "mtplx_runtime.json"
    if contract_path.exists():
        try:
            loaded = json.loads(contract_path.read_text(encoding="utf-8"))
            contract = loaded if isinstance(loaded, dict) else None
        except Exception as exc:
            contract_error = str(exc)
    sidecar_candidates = _mtp_sidecar_candidates(path, contract)
    if not _mtp_sidecar_exists(path, contract):
        missing.append("mtp sidecar")
    return {
        "ok": not missing and contract_error is None,
        "required_files": list(REQUIRED_MTPLX_MODEL_FILES) + [sidecar_candidates[0]],
        "mtp_sidecar_candidates": sidecar_candidates,
        "missing_files": missing,
        "contract_present": contract_path.exists(),
        "contract_arch_id": contract.get("arch_id") if isinstance(contract, dict) else None,
        "contract_error": contract_error,
    }


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def manifest_bytes_on_disk(destination: Path, repo_files: Iterable[RepoFile]) -> int:
    """Bytes already landed for the files this download ships.

    Download progress used to be the byte count of the whole destination
    folder, so anything the folder held beyond the current manifest (shards
    from a superseded revision, staging leftovers from an interrupted hub
    transfer) counted as downloaded: the app showed more bytes than the repo
    has, at 100 percent, while still downloading. Only manifest files count
    here. A landed file counts when its size matches the Hub's (a mismatch is
    a stale copy the download discards and refetches), an in-flight
    ``*.incomplete`` partial counts up to its expected size, nothing else.
    """

    total = 0
    for repo_file in repo_files:
        try:
            target = _safe_destination_for_repo_file(destination, repo_file)
        except RuntimeError:
            continue
        expected = (
            repo_file.size_bytes
            if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes >= 0
            else None
        )
        try:
            if target.is_file():
                size = target.stat().st_size
                if expected is not None and size != expected:
                    size = 0
            else:
                partial = target.with_name(target.name + ".incomplete")
                size = partial.stat().st_size if partial.is_file() else 0
                if expected is not None:
                    size = min(size, expected)
        except OSError:
            continue
        total += size
    return total


def _repo_files_from_snapshot(
    remote_files: dict[str, dict[str, Any]] | None,
) -> list[RepoFile]:
    if not remote_files:
        return []
    return [
        RepoFile(
            path=name,
            size_bytes=entry.get("size") if isinstance(entry.get("size"), int) else None,
            blob_id=entry.get("blob_id") if isinstance(entry.get("blob_id"), str) else None,
            sha256=entry.get("sha256") if isinstance(entry.get("sha256"), str) else None,
        )
        for name, entry in remote_files.items()
    ]


def _recorded_transfer_blobs(destination: Path, repo_id: str) -> dict[str, str]:
    """Blob ids the transfer marker vouches for, by repo path."""

    try:
        recorded = json.loads((destination / TRANSFER_MARKER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(recorded, dict) or recorded.get("repo_id") != repo_id:
        return {}
    files = recorded.get("files")
    if not isinstance(files, dict):
        return {}
    return {
        path: entry["blob_id"]
        for path, entry in files.items()
        if isinstance(entry, dict) and isinstance(entry.get("blob_id"), str) and entry["blob_id"]
    }


def _discard_superseded_transfers(
    destination: Path, *, repo_id: str, repo_files: Iterable[RepoFile]
) -> None:
    """Drop the partial and landed files that do not belong to this snapshot.

    A resumed download used to append the current commit's tail onto any
    ``*.incomplete`` partial it found, whichever commit had written it, and
    accept the result on size alone: a pack repaired in place upstream came
    back as a corrupt file locally. The transfer marker records which blob
    each file is being fetched from. A partial whose blob changed is
    discarded, so is a partial nothing vouches for, and a landed file whose
    recorded blob changed goes too (a head swap keeps the name and often the
    size). Landed files without a record are kept; the size check covers
    them. Idempotent, so it runs before the resume figure is computed and
    again right before the first byte.
    """

    if not destination.is_dir():
        return
    recorded = _recorded_transfer_blobs(destination, repo_id)
    for repo_file in repo_files:
        try:
            target = _safe_destination_for_repo_file(destination, repo_file)
        except RuntimeError:
            continue
        partial = target.with_name(target.name + ".incomplete")
        recorded_blob = recorded.get(repo_file.path)
        same_blob = bool(recorded_blob) and recorded_blob == repo_file.blob_id
        if partial.is_file() and not same_blob:
            partial.unlink()
        if target.is_file() and recorded_blob and repo_file.blob_id and not same_blob:
            target.unlink()


def _record_transfer(
    destination: Path,
    *,
    repo_id: str,
    revision: str | None,
    repo_files: Iterable[RepoFile],
) -> None:
    """Write the transfer marker: which blob every file of this pull comes from.

    Written once the download has created its destination, removed when the
    pull completes, so it is only ever seen by a pull that resumes.
    """

    files = {
        repo_file.path: {"blob_id": repo_file.blob_id}
        for repo_file in repo_files
        if repo_file.blob_id
    }
    (destination / TRANSFER_MARKER_FILE).write_text(
        json.dumps({"repo_id": repo_id, "revision": revision, "files": files}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def stale_transient_bytes(destination: Path, repo_files: Iterable[RepoFile]) -> tuple[int, int]:
    """Leftover transients under ``destination`` as (bytes, file count).

    ``*.incomplete`` partials that belong to no manifest file, and anything
    under the hub cache's ``.cache`` staging tree. They are what inflated the
    download panel; they are reported so the user knows the folder holds
    them, never removed here.
    """

    if not destination.is_dir():
        return 0, 0
    manifest: set[Path] = set()
    for repo_file in repo_files:
        try:
            target = _safe_destination_for_repo_file(destination, repo_file)
        except RuntimeError:
            continue
        manifest.add(target)
        manifest.add(target.with_name(target.name + ".incomplete"))
    total = 0
    count = 0
    for child in destination.rglob("*"):
        try:
            if not child.is_file() or child in manifest:
                continue
            parts = child.relative_to(destination).parts
            if child.suffix != ".incomplete" and ".cache" not in parts:
                continue
            total += child.stat().st_size
            count += 1
        except OSError:
            continue
    return total, count


def _model_bytes_without_transients(path: Path) -> int:
    """The folder's bytes minus the ``.cache`` staging tree and partials, for
    caches whose manifest is unknown (pulls older than the 2.9 marker)."""

    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if ".cache" in child.relative_to(path).parts:
                continue
            if child.is_file() and child.suffix != ".incomplete":
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _emit_download_progress(callback: DownloadProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # Progress reporting must never break a model download.
        return


def _hub_runtime() -> tuple[Any, Callable[..., str], Callable[[], Any], Callable[..., dict[str, str]], Callable[[Any], None]]:
    """Import the small Hub surface used by the app installer.

    Tests often patch ``huggingface_hub`` with a lightweight module object, so
    this helper keeps the imports forgiving while still using the official Hub
    helpers when they are available.
    """

    try:
        hf_hub = importlib.import_module("huggingface_hub")
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for mtplx pull: {exc}") from exc

    try:
        from huggingface_hub.utils import build_hf_headers, hf_raise_for_status
    except Exception:

        def build_hf_headers(**_kwargs: Any) -> dict[str, str]:
            return {}

        def hf_raise_for_status(response: Any) -> None:
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            elif int(getattr(response, "status_code", 200)) >= 400:
                raise RuntimeError(f"Hugging Face request failed with HTTP {response.status_code}")

    required = ["hf_hub_url", "get_session", "HfApi"]
    missing = [name for name in required if not hasattr(hf_hub, name)]
    if missing:
        raise RuntimeError(
            "huggingface_hub is too old for structured mtplx pull "
            f"(missing {', '.join(missing)})"
        )
    return (
        hf_hub.HfApi,
        hf_hub.hf_hub_url,
        hf_hub.get_session,
        build_hf_headers,
        hf_raise_for_status,
    )


def _classify_pull_error(exc: BaseException, repo_id: str) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {401, 403}:
        return (
            f"Hugging Face denied access to {repo_id}. "
            "Sign in with an access token or request access, then retry."
        )
    if status_code == 404:
        return f"Hugging Face could not find {repo_id}. Check the model name, then retry."
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "Not enough disk space to finish the model download. Free space, then retry."
    certificate_failure = _certificate_verify_failure(exc)
    if certificate_failure is not None:
        return (
            "The secure connection to Hugging Face could not be verified. This "
            "usually means a company proxy inspects HTTPS traffic and signs it "
            "with its own certificate. macOS, Safari and curl trust that "
            "certificate through the keychain; Python does not read the keychain. "
            "Ask your IT team for the proxy's root certificate as a .pem file, "
            "then set SSL_CERT_FILE and REQUESTS_CA_BUNDLE to its path and retry. "
            "If the Python package `truststore` is installed next to MTPLX, the "
            "keychain is used automatically and nothing else is needed. "
            # The wrapped error's own text, not the outermost wrapper's: a bug
            # report needs the line that names the certificate problem.
            f"Details: {certificate_failure}"
        )
    return str(exc)


def _certificate_verify_failure(exc: BaseException) -> BaseException | None:
    """The TLS certificate verification failure in ``exc``'s chain, if any (#495).

    httpx and huggingface_hub wrap the ``ssl`` error, sometimes more than
    once, and some wrappers keep only its text.
    """

    import ssl

    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, ssl.SSLCertVerificationError):
            return cursor
        if "CERTIFICATE_VERIFY_FAILED" in str(cursor):
            return cursor
        cursor = cursor.__cause__ or cursor.__context__
    return None


_SYSTEM_TRUST_STATE: dict[str, Any] = {"attempted": False, "active": False, "reason": None}


def use_system_trust_store() -> dict[str, Any]:
    """Verify TLS through the operating system when ``truststore`` is present.

    Behind a TLS-inspecting proxy the proxy's root certificate is in the macOS
    keychain and not in ``certifi``, so every download fails verification
    while curl works (#495). ``truststore`` hands verification to the system
    (Security.framework on macOS), which is what pip has done since 24.2. It
    is used when it is installed and never required: MTPLX does not depend on
    it, so nothing changes for anyone who has not installed it.

    Deliberately not done: building a CA bundle from ``security
    find-certificate``. That dumps every certificate in a keychain whatever
    its trust setting, so it would trust roots an administrator marked as
    never trusted.

    ``MTPLX_SYSTEM_TRUST=0`` switches it off. Idempotent: injecting twice would
    stack the ``ssl.SSLContext`` replacement.
    """

    state = _SYSTEM_TRUST_STATE
    if state["attempted"]:
        return dict(state)
    state["attempted"] = True
    raw = os.environ.get("MTPLX_SYSTEM_TRUST", "1").strip().lower()
    if raw in {"0", "off", "false", "no"}:
        state["reason"] = "disabled_by_env"
        return dict(state)
    try:
        import truststore
    except ImportError:
        state["reason"] = "truststore_not_installed"
        return dict(state)
    truststore.inject_into_ssl()
    state["active"] = True
    state["reason"] = "truststore"
    return dict(state)


def _safe_destination_for_repo_file(destination: Path, repo_file: RepoFile) -> Path:
    raw = str(repo_file.path)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw.strip()
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise RuntimeError(f"unsafe file path in Hugging Face repo: {repo_file.path}")
    target = destination.joinpath(*posix.parts)
    if destination.is_symlink():
        raise RuntimeError(f"unsafe model destination symlink: {destination}")
    canonical_destination = destination.resolve(strict=False)
    cursor = destination
    for part in posix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(
                f"unsafe symlink component in Hugging Face repo path: {repo_file.path}"
            )
    try:
        target.parent.resolve(strict=False).relative_to(canonical_destination)
    except ValueError as exc:
        raise RuntimeError(
            f"unsafe file path in Hugging Face repo: {repo_file.path}"
        ) from exc
    return target


def _emit_current_download_size(
    callback: DownloadProgressCallback | None,
    *,
    repo_id: str,
    destination: Path,
    total_bytes: int | None,
    started_at: float,
    last_emit_at: float,
    last_emit_size: int,
    file_path: str | None = None,
    measure: Callable[[], int] | None = None,
) -> tuple[float, int]:
    now = time.monotonic()
    current_size = measure() if measure is not None else directory_size_bytes(destination)
    interval = max(0.001, now - last_emit_at)
    delta = current_size - last_emit_size
    reported_size = min(current_size, total_bytes) if total_bytes else current_size
    _emit_download_progress(
        callback,
        {
            "event": "progress",
            "repo_id": repo_id,
            "path": str(destination),
            "file": file_path,
            "size_bytes": reported_size,
            "total_bytes": total_bytes,
            "delta_bytes": delta,
            "rate_bps": float(max(0, delta)) / interval,
            "elapsed_s": now - started_at,
            "interval_s": interval,
            "stalled_s": 0,
            "message": "Downloading model files",
        },
    )
    return now, current_size


def _open_hub_stream(session: Any, url: str, headers: dict[str, str]) -> Any:
    stream = getattr(session, "stream", None)
    if callable(stream):
        return stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=60,
        )
    return session.get(url, headers=headers, stream=True, timeout=(10, 60))


def _iter_response_bytes(response: Any) -> Iterator[bytes]:
    iter_content = getattr(response, "iter_content", None)
    if callable(iter_content):
        yield from iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE)
        return
    iter_bytes = getattr(response, "iter_bytes", None)
    if callable(iter_bytes):
        yield from iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE)
        return
    raise RuntimeError("Hugging Face response does not support byte streaming")


def _download_repo_file(
    repo_file: RepoFile,
    *,
    repo_id: str,
    revision: str | None,
    destination: Path,
    session: Any,
    hf_hub_url: Callable[..., str],
    build_hf_headers: Callable[..., dict[str, str]],
    hf_raise_for_status: Callable[[Any], None],
    callback: DownloadProgressCallback | None,
    total_bytes: int | None,
    started_at: float,
    progress_interval_s: float,
    last_emit_at: float,
    last_emit_size: int,
    measure: Callable[[], int] | None = None,
    token: str | bool | None = None,
) -> tuple[float, int]:
    if token is None:
        token = hf_token_for_download()
    target = _safe_destination_for_repo_file(destination, repo_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = repo_file.size_bytes
    if expected_size is not None and target.exists() and target.stat().st_size == expected_size:
        return last_emit_at, last_emit_size
    if expected_size is None and target.exists() and target.stat().st_size > 0:
        return last_emit_at, last_emit_size

    partial = target.with_name(target.name + ".incomplete")
    if target.exists():
        # A size-mismatched final file is a stale version of a file that
        # changed upstream (e.g. a repaired index gaining vision entries),
        # not an interrupted download. Resuming from it would append the
        # remote tail onto old content and corrupt the file, so discard it.
        # Only a leftover *.incomplete partial may be range-resumed.
        target.unlink()
    existing = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and existing > expected_size:
        partial.unlink()
        existing = 0
    # The Hub publishes the sha256 of every LFS blob. Hashing the bytes as
    # they land (the resumed prefix first) turns the size-only acceptance
    # into an exact one without a second pass over the file.
    digest = hashlib.sha256() if repo_file.sha256 else None
    if digest is not None and existing > 0:
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
                digest.update(block)

    def _land(landed: Path) -> None:
        if digest is not None and digest.hexdigest() != repo_file.sha256:
            landed.unlink(missing_ok=True)
            raise RuntimeError(
                f"corrupt download for {repo_file.path}: the bytes on disk do not "
                "match the file on Hugging Face. The partial was discarded; run the pull again."
            )
        landed.replace(target)

    headers = build_hf_headers(token=token)
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    url = hf_hub_url(repo_id=repo_id, filename=repo_file.path, revision=revision)
    response_stream = _open_hub_stream(session, url, headers)
    with response_stream as response:
        status_code = int(getattr(response, "status_code", 200))
        if existing > 0 and status_code == 200:
            partial.unlink(missing_ok=True)
            existing = 0
            if digest is not None:
                digest = hashlib.sha256()
        elif existing > 0 and status_code == 416 and expected_size is not None and existing == expected_size:
            _land(partial)
            return _emit_current_download_size(
                callback,
                repo_id=repo_id,
                destination=destination,
                total_bytes=total_bytes,
                started_at=started_at,
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
                file_path=repo_file.path,
                measure=measure,
            )
        hf_raise_for_status(response)
        mode = "ab" if existing > 0 else "wb"
        with partial.open(mode) as handle:
            for chunk in _iter_response_bytes(response):
                if not chunk:
                    continue
                handle.write(chunk)
                if digest is not None:
                    digest.update(chunk)
                now = time.monotonic()
                if now - last_emit_at >= progress_interval_s:
                    last_emit_at, last_emit_size = _emit_current_download_size(
                        callback,
                        repo_id=repo_id,
                        destination=destination,
                        total_bytes=total_bytes,
                        started_at=started_at,
                        last_emit_at=last_emit_at,
                        last_emit_size=last_emit_size,
                        file_path=repo_file.path,
                        measure=measure,
                    )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"incomplete download for {repo_file.path}: "
            f"expected {expected_size} bytes, got {partial.stat().st_size}"
        )
    _land(partial)
    return _emit_current_download_size(
        callback,
        repo_id=repo_id,
        destination=destination,
        total_bytes=total_bytes,
        started_at=started_at,
        last_emit_at=last_emit_at,
        last_emit_size=last_emit_size,
        file_path=repo_file.path,
        measure=measure,
    )


def _resolve_download_backend(requested: str) -> tuple[str, str | None]:
    backend = requested.strip().lower()
    if backend not in {"auto", "python", "aria2"}:
        raise ValueError(
            "download backend must be one of: auto, python, aria2"
        )
    aria2c = shutil.which("aria2c")
    if backend == "python":
        return "python", None
    if aria2c:
        return "aria2", aria2c
    if backend == "aria2":
        raise RuntimeError(
            "aria2 download backend requested, but aria2c is not installed. "
            "Install it with `brew install aria2` or use --download-backend python."
        )
    return "python", None


def _download_repo_files_with_aria2(
    repo_files: list[RepoFile],
    *,
    executable: str,
    repo_id: str,
    revision: str | None,
    destination: Path,
    hf_hub_url: Callable[..., str],
    build_hf_headers: Callable[..., dict[str, str]],
    token: str | bool | None,
    callback: DownloadProgressCallback | None,
    total_bytes: int | None,
    started_at: float,
    progress_interval_s: float,
    last_emit_at: float,
    last_emit_size: int,
) -> tuple[float, int]:
    from mtplx.aria2_downloader import Aria2Download, run_aria2_downloads

    headers = tuple(
        f"{name}: {value}" for name, value in build_hf_headers(token=token).items()
    )
    downloads: list[Aria2Download] = []
    settled_bytes = 0
    for repo_file in repo_files:
        target = _safe_destination_for_repo_file(destination, repo_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_size = repo_file.size_bytes
        if expected_size is not None and target.exists() and target.stat().st_size == expected_size:
            settled_bytes += expected_size
            continue
        if expected_size is None and target.exists() and target.stat().st_size > 0:
            settled_bytes += target.stat().st_size
            continue

        partial = target.with_name(target.name + ".incomplete")
        if target.exists():
            target.unlink()
        if (
            expected_size is not None
            and partial.exists()
            and partial.stat().st_size > expected_size
        ):
            partial.unlink()
        downloads.append(
            Aria2Download(
                url=hf_hub_url(
                    repo_id=repo_id,
                    filename=repo_file.path,
                    revision=revision,
                ),
                output=partial,
                headers=headers,
                expected_size=expected_size,
                sha256=repo_file.sha256,
                display_name=repo_file.path,
            )
        )

    def report(completed: int, rate_bps: float, file_path: str | None) -> None:
        nonlocal last_emit_at, last_emit_size
        now = time.monotonic()
        current_size = settled_bytes + completed
        interval = max(0.001, now - last_emit_at)
        reported_size = min(current_size, total_bytes) if total_bytes else current_size
        _emit_download_progress(
            callback,
            {
                "event": "progress",
                "repo_id": repo_id,
                "path": str(destination),
                "file": file_path,
                "size_bytes": reported_size,
                "total_bytes": total_bytes,
                "delta_bytes": current_size - last_emit_size,
                "rate_bps": max(0.0, rate_bps),
                "elapsed_s": now - started_at,
                "interval_s": interval,
                "stalled_s": 0,
                "message": "Downloading model files with aria2c",
            },
        )
        last_emit_at = now
        last_emit_size = current_size

    run_aria2_downloads(
        downloads,
        executable=executable,
        progress_callback=report if callback is not None else None,
        progress_interval_s=progress_interval_s,
    )
    for download in downloads:
        partial = download.output
        if not partial.is_file():
            raise RuntimeError(
                f"aria2c did not produce a completed file for {download.display_name}"
            )
        if (
            download.expected_size is not None
            and partial.stat().st_size != download.expected_size
        ):
            raise RuntimeError(
                f"incomplete download for {download.display_name}: expected "
                f"{download.expected_size} bytes, got {partial.stat().st_size}"
            )
        target = partial.with_name(partial.name.removesuffix(".incomplete"))
        partial.replace(target)
    return _emit_current_download_size(
        callback,
        repo_id=repo_id,
        destination=destination,
        total_bytes=total_bytes,
        started_at=started_at,
        last_emit_at=last_emit_at,
        last_emit_size=last_emit_size,
        measure=lambda: manifest_bytes_on_disk(destination, repo_files),
    )


def _download_snapshot_with_structured_progress(
    *,
    repo_id: str,
    revision: str | None,
    destination: Path,
    progress_callback: DownloadProgressCallback | None,
    progress_interval_s: float,
    download_backend: str = "python",
    aria2c_path: str | None = None,
) -> tuple[Path, int | None]:
    HfApi, hf_hub_url, get_session, build_hf_headers, hf_raise_for_status = _hub_runtime()
    try:
        # The metadata call settles which credential this pull uses (the
        # resolved token, or anonymous after a rejected stored token); every
        # file below sends the same one.
        info, token = _model_info_with_anonymous_fallback(
            HfApi(), repo_id=repo_id, revision=revision
        )
    except Exception as exc:
        raise RuntimeError(_classify_pull_error(exc, repo_id)) from exc
    siblings = getattr(info, "siblings", None) or []
    repo_files: list[RepoFile] = []
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not isinstance(name, str) or not name.strip():
            continue
        size = getattr(sibling, "size", None)
        blob_id = getattr(sibling, "blob_id", None)
        repo_files.append(
            RepoFile(
                path=name,
                size_bytes=size if isinstance(size, int) else None,
                blob_id=blob_id if isinstance(blob_id, str) and blob_id else None,
                sha256=_sibling_lfs_sha256(sibling),
            )
        )
    if not repo_files:
        raise RuntimeError(f"Hugging Face repo {repo_id} did not return downloadable files.")
    _discard_superseded_transfers(destination, repo_id=repo_id, repo_files=repo_files)
    _record_transfer(destination, repo_id=repo_id, revision=revision, repo_files=repo_files)

    total_bytes = sum(
        repo_file.size_bytes
        for repo_file in repo_files
        if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes > 0
    ) or None
    session = get_session()

    def measure() -> int:
        return manifest_bytes_on_disk(destination, repo_files)

    started_at = time.monotonic()
    last_emit_at = started_at
    last_emit_size = measure()
    if download_backend == "aria2":
        if not aria2c_path:
            raise RuntimeError("aria2 download backend has no aria2c executable")
        try:
            _download_repo_files_with_aria2(
                repo_files,
                executable=aria2c_path,
                repo_id=repo_id,
                revision=revision,
                destination=destination,
                hf_hub_url=hf_hub_url,
                build_hf_headers=build_hf_headers,
                token=token,
                callback=progress_callback,
                total_bytes=total_bytes,
                started_at=started_at,
                progress_interval_s=max(0.1, progress_interval_s),
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(_classify_pull_error(exc, repo_id)) from exc
        return destination, total_bytes

    for repo_file in repo_files:
        try:
            last_emit_at, last_emit_size = _download_repo_file(
                repo_file,
                repo_id=repo_id,
                revision=revision,
                destination=destination,
                session=session,
                hf_hub_url=hf_hub_url,
                build_hf_headers=build_hf_headers,
                hf_raise_for_status=hf_raise_for_status,
                token=token,
                callback=progress_callback,
                total_bytes=total_bytes,
                started_at=started_at,
                progress_interval_s=max(0.1, progress_interval_s),
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
                measure=measure,
            )
        except Exception as exc:
            raise RuntimeError(_classify_pull_error(exc, repo_id)) from exc
    return destination, total_bytes


@dataclass(frozen=True)
class CachedModel:
    repo_id: str
    path: Path
    size_bytes: int
    has_runtime_contract: bool
    has_config: bool
    validation: dict[str, Any]
    root: Path | None = None
    root_index: int = 0
    is_primary: bool = True

    def to_dict(self) -> dict[str, Any]:
        # Per-model launch resolution promotes the quantized flagships to
        # turbo; a flat DEFAULT_PROFILE_NAME here reported "sustained" for
        # artifacts the engine never launches on sustained. Lazy import:
        # core module, resolver lives in the CLI layer.
        from mtplx.commands.public import resolved_default_profile_name_for_ref

        return {
            "repo_id": self.repo_id,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1_000_000_000, 3),
            "has_runtime_contract": self.has_runtime_contract,
            "has_config": self.has_config,
            "validation": self.validation,
            "root": str(self.root or self.path.parent),
            "root_index": self.root_index,
            "is_primary": self.is_primary,
            "recommended_profile": (
                resolved_default_profile_name_for_ref(self.path)
                if self.validation.get("ok")
                else None
            ),
            "delete_command": f"mtplx remove {self.repo_id}",
        }


def list_cached_models(
    *,
    cache_dir: str | Path | None = None,
    search_dirs: Iterable[str | Path] | None = None,
    include_hf_cache: bool | None = None,
) -> list[CachedModel]:
    rows: list[CachedModel] = []
    seen: set[str] = set()
    roots = model_library_roots(cache_dir, search_dirs=search_dirs)
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name.startswith("models--"):
                snap = _resolve_hf_dir_snapshot(child, require_complete=True)
                if snap is None:
                    continue
                repo_id = hf_repo_id_from_cache_dir(child, snap)
                if not repo_id:
                    continue
                target = snap
            else:
                target = child
                repo_id = child.name.replace("--", "/")
            identity = os.path.normcase(str(target.resolve(strict=False)))
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(
                CachedModel(
                    repo_id=repo_id,
                    path=target,
                    size_bytes=directory_size_bytes(target),
                    has_runtime_contract=(target / "mtplx_runtime.json").exists(),
                    has_config=(target / "config.json").exists(),
                    validation=validate_mtplx_model_files(target),
                    root=root,
                    root_index=root_index,
                    is_primary=root_index == 0,
                )
            )
    scan_hf = (
        (cache_dir is None and not _hf_home_reuse_disabled())
        if include_hf_cache is None
        else (include_hf_cache and not _hf_home_reuse_disabled())
    )
    if scan_hf:
        hf_hub = hf_hub_cache_dir()
        hf_index = len(roots)
        for repo_id, snap in iter_hf_cache_models(hf_hub, require_complete=True):
            identity = os.path.normcase(str(snap.resolve(strict=False)))
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(
                CachedModel(
                    repo_id=repo_id,
                    path=snap,
                    size_bytes=directory_size_bytes(snap),
                    has_runtime_contract=(snap / "mtplx_runtime.json").exists(),
                    has_config=(snap / "config.json").exists(),
                    validation=validate_mtplx_model_files(snap),
                    root=hf_hub,
                    root_index=hf_index,
                    is_primary=False,
                )
            )
    return rows


def _local_matches_remote_index(
    path: Path, repo_id: str, revision: str | None
) -> bool:
    """Best-effort remote freshness check for an explicit pull.

    A pull is a stated intent to sync, so a locally-complete copy must
    still pick up files added to the repo after the first download
    (restored vision towers, repaired contracts). The weight index is
    the cheap proxy: when it changed upstream, fall through to the
    download branch and let snapshot_download fetch only the delta.
    Network failures err on reuse so offline pulls keep working.
    """

    local_index = path / "model.safetensors.index.json"
    if not local_index.is_file():
        return True
    try:
        from huggingface_hub import hf_hub_download

        remote, _token = _call_hub_with_anonymous_fallback(
            lambda token: hf_hub_download(
                repo_id,
                "model.safetensors.index.json",
                revision=revision,
                token=token,
            ),
            hf_token_for_download(),
        )
        return Path(remote).read_bytes() == local_index.read_bytes()
    except Exception:
        return True


def pull_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    progress_callback: DownloadProgressCallback | None = None,
    progress_interval_s: float = 10.0,
    force_sync: bool = False,
    destination: Path | None = None,
    download_backend: str = "python",
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(f"pull requires a Hugging Face repo id or URL, got: {model_ref}")
    # Before the first request of any kind (the freshness query below is one).
    use_system_trust_store()
    raw_cache_dir = cache_dir
    explicit_destination = destination is not None
    revision = _effective_model_revision(repo_id, revision)
    root = ensure_model_root(model_cache_dir(cache_dir).expanduser().absolute())
    if destination is None:
        destination = cached_model_path(repo_id, cache_dir=root)
    destination = Path(destination).expanduser().absolute()
    if destination.parent.resolve() != root.resolve():
        raise ValueError(f"pull destination must be a direct child of {root}")
    if destination.is_symlink():
        raise RuntimeError(
            f"refusing to write model through top-level symlink: {destination}"
        )

    started_size = directory_size_bytes(destination)
    started_disk_bytes = started_size
    marker = read_source_marker(destination)
    remote_sha: str | None = None
    remote_files: dict[str, dict[str, Any]] | None = None
    snapshot_resolved = False

    def _resolve_remote_snapshot() -> None:
        nonlocal remote_sha, remote_files, snapshot_resolved
        if not snapshot_resolved:
            remote_sha, remote_files = _query_repo_snapshot(repo_id, revision=revision)
            snapshot_resolved = True

    def _fresh_against_remote(path: Path, path_marker: dict[str, Any] | None) -> bool:
        # A pull is a stated intent to sync. Prefer the exact commit-sha
        # compare against the pull marker — it sees every changed file,
        # including sidecars the weight index never lists (mtp.safetensors
        # head swaps were invisible to the index-only check). Legacy caches
        # without a sha marker keep the index-byte compare. Network failures
        # err on reuse so offline pulls keep working.
        local_sha = (path_marker or {}).get("resolved_sha")
        if isinstance(local_sha, str) and local_sha:
            _resolve_remote_snapshot()
            return remote_sha is None or remote_sha == local_sha
        return _local_matches_remote_index(path, repo_id, revision)

    def _try_reuse(candidate: Path, path_marker: dict[str, Any] | None, *, writable: bool) -> dict[str, Any] | None:
        if not (
            candidate.exists()
            and _cached_model_ready_for_repo(candidate, repo_id)
            and _source_marker_matches(
                candidate,
                repo_id=repo_id,
                revision=revision,
            )
            and _fresh_against_remote(candidate, path_marker)
        ):
            return None
        if writable:
            (candidate / TRANSFER_MARKER_FILE).unlink(missing_ok=True)
        validation = validate_mtplx_model_files(candidate)
        _validate_pinned_laguna_files(candidate, repo_id)
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "cached MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
        reuse_manifest = _repo_files_from_snapshot((path_marker or {}).get("files"))
        model_bytes = (
            manifest_bytes_on_disk(candidate, reuse_manifest)
            if reuse_manifest
            else _model_bytes_without_transients(candidate)
        )
        disk = directory_size_bytes(candidate)
        stale_b, stale_f = stale_transient_bytes(candidate, reuse_manifest)
        return {
            "resolved": candidate,
            "marker": path_marker,
            "model_bytes": model_bytes,
            "started_size": model_bytes,
            "disk_bytes": disk,
            "stale_bytes": stale_b,
            "stale_files": stale_f,
        }

    reused_info: dict[str, Any] | None = None
    if not force_sync:
        reused_info = _try_reuse(destination, marker, writable=True)
        if (
            reused_info is None
            and not explicit_destination
            and _should_scan_hf_hub_cache(raw_cache_dir)
        ):
            hf_candidate = huggingface_cache_path(repo_id)
            if hf_candidate is not None and hf_candidate != destination:
                reused_info = _try_reuse(
                    hf_candidate,
                    read_source_marker(hf_candidate),
                    writable=False,
                )

    if reused_info is not None:
        resolved = reused_info["resolved"]
        marker = reused_info["marker"]
        model_bytes = reused_info["model_bytes"]
        started_size = reused_info["started_size"]
        disk_bytes = reused_info["disk_bytes"]
        stale_bytes = reused_info["stale_bytes"]
        stale_files = reused_info["stale_files"]
        reused_existing = True
        resumed_existing = False
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": model_bytes,
                "total_bytes": model_bytes,
                "disk_bytes": disk_bytes,
                "stale_bytes": stale_bytes,
                "stale_files": stale_files,
                "delta_bytes": 0,
                "reused_existing": True,
            },
        )
    else:
        reused_existing = False
        selected_backend, aria2c_path = _resolve_download_backend(download_backend)
        # Pin the whole download to one resolved commit so every file comes
        # from the same snapshot even if the repo is pushed to mid-download.
        _resolve_remote_snapshot()
        manifest = _repo_files_from_snapshot(remote_files)
        download_revision = revision if revision is not None else remote_sha
        if manifest:
            # Files from a superseded snapshot go first, so the resume figure
            # counts only what this pull keeps. Then only what this download
            # ships counts toward the resume/start decision, the disk
            # headroom, and the progress the app shows.
            _discard_superseded_transfers(destination, repo_id=repo_id, repo_files=manifest)
            started_size = manifest_bytes_on_disk(destination, manifest)
            started_disk_bytes = directory_size_bytes(destination)
        resumed_existing = destination.exists() and started_size > 0
        if repo_id.casefold() == LAGUNA_S_2_1_REPO_ID.casefold():
            total_bytes: int | None = LAGUNA_S_2_1_REPO_BYTES
        elif remote_files:
            total_bytes = (
                sum(
                    entry["size"]
                    for entry in remote_files.values()
                    if isinstance(entry.get("size"), int) and entry["size"] > 0
                )
                or None
            )
        elif progress_callback is not None:
            total_bytes = _query_repo_total_bytes(repo_id, revision=download_revision)
        else:
            total_bytes = None
        for name, entry in (remote_files or {}).items():
            _safe_destination_for_repo_file(
                destination,
                RepoFile(path=name, size_bytes=entry.get("size")),
            )
        _require_download_disk_headroom(
            root,
            total_bytes=total_bytes,
            started_size_bytes=started_size,
        )

        def _landed_bytes(path: Path) -> int:
            if not manifest:
                return directory_size_bytes(path)
            landed = manifest_bytes_on_disk(path, manifest)
            return min(landed, total_bytes) if total_bytes else landed

        destination.mkdir(parents=True, exist_ok=True)
        _emit_download_progress(
            progress_callback,
            {
                "event": "resume" if resumed_existing else "start",
                "repo_id": repo_id,
                "path": str(destination),
                "size_bytes": started_size,
                "total_bytes": total_bytes,
                "disk_bytes": started_disk_bytes,
                "stale_bytes": stale_transient_bytes(destination, manifest)[0],
                "stale_files": stale_transient_bytes(destination, manifest)[1],
            },
        )
        progress_suppression = (
            _suppress_hf_hub_progress()
            if progress_callback is not None
            else contextlib.nullcontext()
        )
        with progress_suppression:
            if progress_callback is not None or selected_backend == "aria2":
                resolved, total_bytes_from_download = _download_snapshot_with_structured_progress(
                    repo_id=repo_id,
                    revision=download_revision,
                    destination=destination,
                    progress_callback=progress_callback,
                    progress_interval_s=progress_interval_s,
                    download_backend=selected_backend,
                    aria2c_path=aria2c_path,
                )
                if total_bytes_from_download:
                    total_bytes = total_bytes_from_download
            else:
                try:
                    from huggingface_hub import snapshot_download
                except Exception as exc:
                    raise RuntimeError(
                        f"huggingface_hub is required for mtplx pull: {exc}"
                    ) from exc
                path, _token = _call_hub_with_anonymous_fallback(
                    lambda token: snapshot_download(
                        repo_id=repo_id,
                        repo_type="model",
                        revision=download_revision,
                        local_dir=str(destination),
                        token=token,
                    ),
                    hf_token_for_download(),
                )
                resolved = Path(path)
        _emit_download_progress(
            progress_callback,
            {
                "event": "verifying",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": _landed_bytes(resolved),
                "total_bytes": total_bytes,
            },
        )
        validation = validate_mtplx_model_files(resolved)
        if not cached_model_is_complete(resolved):
            raise RuntimeError(
                "downloaded model is incomplete: weight shards are missing or still partial"
            )
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "downloaded MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
        _validate_pinned_laguna_files(resolved, repo_id)
        # Provenance marker on every pull (2.9.0): records the exact commit
        # and per-file blob map this cache was synced to, so update checks
        # can compare revisions instead of guessing from the weight index.
        _write_source_marker(
            resolved,
            repo_id=repo_id,
            revision=revision,
            resolved_sha=remote_sha,
            files=remote_files,
        )
        (resolved / TRANSFER_MARKER_FILE).unlink(missing_ok=True)
        final_size = _landed_bytes(resolved)
        model_bytes = final_size
        disk_bytes = directory_size_bytes(resolved)
        stale_bytes, stale_files = stale_transient_bytes(resolved, manifest)
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": final_size,
                "total_bytes": total_bytes if total_bytes else final_size,
                "disk_bytes": disk_bytes,
                "stale_bytes": stale_bytes,
                "stale_files": stale_files,
                "delta_bytes": final_size - started_size,
            },
        )
    return {
        "repo_id": repo_id,
        "path": str(resolved),
        "cache_dir": str(root),
        "revision": revision,
        "resolved_sha": (
            (marker or {}).get("resolved_sha") if reused_existing else remote_sha
        ),
        "reused_existing": reused_existing,
        "resumed_existing": resumed_existing,
        "started_size_bytes": started_size,
        "size_bytes": model_bytes,
        "disk_bytes": disk_bytes,
        "stale_bytes": stale_bytes,
        "stale_files": stale_files,
        "has_runtime_contract": (resolved / "mtplx_runtime.json").exists(),
        "has_config": (resolved / "config.json").exists(),
        "validation": _pull_validation(resolved, repo_id),
    }


def _names_existing_cache_entry(name: str, root: Path) -> bool:
    """True when ``name`` is exactly one existing direct child of ``root``."""
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or name.startswith(".")
    ):
        return False
    return os.path.lexists(root / name)


def resolve_cached_model_target(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    search_dirs: Iterable[str | Path] | None = None,
    explicit_root: bool = False,
) -> tuple[str, Path]:
    """Resolve one unambiguous, primary-root cache entry without deleting it.

    ``explicit_root`` means the caller named ``cache_dir`` on purpose (the
    CLI's ``--cache-dir``, which the app always passes). Removal only ever
    acts on the primary root, so a named root is the whole answer: copies in
    the additional folders are not candidates and cannot make the ref
    ambiguous. Without it, the "multiple installed copies ... select one
    root explicitly with --cache-dir" refusal could not be satisfied when the
    additional folders came from ``MTPLX_MODEL_DIRS`` or ``model_dirs`` in
    the config file, because those are appended whatever the caller passes.
    """

    raw_ref = str(model_ref).strip()
    roots = model_library_roots(cache_dir, search_dirs=search_dirs)
    if explicit_root:
        roots = roots[:1]
    # A ref that spells an existing entry's directory name means that entry.
    # `mtplx models` and the app both hand back entries by name, and the
    # public-id alias table must never redirect a delete from the folder the
    # user pointed at onto a first-party repo's folder (a hand-named
    # "qwen3.8-27b-mtplx-optimized-speed" resolved to
    # Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed and would have taken the
    # wrong 60 GB). The literal name then goes through the same multi-root
    # checks as any other ref.
    literal_entry = any(_names_existing_cache_entry(raw_ref, root) for root in roots)
    repo_id = None if literal_entry else repo_id_from_model_ref(raw_ref)
    if repo_id is None:
        # Not a Hugging Face id: only a cached directory name ("Org--Name" or
        # a bare branded name) is accepted. Anything carrying a path
        # separator is refused outright, so no traversal spelling needs to
        # be reasoned about individually.
        if not raw_ref or "/" in raw_ref or "\\" in raw_ref:
            raise ValueError(
                f"refusing to remove model ref {model_ref!r}: "
                "invalid cached model reference"
            )
        repo_id = raw_ref.replace("--", "/")
    if any(not segment or segment in {".", ".."} for segment in repo_id.split("/")):
        raise ValueError(
            f"refusing to remove model ref {model_ref!r}: "
            "invalid cached model reference"
        )

    matches: list[tuple[int, Path]] = []
    seen: set[str] = set()
    for root_index, root in enumerate(roots):
        if literal_entry:
            names = [raw_ref]
        else:
            names = [safe_model_name(repo_id)]
            if "/" in repo_id:
                names.append(repo_id.split("/", 1)[1])
        for name in names:
            path = root / name
            if path == root or path.parent.resolve() != root.resolve():
                raise ValueError(
                    f"refusing to remove {path}: model ref {model_ref!r} does "
                    f"not name a model directory inside {root}"
                )
            key = os.path.normcase(str(path.absolute()))
            if key in seen or not (path.exists() or path.is_symlink()):
                continue
            seen.add(key)
            matches.append((root_index, path))

    default_path = roots[0] / safe_model_name(repo_id)
    if len(matches) > 1:
        paths = ", ".join(str(path) for _, path in matches)
        raise ValueError(
            f"multiple installed copies match {repo_id}: {paths}; "
            "select one root explicitly with --cache-dir"
        )
    if not matches:
        return repo_id, default_path
    root_index, path = matches[0]
    if root_index != 0:
        raise ValueError(
            f"model is installed in discovery-only root {path.parent}; "
            "select that root explicitly with --cache-dir to remove it"
        )
    return repo_id, path


def remove_cached_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    search_dirs: Iterable[str | Path] | None = None,
    explicit_root: bool = False,
) -> dict[str, Any]:
    repo_id, path = resolve_cached_model_target(
        model_ref,
        cache_dir=cache_dir,
        search_dirs=search_dirs,
        explicit_root=explicit_root,
    )
    if not (path.exists() or path.is_symlink()):
        return {
            "repo_id": repo_id,
            "path": str(path),
            "removed": False,
            "size_bytes_removed": 0,
        }
    if path.is_symlink():
        size = 0
        path.unlink()
    elif path.is_dir():
        size = directory_size_bytes(path)
        shutil.rmtree(path)
    else:
        raise ValueError(f"cached model entry is not a directory: {path}")
    return {
        "repo_id": repo_id,
        "path": str(path),
        "removed": True,
        "size_bytes_removed": size,
    }


def hf_cache_report(
    *,
    cache_dir: str | Path | None = None,
    search_dirs: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    root = model_cache_dir(cache_dir)
    roots = model_library_roots(cache_dir, search_dirs=search_dirs)
    # The same resolver pull uses, so doctor can never report a token that
    # pull then ignores (or the other way round).
    token_source = hf_token_source()
    token_present = token_source is not None
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        free_bytes: int | None = usage.free
    except OSError:
        free_bytes = None
    return {
        "cache_dir": str(root),
        "model_roots": [str(candidate) for candidate in roots],
        "cache_exists": root.exists(),
        "cache_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "disk_free_bytes": free_bytes,
        "disk_free_gb": round(free_bytes / 1_000_000_000, 3) if free_bytes is not None else None,
        "cached_models": len(
            list_cached_models(
                cache_dir=root,
                search_dirs=roots[1:],
                include_hf_cache=False,
            )
        ),
        "token_present": token_present,
        "token_source": token_source,
        "token_used_by_pull": token_present,
        "token_policy": (
            "mtplx pull sends the HF_TOKEN / HUGGING_FACE_HUB_TOKEN token, else the "
            "`hf auth login` token, else nothing; public models never need one"
        ),
    }
