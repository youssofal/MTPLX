"""Hugging Face model resolution and local cache helpers."""

from __future__ import annotations

import contextlib
import errno
import importlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

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


@dataclass(frozen=True)
class RepoFile:
    path: str
    size_bytes: int | None


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

        info = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=hf_token_for_download(),
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
        files[name] = entry
    return (sha if isinstance(sha, str) and sha else None), (files or None)


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
        info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=hf_token_for_download(),
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


def safe_model_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "--")


def repo_id_from_model_ref(value: str) -> str | None:
    return _hf_repo_id_from_ref(value)


def cached_model_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path:
    return model_cache_dir(cache_dir) / safe_model_name(repo_id)


def hf_token_for_download() -> str | bool:
    """Use explicit env auth only; public pulls should never need HF login."""

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False


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


def _has_incomplete_transfers(path: Path) -> bool:
    """``snapshot_download`` stages in-flight files as ``*.incomplete``.

    Markers inside the hub's ``.cache`` bookkeeping tree are ignored: they
    can outlive a successful resume, and the weight checks verify the final
    files directly. A marker next to the weights, however, means the final
    file never landed.
    """

    try:
        for marker in path.rglob("*.incomplete"):
            if ".cache" in marker.relative_to(path).parts:
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
    if _repo_requires_qwen_mtplx_payload(repo_id):
        return bool(validate_mtplx_model_files(path).get("ok"))
    return True


def resolve_model_path(model_ref: str, *, cache_dir: str | Path | None = None) -> Path:
    local = Path(model_ref).expanduser()
    if local.exists():
        return local
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise FileNotFoundError(f"Model path is not available locally: {local}")
    cached = cached_model_path(repo_id, cache_dir=cache_dir)
    if _cached_model_ready_for_repo(cached, repo_id):
        return cached
    # Branded local builds (forge output, `mtplx models` rows) live under the
    # bare repo basename, not the Org--Name snapshot layout. Bench's default
    # model selection already resolves them for the same id ("installed
    # locally"); quickstart/serve must agree, or the CLI tells a user to
    # re-download 20 GB it already lists. Same contract gate as above.
    if "/" in repo_id:
        branded = cached.parent / repo_id.split("/", 1)[1]
        if branded != cached and _cached_model_ready_for_repo(branded, repo_id):
            return branded
    raise FileNotFoundError(
        f"Model {repo_id} is not cached. Run: mtplx pull {repo_id}"
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
    return str(exc)


def _safe_destination_for_repo_file(destination: Path, repo_file: RepoFile) -> Path:
    target = destination / repo_file.path
    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise RuntimeError(f"unsafe file path in Hugging Face repo: {repo_file.path}") from exc
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
) -> tuple[float, int]:
    now = time.monotonic()
    current_size = directory_size_bytes(destination)
    interval = max(0.001, now - last_emit_at)
    delta = current_size - last_emit_size
    _emit_download_progress(
        callback,
        {
            "event": "progress",
            "repo_id": repo_id,
            "path": str(destination),
            "file": file_path,
            "size_bytes": current_size,
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
) -> tuple[float, int]:
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

    headers = build_hf_headers(token=hf_token_for_download())
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    url = hf_hub_url(repo_id=repo_id, filename=repo_file.path, revision=revision)
    response_stream = _open_hub_stream(session, url, headers)
    with response_stream as response:
        status_code = int(getattr(response, "status_code", 200))
        if existing > 0 and status_code == 200:
            partial.unlink(missing_ok=True)
            existing = 0
        elif existing > 0 and status_code == 416 and expected_size is not None and existing == expected_size:
            partial.replace(target)
            return _emit_current_download_size(
                callback,
                repo_id=repo_id,
                destination=destination,
                total_bytes=total_bytes,
                started_at=started_at,
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
                file_path=repo_file.path,
            )
        hf_raise_for_status(response)
        mode = "ab" if existing > 0 else "wb"
        with partial.open(mode + "") as handle:
            for chunk in _iter_response_bytes(response):
                if not chunk:
                    continue
                handle.write(chunk)
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
                    )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"incomplete download for {repo_file.path}: "
            f"expected {expected_size} bytes, got {partial.stat().st_size}"
        )
    partial.replace(target)
    return _emit_current_download_size(
        callback,
        repo_id=repo_id,
        destination=destination,
        total_bytes=total_bytes,
        started_at=started_at,
        last_emit_at=last_emit_at,
        last_emit_size=last_emit_size,
        file_path=repo_file.path,
    )


def _download_snapshot_with_structured_progress(
    *,
    repo_id: str,
    revision: str | None,
    destination: Path,
    progress_callback: DownloadProgressCallback | None,
    progress_interval_s: float,
) -> tuple[Path, int | None]:
    HfApi, hf_hub_url, get_session, build_hf_headers, hf_raise_for_status = _hub_runtime()
    try:
        info = HfApi().model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
            token=hf_token_for_download(),
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
        repo_files.append(RepoFile(path=name, size_bytes=size if isinstance(size, int) else None))
    if not repo_files:
        raise RuntimeError(f"Hugging Face repo {repo_id} did not return downloadable files.")

    total_bytes = sum(
        repo_file.size_bytes
        for repo_file in repo_files
        if isinstance(repo_file.size_bytes, int) and repo_file.size_bytes > 0
    ) or None
    session = get_session()
    started_at = time.monotonic()
    last_emit_at = started_at
    last_emit_size = directory_size_bytes(destination)
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
                callback=progress_callback,
                total_bytes=total_bytes,
                started_at=started_at,
                progress_interval_s=max(0.1, progress_interval_s),
                last_emit_at=last_emit_at,
                last_emit_size=last_emit_size,
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
            "recommended_profile": (
                resolved_default_profile_name_for_ref(self.path)
                if self.validation.get("ok")
                else None
            ),
            "delete_command": f"mtplx remove {self.repo_id}",
        }


def list_cached_models(*, cache_dir: str | Path | None = None) -> list[CachedModel]:
    root = model_cache_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[CachedModel] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        repo_id = child.name.replace("--", "/")
        rows.append(
            CachedModel(
                repo_id=repo_id,
                path=child,
                size_bytes=directory_size_bytes(child),
                has_runtime_contract=(child / "mtplx_runtime.json").exists(),
                has_config=(child / "config.json").exists(),
                validation=validate_mtplx_model_files(child),
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

        remote = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            revision=revision,
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
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(f"pull requires a Hugging Face repo id or URL, got: {model_ref}")
    revision = _effective_model_revision(repo_id, revision)
    root = model_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if destination is None:
        destination = cached_model_path(repo_id, cache_dir=root)

    started_size = directory_size_bytes(destination)
    marker = read_source_marker(destination)
    remote_sha: str | None = None
    remote_files: dict[str, dict[str, Any]] | None = None
    snapshot_resolved = False

    def _resolve_remote_snapshot() -> None:
        nonlocal remote_sha, remote_files, snapshot_resolved
        if not snapshot_resolved:
            remote_sha, remote_files = _query_repo_snapshot(repo_id, revision=revision)
            snapshot_resolved = True

    def _fresh_against_remote() -> bool:
        # A pull is a stated intent to sync. Prefer the exact commit-sha
        # compare against the pull marker — it sees every changed file,
        # including sidecars the weight index never lists (mtp.safetensors
        # head swaps were invisible to the index-only check). Legacy caches
        # without a sha marker keep the index-byte compare. Network failures
        # err on reuse so offline pulls keep working.
        local_sha = (marker or {}).get("resolved_sha")
        if isinstance(local_sha, str) and local_sha:
            _resolve_remote_snapshot()
            return remote_sha is None or remote_sha == local_sha
        return _local_matches_remote_index(destination, repo_id, revision)

    if (
        not force_sync
        and destination.exists()
        and _cached_model_ready_for_repo(destination, repo_id)
        and _source_marker_matches(
            destination,
            repo_id=repo_id,
            revision=revision,
        )
        and _fresh_against_remote()
    ):
        resolved = destination
        reused_existing = True
        resumed_existing = False
        validation = validate_mtplx_model_files(resolved)
        _validate_pinned_laguna_files(resolved, repo_id)
        if repo_id.lower().startswith("youssofal/qwen3.6-27b-mtplx") and not validation["ok"]:
            raise RuntimeError(
                "cached MTPLX model is incomplete: "
                + ", ".join(validation["missing_files"] or [str(validation.get("contract_error"))])
            )
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": directory_size_bytes(resolved),
                "total_bytes": directory_size_bytes(resolved),
                "delta_bytes": 0,
                "reused_existing": True,
            },
        )
    else:
        reused_existing = False
        resumed_existing = destination.exists() and started_size > 0
        # Pin the whole download to one resolved commit so every file comes
        # from the same snapshot even if the repo is pushed to mid-download.
        _resolve_remote_snapshot()
        download_revision = revision if revision is not None else remote_sha
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
        _require_download_disk_headroom(
            root,
            total_bytes=total_bytes,
            started_size_bytes=started_size,
        )
        destination.mkdir(parents=True, exist_ok=True)
        _emit_download_progress(
            progress_callback,
            {
                "event": "resume" if resumed_existing else "start",
                "repo_id": repo_id,
                "path": str(destination),
                "size_bytes": started_size,
                "total_bytes": total_bytes,
            },
        )
        progress_suppression = (
            _suppress_hf_hub_progress()
            if progress_callback is not None
            else contextlib.nullcontext()
        )
        with progress_suppression:
            if progress_callback is not None:
                resolved, total_bytes_from_download = _download_snapshot_with_structured_progress(
                    repo_id=repo_id,
                    revision=download_revision,
                    destination=destination,
                    progress_callback=progress_callback,
                    progress_interval_s=progress_interval_s,
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
                path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=download_revision,
                    local_dir=str(destination),
                    token=hf_token_for_download(),
                )
                resolved = Path(path)
        _emit_download_progress(
            progress_callback,
            {
                "event": "verifying",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": directory_size_bytes(resolved),
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
        final_size = directory_size_bytes(resolved)
        _emit_download_progress(
            progress_callback,
            {
                "event": "complete",
                "repo_id": repo_id,
                "path": str(resolved),
                "size_bytes": final_size,
                "total_bytes": total_bytes if total_bytes else final_size,
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
        "size_bytes": directory_size_bytes(resolved),
        "has_runtime_contract": (resolved / "mtplx_runtime.json").exists(),
        "has_config": (resolved / "config.json").exists(),
        "validation": _pull_validation(resolved, repo_id),
    }


def remove_cached_model(model_ref: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref) or model_ref.replace("--", "/")
    path = cached_model_path(repo_id, cache_dir=cache_dir)
    existed = path.exists()
    size = directory_size_bytes(path) if existed else 0
    if existed:
        shutil.rmtree(path)
    return {
        "repo_id": repo_id,
        "path": str(path),
        "removed": existed,
        "size_bytes_removed": size,
    }


def hf_cache_report(*, cache_dir: str | Path | None = None) -> dict[str, Any]:
    root = model_cache_dir(cache_dir)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    token_source = "environment" if token_present else None
    if not token_present:
        try:
            from huggingface_hub import get_token

            token_present = bool(get_token())
            token_source = "huggingface_hub" if token_present else None
        except Exception:
            token_present = False
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
        free_bytes: int | None = usage.free
    except OSError:
        free_bytes = None
    return {
        "cache_dir": str(root),
        "cache_exists": root.exists(),
        "cache_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "disk_free_bytes": free_bytes,
        "disk_free_gb": round(free_bytes / 1_000_000_000, 3) if free_bytes is not None else None,
        "cached_models": len(list_cached_models(cache_dir=root)),
        "token_present": token_present,
        "token_source": token_source,
    }
