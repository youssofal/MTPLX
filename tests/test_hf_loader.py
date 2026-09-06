from __future__ import annotations

import hashlib
import json
import sys
import time
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from mtplx.hf_loader import (
    RepoFile,
    _call_hub_with_anonymous_fallback,
    cached_model_is_complete,
    cached_model_path,
    directory_size_bytes,
    hf_token_for_download,
    hf_token_source,
    hf_cache_report,
    list_cached_models,
    manifest_bytes_on_disk,
    pull_model,
    remove_cached_model,
    repo_id_from_model_ref,
    resolve_model_path,
    safe_model_name,
    validate_mtplx_model_files,
)
from mtplx.profiles import (
    LEGACY_OPTIMIZED_HF_MODEL_ID,
    OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    QUALITY_HF_MODEL_ID,
)


class _FakeHubResponse:
    def __init__(self, chunks: list[bytes | tuple[bytes, float]], status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            delay = 0.0
            if isinstance(chunk, tuple):
                data, delay = chunk
            else:
                data = chunk
            if delay > 0:
                time.sleep(delay)
            yield data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHubSession:
    def __init__(self, files: dict[str, bytes | list[bytes | tuple[bytes, float]]]):
        self.files = files
        self.requests: list[dict[str, object]] = []

    def get(self, url: str, **kwargs):
        filename = url.removeprefix("fake://")
        self.requests.append({"url": url, **kwargs})
        payload = self.files[filename]
        chunks: list[bytes | tuple[bytes, float]]
        if isinstance(payload, bytes):
            chunks = [payload]
        else:
            chunks = payload
        range_header = str((kwargs.get("headers") or {}).get("Range") or "")
        if range_header.startswith("bytes=") and isinstance(payload, bytes):
            offset = int(range_header[len("bytes="):].rstrip("-"))
            if offset >= len(payload):
                return _FakeHubResponse([], status_code=416)
            return _FakeHubResponse([payload[offset:]], status_code=206)
        return _FakeHubResponse(chunks)


def _install_fake_hub(
    monkeypatch,
    files: dict[str, bytes | list[bytes | tuple[bytes, float]]],
    *,
    captured: dict[str, object] | None = None,
    blob_ids: dict[str, str] | None = None,
    sha256: dict[str, str] | None = None,
) -> _FakeHubSession:
    captured = captured if captured is not None else {}
    session = _FakeHubSession(files)
    hub = ModuleType("huggingface_hub")
    hub.__path__ = []

    class FakeHfApi:
        def model_info(self, **kwargs):
            captured["model_info_token"] = kwargs.get("token")
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(
                        rfilename=name,
                        size=sum(len(item[0] if isinstance(item, tuple) else item) for item in payload) if isinstance(payload, list) else len(payload),
                        blob_id=(blob_ids or {}).get(name),
                        lfs={"sha256": sha256[name]} if sha256 and name in sha256 else None,
                    )
                    for name, payload in files.items()
                ]
            )

    def fake_hf_hub_url(*, repo_id, filename, revision=None):
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        return f"fake://{filename}"

    hub.HfApi = FakeHfApi
    hub.hf_hub_url = fake_hf_hub_url
    hub.get_session = lambda: session
    hub.snapshot_download = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("structured progress should not use snapshot_download")
    )

    utils = ModuleType("huggingface_hub.utils")

    def fake_build_hf_headers(**kwargs):
        captured["headers_token"] = kwargs.get("token")
        return {}

    def fake_hf_raise_for_status(response):
        response.raise_for_status()

    utils.build_hf_headers = fake_build_hf_headers
    utils.hf_raise_for_status = fake_hf_raise_for_status
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return session


def test_repo_id_from_model_ref_accepts_hf_url_and_repo_id():
    assert repo_id_from_model_ref("mtplx/example") == "mtplx/example"
    assert (
        repo_id_from_model_ref("https://huggingface.co/mtplx/example/tree/main")
        == "mtplx/example"
    )
    assert repo_id_from_model_ref("models/local-model") is None


def test_repo_id_from_model_ref_maps_known_public_aliases():
    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Quality") == QUALITY_HF_MODEL_ID
    assert (
        repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Speed-V2")
        == OPTIMIZED_SPEED_V2_HF_MODEL_ID
    )
    assert (
        repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Speed")
        == OPTIMIZED_SPEED_V1_HF_MODEL_ID
    )
    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized") == LEGACY_OPTIMIZED_HF_MODEL_ID


def test_known_public_alias_wins_over_bare_cwd_folder(tmp_path: Path, monkeypatch):
    (tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality").mkdir()
    monkeypatch.chdir(tmp_path)

    assert repo_id_from_model_ref("Qwen3.6-27B-MTPLX-Optimized-Quality") == QUALITY_HF_MODEL_ID
    assert repo_id_from_model_ref("./Qwen3.6-27B-MTPLX-Optimized-Quality") is None


def test_safe_model_name_and_cache_path(tmp_path: Path):
    assert safe_model_name("mtplx/example") == "mtplx--example"
    assert cached_model_path("mtplx/example", cache_dir=tmp_path) == tmp_path / "mtplx--example"


def test_resolve_model_path_uses_cache_for_hf_refs(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"1234")

    assert resolve_model_path("mtplx/example", cache_dir=tmp_path) == cached


def test_resolve_model_path_rejects_unpinned_laguna_cache(tmp_path: Path):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_REPO_ID

    cached = cached_model_path(LAGUNA_S_2_1_REPO_ID, cache_dir=tmp_path)
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="not cached"):
        resolve_model_path(LAGUNA_S_2_1_REPO_ID, cache_dir=tmp_path)


def test_cached_model_is_complete_rejects_interrupted_indexed_download(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_rejects_partial_index_even_with_one_shard(
    tmp_path: Path,
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {'
        '"a": "model-00001-of-00002.safetensors", '
        '"b": "model-00002-of-00002.safetensors"'
        '}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_rejects_incomplete_transfer_marker(tmp_path: Path):
    # A partial whose final file has not landed is an interrupted transfer.
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.incomplete").write_bytes(b"partial")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_ignores_partial_next_to_landed_weights(tmp_path: Path):
    # A partial next to a landed final file is a leftover of an earlier
    # attempt, not a transfer: the downloader replaces its partial into the
    # final atomically and unlinks a stale final before re-fetching. Treating
    # the leftover as "partial" kept a byte-complete folder on an endless
    # Retry in the app.
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")
    (cached / "model.safetensors.incomplete").write_bytes(b"partial")

    assert cached_model_is_complete(cached) is True


def test_cached_model_is_complete_rejects_shards_that_sort_before_index(
    tmp_path: Path,
):
    # Interrupted pull of Qwen/Qwen3.5-122B-A10B: shard names like
    # "model.safetensors-00001-of-00039.safetensors" download before
    # "model.safetensors.index.json", so a cancel leaves complete shards,
    # no index, and no .incomplete marker.
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors-00001-of-00039.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is False


def test_cached_model_is_complete_accepts_single_file_model(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")

    assert cached_model_is_complete(cached) is True


def test_pull_model_reuses_complete_destination_without_redownload(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    def fail_snapshot_download(**_kwargs):
        raise AssertionError("complete cached model should not download again")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_snapshot_download),
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["path"] == str(cached)
    assert result["reused_existing"] is True
    assert result["resumed_existing"] is False


def test_pull_model_pins_laguna_revision_and_records_source(
    tmp_path: Path, monkeypatch
):
    from mtplx.models.laguna_config import (
        LAGUNA_S_2_1_REPO_ID,
        LAGUNA_S_2_1_REVISION,
        LAGUNA_S_2_1_SHARD_SIZES,
    )

    # This test exercises revision pinning and source-marker recording, not the
    # disk preflight (covered separately). Mock free space so it stays hermetic
    # regardless of the host's actual free disk.
    monkeypatch.setattr(
        "mtplx.hf_loader.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=256 * 1024**3),
    )

    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        from mtplx.models import laguna_config

        captured.update(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        weight_map = {}
        for index, (name, size) in enumerate(LAGUNA_S_2_1_SHARD_SIZES.items()):
            with (destination / name).open("wb") as handle:
                handle.truncate(size)
            weight_map[f"model.layer.{index}"] = name
        (destination / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )
        (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
        (destination / "tokenizer_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (destination / "generation_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (destination / "special_tokens_map.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (destination / "chat_template.jinja").write_text(
            "{{ messages }}",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            laguna_config,
            "LAGUNA_S_2_1_SIDECAR_SHA256",
            {
                name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                for name in laguna_config.LAGUNA_S_2_1_SIDECAR_SHA256
            },
        )
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = pull_model(LAGUNA_S_2_1_REPO_ID, cache_dir=tmp_path)

    assert captured["revision"] == LAGUNA_S_2_1_REVISION
    assert result["revision"] == LAGUNA_S_2_1_REVISION
    assert result["validation"]["ok"] is True
    assert result["validation"]["missing_files"] == []
    assert result["validation"]["runtime_compatibility"] == "native-ar-only"
    marker = json.loads(
        (Path(result["path"]) / ".mtplx-source.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker == {
        "repo_id": LAGUNA_S_2_1_REPO_ID,
        "revision": LAGUNA_S_2_1_REVISION,
    }

    with pytest.raises(ValueError, match="pinned to revision"):
        pull_model(
            LAGUNA_S_2_1_REPO_ID,
            cache_dir=tmp_path,
            revision="main",
        )


def test_pull_model_rejects_laguna_download_without_disk_headroom(
    tmp_path: Path, monkeypatch
):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_REPO_ID

    monkeypatch.setattr(
        "mtplx.hf_loader.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=8 * 1024**3),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **_kwargs: pytest.fail(
                "download started before disk-space preflight"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="free disk space"):
        pull_model(LAGUNA_S_2_1_REPO_ID, cache_dir=tmp_path)


def test_pull_model_resumes_incomplete_destination(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    download_cache = cached / ".cache" / "huggingface" / "download"
    download_cache.mkdir(parents=True)
    (download_cache / "model-00001-of-00001.safetensors.incomplete").write_bytes(
        b"partial"
    )
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["path"] == str(cached)
    assert result["reused_existing"] is False
    assert result["resumed_existing"] is True
    assert result["started_size_bytes"] > 0
    assert events[0]["event"] == "resume"
    assert "progress" in [event["event"] for event in events]
    assert [event["event"] for event in events[-2:]] == ["verifying", "complete"]


def test_pull_model_resumes_qwen_mtplx_folder_missing_required_sidecars(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / safe_model_name(QUALITY_HF_MODEL_ID)
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "tokenizer.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
            "mtp.safetensors": b"mtp",
            "mtplx_runtime.json": b"{}\n",
        },
    )
    events: list[dict] = []

    result = pull_model(
        QUALITY_HF_MODEL_ID,
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["reused_existing"] is False
    assert result["resumed_existing"] is True
    assert result["validation"]["ok"] is True
    assert [event["event"] for event in events[-2:]] == ["verifying", "complete"]


def test_pull_model_structured_stream_reports_written_bytes(
    tmp_path: Path, monkeypatch
):
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": [
                (b"a" * 16, 0.02),
                (b"a" * 48, 0.02),
            ],
        },
    )
    events: list[dict] = []

    pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0.01,
    )

    progress_events = [event for event in events if event["event"] == "progress"]
    assert any(event.get("delta_bytes", 0) > 0 for event in progress_events)
    assert all(event.get("message") == "Downloading model files" for event in progress_events)


def test_manifest_bytes_on_disk_counts_only_manifest_files(tmp_path: Path):
    destination = tmp_path / "mtplx--example"
    (destination / "sub").mkdir(parents=True)
    (destination / "config.json").write_bytes(b"{}\n")
    (destination / "sub" / "shard.safetensors.incomplete").write_bytes(b"12345")
    (destination / "stale.safetensors").write_bytes(b"0123456789")
    (destination / "model-00001-of-00002.safetensors").write_bytes(b"x" * 200)
    leftovers = destination / ".cache" / "huggingface" / "download"
    leftovers.mkdir(parents=True)
    (leftovers / "model.safetensors.incomplete").write_bytes(b"y" * 300)
    manifest = [
        RepoFile(path="config.json", size_bytes=3),
        RepoFile(path="sub/shard.safetensors", size_bytes=3),
        RepoFile(path="stale.safetensors", size_bytes=4),
        RepoFile(path="missing.json", size_bytes=9),
    ]

    assert directory_size_bytes(destination) > 500
    # landed and matching (3) + partial capped at its expected size (3);
    # the size-mismatched stale copy, the missing file, the shard from a
    # superseded revision, and the hub staging leftover count nothing.
    assert manifest_bytes_on_disk(destination, manifest) == 6


def test_pull_model_progress_never_exceeds_repo_size_with_stale_files(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "model-00001-of-00003.safetensors").write_bytes(b"s" * 200)
    leftovers = cached / ".cache" / "huggingface" / "download"
    leftovers.mkdir(parents=True)
    (leftovers / "model-00001-of-00001.safetensors.incomplete").write_bytes(b"l" * 300)
    index = b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n'
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": index,
            "model-00001-of-00001.safetensors": [
                (b"w" * 32, 0.02),
                (b"w" * 32, 0.02),
            ],
        },
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0.01,
    )

    total = 3 + len(index) + 64
    assert result["started_size_bytes"] == 0
    assert result["resumed_existing"] is False
    assert events[0]["event"] == "start"
    assert events[0]["size_bytes"] == 0
    assert directory_size_bytes(cached) > total
    sized = [event for event in events if event.get("size_bytes") is not None and event.get("total_bytes")]
    assert sized
    assert all(event["size_bytes"] <= event["total_bytes"] for event in sized)
    assert events[-1]["event"] == "complete"
    assert events[-1]["size_bytes"] == total
    assert events[-1]["total_bytes"] == total


def _stub_login_token(monkeypatch, token: str | None) -> None:
    """Pin the `hf auth login` token the real huggingface_hub would read from
    disk, so these tests never depend on this machine's login state."""

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)


def test_hf_token_resolution_is_env_then_login_then_anonymous(monkeypatch):
    # Three token policies used to coexist: pull sent only HF_TOKEN, doctor
    # reported the login token as present, inspect sent the login token. One
    # resolver now serves them all, and it is the library's own order.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    _stub_login_token(monkeypatch, None)
    assert hf_token_for_download() is False
    assert hf_token_source() is None

    _stub_login_token(monkeypatch, "hf_login")
    assert hf_token_for_download() == "hf_login"
    assert hf_token_source() == "login"

    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    assert hf_token_for_download() == "hf_explicit"
    assert hf_token_source() == "environment"

    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_env")
    assert hf_token_for_download() == "hf_legacy_env"
    assert hf_token_source() == "environment"


class _Refused(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


def test_rejected_stored_token_is_retried_anonymously_for_public_repos():
    calls: list[str | bool] = []

    def hub_call(token):
        calls.append(token)
        if token:
            raise _Refused(401)
        return "public-metadata"

    result, token_used = _call_hub_with_anonymous_fallback(hub_call, "hf_stale")

    assert result == "public-metadata"
    assert token_used is False
    assert calls == ["hf_stale", False]


def test_gated_repo_reports_the_original_refusal_after_the_anonymous_retry():
    def hub_call(token):
        raise _Refused(403 if token else 401)

    with pytest.raises(_Refused) as excinfo:
        _call_hub_with_anonymous_fallback(hub_call, "hf_valid_but_not_approved")

    assert excinfo.value.response.status_code == 403


def test_anonymous_fallback_never_retries_without_a_token_or_on_other_errors():
    calls: list[str | bool] = []

    def refused(token):
        calls.append(token)
        raise _Refused(401)

    with pytest.raises(_Refused):
        _call_hub_with_anonymous_fallback(refused, False)
    assert calls == [False]

    calls.clear()

    def offline(token):
        calls.append(token)
        raise OSError("offline")

    with pytest.raises(OSError):
        _call_hub_with_anonymous_fallback(offline, "hf_token")
    assert calls == ["hf_token"]


def test_pull_model_sends_the_login_token_when_one_is_stored(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured: dict[str, object] = {}
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
        captured=captured,
    )
    sys.modules["huggingface_hub"].get_token = lambda: "hf_login"

    pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=lambda _e: None, progress_interval_s=0)

    assert captured["model_info_token"] == "hf_login"
    assert captured["headers_token"] == "hf_login"


def test_pull_model_falls_back_to_anonymous_when_the_stored_token_is_rejected(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured: dict[str, object] = {}
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
        captured=captured,
    )
    hub = sys.modules["huggingface_hub"]
    hub.get_token = lambda: "hf_revoked"
    original_model_info = hub.HfApi.model_info
    tokens_seen: list[str | bool] = []

    def refusing_model_info(self, **kwargs):
        tokens_seen.append(kwargs.get("token"))
        if kwargs.get("token"):
            raise _Refused(401)
        return original_model_info(self, **kwargs)

    hub.HfApi.model_info = refusing_model_info

    result = pull_model(
        "mtplx/example", cache_dir=tmp_path, progress_callback=lambda _e: None, progress_interval_s=0
    )

    assert result["path"] == str(tmp_path / "mtplx--example")
    # Each metadata call (freshness marker, then the download) tried the
    # stored token once and fell back to anonymous.
    assert len(tokens_seen) >= 2
    assert tokens_seen[::2] == ["hf_revoked"] * (len(tokens_seen) // 2)
    assert tokens_seen[1::2] == [False] * (len(tokens_seen) // 2)
    # Every file fetch sends the credential that actually worked.
    assert captured["headers_token"] is False


def test_hf_cache_report_tells_the_truth_about_the_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    _stub_login_token(monkeypatch, None)
    anonymous = hf_cache_report(cache_dir=tmp_path)
    assert anonymous["token_present"] is False
    assert anonymous["token_source"] is None
    assert anonymous["token_used_by_pull"] is False

    _stub_login_token(monkeypatch, "hf_login")
    login = hf_cache_report(cache_dir=tmp_path)
    assert login["token_present"] is True
    assert login["token_source"] == "login"
    assert login["token_used_by_pull"] is True
    assert "hf auth login" in login["token_policy"]

    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    env = hf_cache_report(cache_dir=tmp_path)
    assert env["token_source"] == "environment"
    assert env["token_used_by_pull"] is True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("environment", "hugging face token: from HF_TOKEN (used by mtplx pull)"),
        ("login", "hugging face token: from `hf auth login` (used by mtplx pull)"),
        (None, "hugging face token: none (public models need none"),
    ],
)
def test_doctor_human_output_states_the_token_policy(capsys, source, expected):
    from mtplx.commands.public import _render_doctor_report

    report = {
        "environment": {},
        "huggingface": {"token_source": source, "token_present": source is not None},
        "thermal_control": {},
        "tools": {},
    }

    assert _render_doctor_report(SimpleNamespace(summary=False, deep=False), report) == 0
    assert expected in capsys.readouterr().out


def test_pull_model_downloads_public_models_anonymously_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    captured: dict[str, object] = {}
    _install_fake_hub(
        monkeypatch,
        {
            "config.json": b"{}\n",
            "model.safetensors.index.json": b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
            "model-00001-of-00001.safetensors": b"weights",
        },
        captured=captured,
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["path"] == str(tmp_path / "mtplx--example")
    assert captured["model_info_token"] is False
    assert captured["headers_token"] is False
    assert events[0]["total_bytes"] == 81


def test_resolve_model_path_reports_missing_cache(tmp_path: Path):
    try:
        resolve_model_path("mtplx/example", cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "mtplx pull mtplx/example" in str(exc)
    else:
        raise AssertionError("expected missing cache error")


def test_resolve_model_path_rejects_missing_local_path(tmp_path: Path):
    missing = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality"
    try:
        resolve_model_path(str(missing), cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "not available locally" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected missing local path error")


def test_list_and_remove_cached_models(tmp_path: Path):
    (tmp_path / ".tmp").mkdir()
    model = tmp_path / "mtplx--example"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text("{}\n", encoding="utf-8")
    (model / "small.bin").write_bytes(b"1234")

    rows = list_cached_models(cache_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0].repo_id == "mtplx/example"
    assert rows[0].has_config is True
    assert rows[0].has_runtime_contract is True
    assert rows[0].validation["missing_files"]
    assert rows[0].to_dict()["recommended_profile"] is None
    assert rows[0].size_bytes >= 4

    removed = remove_cached_model("mtplx/example", cache_dir=tmp_path)
    assert removed["removed"] is True
    assert not model.exists()


def _traversal_cache(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A ~/.mtplx-shaped tree: cache dir, a sentinel model, a sibling file."""

    home = tmp_path / "home" / ".mtplx"
    cache = home / "models"
    cache.mkdir(parents=True)
    config = home / "config.toml"
    config.write_text('model = "mtplx/example"\n', encoding="utf-8")
    model = cache / "mtplx--sentinel"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"1234")
    return home, cache, model


@pytest.mark.parametrize("ref", [".", "/", "..", "", "./", "//"])
def test_remove_cached_model_refuses_paths_outside_cache(tmp_path: Path, ref: str):
    # safe_model_name only swaps "/" for "--", so these refs used to resolve
    # onto the models cache itself or its parent (~/.mtplx: bin, config.toml,
    # session-bank, logs) and rmtree took the lot, exit 0, "removed" printed.
    home, cache, model = _traversal_cache(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        remove_cached_model(ref, cache_dir=cache)

    assert repr(ref) in str(excinfo.value)
    assert home.exists()
    assert cache.exists()
    assert (home / "config.toml").exists()
    assert (model / "weights.bin").read_bytes() == b"1234"


def test_remove_cached_model_contains_dotdot_refs_inside_cache(tmp_path: Path):
    # "../.." is not an escape: safe_model_name folds it to the literal child
    # name "..--..", which stays inside the cache. It must therefore be a
    # plain miss, not a deletion and not a traversal.
    home, cache, model = _traversal_cache(tmp_path)

    result = remove_cached_model("../..", cache_dir=cache)

    assert result["removed"] is False
    assert Path(result["path"]).parent == cache.resolve()
    assert home.exists()
    assert (home / "config.toml").exists()
    assert (model / "weights.bin").read_bytes() == b"1234"


def test_remove_cached_model_removes_legitimate_ref(tmp_path: Path):
    home, cache, model = _traversal_cache(tmp_path)

    result = remove_cached_model("mtplx/sentinel", cache_dir=cache)

    assert Path(result["path"]).parent == cache.resolve()
    assert result["repo_id"] == "mtplx/sentinel"
    assert result["removed"] is True
    assert result["size_bytes_removed"] == 4
    assert not model.exists()
    # Only the model goes; the cache and its siblings survive.
    assert cache.exists()
    assert (home / "config.toml").exists()

    # Missing-entry behaviour is unchanged: a second call is a clean miss.
    again = remove_cached_model("mtplx/sentinel", cache_dir=cache)
    assert again["removed"] is False
    assert again["size_bytes_removed"] == 0


def test_remove_cli_requires_confirmation_without_yes(tmp_path: Path, monkeypatch, capsys):
    from mtplx.cli import main

    home, cache, model = _traversal_cache(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    exit_code = main(["remove", "mtplx/sentinel", "--cache-dir", str(cache)])

    assert exit_code == 1
    assert model.exists()
    assert "--yes" in capsys.readouterr().err


def test_remove_cli_prompt_declines_and_accepts(tmp_path: Path, monkeypatch, capsys):
    from mtplx.cli import main

    home, cache, model = _traversal_cache(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert main(["remove", "mtplx/sentinel", "--cache-dir", str(cache)]) == 1
    assert model.exists()

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert main(["remove", "mtplx/sentinel", "--cache-dir", str(cache)]) == 0
    assert not model.exists()
    assert cache.exists()
    assert (home / "config.toml").exists()


def test_remove_cli_refuses_traversal_even_with_yes(tmp_path: Path, capsys):
    from mtplx.cli import main

    home, cache, model = _traversal_cache(tmp_path)

    exit_code = main(["remove", "..", "--cache-dir", str(cache), "--yes"])

    assert exit_code == 1
    assert "refusing to remove" in capsys.readouterr().err
    assert home.exists()
    assert (home / "config.toml").exists()
    assert (model / "weights.bin").read_bytes() == b"1234"


def test_remove_cached_model_unlinks_symlink_without_deleting_target(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "weights.safetensors").write_bytes(b"weights")
    cache = tmp_path / "cache"
    cache.mkdir()
    link = cache / "mtplx--example"
    link.symlink_to(external, target_is_directory=True)

    removed = remove_cached_model("mtplx/example", cache_dir=cache)

    assert removed["removed"] is True
    assert removed["size_bytes_removed"] == 0
    assert not link.exists()
    assert not link.is_symlink()
    assert (external / "weights.safetensors").read_bytes() == b"weights"


def test_remove_cached_model_refuses_non_directory_cache_entry(tmp_path: Path):
    entry = tmp_path / "mtplx--example"
    entry.write_text("not a model directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        remove_cached_model("mtplx/example", cache_dir=tmp_path)

    assert entry.read_text(encoding="utf-8") == "not a model directory"


def test_hf_cache_report_is_no_network(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    cache = tmp_path / "missing-cache"
    report = hf_cache_report(cache_dir=cache)

    assert report["cache_dir"] == str(cache)
    assert report["cache_exists"] is False
    assert report["cached_models"] == 0
    assert "token_present" in report
    assert "disk_free_bytes" in report


def test_validate_mtplx_model_files_reports_required_payload(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
    ):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text('{"arch_id": "qwen3-next-mtp"}\n', encoding="utf-8")

    validation = validate_mtplx_model_files(model)

    assert validation["ok"] is True
    assert validation["missing_files"] == []
    assert validation["contract_arch_id"] == "qwen3-next-mtp"


def test_validate_mtplx_model_files_accepts_configured_nested_mtp_sidecar(tmp_path: Path):
    model = tmp_path / "model"
    (model / "mtp").mkdir(parents=True)
    (model / "config.json").write_text(
        '{"mlx_lm_extra_tensors": {"mtp_file": "mtp/weights.safetensors"}}\n',
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "model.safetensors.index.json"):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text('{"arch_id": "qwen3-next-mtp"}\n', encoding="utf-8")
    (model / "mtp" / "weights.safetensors").write_bytes(b"mtp")

    validation = validate_mtplx_model_files(model)

    assert validation["ok"] is True
    assert validation["missing_files"] == []
    assert validation["mtp_sidecar_candidates"][0] == "mtp/weights.safetensors"


def _write_complete_single(root: Path, shards: int = 1) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    weight_map = {
        f"w{i}": f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    for name in set(weight_map.values()):
        (root / name).write_bytes(b"weights")


def test_cached_model_is_complete_accepts_assistant_pair_bundle(tmp_path: Path):
    # QA-112: Gemma-4 pair bundles have no top-level config.json; the old
    # check failed them at 100% with "weight shards missing".
    bundle = tmp_path / "Youssofal--Gemma4-MTPLX-Optimized-Speed"
    bundle.mkdir()
    (bundle / "mtplx_pair.json").write_text(
        json.dumps({"layout": {"target": "target", "assistant": "assistant"}}),
        encoding="utf-8",
    )
    _write_complete_single(bundle / "target", shards=4)
    _write_complete_single(bundle / "assistant", shards=1)

    assert cached_model_is_complete(bundle) is True


def test_cached_model_is_complete_rejects_pair_bundle_missing_assistant_shard(
    tmp_path: Path,
):
    bundle = tmp_path / "Youssofal--Gemma4-MTPLX-Optimized-Speed"
    bundle.mkdir()
    (bundle / "mtplx_pair.json").write_text(
        json.dumps({"layout": {"target": "target", "assistant": "assistant"}}),
        encoding="utf-8",
    )
    _write_complete_single(bundle / "target", shards=4)
    # assistant half: index references a shard that never downloaded.
    assistant = bundle / "assistant"
    assistant.mkdir()
    (assistant / "config.json").write_text("{}\n", encoding="utf-8")
    (assistant / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors"}}), encoding="utf-8"
    )

    assert cached_model_is_complete(bundle) is False


# --- pull provenance markers + sha freshness (2.9.0 model updater) ---------


def _write_complete_pack(root: Path, name: str = "mtplx--example") -> Path:
    pack = root / name
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "config.json").write_text("{}\n", encoding="utf-8")
    (pack / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (pack / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return pack


def _install_sha_hub(
    monkeypatch,
    *,
    sha: str | None,
    files: dict[str, tuple[int, str]] | None = None,
    snapshot_writer=None,
    captured: dict | None = None,
):
    captured = captured if captured is not None else {}

    class FakeHfApi:
        def model_info(self, **kwargs):
            captured["model_info_revision"] = kwargs.get("revision")
            if sha is None:
                raise RuntimeError("offline")
            siblings = [
                SimpleNamespace(rfilename=name, size=size, blob_id=blob)
                for name, (size, blob) in (files or {}).items()
            ]
            return SimpleNamespace(sha=sha, siblings=siblings)

    def fail_snapshot(**_kwargs):
        raise AssertionError("snapshot_download must not run for this case")

    def snapshot(**kwargs):
        captured["snapshot_revision"] = kwargs.get("revision")
        return snapshot_writer(**kwargs)

    hub = SimpleNamespace(
        HfApi=FakeHfApi,
        snapshot_download=snapshot if snapshot_writer else fail_snapshot,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return captured


def test_pull_model_records_provenance_marker_and_pins_commit(
    tmp_path: Path, monkeypatch
):
    def writer(**kwargs):
        destination = Path(kwargs["local_dir"])
        _write_complete_pack(destination.parent, destination.name)
        return str(destination)

    captured = _install_sha_hub(
        monkeypatch,
        sha="commit-aaa",
        files={"model-00001-of-00001.safetensors": (7, "blob-1")},
        snapshot_writer=writer,
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert captured["snapshot_revision"] == "commit-aaa"
    assert result["resolved_sha"] == "commit-aaa"
    marker = json.loads(
        (Path(result["path"]) / ".mtplx-source.json").read_text(encoding="utf-8")
    )
    assert marker["repo_id"] == "mtplx/example"
    assert marker["revision"] is None
    assert marker["resolved_sha"] == "commit-aaa"
    assert marker["files"]["model-00001-of-00001.safetensors"]["blob_id"] == "blob-1"
    assert "engine_version" in marker and "pulled_at" in marker


def test_pull_model_marker_sha_stale_triggers_sync(tmp_path: Path, monkeypatch):
    pack = _write_complete_pack(tmp_path)
    (pack / ".mtplx-source.json").write_text(
        json.dumps(
            {"repo_id": "mtplx/example", "revision": None, "resolved_sha": "commit-aaa"}
        ),
        encoding="utf-8",
    )

    def writer(**kwargs):
        return str(Path(kwargs["local_dir"]))

    captured = _install_sha_hub(
        monkeypatch,
        sha="commit-bbb",
        files={"model-00001-of-00001.safetensors": (7, "blob-2")},
        snapshot_writer=writer,
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["reused_existing"] is False
    assert captured["snapshot_revision"] == "commit-bbb"
    marker = json.loads((pack / ".mtplx-source.json").read_text(encoding="utf-8"))
    assert marker["resolved_sha"] == "commit-bbb"


def test_pull_model_marker_sha_current_reuses(tmp_path: Path, monkeypatch):
    pack = _write_complete_pack(tmp_path)
    (pack / ".mtplx-source.json").write_text(
        json.dumps(
            {"repo_id": "mtplx/example", "revision": None, "resolved_sha": "commit-aaa"}
        ),
        encoding="utf-8",
    )
    _install_sha_hub(monkeypatch, sha="commit-aaa")

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["reused_existing"] is True
    assert result["resolved_sha"] == "commit-aaa"


def test_pull_model_marker_sha_offline_errs_on_reuse(tmp_path: Path, monkeypatch):
    pack = _write_complete_pack(tmp_path)
    (pack / ".mtplx-source.json").write_text(
        json.dumps(
            {"repo_id": "mtplx/example", "revision": None, "resolved_sha": "commit-aaa"}
        ),
        encoding="utf-8",
    )
    _install_sha_hub(monkeypatch, sha=None)

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["reused_existing"] is True


def test_pull_model_force_sync_skips_reuse(tmp_path: Path, monkeypatch):
    pack = _write_complete_pack(tmp_path)
    (pack / ".mtplx-source.json").write_text(
        json.dumps(
            {"repo_id": "mtplx/example", "revision": None, "resolved_sha": "commit-aaa"}
        ),
        encoding="utf-8",
    )

    def writer(**kwargs):
        return str(Path(kwargs["local_dir"]))

    captured = _install_sha_hub(
        monkeypatch,
        sha="commit-aaa",
        files={"model-00001-of-00001.safetensors": (7, "blob-1")},
        snapshot_writer=writer,
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path, force_sync=True)

    assert result["reused_existing"] is False
    assert captured["snapshot_revision"] == "commit-aaa"


_SHARD = "model-00001-of-00001.safetensors"
_INDEX = b'{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n'


def _remote_pack(weights: bytes) -> dict[str, bytes]:
    return {"config.json": b"{}\n", "model.safetensors.index.json": _INDEX, _SHARD: weights}


def _shard_requests(session) -> list[dict]:
    return [request for request in session.requests if request["url"].endswith(_SHARD)]


def test_pull_model_discards_a_partial_from_a_superseded_blob(tmp_path: Path, monkeypatch):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / f"{_SHARD}.incomplete").write_bytes(b"stale-")
    (cached / ".mtplx-transfer.json").write_text(
        json.dumps({"repo_id": "mtplx/example", "revision": "old", "files": {_SHARD: {"blob_id": "blob-old"}}}),
        encoding="utf-8",
    )
    session = _install_fake_hub(
        monkeypatch, _remote_pack(b"fresh-weights"), blob_ids={_SHARD: "blob-new"}
    )
    events: list[dict] = []

    pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=events.append, progress_interval_s=0)

    assert (cached / _SHARD).read_bytes() == b"fresh-weights"
    assert "Range" not in (_shard_requests(session)[0].get("headers") or {})
    assert events[0]["event"] == "start"
    assert events[0]["size_bytes"] == 0
    assert not (cached / ".mtplx-transfer.json").exists()


def test_pull_model_resumes_a_partial_from_the_same_blob_and_verifies_it(tmp_path: Path, monkeypatch):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / f"{_SHARD}.incomplete").write_bytes(b"fresh-")
    (cached / ".mtplx-transfer.json").write_text(
        json.dumps({"repo_id": "mtplx/example", "revision": None, "files": {_SHARD: {"blob_id": "blob-1"}}}),
        encoding="utf-8",
    )
    session = _install_fake_hub(
        monkeypatch,
        _remote_pack(b"fresh-weights"),
        blob_ids={_SHARD: "blob-1"},
        sha256={_SHARD: hashlib.sha256(b"fresh-weights").hexdigest()},
    )
    events: list[dict] = []

    result = pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=events.append, progress_interval_s=0)

    assert result["resumed_existing"] is True
    assert events[0]["event"] == "resume"
    assert events[0]["size_bytes"] == len(b"fresh-")
    assert (_shard_requests(session)[0]["headers"] or {}).get("Range") == "bytes=6-"
    assert (cached / _SHARD).read_bytes() == b"fresh-weights"
    assert not (cached / f"{_SHARD}.incomplete").exists()


def test_pull_model_discards_a_partial_nothing_vouches_for(tmp_path: Path, monkeypatch):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / f"{_SHARD}.incomplete").write_bytes(b"who-knows")
    session = _install_fake_hub(monkeypatch, _remote_pack(b"fresh-weights"), blob_ids={_SHARD: "blob-1"})

    pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=lambda _e: None, progress_interval_s=0)

    assert "Range" not in (_shard_requests(session)[0].get("headers") or {})
    assert (cached / _SHARD).read_bytes() == b"fresh-weights"


def test_pull_model_replaces_a_landed_file_whose_blob_changed(tmp_path: Path, monkeypatch):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / _SHARD).write_bytes(b"old-weights!!")  # same size as the new blob
    (cached / ".mtplx-transfer.json").write_text(
        json.dumps({"repo_id": "mtplx/example", "revision": "old", "files": {_SHARD: {"blob_id": "blob-old"}}}),
        encoding="utf-8",
    )
    _install_fake_hub(monkeypatch, _remote_pack(b"fresh-weights"), blob_ids={_SHARD: "blob-new"})

    pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=lambda _e: None, progress_interval_s=0)

    assert (cached / _SHARD).read_bytes() == b"fresh-weights"


def test_pull_model_rejects_a_landed_file_whose_sha256_differs(tmp_path: Path, monkeypatch):
    cached = tmp_path / "mtplx--example"
    _install_fake_hub(
        monkeypatch, _remote_pack(b"fresh-weights"), blob_ids={_SHARD: "blob-1"}, sha256={_SHARD: "0" * 64}
    )

    with pytest.raises(RuntimeError, match="corrupt download"):
        pull_model("mtplx/example", cache_dir=tmp_path, progress_callback=lambda _e: None, progress_interval_s=0)

    assert not (cached / _SHARD).exists()
    assert not (cached / f"{_SHARD}.incomplete").exists()
