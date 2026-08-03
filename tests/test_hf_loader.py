from __future__ import annotations

import hashlib
import json
import sys
import time
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from mtplx.hf_loader import (
    cached_model_is_complete,
    cached_model_path,
    hf_token_for_download,
    hf_cache_report,
    list_cached_models,
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
        return _FakeHubResponse(chunks)


def _install_fake_hub(
    monkeypatch,
    files: dict[str, bytes | list[bytes | tuple[bytes, float]]],
    *,
    captured: dict[str, object] | None = None,
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
                    SimpleNamespace(rfilename=name, size=sum(len(item[0] if isinstance(item, tuple) else item) for item in payload) if isinstance(payload, list) else len(payload))
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
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"weights")
    (cached / "model.safetensors.incomplete").write_bytes(b"partial")

    assert cached_model_is_complete(cached) is False


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


def test_hf_token_for_download_uses_explicit_env_only(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    assert hf_token_for_download() is False

    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    assert hf_token_for_download() == "hf_explicit"


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
