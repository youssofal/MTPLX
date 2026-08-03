from __future__ import annotations

from mtplx.diagnostics import (
    DEFAULT_SPEED_MODEL_SIZE_BYTES,
    build_diagnostics_payload,
    required_download_free_bytes,
    write_doctor_bundle,
)


def _write_minimal_complete_model(path) -> None:
    path.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "mtplx_runtime.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"model")
    (path / "mtp.safetensors").write_bytes(b"mtp")


def test_required_download_free_bytes_has_temp_and_headroom() -> None:
    required = required_download_free_bytes(DEFAULT_SPEED_MODEL_SIZE_BYTES)

    assert required > DEFAULT_SPEED_MODEL_SIZE_BYTES
    assert required >= int(DEFAULT_SPEED_MODEL_SIZE_BYTES * 2.5)


def test_diagnostics_payload_has_production_checks(tmp_path) -> None:
    payload = build_diagnostics_payload(
        model_cache=tmp_path,
        mlx_info={"mlx_error": "missing"},
        thermal_control={"available": False},
    )

    assert payload["support_matrix"]["supported"]["default_model"] == (
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2"
    )
    assert payload["support_matrix"]["supported"]["default_profile"] == "sustained"
    ids = {check["id"] for check in payload["checks"]}
    assert {
        "os.macos_version",
        "python.native_arm64",
        "python.version",
        "mlx.import",
        "resource.memory",
        "resource.model_cache_disk",
        "model.cache",
        "model.default_repo",
        "docker.binary",
        "thermal.control",
        "power.low_power_mode",
        "power.thermal_pressure",
    }.issubset(ids)


def test_default_repo_check_rejects_stale_public_namespace(tmp_path) -> None:
    payload = build_diagnostics_payload(model_cache=tmp_path)
    check = next(item for item in payload["checks"] if item["id"] == "model.default_repo")

    assert check["status"] == "pass"
    assert check["observed"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2"
    assert not check["observed"].startswith("mtplx/")


def test_model_cache_check_passes_when_startup_default_is_complete(
    tmp_path, monkeypatch
) -> None:
    local_model = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    _write_minimal_complete_model(local_model)
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", str(local_model))
    # Isolate from the developer machine's real ~/.mtplx cache so the
    # startup-default branch (not the HF-cache branch) is exercised.
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path / "isolated-empty-cache"))

    payload = build_diagnostics_payload()

    check = next(item for item in payload["checks"] if item["id"] == "model.cache")
    assert check["status"] == "pass"
    assert check["observed"]["startup_default_model"]["path"] == str(local_model)
    assert check["observed"]["startup_default_model"]["ok"] is True
    assert "normal startup path has a complete local model" in check["fix"]


def test_model_cache_check_honors_explicit_cache_dir_even_with_local_default(
    tmp_path, monkeypatch
) -> None:
    local_model = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    _write_minimal_complete_model(local_model)
    explicit_cache = tmp_path / "explicit-cache"
    explicit_cache.mkdir()
    monkeypatch.setenv("MTPLX_OPTIMIZED_SPEED_MODEL", str(local_model))

    payload = build_diagnostics_payload(
        model_cache=explicit_cache,
        include_startup_default_model=False,
    )

    check = next(item for item in payload["checks"] if item["id"] == "model.cache")
    assert check["status"] == "warn"
    assert check["observed"]["startup_default_model"] is None
    assert check["fix"] == "Download the default model before first run."


def test_write_doctor_bundle_creates_redacted_zip(tmp_path) -> None:
    report = {
        "environment": {"project_root": "/Users/example/private"},
        "host": {"python_executable": "/Users/example/.venv/bin/python"},
    }

    bundle = write_doctor_bundle(report=report, output_dir=tmp_path)

    assert bundle["redacted"] is True
    assert bundle["bundle_zip"].endswith(".zip")
    assert (tmp_path / bundle["bundle_id"] / "doctor.json").exists()
    assert (tmp_path / f"{bundle['bundle_id']}.zip").exists()
