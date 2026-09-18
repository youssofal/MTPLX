"""Live provider contract, including the actual production FastAPI wiring."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from mtplx.runtime_systems import RuntimeSystemsRegistry
from mtplx.server.runtime_status import ServingStatusProvider


def state():
    return SimpleNamespace(
        runtime_systems=RuntimeSystemsRegistry(),
        runtime=SimpleNamespace(mtp_enabled=True),
        args=SimpleNamespace(
            generation_mode="mtp", model="/private/model", api_key="secret"
        ),
        foreground_active=0,
        requests_completed=0,
        requests_cancelled=0,
    )


def status(value):
    return value.runtime_systems.snapshot()["systems"]["serving"]["status"]


def test_provider_tracks_existing_serving_lifecycle_without_mutating_it():
    value = state()
    provider = ServingStatusProvider(value)
    provider.publish()
    assert status(value)["phase"] == "idle"
    value.foreground_active = 2
    value.requests_completed = 7
    value.requests_cancelled = 1
    provider.publish()
    assert status(value)["phase"] == "busy"
    assert status(value)["active_requests"] == 2
    assert status(value)["requests_completed"] == 7
    assert status(value)["requests_cancelled"] == 1
    assert value.foreground_active == 2
    value.runtime = None
    provider.publish()
    assert status(value)["available"] is False
    assert status(value)["mtp_enabled"] is False


def test_provider_counts_dashboard_and_foreground_without_model_lock():
    value = state()
    value.foreground_count = lambda: 1
    value.dashboard = SimpleNamespace(in_flight=SimpleNamespace(count=lambda: 3))

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("Provider must never acquire the model lock")

    value.model_lock = ForbiddenLock()
    ServingStatusProvider(value).publish()
    assert status(value)["active_requests"] == 3


def test_failed_refresh_replaces_stale_status_and_redacts_errors():
    value = state()
    provider = ServingStatusProvider(value)
    provider.publish()

    def broken():
        raise RuntimeError("/private/path?api_key=secret")

    value.foreground_count = broken
    provider.publish()
    assert status(value) == {
        "available": False,
        "enabled": False,
        "wired": True,
        "phase": "unavailable",
        "reason": "status_read_failed",
    }
    assert "secret" not in json.dumps(status(value))
    value.foreground_count = lambda: 0
    provider.publish()
    assert status(value)["phase"] == "idle"


def test_unknown_values_are_not_fabricated_as_valid_metrics():
    value = state()
    value.foreground_active = None
    value.requests_completed = True
    value.requests_cancelled = -1
    value.args.generation_mode = "secret"
    ServingStatusProvider(value).publish()
    result = status(value)
    assert result["phase"] == "unknown"
    assert result["active_requests"] is None
    assert result["requests_completed"] is None
    assert result["requests_cancelled"] is None
    assert result["generation_mode"] is None
    assert "private" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_released_runtime_is_not_advertised_available():
    value = state()
    value.aime_parent_runtime_released = True
    ServingStatusProvider(value).publish()
    assert status(value)["phase"] == "unavailable"
    assert status(value)["enabled"] is False


def test_concurrent_refresh_is_bounded_and_snapshots_are_detached():
    value = state()
    provider = ServingStatusProvider(value)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: provider.publish(), range(64)))
    snapshot = value.runtime_systems.snapshot()
    assert snapshot["system_count"] == 1
    assert snapshot["revision"] == 64
    snapshot["systems"]["serving"]["status"]["active_requests"] = 999
    assert status(value)["active_requests"] == 0


def test_production_app_refreshes_real_state_and_keeps_auth_boundary():
    from test_server_openai import _fake_state

    from mtplx.server.openai import create_app

    value = _fake_state(api_key="test-key")
    value.foreground_count = lambda: value.foreground_active
    value.foreground_active = 0
    value.requests_completed = 4
    client = TestClient(create_app(value))
    assert client.get("/v1/mtplx/systems").status_code == 401
    assert value.runtime_systems.snapshot()["system_count"] == 0
    response = client.get(
        "/v1/mtplx/systems", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200
    assert response.json()["systems"]["serving"]["status"]["requests_completed"] == 4
    value.foreground_active = 3
    value.requests_completed = 5
    response = client.get(
        "/v1/mtplx/systems", headers={"Authorization": "Bearer test-key"}
    )
    payload = response.json()["systems"]["serving"]["status"]
    assert payload["phase"] == "busy"
    assert payload["active_requests"] >= 3
    assert payload["requests_completed"] == 5
    assert "test-key" not in response.text


def test_provider_import_does_not_import_mlx():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from mtplx.server.runtime_status import ServingStatusProvider; "
                "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
