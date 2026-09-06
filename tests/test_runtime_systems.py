from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mtplx.runtime_systems import (
    RuntimeSystemsRegistry,
    install_runtime_systems_endpoint,
    runtime_systems_snapshot,
)


def test_update_replaces_one_provider_neutral_status() -> None:
    registry = RuntimeSystemsRegistry(clock=lambda: 12.5)
    source = {"enabled": True, "metrics": {"hits": 3}}

    revision = registry.update("cache.primary", source)
    source["metrics"]["hits"] = 99

    payload = registry.snapshot()
    assert revision == 1
    assert payload == {
        "ts": 12.5,
        "revision": 1,
        "updated_at_s": 12.5,
        "system_count": 1,
        "systems": {
            "cache.primary": {
                "revision": 1,
                "updated_at_s": 12.5,
                "status": {"enabled": True, "metrics": {"hits": 3}},
            }
        },
    }


def test_snapshot_is_detached_from_registry_state() -> None:
    registry = RuntimeSystemsRegistry()
    registry.update("scheduler", {"queue": {"depth": 2}})

    first = registry.snapshot()
    first["systems"]["scheduler"]["status"]["queue"]["depth"] = 500

    assert registry.snapshot()["systems"]["scheduler"]["status"]["queue"]["depth"] == 2


def test_registry_bounds_names_payloads_and_cardinality() -> None:
    registry = RuntimeSystemsRegistry(max_systems=1, max_status_bytes=16)
    registry.update("valid-name", {"ok": True})

    with pytest.raises(ValueError, match="limited"):
        registry.update("another", {})
    with pytest.raises(ValueError, match="system name"):
        registry.update("not valid", {})
    with pytest.raises(ValueError, match="max_status_bytes"):
        registry.update("valid-name", {"value": "x" * 20})
    with pytest.raises(ValueError, match="JSON-compatible"):
        registry.update("valid-name", {"value": object()})


def test_concurrent_updates_produce_a_consistent_snapshot() -> None:
    registry = RuntimeSystemsRegistry(max_systems=16)

    def publish(index: int) -> None:
        for value in range(100):
            registry.update(f"worker.{index}", {"value": value})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(8)))

    payload = registry.snapshot()
    assert payload["revision"] == 800
    assert payload["system_count"] == 8
    assert all(item["status"]["value"] == 99 for item in payload["systems"].values())


def test_remove_advances_revision_only_when_present() -> None:
    registry = RuntimeSystemsRegistry()
    registry.update("worker", {"ready": True})

    assert registry.remove("worker") is True
    assert registry.remove("worker") is False
    assert registry.snapshot()["revision"] == 2


def test_missing_registry_returns_an_empty_snapshot() -> None:
    payload = runtime_systems_snapshot(SimpleNamespace())
    assert payload["system_count"] == 0
    assert payload["systems"] == {}


def test_http_endpoint_returns_the_live_registry_snapshot() -> None:
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    state = SimpleNamespace(runtime_systems=RuntimeSystemsRegistry())
    state.runtime_systems.update("decoder", {"ready": True})
    app = fastapi.FastAPI()
    install_runtime_systems_endpoint(app, state)

    response = testclient.TestClient(app).get("/v1/mtplx/systems")

    assert response.status_code == 200
    assert response.json()["systems"]["decoder"]["status"] == {"ready": True}
