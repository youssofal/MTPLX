"""Tests for mtplx.supervisor.registry: state, LRU order, pins, idle."""

from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.supervisor.registry import (
    EngineRecord,
    EngineRegistry,
    EngineSpec,
    EngineState,
)


def _spec(model_id: str, resident_bytes: int = 1024) -> EngineSpec:
    return EngineSpec(model_id=model_id, path=Path(f"/models/{model_id}"), resident_bytes=resident_bytes)


def test_add_registers_a_record_as_installed():
    registry = EngineRegistry()
    record = registry.add(_spec("a"))
    assert record.state == EngineState.INSTALLED
    assert record.pins == 0
    assert registry.get("a") is record


def test_add_replaces_an_existing_id():
    registry = EngineRegistry()
    registry.add(_spec("a", resident_bytes=1))
    registry.set_state("a", EngineState.READY)
    replaced = registry.add(_spec("a", resident_bytes=2))
    assert replaced.state == EngineState.INSTALLED
    assert registry.get("a").spec.resident_bytes == 2


def test_get_unknown_id_returns_none():
    registry = EngineRegistry()
    assert registry.get("missing") is None


def test_resolve_ready_engine_is_loaded():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.READY)
    record, verdict = registry.resolve("a", default_id="a", strict=False)
    assert verdict == "loaded"
    assert record.spec.model_id == "a"


def test_resolve_loading_engine_is_loaded():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.LOADING)
    record, verdict = registry.resolve("a", default_id="a", strict=False)
    assert verdict == "loaded"


def test_resolve_installed_engine_is_installed():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    record, verdict = registry.resolve("a", default_id="a", strict=False)
    assert verdict == "installed"
    assert record.state == EngineState.INSTALLED


def test_resolve_failed_engine_is_installed():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.FAILED, failure_reason="boom")
    record, verdict = registry.resolve("a", default_id="a", strict=False)
    assert verdict == "installed"


def test_resolve_none_falls_back_to_default():
    registry = EngineRegistry()
    registry.add(_spec("default-model"))
    record, verdict = registry.resolve(None, default_id="default-model", strict=False)
    assert verdict == "fallback"
    assert record.spec.model_id == "default-model"


def test_resolve_unknown_falls_back_to_default():
    registry = EngineRegistry()
    registry.add(_spec("default-model"))
    record, verdict = registry.resolve("nope", default_id="default-model", strict=False)
    assert verdict == "fallback"
    assert record.spec.model_id == "default-model"


def test_resolve_unknown_strict_returns_unknown():
    registry = EngineRegistry()
    registry.add(_spec("default-model"))
    record, verdict = registry.resolve("nope", default_id="default-model", strict=True)
    assert verdict == "unknown"
    assert record is None


def test_resolve_missing_default_is_unknown():
    registry = EngineRegistry()
    record, verdict = registry.resolve(None, default_id="ghost", strict=False)
    assert verdict == "unknown"
    assert record is None


def test_resolve_draining_engine_returns_draining_not_installed():
    """C1: a DRAINING record must not be treated as loadable/installed --
    resolve() reports it as its own "draining" verdict so a caller (F2's
    proxy) can 503 it instead of JIT-reloading underneath the drain."""
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.READY)
    registry.set_state("a", EngineState.DRAINING)
    record, verdict = registry.resolve("a", default_id="a", strict=False)
    assert verdict == "draining"
    assert record.state == EngineState.DRAINING


def test_add_alias_resolves_to_canonical_record():
    registry = EngineRegistry()
    registry.add(_spec("pack-dir-name"))
    registry.set_state("pack-dir-name", EngineState.READY)
    registry.add_alias("pack-dir-name", "engine-served-id")
    record, verdict = registry.resolve("engine-served-id", default_id="pack-dir-name", strict=False)
    assert verdict == "loaded"
    assert record.spec.model_id == "pack-dir-name"
    assert "engine-served-id" in registry.get("pack-dir-name").aliases


def test_add_alias_is_idempotent():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add_alias("a", "alias-1")
    registry.add_alias("a", "alias-1")
    assert registry.get("a").aliases == ["alias-1"]


def test_add_alias_unknown_model_raises():
    registry = EngineRegistry()
    with pytest.raises(KeyError):
        registry.add_alias("missing", "alias-1")


def test_add_alias_colliding_with_existing_model_id_is_ignored():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add(_spec("b"))
    registry.add_alias("a", "b")
    assert registry.get("a").aliases == []


def test_add_alias_colliding_with_another_records_alias_is_ignored():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add(_spec("b"))
    registry.add_alias("a", "shared-alias")
    registry.add_alias("b", "shared-alias")
    assert registry.get("a").aliases == ["shared-alias"]
    assert registry.get("b").aliases == []


def test_add_alias_equal_to_own_model_id_is_a_no_op():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add_alias("a", "a")
    assert registry.get("a").aliases == []


def test_pin_increments_and_unknown_id_raises():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.pin("a")
    registry.pin("a")
    assert registry.get("a").pins == 2
    with pytest.raises(KeyError):
        registry.pin("missing")


def test_unpin_never_goes_below_zero():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.unpin("a")
    assert registry.get("a").pins == 0
    registry.pin("a")
    registry.unpin("a")
    registry.unpin("a")
    assert registry.get("a").pins == 0


def test_unpin_unknown_id_is_a_no_op():
    registry = EngineRegistry()
    registry.unpin("missing")  # must not raise


def test_touch_updates_last_used():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.get("a").last_used = 0.0
    registry.touch("a")
    assert registry.get("a").last_used > 0.0


def test_touch_unknown_id_raises():
    registry = EngineRegistry()
    with pytest.raises(KeyError):
        registry.touch("missing")


def test_set_state_updates_port_and_pid():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.LOADING, port=50001, pid=4242)
    record = registry.get("a")
    assert record.state == EngineState.LOADING
    assert record.port == 50001
    assert record.pid == 4242


def test_set_state_ready_clears_failure_reason_but_not_pins():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.pin("a")
    registry.set_state("a", EngineState.FAILED, failure_reason="crashed")
    registry.set_state("a", EngineState.READY)
    record = registry.get("a")
    assert record.failure_reason is None
    assert record.pins == 1


def test_set_state_failed_records_reason():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.FAILED, failure_reason="out_of_memory")
    assert registry.get("a").failure_reason == "out_of_memory"


def test_set_state_unknown_id_raises():
    registry = EngineRegistry()
    with pytest.raises(KeyError):
        registry.set_state("missing", EngineState.READY)


def test_lru_unpinned_excludes_non_ready_and_pinned():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add(_spec("b"))
    registry.add(_spec("c"))
    registry.set_state("a", EngineState.READY)
    registry.set_state("b", EngineState.READY)
    registry.set_state("c", EngineState.LOADING)
    registry.pin("b")
    unpinned = registry.lru_unpinned(exclude=set())
    assert [r.spec.model_id for r in unpinned] == ["a"]


def test_lru_unpinned_sorts_oldest_first():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add(_spec("b"))
    registry.set_state("a", EngineState.READY)
    registry.set_state("b", EngineState.READY)
    registry.get("a").last_used = 200.0
    registry.get("b").last_used = 100.0
    ordered = registry.lru_unpinned(exclude=set())
    assert [r.spec.model_id for r in ordered] == ["b", "a"]


def test_lru_unpinned_honors_exclude():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.READY)
    assert registry.lru_unpinned(exclude={"a"}) == []


def test_idle_ttl_zero_or_negative_returns_empty():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.set_state("a", EngineState.READY)
    registry.get("a").last_used = 0.0
    assert registry.idle(now=10_000.0, ttl_s=0, exclude=set()) == []
    assert registry.idle(now=10_000.0, ttl_s=-5, exclude=set()) == []


def test_idle_returns_only_stale_unpinned_ready_records():
    registry = EngineRegistry()
    registry.add(_spec("stale"))
    registry.add(_spec("fresh"))
    registry.add(_spec("pinned"))
    registry.set_state("stale", EngineState.READY)
    registry.set_state("fresh", EngineState.READY)
    registry.set_state("pinned", EngineState.READY)
    now = 10_000.0
    registry.get("stale").last_used = 0.0
    registry.get("fresh").last_used = now - 5.0
    registry.get("pinned").last_used = 0.0
    registry.pin("pinned")
    idle = registry.idle(now=now, ttl_s=100.0, exclude=set())
    assert [r.spec.model_id for r in idle] == ["stale"]


def test_records_lists_everything_added():
    registry = EngineRegistry()
    registry.add(_spec("a"))
    registry.add(_spec("b"))
    ids = {r.spec.model_id for r in registry.records()}
    assert ids == {"a", "b"}
