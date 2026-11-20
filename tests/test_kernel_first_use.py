import json
import os
from pathlib import Path

import pytest

from mtplx.kernels.first_use import Mismatch, ProofCache


def test_only_a_matching_gpu_pack_and_shape_reuses_a_proof(tmp_path):
    calls = []
    def prove():
        calls.append(1)
        return ["packed"], None

    for gpu, shape in (("a", 4), ("a", 4), ("b", 4), ("a", 3)):
        assert ProofCache(gpu, tmp_path).decide("projection", shape, prove) == (["packed"], None)
    assert len(calls) == 3
    for path in tmp_path.glob("*/projection-*.json"):
        data = json.loads(path.read_bytes())
        data["result"][0] = ["different arithmetic"]
        path.write_text(json.dumps(data))
    assert ProofCache("a", tmp_path).decide("projection", 4, prove) == (["packed"], None)
    assert len(calls) == 4


def test_a_speed_race_requires_a_measured_margin_and_a_bit_proof(tmp_path):
    cache = ProofCache("a", tmp_path)
    def failed(_):
        raise AssertionError("race reached a failed bit check")

    assert cache.decide("tile", 1, lambda: (None, "bits differ"), race=failed) == (None, "bits differ")
    for shape, candidate, decision in ((2, 96, "tie"), (3, 95, "win"), (5, 105, "loss"), (6, 104, "tie")):
        plan, why = cache.decide("tile", shape, lambda: (True, None), race=lambda _: (100, candidate))
        assert (plan is True) == (decision == "win")
        assert (why is None) == (decision == "win")
        receipt = cache.race_report("tile")[-1]
        assert receipt["decision"] == decision and receipt["confirmed"]
        assert receipt["stock_gpu_ns"] == 100 and receipt["candidate_gpu_ns"] == candidate
        assert receipt["measured_s"] >= 0 and not receipt["from_cache"]
        next_load = ProofCache("a", tmp_path)
        assert next_load.decide("tile", shape, lambda: pytest.fail("cached proof repeated"), race=failed) == (plan, why)
        assert next_load.race_report("tile") == [{**receipt, "from_cache": True}]
    assert cache.decide("tile", 4, lambda: (True, None), race=lambda _: (100, 0))[0] is None
    assert ProofCache("a", tmp_path).decide("tile", 4, lambda: (True, None), race=lambda _: (100, 80)) == (True, None)

def test_editing_metal_source_invalidates_a_cached_pass_in_the_same_package_version(tmp_path, monkeypatch):
    from mtplx.kernels import first_use

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "weights.safetensors").touch()
    code = tmp_path / "code"
    code.mkdir()
    kernel = code / "verify.metal"
    kernel.write_text("original")
    digest = first_use._source_digest
    monkeypatch.setattr(first_use, "_source_digest", lambda *roots: digest(code))
    monkeypatch.setattr(first_use.Path, "home", lambda: tmp_path)
    calls = []
    def prove():
        calls.append(1)
        return True, None
    for contents in ("original", "original", "edited"):
        kernel.write_text(contents)
        assert ProofCache.for_pack(pack).decide("verify", [4], prove) == (True, None)
    assert len(calls) == 2


@pytest.mark.parametrize("directory", [True, False])
def test_transient_gpu_failures_retry_on_next_load_only(tmp_path, directory):
    root = tmp_path if directory else None
    calls = []
    def once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("[metal::malloc] resource limit")
        return True, None

    first = ProofCache("gpu", root)
    why = first.decide("projection", [4], once)[1]
    assert "metal::malloc" in why
    assert first.decide("projection", [4], once) == (None, why)
    assert not list(tmp_path.glob("*/projection-*.json"))
    assert ProofCache("gpu", root).decide("projection", [4], once) == (True, None)
    assert len(calls) == 2


def test_only_completed_comparisons_persist_a_decline(tmp_path):
    for shape, reason in enumerate(("a timer refused", Mismatch("stock bits differ"))):
        first = ProofCache("gpu", tmp_path)
        assert first.decide("input", shape, lambda: (None, reason)) == (None, reason)
        second = ProofCache("gpu", tmp_path)
        assert second.decide("input", shape, lambda: (True, None)) == (
            (None, reason) if isinstance(reason, Mismatch) else (True, None)
        )


def test_unreadable_build_metadata_keeps_live_checks_and_reports_the_reason(tmp_path, monkeypatch):
    from mtplx.kernels import first_use

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.safetensors").touch()
    def missing(*_):
        raise OSError("source disappeared")
    monkeypatch.setattr(first_use, "_source_digest", missing)
    proofs = ProofCache.for_pack(tmp_path)
    assert proofs.directory is None and "source disappeared" in proofs.identity_error
    assert proofs.decide("input", 4, lambda: (True, None)) == (True, None)
    assert proofs.decide("input", 4, lambda: pytest.fail("memory memo missed")) == (True, None)


def test_identity_tracks_content_versions_and_gpu_overrides(tmp_path, monkeypatch):
    from mtplx.kernels import first_use

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    blobs = tmp_path / "hub" / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / ("a" * 64)
    blob.write_bytes(b"shard")
    (pack / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr(first_use, "_source_digest", lambda *roots: "build")
    monkeypatch.setattr(first_use, "version", lambda _: "first")
    prior = ProofCache.for_pack(pack).identity
    assert prior["files"][0] == ("model.safetensors", "a" * 64)
    os.utime(blob)
    assert ProofCache.for_pack(pack).identity == prior
    import mlx.core as mx
    with monkeypatch.context() as patch:
        patch.setattr(mx, "__version__", mx.__version__ + ".changed")
        assert ProofCache.for_pack(pack).identity != prior
    monkeypatch.setattr(first_use, "version", lambda _: "second")
    assert ProofCache.for_pack(pack).identity != prior
    monkeypatch.setattr(first_use, "version", lambda _: "first")
    monkeypatch.setenv("MLX_METAL_GPU_ARCH", "test-architecture")
    assert ProofCache.for_pack(pack).identity != prior
    monkeypatch.delenv("MLX_METAL_GPU_ARCH")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "0" if prior["rehearsal"] else "1")
    assert ProofCache.for_pack(pack).identity != prior
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1" if prior["rehearsal"] else "0")
    sidecar = tmp_path / "linked-sidecar"
    sidecar.mkdir()
    (pack / "mtp").symlink_to(sidecar, target_is_directory=True)
    (sidecar / "weights.safetensors").write_bytes(b"new sidecar")
    assert ProofCache.for_pack(pack).identity != prior
    prior = ProofCache.for_pack(pack).identity
    (sidecar / "weights.safetensors").write_bytes(b"replaced sidecar")
    assert ProofCache.for_pack(pack).identity != prior
    prior = ProofCache.for_pack(pack).identity
    (pack / "config.json").write_text('{"updated":true}')
    assert ProofCache.for_pack(pack).identity != prior


def test_expired_owned_identities_are_pruned_while_unknown_files_survive(tmp_path):
    from mtplx.kernels.first_use import _digest

    old = ProofCache("old GPU", tmp_path)
    old.decide("input", [1], lambda: (True, None))
    record, = old.directory.glob("input-*.json")
    data = json.loads(record.read_bytes())
    assert (data["cut"], data["shape"]) == ("input", [1])
    os.utime(old.directory / "identity.json", (0, 0))
    legacy = tmp_path / ("f" * 64 + ".json")
    legacy.write_text("old format")
    os.utime(legacy, (0, 0))
    valid = [tmp_path / (digit * 64 + ".json") for digit in ("a", "b", "c")]
    for index, path in enumerate(valid):
        payload = {"result": [True, None]}
        if index:
            payload["race"] = None
        payload["checksum"] = _digest([path.stem, *payload.values()])
        path.write_text(json.dumps(payload))
        if index < 2:
            os.utime(path, (0, 0))
    ProofCache("current GPU", tmp_path).decide("input", 1, lambda: (True, None))
    assert not old.directory.exists()
    assert [p.exists() for p in valid] == [False, False, True]
    assert legacy.read_text() == "old format"


def test_overlapping_prices_remain_cached_until_the_next_daily_retry(tmp_path, monkeypatch):
    from mtplx.kernels import first_use

    # The GPU arms separate, but including submission makes their ranges touch.
    overlap = first_use.GpuRace((100, 120), (50, 60), 20, (0, 0), (0, 40))
    now = 1000
    monkeypatch.setattr(first_use.time, "time", lambda: now)
    first = ProofCache("device", tmp_path)
    plan, why = first.decide("tile", 1, lambda: (True, None), race=lambda _: overlap)
    receipt, = first.race_report("tile")
    assert plan is None and "inconclusive" in why and not receipt["confirmed"]
    assert receipt["stock_encode_gpu_max_ns"] == 120 and receipt["candidate_encode_gpu_max_ns"] == 100
    assert receipt["retry_after"] == now + 86400
    now += 86399
    warm = ProofCache("device", tmp_path)
    assert warm.decide("tile", 1, lambda: pytest.fail("early retry"), race=lambda _: overlap) == (plan, why)
    assert warm.race_report("tile") == [{**receipt, "from_cache": True}]
    now += 1
    def interrupted(_):
        raise RuntimeError("timestamp unavailable")
    failed = ProofCache("device", tmp_path)
    assert "timestamp unavailable" in failed.decide("tile", 1, lambda: (True, None), race=interrupted)[1]
    assert failed.race_report("tile")[0]["retry_after"] == now + 86400
    warm = ProofCache("device", tmp_path)
    assert "timestamp unavailable" in warm.decide("tile", 1, lambda: pytest.fail("repeated failed refresh"), race=interrupted)[1]
    now += 86400
    retry = ProofCache("device", tmp_path)
    assert retry.decide("tile", 1, lambda: (True, None), race=lambda _: (100, 80)) == (True, None)
    now += 7 * 86400
    settled = ProofCache("device", tmp_path)
    assert settled.decide("tile", 1, lambda: pytest.fail("settled win retried"), race=lambda _: overlap) == (True, None)



def test_a_launch_sensitive_price_uses_each_buffers_encode_and_gpu_time(tmp_path):
    from mtplx.kernels.first_use import GpuRace

    times = GpuRace((100,) * 9, (103,) * 9, 8, (30,) * 9, (4,) * 9)
    cache = ProofCache("gpu", tmp_path)
    assert cache.decide("tiled", [4], lambda: (True, None), race=lambda _: times) == (True, None)
    receipt = cache.race_report("tiled")[0]
    assert (receipt["stock_gpu_ns"], receipt["candidate_gpu_ns"]) == (100, 103)
    assert (receipt["stock_encode_gpu_ns"], receipt["candidate_encode_gpu_ns"]) == (130, 107)
    assert receipt["price_metric"] == "encode_plus_gpu"
