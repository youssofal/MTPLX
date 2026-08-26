from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.attention_context import attention_phase  # noqa: E402
import mtplx.deepseek_v4_dflash2 as dflash2  # noqa: E402
from mtplx.kernels import deepseek_v4_qkv_prologue as qkv_core  # noqa: E402
from mtplx.models import deepseek_v4 as target_model  # noqa: E402
from mtplx.models import deepseek_v4_dspark as dspark_model  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_target_qkv_route_projects_once_and_finalizes_by_runtime_phase() -> None:
    events = []
    q_rank = mx.zeros((1, 2, 1024), dtype=mx.bfloat16)
    latent = mx.zeros((1, 2, 512), dtype=mx.bfloat16)
    q_final = mx.zeros((1, 2, 64, 512), dtype=mx.bfloat16)
    records = mx.zeros((1, 2, 432), dtype=mx.uint8)
    cos = mx.zeros((2, 32), dtype=mx.float32)
    sin = mx.zeros((2, 32), dtype=mx.float32)

    def project_learned(hidden):
        events.append(("project", hidden))
        return q_rank, latent

    def finalizer(label):
        def run(*args):
            events.append((label, args))
            return q_final, records

        return run

    attention = SimpleNamespace(
        n_heads=64,
        head_dim=512,
        _mia_qkv_plan=SimpleNamespace(
            project_learned=project_learned,
            prefill_records=finalizer("prefill"),
            target_records=finalizer("target"),
        ),
        wq_b=lambda values: mx.zeros(
            (*values.shape[:-1], 64 * 512), dtype=mx.bfloat16
        ),
        _mia_token_rope_tables=lambda start, rows: (
            mx.arange(start, start + rows, dtype=mx.int32),
            cos,
            sin,
        ),
    )
    hidden = mx.zeros((1, 2, 4096), dtype=mx.bfloat16)
    cache = SimpleNamespace(offset=11)

    with attention_phase("prefill"):
        actual = target_model.DeepseekV4Attention._mia_cached_qkv_records(
            attention, hidden, cache
        )

    assert actual[0] is q_final
    assert tuple(actual[0].shape) == (1, 2, 64, 512)
    assert actual[1] is q_rank
    assert actual[2] is records
    assert [event[0] for event in events] == ["project", "prefill"]
    finalized = events[1][1]
    assert tuple(finalized[0].shape) == (1, 2, 64, 512)
    assert finalized[1] is latent
    assert finalized[2] is cos
    assert finalized[3] is sin


def test_finalized_target_records_preserve_visibility_and_retention_bookkeeping() -> None:
    events = []
    visible = object()

    class Window:
        def paged_records(self, start, stop):
            events.append(("paged", start, stop))
            return visible

        def slice(self, *_args):
            raise AssertionError("exact cache update gathered the fixed window")

        def drop_before(self, start):
            events.append(("drop", start))

    cache = SimpleNamespace(
        offset=10,
        window_size=4,
        rollback_capacity=2,
        window=Window(),
        window_start=0,
        _write_window_records=lambda records, *, absolute_start: events.append(
            ("write", records, absolute_start)
        ),
    )
    records = mx.zeros((1, 2, 432), dtype=mx.uint8)

    actual = target_model.DeepseekV4NVFP4Cache._update_fixed_window_records(
        cache, records
    )

    assert actual == (visible, 7)
    assert events == [
        ("write", records, 10),
        ("paged", 7, 12),
        ("drop", 6),
    ]
    assert cache.window_start == 6


def test_fixed_target_cache_poison_legacy_latent_rope_route(monkeypatch) -> None:
    monkeypatch.setattr(
        target_model,
        "install_stock432_record_packer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed record-only cache installed the old packer")
        ),
    )
    cache = target_model.DeepseekV4NVFP4Cache(
        window_size=128,
        compress_ratio=0,
        head_dim=512,
        max_batch_tokens=8,
    )

    assert cache._pack_window_records is None
    assert cache._update_window_impl.__name__ == "_fixed_window_requires_records"
    assert cache._write_window_records.keywords["owner"] is cache.window
    with pytest.raises(RuntimeError, match="finalized stock432 records only"):
        cache.update_window(object(), object())


def test_fixed_target_verify_trim_recovers_full_rollback_reserve() -> None:
    cache = target_model.DeepseekV4NVFP4Cache(
        window_size=128,
        compress_ratio=0,
        head_dim=512,
        rollback_capacity=64,
        max_batch_tokens=8_224,
    )
    prompt = mx.zeros((1, 300, 432), dtype=mx.uint8)
    prompt[0, :, 0] = mx.arange(300, dtype=mx.uint8)
    cache.update_window_records(prompt)
    cache.advance(300)

    verify = mx.zeros((1, 6, 432), dtype=mx.uint8)
    verify[0, :, 0] = mx.arange(300, 306, dtype=mx.uint8)
    cache.update_window_records(verify)
    cache.advance(6)
    assert (cache.window.start, cache.window.end) == (114, 306)

    assert cache._trim_installed(5) == 5
    assert (cache.offset, cache.window.start, cache.window.end) == (301, 109, 301)
    assert cache.window_start == 109

    assert cache.trim(64) == 64
    assert (cache.offset, cache.window.start, cache.window.end) == (237, 45, 237)
    assert cache.window_start == 45
    assert cache.window._paged_records.length == 192
    np.testing.assert_array_equal(
        np.array(cache.window.slice(45, 237)[0, :, 0]),
        np.arange(45, 237, dtype=np.uint8),
    )


def test_dspark_exact_context_uses_kv_only_plan_and_k5_records_are_temporary() -> None:
    events = []
    latent = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    records = mx.zeros((1, 3, 432), dtype=mx.uint8)
    cos = mx.zeros((3, 32), dtype=mx.float32)
    sin = mx.zeros((3, 32), dtype=mx.float32)
    attention = SimpleNamespace(
        _mia_token_rope_tables=lambda start, rows: (
            mx.arange(start, start + rows, dtype=mx.int32),
            cos,
            sin,
        ),
        _mia_qkv_plan=SimpleNamespace(
            project_kv=lambda hidden: events.append(("kv_only", hidden)) or latent,
            context_records=lambda *args: events.append(("records", args)) or records,
        ),
    )
    hidden = mx.zeros((1, 3, 4096), dtype=mx.bfloat16)

    actual = dspark_model.DeepseekV4DSparkAttention._mia_context_records(
        attention, hidden, 17
    )

    assert actual is records
    assert events == [
        ("kv_only", hidden),
        ("records", (latent, cos, sin)),
    ]
    source = inspect.getsource(dspark_model.DeepseekV4DSparkAttention._run_k5)
    assert ".proposal_records(" in source
    assert "_pack_draft_records" not in source
    assert "cache.ring.records" in source
    assert "_commit" not in source


def test_exact_dflash_context_uses_final_records_not_legacy_latent_rope() -> None:
    source = inspect.getsource(dflash2.DeepseekV4DSparkBackend._append_context)

    assert source.count("project_context_records(") == 2
    assert "_install_prefill_records(" in source
    assert "_commit_records(" in source
    assert "project_kv(" not in source
    assert "_install_prefill_tail(" not in source
    assert "_commit_main(" not in source


def test_installer_binds_46_unique_weight_owned_qkv_plans(monkeypatch) -> None:
    class Owner:
        split = 1024

        def project_fused(self, values):
            return values

    monkeypatch.setattr(target_model, "MiaStackedMXFP8Projection", Owner)
    raw = qkv_core.MiaQKVPrologue(
        learned_norm=lambda *_args, **_kwargs: None,
        kv_norm=lambda *_args, **_kwargs: None,
        target_records=lambda *_args, **_kwargs: None,
        prefill_records=lambda *_args, **_kwargs: None,
        proposal_records=lambda *_args, **_kwargs: None,
        context_records=lambda *_args, **_kwargs: None,
        q_rank=1024,
        heads=64,
        head_dim=512,
        rope_dim=64,
        proposal_rows=5,
        context_rows=128,
        prefill_tile_rows=1024,
    )
    monkeypatch.setattr(qkv_core, "install_mia_qkv_prologue", lambda **_kwargs: raw)

    class Attention:
        def __init__(self):
            self._mia_input_projection = Owner()
            self.q_norm = SimpleNamespace(
                weight=mx.ones((1024,), dtype=mx.bfloat16)
            )
            self.kv_norm = SimpleNamespace(
                weight=mx.ones((512,), dtype=mx.bfloat16)
            )
            self.q_lora_rank = 1024
            self.n_heads = 64
            self.head_dim = 512
            self.rope_head_dim = 64
            self.eps = 1.0e-6
            self._mia_qkv_plan = None

        def install_mia_qkv_prologue(self, plan):
            self._mia_qkv_plan = plan

    target = tuple(SimpleNamespace(attn=Attention()) for _ in range(43))
    draft = tuple(SimpleNamespace(attn=Attention()) for _ in range(3))
    model = SimpleNamespace(
        layers=target,
        dspark=SimpleNamespace(stages=draft),
    )

    receipt = target_model.install_mia_qkv_prologue_routes(model)

    assert receipt["target_attention"] == 43
    assert receipt["draft_attention"] == 3
    assert receipt["unique_plan_count"] == 46
    assert receipt["prefill_cutoff"] == 1024
    assert receipt["proposal_rows"] == 5
    assert receipt["context_rows"] == 128
    assert len(set(receipt["q_weight_ids"])) == 46
    assert len(set(receipt["kv_weight_ids"])) == 46
    assert len(set(receipt["projection_owner_ids"])) == 46
