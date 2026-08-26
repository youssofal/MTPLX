from __future__ import annotations

import hashlib
import inspect
import json
import os
import struct
from types import SimpleNamespace

import pytest

import mlx.core as mx

import mtplx.deepseek_v4_exl3 as exl3
import mtplx.deepseek_v4_mia_engine as mia_engine
from mtplx.models import deepseek_v4 as target_module
from mtplx.deepseek_v4_mia_engine import MiaDeepseekV4EnginePlan


def test_target_arena_settles_fixed_pages_journals_and_live_frontiers(
    monkeypatch,
) -> None:
    layers = tuple(
        SimpleNamespace(
            attn=SimpleNamespace(
                window_size=8,
                compress_ratio=ratio,
                head_dim=512,
            )
        )
        for ratio in (0, 4, 128)
    )
    arena = mia_engine.MiaTargetCacheArena(
        layers,
        capacity_tokens=128,
        max_batch_tokens=16,
    )
    ratio4 = arena._caches[1]
    ratio128 = arena._caches[2]
    ratio4.comp.cur_kv = mx.zeros((1, 3, 1024), dtype=mx.float32)
    ratio4.comp.cur_score = mx.zeros((1, 3, 1024), dtype=mx.float32)
    ratio4.index_comp.prev_kv = mx.zeros((1, 4, 256), dtype=mx.float32)
    ratio4.index_comp.prev_score = mx.zeros((1, 4, 256), dtype=mx.float32)
    ratio128.comp.cur_kv = mx.zeros((1, 127, 512), dtype=mx.float32)
    ratio128.comp.cur_score = mx.zeros((1, 127, 512), dtype=mx.float32)
    logits = mx.zeros((1, 1, 64), dtype=mx.float32)
    tap = mx.zeros((1, 1, 2), dtype=mx.float32)
    calls = []
    scheduled = []
    monkeypatch.setattr(
        arena,
        "_eval_prefill_roots",
        lambda *arrays: calls.append(arrays),
    )
    monkeypatch.setattr(
        arena,
        "_schedule_verify_roots",
        lambda *arrays: scheduled.append(arrays),
        raising=False,
    )

    arena.settle_prefill_chunk(logits, tap)
    pending_frontiers = []
    for lane in arena._prefill_frontier_lanes:
        lane._pending_m6 = SimpleNamespace(
            combined_kv=mx.zeros(
                (1, 6, lane.state_width),
                dtype=mx.float32,
            ),
            combined_score=mx.zeros(
                (1, 6, lane.state_width),
                dtype=mx.float32,
            ),
        )
        pending_frontiers.extend(
            (
                lane._pending_m6.combined_kv,
                lane._pending_m6.combined_score,
            )
        )
    arena.schedule_verify_chunk(logits)

    assert len(calls) == 1
    assert len(scheduled) == 1
    roots = calls[0]
    assert roots[:2] == (logits, tap)
    for cache in arena._caches:
        assert any(root is cache.window._pages for root in roots)
    for cache in (ratio4, ratio128):
        assert any(root is cache.compressed.pages for root in roots)
        for journal in cache.comp.journal_buffers:
            assert any(root is journal for root in roots)
    assert any(root is ratio4.index_compressed.pages for root in roots)
    for journal in ratio4.index_comp.journal_buffers:
        assert any(root is journal for root in roots)
    for frontier in (
        ratio4.comp.cur_kv,
        ratio4.comp.cur_score,
        ratio4.index_comp.prev_kv,
        ratio4.index_comp.prev_score,
        ratio128.comp.cur_kv,
        ratio128.comp.cur_score,
    ):
        assert any(root is frontier for root in roots)
    assert scheduled[0][0] is logits
    assert all(any(root is expected for root in scheduled[0]) for expected in roots[2:])
    assert all(
        any(root is expected for root in scheduled[0])
        for expected in pending_frontiers
    )


def test_m6_cache_schedule_reuses_exact_ratio_slices_across_43_layers(
    monkeypatch,
) -> None:
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        eval_calls = []
        original_eval = mx.eval

        def capture_eval(*arrays):
            eval_calls.append(arrays)
            return original_eval(*arrays)

        monkeypatch.setattr(mx, "eval", capture_eval)
        ratios = (0, 0) + (4, 128) * 20 + (4,)
        layers = tuple(
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_size=128,
                    compress_ratio=ratio,
                    head_dim=512,
                )
            )
            for ratio in ratios
        )
        arena = mia_engine.MiaTargetCacheArena(
            layers,
            capacity_tokens=512,
            max_batch_tokens=6,
        )
        caches = arena.acquire(layers)
        schedule = arena._m6_schedule
        ratio4_tables = schedule._ratio4
        ratio128_tables = schedule._ratio128

        for tables in (ratio4_tables, ratio128_tables):
            evaluated = (
                tables.ape_slots,
                tables.journal_slots,
                tables.compressed_blocks,
                tables.compressed_offsets,
                tables.causal_lengths,
                tables.batch_slots,
            )
            assert any(
                len(call) == len(evaluated)
                and all(
                    observed is expected
                    for observed, expected in zip(call, evaluated, strict=True)
                )
                for call in eval_calls
            )
            assert all(value.dtype == mx.int32 for value in evaluated)
        assert all(cache._mia_m6_schedule is schedule for cache in caches)
        assert all(
            cache.comp._mia_m6_tables is ratio4_tables
            and cache.index_comp._mia_m6_tables is ratio4_tables
            for ratio, cache in zip(ratios, caches, strict=True)
            if ratio == 4
        )
        assert all(
            cache.comp._mia_m6_tables is ratio128_tables
            for ratio, cache in zip(ratios, caches, strict=True)
            if ratio == 128
        )

        caches[0].offset = 127
        cycle = arena.begin_verify()

        assert cycle is arena.current_m6_cycle
        assert cycle.start_offset == 127
        assert cycle.stop_offset == 133
        assert cycle.ratio4.ape_slots.tolist() == [3, 0, 1, 2, 3, 0]
        assert cycle.ratio4.journal_slots.tolist() == [55, 56, 57, 58, 59, 60]
        assert cycle.ratio4.first_window == 31
        assert cycle.ratio4.emitted_rows == 2
        assert cycle.ratio4.compressed_blocks.tolist() == [0, 0]
        assert cycle.ratio4.compressed_offsets.tolist() == [31, 32]
        assert cycle.ratio4.causal_lengths.tolist() == [32, 32, 32, 32, 33, 33]
        assert cycle.ratio128.ape_slots.tolist() == [127, 0, 1, 2, 3, 4]
        assert cycle.ratio128.journal_slots.tolist() == [127, 128, 129, 130, 131, 132]
        assert cycle.ratio128.first_window == 0
        assert cycle.ratio128.emitted_rows == 1
        assert cycle.ratio128.compressed_blocks.tolist() == [0]
        assert cycle.ratio128.compressed_offsets.tolist() == [0]
        assert cycle.ratio128.causal_lengths.tolist() == [1, 1, 1, 1, 1, 1]
        assert all(
            lane is None if ratio == 0 else lane is cycle.ratio4
            for ratio, lane in zip(ratios, cycle.by_layer, strict=True)
            if ratio != 128
        )
        assert all(
            lane is cycle.ratio128
            for ratio, lane in zip(ratios, cycle.by_layer, strict=True)
            if ratio == 128
        )

        caches[0].offset = 70
        ratio4_wrap = arena.begin_verify()
        assert ratio4_wrap.ratio4.journal_slots.tolist() == [70, 71, 0, 1, 2, 3]

        caches[0].offset = 190
        ratio128_wrap = arena.begin_verify()
        assert ratio128_wrap.ratio128.journal_slots.tolist() == [
            190,
            191,
            0,
            1,
            2,
            3,
        ]

        caches[0].offset = 255
        page_crossing = arena.begin_verify()
        assert page_crossing.ratio4.first_window == 63
        assert page_crossing.ratio4.emitted_rows == 2
        assert page_crossing.ratio4.compressed_blocks.tolist() == [0, 1]
        assert page_crossing.ratio4.compressed_offsets.tolist() == [63, 0]

        caches[0].offset = 506
        maximum = arena.begin_verify()
        assert maximum.start_offset == 506
        assert maximum.stop_offset == 512
        assert maximum.ratio4.causal_lengths.tolist() == [
            126,
            127,
            127,
            127,
            127,
            128,
        ]
        assert maximum.ratio128.causal_lengths.tolist() == [3, 3, 3, 3, 3, 4]
        for lane in (maximum.ratio4, maximum.ratio128):
            assert len(lane.ape_slots) == 6
            assert len(lane.journal_slots) == 6
            assert len(lane.causal_lengths) == 6
            assert len(lane.batch_slots) == 6

        arena.release(caches)

        assert arena.current_m6_cycle is None
    finally:
        mx.set_default_device(previous_device)


def test_mia_target_ratio_order_rejects_same_count_reordering() -> None:
    reordered = list(mia_engine._MIA_TARGET_COMPRESS_RATIOS)
    reordered[2], reordered[3] = reordered[3], reordered[2]

    with pytest.raises(ValueError, match="compress-ratio order changed"):
        mia_engine._require_mia_target_ratio_order(tuple(reordered))


def _safetensors_bytes(header: dict, payload: bytes) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded = encoded + b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(padded)) + padded + payload


class _FakeArray:
    def __init__(self, shape=(1,)):
        self.shape = tuple(shape)

    def __getitem__(self, _key):
        return self

    def astype(self, _dtype):
        return self


class _FakeTargetArena:
    def __init__(self, events):
        self.events = events
        self.cache = [object()]
        self.current_m6_cycle = None

    def acquire(self, _layers):
        self.events.append("target.acquire")
        return self.cache

    def release(self, caches):
        assert caches is self.cache
        self.events.append("target.release")

    def settle_prefill_chunk(self, *_outputs):
        self.events.append("target.settle")

    def begin_verify(self):
        self.current_m6_cycle = SimpleNamespace(start_offset=127)
        self.events.append("target.begin_verify")
        return self.current_m6_cycle

    def commit_verify(self, accepted_rows):
        self.events.append(("target.commit_verify", accepted_rows))
        self.current_m6_cycle = None
        return accepted_rows


def _plan(events):
    return MiaDeepseekV4EnginePlan(
        context_capacity_tokens=384_000,
        target_physical_capacity_tokens=384_005,
        max_batch_tokens=8_224,
        max_sequences=1,
        page_geometry=(),
        workspace_geometry=(),
        indexer_workspace=None,
        indexer_rope_table=None,
        mla_workspace=None,
        target_cache_arena=_FakeTargetArena(events),
        prewarm_signatures=(),
        installed_routes=(),
        target_artifact="target",
        draft_artifact="draft",
        artifact_small_file_sha256=(),
        identity="test-plan",
    )


def _patch_fake_mx(monkeypatch):
    monkeypatch.setattr(mx, "zeros", lambda shape, **_kwargs: _FakeArray(shape))
    monkeypatch.setattr(mx, "arange", lambda size, **_kwargs: _FakeArray((size,)))
    monkeypatch.setattr(mx, "array", lambda value, **_kwargs: _FakeArray((len(value),)))
    monkeypatch.setattr(mx, "argmax", lambda *_args, **_kwargs: _FakeArray((1,)))
    monkeypatch.setattr(mx, "concatenate", lambda *_args, **_kwargs: _FakeArray((1, 6)))
    monkeypatch.setattr(mx, "eval", lambda *_args, **_kwargs: None)


def test_engine_plan_converts_absolute_target_length_to_m6_acceptance() -> None:
    events = []
    plan = _plan(events)
    cache = plan.target_cache_arena.cache

    cycle = plan.begin_target_verify(cache)
    acceptance = plan.commit_target_verify(cache, target_len=131)

    assert cycle.start_offset == 127
    assert acceptance == 4
    assert events == [
        "target.begin_verify",
        ("target.commit_verify", 4),
    ]


def test_engine_plan_seals_quad_qmv_owners_and_prewarm_signatures() -> None:
    build_source = inspect.getsource(mia_engine.build_mia_engine_plan)
    identity_source = inspect.getsource(mia_engine._mia_engine_identity)
    prewarm_source = inspect.getsource(MiaDeepseekV4EnginePlan.prewarm)

    assert "_mia_exl3_m6_fused" in build_source
    assert "EXL3SwitchGLU.direct_m6_clamp10" in build_source
    assert "_InstalledM6QuadQMVPlan" in build_source
    assert "EXL3_M6_QUAD_DESCRIPTOR_SHA256" in build_source
    assert 'getattr(quad_plan, "dual_fc1_input", None)' in build_source
    assert 'getattr(quad_plan, "dual_fc1_inner", None)' in build_source
    assert 'getattr(quad_plan, "activation_down", None)' in build_source
    assert 'getattr(quad_plan, "down_inner", None)' in build_source
    assert 'getattr(quad_plan, "direct_final_tail", None)' in build_source
    assert "MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256" in build_source
    assert "MIA_EXL3_M6_QUAD_DESCRIPTOR_SHA256" in identity_source
    assert "EXL3_M6_STAGE_VECTOR_BYTES" in build_source
    assert "EXL3_M6_STAGE_VECTORS_PER_K_TILE" in build_source
    assert 'getattr(quad_plan, "stage_vector_bytes", None)' in build_source
    assert 'getattr(quad_plan, "stage_vectors_per_k_tile", None)' in build_source
    assert "_mia_exl3_trellis_fused" in build_source
    assert "EXL3SwitchGLU.fused" in build_source
    assert "_mia_fullgraph_propose_k5" in build_source
    assert '!= "_mia_propose_k5"' not in build_source
    assert "_run_dspark_k5_nvfp4_mla_graph" in build_source
    assert "_gate_up_impl" in build_source
    assert (
        "target_exl3_prefill_trellis_bm8_bm64_verify_m6_direct_5stage_"
        "staged_fc1_clamp10_bf16tail_u4_stage16b_96x8"
        in build_source
    )
    assert (
        "target-exl3-prefill-trellis-bm8-bm64-verify-m6-direct-5stage-"
        "staged-fc1-clamp10-bf16tail-u4-stage16b-96x8"
        in identity_source
    )
    assert 'MiaPrewarmSignature("target_prefill_m6_bm8", 6, "prefill")' in (
        build_source
    )
    assert (
        'MiaPrewarmSignature("target_prefill_m128_bm64", MIA_WINDOW, "prefill")'
        in build_source
    )
    assert (
        '"target_verify_m6_direct_5stage_staged_fc1_clamp10"'
        in build_source
    )
    assert 'with attention_phase("prefill"):' in prewarm_source
    assert "first_layer.ffn(" in prewarm_source
    assert '"prefill_m6_rows": prefill_m6_rows' in prewarm_source


def test_shared_indexer_rope_table_is_built_once_and_owned_by_every_ratio4_layer(
    monkeypatch,
):
    inv_freq = object()
    shared_table = object()
    workspace = object()
    build_calls = []
    install_calls = []

    class FakeIndexer:
        def __init__(self):
            self._inv_freq = inv_freq

        def install_mia_paged_topk(self, installed_workspace, rope_table):
            install_calls.append((installed_workspace, rope_table))

    layers = tuple(
        SimpleNamespace(attn=SimpleNamespace(indexer=FakeIndexer()))
        for _ in range(3)
    )
    monkeypatch.setattr(
        mia_engine,
        "precompute_indexer_rope_table",
        lambda frequencies, *, max_positions: (
            build_calls.append((frequencies, max_positions)) or shared_table
        ),
    )

    actual = mia_engine._install_shared_indexer_resources(
        layers,
        (4, 128, 4),
        workspace,
        inv_freq,
        max_positions=384_005,
    )

    assert actual is shared_table
    assert build_calls == [(inv_freq, 384_005)]
    assert install_calls == [
        (workspace, shared_table),
        (workspace, shared_table),
    ]


def test_exact_engine_binds_one_base_and_one_compress_rope_provider() -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        bindings = []

        class FakeAttention:
            def __init__(self, ratio):
                self.compress_ratio = ratio

            def install_mia_rope_provider(self, provider):
                bindings.append((self.compress_ratio, provider))

        model = SimpleNamespace(
            args=SimpleNamespace(
                qk_rope_head_dim=64,
                rope_theta=10_000.0,
                compress_rope_theta=160_000.0,
                original_seq_len=4_096,
                rope_factor=16.0,
                beta_fast=32,
                beta_slow=1,
            ),
            layers=tuple(
                SimpleNamespace(attn=FakeAttention(ratio))
                for ratio in (0, 4, 128, 4, 0)
            ),
            dspark=SimpleNamespace(
                stages=tuple(
                    SimpleNamespace(attn=FakeAttention(0)) for _ in range(3)
                )
            ),
        )

        base, compress = target_module.install_mia_target_rope_providers(
            model,
            physical_max_positions=384_005,
        )

        assert model._mia_base_rope_provider is base
        assert model._mia_compress_rope_provider is compress
        assert base is not compress
        assert base.max_positions == 384_005
        assert compress.max_positions == 384_005
        draft = model._mia_draft_rope_provider
        assert draft.max_positions == 384_005
        assert draft is not base
        assert [
            (ratio, provider is base, provider is compress, provider is draft)
            for ratio, provider in bindings
        ] == [
            (0, True, False, False),
            (4, False, True, False),
            (128, False, True, False),
            (4, False, True, False),
            (0, True, False, False),
            (0, False, False, True),
            (0, False, False, True),
            (0, False, False, True),
        ]
        positions, _cos, _sin = draft.token_tables(384_000, 5)
        assert positions.tolist() == list(range(384_000, 384_005))
        target_positions, _cos, _sin = base.token_tables(384_000, 5)
        assert target_positions.tolist() == list(range(384_000, 384_005))
    finally:
        mx.set_default_device(previous)


def test_exact_target_forward_starts_one_shared_rope_epoch_per_chunk() -> None:
    events = []

    class Provider:
        def begin_forward(self):
            events.append("rope")

    model = target_module.Model.__new__(target_module.Model)
    model._mia_base_rope_provider = Provider()
    model._mia_compress_rope_provider = Provider()
    model._mia_draft_rope_provider = Provider()
    model.model = SimpleNamespace(
        _run_mia_hc_target_tail_taps=lambda inputs, cache: (
            events.append((inputs, cache)) or "result"
        )
    )
    inputs = object()
    cache = object()

    assert model._mia_target_forward(inputs, cache) == "result"
    assert events == ["rope", "rope", "rope", (inputs, cache)]


def test_exact_stacked_projection_installer_binds_all_named_owners(monkeypatch):
    validated = []
    built = []

    class FakeStack:
        @staticmethod
        def validate_pair(first, second):
            validated.append((first, second))

        def __init__(self, first, second):
            built.append((first, second))

    monkeypatch.setattr(target_module, "MiaStackedMXFP8Projection", FakeStack)
    monkeypatch.setattr(target_module, "MiaStackedDenseProjection", FakeStack)

    class FakeCompressor:
        def __init__(self, name):
            self.wkv = f"{name}.wkv"
            self.wgate = f"{name}.wgate"
            self.owner = None

        def install_mia_stacked_projection(self, owner):
            self.owner = owner

    class FakeSharedExpert:
        def __init__(self, name):
            self.gate_proj = f"{name}.gate_proj"
            self.up_proj = f"{name}.up_proj"
            self.owner = None

        def install_mia_stacked_gate_up(self, owner):
            self.owner = owner
            self._gate_up_impl = owner

    class FakeAttention:
        def __init__(self, name, ratio):
            self.compress_ratio = ratio
            self.wq_a = f"{name}.wq_a"
            self.wkv = f"{name}.wkv"
            self.owner = None
            if ratio:
                self.compressor = FakeCompressor(f"{name}.compressor")
            if ratio == 4:
                self.indexer = SimpleNamespace(
                    compressor=FakeCompressor(f"{name}.indexer.compressor")
                )

        def install_mia_stacked_projection(self, owner):
            self.owner = owner

    ratios = (0, 0) + (4,) * 21 + (128,) * 20
    layers = tuple(
        SimpleNamespace(
            attn=FakeAttention(f"target.{index}", ratio),
            ffn=SimpleNamespace(
                shared_experts=FakeSharedExpert(f"target.{index}.shared")
            ),
        )
        for index, ratio in enumerate(ratios)
    )
    stages = tuple(
        SimpleNamespace(
            attn=FakeAttention(f"draft.{index}", 0),
            ffn=SimpleNamespace(
                shared_experts=FakeSharedExpert(f"draft.{index}.shared")
            ),
        )
        for index in range(3)
    )
    model = SimpleNamespace(
        layers=layers,
        dspark=SimpleNamespace(stages=stages),
    )

    receipt = target_module.install_mia_stacked_projections(model)

    assert receipt == {
        "target_attention": 43,
        "draft_attention": 3,
        "shared_expert": 46,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }
    assert len(validated) == len(built) == 154
    assert all(layer.attn.owner is not None for layer in layers)
    assert all(stage.attn.owner is not None for stage in stages)
    assert all(layer.ffn.shared_experts.owner is not None for layer in layers)
    assert all(stage.ffn.shared_experts.owner is not None for stage in stages)
    assert all(
        layer.ffn.shared_experts._gate_up_impl
        is layer.ffn.shared_experts.owner
        for layer in layers
    )
    assert all(
        stage.ffn.shared_experts._gate_up_impl
        is stage.ffn.shared_experts.owner
        for stage in stages
    )
    assert all(
        layer.attn.compressor.owner is not None
        for layer in layers
        if layer.attn.compress_ratio
    )
    assert all(
        layer.attn.indexer.compressor.owner is not None
        for layer in layers
        if layer.attn.compress_ratio == 4
    )


def test_ratio_specialized_mla_route_contract_rejects_generic_callables():
    expected = {
        0: (
            "_mia_cached_forward_uncompressed",
            "_mia_cached_attention_ratio0",
            "_mia_uncached_compressed",
            "_run_installed_window_nvfp4_sparse_mla",
            "_run_installed_window_nvfp4_prefill_mla",
        ),
        4: (
            "_mia_cached_forward_ratio4",
            "_mia_cached_attention_ratio4",
            "_mia_uncached_compressed",
            "_run_installed_indexed_paged_nvfp4_sparse_mla",
            "_run_installed_indexed_paged_nvfp4_prefill_mla",
        ),
        128: (
            "_mia_cached_forward_ratio128",
            "_mia_cached_attention_ratio128",
            "_mia_uncached_compressed",
            "_run_installed_sequential_paged_nvfp4_sparse_mla",
            "_run_installed_sequential_paged_nvfp4_prefill_mla",
        ),
    }

    assert mia_engine._MIA_ATTENTION_ROUTE_CONTRACTS == expected
    installed_names = {
        name
        for route_contract in expected.values()
        for name in route_contract
    }
    assert installed_names.isdisjoint(
        {
            "_mia_cached_attention",
            "_run_nvfp4_sparse_mla",
            "_run_paged_nvfp4_sparse_mla",
            "_run_nvfp4_prefill_mla",
            "_run_paged_nvfp4_prefill_mla",
        }
    )


def test_prewarm_releases_target_lease_when_first_forward_fails(monkeypatch):
    _patch_fake_mx(monkeypatch)
    events = []

    class FailingModel:
        layers = (object(),)

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("target compile failed")

    with pytest.raises(RuntimeError, match="target compile failed"):
        _plan(events).prewarm(FailingModel())

    assert events == ["target.acquire", "target.release"]


def test_prewarm_releases_both_leases_when_proposal_fails(monkeypatch):
    _patch_fake_mx(monkeypatch)
    events = []
    fake = _FakeArray()

    class FakeSelector:
        def _select_rows(self, *_args, **_kwargs):
            return SimpleNamespace(indices=fake, lengths=fake)

    class FakeMHC:
        def post_pre_ffn(self, *_args, **_kwargs):
            return fake, fake, fake, fake

    class FakeDraftOwner:
        def release_mia_cache(self, _cache):
            events.append("draft.release")

    class FailingModel:
        def __init__(self):
            first = SimpleNamespace(
                attn_hc=fake,
                attn_norm=fake,
                ffn_hc=fake,
                ffn_norm=fake,
                ffn=lambda *_args: events.append("ffn.prefill_m6") or fake,
                attn=SimpleNamespace(
                    _output_projection_impl=lambda *_args: (
                        events.append("wo.m16") or fake
                    ),
                    _mia_qkv_plan=SimpleNamespace(
                        prefill_records=lambda *_args: (
                            events.append("qkv.m1024") or (fake, fake)
                        )
                    ),
                ),
            )
            selector_layer = SimpleNamespace(
                attn=SimpleNamespace(indexer=FakeSelector())
            )
            self.layers = (first, first, selector_layer)
            self.model = SimpleNamespace(
                layers=(first,),
                _mia_mhc=FakeMHC(),
            )
            self.dspark = FakeDraftOwner()
            self._mia_base_rope_provider = SimpleNamespace(
                token_tables=lambda *_args: (fake, fake, fake)
            )

        def __call__(self, *_args, **_kwargs):
            return fake, (fake,)

        def mia_dflash_forward(self, *_args, **_kwargs):
            events.append("target.verify_m6_piecewise")
            return fake, (fake,)

        def make_dspark_cache(self):
            events.append("draft.acquire")
            return [SimpleNamespace(ring=SimpleNamespace(records=fake))]

        def prefill_dspark(self, *_args, **_kwargs):
            return None

        def propose_dspark_k5(self, *_args, **_kwargs):
            raise RuntimeError("proposal compile failed")

    with pytest.raises(RuntimeError, match="proposal compile failed"):
        _plan(events).prewarm(FailingModel())

    assert events == [
        "target.acquire",
        "target.settle",
        "ffn.prefill_m6",
        "wo.m16",
        "qkv.m1024",
        "draft.acquire",
        "draft.release",
        "target.release",
    ]

    events.clear()
    successful = FailingModel()
    successful.propose_dspark_k5 = lambda *_args, **_kwargs: SimpleNamespace(
        future_tokens=fake,
        neural_logits=fake,
    )

    receipt = _plan(events).prewarm(successful)

    assert receipt["verify_rows"] == mia_engine.MIA_DSPARK_BLOCK + 1
    assert receipt["prefill_m6_rows"] == 6
    assert events == [
        "target.acquire",
        "target.settle",
        "ffn.prefill_m6",
        "wo.m16",
        "qkv.m1024",
        "draft.acquire",
        "target.begin_verify",
        "target.verify_m6_piecewise",
        "target.settle",
        "draft.release",
        "target.release",
    ]


def test_prewarm_release_failures_do_not_replace_the_primary_error():
    class FailingPlan:
        def release_target_cache(self, _cache):
            raise RuntimeError("target release failed")

    class FailingDraftOwner:
        def release_mia_cache(self, _cache):
            raise RuntimeError("draft release failed")

    primary = RuntimeError("proposal failed")

    mia_engine._release_prewarm_leases(
        FailingPlan(),
        SimpleNamespace(dspark=FailingDraftOwner()),
        [object()],
        [object()],
        primary,
    )

    assert str(primary) == "proposal failed"
    assert primary.__notes__ == [
        "prewarm cache release also failed: RuntimeError: draft release failed",
        "prewarm cache release also failed: RuntimeError: target release failed",
    ]


def test_verified_safetensors_rejects_path_swap_and_loads_only_hashed_fd(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    pinned = b"pinned-shard-bytes"
    swapped = b"swapped-shard-byte"
    assert len(pinned) == len(swapped)
    shard.write_bytes(pinned)
    replacement.write_bytes(swapped)

    observed_payloads = []

    def fake_load(stream, *, format):
        assert format == "safetensors"
        os.replace(replacement, shard)
        observed_payloads.append(stream.read())
        return {"payload": observed_payloads[-1]}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="changed while loading"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=len(pinned),
            expected_sha256=hashlib.sha256(pinned).hexdigest(),
        )

    assert observed_payloads == [pinned]
    assert shard.read_bytes() == swapped


def test_verified_safetensors_rejects_digest_before_loading(monkeypatch, tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"wrong bytes")
    load_calls = 0

    def fake_load(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        return {}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="checksum changed"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=shard.stat().st_size,
            expected_sha256=hashlib.sha256(b"expected bytes").hexdigest(),
        )

    assert load_calls == 0


def test_verified_safetensors_loads_tiny_real_file_from_same_descriptor(tmp_path):
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(str(shard), {"value": mx.array([7], dtype=mx.int32)})
    expected_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()

    loaded = exl3._load_verified_safetensors(
        shard,
        expected_bytes=shard.stat().st_size,
        expected_sha256=expected_sha256,
    )
    mx.eval(loaded["value"])

    assert loaded["value"].item() == 7


def test_verified_safetensors_rejects_in_place_change_during_load(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "model.safetensors"
    pinned = b"pinned-shard-bytes"
    changed = b"changed-shard-byte"
    assert len(pinned) == len(changed)
    shard.write_bytes(pinned)

    def fake_load(stream, *, format):
        assert format == "safetensors"
        shard.write_bytes(changed)
        stream.seek(0)
        return {"payload": stream.read()}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    with pytest.raises(ValueError, match="changed while loading"):
        exl3._load_verified_safetensors(
            shard,
            expected_bytes=len(pinned),
            expected_sha256=hashlib.sha256(pinned).hexdigest(),
        )


def test_verified_safetensors_enforces_canonical_pin_in_existing_single_pass(
    monkeypatch,
    tmp_path,
):
    original = tmp_path / "original.safetensors"
    reordered = tmp_path / "reordered.safetensors"
    changed_header = tmp_path / "changed-header.safetensors"
    changed_payload = tmp_path / "changed-payload.safetensors"
    first_header = {
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    }
    reordered_header = {
        "b": {"data_offsets": [1, 2], "shape": [1], "dtype": "U8"},
        "a": {"shape": [1], "dtype": "U8", "data_offsets": [0, 1]},
    }
    original.write_bytes(_safetensors_bytes(first_header, b"AB"))
    reordered.write_bytes(_safetensors_bytes(reordered_header, b"AB"))
    changed_header.write_bytes(
        _safetensors_bytes(
            {
                **reordered_header,
                "b": {
                    "data_offsets": [0, 1],
                    "shape": [1],
                    "dtype": "U8",
                },
            },
            b"AB",
        )
    )
    changed_payload.write_bytes(_safetensors_bytes(reordered_header, b"AC"))
    canonical_sha256 = (
        "f291c11aa84a6ca9259cb713843859743223b11d2a4e74c0b8cf97074778c520"
    )
    loaded_payloads = []

    def fake_load(stream, *, format):
        assert format == "safetensors"
        loaded_payloads.append(stream.read())
        return {"loaded": True}

    monkeypatch.setattr(exl3.mx, "load", fake_load)

    for path in (original, reordered):
        payload = path.read_bytes()
        assert exl3._load_verified_safetensors(
            path,
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_canonical_sha256=canonical_sha256,
        ) == {"loaded": True}
    for path in (changed_header, changed_payload):
        payload = path.read_bytes()
        with pytest.raises(ValueError, match="canonical checksum changed"):
            exl3._load_verified_safetensors(
                path,
                expected_bytes=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_canonical_sha256=canonical_sha256,
            )

    assert loaded_payloads == [original.read_bytes(), reordered.read_bytes()]


def test_artifact_seals_normalize_only_materialization_paths_and_raw_hashes():
    shard_seals = {
        "target.safetensors": "a" * 64,
    }

    def target_manifest(source: str, raw_sha256: str) -> dict:
        return {
            "files": [
                {
                    "sha256": raw_sha256,
                    "bytes": 17,
                    "name": "target.safetensors",
                }
            ],
            "tensor_count": 3,
            "source_tp": 4,
            "format": "rank-sliced-exl3-tp1-v1",
            "tensor_bytes": 9,
            "target_tp": 1,
            "source": source,
        }

    local = target_manifest(
        "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI",
        "1" * 64,
    )
    packaged = target_manifest(
        "/hf-cache/hub/models--0xSero--deepseek-v4-flash-0731-spark/"
        "snapshots/22f28d32b9b29b4352eaa380ff8c2c170b2847ab",
        "2" * 64,
    )

    assert mia_engine._target_artifact_seal_sha256(
        local, shard_seals
    ) == mia_engine._target_artifact_seal_sha256(packaged, shard_seals)
    relocated = target_manifest("/arbitrary/relocated/source", "3" * 64)
    assert mia_engine._target_artifact_seal_sha256(
        local, shard_seals
    ) == mia_engine._target_artifact_seal_sha256(relocated, shard_seals)
    changed_target = target_manifest(local["source"], "1" * 64)
    changed_target["files"][0]["bytes"] = 18
    assert mia_engine._target_artifact_seal_sha256(
        local, shard_seals
    ) != mia_engine._target_artifact_seal_sha256(changed_target, shard_seals)
    with pytest.raises(ValueError, match="source contract"):
        mia_engine._target_artifact_seal_sha256(
            target_manifest("", "1" * 64),
            shard_seals,
        )

    plan = {
        "current_to_new_expert_id": {"6": 0, "7": 1},
        "draft_experts": 2,
        "selected_current_expert_ids": [6, 7],
        "selected_original_expert_ids": [164, 27],
        "selection_policy": "pinned policy",
        "sha256": {"draft.safetensors": "1" * 64},
        "source": "/first/source",
        "source_experts": 216,
        "source_plan": "/first/source/REAP_PLAN.json",
        "structured_categories": ["agentic", "tools"],
        "structured_per_category": 1,
        "tensor_count": 3,
        "total_size": 9,
    }
    relocated_plan = dict(reversed(plan.items()))
    relocated_plan["source"] = "/second/source"
    relocated_plan["source_plan"] = "/second/source/REAP_PLAN.json"
    relocated_plan["sha256"] = {"draft.safetensors": "2" * 64}
    assert mia_engine._draft_plan_seal_sha256(plan) == (
        mia_engine._draft_plan_seal_sha256(relocated_plan)
    )
    changed_plan = dict(plan)
    changed_plan["selected_original_expert_ids"] = [164, 28]
    assert mia_engine._draft_plan_seal_sha256(plan) != (
        mia_engine._draft_plan_seal_sha256(changed_plan)
    )


def test_artifact_metadata_pins_semantic_manifests_and_defers_shard_digest(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    target_names = [f"carried-{index:03d}.safetensors" for index in range(1, 6)]
    target_names.extend(
        f"exl3-layer-{layer:03d}-tp1-rank0.safetensors"
        for layer in range(43)
    )
    for name in target_names:
        (target / name).write_bytes(b"t")
    (draft / "dspark-draft.safetensors").write_bytes(b"d")

    target_weight_map = {
        f"tensor.{index}": target_names[index % len(target_names)]
        for index in range(117_005)
    }
    target_documents = {
        "config.json": {},
        "tokenizer.json": {"version": "1.0", "model": {}},
        "tokenizer_config.json": {"tokenizer_class": "PreTrainedTokenizerFast"},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 106_084_465_528},
            "weight_map": target_weight_map,
        },
        "rank-sliced-tp1-manifest.json": {
            "format": "rank-sliced-exl3-tp1-v1",
            "source": "/relocatable/source",
            "source_tp": 4,
            "target_tp": 1,
            "tensor_count": 117_005,
            "tensor_bytes": 106_084_465_528,
            "files": [
                {
                    "name": name,
                    "bytes": 1,
                    "sha256": hashlib.sha256(b"not-the-shard").hexdigest(),
                }
                for name in target_names
            ],
        },
        "EXL3_MANIFEST.json": {},
    }
    draft_weight_map = {
        f"tensor.{index}": "dspark-draft.safetensors"
        for index in range(1_249)
    }
    draft_documents = {
        "config.json": {},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 1},
            "weight_map": draft_weight_map,
        },
        "DSPARK_DRAFT_PLAN.json": {
            "draft_experts": 64,
            "source_experts": 216,
            "tensor_count": 1_249,
            "total_size": 1,
            "sha256": {
                "dspark-draft.safetensors": hashlib.sha256(
                    b"not-the-draft"
                ).hexdigest()
            },
        },
    }

    def write_documents(root, documents):
        for name, document in documents.items():
            (root / name).write_text(json.dumps(document), encoding="utf-8")
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in documents
        }

    target_pins = write_documents(target, target_documents)
    target_pins.pop("rank-sliced-tp1-manifest.json")
    target_canonical = {
        name: hashlib.sha256(f"canonical:{name}".encode()).hexdigest()
        for name in target_names
    }
    monkeypatch.setattr(mia_engine, "_TARGET_SMALL_FILE_PINS", target_pins)
    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_ARTIFACT_SEAL",
        mia_engine._target_artifact_seal_sha256(
            target_documents["rank-sliced-tp1-manifest.json"], target_canonical
        ),
    )
    draft_pins = write_documents(draft, draft_documents)
    draft_pins.pop("DSPARK_DRAFT_PLAN.json")
    monkeypatch.setattr(
        mia_engine,
        "_DRAFT_CANONICAL_PLAN_SEAL",
        mia_engine._draft_plan_seal_sha256(
            draft_documents["DSPARK_DRAFT_PLAN.json"]
        ),
    )
    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_SHARD_PINS",
        target_canonical,
        raising=False,
    )
    monkeypatch.setattr(mia_engine, "_DRAFT_SMALL_FILE_PINS", draft_pins)
    monkeypatch.setattr(
        mia_engine,
        "_DRAFT_CANONICAL_SHARD_PIN",
        hashlib.sha256(b"canonical:draft").hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(mia_engine, "MIA_DRAFT_SHARD_BYTES", 1)

    validation = mia_engine.validate_pinned_mia_artifacts(target, draft)

    assert len(validation.target_shards) == 48
    assert len(validation.target_weight_map) == 117_005
    assert len(validation.draft_weight_map) == 1_249
    target_small_files = dict(validation.target_small_file_sha256)
    assert target_small_files["tokenizer.json"] == hashlib.sha256(
        (target / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert target_small_files["tokenizer_config.json"] == hashlib.sha256(
        (target / "tokenizer_config.json").read_bytes()
    ).hexdigest()
    assert target_small_files["rank-sliced-tp1-manifest.json"] == hashlib.sha256(
        (target / "rank-sliced-tp1-manifest.json").read_bytes()
    ).hexdigest()
    assert dict(validation.draft_small_file_sha256)[
        "DSPARK_DRAFT_PLAN.json"
    ] == hashlib.sha256((draft / "DSPARK_DRAFT_PLAN.json").read_bytes()).hexdigest()
    assert validation.target_shards[0].sha256 == hashlib.sha256(
        b"not-the-shard"
    ).hexdigest()
    assert validation.target_shards[0].canonical_sha256 == (
        target_canonical[validation.target_shards[0].name]
    )
    assert validation.draft_shards[0].canonical_sha256 == (
        hashlib.sha256(b"canonical:draft").hexdigest()
    )

    replacement = tmp_path / "replacement-tokenizer.json"
    replacement.write_bytes((target / "tokenizer.json").read_bytes())
    os.replace(replacement, target / "tokenizer.json")
    with pytest.raises(ValueError, match="tokenizer file identity changed"):
        mia_engine.revalidate_pinned_mia_tokenizer_files(validation)

    (target / "tokenizer.json").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Mia target file changed: tokenizer.json"):
        mia_engine.validate_pinned_mia_artifacts(target, draft)


def test_engine_identity_includes_every_pinned_small_file(monkeypatch):
    original = mia_engine._mia_engine_identity(384_000, 8_224)
    changed = dict(mia_engine._TARGET_SMALL_FILE_PINS)
    changed["tokenizer.json"] = "0" * 64
    monkeypatch.setattr(mia_engine, "_TARGET_SMALL_FILE_PINS", changed)

    assert mia_engine._mia_engine_identity(384_000, 8_224) != original


def test_engine_identity_uses_the_installed_dflash_source_pin():
    from mtplx.dflash_identity import PINNED_DFLASH_COMMIT

    assert mia_engine.MIA_DFLASH_COMMIT == PINNED_DFLASH_COMMIT


def test_engine_identity_includes_canonical_shard_seals(monkeypatch):
    target_artifact_seal = mia_engine._TARGET_CANONICAL_ARTIFACT_SEAL
    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_SHARD_PINS",
        {"target.safetensors": "a" * 64},
    )
    monkeypatch.setattr(mia_engine, "_DRAFT_CANONICAL_SHARD_PIN", "b" * 64)
    original = mia_engine._mia_engine_identity(384_000, 8_224)

    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_SHARD_PINS",
        {"target.safetensors": "c" * 64},
    )
    assert mia_engine._mia_engine_identity(384_000, 8_224) != original

    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_SHARD_PINS",
        {"target.safetensors": "a" * 64},
    )
    monkeypatch.setattr(mia_engine, "_DRAFT_CANONICAL_SHARD_PIN", "d" * 64)
    assert mia_engine._mia_engine_identity(384_000, 8_224) != original

    monkeypatch.setattr(mia_engine, "_DRAFT_CANONICAL_SHARD_PIN", "b" * 64)
    monkeypatch.setattr(mia_engine, "_TARGET_CANONICAL_ARTIFACT_SEAL", "e" * 64)
    assert mia_engine._mia_engine_identity(384_000, 8_224) != original

    monkeypatch.setattr(
        mia_engine,
        "_TARGET_CANONICAL_ARTIFACT_SEAL",
        target_artifact_seal,
    )
    monkeypatch.setattr(mia_engine, "_DRAFT_CANONICAL_PLAN_SEAL", "f" * 64)
    assert mia_engine._mia_engine_identity(384_000, 8_224) != original
