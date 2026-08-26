from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.benchmarks.dflash2_runtime import (  # noqa: E402
    MTPLXDFlash2Bundle,
    build_deepseek_v4_dflash2_runtime_context,
    load_mtplx_deepseek_v4_dflash2_bundle,
)
from mtplx.dflash_identity import PINNED_DFLASH_IDENTITY  # noqa: E402
from mtplx.deepseek_v4_dflash2 import (  # noqa: E402
    DeepseekV4DSparkBackend,
    DeepseekV4DSparkDraftAdapter,
    DeepseekV4TargetTapRows,
    DeepseekV4TargetOps,
    generate_deepseek_v4_dflash2,
)
from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache  # noqa: E402
from mtplx.models.deepseek_v4_dspark import DeepseekV4DSparkCache  # noqa: E402


class _FakeMiaEnginePlan:
    identity = "test-mia-deepseek-v4-engine-plan"
    context_capacity_tokens = 384_000
    target_physical_capacity_tokens = 384_005

    def __init__(self, target_cache_factory=None, events=None) -> None:
        self._target_cache_factory = target_cache_factory
        self.events = events if events is not None else []
        self.target_cache_layers = []
        self.released_target_caches = []
        self.prefill_settlement_calls = []
        self.verify_settlement_calls = []
        self.begin_verify_calls = []
        self.commit_verify_calls = []

    def make_target_cache(self, layers):
        self.target_cache_layers.append(layers)
        if self._target_cache_factory is None:
            raise AssertionError("this fixture does not allocate target caches")
        return self._target_cache_factory()

    def release_target_cache(self, caches) -> None:
        self.released_target_caches.append(caches)

    def settle_target_prefill_chunk(self, *arrays) -> None:
        self.prefill_settlement_calls.append(arrays)

    def schedule_target_verify_chunk(self, *arrays) -> None:
        self.verify_settlement_calls.append(arrays)
        self.events.append("target.schedule_verify")

    def begin_target_verify(self, caches) -> None:
        self.begin_verify_calls.append(caches)
        self.events.append("target.begin_verify")

    def commit_target_verify(self, caches, target_len) -> None:
        self.commit_verify_calls.append((caches, target_len))
        self.events.append("target.commit_verify")


def _seal_fake_target(target, *, target_cache_factory=None) -> _FakeMiaEnginePlan:
    plan = _FakeMiaEnginePlan(
        target_cache_factory,
        events=getattr(target, "events", None),
    )
    target._mia_engine_plan = plan
    target._mia_prewarm_receipt = {"identity": plan.identity}
    return plan


class _FakeDeepseekTarget:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            model_type="deepseek_v4",
            dspark_target_layer_ids=(40, 41, 42),
        )
        self.released_draft_caches = []
        self.dspark = SimpleNamespace(
            stages=(object(), object(), object()),
            release_mia_cache=self.released_draft_caches.append,
        )
        self.model = SimpleNamespace(embed_tokens=object())
        self.layers = (object(),)
        self._target_cache_type = DeepseekV4NVFP4Cache
        self.calls: list[tuple[int, bool]] = []
        self.events: list[str] = []
        self.cache_capacities: list[int | None] = []
        self.plan = _seal_fake_target(
            self,
            target_cache_factory=lambda: self.make_cache(
                capacity_tokens=_FakeMiaEnginePlan.context_capacity_tokens
            ),
        )

    def make_cache(self, *, capacity_tokens=None):
        self.cache_capacities.append(capacity_tokens)
        return [
            DeepseekV4NVFP4Cache(
                window_size=128,
                compress_ratio=0,
                head_dim=512,
            )
        ]

    def __call__(
        self,
        input_ids,
        *,
        cache,
        return_hidden,
        logits_keep=None,
    ):
        self.calls.append((int(input_ids.shape[1]), logits_keep == 1))
        rows = int(input_ids.shape[1])
        logits = mx.zeros((1, 1 if logits_keep == 1 else rows, 64))
        taps = tuple(
            mx.full((1, rows, 2), float(layer_id))
            for layer_id in (40, 41, 42)
        )
        return logits, taps

    def mia_dflash_forward(self, input_ids, cache, *, logits_last_only):
        self.events.append("target.forward")
        return self(
            input_ids,
            cache=cache,
            return_hidden=True,
            logits_keep=1 if logits_last_only else None,
        )


def test_target_ops_uses_physical_m6_and_ordered_deepseek_taps() -> None:
    model = _FakeDeepseekTarget()
    ops = DeepseekV4TargetOps(model)
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        cache_capacity_tokens=64,
    )

    logits, captured = ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[29, 31, 32, 33, 34, 35]], dtype=mx.int32),
        target_cache=cache,
        capture_layer_ids={41, 42, 43},
    )
    features = ops.extract_context_feature(captured, [40, 41, 42])
    ops.settle_prefill_chunk(cache, logits, captured)
    posterior = mx.argmax(logits[0], axis=-1)
    ops.schedule_verify_chunk(cache, posterior)
    ops.restore_after_acceptance(
        cache,
        target_len=3,
        acceptance_length=2,
        drafted_tokens=5,
    )

    assert ops.supports_model(model)
    assert ops.family(model) == "deepseek_v4_dspark"
    assert model.calls == [(6, False)]
    assert model.cache_capacities == [384_000]
    assert model.plan.target_cache_layers == [model.layers]
    assert tuple(logits.shape) == (1, 6, 64)
    assert set(captured) == {41, 42, 43}
    assert model.plan.prefill_settlement_calls == [
        (logits, captured[41], captured[42], captured[43])
    ]
    assert model.plan.verify_settlement_calls == [(posterior,)]
    assert model.plan.begin_verify_calls == [cache]
    assert model.plan.commit_verify_calls == [(cache, 3)]
    assert model.events == [
        "target.begin_verify",
        "target.forward",
        "target.schedule_verify",
        "target.commit_verify",
    ]
    assert tuple(features.shape) == (1, 6, 6)
    np.testing.assert_array_equal(
        np.array(features[0, 0]),
        np.array([[40, 40], [41, 41], [42, 42]], dtype=np.float32),
    )


def test_target_ops_commits_once_without_per_cache_trim(monkeypatch) -> None:
    model = _FakeDeepseekTarget()
    ops = DeepseekV4TargetOps(model)
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
    )
    owner = cache[0]
    monkeypatch.setattr(
        owner,
        "_trim_installed",
        lambda _trim_count: pytest.fail("DFlash must not trim Mia caches per layer"),
    )

    elapsed_ns = ops.restore_after_acceptance(
        cache,
        target_len=131,
        acceptance_length=2,
        drafted_tokens=5,
    )

    assert isinstance(owner, DeepseekV4NVFP4Cache)
    assert owner.window.mode == "nvfp4_stock432"
    assert owner.window.record_bytes == 432
    assert model.plan.commit_verify_calls == [(cache, 131)]
    assert elapsed_ns == 0
    capabilities = ops.capabilities_for(model)
    assert capabilities.supports_dflash is True
    assert capabilities.supports_kv_trim is True
    assert capabilities.supports_target_hidden_capture is True
    assert capabilities.supports_prefix_snapshot is False
    assert capabilities.supports_chunked_prefill is True
    assert capabilities.supports_tree_verify is False


def test_target_ops_releases_cache_when_acquired_layout_is_invalid() -> None:
    model = _FakeDeepseekTarget()
    invalid_cache = [object()]
    plan = _seal_fake_target(
        model,
        target_cache_factory=lambda: invalid_cache,
    )
    ops = DeepseekV4TargetOps(model)

    with pytest.raises(
        ValueError,
        match="requires Mia stock432 target caches",
    ):
        ops.make_cache(
            model,
            enable_speculative_linear_cache=True,
        )

    assert plan.released_target_caches == [invalid_cache]


def test_target_ops_preserves_validation_error_when_cache_release_fails() -> None:
    model = _FakeDeepseekTarget()
    invalid_cache = [object()]
    plan = _seal_fake_target(
        model,
        target_cache_factory=lambda: invalid_cache,
    )

    def fail_release(caches) -> None:
        plan.released_target_caches.append(caches)
        raise RuntimeError("target release failed")

    plan.release_target_cache = fail_release
    ops = DeepseekV4TargetOps(model)

    with pytest.raises(
        ValueError,
        match="requires Mia stock432 target caches",
    ) as exc_info:
        ops.make_cache(
            model,
            enable_speculative_linear_cache=True,
        )

    assert plan.released_target_caches == [invalid_cache]
    assert any(
        "target cache release also failed: RuntimeError: target release failed" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


def test_target_ops_cleanup_attempts_both_arena_releases() -> None:
    model = _FakeDeepseekTarget()
    ops = DeepseekV4TargetOps(model)
    target_cache = [object()]
    draft_cache = [object()]
    releases = []

    def fail_target_release(caches) -> None:
        releases.append(("target", caches))
        raise RuntimeError("target release failed")

    def fail_draft_release(caches) -> None:
        releases.append(("draft", caches))
        raise RuntimeError("draft release failed")

    ops._release_target_cache = fail_target_release
    ops._release_draft_cache = fail_draft_release

    with pytest.raises(RuntimeError, match="target release failed") as exc_info:
        ops.cleanup_generation_caches(target_cache, draft_cache)

    assert releases == [("target", target_cache), ("draft", draft_cache)]
    assert any(
        "additional generation cache release failed: "
        "RuntimeError: draft release failed" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


class _FakeDSparkAttention:
    window_size = 128
    head_dim = 512

    def prefill_context(self, projected, cache) -> None:
        cache.prefill(*self.project_kv(projected, mx.arange(projected.shape[1])))

    def project_kv(self, projected, positions):
        if isinstance(positions, int):
            positions = mx.arange(positions, positions + projected.shape[1])
        assert int(projected.shape[1]) == int(positions.shape[0])
        assert int(projected.shape[-1]) == 512
        return projected, projected[..., -64:]

    def project_context_records(self, projected, start_pos):
        assert isinstance(start_pos, int)
        assert int(projected.shape[-1]) == 512
        return mx.zeros(
            (1, int(projected.shape[1]), 432), dtype=mx.uint8
        )


class _FakeDSparkOwner:
    def __init__(self) -> None:
        self.stages = tuple(
            SimpleNamespace(attn=_FakeDSparkAttention()) for _ in range(3)
        )
        self.projected_rows: mx.array | None = None
        self.proposal_positions: list[int] = []
        self.released_mia_caches = []

    def make_cache(self):
        return [
            DeepseekV4DSparkCache(window_size=128, head_dim=512) for _ in range(3)
        ]

    def release_mia_cache(self, caches) -> None:
        self.released_mia_caches.append(caches)

    def propose_k5(
        self,
        primary_token_ids,
        embed_tokens,
        lm_head,
        caches,
        *,
        start_pos,
    ):
        del embed_tokens, lm_head, caches
        self.proposal_positions.append(int(start_pos))
        assert int(primary_token_ids.item()) == 29
        return SimpleNamespace(
            future_tokens=mx.array([[31, 32, 33, 34, 35]], dtype=mx.uint32)
        )


class _FakeStageZero:
    def __init__(self, owner: _FakeDSparkOwner) -> None:
        self.owner = owner
        self.attn = _FakeDSparkAttention()

    def _run_fuse_main_rows(self, target_rows):
        self.owner.projected_rows = target_rows
        rows = int(target_rows.shape[1])
        return mx.zeros((1, rows, 512), dtype=mx.bfloat16)


def _fake_dspark_target():
    owner = _FakeDSparkOwner()
    owner.stages = (
        _FakeStageZero(owner),
        SimpleNamespace(attn=_FakeDSparkAttention()),
        SimpleNamespace(attn=_FakeDSparkAttention()),
    )
    target = SimpleNamespace(
        args=SimpleNamespace(
            hidden_size=2,
            dspark_target_layer_ids=(40, 41, 42),
            dspark_noise_token_id=128799,
        ),
        dspark=owner,
        layers=(),
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
        _target_cache_type=DeepseekV4NVFP4Cache,
    )
    _seal_fake_target(target)
    return target, owner


def _fake_target_taps(rows: int) -> DeepseekV4TargetTapRows:
    return DeepseekV4TargetTapRows(
        tuple(
            mx.full((1, rows, 2), value)
            for value in (40.0, 41.0, 42.0)
        )
    )


def test_draft_adapter_advertises_m6_but_projects_three_dspark_taps() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    target_taps = _fake_target_taps(2)

    projected = draft.project_target_hidden(target_taps)
    latent = target.dspark.stages[0]._run_fuse_main_rows(projected.fuse_tail(0))

    assert draft.block_size == 6
    assert draft.mask_token_id == 128799
    assert tuple(draft.target_layer_ids) == (40, 41, 42)
    assert draft.capabilities.default_block_tokens == 6
    assert draft.capabilities.max_block_tokens == 6
    assert draft.capabilities.supports_copyspec is False
    assert draft.capabilities.supports_ddtree is False
    assert draft.capabilities.supports_early_rollback_launch is False
    assert draft.capabilities.fixed_physical_block is True
    assert projected is target_taps
    assert tuple(latent.shape) == (1, 2, 512)
    assert owner.projected_rows is not None
    assert tuple(float(tap[0, 0, 0].item()) for tap in projected.taps) == (
        40.0,
        41.0,
        42.0,
    )


def test_draft_backend_releases_cache_when_acquired_layout_is_invalid() -> None:
    target, owner = _fake_dspark_target()
    invalid_cache = [object(), object(), object()]
    owner.make_cache = lambda: invalid_cache
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()

    with pytest.raises(
        ValueError,
        match="must start empty in Mia stock432 format",
    ):
        backend.make_cache(
            draft_model=draft,
            sink_size=0,
            window_size=128,
        )

    assert owner.released_mia_caches == [invalid_cache]


def test_draft_backend_preserves_validation_error_when_cache_release_fails() -> None:
    target, owner = _fake_dspark_target()
    invalid_cache = [object(), object(), object()]
    owner.make_cache = lambda: invalid_cache

    def fail_release(caches) -> None:
        owner.released_mia_caches.append(caches)
        raise RuntimeError("draft release failed")

    owner.release_mia_cache = fail_release
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()

    with pytest.raises(
        ValueError,
        match="must start empty in Mia stock432 format",
    ) as exc_info:
        backend.make_cache(
            draft_model=draft,
            sink_size=0,
            window_size=128,
        )

    assert owner.released_mia_caches == [invalid_cache]
    assert any(
        "draft cache release also failed: RuntimeError: draft release failed" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


def test_draft_backend_appends_committed_context_once_and_returns_five_tokens(
    monkeypatch,
) -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )
    arguments = dict(
        target_model=target,
        target_ops=DeepseekV4TargetOps(target),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        block_len=6,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    scheduled = []
    monkeypatch.setattr(mx, "async_eval", lambda *arrays: scheduled.append(arrays))

    first = backend.draft_greedy(
        **arguments,
        draft_context=_fake_target_taps(4),
    )
    second = backend.draft_greedy(
        **arguments,
        draft_context=_fake_target_taps(2),
    )

    assert tuple(np.array(first)) == (31, 32, 33, 34, 35)
    assert tuple(np.array(second)) == (31, 32, 33, 34, 35)
    assert owner.proposal_positions == [4, 6]
    assert [cache.prefill_length for cache in caches] == [6, 6, 6]
    assert all(cache.ring.mode == "nvfp4_stock432_fixed_ring" for cache in caches)
    assert all(cache.ring.record_bytes == 432 for cache in caches)
    assert len(scheduled) == 2
    for call in scheduled:
        assert len(call) == 4
        for cache in caches:
            assert any(root is cache.ring.records for root in call)


def test_draft_backend_streams_prefill_chunks_without_retaining_or_reappending_prompt() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )
    store = backend.make_target_feature_store(
        prompt_len=20,
        project_context=draft.project_target_hidden,
        draft_model=draft,
        draft_cache=caches,
    )
    raw = _fake_target_taps(20)

    store.write_prompt_slice(start=0, end=10, features=raw[:, :10])
    current = store.write_prompt_slice(start=10, end=20, features=raw[:, 10:])
    drafted = backend.draft_greedy(
        target_model=target,
        target_ops=DeepseekV4TargetOps(target),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        draft_context=current,
        block_len=6,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )

    assert tuple(current.shape) == (1, 0, 6)
    assert [cache.prefill_length for cache in caches] == [20, 20, 20]
    assert owner.proposal_positions == [20]
    assert tuple(np.array(drafted)) == (31, 32, 33, 34, 35)


def test_draft_backend_returns_requested_prefix_for_dflash_final_tail() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )

    tail = backend.draft_greedy(
        target_model=target,
        target_ops=DeepseekV4TargetOps(target),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        draft_context=_fake_target_taps(4),
        block_len=3,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )

    assert tuple(np.array(tail)) == (31, 32)
    assert owner.proposal_positions == [4]


def test_draft_backend_capture_reuses_the_same_dspark_proposal_path() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )

    drafted, top_ids, top_logprobs = backend.draft_greedy_capture(
        target_model=target,
        target_ops=DeepseekV4TargetOps(target),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        draft_context=_fake_target_taps(4),
        block_len=6,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
        top_width=8,
    )

    assert tuple(np.array(drafted)) == (31, 32, 33, 34, 35)
    assert top_ids is None
    assert top_logprobs is None
    assert owner.proposal_positions == [4]


def test_deepseek_bundle_reuses_mtplx_target_and_dflash2_engine_types(
    monkeypatch,
) -> None:
    from mtplx.benchmarks import dflash2_runtime

    target, owner = _fake_dspark_target()
    target.args.model_type = "deepseek_v4"
    target._target_cache_type = DeepseekV4NVFP4Cache

    def reject_capacity_free_probe():
        raise AssertionError("bundle binding must not allocate a capacity-free cache")

    target.make_cache = reject_capacity_free_probe
    tokenizer = object()
    runtime = SimpleNamespace(model=target, tokenizer=tokenizer)
    calls = []
    monkeypatch.setattr(
        dflash2_runtime,
        "_load_mtplx_deepseek_runtime",
        lambda path: calls.append(path) or runtime,
    )
    monkeypatch.setattr(
        dflash2_runtime,
        "require_pinned_dflash_install",
        lambda: pytest.fail("the accepted preflight receipt was read a second time"),
        raising=False,
    )

    bundle = load_mtplx_deepseek_v4_dflash2_bundle(
        "/models/deepseek-v4",
        dflash_identity=PINNED_DFLASH_IDENTITY,
    )

    assert isinstance(bundle, MTPLXDFlash2Bundle)
    assert bundle.runtime is runtime
    assert bundle.target_model is target
    assert bundle.tokenizer is tokenizer
    assert isinstance(bundle.target_ops, DeepseekV4TargetOps)
    assert isinstance(bundle.draft_model, DeepseekV4DSparkDraftAdapter)
    assert isinstance(bundle.draft_backend, DeepseekV4DSparkBackend)
    assert bundle.checkpoint_block_size == 6
    assert bundle.target_layer_ids == (40, 41, 42)
    assert bundle.draft_meta["kind"] == "deepseek_v4_dspark"
    assert calls == ["/models/deepseek-v4"]
    assert bundle.runtime_context.dflash_identity is PINNED_DFLASH_IDENTITY
    assert len(owner.released_mia_caches) == 1


def test_deepseek_bundle_loader_selects_dspark_at_construction(monkeypatch) -> None:
    from mtplx import runtime as runtime_module
    from mtplx.benchmarks import dflash2_runtime

    loaded = object()
    calls = []

    def fake_load(model_path, *, mtp, dspark):
        calls.append((model_path, mtp, dspark))
        return loaded

    monkeypatch.setattr(runtime_module, "load", fake_load)

    assert dflash2_runtime.load_mtplx_deepseek_runtime("model") is loaded
    assert calls == [("model", True, True)]


def test_deepseek_runtime_context_fixes_dflash_m6_without_generic_kv_quantizer() -> None:
    context = build_deepseek_v4_dflash2_runtime_context()

    assert context.runtime.verify_mode == "dflash"
    assert context.runtime.verify_len_cap == 6
    assert context.runtime.copyspec_mode == "off"
    assert context.runtime.quantize_kv_cache is False
    assert context.runtime.prefix_cache is False
    assert context.runtime.dflash_max_ctx == 0
    assert context.dflash_identity.vcs == "git"


def _fake_generation_bundle(tokenizer):
    context = build_deepseek_v4_dflash2_runtime_context()
    target_model = SimpleNamespace(_mia_engine_plan=_FakeMiaEnginePlan())
    return (
        SimpleNamespace(
            target_model=target_model,
            target_ops=object(),
            tokenizer=tokenizer,
            draft_model=object(),
            draft_backend=object(),
            runtime_context=context,
        ),
        context,
    )


def test_generation_adapter_translates_existing_dflash_events_without_scheduling(
    monkeypatch,
) -> None:
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=(11, 12),
        generation_tokens=2,
        accepted_from_draft=1,
        acceptance_ratio=0.5,
        cycles_completed=1,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(1,),
        peak_memory_gb=2.0,
    )
    events = [
        TokenEvent(11, 1, 0.0, 0),
        TokenEvent(12, 2, 0.5, 1),
        summary,
    ]
    calls = []

    def fake_stream(**kwargs):
        calls.append(kwargs)
        return iter(events)

    monkeypatch.setattr(adapter_module, "_stream_dflash_generate", fake_stream)
    callback_tokens = []

    def should_cancel() -> bool:
        return False

    bundle, context = _fake_generation_bundle(
        SimpleNamespace(decode=lambda values: f"decoded:{values}")
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=2,
        token_callback=callback_tokens.append,
        prefill_step_size=256,
        should_cancel=should_cancel,
        runtime_context=context,
    )

    assert output.tokens == [11, 12]
    assert output.text == "decoded:[11, 12]"
    assert callback_tokens == [[11], [12]]
    assert output.final_state is None
    stats = output.stats
    assert stats.mode == "dspark"
    assert stats.generated_tokens == 2
    assert stats.accepted_drafts == 1
    assert stats.drafted_tokens == 5
    assert stats.rejected_drafts == 4
    assert stats.drafted_by_depth == [1, 1, 1, 1, 1]
    assert stats.verify_calls == 1
    assert stats.speculative_depth == 5
    assert stats.decode_elapsed_s == pytest.approx(0.009)
    assert stats.decode_tok_s == pytest.approx(2 / 0.009)
    assert stats.peak_memory_bytes == 2_000_000_000
    assert stats.events == [summary.to_payload()]
    assert len(calls) == 1
    assert calls[0]["block_tokens"] == 6
    assert calls[0]["prompt_tokens_override"] == [1, 2, 3]
    assert calls[0]["quantize_kv_cache"] is False
    assert calls[0]["prefill_step_size"] == 256
    assert calls[0]["should_cancel"] is should_cancel


def test_generation_adapter_stops_at_first_stop_token_and_suppresses_suffix(
    monkeypatch,
) -> None:
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=(11, 12, 13),
        generation_tokens=3,
        accepted_from_draft=2,
        acceptance_ratio=2 / 3,
        cycles_completed=1,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(2,),
        peak_memory_gb=2.0,
    )
    events = [
        TokenEvent(11, 1, 0.0, 0),
        TokenEvent(12, 2, 0.5, 1),
        TokenEvent(13, 3, 2 / 3, 1),
        summary,
    ]
    monkeypatch.setattr(
        adapter_module,
        "_stream_dflash_generate",
        lambda **_kwargs: iter(events),
    )
    callback_tokens = []
    bundle, context = _fake_generation_bundle(
        SimpleNamespace(decode=lambda values: f"decoded:{values}")
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=3,
        stop_token_ids=[12],
        token_callback=callback_tokens.append,
        runtime_context=context,
    )

    assert output.tokens == [11]
    assert output.text == "decoded:[11]"
    assert callback_tokens == [[11]]
    assert output.finish_reason == "stop"


def test_generation_adapter_accounts_for_fixed_m6_terminal_cycles(monkeypatch) -> None:
    from dflash_mlx.engine.events import SummaryEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=tuple(range(6)),
        generation_tokens=6,
        accepted_from_draft=2,
        acceptance_ratio=2 / 6,
        cycles_completed=4,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(0, 1, 0, 1),
    )
    monkeypatch.setattr(
        adapter_module,
        "_stream_dflash_generate",
        lambda **_kwargs: iter([summary]),
    )
    bundle, context = _fake_generation_bundle(
        SimpleNamespace(decode=lambda values: str(values))
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=6,
        runtime_context=context,
    )

    assert output.stats.drafted_tokens == 20
    assert output.stats.rejected_drafts == 18
    assert output.stats.drafted_by_depth == [4, 4, 4, 4, 4]


@pytest.mark.parametrize("remaining_tokens", range(1, 7))
def test_generation_adapter_keeps_logical_384k_admission_with_physical_m6_headroom(
    monkeypatch,
    remaining_tokens,
) -> None:
    from dflash_mlx.engine.events import SummaryEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    observed = []
    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=384_000 - remaining_tokens,
        generated_token_ids=tuple(range(remaining_tokens)),
        generation_tokens=remaining_tokens,
        accepted_from_draft=0,
        acceptance_ratio=0.0,
        cycles_completed=1,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(0,),
    )

    def fake_stream(**kwargs):
        observed.append(kwargs)
        return iter([summary])

    monkeypatch.setattr(adapter_module, "_stream_dflash_generate", fake_stream)
    bundle, context = _fake_generation_bundle(
        SimpleNamespace(decode=lambda values: str(values))
    )
    prompt = [1] * (384_000 - remaining_tokens)

    output = generate_deepseek_v4_dflash2(
        bundle,
        prompt,
        max_tokens=remaining_tokens,
        runtime_context=context,
    )

    assert len(output.tokens) == remaining_tokens
    assert observed[0]["block_tokens"] == 6
    plan = bundle.target_model._mia_engine_plan
    assert plan.context_capacity_tokens == len(prompt) + remaining_tokens
    assert (
        plan.target_physical_capacity_tokens
        == plan.context_capacity_tokens + 5
    )
