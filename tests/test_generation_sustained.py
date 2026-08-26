from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.generation import (
    _clear_cache_every,
    _defer_verify_hidden_eval_enabled,
    _make_target_prefill_cache,
    _maybe_repage_target_prefill_cache,
    _prefill,
    _prefill_chunk_cache_cleanup_every,
    _prefill_chunk_size,
    _prefill_committed_mtp_history_streaming,
    _session_restore_cache_factory,
    _sustained_prefill_layout,
    _trim_cache_to_offset,
    generate_ar,
    generate_mtpk,
    restore_or_prefill_prompt_state,
)
from mtplx.deepseek_v4_mia_engine import MiaDeepseekV4EnginePlan
from mtplx.mtp_patch import MTPContract
from mtplx.profiles import DEFAULT_HF_MODEL_ID
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        position_offset=None,
    ):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        self.calls.append(
            {
                "tokens": int(input_ids.shape[1]),
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class KwargsOnlyTinyModel(TinyModel):
    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        **kwargs,
    ):
        return super().__call__(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )


class AcceptingTinyMTPModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        length = int(next_token_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        logits = mx.zeros((1, length, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class OffsetCache:
    def __init__(self):
        self.offset = 0
        self.trimmed: list[int] = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(self.offset), int(n))
        self.offset -= n
        self.trimmed.append(n)
        return n


def test_trim_cache_to_offset_preflights_all_bounded_entries_atomically():
    from mtplx.models.deepseek_v4 import DeepseekV4Cache

    first = DeepseekV4Cache(
        window_size=16,
        compress_ratio=0,
        head_dim=8,
        rollback_capacity=10,
    )
    second = DeepseekV4Cache(
        window_size=16,
        compress_ratio=0,
        head_dim=8,
        rollback_capacity=2,
    )
    first.offset = second.offset = 10
    assert first.max_rollback == 10
    assert second.max_rollback == 2

    assert _trim_cache_to_offset([first, second], 5) is False
    assert [first.offset, second.offset] == [10, 10]


def test_trim_cache_to_offset_rejects_zero_delta_entry_without_trim_atomically():
    first = OffsetCache()
    first.offset = 10
    second = SimpleNamespace(offset=5, trim=None)

    assert _trim_cache_to_offset([first, second], 5) is False
    assert first.offset == 10
    assert first.trimmed == []


class RejectingTinyMTPModel(AcceptingTinyMTPModel):
    def __init__(self):
        super().__init__()
        self.target_cache = [OffsetCache()]

    def make_cache(self):
        return self.target_cache

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if cache:
            for entry in cache:
                entry.offset += int(input_ids.shape[1])
        return super().__call__(input_ids, cache=cache, **kwargs)

    def mtp_forward(self, *args, **kwargs):
        result = super().mtp_forward(*args, **kwargs)
        if isinstance(result, tuple):
            logits, hidden = result
            logits = mx.zeros_like(logits) + mx.array(
                [0.0, 0.0, 1.0, 0.0],
                dtype=mx.float32,
            )
            return logits, hidden
        return mx.zeros_like(result) + mx.array(
            [0.0, 0.0, 1.0, 0.0],
            dtype=mx.float32,
        )


class TargetOnlyRuntime:
    """A runtime with no MTP head, returning logits ONLY — like Laguna's.

    Mirrors ``_TargetOnlyRuntime`` in test_laguna_fused.py: asking it for hidden
    states is the bug itself, so it says so loudly rather than quietly handing
    back something unpackable.
    """

    def __init__(self, model: TinyModel):
        self.model = model
        self.mtp_enabled = False
        self.model_path = Path("tiny-target-only")
        self.contract = MTPContract()
        self.diagnostic_counters: dict[str, int] = {}

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        assert not return_hidden, (
            "the warm-restore prefill must not ask a target-only runtime for "
            "hidden states; its forward_ar returns logits alone"
        )
        assert hidden_variant is None, (
            "hidden_variant must not travel to a target-only runtime: the "
            "generic runtime forwards it to a model that cannot accept it"
        )
        return self.model(
            input_ids,
            cache=cache,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
        )

    def make_cache(self):
        return self.model.make_cache()

    def repage_target_prefill_cache(self, _cache):
        return False

    @staticmethod
    def target_cache_lifecycle():
        return nullcontext()


def _runtime(model: TinyModel, *, mtp_enabled: bool = True) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )


class _SealedTargetArena:
    def __init__(self, events: list[str]):
        self.events = events
        self.cache: list[object] = []
        self.leased = False

    def acquire(self, _layers):
        if self.leased:
            raise RuntimeError("sealed target cache is already leased")
        self.leased = True
        self.events.append("target.acquire")
        return self.cache

    def release(self, cache):
        assert cache is self.cache
        assert self.leased
        self.leased = False
        self.events.append("target.release")

    def release_active(self):
        if self.leased:
            self.release(self.cache)


class _SealedTargetTinyModel(TinyModel):
    layers = ()

    def __init__(self, plan: MiaDeepseekV4EnginePlan):
        super().__init__()
        self._mia_engine_plan = plan
        self.fail_forward = False

    def make_cache(self):
        return self._mia_engine_plan.make_target_cache(self.layers)

    def __call__(self, *args, **kwargs):
        if self.fail_forward:
            raise RuntimeError("sealed target forward failed")
        return super().__call__(*args, **kwargs)


def _sealed_target_runtime():
    events: list[str] = []
    arena = _SealedTargetArena(events)
    plan = MiaDeepseekV4EnginePlan(
        context_capacity_tokens=384_000,
        target_physical_capacity_tokens=384_005,
        max_batch_tokens=8_224,
        max_sequences=1,
        page_geometry=(),
        workspace_geometry=(),
        indexer_workspace=None,
        indexer_rope_table=None,
        mla_workspace=None,
        target_cache_arena=arena,
        prewarm_signatures=(),
        installed_routes=(),
        target_artifact="target",
        draft_artifact="draft",
        artifact_small_file_sha256=(),
        identity="sealed-target-test",
    )
    model = _SealedTargetTinyModel(plan)
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("sealed-target"),
        mtp_enabled=False,
        contract=MTPContract(),
        backend_id="deepseek_v4_dspark",
    )
    return runtime, model, plan, arena, events


def test_contiguous_then_repage_cache_layout_restores_paged_env(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

        def repage_target_prefill_cache(self, received_cache):
            configure(received_cache)
            return True

    def configure(received_cache):
        events.append(("repage", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
        assert received_cache is cache
        return {"enabled": 1, "entries": 0, "skipped": 0}

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    runtime = Runtime()
    made_cache = _make_target_prefill_cache(runtime)
    elapsed = _maybe_repage_target_prefill_cache(runtime, made_cache)

    assert elapsed >= 0.0
    assert events == [("make_cache", "0"), ("repage", "1")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_contiguous_dense_decode_cache_layout_does_not_repage(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

        def repage_target_prefill_cache(self, received_cache):
            configure(received_cache)
            return True

    def configure(_received_cache):
        raise AssertionError("dense decode layout must not repage after prefill")

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    runtime = Runtime()
    made_cache = _make_target_prefill_cache(runtime)
    elapsed = _maybe_repage_target_prefill_cache(runtime, made_cache)

    assert elapsed == 0.0
    assert events == [("make_cache", "0")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_session_restore_uses_prefill_layout_cache_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            cache_factory = kwargs["cache_factory"]
            assert callable(cache_factory)
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=cache_factory(),
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "clone"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "clone"


@pytest.mark.parametrize("wrapped", [False, True])
def test_sealed_mia_session_restore_keeps_runtime_cache_factory(monkeypatch, wrapped):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    target_model = TinyModel()
    target_model._mia_engine_plan = object()
    runtime_model = (
        SimpleNamespace(language_model=target_model) if wrapped else target_model
    )
    rt = _runtime(runtime_model, mtp_enabled=True)

    assert _session_restore_cache_factory(rt) is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_sealed_mia_prefill_cache_is_never_generically_repaged(monkeypatch, wrapped):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    target_model = TinyModel()
    target_model._mia_engine_plan = object()
    runtime_model = (
        SimpleNamespace(language_model=target_model) if wrapped else target_model
    )

    class Runtime:
        model = runtime_model

        def repage_target_prefill_cache(self, _cache):
            raise AssertionError("the sealed Mia cache arena must not be repaged")

    assert _maybe_repage_target_prefill_cache(Runtime(), [object()]) == 0.0


@pytest.mark.parametrize("wrapped", [False, True])
def test_sealed_mia_session_restore_integrates_without_factory_or_repage(
    monkeypatch,
    wrapped,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    target_model = TinyModel()
    target_model._mia_engine_plan = object()
    runtime_model = (
        SimpleNamespace(language_model=target_model) if wrapped else target_model
    )
    rt = _runtime(runtime_model, mtp_enabled=True)
    rt.repage_target_prefill_cache = lambda _cache: pytest.fail(
        "the sealed Mia cache arena must not be repaged"
    )
    prompt_ids = [0, 1, 2, 3, 4]
    restored_cache = ["sealed-mia-cache"]
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=len(prompt_ids))

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=len(prompt_ids)),
                cache=restored_cache,
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=None,
                restore_mode="clone",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        prompt_ids,
        mtp_history_policy="cycle",
        session_bank=Bank(),
        restore_mode="clone",
        store_prefix_snapshot=False,
    )

    assert captured["cache_factory"] is None
    assert prompt_state.trunk_cache is restored_cache
    assert prompt_state.cache_hit is True
    assert prompt_state.suffix_tokens == 0


def test_live_frontier_reference_restore_survives_prefill_layout_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            assert callable(kwargs["cache_factory"])
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=["live-frontier-cache"],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="reference_lease",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "reference_lease"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "reference_lease"


def test_auto_sustained_prefill_policy_keeps_dense_decode_through_128k(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY", "auto")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", "16384")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", "256")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    # Dense-layout cleanup cadence: every 4 chunks (2026-07-05 A/B — the
    # per-chunk synchronize+clear_cache cost 5-21% prefill throughput with
    # byte-identical peak memory; receipts in MEASUREMENTS).
    assert _prefill_chunk_cache_cleanup_every() == 4
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 4
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "196608")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_cache_cleanup_every() == 2
    assert _clear_cache_every() == 0


def test_auto_sustained_prefill_policy_repages_when_paged_kv_quant_is_enabled(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q4")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "off")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_non_sustained_long_context_prefill_is_blocked_before_full_hidden_eval(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.delenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    with pytest.raises(
        RuntimeError, match="Blocked unsafe long-context MTP prefill path"
    ):
        restore_or_prefill_prompt_state(
            _runtime(model, mtp_enabled=True),
            list(range(8)),
            mtp_history_policy="committed",
        )

    assert model.calls == []


def test_non_sustained_long_context_prefill_guard_has_explicit_escape_hatch(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", "1")
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    restore_or_prefill_prompt_state(
        _runtime(model, mtp_enabled=True),
        list(range(8)),
        mtp_history_policy="committed",
    )

    assert model.calls


def test_generate_ar_does_not_request_hidden_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_AR_RETURN_HIDDEN", raising=False)
    monkeypatch.delenv("MTPLX_DIAGNOSTIC_AR_RETURN_HIDDEN", raising=False)
    model = TinyModel()

    out = generate_ar(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )

    assert out.stats.ar_return_hidden is False
    assert out.stats.forward_ar_hidden_calls == 0
    assert out.stats.forward_ar_plain_calls >= 1
    assert out.stats.prompt_target_prefill_time_s == out.stats.prompt_eval_time_s
    assert out.stats.prompt_mtp_history_time_s == 0.0
    assert out.stats.prompt_target_prefill_tok_s > 0.0
    assert out.stats.tok_s == out.stats.decode_tok_s
    assert out.stats.decode_elapsed_s == pytest.approx(
        out.stats.elapsed_s - out.stats.prompt_eval_time_s
    )
    assert out.stats.end_to_end_tok_s <= out.stats.decode_tok_s
    assert all(call["return_hidden"] is False for call in model.calls)


def test_score_prompt_logprobs_alignment_and_normalization():
    """Prompt scoring: position i predicts token i+1; per-row logprobs are a
    valid distribution slice (sorted descending, <= 0); token_logprobs match
    the target token's entry when it appears in the top-K."""

    from mtplx.generation import score_prompt_logprobs

    rt = _runtime(TinyModel(), mtp_enabled=True)
    prompt = [0, 1, 2, 3]

    scored = score_prompt_logprobs(rt, prompt, top_k=4, chunk_size=2)

    assert scored["prompt_tokens"] == 4
    assert len(scored["positions"]) == 3
    assert len(scored["token_logprobs"]) == 3
    for index, entries in enumerate(scored["positions"]):
        values = [logprob for _token, logprob in entries]
        assert values == sorted(values, reverse=True)
        assert all(value <= 1e-6 for value in values)
        # top_k == vocab here, so the target token must be present and its
        # entry must equal the reported token logprob.
        target = prompt[index + 1]
        by_token = dict(entries)
        assert target in by_token
        assert by_token[target] == pytest.approx(
            scored["token_logprobs"][index], abs=1e-5
        )


def test_sealed_target_ar_dspark_ar_sequence_releases_each_target_lease():
    runtime, model, plan, arena, events = _sealed_target_runtime()
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=4)

    first = generate_ar(
        runtime,
        [0],
        max_tokens=1,
        sampler=sampler,
        stop_token_ids=set(),
    )
    dspark_cache = plan.make_target_cache(model.layers)
    plan.release_target_cache(dspark_cache)
    second = generate_ar(
        runtime,
        [0],
        max_tokens=1,
        sampler=sampler,
        stop_token_ids=set(),
    )

    assert first.tokens == second.tokens == [1]
    assert arena.leased is False
    assert events == [
        "target.acquire",
        "target.release",
        "target.acquire",
        "target.release",
        "target.acquire",
        "target.release",
    ]


def test_sealed_target_ar_error_releases_before_next_acquire():
    runtime, model, plan, arena, events = _sealed_target_runtime()

    def fail_callback(_tokens):
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        generate_ar(
            runtime,
            [0],
            max_tokens=1,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
            stop_token_ids=set(),
            token_callback=fail_callback,
        )

    next_cache = plan.make_target_cache(model.layers)
    plan.release_target_cache(next_cache)
    assert arena.leased is False
    assert events == [
        "target.acquire",
        "target.release",
        "target.acquire",
        "target.release",
    ]


def test_sealed_prompt_scoring_releases_before_dspark_acquire():
    from mtplx.generation import score_prompt_logprobs

    runtime, model, plan, arena, events = _sealed_target_runtime()

    scored = score_prompt_logprobs(runtime, [0, 1, 2], top_k=4, chunk_size=16)
    dspark_cache = plan.make_target_cache(model.layers)
    plan.release_target_cache(dspark_cache)

    assert scored["prompt_tokens"] == 3
    assert arena.leased is False
    assert events == [
        "target.acquire",
        "target.release",
        "target.acquire",
        "target.release",
    ]


def test_sealed_prompt_scoring_error_releases_before_next_acquire():
    from mtplx.generation import score_prompt_logprobs

    runtime, model, plan, arena, events = _sealed_target_runtime()
    model.fail_forward = True

    with pytest.raises(RuntimeError, match="sealed target forward failed"):
        score_prompt_logprobs(runtime, [0, 1], top_k=4, chunk_size=16)

    model.fail_forward = False
    next_cache = plan.make_target_cache(model.layers)
    plan.release_target_cache(next_cache)
    assert arena.leased is False
    assert events == [
        "target.acquire",
        "target.release",
        "target.acquire",
        "target.release",
    ]


def test_generate_ar_restores_warm_prefix_from_session_bank():
    """#246: the AR lane used to full-prefill unconditionally and hardcode
    cached_tokens 0 / cache_hit false. With a bank hit it must restore the
    prefix and report real numbers."""

    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            cache_factory = kwargs.get("cache_factory")
            cache = cache_factory() if callable(cache_factory) else _rt.make_cache()
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=cache,
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=None,
                mtp_history_cache=None,
                restore_mode="clone",
            )

    out = generate_ar(
        rt,
        [0, 1, 2, 3, 4],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
        session_bank=Bank(),
        session_id="ar-warm-session",
    )

    assert out.stats.session_cache_hit is True
    assert out.stats.cached_tokens > 0
    assert out.stats.new_prefill_tokens < 5
    assert len(out.tokens) == 2


def test_generate_ar_cold_output_identical_with_and_without_bank():
    """Empty-bank receipt: routing AR through restore_or_prefill must not
    change cold-path outputs."""

    class EmptyBank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, _rt, _prompt_ids, **kwargs):
            return None

    prompt = [0, 1, 2, 3]
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=4)
    out_no_bank = generate_ar(
        _runtime(TinyModel(), mtp_enabled=True),
        list(prompt),
        max_tokens=3,
        sampler=sampler,
        stop_token_ids=set(),
    )
    out_empty_bank = generate_ar(
        _runtime(TinyModel(), mtp_enabled=True),
        list(prompt),
        max_tokens=3,
        sampler=sampler,
        stop_token_ids=set(),
        session_bank=EmptyBank(),
        session_id="ar-cold-session",
    )

    assert out_no_bank.tokens == out_empty_bank.tokens
    assert out_empty_bank.stats.cached_tokens == 0
    assert out_empty_bank.stats.session_cache_hit is False


def test_generate_ar_captures_final_state_for_bank_commit():
    """capture_final_state must produce a committable state whose token ids
    match the generated tokens exactly (the committer refuses mismatches)."""

    out = generate_ar(
        _runtime(TinyModel(), mtp_enabled=True),
        [0, 1, 2],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
        capture_final_state=True,
    )

    assert out.final_state is not None
    assert out.final_state.safe_to_commit is True
    assert out.final_state.generated_token_ids == tuple(out.tokens)
    assert out.final_state.final_committed_mtp_cache is None
    assert out.final_state.mtp_history_policy == "cycle"


def test_default_qwen27b_ar_decode_trace_does_not_crash(tmp_path, monkeypatch):
    trace_path = tmp_path / "qwen27b-ar.jsonl"
    monkeypatch.setenv("MTPLX_DECODE_TRACE_JSONL", str(trace_path))
    monkeypatch.setenv("MTPLX_DECODE_TRACE_INTERVAL_S", "0.1")
    runtime = _runtime(TinyModel(), mtp_enabled=True)
    runtime.model_path = Path(DEFAULT_HF_MODEL_ID)

    output = generate_ar(
        runtime,
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert output.tokens == [1, 1]
    assert rows[-1]["final"] is True
    assert rows[-1]["generated_tokens_total"] == 2
    assert rows[-1]["target_distribution_materialized_rows_delta"] == 0


def test_lazy_bonus_verify_shortens_full_accept_verify_input(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 3, 1]
    assert out.stats.verify_calls == 1
    assert out.stats.commit_time_s > 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is True
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 3
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 3


def test_lazy_target_distributions_inline_bonus_avoids_bonus_reforward(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 4]
    assert out.stats.verify_calls == 1
    assert out.stats.lazy_bonus_commit_time_s == 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is False
    assert (
        out.stats.events[0]["lazy_bonus_verify"]["disabled_by"]
        == "lazy_target_distributions"
    )
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 4
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})
    assert out.stats.events[0]["target_distribution_materialized"]["mode"] == (
        "lazy_accept_bonus_path"
    )


def test_lazy_target_distributions_stop_after_first_rejection(monkeypatch):
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:1] == [1]
    assert out.stats.events[0]["rejected_at_depth"] == 1
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.target_distribution_materialized_windows == 1
    assert out.stats.events[0]["target_distribution_materialized"]["rows"] == 1


@pytest.mark.parametrize(
    ("model_cls", "sampler"),
    [
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)),
        (RejectingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=1.0, top_k=1)),
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)),
    ],
)
def test_lazy_target_distributions_match_dense_reference(
    monkeypatch,
    model_cls,
    sampler,
):
    def run_once(*, lazy: bool):
        monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
        monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
        monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
        if lazy:
            monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
        else:
            monkeypatch.delenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", raising=False)
        return generate_mtpk(
            _runtime(model_cls(), mtp_enabled=True),
            [0],
            max_tokens=5,
            sampler=sampler,
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
            seed=123,
        )

    dense = run_once(lazy=False)
    lazy = run_once(lazy=True)

    assert lazy.tokens == dense.tokens
    assert lazy.stats.accepted_by_depth == dense.stats.accepted_by_depth
    assert lazy.stats.drafted_by_depth == dense.stats.drafted_by_depth
    assert lazy.stats.rejected_drafts == dense.stats.rejected_drafts
    assert lazy.stats.bonus_tokens == dense.stats.bonus_tokens
    assert lazy.finish_reason == dense.finish_reason


def test_lazy_bonus_verify_skips_d1_by_default(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1, 1]
    lazy = out.stats.events[0]["lazy_bonus_verify"]
    assert lazy["enabled"] is False
    assert lazy["min_depth"] == 2
    assert lazy["verify_input_tokens"] == 2
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})


def test_omit_speculative_bonus_skips_bonus_distribution_row(monkeypatch):
    monkeypatch.setenv("MTPLX_OMIT_SPECULATIVE_BONUS", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.events[0]["speculative_bonus"] == {
        "omitted": True,
        "distribution_row_needed": False,
    }
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 1
    assert "bonus_token" not in out.stats.events[0]
    assert out.stats.bonus_tokens == 0


def test_trim_commit_keeps_rejected_verify_prefix_without_reforward(monkeypatch):
    monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=1,
        mtp_history_policy="cycle",
        verify_strategy="trim_commit",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert [call["tokens"] for call in model.calls] == [1, 2]
    assert model.target_cache[0].trimmed == [1]
    assert out.stats.events[0]["capture_repair"] == "trimmed_prefix_commit"
    assert "repair_forward" not in out.stats.events[0].get("timing_s", {})


def test_mtpk_draft_time_is_decode_only_and_excludes_prompt_mtp_history(monkeypatch):
    """draft_time_s must be a decode-window bucket.

    Prefill MTP-history time is already reported separately in
    prompt_mtp_history_time_s (and subtracted from prompt_target_prefill).
    Folding it into draft_time_s as well made exported stats look impossible
    at long context (256k Ivan-ladder row: draft 83s inside a 19s decode
    window) and disagreed with generate_mtp1/generate_mtpa, which both report
    decode-only draft time.
    """
    import mtplx.generation as generation_mod

    real_restore = generation_mod.restore_or_prefill_prompt_state

    def fake_restore(*args, **kwargs):
        state = real_restore(*args, **kwargs)
        state.prompt_mtp_history_time_s = 123.0
        return state

    monkeypatch.setattr(
        generation_mod, "restore_or_prefill_prompt_state", fake_restore
    )

    out = generate_mtpk(
        _runtime(AcceptingTinyMTPModel(), mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    # The prompt-side bucket carries the injected time untouched...
    assert out.stats.prompt_mtp_history_time_s == 123.0
    # ...and the decode-side draft bucket does not absorb it.
    assert out.stats.draft_time_s < 60.0
    # prompt_eval here is tiny, so the target-prefill share clamps to zero.
    assert out.stats.prompt_target_prefill_time_s == 0.0


def test_sustained_prefill_chunks_without_full_prompt_logits(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert model.calls[-1]["logits_keep"] == 1
    assert rt.diagnostic_counters["prefill_chunks"] == 2
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0
    assert rt.diagnostic_counters["final_logits_tokens_emitted"] == 1


def test_warm_restored_suffix_prefill_is_chunked_and_typed_for_abort(monkeypatch):
    # kvcache-v2: suffixes <= MTPLX_SMALL_SUFFIX_FUSED_MAX fuse into one
    # forward; this test guards the chunked lane used above that threshold.
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    prefill_events: list[dict[str, object]] = []

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6],
        mtp_history_policy="committed",
        session_bank=Bank(),
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 3
    assert prompt_state.suffix_tokens == 4
    assert [call["tokens"] for call in model.calls] == [2, 1, 1]
    assert [call["return_hidden"] for call in model.calls] == [True, True, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert appended == [[3], [4, 5], [6]]
    assert rt.diagnostic_counters["restored_suffix_prefill_chunks"] == 2
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    assert [event["tokens_done"] for event in chunk_events] == [3, 5, 6, 7]
    assert [event["tokens_total"] for event in chunk_events] == [7, 7, 7, 7]
    assert [event["cached_tokens"] for event in chunk_events] == [3, 3, 3, 3]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [4, 4, 4, 4]
    assert chunk_events[-1]["live_prefill_tok_s"] is not None


@pytest.mark.parametrize(
    ("lane", "fused_max", "suffix_len"),
    [("fused", 64, 2), ("chunked", 0, 4)],
)
def test_warm_restore_never_asks_a_target_only_runtime_for_hidden(
    monkeypatch, lane, fused_max, suffix_len
):
    """Regression for the live serving crash at _prefill_restored_prompt_suffix.

    Every warm restore asked the runtime for hidden states unconditionally, on
    both lanes. A target-only runtime returns logits alone, so unpacking that
    into ``(logits, hidden)`` raised ``ValueError: not enough values to unpack
    (expected 2, got 1)``. The double asserts the request is never made at all
    — including hidden_variant, which the generic runtime would forward to a
    model that cannot accept it.
    """

    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", str(fused_max))
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    model = TinyModel()
    rt = TargetOnlyRuntime(model)

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                # An AR turn banks the trunk cache only: no hidden was stored.
                hidden=None,
                mtp_history_cache=None,
                restore_mode="clone",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(3 + suffix_len)),
        # The server hands AR runtimes the committed policy; the chokepoint
        # guard downgrades it to cycle, and the suffix prefill must honor that.
        mtp_history_policy="committed",
        session_bank=Bank(),
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.mtp_history_policy == "cycle"
    assert prompt_state.cached_tokens == 3
    assert prompt_state.suffix_tokens == suffix_len
    assert prompt_state.hidden is None
    assert prompt_state.logits is not None
    assert not any(call["return_hidden"] for call in model.calls)
    if lane == "fused":
        assert rt.diagnostic_counters["restored_suffix_prefill_fused"] == 1
    else:
        assert rt.diagnostic_counters["restored_suffix_prefill_chunks"] >= 1


def test_restore_prefers_larger_near_gap_over_shorter_exact_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []

    exact_entry = SimpleNamespace(prefix_len=3)
    near_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-1",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            # kvcache-v2: with boundary-true restore on (default), the
            # block-prefix lane is env-decided (default on) for every
            # client, not just OpenCode-compact — restores fail closed at
            # the entry layer instead (issue #138).
            assert kwargs["allow_block_prefix"] is True
            return [(near_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()
    prefill_events: list[dict[str, object]] = []

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 7
    assert prompt_state.suffix_tokens == 2
    assert prompt_state.restore_mode == "near_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.prefix_restore_calls == [(7, "clone")]
    assert near_entry.hits == 1
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    # kvcache-v2 fused small-suffix prefill emits one progress event for the
    # whole (tiny) suffix instead of per-chunk events.
    assert [event["tokens_done"] for event in chunk_events] == [7, 9]
    assert [event["cached_tokens"] for event in chunk_events] == [7, 7]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [2, 2]


def test_opencode_compact_restore_prefers_block_prefix_over_short_exact(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    opencode_compact_policy = (
        "tool_prompt_mode=compact;"
        "tool_contract=compact_tool_contract:schema_free:v1;"
        "opencode_prompt_contract=opencode_agent"
    )

    exact_entry = SimpleNamespace(prefix_len=3)
    block_entry = SimpleNamespace(
        prefix_len=20,
        token_ids=tuple(range(20)),
        session_id="opencode-session",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=opencode_compact_policy,
        snapshot_epoch=20,
        mtp_snapshot_epoch=20,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []
            self.near_allow_block: list[bool] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_allow_block.append(bool(kwargs["allow_block_prefix"]))
            if not kwargs["allow_block_prefix"]:
                return []
            return [(block_entry, 8)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=opencode_compact_policy,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 8
    assert prompt_state.suffix_tokens == 4
    assert prompt_state.restore_mode == "block_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.near_allow_block == [True]
    assert bank.prefix_restore_calls == [(8, "clone")]
    assert block_entry.hits == 1
    # kvcache-v2 fused small-suffix prefill appends the post-first-token
    # history rows in one call instead of body/final chunks — same rows, same
    # hidden positions, one eval barrier.
    assert appended == [[8], [9, 10, 11]]


def _make_frozen_prefix_bank_fixture(rt, *, policy_fingerprint=None):
    """Bank shape from issue #138: a stale short exact-prefix entry plus a
    much longer entry sharing a bigger prompt prefix (gap > tiny-gap limit).
    Before the fix, non-OpenCode clients could only take the tiny-gap lane,
    so every restore froze on the short exact entry."""
    exact_entry = SimpleNamespace(prefix_len=3)
    block_entry = SimpleNamespace(
        prefix_len=20,
        token_ids=tuple(range(20)),
        session_id="agent-session",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=policy_fingerprint,
        snapshot_epoch=20,
        mtp_snapshot_epoch=20,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []
            self.near_allow_block: list[bool] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_allow_block.append(bool(kwargs["allow_block_prefix"]))
            if not kwargs["allow_block_prefix"]:
                return []
            return [(block_entry, 8)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    return Bank(), exact_entry, block_entry


def _install_history_stub(monkeypatch):
    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        assert hidden_states.shape[1] == len(token_ids)
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)


def test_generic_client_escapes_stale_short_exact_prefix_via_block_restore(
    monkeypatch,
):
    """Issue #138: Pi/little-coder style clients (no OpenCode-compact
    fingerprint) froze on the oldest short exact prefix while longer banked
    prefixes went unused, re-prefilling a growing suffix every turn. With
    boundary-true restore on (the v2 default), the block-prefix lane is safe
    and must engage for every client."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, _exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 8
    assert prompt_state.restore_mode == "block_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.near_allow_block == [True]
    assert bank.prefix_restore_calls == [(8, "clone")]
    assert block_entry.hits == 1


def test_generic_client_block_restore_respects_boundary_true_off_switch(
    monkeypatch,
):
    """With MTPLX_SESSION_BOUNDARY_TRUE_RESTORE=0 the pre-v2 caution comes
    back for non-OpenCode clients: tiny-gap only, exact restore wins."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cached_tokens == exact_entry.prefix_len
    assert bank.restore_calls == 1
    assert bank.near_allow_block[0] is False
    assert block_entry.hits == 0


def test_generic_client_block_restore_respects_block_prefix_kill_switch(
    monkeypatch,
):
    """MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0 must still disable the block
    lane for generic clients even with boundary-true restore on."""
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    _install_history_stub(monkeypatch)
    fingerprint = "tool_prompt_mode=hybrid;client=pi"
    bank, exact_entry, block_entry = _make_frozen_prefix_bank_fixture(
        rt, policy_fingerprint=fingerprint
    )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=fingerprint,
    )

    assert prompt_state.cached_tokens == exact_entry.prefix_len
    assert bank.restore_calls == 1
    assert bank.near_allow_block[0] is False
    assert block_entry.hits == 0


def test_ssd_near_prefix_restore_time_is_cache_time_not_decode_time(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    ssd_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-ssd",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        cache_source="ssd",
        ssd_cache_hit=True,
        ssd_restore_s=1.25,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **_kwargs):
            return [(ssd_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert int(prefix_len) == 7
            assert mode == "clone"
            assert cache_factory is None or callable(cache_factory)
            return [], [], "clone"

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=Bank(),
    )

    assert prompt_state.cache_source == "ssd"
    assert prompt_state.ssd_cache_hit is True
    assert prompt_state.ssd_restore_s == 1.25
    assert prompt_state.cache_restore_time_s >= 1.25


def test_block_prefix_restore_matches_target_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def __init__(self):
            self.near_kwargs: list[dict[str, object]] = []

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_kwargs.append(kwargs)
            return []

    bank = Bank()
    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
    )

    assert prompt_state.cache_hit is False
    assert prompt_state.cached_tokens == 0
    assert bank.near_kwargs
    assert bank.near_kwargs[-1]["allow_block_prefix"] is True


def test_sustained_prefill_chunk_cache_cleanup_is_explicit(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP", "1")
    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        "mtplx.generation.mx.clear_cache", lambda: calls.append("clear")
    )
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert calls == ["sync", "clear", "sync", "clear"]
    assert rt.diagnostic_counters["prefill_chunk_cache_cleanup_events"] == 2


def test_sustained_prefill_stock_cache_only_requires_unsafe_allow(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_stock_cache_only_is_explicit_unsafe(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_stock_cache_only_calls"] == 2


def test_sustained_prefill_omlx_external_is_safe_profile_path(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_OMLX_EXTERNAL", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_omlx_external_calls"] == 2
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_forwards_logits_controls_through_patched_kwargs_wrapper(
    monkeypatch,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = KwargsOnlyTinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0


def test_last_window_mtp_history_skips_discarded_chunk_hidden(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[tuple[list[int], int | None]] = []

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        phase,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert phase == "prefill"
        appended.append((list(token_ids), position_offset))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    _prefill_committed_mtp_history_streaming(
        rt,
        list(range(9)),
        mtp_hidden_variant="post_norm",
        history_window_tokens=3,
    )

    assert [call["tokens"] for call in model.calls] == [2, 2, 2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert appended == [([6], 5), ([7, 8], 6)]


def test_32k_prefill_peak_memory_bounded():
    """
    Regression guard for the Ivan/Benchand 32K memory balloon.
    Run only on the Apple Silicon long-context QA machine.
    """
    if os.environ.get("MTPLX_RUN_32K_MEMORY_QA") != "1":
        pytest.skip("set MTPLX_RUN_32K_MEMORY_QA=1 on the long-context QA Mac")
    model_path = os.environ.get("MTPLX_32K_QA_MODEL")
    if not model_path:
        pytest.skip("set MTPLX_32K_QA_MODEL to a local runnable MTPLX model")

    from mtplx.runtime import load

    rt = load(model_path, mtp=True)
    text = "def f(x): return x + 1\n" * 4096
    prompt_ids = rt.tokenizer.encode(text)[:32768]
    if len(prompt_ids) < 32000:
        pytest.skip("QA prompt did not tokenize to 32K tokens")

    mx.reset_peak_memory()
    os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2048"
    os.environ["MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"] = "0"
    _prefill(rt, prompt_ids, return_hidden=True)
    peak_gb = mx.get_peak_memory() / (1024**3)

    assert peak_gb < 35.0, f"32K Sustained prefill peak was {peak_gb:.1f} GB"
