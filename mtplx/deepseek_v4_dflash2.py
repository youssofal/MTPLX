"""DeepSeek V4 DSpark adapters for the existing DFlash2 engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.engine.target_features import StreamingTargetFeatureStore
from dflash_mlx.model import DraftRuntimeCapabilities

from mtplx.dflash_identity import (
    InstalledDFlashIdentity,
    PINNED_DFLASH_COMMIT,
)
from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache


_TARGET_LAYER_IDS = (40, 41, 42)
_CAPTURE_LAYER_IDS = tuple(layer_id + 1 for layer_id in _TARGET_LAYER_IDS)
_PHYSICAL_VERIFY_WIDTH = 6
_DFLASH_RUNTIME_IDENTITY = (
    "mia-deepseek-v4:dflash:fixed-linear:m6:prefill1024:window128:"
    f"stock432:copyspec-zero-owner-{PINNED_DFLASH_COMMIT}:"
    "adaptive-off:fallback-off"
)


@dataclass(frozen=True, slots=True)
class DeepseekV4DFlashRuntimeContext:
    """Frozen construction receipt for the sole Mia DFlash execution lane."""

    runtime: Any
    diagnostics: Any
    verify: Any
    metal_limits: Any
    dflash_identity: InstalledDFlashIdentity
    identity: str

    @classmethod
    def install(
        cls,
        context: Any,
        *,
        dflash_identity: InstalledDFlashIdentity,
    ) -> "DeepseekV4DFlashRuntimeContext":
        runtime = context.runtime
        required_runtime = {
            "prefill_step_size": 1024,
            "draft_sink_size": 0,
            "draft_window_size": 128,
            "verify_len_cap": _PHYSICAL_VERIFY_WIDTH,
            "prefix_cache": False,
            "prefix_cache_l2": False,
            "clear_cache_boundaries": False,
            "target_fa_window": 0,
            # Zero disables the generic DFlash AR cutoff. The model-owned page
            # plan rejects over-capacity requests before DFlash is entered.
            "dflash_max_ctx": 0,
            "verify_mode": "dflash",
            "copyspec_mode": "off",
            "quantize_kv_cache": False,
        }
        changed = {
            name: getattr(runtime, name, None)
            for name, expected in required_runtime.items()
            if getattr(runtime, name, None) != expected
        }
        if changed:
            raise ValueError(
                "DeepSeek V4 DFlash runtime does not match the installed Mia route: "
                f"{changed}"
            )
        if getattr(context.verify, "mode", None) != "dflash":
            raise ValueError("DeepSeek V4 DFlash requires the fixed linear verifier")
        diagnostics = context.diagnostics
        if (
            getattr(diagnostics, "mode", None) != "off"
            or bool(getattr(diagnostics, "memory_waterfall", False))
            or bool(getattr(getattr(diagnostics, "trace", None), "cycle_events", False))
        ):
            raise ValueError("DeepSeek V4 DFlash diagnostics must be off at installation")
        return cls(
            runtime=runtime,
            diagnostics=diagnostics,
            verify=context.verify,
            metal_limits=context.metal_limits,
            dflash_identity=dflash_identity,
            identity=_DFLASH_RUNTIME_IDENTITY,
        )


class DeepseekV4TargetTapRows(tuple):
    """Three ordered target-tap views without a full-width concatenation.

    It is deliberately a tuple-native MLX tree.  The shared DFlash scheduler
    can therefore evaluate the three arrays without a model-specific runtime
    branch, while tuple-shaped tensor indexing still returns a structured view.
    """

    def __new__(cls, taps: tuple[mx.array, mx.array, mx.array]):
        return super().__new__(cls, taps)

    @property
    def taps(self) -> tuple[mx.array, mx.array, mx.array]:
        return self[0], self[1], self[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        first = self.taps[0]
        return int(first.shape[0]), int(first.shape[1]), sum(
            int(tap.shape[-1]) for tap in self.taps
        )

    @property
    def rows(self) -> int:
        return int(self.taps[0].shape[1])

    def __getitem__(self, key: Any):
        if isinstance(key, int):
            return super().__getitem__(key)
        return DeepseekV4TargetTapRows(tuple(tap[key] for tap in self))

    def fuse_tail(self, first_row: int) -> mx.array:
        return mx.concatenate(
            tuple(tap[:, int(first_row) :, :] for tap in self.taps),
            axis=-1,
        )


@dataclass
class DeepseekV4StreamingTargetFeatureStore(StreamingTargetFeatureStore):
    """Existing streaming store specialized to structured DeepSeek taps."""

    def _project(self, features: Any) -> DeepseekV4TargetTapRows:
        return features

    def write_prompt_slice(
        self,
        *,
        start: int,
        end: int,
        features: Any,
    ) -> DeepseekV4TargetTapRows:
        start = int(start)
        end = int(end)
        projected = self._project(features)
        self.consume_prompt_chunk(start=start, end=end, features=projected)
        self._current_hidden = projected[:, :0, :]
        return self._current_hidden

    def commit_generation(
        self,
        committed_hidden: Any,
        *,
        collect_snapshot: bool,
    ) -> None:
        del collect_snapshot
        self._current_hidden = self._project(committed_hidden)


class DeepseekV4TargetOps:
    """Bind a construction-qualified DeepSeek V4 target to DFlash2."""

    backend_name = "deepseek_v4_dspark"

    def __init__(self, target_model: Any) -> None:
        plan = getattr(target_model, "_mia_engine_plan", None)
        receipt = getattr(target_model, "_mia_prewarm_receipt", None)
        if plan is None or not isinstance(receipt, dict):
            raise ValueError(
                "DeepSeek V4 DFlash2 requires the sealed and prewarmed Mia engine"
            )
        if str(receipt.get("identity")) != str(plan.identity):
            raise ValueError("DeepSeek V4 Mia prewarm receipt does not match its plan")
        self._target_model = target_model
        self._plan = plan
        self._make_target_cache = plan.make_target_cache
        self._release_target_cache = plan.release_target_cache
        self._settle_target_prefill_chunk = plan.settle_target_prefill_chunk
        self._schedule_target_verify_chunk = plan.schedule_target_verify_chunk
        self._begin_target_verify = plan.begin_target_verify
        self._commit_target_verify = plan.commit_target_verify
        self._release_draft_cache = target_model.dspark.release_mia_cache

    def model_type(self, target_model: Any) -> str:
        return str(getattr(getattr(target_model, "args", None), "model_type", "")).lower()

    def supports_model(self, target_model: Any) -> bool:
        stages = tuple(getattr(getattr(target_model, "dspark", None), "stages", ()))
        layer_ids = tuple(
            int(value)
            for value in (
                getattr(
                    getattr(target_model, "args", None),
                    "dspark_target_layer_ids",
                    (),
                )
                or ()
            )
        )
        return (
            target_model is self._target_model
            and self.model_type(target_model) == "deepseek_v4"
            and len(stages) == 3
            and layer_ids == _TARGET_LAYER_IDS
            and getattr(target_model, "_target_cache_type", None)
            is DeepseekV4NVFP4Cache
        )

    def family(self, target_model: Any) -> str:
        del target_model
        return self.backend_name

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=False,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            supports_verify_linear=True,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
            supports_chunked_prefill=True,
            supports_fixed_linear_runtime=True,
            fixed_linear_restore_without_arming=True,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_model(self, target_model: Any) -> Any:
        return target_model.model

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self,
        target_model: Any,
        hidden_states: mx.array,
    ) -> mx.array:
        return target_model.lm_head(hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
        cache_capacity_tokens: Optional[int] = None,
    ) -> list[Any]:
        del enable_speculative_linear_cache
        if target_model is not self._target_model:
            raise ValueError("DeepSeek V4 target ops are bound to one Mia engine")
        if quantize_kv_cache:
            raise ValueError(
                "DeepSeek V4 target K/V is already Mia stock432 NVFP4 from offset zero"
            )
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("DeepSeek V4 uses its model-defined attention windows")
        if (
            cache_capacity_tokens is not None
            and int(cache_capacity_tokens) > int(self._plan.context_capacity_tokens)
        ):
            raise ValueError(
                "DeepSeek V4 request exceeds the installed Mia context capacity"
            )
        cache = self._make_target_cache(target_model.layers)
        try:
            if not cache or not all(
                isinstance(entry, DeepseekV4NVFP4Cache) for entry in cache
            ):
                raise ValueError(
                    "DeepSeek V4 DFlash2 requires Mia stock432 target caches"
                )
        except BaseException as primary_error:
            try:
                self._release_target_cache(cache)
            except BaseException as release_error:
                primary_error.add_note(
                    "target cache release also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )
            raise
        return cache

    def install_speculative_hooks(self, target_model: Any) -> None:
        del target_model

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        del input_embeddings
        return self._forward_with_hidden_capture_phase(
            target_model,
            input_ids=input_ids,
            cache=cache,
            input_embeddings=None,
            capture_layer_ids=capture_layer_ids,
            logits_last_only=logits_last_only,
            phase="prefill",
        )

    def settle_prefill_chunk(
        self,
        cache_entries: list[Any],
        logits: mx.array,
        captured: dict[int, mx.array],
    ) -> None:
        del cache_entries
        self._settle_target_prefill_chunk(logits, *captured.values())

    def schedule_verify_chunk(
        self,
        cache_entries: list[Any],
        posterior: mx.array,
    ) -> None:
        del cache_entries
        self._schedule_target_verify_chunk(posterior)

    def _forward_with_hidden_capture_phase(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array],
        cache: Optional[list[Any]],
        input_embeddings: Optional[mx.array],
        capture_layer_ids: Optional[set[int]],
        logits_last_only: bool,
        phase: str,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        from mtplx.attention_context import attention_phase

        del input_embeddings
        with attention_phase(phase):
            logits, taps = target_model.mia_dflash_forward(
                input_ids,
                cache,
                logits_last_only=logits_last_only,
            )
        del capture_layer_ids
        return logits, dict(zip(_CAPTURE_LAYER_IDS, taps))

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        self._begin_target_verify(target_cache)
        return self._forward_with_hidden_capture_phase(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            input_embeddings=None,
            capture_layer_ids=capture_layer_ids,
            logits_last_only=False,
            phase="decode_verify",
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("DeepSeek V4 DSpark does not support DDTree")

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("DeepSeek V4 DSpark does not support DDTree")

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> DeepseekV4TargetTapRows:
        del target_layer_ids
        return DeepseekV4TargetTapRows(
            tuple(captured_dict[layer_id + 1] for layer_id in _TARGET_LAYER_IDS)
        )

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        del cache_entries, prefix_len

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        del acceptance_length, drafted_tokens
        self._commit_target_verify(cache_entries, target_len)
        return 0

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        release_errors: list[BaseException] = []
        if target_cache:
            try:
                self._release_target_cache(target_cache)
            except BaseException as error:
                release_errors.append(error)
        if draft_cache:
            try:
                self._release_draft_cache(draft_cache)
            except BaseException as error:
                release_errors.append(error)
        if release_errors:
            primary_error, *additional_errors = release_errors
            for error in additional_errors:
                primary_error.add_note(
                    "additional generation cache release failed: "
                    f"{type(error).__name__}: {error}"
                )
            raise primary_error


class DeepseekV4DSparkDraftAdapter:
    """Expose the installed K5 DSpark owner through DFlash2's draft metadata."""

    def __init__(self, target_model: Any) -> None:
        owner = getattr(target_model, "dspark", None)
        stages = tuple(getattr(owner, "stages", ()))
        if len(stages) != 3:
            raise ValueError("DeepSeek V4 DFlash2 requires exactly three DSpark stages")
        target_layer_ids = tuple(
            int(value)
            for value in (
                getattr(target_model.args, "dspark_target_layer_ids", ()) or ()
            )
        )
        if target_layer_ids != _TARGET_LAYER_IDS:
            raise ValueError("DeepSeek V4 DFlash2 requires target taps 40/41/42")
        make_cache = getattr(owner, "make_cache", None)
        release_cache = getattr(owner, "release_mia_cache", None)
        if not callable(make_cache) or not callable(release_cache):
            raise ValueError(
                "DeepSeek V4 DFlash2 requires the sealed DSpark cache arena"
            )

        self.target_model = target_model
        self.owner = owner
        self._make_cache = make_cache
        self._release_cache = release_cache
        self.target_layer_ids = list(_TARGET_LAYER_IDS)
        self.block_size = _PHYSICAL_VERIFY_WIDTH
        self.mask_token_id = int(target_model.args.dspark_noise_token_id)
        self.args = SimpleNamespace(
            sliding_window=max(int(stage.attn.window_size) for stage in stages),
            layer_types=("sliding_attention",) * len(stages),
        )
        self.capabilities = DraftRuntimeCapabilities(
            default_block_tokens=_PHYSICAL_VERIFY_WIDTH,
            max_block_tokens=_PHYSICAL_VERIFY_WIDTH,
            supports_copyspec=False,
            supports_ddtree=False,
            supports_early_rollback_launch=False,
            fixed_physical_block=True,
        )

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        if target_model is not self.target_model or not isinstance(
            target_ops, DeepseekV4TargetOps
        ):
            raise ValueError("DSpark draft adapter is bound to one DeepSeek target")

    def project_target_hidden(
        self,
        target_hidden: DeepseekV4TargetTapRows,
    ) -> DeepseekV4TargetTapRows:
        # The streaming owner must slice first: projecting all 8,224 captured
        # rows here would immediately discard 8,096 of them for a 128-row ring.
        return target_hidden


class DeepseekV4DSparkBackend:
    """Append accepted target context and invoke the installed DSpark K5 model."""

    def make_target_feature_store(
        self,
        *,
        prompt_len: int,
        project_context: Any,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
    ):
        def consume_prompt_chunk(
            *,
            start: int,
            end: int,
            features: DeepseekV4TargetTapRows,
        ) -> None:
            del start, end
            self._append_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                draft_context=features,
            )
            mx.eval(*(cache.ring.records for cache in draft_cache))

        return DeepseekV4StreamingTargetFeatureStore(
            prompt_len=int(prompt_len),
            project_context=project_context,
            consume_prompt_chunk=consume_prompt_chunk,
        )

    def make_cache(
        self,
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        sink_size: int,
        window_size: int,
        allow_full_context_layers: bool = False,
    ) -> list[Any]:
        del sink_size, window_size
        if allow_full_context_layers:
            raise ValueError("DSpark stages use their fixed sliding attention window")
        caches = draft_model._make_cache()
        try:
            if len(caches) != 3:
                raise ValueError("DeepSeek V4 DFlash2 requires three DSpark caches")
            for cache in caches:
                ring = getattr(cache, "ring", None)
                if (
                    getattr(ring, "mode", None) != "nvfp4_stock432_fixed_ring"
                    or getattr(ring, "record_bytes", None) != 432
                    or len(ring) != 0
                ):
                    raise ValueError(
                        "DSpark DFlash2 caches must start empty in Mia stock432 format"
                    )
        except BaseException as primary_error:
            try:
                draft_model._release_cache(caches)
            except BaseException as release_error:
                primary_error.add_note(
                    "draft cache release also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )
            raise
        return caches

    @staticmethod
    def _append_context(
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        draft_context: DeepseekV4TargetTapRows,
    ) -> int:
        prior_length = int(draft_cache[0].prefill_length)
        context_rows = draft_context.rows
        if context_rows == 0:
            return prior_length

        tail_count = min(context_rows, int(draft_model.args.sliding_window))
        tail_offset = context_rows - tail_count
        tail_start = prior_length + tail_offset
        tail_context = draft_context.fuse_tail(tail_offset)
        main_hidden = draft_model.owner.stages[0]._run_fuse_main_rows(tail_context)
        if prior_length == 0:
            for stage, cache in zip(
                draft_model.owner.stages,
                draft_cache,
            ):
                records = stage.attn.project_context_records(
                    main_hidden, tail_start
                )
                cache._install_prefill_records(
                    records,
                    absolute_start=tail_start,
                    total_length=context_rows,
                )
        else:
            for stage, cache in zip(
                draft_model.owner.stages,
                draft_cache,
            ):
                records = stage.attn.project_context_records(
                    main_hidden, tail_start
                )
                cache._commit_records(tail_start, records)
        return prior_length + context_rows

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: DeepseekV4TargetTapRows,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        del target_ops, mask_token_tail, suppress_token_mask, async_launch
        requested_width = int(block_len)

        start_pos = self._append_context(
            draft_model=draft_model,
            draft_cache=draft_cache,
            draft_context=draft_context,
        )
        proposal = draft_model.owner.propose_k5(
            staged_first,
            target_model.model.embed_tokens,
            target_model.lm_head,
            draft_cache,
            start_pos=start_pos,
        )
        full_draft = proposal.future_tokens.squeeze(0).astype(mx.uint32)
        drafted = full_draft[: requested_width - 1]
        mx.async_eval(
            drafted,
            *(cache.ring.records for cache in draft_cache),
        )
        return drafted

    def draft_greedy_capture(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: DeepseekV4TargetTapRows,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        top_width: int,
    ) -> tuple[mx.array, None, None]:
        """Run the same DSpark proposal while DFlash captures target logits."""

        del top_width
        drafted = self.draft_greedy(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
            suppress_token_mask=suppress_token_mask,
            async_launch=async_launch,
        )
        return drafted, None, None

    def advance_context(
        self,
        *,
        draft_model: DeepseekV4DSparkDraftAdapter,
        draft_cache: list[Any],
        draft_context: DeepseekV4TargetTapRows,
    ) -> None:
        self._append_context(
            draft_model=draft_model,
            draft_cache=draft_cache,
            draft_context=draft_context,
        )


def _stream_dflash_generate(**kwargs):
    from dflash_mlx.runtime import stream_dflash_generate

    return stream_dflash_generate(**kwargs)


def generate_deepseek_v4_dflash2(
    bundle: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    runtime_context: Any,
    stop_token_ids: Optional[list[int]] = None,
    token_callback: Any = None,
    prefill_step_size: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
):
    """Translate the unchanged DFlash2 event stream into MTPLX output types."""

    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids
    from mtplx.generation import GenerationOutput, GenerationStats

    if not isinstance(runtime_context, DeepseekV4DFlashRuntimeContext):
        raise ValueError("DeepSeek V4 DFlash2 requires its installed runtime context")
    if runtime_context is not getattr(bundle, "runtime_context", None):
        raise ValueError("DeepSeek V4 DFlash2 runtime context is not owned by its bundle")
    capacity = int(bundle.target_model._mia_engine_plan.context_capacity_tokens)
    requested_span = len(prompt_ids) + int(max_tokens)
    if requested_span > capacity:
        raise ValueError(
            f"DeepSeek V4 DFlash2 request span {requested_span} exceeds "
            f"the installed {capacity}-token page plan"
        )
    resolved_stop_ids = (
        [int(value) for value in stop_token_ids]
        if stop_token_ids is not None
        else get_stop_token_ids(bundle.tokenizer)
    )
    stop_set = set(resolved_stop_ids)
    summary = None
    stop_seen = False
    for event in _stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=[int(value) for value in prompt_ids],
        prompt="",
        use_chat_template=False,
        max_new_tokens=int(max_tokens),
        block_tokens=_PHYSICAL_VERIFY_WIDTH,
        stop_token_ids=resolved_stop_ids,
        quantize_kv_cache=False,
        runtime_context=runtime_context,
        prefill_step_size=prefill_step_size,
        should_cancel=should_cancel,
    ):
        if isinstance(event, TokenEvent):
            token_id = int(event.token_id)
            if token_id in stop_set:
                stop_seen = True
            elif token_callback is not None and not stop_seen:
                token_callback([token_id])
        elif isinstance(event, SummaryEvent):
            if summary is not None:
                raise RuntimeError("DFlash2 emitted more than one summary")
            summary = event

    if summary is None:
        raise RuntimeError("DFlash2 stream ended without a summary")
    if summary.fallback_ar:
        raise RuntimeError(
            "DeepSeek V4 DFlash2 refused its installed lane: "
            f"{summary.fallback_reason or 'unspecified fallback'}"
        )
    if int(summary.block_tokens or 0) != _PHYSICAL_VERIFY_WIDTH:
        raise RuntimeError("DeepSeek V4 DFlash2 did not execute physical M6")

    physical_tokens = [int(value) for value in summary.generated_token_ids]
    first_stop = next(
        (index for index, token_id in enumerate(physical_tokens) if token_id in stop_set),
        None,
    )
    tokens = physical_tokens if first_stop is None else physical_tokens[:first_stop]
    elapsed_s = float(summary.elapsed_us) / 1_000_000.0
    prompt_s = float(summary.phase_timings_us.get("prefill", 0.0)) / 1_000_000.0
    decode_s = max(0.0, elapsed_s - prompt_s)
    cycles = int(summary.cycles_completed)
    acceptance_history = tuple(int(value) for value in summary.acceptance_history)
    accepted_by_depth = [
        sum(1 for accepted in acceptance_history if accepted >= depth)
        for depth in range(1, _PHYSICAL_VERIFY_WIDTH)
    ]
    accepted = int(summary.accepted_from_draft)
    drafted = cycles * (_PHYSICAL_VERIFY_WIDTH - 1)
    drafted_by_depth = [cycles] * (_PHYSICAL_VERIFY_WIDTH - 1)
    generated = len(tokens)
    stats = GenerationStats(
        mode="dspark",
        generated_tokens=generated,
        elapsed_s=elapsed_s,
        tok_s=(generated / elapsed_s if elapsed_s > 0 else 0.0),
        decode_elapsed_s=decode_s,
        decode_tok_s=(generated / decode_s if decode_s > 0 else 0.0),
        end_to_end_tok_s=(generated / elapsed_s if elapsed_s > 0 else 0.0),
        accepted_drafts=accepted,
        rejected_drafts=max(0, drafted - accepted),
        drafted_tokens=drafted,
        verify_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("verify", 0.0)
        )
        / 1_000_000.0,
        draft_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("draft", 0.0)
        )
        / 1_000_000.0,
        prompt_eval_time_s=prompt_s,
        prompt_tps=(
            int(summary.prompt_token_count) / prompt_s if prompt_s > 0 else 0.0
        ),
        rollback_time_s=float(
            (summary.cycle_profile_totals_us or {}).get("rollback", 0.0)
        )
        / 1_000_000.0,
        peak_memory_bytes=int(float(summary.peak_memory_gb or 0.0) * 1_000_000_000),
        speculative_depth=_PHYSICAL_VERIFY_WIDTH - 1,
        requested_speculative_depth=_PHYSICAL_VERIFY_WIDTH - 1,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        verify_calls=cycles,
        verify_hidden_mode="dflash2_deepseek_taps_40_41_42",
        events=[summary.to_payload()],
    )
    return GenerationOutput(
        tokens=tokens,
        text=bundle.tokenizer.decode(tokens),
        stats=stats,
        final_state=None,
        finish_reason="stop" if first_stop is not None else "length",
    )
