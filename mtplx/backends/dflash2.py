"""First-class DFlash2 runtime adapter.

The DFlash algorithm remains in the official ``dflash`` package.  This module
only owns the MTPLX bundle contract, runtime wrapper, and telemetry mapping;
imports of MLX and DFlash are deliberately kept inside load/generation calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from mtplx.dflash2_bundle import (
    DFLASH2_ARCH_ID,
    DFLASH2_BACKEND,
    DFLASH2_DRAFT_LAYERS,
    dflash2_bundle_inspection,
    load_dflash2_metadata,
    resolve_dflash2_bundle_paths,
)

BACKEND_NAME = DFLASH2_BACKEND
ARCH_ID = DFLASH2_ARCH_ID
DEFAULT_DFLASH2_BLOCK_SIZE = DFLASH2_DRAFT_LAYERS
DEFAULT_DRAFT_BLOCK_SIZE = DEFAULT_DFLASH2_BLOCK_SIZE
DFlash2Quantization = Literal["4bit", "8bit", "unquantized"]
_DFLASH2_QUANTIZATIONS = {"4bit", "8bit", "unquantized"}
_DFLASH2_LOAD_DRAFT_LOCK = RLock()


class DFlash2Unsupported(RuntimeError):
    """Raised when an MTPLX feature is not supported by DFlash2."""


def _normalise_quantization(value: Any) -> DFlash2Quantization:
    if value is None:
        return "unquantized"
    if isinstance(value, int):
        value = f"{value}bit"
    text = str(value).strip().lower().replace("-", "").replace(" ", "")
    aliases = {
        "4": "4bit",
        "q4": "4bit",
        "4bit": "4bit",
        "8": "8bit",
        "q8": "8bit",
        "8bit": "8bit",
        "bf16": "unquantized",
        "fp16": "unquantized",
        "fp32": "unquantized",
        "none": "unquantized",
        "unquantized": "unquantized",
    }
    result = aliases.get(text)
    if result is None or result not in _DFLASH2_QUANTIZATIONS:
        raise ValueError(
            "DFlash2 draft_quantization must be one of '4bit', '8bit', or 'unquantized'"
        )
    return result  # type: ignore[return-value]


def _manifest_settings(metadata: dict[str, Any]) -> tuple[DFlash2Quantization, int]:
    draft = metadata.get("draft")
    draft = draft if isinstance(draft, dict) else {}
    quantization = metadata.get(
        "draft_quantization",
        metadata.get("draft_precision", draft.get("quantization", draft.get("precision"))),
    )
    raw_block_size = metadata.get(
        "draft_block_size",
        metadata.get("block_size", draft.get("draft_block_size", draft.get("block_size"))),
    )
    try:
        block_size = DEFAULT_DFLASH2_BLOCK_SIZE if raw_block_size is None else int(raw_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("DFlash2 block_size must be an integer") from exc
    if block_size < 1:
        raise ValueError("DFlash2 block_size must be positive")
    return _normalise_quantization(quantization), block_size


resolve_dflash2_paths = resolve_dflash2_bundle_paths
load_dflash2_manifest = load_dflash2_metadata


@dataclass(frozen=True)
class DFlash2RuntimeConfig:
    target_model_path: Path
    draft_model_path: Path
    draft_block_size: int = DEFAULT_DFLASH2_BLOCK_SIZE
    draft_quantization: DFlash2Quantization = "unquantized"
    prefill_step_size: int = 2048
    backend: str = "dflash2"

    @classmethod
    def from_paths(
        cls,
        *,
        target_model_path: str | Path,
        draft_model_path: str | Path,
        draft_block_size: int = DEFAULT_DFLASH2_BLOCK_SIZE,
        draft_quantization: Any = "unquantized",
        quantization: Any | None = None,
        prefill_step_size: int = 2048,
    ) -> DFlash2RuntimeConfig:
        return cls(
            target_model_path=Path(target_model_path),
            draft_model_path=Path(draft_model_path),
            draft_block_size=int(draft_block_size),
            draft_quantization=_normalise_quantization(
                draft_quantization if quantization is None else quantization
            ),
            prefill_step_size=int(prefill_step_size),
        )

    @property
    def quantization(self) -> DFlash2Quantization:
        return self.draft_quantization

    @property
    def backend_id(self) -> str:
        return self.backend

    @property
    def target_path(self) -> Path:
        return self.target_model_path

    @property
    def draft_path(self) -> Path:
        return self.draft_model_path

    def validate_static(self) -> None:
        if self.backend != "dflash2":
            raise ValueError("DFlash2 backend must be 'dflash2'")
        if self.draft_block_size < 1:
            raise ValueError("DFlash2 draft_block_size must be positive")
        _normalise_quantization(self.draft_quantization)
        if self.prefill_step_size < 1:
            raise ValueError("DFlash2 prefill_step_size must be positive")
        if not self.target_model_path.is_dir():
            raise FileNotFoundError(f"DFlash2 target path does not exist: {self.target_model_path}")
        if not self.draft_model_path.is_dir():
            raise FileNotFoundError(f"DFlash2 draft path does not exist: {self.draft_model_path}")


@dataclass
class DFlash2Telemetry:
    chunks: int = 0
    generated_tokens: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_bytes: int = 0
    finish_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "dflash2",
            "chunks": self.chunks,
            "generated_tokens": self.generated_tokens,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tps": self.prompt_tps,
            "generation_tps": self.generation_tps,
            "peak_memory_bytes": self.peak_memory_bytes,
            "finish_reason": self.finish_reason,
            "events": list(self.events),
        }


class DFlash2Runtime:
    """MTPLX wrapper around an MLX target and an official DFlash2 draft."""

    def __init__(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        draft_model: Any,
        config: DFlash2RuntimeConfig,
    ) -> None:
        self.model = target_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft = draft_model
        self.config = config
        self.model_path = config.target_model_path
        self.path = config.target_model_path
        self.bundle_path: Path | None = None
        self.backend_id = "dflash2"
        self.mtp_enabled = True
        self.dflash2_external_draft = True
        self.telemetry = DFlash2Telemetry()
        self.diagnostic_counters: dict[str, int] = {}

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = self.diagnostic_counters.get(key, 0) + int(amount)


def load_dflash2_bundle(
    bundle_root: str | Path,
    *,
    draft_block_size: int | None = None,
    draft_quantization: Any | None = None,
) -> DFlash2Runtime:
    """Load a local DFlash2 bundle using only the official package APIs."""

    root = Path(bundle_root).expanduser()
    resolved = resolve_dflash2_bundle_paths(root)
    if resolved is None:
        raise ValueError(f"not a DFlash2 bundle: {bundle_root}")
    inspection = dflash2_bundle_inspection(
        model_ref=str(root),
        bundle_root=root,
        paths=resolved,
    )
    compatibility = inspection.get("compatibility", {})
    if not isinstance(compatibility, dict) or not compatibility.get("can_run"):
        message = compatibility.get("message") if isinstance(compatibility, dict) else None
        raise ValueError(message or f"DFlash2 bundle rejected: {root}")
    metadata = resolved["metadata"]
    manifest_quantization, manifest_block_size = _manifest_settings(metadata)
    config = DFlash2RuntimeConfig.from_paths(
        target_model_path=resolved["target_model"],
        draft_model_path=resolved["draft_model"],
        draft_block_size=(manifest_block_size if draft_block_size is None else int(draft_block_size)),
        draft_quantization=(
            manifest_quantization if draft_quantization is None else draft_quantization
        ),
    )
    config.validate_static()
    # The official API is imported only after inspection and static validation.
    try:
        import dflash.model_mlx as dflash_model_mlx
    except ImportError as exc:
        raise RuntimeError(
            "DFlash2 requires the optional 'dflash2' dependency (dflash==0.1.0)"
        ) from exc
    target_model, tokenizer = dflash_model_mlx.load(str(config.target_model_path))
    draft_model = _load_dflash2_draft(
        dflash_model_mlx,
        dflash_model_mlx.load_draft,
        config.draft_model_path,
    )
    if config.draft_quantization in {"4bit", "8bit"}:
        try:
            import mlx.core as mx
            from mlx import nn
        except ImportError as exc:
            raise RuntimeError(
                "DFlash2 draft quantization requires the optional MLX dependency"
            ) from exc
        nn.quantize(
            draft_model,
            group_size=64,
            bits=4 if config.draft_quantization == "4bit" else 8,
        )
        mx.eval(draft_model.parameters())
    runtime = DFlash2Runtime(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        config=config,
    )
    runtime.bundle_path = root
    runtime.dflash2_metadata = metadata
    return runtime


load_dflash2 = load_dflash2_bundle


def _load_dflash2_draft(
    dflash_module: Any,
    draft_loader: Callable[[str], Any],
    draft_model_path: str | Path,
) -> Any:
    """Call the official draft loader without letting local paths hit HF."""

    local_path = Path(draft_model_path).expanduser()
    if not local_path.is_dir():
        return draft_loader(str(draft_model_path))
    local_directory = str(local_path.resolve())
    with _DFLASH2_LOAD_DRAFT_LOCK:
        original_snapshot_download = dflash_module.snapshot_download
        dflash_module.snapshot_download = lambda *_args, **_kwargs: local_directory
        try:
            return draft_loader(str(draft_model_path))
        finally:
            dflash_module.snapshot_download = original_snapshot_download


def _unsupported_options(
    *,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
) -> None:
    if constraint is not None:
        raise DFlash2Unsupported("constrained decoding is not supported on the dflash2 backend")
    if (
        session_bank is not None
        or capture_final_state
        or commit_prompt_state_to_bank
    ):
        raise DFlash2Unsupported("sessions and final-state capture are not supported on the dflash2 backend")


def _response_tokens(response: Any) -> list[int]:
    values = getattr(response, "tokens", None)
    if values is not None:
        return [int(value) for value in values]
    token = getattr(response, "token", None)
    return [] if token is None else [int(token)]


def _decode_tokens(tokenizer: Any, tokens: list[int], fallback: str) -> str:
    try:
        return str(tokenizer.decode(tokens))
    except (AttributeError, TypeError, ValueError):
        return fallback


def _default_stop_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError):
        return set()


def _response_drafted_count(
    response: Any,
    *,
    mode: Literal["ar", "mtpk"],
    block_size: int,
    max_tokens: int,
    previous_generated: int,
    chunk_length: int,
) -> int:
    if mode != "mtpk":
        return 0
    for name in ("drafted_tokens", "draft_tokens", "proposed_tokens", "proposal_tokens"):
        value = getattr(response, name, None)
        if isinstance(value, (list, tuple)):
            return len(value)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    accepted = getattr(response, "accepted", None)
    if accepted is None and chunk_length <= 1:
        return 0
    # DFlash proposes bs - 1 tokens and emits those accepted tokens plus one
    # target bonus.  generation_tokens is cumulative when supplied by DFlash;
    # the local count keeps test doubles and older releases compatible.
    generated_before = previous_generated
    generation_total = getattr(response, "generation_tokens", None)
    if generation_total is not None:
        try:
            generated_before = max(0, int(generation_total) - chunk_length)
        except (TypeError, ValueError):
            pass
    return max(0, min(block_size - 1, max_tokens - generated_before))


def _accepted_draft_count(response: Any, drafted_count: int) -> int:
    explicit = getattr(response, "accepted_drafts", None)
    if explicit is not None:
        try:
            return max(0, min(drafted_count, int(explicit)))
        except (TypeError, ValueError):
            pass
    accepted = getattr(response, "accepted", None)
    if accepted is None:
        return 0
    try:
        # Official DFlash's accepted count includes the target bonus token.
        return max(0, min(drafted_count, int(accepted) - 1))
    except (TypeError, ValueError):
        return 0


def _generate_stream(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    block_size: int | None,
    seed: int,
    stop_token_ids: set[int] | None,
    token_callback: Callable[[list[int]], None] | None,
    abort_check: Callable[[], bool] | None,
    mode: Literal["ar", "mtpk"],
) -> Any:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    started = time.perf_counter()
    telemetry = DFlash2Telemetry(prompt_tokens=len(prompt_ids))
    generated: list[int] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    prompt_tps = 0.0
    accepted = 0
    drafted = 0
    previous_generated = 0
    peak_memory = 0.0
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("DFlash2 requires the optional MLX dependency") from exc
    mx.random.seed(int(seed))
    effective_stop_ids = (
        _default_stop_token_ids(runtime.tokenizer)
        if stop_token_ids is None
        else {int(value) for value in stop_token_ids}
    )

    if mode == "mtpk":
        try:
            from dflash.model_mlx import stream_generate
        except ImportError as exc:
            raise RuntimeError(
                "DFlash2 requires the optional 'dflash2' dependency (dflash==0.1.0)"
            ) from exc
        stream = stream_generate(
            runtime.model,
            runtime.draft_model,
            runtime.tokenizer,
            prompt_ids,
            block_size=runtime.config.draft_block_size,
            max_tokens=max_tokens,
            temperature=float(sampler.temperature),
            top_p=float(sampler.top_p),
            top_k=int(sampler.top_k),
            prefill_step_size=runtime.config.prefill_step_size,
        )
    else:
        try:
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("DFlash2 target AR requires mlx-lm") from exc
        stream = stream_generate(
            runtime.model,
            runtime.tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=make_sampler(
                temp=float(sampler.temperature),
                top_p=float(sampler.top_p),
                top_k=int(sampler.top_k),
            ),
            prefill_step_size=runtime.config.prefill_step_size,
        )

    for response in stream:
        if abort_check is not None and abort_check():
            finish_reason = "cancelled"
            break
        chunk = _response_tokens(response)
        raw_chunk_length = len(chunk)
        if not chunk:
            finish_reason = getattr(response, "finish_reason", None) or finish_reason
            continue
        remaining = max_tokens - len(generated)
        chunk = chunk[:remaining]
        stop_index = next(
            (index for index, token in enumerate(chunk) if token in effective_stop_ids),
            None,
        )
        if stop_index is not None:
            chunk = chunk[: stop_index + 1]
            finish_reason = "stop"
        if not chunk:
            break
        drafted_count = _response_drafted_count(
            response,
            mode=mode,
            block_size=runtime.config.draft_block_size,
            max_tokens=max_tokens,
            previous_generated=previous_generated,
            chunk_length=raw_chunk_length,
        )
        accepted_count = _accepted_draft_count(response, drafted_count)
        generated.extend(chunk)
        text_parts.append(str(getattr(response, "text", "") or ""))
        telemetry.chunks += 1
        accepted += accepted_count
        drafted += drafted_count
        prompt_tps = float(getattr(response, "prompt_tps", 0.0) or 0.0)
        response_tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
        peak_memory = float(getattr(response, "peak_memory", 0.0) or 0.0)
        telemetry.events.append(
            {
                "tokens": len(chunk),
                "drafted": drafted_count,
                "drafted_positions": list(
                    range(
                        len(prompt_ids) + previous_generated,
                        len(prompt_ids) + previous_generated + drafted_count,
                    )
                ),
                "accepted": accepted_count,
                "generation_tps": response_tps,
            }
        )
        if token_callback is not None:
            callback_chunk = [token for token in chunk if token not in effective_stop_ids]
            if callback_chunk:
                token_callback(callback_chunk)
        previous_generated += len(chunk)
        if finish_reason is None:
            finish_reason = getattr(response, "finish_reason", None) or finish_reason
        if finish_reason is not None or len(generated) >= max_tokens:
            break

    elapsed = max(time.perf_counter() - started, 1e-12)
    telemetry.generated_tokens = len(generated)
    telemetry.drafted_tokens = drafted
    telemetry.accepted_tokens = accepted
    telemetry.prompt_tps = prompt_tps
    telemetry.generation_tps = (
        len(generated) / elapsed if len(generated) else 0.0
    )
    telemetry.peak_memory_bytes = int(peak_memory * 1_000_000_000)
    telemetry.finish_reason = finish_reason or ("length" if len(generated) >= max_tokens else "stop")
    runtime.telemetry = telemetry
    runtime._count("dflash2_chunks", telemetry.chunks)
    runtime._count("dflash2_generated_tokens", telemetry.generated_tokens)

    from mtplx.generation import GenerationOutput, GenerationStats

    prompt_eval_time = len(prompt_ids) / prompt_tps if prompt_tps > 0 else 0.0
    stats = GenerationStats(
        mode=mode,
        generated_tokens=len(generated),
        elapsed_s=elapsed,
        tok_s=len(generated) / elapsed,
        decode_elapsed_s=max(0.0, elapsed - prompt_eval_time),
        decode_tok_s=(
            len(generated) / max(1e-12, elapsed - prompt_eval_time)
            if elapsed > prompt_eval_time
            else 0.0
        ),
        end_to_end_tok_s=len(generated) / elapsed,
        runtime_mtp_enabled=mode == "mtpk",
        prompt_eval_time_s=prompt_eval_time,
        prompt_tps=prompt_tps,
        accepted_drafts=accepted,
        drafted_tokens=drafted,
        draft_time_s=max(0.0, elapsed - prompt_eval_time),
        verify_calls=telemetry.chunks if mode == "mtpk" else 0,
        peak_memory_bytes=telemetry.peak_memory_bytes,
        draft_core={"backend": "dflash2", **telemetry.to_dict()},
        events=list(telemetry.events),
    )
    return GenerationOutput(
        tokens=generated,
        text=_decode_tokens(
            runtime.tokenizer,
            generated[:-1] if generated and generated[-1] in effective_stop_ids else generated,
            "".join(text_parts),
        ),
        stats=stats,
        finish_reason=telemetry.finish_reason,
    )


def generate_dflash2(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    abort_check: Callable[[], bool] | None = None,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
    speculative_depth: int | None = None,
) -> Any:
    _unsupported_options(
        constraint=constraint,
        session_bank=session_bank,
        session_id=session_id,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        capture_final_state=capture_final_state,
        session_restore_mode=session_restore_mode,
        commit_prompt_state_to_bank=commit_prompt_state_to_bank,
    )
    return _generate_stream(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        block_size=None,
        seed=seed,
        stop_token_ids=stop_token_ids,
        token_callback=token_callback,
        abort_check=abort_check,
        mode="mtpk",
    )


def generate_dflash2_ar(
    runtime: DFlash2Runtime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    abort_check: Callable[[], bool] | None = None,
    constraint: Any = None,
    session_bank: Any = None,
    session_id: str | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    session_restore_mode: str = "clone",
    commit_prompt_state_to_bank: bool = False,
    **_: Any,
) -> Any:
    _unsupported_options(
        constraint=constraint,
        session_bank=session_bank,
        session_id=session_id,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        capture_final_state=capture_final_state,
        session_restore_mode=session_restore_mode,
        commit_prompt_state_to_bank=commit_prompt_state_to_bank,
    )
    return _generate_stream(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        block_size=None,
        seed=seed,
        stop_token_ids=stop_token_ids,
        token_callback=token_callback,
        abort_check=abort_check,
        mode="ar",
    )
