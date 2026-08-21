"""Draft-only LM-head helpers for MTPLX speculative proposals."""

from __future__ import annotations

import time
from typing import Any


def normalize_draft_lm_head_spec(
    value: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a validated draft LM-head spec from profile/contract metadata."""
    if value is None:
        return fallback
    if not isinstance(value, dict):
        raise ValueError("draft LM-head spec must be an object")
    if "bits" not in value:
        raise ValueError("draft LM-head spec missing bits")
    bits = int(value["bits"])
    group_size = int(value.get("group_size", 64))
    mode = str(value.get("mode", "affine"))
    if bits <= 0:
        raise ValueError("draft LM-head bits must be positive")
    if group_size <= 0:
        raise ValueError("draft LM-head group_size must be positive")
    if mode not in {"affine", "symmetric"}:
        raise ValueError("draft LM-head mode must be 'affine' or 'symmetric'")
    return {"bits": bits, "group_size": group_size, "mode": mode}


def draft_lm_head_spec_from_runtime_contract(
    contract_data: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a model-specific draft-head recommendation from contract data."""
    if not isinstance(contract_data, dict):
        return fallback
    return normalize_draft_lm_head_spec(
        contract_data.get("recommended_draft_lm_head"),
        fallback=fallback,
    )


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _make_requantized_head(module: Any, *, bits: int, group_size: int, mode: str) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    started = time.perf_counter()
    if (
        int(module.bits) == int(bits)
        and int(module.group_size) == int(group_size)
        and str(module.mode) == str(mode)
    ):
        report = {
            "original": {
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "mode": str(module.mode),
                "weight_shape": list(module.weight.shape),
                "scales_shape": list(module.scales.shape),
            },
            "draft_only": {
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "mode": str(module.mode),
                "weight_shape": list(module.weight.shape),
                "scales_shape": list(module.scales.shape),
            },
            "reused_existing_quantization": True,
            "elapsed_s": time.perf_counter() - started,
        }
        return module, report
    dense = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    ).astype(mx.bfloat16)
    mx.eval(dense)
    linear = nn.Linear(int(dense.shape[1]), int(dense.shape[0]), bias=("bias" in module))
    linear.weight = dense
    if "bias" in module:
        linear.bias = module.bias
    quantized = nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(quantized.weight, quantized.scales, quantized.biases)
    report = {
        "original": {
            "bits": int(module.bits),
            "group_size": int(module.group_size),
            "mode": str(module.mode),
            "weight_shape": list(module.weight.shape),
            "scales_shape": list(module.scales.shape),
        },
        "draft_only": {
            "bits": int(quantized.bits),
            "group_size": int(quantized.group_size),
            "mode": str(quantized.mode),
            "weight_shape": list(quantized.weight.shape),
            "scales_shape": list(quantized.scales.shape),
        },
        "elapsed_s": time.perf_counter() - started,
    }
    return quantized, report


def _quantize_linear_like_head(
    module: Any,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> tuple[Any, dict[str, Any]]:
    import mlx.nn as nn

    try:
        from .mtp_adapters import LoRALinear
    except Exception:  # pragma: no cover - defensive for minimal import contexts
        LoRALinear = None  # type: ignore[assignment]

    if LoRALinear is not None and isinstance(module, LoRALinear):
        draft_base, report = _quantize_linear_like_head(
            module.base,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
        module.base = draft_base
        report = {
            "wrapper": "LoRALinear",
            "base": report,
            "draft_only": report["draft_only"],
        }
        return module, report
    if isinstance(module, nn.QuantizedLinear):
        return _make_requantized_head(
            module,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
    if isinstance(module, nn.Linear):
        return _make_quantized_dense_head(
            module,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
    raise TypeError(f"head is not Linear/QuantizedLinear: {type(module)!r}")


def _make_quantized_dense_head(module: Any, *, bits: int, group_size: int, mode: str) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    started = time.perf_counter()
    dense = module.weight.astype(mx.bfloat16)
    mx.eval(dense)
    linear = nn.Linear(int(dense.shape[1]), int(dense.shape[0]), bias=("bias" in module))
    linear.weight = dense
    if "bias" in module:
        linear.bias = module.bias.astype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(quantized.weight, quantized.scales, quantized.biases)
    report = {
        "source": "dense_lm_head",
        "original": {
            "bits": "dense",
            "dtype": str(module.weight.dtype),
            "weight_shape": list(module.weight.shape),
            "scales_shape": None,
        },
        "draft_only": {
            "bits": int(quantized.bits),
            "group_size": int(quantized.group_size),
            "mode": str(quantized.mode),
            "weight_shape": list(quantized.weight.shape),
            "scales_shape": list(quantized.scales.shape),
        },
        "reused_existing_quantization": False,
        "elapsed_s": time.perf_counter() - started,
    }
    return quantized, report


def _embedding_report(module: Any) -> dict[str, Any]:
    return {
        "bits": int(getattr(module, "bits")),
        "group_size": int(getattr(module, "group_size")),
        "mode": str(getattr(module, "mode")),
        "weight_shape": list(module.weight.shape),
        "scales_shape": list(module.scales.shape),
    }


def _make_embedding_as_linear_head(
    module: Any,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    class _EmbeddingAsLinear(nn.Module):
        def __init__(self, embedding: Any):
            super().__init__()
            self.embedding = embedding

        def __call__(self, x):
            return self.embedding.as_linear(x)

    started = time.perf_counter()
    if isinstance(module, nn.QuantizedEmbedding):
        original = _embedding_report(module)
        if (
            int(module.bits) == int(bits)
            and int(module.group_size) == int(group_size)
            and str(module.mode) == str(mode)
        ):
            return _EmbeddingAsLinear(module), {
                "source": "tied_embedding",
                "original": original,
                "draft_only": original,
                "reused_existing_quantization": True,
                "elapsed_s": time.perf_counter() - started,
            }
        dense = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        ).astype(mx.bfloat16)
        mx.eval(dense)
    elif isinstance(module, nn.Embedding):
        dense = module.weight.astype(mx.bfloat16)
        original = {
            "bits": "bf16",
            "group_size": None,
            "mode": "none",
            "weight_shape": list(module.weight.shape),
            "scales_shape": None,
        }
    else:
        raise TypeError(f"embed_tokens is not Embedding/QuantizedEmbedding: {type(module)!r}")

    embedding = nn.Embedding(int(dense.shape[0]), int(dense.shape[1]))
    embedding.weight = dense
    quantized = nn.QuantizedEmbedding.from_embedding(
        embedding,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    mx.eval(quantized.weight, quantized.scales, quantized.biases)
    return _EmbeddingAsLinear(quantized), {
        "source": "tied_embedding",
        "original": original,
        "draft_only": _embedding_report(quantized),
        "reused_existing_quantization": False,
        "elapsed_s": time.perf_counter() - started,
    }


def _install_draft_lm_head(rt: Any, *, bits: int, group_size: int, mode: str) -> dict[str, Any]:
    import mlx.nn as nn

    text = _text_model(rt.model)
    mtp_layers = getattr(getattr(text, "mtp", None), "layers", None)
    step_shared_heads = [
        (idx, layer, getattr(layer, "shared_head_head", None))
        for idx, layer in enumerate(mtp_layers or [])
        if getattr(layer, "shared_head_head", None) is not None
    ]
    if step_shared_heads:
        started = time.perf_counter()
        reports: list[dict[str, Any]] = []
        for idx, layer, head in step_shared_heads:
            draft_head, report = _quantize_linear_like_head(
                head,
                bits=bits,
                group_size=group_size,
                mode=mode,
            )
            # opt-in reduced-vocab draft head, active only via MTPLX_DRAFT_VOCAB_IDS
            try:
                from .reduced_vocab_draft import (
                    maybe_wrap_draft_head as _mtplx_ext_wrap,
                )

                draft_head = _mtplx_ext_wrap(text, draft_head)
            except Exception:  # pragma: no cover - never fail the load
                pass
            layer.shared_head_head = draft_head
            reports.append({"layer": idx, **report})
        text._mtplx_step_mtp_draft_shared_heads = {
            "bits": int(bits),
            "group_size": int(group_size),
            "mode": str(mode),
            "layers": len(reports),
        }
        return {
            "source": "step_mtp_shared_head",
            "layers": reports,
            "elapsed_s": time.perf_counter() - started,
        }

    module = getattr(text, "lm_head", None)
    if module is not None:
        if isinstance(module, nn.QuantizedLinear):
            draft_head, report = _make_requantized_head(
                module,
                bits=bits,
                group_size=group_size,
                mode=mode,
            )
        elif isinstance(module, nn.Linear):
            draft_head, report = _make_quantized_dense_head(
                module,
                bits=bits,
                group_size=group_size,
                mode=mode,
            )
        else:
            raise TypeError(f"lm_head is not Linear/QuantizedLinear: {type(module)!r}")
    elif bool(getattr(getattr(text, "args", None), "tie_word_embeddings", False)):
        embed_tokens = getattr(getattr(text, "model", None), "embed_tokens", None)
        draft_head, report = _make_embedding_as_linear_head(
            embed_tokens,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )
    else:
        raise AttributeError("model has no lm_head and does not tie output projection to embeddings")
    try:
        from .reduced_vocab_draft import maybe_wrap_draft_head as _mtplx_ext_wrap

        _mtplx_ext_head = _mtplx_ext_wrap(text, draft_head)
    except Exception as _mtplx_ext_exc:  # pragma: no cover - never fail the load
        import sys as _mtplx_ext_sys

        print(
            f"[reduced-vocab] hook failed: {_mtplx_ext_exc!r}",
            file=_mtplx_ext_sys.stderr,
            flush=True,
        )
        _mtplx_ext_head = draft_head
    if _mtplx_ext_head is not draft_head:
        draft_head = _mtplx_ext_head
        if isinstance(report, dict):
            report = dict(report)
            report["reduced_vocab"] = dict(
                getattr(text, "_mtplx_reduced_vocab", {}) or {}
            )
    text._mtplx_draft_lm_head = draft_head
    return report
