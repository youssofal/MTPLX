"""Draft-only LM-head helpers for MTPLX speculative proposals."""

from __future__ import annotations

import os
import sys
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
    quantized = _requantize_in_row_chunks(
        module,
        bits=bits,
        group_size=group_size,
        mode=mode,
    )
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


#: Rows per requantization chunk. Affine/symmetric quantization groups run
#: along the input axis, so every output row quantizes independently and a
#: row-chunked pass is bit-identical to the whole-tensor pass; it just never
#: holds the dense bf16 head in memory (2.4 GiB for a 248k x 5120 head).
_REQUANTIZE_ROWS_PER_CHUNK = 8192


class ZeroedDraftHeadError(RuntimeError):
    """The requantized draft head read back as all zeros (a silent Metal OOM)."""


def _quantized_linear_from_parts(
    *,
    weight: Any,
    scales: Any,
    biases: Any | None,
    bias: Any | None,
    input_dims: int,
    output_dims: int,
    group_size: int,
    bits: int,
    mode: str,
) -> Any:
    """Build an ``nn.QuantizedLinear`` around already-quantized parts.

    ``nn.QuantizedLinear(...)`` allocates and quantizes a random dense weight
    of the full size before anything can be assigned, which is exactly the
    allocation this module must avoid on tight machines; ``from_linear``
    needs the dense tensor. So the module is assembled directly.
    """

    import mlx.nn as nn

    module = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(module)
    module.group_size = int(group_size)
    module.bits = int(bits)
    module.mode = str(mode)
    module.weight = weight
    module.scales = scales
    if biases is not None:
        module.biases = biases
    if bias is not None:
        module.bias = bias
    module.freeze()
    return module


def _requantize_in_row_chunks(
    module: Any,
    *,
    bits: int,
    group_size: int,
    mode: str,
    rows_per_chunk: int = _REQUANTIZE_ROWS_PER_CHUNK,
) -> Any:
    """Requantize a ``QuantizedLinear`` head to ``bits``/``group_size`` in row chunks.

    Issue #483 (M1 Pro, 32 GB): the whole-tensor path dequantized the 8-bit
    head to a 2.4 GiB dense bf16 intermediate on top of the resident model,
    the 4-bit quantize of that intermediate hit a Metal OOM that surfaced
    only in the command-buffer callback, and the helper returned a head
    whose weight, scales and biases all read back as zero. Draft acceptance
    fell to under 1% and MTP ran at 4-6 tok/s while the target was healthy.
    Each chunk here dequantizes and requantizes ``rows_per_chunk`` output
    rows (about 80 MB of bf16 at the 27B head shape), the parts are
    concatenated, and the result is bit-identical to the whole-tensor pass.
    A head that still reads back as zero raises :class:`ZeroedDraftHeadError`
    so the caller can fall back instead of serving a dead drafter.
    """

    import mlx.core as mx

    total_rows = int(module.weight.shape[0])
    weights: list[Any] = []
    scales: list[Any] = []
    biases: list[Any] = []
    has_biases = "biases" in module and module.biases is not None
    for start in range(0, total_rows, int(rows_per_chunk)):
        stop = min(total_rows, start + int(rows_per_chunk))
        dense_chunk = mx.dequantize(
            module.weight[start:stop],
            module.scales[start:stop],
            module.biases[start:stop] if has_biases else None,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        ).astype(mx.bfloat16)
        parts = mx.quantize(dense_chunk, group_size=group_size, bits=bits, mode=mode)
        weights.append(parts[0])
        scales.append(parts[1])
        if len(parts) > 2 and parts[2] is not None:
            biases.append(parts[2])
        mx.eval(*parts)
        del dense_chunk
    weight = mx.concatenate(weights, axis=0) if len(weights) > 1 else weights[0]
    scale = mx.concatenate(scales, axis=0) if len(scales) > 1 else scales[0]
    bias_parts = (
        (mx.concatenate(biases, axis=0) if len(biases) > 1 else biases[0])
        if biases
        else None
    )
    mx.eval(weight, scale, *( [bias_parts] if bias_parts is not None else [] ))
    # A silent Metal OOM leaves every part zeroed; a real head never has
    # all-zero scales. Check the scales (small) rather than the weight.
    if not bool(mx.any(scale != 0).item()):
        raise ZeroedDraftHeadError(
            "requantized draft LM head read back as all zeros "
            f"({total_rows} rows, {bits}-bit/g{group_size}); the device ran out "
            "of memory while building it"
        )
    input_dims = int(module.scales.shape[1]) * int(module.group_size)
    return _quantized_linear_from_parts(
        weight=weight,
        scales=scale,
        biases=bias_parts,
        bias=module.bias if "bias" in module else None,
        input_dims=input_dims,
        output_dims=total_rows,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


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


def _is_rotated_packed_head(module: Any) -> bool:
    try:
        from .models.prism_hadamard_qwen35 import HadamardQuantizedLinear
    except Exception:  # pragma: no cover - the model module needs mlx-lm
        return False
    return isinstance(module, HadamardQuantizedLinear)


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
        if _is_rotated_packed_head(module):
            # Prism ML's rotated ternary head (Bonsai) is already 2.25 bits
            # per weight: a 4-bit draft copy would read twice the bytes per
            # draft position, and a copy built from the raw packed rows
            # without the activation transform would draft from garbage. The
            # drafter shares the target head, which is exact (same law,
            # verified by the target) and is the smallest head available.
            draft_head = module
            packed = {
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "mode": str(module.mode),
                "weight_shape": list(module.weight.shape),
                "scales_shape": list(module.scales.shape),
            }
            report = {
                "source": "rotated_packed_lm_head",
                "original": dict(packed),
                "draft_only": dict(packed),
                "reused_existing_quantization": True,
                "requested": {"bits": int(bits), "group_size": int(group_size), "mode": str(mode)},
            }
        elif isinstance(module, nn.QuantizedLinear):
            try:
                draft_head, report = _make_requantized_head(
                    module,
                    bits=bits,
                    group_size=group_size,
                    mode=mode,
                )
            except (ZeroedDraftHeadError, RuntimeError) as exc:
                # Fail closed to the head that is already resident: a
                # drafter that shares the target's quantized head is exact
                # (same law, verified by the target) and merely reads more
                # bytes per draft position. Never serve a zeroed head
                # (issue #483: 0.4% acceptance, 4 tok/s on a 32 GB M1 Pro).
                print(
                    "[draft-lm-head] requantization failed "
                    f"({type(exc).__name__}: {exc}); reusing the target "
                    f"{int(module.bits)}-bit/g{int(module.group_size)} head as "
                    "the draft head",
                    file=sys.stderr,
                    flush=True,
                )
                draft_head = module
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
                    "requantization_failed": f"{type(exc).__name__}: {exc}",
                }
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
    text._mtplx_draft_lm_head = draft_head
    from .frspec_draft import frspec_enabled, install_frspec_draft_head

    if frspec_enabled():
        report = dict(report)
        report["frspec"] = install_frspec_draft_head(
            text, model_path=getattr(rt, "model_path", None)
        )
        if not report["frspec"].get("installed"):
            raise RuntimeError(
                "FR-Spec draft head installation failed: "
                f"{report['frspec'].get('reason', 'unknown reason')}"
            )
        print(f"[frspec] install report: {report['frspec']}", file=sys.stderr, flush=True)
    else:
        print(
            "[frspec] disabled (MTPLX_FRSPEC_DRAFT="
            f"{os.environ.get('MTPLX_FRSPEC_DRAFT')!r})",
            file=sys.stderr,
            flush=True,
        )
    return report
