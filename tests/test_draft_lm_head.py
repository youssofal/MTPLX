from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mtplx.draft_lm_head import (
    _install_draft_lm_head,
    _make_compact_quantized_draft_head,
)
from mtplx.mtp_adapters import LoRALinear


class _TiedEmbeddingText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding = nn.Embedding(32, 32)
        embedding.weight = mx.ones((32, 32), dtype=mx.bfloat16)
        self.model = SimpleNamespace(
            embed_tokens=nn.QuantizedEmbedding.from_embedding(
                embedding,
                group_size=32,
                bits=4,
                mode="affine",
            )
        )
        self.args = SimpleNamespace(tie_word_embeddings=True)


class _StepMTPText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_head = nn.Linear(32, 64, bias=False)
        shared_head.weight = mx.ones((64, 32), dtype=mx.bfloat16)
        self.mtp = SimpleNamespace(
            layers=[
                SimpleNamespace(shared_head_head=shared_head),
            ]
        )
        self.lm_head = nn.Linear(32, 64, bias=False)


def test_compact_quantized_draft_head_preserves_selected_rows_and_id_map() -> None:
    dense = mx.arange(16 * 32, dtype=mx.float32).reshape(16, 32).astype(mx.bfloat16)
    linear = nn.Linear(32, 16, bias=False)
    linear.weight = dense
    full = nn.QuantizedLinear.from_linear(
        linear,
        group_size=32,
        bits=4,
        mode="affine",
    )

    compact, token_map, report = _make_compact_quantized_draft_head(
        full,
        prefix_count=8,
        control_start=12,
        control_end=14,
        padded_count=16,
    )
    x = mx.ones((1, 1, 32), dtype=mx.bfloat16)
    expected = full(x)[..., [*range(8), 12, 13]]
    actual = compact(x)
    mx.eval(expected, actual, token_map)

    assert mx.array_equal(actual, expected).item()
    assert token_map.tolist() == [*range(8), 12, 13]
    assert report["real_rows"] == 10
    assert report["padded_rows"] == 16


def test_install_draft_lm_head_supports_tied_quantized_embeddings() -> None:
    rt = SimpleNamespace(model=_TiedEmbeddingText())

    report = _install_draft_lm_head(rt, bits=4, group_size=32, mode="affine")
    head = rt.model._mtplx_draft_lm_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "tied_embedding"
    assert report["reused_existing_quantization"] is True
    assert logits.shape == (1, 2, 32)


def test_install_draft_lm_head_quantizes_dense_linear_head() -> None:
    linear = nn.Linear(32, 64, bias=False)
    linear.weight = mx.ones((64, 32), dtype=mx.bfloat16)
    rt = SimpleNamespace(model=SimpleNamespace(lm_head=linear))

    report = _install_draft_lm_head(rt, bits=4, group_size=32, mode="affine")
    head = rt.model._mtplx_draft_lm_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "dense_lm_head"
    assert report["reused_existing_quantization"] is False
    assert logits.shape == (1, 2, 64)


def test_draft_head_install_has_no_pending_qwen38_source_route() -> None:
    import mtplx.draft_lm_head as draft_lm_head

    source = Path(draft_lm_head.__file__).read_text(encoding="utf-8")
    assert "_finish_pending_qwen38_source_route" not in source
    assert "_mtplx_qwen38_pending_source_route" not in source


def test_install_draft_lm_head_quantizes_step_mtp_shared_heads() -> None:
    rt = SimpleNamespace(model=_StepMTPText())

    report = _install_draft_lm_head(rt, bits=3, group_size=32, mode="affine")
    head = rt.model.mtp.layers[0].shared_head_head
    logits = head(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "step_mtp_shared_head"
    assert report["layers"][0]["layer"] == 0
    assert report["layers"][0]["draft_only"]["bits"] == 3
    assert head.bits == 3
    assert rt.model.lm_head.__class__ is nn.Linear
    assert logits.shape == (1, 2, 64)


def test_install_draft_lm_head_quantizes_lora_wrapped_step_shared_heads() -> None:
    rt = SimpleNamespace(model=_StepMTPText())
    original = rt.model.mtp.layers[0].shared_head_head
    rt.model.mtp.layers[0].shared_head_head = LoRALinear(original, rank=2)

    report = _install_draft_lm_head(rt, bits=3, group_size=32, mode="affine")
    wrapped = rt.model.mtp.layers[0].shared_head_head
    logits = wrapped(mx.ones((1, 2, 32), dtype=mx.bfloat16))
    mx.eval(logits)

    assert report["source"] == "step_mtp_shared_head"
    assert report["layers"][0]["wrapper"] == "LoRALinear"
    assert report["layers"][0]["draft_only"]["bits"] == 3
    assert isinstance(wrapped, LoRALinear)
    assert wrapped.base.bits == 3
    assert logits.shape == (1, 2, 64)
