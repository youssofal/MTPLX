"""A segmented conversation past the dense-decode ceiling stays on segments (MTPLX_SEGMENTED_KV).

2026-10-10, Qwen3.8-27B on a 64 GB Mac: the auto ceiling is 157,286 tokens. A follow-up turn of
161,835 tokens (161,751 restored, 84 new) was laid out ``contiguous_then_repage``; the repage
builds paged caches from each layer's ``keys``, which a segment cache answers by gathering every
segment, and the admission priced that copy at 11.6 GiB and refused the turn with 10.1 GiB free.
"""

from __future__ import annotations

import pytest

import mtplx.segmented_kv as segmented_kv_module
from mtplx.generation import (
    _maybe_repage_target_prefill_cache,
    _sustained_prefill_layout,
    prefill_cache_layout,
)
from mtplx.segmented_kv import SegmentedKVCache

GIB = 1024**3
CEILING = 157_286
PROMPT, RESTORED = 161_835, 161_751
# Qwen3.8-27B: 16 full-attention layers x K+V x 4 KV heads x 256 x bf16, plus 4 KiB of MTP history.
ROW = 65_536 + 4_096
OUTPUT_ROWS = 16_384 + 3


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", str(CEILING))
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    # segmented_kv_enabled() needs a verdict on the loaded model (without one it stays off).
    monkeypatch.setattr(segmented_kv_module, "_MODEL_SUPPORT", {"supported": True, "reasons": []})


def _switch_off(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_SEGMENTED_KV", raising=False)


def test_segments_keep_the_dense_layout_past_the_ceiling(monkeypatch) -> None:
    assert _sustained_prefill_layout(CEILING) == "contiguous_dense_decode"
    assert _sustained_prefill_layout(PROMPT) == "contiguous_dense_decode"
    assert prefill_cache_layout(None, PROMPT) == "contiguous_dense_decode"
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(PROMPT))
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_the_stock_cache_still_repages_past_the_ceiling(monkeypatch) -> None:
    _switch_off(monkeypatch)
    assert _sustained_prefill_layout(CEILING) == "contiguous_dense_decode"
    assert _sustained_prefill_layout(PROMPT) == "contiguous_then_repage"
    monkeypatch.setenv("MTPLX_SEGMENTED_KV", "1")
    monkeypatch.setattr(segmented_kv_module, "_MODEL_SUPPORT", None)  # no verdict: stock
    assert _sustained_prefill_layout(PROMPT) == "contiguous_then_repage"


def test_quantized_kv_keeps_its_repage_with_segments(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    assert _sustained_prefill_layout(PROMPT) == "contiguous_then_repage"


def test_an_explicit_layout_is_left_alone(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    assert _sustained_prefill_layout(PROMPT) == "contiguous_then_repage"


class _Runtime:
    def __init__(self) -> None:
        self.repages = 0

    def repage_target_prefill_cache(self, cache) -> bool:
        self.repages += 1
        return True


def test_a_segmented_prefill_past_the_ceiling_is_not_repaged(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(PROMPT))
    runtime = _Runtime()
    cache = [SegmentedKVCache()]
    assert _maybe_repage_target_prefill_cache(runtime, cache) == 0.0
    assert runtime.repages == 0
    assert isinstance(cache[0], SegmentedKVCache)


def _incident_growth(layout: str) -> dict:
    from mtplx.server.openai import _admission_growth, _AdmissionGeometry

    geometry = _AdmissionGeometry(
        live_bytes_per_token=ROW,
        paged_bytes_per_token=ROW,
        context_transient_bytes_per_token=0,
        flat_transient_bytes=0,
        weights_bytes=0,
        aux_bytes_per_token=4_096,
    )
    return _admission_growth(
        geometry,
        prompt_tokens=PROMPT,
        reused_tokens=RESTORED,
        restore_copies_prefix=False,
        layout=layout,
        source_layout=None,
        output_tokens=OUTPUT_ROWS,
        publish=True,
        scratch_bytes=GIB // 2,
        lease={"paged": False},  # the live segmented cache the lease extends
        segmented_kv=True,
    )


def test_the_incident_turn_is_priced_without_a_history_copy() -> None:
    model = _incident_growth(prefill_cache_layout(None, PROMPT))
    assert model["repage_copy_bytes"] == 0
    assert model["restore_copy_bytes"] == 0
    # The new rows, the answer's rows the snapshot seals, and the prefill scratch: about 1 GiB.
    assert model["publish_copy_bytes"] == (PROMPT - RESTORED + OUTPUT_ROWS) * ROW
    assert model["growth_bytes"] < 2 * GIB
    # The repage layout the turn got: a paged copy of the whole history.
    repaged = _incident_growth("contiguous_then_repage")
    assert repaged["repage_copy_bytes"] == (PROMPT + OUTPUT_ROWS) * ROW
    assert repaged["growth_bytes"] > 11 * GIB
