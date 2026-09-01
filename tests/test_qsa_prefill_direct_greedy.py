"""Greedy-token fixture for the direct Steel QSA prefill lane.

Production geometry (24q / 2kv / D256, indexer budget 2048 / ratio 4) so
the native kernel's support check can actually fire. One Attention layer +
a tiny embed/lm_head, not the 125B checkpoint — that would dual-load next
to live :8002.

Protocol: warm 2049 tokens (below the sparse producer gate), then a 32-token
prefill chunk whose earliest query is past the dense/sparse boundary, then
8 greedy decode steps. Direct vs gather vs dense must emit the same token
ids. Direct must show ``direct_kernel > 0`` and no gather/dense fallback.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map

import mtplx.kernels.qsa_prefill_direct as direct
import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.attention_context import attention_phase
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

pytestmark = pytest.mark.skipif(
    direct._EXT is None
    or not hasattr(direct._EXT, "qwen4_qsa_sparse_gqa_attention"),
    reason="no built mtplx_qsa_kernels extension on this machine",
)

_HIDDEN = 2560
_VOCAB = 512
_WARM = 2049
_SPARSE = 32
_GREEDY = 8
_SCALE_SEED = 7


def _prod_args() -> TextArgs:
    return TextArgs(
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        vocab_size=_VOCAB,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )


@pytest.fixture(scope="module")
def native_lane():
    assert direct.qsa_prefill_direct_preflight() is True
    return True


def _arm_env(monkeypatch, *, direct_on: bool, gather_on: bool) -> None:
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "8")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT", "999999")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_DIRECT", "1" if direct_on else "0")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_GATHER", "1" if gather_on else "0")
    monkeypatch.delenv("MTPLX_QSA_FLASH", raising=False)


def _run_arm(layer, embed, lm_head, prompt, *, phase_prefill="prefill"):
    qwen4_exp._QSA_PREFILL_COUNTS.clear()
    cache = QSACache(compress_ratio=layer.indexer.ratio)
    warm, sparse = prompt[:, :_WARM], prompt[:, _WARM:]
    with attention_phase(phase_prefill):
        out = layer(embed(warm), cache)
        mx.eval(out)
        out = layer(embed(sparse), cache)
        mx.eval(out)
    tokens = []
    last = int(mx.argmax(lm_head(out[:, -1, :]), axis=-1).item())
    tokens.append(last)
    with attention_phase("ar_decode"):
        for _ in range(_GREEDY - 1):
            step = mx.array([[last]], dtype=mx.int32)
            hid = layer(embed(step), cache)
            mx.eval(hid)
            last = int(mx.argmax(lm_head(hid[:, -1, :]), axis=-1).item())
            tokens.append(last)
    return tokens, dict(qwen4_exp.qsa_prefill_engagement())


def test_greedy_tokens_match_across_direct_gather_dense(native_lane, monkeypatch):
    mx.set_default_device(mx.gpu)
    mx.random.seed(_SCALE_SEED)
    args = _prod_args()
    layer = Attention(args)
    embed = nn.Embedding(_VOCAB, _HIDDEN)
    lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def _bf16(module):
        module.update(tree_map(lambda p: p.astype(mx.bfloat16), module.parameters()))

    _bf16(layer)
    _bf16(embed)
    _bf16(lm_head)
    mx.eval(layer.parameters(), embed.parameters(), lm_head.parameters())
    assert layer.q_proj.weight.dtype == mx.bfloat16

    mx.random.seed(_SCALE_SEED + 1)
    prompt = mx.random.randint(0, _VOCAB, (1, _WARM + _SPARSE))

    _arm_env(monkeypatch, direct_on=True, gather_on=False)
    direct_toks, direct_eng = _run_arm(layer, embed, lm_head, prompt)

    _arm_env(monkeypatch, direct_on=False, gather_on=True)
    gather_toks, gather_eng = _run_arm(layer, embed, lm_head, prompt)

    _arm_env(monkeypatch, direct_on=False, gather_on=False)
    dense_toks, dense_eng = _run_arm(layer, embed, lm_head, prompt)

    print("direct_tokens", direct_toks, "eng", direct_eng)
    print("gather_tokens", gather_toks, "eng", gather_eng)
    print("dense_tokens", dense_toks, "eng", dense_eng)

    assert direct_eng.get("direct_kernel", 0) >= 1, direct_eng
    assert direct_eng.get("gather_tier", 0) == 0, direct_eng
    assert direct_eng.get("dense_fallback", 0) == 0, direct_eng
    assert gather_eng.get("direct_kernel", 0) == 0, gather_eng
    assert gather_eng.get("gather_tier", 0) >= 1, gather_eng
    assert dense_eng.get("direct_kernel", 0) == 0, dense_eng
    assert dense_eng.get("gather_tier", 0) == 0, dense_eng
    assert dense_eng.get("dense_fallback", 0) >= 1, dense_eng

    assert direct_toks == gather_toks == dense_toks, (
        direct_toks,
        gather_toks,
        dense_toks,
        direct_eng,
        gather_eng,
        dense_eng,
    )
