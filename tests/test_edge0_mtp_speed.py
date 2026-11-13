"""Edge0-120B MTP speed lanes that live outside the latent block.

FR-Spec (``mtplx.frspec_draft``): Edge0 ships the Flash-Next MTP head's shared
``lm_head`` as Q8_0 (8-bit, group 32), and its engine drafts from a restricted
vocabulary (``mtp/draft_vocab.bin``). The pruned draft head accepts group 32
for the Edge0 model type only; every other model keeps the 8-bit/group-64
native contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mtplx import frspec_draft

VOCAB, HIDDEN = 512, 128


class _Text:
    def __init__(self, model_type: str, bits: int, group_size: int):
        self.args = SimpleNamespace(model_type=model_type)
        lin = nn.Linear(HIDDEN, VOCAB, bias=False)
        lin.weight = mx.array(np.random.default_rng(0).standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.1)
        self.head = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
        mx.eval(self.head.parameters())
        self.bound = None

    def _mtplx_native_mtp_draft_head(self):
        return self.head

    def _mtplx_bind_draft_lm_head(self, head):
        self.bound = head


@pytest.fixture()
def vocab(tmp_path, monkeypatch):
    ids = sorted(np.random.default_rng(1).choice(VOCAB, 200, replace=False).tolist())
    path = tmp_path / "draft_vocab.json"
    path.write_text(json.dumps({"ids": ids}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(path))
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    monkeypatch.delenv("MTPLX_FRSPEC_LEGACY", raising=False)
    return ids


@pytest.mark.parametrize("model_type", ["edge0_latent_text", "edge0_latent"])
def test_edge0_q8_group32_head_prunes(vocab, model_type):
    text = _Text(model_type, bits=8, group_size=32)
    report = frspec_draft.install_frspec_draft_head(text)
    assert report["installed"] and report["group_size"] == 32 and report["n"] == len(vocab)
    assert text.bound is text._mtplx_frspec_draft_head
    x = mx.array(np.random.default_rng(2).standard_normal((1, 3, HIDDEN)).astype(np.float32))
    full = np.array(text.head(x))
    got = np.array(text.bound(x))
    assert got.shape == full.shape
    np.testing.assert_array_equal(got[..., vocab], full[..., vocab])  # the kept rows: the same matmul rows
    others = np.setdiff1d(np.arange(VOCAB), vocab)
    assert (got[..., others] <= -1e29).all()  # pruned rows can never be drafted


def test_other_models_keep_the_group64_contract(vocab):
    text = _Text("qwen4_exp_text", bits=8, group_size=32)
    report = frspec_draft.install_frspec_draft_head(text)
    assert not report["installed"] and report["reason"] == "native_head_contract"
    assert text.bound is None


def test_edge0_still_requires_8_bit(vocab):
    text = _Text("edge0_latent_text", bits=4, group_size=32)
    report = frspec_draft.install_frspec_draft_head(text)
    assert not report["installed"] and report["reason"] == "native_head_contract"
