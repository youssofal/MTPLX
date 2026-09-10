from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.draft_lm_head import _install_draft_lm_head
from mtplx.frspec_draft import install_frspec_draft_head, load_frspec_ids
from mtplx.models.qwen4_exp import TextModel


def test_qwen4_native_mtp_forward_uses_bound_draft_head() -> None:
    stock_calls: list[object] = []
    draft_calls: list[object] = []

    class FakeMTP:
        def fuse_and_run(self, hidden, embedding, cache):
            return (hidden, embedding, cache)

        def hyper_connection_mixer(self, hidden):
            return ("collapsed", hidden)

    text = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=lambda token_ids: ("embedding", token_ids)),
        mtp=FakeMTP(),
        _head_logits=lambda hidden: stock_calls.append(hidden) or "stock",
        _mtp_draft_head_logits=lambda hidden: draft_calls.append(hidden) or "draft",
    )

    logits, _hidden = TextModel.mtp_forward(
        text,
        "hidden",
        "token",
        mtp_cache="cache",
        return_hidden=True,
    )

    assert logits == "draft"
    assert len(draft_calls) == 1
    assert stock_calls == []


def test_frspec_install_binds_full_shape_head_once(monkeypatch, tmp_path) -> None:
    linear = nn.Linear(64, 8, bias=False)
    linear.weight = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    native = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    configured = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    mx.eval(native.parameters(), configured.parameters())

    vocab = tmp_path / "draft-vocab.json"
    vocab.write_text(json.dumps({"ids": [1, 6]}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(vocab))
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    monkeypatch.delenv("MTPLX_FRSPEC_LEGACY", raising=False)

    bound: list[object] = []
    text = SimpleNamespace(
        _mtplx_draft_lm_head=configured,
        _mtplx_native_mtp_draft_head=lambda: native,
        _mtplx_bind_draft_lm_head=bound.append,
    )
    report = install_frspec_draft_head(text)

    assert report["installed"] is True
    assert report["bits"] == 8
    assert report["group_size"] == 64
    assert report["mode"] == "affine"
    assert report["source"] == "native_mtp_head"
    assert report["output_mode"] == "full"
    assert len(bound) == 1
    installed = bound[0]

    x = mx.arange(64, dtype=mx.float32).reshape(1, 1, 64) / 64
    full = native(x)
    reduced = installed(x)
    mx.eval(full, reduced)

    assert tuple(reduced.shape) == (1, 1, 8)
    assert mx.array_equal(reduced[..., [1, 6]], full[..., [1, 6]]).item()
    assert bool(mx.all(reduced[..., [0, 2, 3, 4, 5, 7]] < -1e20).item())


def test_frspec_accepts_q4_g64_native_head(monkeypatch, tmp_path) -> None:
    # Bare-Speed ships an affine Q4/g64 native MTP head. The pruning carries
    # the head's own bits/group_size, so it installs exactly like the Q8/g64
    # Optimized-Speed head.
    linear = nn.Linear(64, 8, bias=False)
    linear.weight = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    native = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    mx.eval(native.parameters())
    vocab = tmp_path / "draft-vocab.json"
    vocab.write_text(json.dumps({"ids": [1, 6]}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(vocab))
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    monkeypatch.delenv("MTPLX_FRSPEC_LEGACY", raising=False)

    report = install_frspec_draft_head(
        SimpleNamespace(_mtplx_native_mtp_draft_head=lambda: native)
    )

    assert report["installed"] is True
    assert report["bits"] == 4
    assert report["group_size"] == 64
    assert report["mode"] == "affine"
    assert report["source"] == "native_mtp_head"
    assert report["n"] == 2
    assert report["vocab_rows"] == 8


def test_frspec_rejects_wrong_group_size_native_head(monkeypatch, tmp_path) -> None:
    # Only affine g64 (Q8 or Q4) is prunable; a g32 head is still refused so a
    # mismatched pack fails the model LOAD rather than serving a mispruned head.
    linear = nn.Linear(64, 8, bias=False)
    native = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=4)
    vocab = tmp_path / "draft-vocab.json"
    vocab.write_text(json.dumps({"ids": [1, 6]}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(vocab))

    report = install_frspec_draft_head(
        SimpleNamespace(_mtplx_native_mtp_draft_head=lambda: native)
    )

    assert report == {
        "installed": False,
        "reason": "native_head_contract",
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
    }


def test_frspec_rejects_non_affine_native_head(monkeypatch, tmp_path) -> None:
    linear = nn.Linear(64, 8, bias=False)
    native = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
    native.mode = "mxfp8"
    vocab = tmp_path / "draft-vocab.json"
    vocab.write_text(json.dumps({"ids": [1, 6]}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(vocab))

    report = install_frspec_draft_head(
        SimpleNamespace(_mtplx_native_mtp_draft_head=lambda: native)
    )

    assert report == {
        "installed": False,
        "reason": "native_head_contract",
        "bits": 8,
        "group_size": 64,
        "mode": "mxfp8",
    }


def test_builtin_qwen38_code_vocab_is_packaged(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", "builtin:qwen38-code-64k")
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)

    ids = load_frspec_ids()

    assert ids is not None
    assert len(ids) == 65_536
    assert min(ids) == 0
    assert max(ids) == 248_319
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "65535", "65537"])
def test_builtin_qwen38_code_vocab_rejects_invalid_or_truncated_n(
    monkeypatch, value
) -> None:
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", "builtin:qwen38-code-64k")
    monkeypatch.setenv("MTPLX_FRSPEC_N", value)

    assert load_frspec_ids() is None


def test_enabled_frspec_fails_at_construction_when_vocab_is_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "1")
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(tmp_path / "missing.json"))
    text = SimpleNamespace(
        lm_head=nn.Linear(32, 8, bias=False),
        args=SimpleNamespace(tie_word_embeddings=False),
    )

    with pytest.raises(RuntimeError, match="FR-Spec draft head installation failed"):
        _install_draft_lm_head(
            SimpleNamespace(model=text),
            bits=4,
            group_size=32,
            mode="affine",
        )
