"""FR-Spec on the configured draft-head route (generic ``mtp_patch`` forward).

The dense qwen3_5 forward projects every draft step through
``text._mtplx_draft_lm_head`` and has no native bind hook, so the install must
put the full-vocabulary wrapper in that slot itself; before this change the
pruned head was built and never used unless ``MTPLX_FRSPEC_LEGACY=1``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mtplx import frspec_draft
from mtplx.frspec_draft import install_frspec_draft_head


def _configured_head(vocab: int = 8, dims: int = 64) -> nn.QuantizedLinear:
    linear = nn.Linear(dims, vocab, bias=False)
    linear.weight = mx.arange(vocab * dims, dtype=mx.float32).reshape(vocab, dims) / 100
    head = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    mx.eval(head.parameters())
    return head


@pytest.fixture
def ranked_vocab(monkeypatch, tmp_path):
    vocab = tmp_path / "draft-vocab.json"
    vocab.write_text(json.dumps({"ids": [1, 6]}))
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(vocab))
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    monkeypatch.delenv("MTPLX_FRSPEC_LEGACY", raising=False)
    return [1, 6]


def test_configured_route_projects_every_draft_step_through_the_pruned_head(
    ranked_vocab,
) -> None:
    configured = _configured_head()
    text = SimpleNamespace(_mtplx_draft_lm_head=configured)

    report = install_frspec_draft_head(text)

    assert report["installed"] is True
    assert report["source"] == "configured_draft_head"
    assert report["binding"] == "configured_draft_head"
    assert report["legacy_swap"] is False
    live = text._mtplx_draft_lm_head
    assert live is text._mtplx_frspec_draft_head
    assert text._mtplx_frspec_saved_head is configured

    x = mx.arange(64, dtype=mx.float32).reshape(1, 1, 64) / 64
    full = configured(x)
    pruned = live(x)
    mx.eval(full, pruned)
    # Full-vocabulary output domain: kept rows equal the configured head,
    # pruned rows can never be sampled, ids stay real token ids.
    assert tuple(pruned.shape) == (1, 1, 8)
    assert mx.array_equal(pruned[..., ranked_vocab], full[..., ranked_vocab]).item()
    assert bool(mx.all(pruned[..., [0, 2, 3, 4, 5, 7]] < -1e20).item())


def test_mtp_patch_draft_forward_reads_the_bound_head(ranked_vocab) -> None:
    """The generic draft forward's head lookup resolves to the wrapper."""

    import inspect

    from mtplx import mtp_patch

    source = inspect.getsource(mtp_patch)
    assert 'getattr(self, "_mtplx_draft_lm_head", None)' in source

    text = SimpleNamespace(_mtplx_draft_lm_head=_configured_head())
    install_frspec_draft_head(text)
    head = getattr(text, "_mtplx_draft_lm_head", None)
    assert head is text._mtplx_frspec_draft_head
    assert hasattr(head, "take_prescatter_row")


def test_native_hook_route_is_unchanged(ranked_vocab) -> None:
    configured = _configured_head()
    bound: list[object] = []
    text = SimpleNamespace(
        _mtplx_draft_lm_head=configured,
        _mtplx_native_mtp_draft_head=lambda: nn.QuantizedLinear.from_linear(
            nn.Linear(64, 8, bias=False), group_size=64, bits=8
        ),
        _mtplx_bind_draft_lm_head=bound.append,
    )

    report = install_frspec_draft_head(text)

    assert report["binding"] == "native_mtp_hook"
    assert bound == [text._mtplx_frspec_draft_head]
    # The configured slot keeps the unpruned head on the native route.
    assert text._mtplx_draft_lm_head is configured
    assert not hasattr(text, "_mtplx_frspec_saved_head")


def test_rotated_packed_head_is_refused(monkeypatch, ranked_vocab) -> None:
    text = SimpleNamespace(_mtplx_draft_lm_head=_configured_head())
    monkeypatch.setattr(frspec_draft, "_is_rotated_head", lambda head: True)

    report = install_frspec_draft_head(text)

    assert report == {"installed": False, "reason": "rotated_packed_head"}
    assert not hasattr(text, "_mtplx_frspec_draft_head")


def test_draft_head_identity_follows_the_pruned_head(ranked_vocab) -> None:
    from mtplx.server.openai import _draft_head_identity

    configured = _configured_head()
    text = SimpleNamespace(_mtplx_draft_lm_head=configured)
    before = _draft_head_identity(SimpleNamespace(model=text))
    install_frspec_draft_head(text)
    after = _draft_head_identity(SimpleNamespace(model=text))
    empty = _draft_head_identity(
        SimpleNamespace(model=SimpleNamespace(_mtplx_draft_lm_head=nn.Module()))
    )

    assert after not in (before, empty)
