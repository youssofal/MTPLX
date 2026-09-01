"""The invariant that licenses the direct kernel to ignore ``block_valid``.

MTPLX's public block contract tolerates arbitrary validity holes; the
vendored Steel kernel cannot represent them. It takes only uint32 ids and
derives validity POSITIONALLY: it reads exactly the first
``min(512, (q_abs + 1) // ratio)`` slots of each row. So the producers must
emit a chronological VALID PREFIX — ascending, causally visible ids in
``[0, valid_count)`` and padding after.

Both production selectors do. This module proves it against the real
selectors rather than against a restatement of the rule, because a future
selector variant that emitted a hole inside the prefix would make the direct
kernel attend block 0 exactly where the MPP and gather tiers mask it — a fast,
plausible, wrong answer.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.models.qwen4_exp import QSACache, QSAIndexer, TextArgs

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="the QSA selectors require the Metal GPU",
)

_RATIO = 4
_BUDGET = 32  # block_topk == budget // ratio == 8
_PRIMED = 2176  # comfortably past the 2049 minimum crossover floor


def _args() -> TextArgs:
    return TextArgs(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=_BUDGET,
        indexer_compress_ratio=_RATIO,
    )


@pytest.fixture()
def indexer() -> QSAIndexer:
    mx.random.seed(20260901)
    module = QSAIndexer(_args())
    module.q_layernorm.weight = module.q_layernorm.weight.astype(mx.float16)
    module.k_layernorm.weight = module.k_layernorm.weight.astype(mx.float16)
    mx.eval(module.parameters())
    return module


def _rows(module: QSAIndexer, rows: int, seed: int):
    mx.random.seed(seed)
    hidden_size = int(module.index_qk_proj.weight.shape[1])
    qk_width = (module.n_heads + module.kv_heads) * module.head_dim
    hidden = (mx.random.normal((1, rows, hidden_size)) * 0.2).astype(mx.float16)
    qk_rows = (mx.random.normal((1, rows, qk_width)) * 0.2).astype(mx.float16)
    mx.eval(hidden, qk_rows)
    return hidden, qk_rows


def _lane_env(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_ROWS", "2")
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setenv("MTPLX_FUSED_QSA_INDEXER", "0")
    monkeypatch.setenv("MTPLX_COMPILED_QSA_INDEXER", "0")
    monkeypatch.setenv("MTPLX_QSA_MTP_PRECOMPUTE", "0")


def _select(module: QSAIndexer, monkeypatch, *, rows: int, force_eager: bool):
    """Drive the real producer and return its (block_ids, block_valid)."""

    _lane_env(monkeypatch)
    if force_eager:
        monkeypatch.setattr(
            QSAIndexer, "_prefill_selector_supported", lambda *a, **kw: False
        )
    cache = QSACache(module.ratio)
    hidden, qk_rows = _rows(module, _PRIMED, seed=5)
    with attention_phase("prefill"):
        module(hidden, 0, cache, qk_rows=qk_rows)
        leaves = [f for f in (cache.raw_keys, cache.pooled) if f is not None]
        mx.eval(*leaves)
        cache.kv.offset = _PRIMED

        hidden, qk_rows = _rows(module, rows, seed=9)
        out = module(hidden, _PRIMED, cache, qk_rows=qk_rows)
    return out


def _assert_prefix_contract(block_ids, block_valid, *, pos_start: int) -> None:
    """Exactly the rule the kernel assumes, checked on the host."""

    mx.eval(block_ids, block_valid)
    ids = block_ids.tolist()
    valid = block_valid.tolist()
    slots = len(ids[0])
    for row, (row_ids, row_valid) in enumerate(zip(ids, valid)):
        q_abs = pos_start + row
        complete = (q_abs + 1) // _RATIO
        expected = min(slots, complete)

        count = sum(1 for ok in row_valid if ok)
        assert count == expected, f"row {row}: valid count {count} != {expected}"
        assert row_valid[:count] == [True] * count, f"row {row}: hole in prefix"
        assert row_valid[count:] == [False] * (slots - count), (
            f"row {row}: validity after the prefix"
        )

        selected = row_ids[:count]
        assert selected == sorted(selected), f"row {row}: ids not ascending"
        assert len(set(selected)) == count, f"row {row}: duplicate id"
        assert all(0 <= b < complete for b in selected), (
            f"row {row}: id outside the causally complete range"
        )


@pytest.mark.parametrize("rows", [64, 96])
def test_eager_selector_emits_a_valid_prefix(indexer, monkeypatch, rows):
    out = _select(indexer, monkeypatch, rows=rows, force_eager=True)
    assert isinstance(out, tuple) and out[0] == "flash_prefill"
    _assert_prefix_contract(out[1], out[2], pos_start=_PRIMED)


def test_metal_selector_emits_a_valid_prefix(indexer, monkeypatch):
    """Skips where the Metal prefill selector's own support check refuses;
    the eager oracle above still pins the contract on that hardware."""

    out = _select(indexer, monkeypatch, rows=64, force_eager=False)
    assert isinstance(out, tuple) and out[0] == "flash_prefill"
    _assert_prefix_contract(out[1], out[2], pos_start=_PRIMED)


def test_both_selectors_agree_on_the_prefix_shape(indexer, monkeypatch):
    """The direct kernel reads slots positionally, so the two producers must
    not disagree about HOW MANY slots are live, even if they tie-break the
    selected set differently."""

    eager = _select(indexer, monkeypatch, rows=64, force_eager=True)
    metal = _select(indexer, monkeypatch, rows=64, force_eager=False)
    mx.eval(eager[2], metal[2])
    assert bool(mx.array_equal(eager[2], metal[2]).item())


def test_below_budget_context_never_emits_the_block_tuple(indexer, monkeypatch):
    """Where the kernel's ``complete < topk`` branch would matter, MTPLX
    does not route here at all: the indexer's dense-skip returns None
    because the visible prefix already fits the budget (dense == sparse).

    That is why the short-prefix arithmetic is covered at the adapter level
    (tests/test_qsa_prefill_direct.py) rather than end to end — it is a
    defensive branch of the vendored kernel, not a production regime.
    """

    _lane_env(monkeypatch)
    tokens = indexer.block_topk * _RATIO  # exactly at the dense/sparse edge
    cache = QSACache(indexer.ratio)
    hidden, qk_rows = _rows(indexer, tokens, seed=5)
    with attention_phase("prefill"):
        indexer(hidden, 0, cache, qk_rows=qk_rows)
        mx.eval(*[f for f in (cache.raw_keys, cache.pooled) if f is not None])
        cache.kv.offset = tokens
        hidden, qk_rows = _rows(indexer, 4, seed=3)
        out = indexer(hidden, tokens, cache, qk_rows=qk_rows)

    # Either the dense-skip (None) or a plain dense mask — never the compact
    # block tuple the direct kernel consumes.
    assert not (isinstance(out, tuple) and out and out[0] == "flash_prefill")


def test_production_regime_prefixes_are_always_full(indexer, monkeypatch):
    """The kernel's valid_blocks = min(topk, complete) must match the
    producer's fill; past the crossover that always resolves to topk."""

    _lane_env(monkeypatch)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
    monkeypatch.setattr(
        QSAIndexer, "_prefill_selector_supported", lambda *a, **kw: False
    )
    cache = QSACache(indexer.ratio)
    hidden, qk_rows = _rows(indexer, _PRIMED, seed=5)
    with attention_phase("prefill"):
        indexer(hidden, 0, cache, qk_rows=qk_rows)
        mx.eval(*[f for f in (cache.raw_keys, cache.pooled) if f is not None])
        cache.kv.offset = _PRIMED
        hidden, qk_rows = _rows(indexer, 64, seed=3)
        out = indexer(hidden, _PRIMED, cache, qk_rows=qk_rows)

    assert isinstance(out, tuple) and out[0] == "flash_prefill"
    counts = out[2].astype(mx.int32).sum(axis=-1)
    mx.eval(counts)
    # Every row here is far past the budget, so every prefix is full: this is
    # the regime production actually serves.
    assert set(counts.tolist()) == {indexer.block_topk}
