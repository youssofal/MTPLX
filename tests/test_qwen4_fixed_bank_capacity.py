"""The fixed QSA bank's capacity on the rows-gather lane is a multiple of the K/V growth step.

Why: MLX writes a ``slice_update`` in place only when the array's buffer is at most 16,384 bytes larger than the
array, and its buffer cache recycles buffers up to two pages larger than asked for. A bank cut to a multiple of 4
rows from a 256-row buffer could land in such a buffer for good, and then every K and V bank was copied on every
verify round (2026-09-20: 24 x 135 MB per round at 128K). These tests pin the rule, its scope (rows-gather only:
the dense lane's capacity is untouched), and that the rows-gather result does not depend on the capacity.
"""

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.graphbank as graphbank
from mtplx.graphbank import TensorOffsetQSACache
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

PREFILL = 13
STEP = 4
RESERVE = 32


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    layer = Attention(_tiny_args())
    layer.update(mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters()))
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


@pytest.fixture()
def rows_gather_lane(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    monkeypatch.delenv("MTPLX_QSA_M4_FUSED_KV_GATHER", raising=False)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.bfloat16)


def _prefilled(attn, seed: int = 2) -> QSACache:
    entry = QSACache(compress_ratio=attn.indexer.ratio)
    attn(_hidden(PREFILL, seed), entry)
    return entry


def _old_capacity(needed: int, ratio: int) -> int:
    return ((needed + ratio - 1) // ratio) * ratio


def test_the_rule_in_numbers():
    rule = TensorOffsetQSACache._bank_capacity
    # The founder's ladder cells: prompts of 2**k - 35 tokens plus 516 reserved rows were all 28 rows short of
    # the step, inside the 17-to-47-row window where the recycled buffer can never donate.
    for prompt in (65_502, 131_039):
        needed = prompt + 516
        assert 17 <= 256 - needed % 256 <= 47
        cap = rule(needed, 4, 256, rows_gather=True)
        assert cap % 256 == 0 and cap % 4 == 0
        assert needed <= cap < needed + 256
        # 1,024 bytes a row (2 KV heads x 256 x bfloat16): a whole number of 16,384-byte pages.
        assert (cap * 1024) % 16384 == 0
    # Dense lane: exactly the old formula.
    for needed in (45, 4_577, 16_866):
        assert rule(needed, 4, 256, rows_gather=False) == _old_capacity(needed, 4)
    # A ratio that does not divide the step still yields a multiple of both.
    assert rule(1000, 6, 256, rows_gather=True) % 768 == 0


def test_rows_gather_bank_is_sized_on_the_step(attn, rows_gather_lane):
    bank = TensorOffsetQSACache.from_qsa_cache(_prefilled(attn), reserve_tokens=RESERVE)

    assert bank.fixed_rows_gather is True
    assert bank.capacity == 256
    assert int(bank.kv.keys.shape[2]) == int(bank.kv.values.shape[2]) == 256
    assert int(bank.raw_keys.shape[1]) == 256
    assert int(bank.pooled.shape[1]) == 256 // attn.indexer.ratio
    assert bank.size() == PREFILL


def test_dense_bank_capacity_is_what_it_was(attn, monkeypatch):
    monkeypatch.delenv("MTPLX_QSA_GATHER", raising=False)
    monkeypatch.delenv("MTPLX_QSA_GATHER_MIN_CONTEXT", raising=False)

    bank = TensorOffsetQSACache.from_qsa_cache(_prefilled(attn), reserve_tokens=RESERVE)

    assert bank.fixed_rows_gather is False
    assert bank.capacity == _old_capacity(PREFILL + RESERVE, attn.indexer.ratio) == 46


def test_growth_keeps_a_rows_gather_bank_on_the_step(attn, rows_gather_lane):
    bank = TensorOffsetQSACache.from_qsa_cache(_prefilled(attn), reserve_tokens=RESERVE)
    keys_before = mx.array(bank.kv.keys[:, :, :PREFILL])
    mx.eval(keys_before)

    assert bank.ensure_capacity(200) is False  # still fits: nothing moves
    assert bank.ensure_capacity(257) is True
    assert bank.capacity == 512
    assert int(bank.kv.keys.shape[2]) == 512
    assert int(bank.pooled.shape[1]) == 512 // attn.indexer.ratio
    assert bool(mx.array_equal(bank.kv.keys[:, :, :PREFILL], keys_before).item())


def test_rows_gather_result_does_not_depend_on_the_capacity(attn, rows_gather_lane, monkeypatch):
    """The same verify windows through a bank on the step and a bank at the old capacity: bit-identical."""

    steps = [_hidden(STEP, 20 + i) for i in range(4)]

    on_step = TensorOffsetQSACache.from_qsa_cache(_prefilled(attn), reserve_tokens=RESERVE)
    monkeypatch.setattr(
        TensorOffsetQSACache,
        "_bank_capacity",
        staticmethod(
            lambda needed, ratio, kv_step, *, rows_gather, bucket=0, dense_bucket=0: _old_capacity(
                needed, ratio
            )
        ),
    )
    old = TensorOffsetQSACache.from_qsa_cache(_prefilled(attn), reserve_tokens=RESERVE)
    assert on_step.capacity == 256 and old.capacity == 46
    assert on_step.fixed_rows_gather and old.fixed_rows_gather

    for x in steps:
        out_step = attn(x, on_step)
        out_old = attn(x, old)
        assert out_step.dtype == out_old.dtype
        assert bool(mx.array_equal(out_step, out_old).item())
    valid = on_step.size() // attn.indexer.ratio
    assert bool(mx.array_equal(on_step.pooled[:, :valid], old.pooled[:, :valid]).item())


def test_promotion_through_the_bank_entry_point_uses_the_rule(attn, rows_gather_lane):
    cache = [_prefilled(attn)]
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache, reserve_tokens=STEP, initial_reserve_tokens=RESERVE
    )

    assert promoted == 1 and failures == {}
    assert isinstance(cache[0], TensorOffsetQSACache)
    assert cache[0].capacity % 256 == 0
