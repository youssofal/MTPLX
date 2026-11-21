"""A conversation that outgrows its held buffers keeps the compiled verifier (mtplx/one_copy.py).

On 2026-10-01 the founder's Flash-Next Pi session reached a screenshot turn of
147,396 tokens. The conversation's QSA buffers held 147,456 rows: the prompt
fit, but not the compiled verifier's 1,024-row reserve. The bank can adopt only
buffers that already hold its planned rows, so the admission priced a padded
copy of the whole bank (4,219,207,680 bytes), the memory gate refused it, and
that turn and all 39 after it verified eagerly, about 4.4 ms slower a round.
After an eager turn the stock buffers also grew their own way (keys and values
by 256 rows, the index keys by doubling), so no later turn could adopt them.

Now the held buffers are resized to the bank's rows one layer at a time before
the bank adopts them, and the admission prices exactly that: the rows they
gain plus one layer's new buffers beside its old ones. This file pins it:

- the bill on the recorded geometry of that turn, and the copying store still
  reproducing the refusal the request log recorded;
- the resize itself: what each layer gets, one layer at a time, a refusal or a
  failure between layers leaving every layer whole, never a cut into held rows,
  and the allocator holding no more than the bill at any point;
- on the tiny Flash-Next pack with two QSA layers, through ``generate_mtpk``
  and the session bank: the transition admitted where the copy is refused; the
  resized-and-adopted bank giving the tokens, the logits, every round's state
  and every cache leaf of a fresh promotion at the same capacity; recovery
  after a refused turn (index keys it doubled cut for a smaller bank) and
  after a resize stopped between layers; zero and one-token warm suffixes; a
  failure during the resize; image positions; compiled copy windows; and the
  next turn's restore.
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions, one_copy, runtime_options
from mtplx.graphbank import FixedM4CapacityPlan, TensorOffsetKVCache, TensorOffsetQSACache
from mtplx.models.qwen4_exp import QSACache
from mtplx.one_copy import held_qsa_rows, resize_bill, resize_qsa_buffers
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from test_qwen4_fixed_m4_capacity_bucket import _bits, _recorded_rounds

GB = 1_000_000_000
# Qwen 3.8's native sampler (Flash-Next's own).
FLASH = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
SEED = 1234


# -- the 2026-10-01 turn, priced on its recorded geometry (nothing allocated) -------

LIMIT = 96_636_764_160  # the Metal limit the serve path pinned: 90 GiB
LINE = 93_737_661_235  # request log row 102 threshold_bytes (0.97 of the limit)
LIVE = 93_408_488_308  # row 102 live_bytes_before
LIVE_RELEASED = 90_392_065_662  # row 102 live_bytes_after (the allocator cache released)
PROMPT = 147_396  # row 102 prompt_tokens
HELD = 147_456  # row 101's bank capacity, handed back whole
FN_ROW = 28_416  # bytes per row of the 12 QSA layers: 12 x (2,048 + 256 + 64)


class _Held:
    """A buffer's shape and size without allocating it."""

    def __init__(self, *shape, dtype=mx.bfloat16):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.nbytes = dtype.size * math.prod(shape)


def _flash_next_layer(rows: int, offset: int, *, raw: int | None = None, pooled: int | None = None):
    """A Flash-Next QSA layer holding ``offset`` tokens in ``rows``-row buffers (shapes only)."""

    raw = rows if raw is None else raw
    pooled = raw // 4 if pooled is None else pooled
    entry = QSACache(4)
    entry.kv.keys = _Held(1, 2, rows, 256)
    entry.kv.values = _Held(1, 2, rows, 256)
    entry.kv.offset = offset
    entry.raw_keys = _Held(1, raw, 128)
    entry.pooled = _Held(1, pooled, 128)
    entry.pooled_len = offset // 4
    return entry


def _flash_next_cache(layers):
    """Flash-Next's trunk cache shape: three recurrent layers before each QSA layer."""

    cache = []
    for layer in layers:
        cache.extend([None, None, None, layer])
    return cache


def _flash_next_rt():
    args = SimpleNamespace(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 12,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    return SimpleNamespace(
        model=SimpleNamespace(args=args),
        metal_memory_limit_bytes=LIMIT,
        qwen4_fixed_m4_compiled_verify=True,
    )


class _Bank:
    """Row 102's session bank: 5,841,036,432 bytes, nothing it can give back."""

    total_nbytes = 5_841_036_432

    def __init__(self):
        self.reclaims = 0

    def shrink_for_admission(self, target, *, protect_tokens, reason):
        self.reclaims += 1
        return 0, 0


@pytest.fixture()
def gather(monkeypatch):
    """The Flash-Next serve lane: rows-gather from 16K, the 8,192-row bucket."""

    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    for name in (
        "MTPLX_QSA_GATHER_MIN_CONTEXT",
        "MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET",
        "MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
        "MTPLX_QSA_SCORE_TILE_ROWS",
        "MTPLX_ONE_COPY",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield monkeypatch
    demotions.reset()


def _gate(monkeypatch, *, held_cache, live=LIVE, prompt=PROMPT):
    """The lane's memory gate on row 102's machine: releasing the allocator
    cache frees what it freed there."""

    reading = {"live": live}

    def release():
        freed = reading["live"] - LIVE_RELEASED
        reading["live"] = LIVE_RELEASED
        return max(0, freed)

    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: reading["live"])
    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", release)
    rt = _flash_next_rt()
    plan = FixedM4CapacityPlan.for_request(92_660, runtime=rt)
    bank = _Bank()
    receipt: dict = {}
    fits = generation._qwen4_fixed_m4_lane_fits(
        rt, prompt_tokens=prompt, session_bank=bank, prompt_ids=[1, 2, 3],
        receipt=receipt, capacity_plan=plan, held_cache=held_cache,
    )
    return fits, receipt, plan, bank


def test_the_147k_turn_is_admitted_as_a_resize(gather):
    cache = _flash_next_cache([_flash_next_layer(HELD, PROMPT) for _ in range(12)])
    assert held_qsa_rows(cache) == HELD
    assert int(LIMIT * generation._QWEN4_FIXED_M4_PRESSURE_FRACTION) == LINE

    fits, receipt, plan, bank = _gate(gather, held_cache=cache)

    # The bucketed resize (155,648 rows: 0.582 GB) is over the line until the
    # allocator cache is released, and the lane's rule drops the bucket before
    # that release; the unbucketed resize then fits once it is released.
    assert fits is True
    assert receipt["promotion"] == "resized"
    assert receipt["held_rows"] == HELD
    assert receipt["promotion_rows"] == 148_480
    assert receipt["capacity_bucket"] == 0 and plan.bucket == 0
    # 11 layers gain 1,024 rows, the twelfth is written beside its old buffers.
    layer_new, layer_old = 148_480 * FN_ROW // 12, HELD * FN_ROW // 12
    assert receipt["promotion_bytes"] == 11 * (layer_new - layer_old) + layer_new == 378_273_792
    assert receipt["live_bytes_before"] == LIVE and receipt["threshold_bytes"] == LINE
    assert bank.reclaims == 0  # no idle session was evicted for it
    assert plan.resize_rows == 148_480

    # With nothing cached the bucketed resize fits and the bucket is kept.
    fits, receipt, plan, _bank = _gate(gather, held_cache=cache, live=LIVE_RELEASED)
    assert fits is True and receipt["promotion"] == "resized"
    assert receipt["promotion_rows"] == 155_648 and plan.bucket == 8192
    layer_new = 155_648 * FN_ROW // 12
    assert receipt["promotion_bytes"] == 11 * (layer_new - layer_old) + layer_new == 581_959_680
    assert plan.resize_rows == 155_648


def test_the_copying_store_still_prices_the_padded_copy_and_refuses(gather):
    """Row 102 of the 2026-10-01 request log, field for field."""

    fits, receipt, plan, bank = _gate(gather, held_cache=None)

    assert fits is False
    assert receipt["promotion"] == "copied"
    assert receipt["promotion_bytes"] == 4_219_207_680 == 148_480 * FN_ROW
    assert receipt["promotion_rows"] == 148_480 and receipt["capacity_bucket"] == 0
    assert receipt["live_bytes_before"] == LIVE
    assert receipt["live_bytes_after"] == LIVE_RELEASED
    assert receipt["threshold_bytes"] == LINE
    assert receipt["bank_bytes_before"] == receipt["bank_bytes_after"] == 5_841_036_432
    assert receipt["chain_entries_evicted"] == receipt["terminal_entries_evicted"] == 0
    assert "held_rows" not in receipt and plan.resize_rows == 0
    assert bank.reclaims == 1
    assert demotions.snapshot()["counts"]["fixed_m4_lane_skipped"] == 1


def test_unequal_layers_are_priced_on_what_each_one_changes(gather):
    """After an eager turn: keys and values grown past the prompt in 256-row
    steps, the index keys doubled."""

    eager = [
        _flash_next_layer(PROMPT + 512, PROMPT, raw=2 * HELD, pooled=2 * HELD // 4)
        for _ in range(12)
    ]
    cache = _flash_next_cache(eager)
    assert held_qsa_rows(cache) is None
    rows = 155_648
    kv_row, raw_row, pooled_row = 2 * 2 * 256 * 2, 128 * 2, 128 * 2  # bytes per row or block
    kv_new, kv_old = rows * kv_row, (PROMPT + 512) * kv_row
    index_new = rows * raw_row + rows // 4 * pooled_row
    index_old = 2 * HELD * raw_row + 2 * HELD // 4 * pooled_row
    per_layer = (kv_new + index_new) - (kv_old + index_old)
    assert per_layer < 0  # the doubled index buffers are cut back
    assert resize_bill(cache, rows) == kv_new + index_new  # the first layer's new buffers

    # A resize that stopped after six layers: only the other six are priced.
    resized = [_flash_next_layer(rows, PROMPT) for _ in range(6)]
    plain = [_flash_next_layer(HELD, PROMPT) for _ in range(6)]
    layer_new, layer_old = rows * FN_ROW // 12, HELD * FN_ROW // 12
    assert resize_bill(_flash_next_cache(resized + plain), rows) == (
        5 * (layer_new - layer_old) + layer_new
    )
    assert resize_bill(_flash_next_cache(resized), rows) == 0


def test_the_dense_lane_and_unresizable_caches_keep_the_copy(gather):
    # Below the rows-gather switch the prefill does not size the buffers for
    # the bank, and the bank's width is arithmetic: the copy stays.
    small = _flash_next_cache([_flash_next_layer(8_448, 8_000) for _ in range(12)])
    fits, receipt, _plan, _bank = _gate(gather, held_cache=small, live=10 * GB, prompt=8_000)
    assert fits is True and receipt["promotion"] == "copied"
    assert receipt["promotion_bytes"] == 9_024 * FN_ROW

    # A layer already promoted to a bank cannot be resized here.
    promoted = _flash_next_cache([_flash_next_layer(HELD, PROMPT) for _ in range(12)])
    promoted[7] = TensorOffsetQSACache.__new__(TensorOffsetQSACache)
    assert resize_bill(promoted, 155_648) is None
    with pytest.raises(ValueError, match="cannot be resized"):
        resize_qsa_buffers(promoted, 155_648)


def test_a_bucket_drop_that_lands_on_the_held_rows_adopts_them(gather):
    # A turn that ran with the bucket dropped handed back 148,480 rows; the
    # next prompt plans 155,648 with the bucket and 148,480 without it.
    cache = _flash_next_cache([_flash_next_layer(148_480, PROMPT) for _ in range(12)])
    fits, receipt, plan, bank = _gate(gather, held_cache=cache, live=LINE - 100_000_000)
    assert fits is True
    assert receipt["promotion"] == "adopted" and receipt["promotion_bytes"] == 0
    assert receipt["promotion_rows"] == 148_480 and plan.bucket == 0
    assert plan.resize_rows == 0 and bank.reclaims == 0


def test_pooled_keys_short_of_the_bank_are_a_billed_resize(gather):
    """Keys, values and raw index keys at the bank's rows, the pooled index
    keys short of its blocks (a suffix that completed no block): the bank
    used to grow them as it was built, with the admission billing 0."""

    rows = 155_648
    short = _flash_next_cache(
        [_flash_next_layer(rows, PROMPT, pooled=HELD // 4) for _ in range(12)]
    )
    fits, receipt, plan, bank = _gate(gather, held_cache=short, live=LIVE_RELEASED)
    assert fits is True
    assert receipt["promotion"] == "resized" and receipt["held_rows"] == rows
    pooled_new, pooled_old = rows // 4 * 128 * 2, HELD // 4 * 128 * 2
    assert receipt["promotion_bytes"] == 11 * (pooled_new - pooled_old) + pooled_new
    assert receipt["promotion_bytes"] == 15_728_640
    assert plan.resize_rows == rows and plan.bucket == 8192

    # Every buffer held: adopted, and nothing is billed or allocated.
    whole = _flash_next_cache([_flash_next_layer(rows, PROMPT) for _ in range(12)])
    fits, receipt, plan, bank = _gate(gather, held_cache=whole, live=LIVE_RELEASED)
    assert fits is True and receipt["promotion"] == "adopted"
    assert receipt["promotion_bytes"] == 0 and plan.resize_rows == 0


def _flash_next_bank(rows: int, offset: int):
    """An installed Flash-Next QSA bank of ``rows`` rows (shapes only)."""

    kv = TensorOffsetKVCache(_Held(1, 2, rows, 256), _Held(1, 2, rows, 256), offset)
    return TensorOffsetQSACache(
        kv, _Held(1, rows, 128), _Held(1, rows // 4, 128), compress_ratio=4,
        rows_gather=True, rows_gather_kv_m4=None, capacity_bucket=8192,
    )


def test_an_installed_growth_is_billed_at_its_peak(gather):
    """The banks the transition installed, growing by one 8,192-row bucket
    during the answer: the layers grown before the last keep their gain and
    the last is written beside its old banks. The admission used to price
    one layer's new banks alone (generation._qwen4_fixed_m4_growth_fits,
    until the 2026-10-01 review)."""

    rows, grown = 155_648, 163_840
    banks = [_flash_next_bank(rows, PROMPT) for _ in range(12)]
    growing = graphbank._fixed_m4_growing(banks, rows + 1)
    assert [target for target, _bank in growing] == [grown] * 12
    layer_new, layer_old = grown * FN_ROW // 12, rows * FN_ROW // 12
    assert banks[0].growth_bytes(grown) == (layer_new, layer_old) == (387_973_120, 368_574_464)
    assert graphbank._fixed_m4_growth_bill(growing) == 11 * (layer_new - layer_old) + layer_new
    assert graphbank._fixed_m4_growth_bill(growing) == 601_358_336
    # Without the bucket the same answer grows to the rows it needs.
    unbucketed = graphbank._fixed_m4_growing(banks, rows + 1, bucket=0)
    assert [target for target, _bank in unbucketed] == [rows + 256] * 12


# -- the resize -------------------------------------------------------------------


def _layer(offset, rows, *, raw=None, pooled=None, seed=0):
    """A stock QSA layer holding ``offset`` tokens; buffers of ``rows`` (index: ``raw``)."""

    mx.random.seed(seed)
    raw = rows if raw is None else raw
    pooled = raw // 4 if pooled is None else pooled
    entry = QSACache(4)
    entry.kv.keys = mx.random.normal((1, 2, rows, 16)).astype(mx.bfloat16)
    entry.kv.values = mx.random.normal((1, 2, rows, 16)).astype(mx.bfloat16)
    entry.kv.offset = offset
    entry.raw_keys = mx.random.normal((1, raw, 8)).astype(mx.bfloat16)
    entry.pooled = mx.random.normal((1, pooled, 8)).astype(mx.bfloat16)
    entry.pooled_len = offset // 4
    entry.indexer_budget = 8
    entry.pooled_f32_t = mx.swapaxes(entry.pooled.astype(mx.float32), 1, 2)[:, None]
    mx.eval(entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled, entry.pooled_f32_t)
    return entry


def _address(value) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint8), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _buffers(entry):
    return (entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled)


def _recurrent():
    entry = ArraysCache(size=2)
    entry[0] = mx.zeros((1, 3, 8), dtype=mx.bfloat16)
    entry[1] = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
    return entry


def test_each_layer_gets_what_the_bank_would_have_copied_in_buffers_of_its_own(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    for name in ("MTPLX_QSA_M4_FUSED_KV_GATHER", "MTPLX_QSA_SCORE_TILE_ROWS"):
        monkeypatch.delenv(name, raising=False)
    grow = _layer(300, 512, seed=1)
    eager = _layer(300, 768, raw=1024, pooled=256, seed=2)  # after an eager turn
    ready = _layer(300, 1024, seed=3)
    cache = [_recurrent(), grow, _recurrent(), eager, ready]
    before = {
        id(entry): [(_bits(b), _address(b)) for b in _buffers(entry)] for entry in (grow, eager)
    }
    ready_buffers = _buffers(ready)
    ready_mirror = ready.pooled_f32_t
    asked = []

    def admit(need):
        # Asked before anything of the layer is allocated.
        layer = grow if not asked else eager
        for buffer, (bits, _address_) in zip(_buffers(layer), before[id(layer)]):
            assert np.array_equal(_bits(buffer), bits)
        asked.append(need)
        return True

    assert resize_qsa_buffers(cache, 1024, admit=admit) == 0
    # keys and values (2 x 1 x 2 x 1024 x 16 bf16), raw (1024 x 8), pooled (256 x 8)
    full = 2 * 2 * 1024 * 16 * 2 + 1024 * 8 * 2 + 256 * 8 * 2
    assert asked == [full, 2 * 2 * 1024 * 16 * 2]  # the eager layer's index buffers already fit
    originals = (_layer(300, 512, seed=1), _layer(300, 768, raw=1024, pooled=256, seed=2))
    for entry, old in zip((grow, eager), originals):
        for got, want, axis, rows in zip(
            _buffers(entry), _buffers(old), (2, 2, 1, 1), (1024, 1024, 1024, 256)
        ):
            assert got.shape[axis] == rows
            reference = TensorOffsetQSACache._fixed_bank(want, rows, axis)
            assert np.array_equal(_bits(got), _bits(reference))
        assert entry.kv.offset == 300 and entry.pooled_len == 75
        assert TensorOffsetQSACache.held_rows(entry) == 1024
    # New allocations, never views of the old buffers.
    for got, (_bits_, address) in zip(_buffers(grow), before[id(grow)]):
        assert _address(got) != address
    for got, (_bits_, address) in zip(_buffers(eager)[:2], before[id(eager)][:2]):
        assert _address(got) != address
    assert grow.pooled_f32_t is None  # derived from the old pooled buffer
    # A layer that already holds the rows is not touched.
    assert all(a is b for a, b in zip(_buffers(ready), ready_buffers))
    assert ready.pooled_f32_t is ready_mirror
    assert held_qsa_rows(cache) == 1024
    # The bank adopts every one of them.
    for entry in (grow, eager, ready):
        bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
        assert bank.promotion == "adopted" and bank.capacity == 1024


def test_a_cut_copies_into_a_buffer_of_its_own():
    entry = _layer(300, 512, raw=2048, pooled=512, seed=4)
    raw, pooled = entry.raw_keys, entry.pooled
    assert resize_qsa_buffers([entry], 512) == 0
    assert entry.raw_keys.shape == (1, 512, 8) and entry.pooled.shape == (1, 128, 8)
    assert np.array_equal(_bits(entry.raw_keys), _bits(raw[:, :512]))
    assert np.array_equal(_bits(entry.pooled), _bits(pooled[:, :128]))
    assert _address(entry.raw_keys) != _address(raw)
    assert _address(entry.pooled) != _address(pooled)


def test_a_refusal_between_layers_leaves_every_layer_whole():
    layers = [_layer(300, 512, seed=s) for s in (5, 6, 7)]
    old = [_buffers(layer) for layer in layers]
    calls = []

    def admit(need):
        calls.append(need)
        return len(calls) < 2

    assert resize_qsa_buffers(layers, 1024, admit=admit) == 2
    assert TensorOffsetQSACache.held_rows(layers[0]) == 1024
    for layer, buffers in zip(layers[1:], old[1:]):
        assert all(a is b for a, b in zip(_buffers(layer), buffers))
    assert held_qsa_rows(layers) is None
    # The next resize finishes the job from the unequal layout.
    assert resize_qsa_buffers(layers, 1024) == 0 and held_qsa_rows(layers) == 1024


def test_a_failure_while_a_layer_is_written_leaves_that_layer_as_it_was(monkeypatch):
    layers = [_layer(300, 512, seed=s) for s in (8, 9)]
    old = [_buffers(layer) for layer in layers]
    real = mx.eval
    calls = {"n": 0}

    def failing(*arrays):
        # Metal allocates when the new buffers are written: a failed
        # allocation surfaces here.
        calls["n"] += 1
        if calls["n"] == 2:  # the second layer's buffers
            raise RuntimeError("[metal::malloc] Unable to allocate")
        return real(*arrays)

    monkeypatch.setattr(mx, "eval", failing)
    with pytest.raises(RuntimeError, match="Unable to allocate"):
        resize_qsa_buffers(layers, 1024)
    monkeypatch.setattr(mx, "eval", real)
    assert TensorOffsetQSACache.held_rows(layers[0]) == 1024
    assert all(a is b for a, b in zip(_buffers(layers[1]), old[1]))
    assert layers[1].pooled_f32_t is not None


def _resize_transient() -> dict:
    """What the allocator holds while three layers are resized from 4,096 rows to 8,192.

    Run by test_the_resize_allocates_its_bill_and_no_more in a process of its
    own. Buffer sizes are whole pages, which the allocator rounds every
    larger allocation to, so its counts match the bill byte for byte.
    """

    layers = [_layer(4000, 4096, seed=s) for s in (11, 12, 13)]
    for layer in layers:
        # The float32 mirror goes with the pooled keys it was made from; the
        # bill leaves it out, and so does this count.
        layer.pooled_f32_t = None
    bill = resize_bill(layers, 8192)
    needs, readings = [], []
    gc.collect()
    mx.synchronize()
    start = mx.get_active_memory()
    mx.reset_peak_memory()

    def admit(need):
        needs.append(need)
        readings.append(mx.get_active_memory() - start)
        return True

    left = resize_qsa_buffers(layers, 8192, admit=admit)
    return {
        "left": left,
        "bill": bill,
        "needs": needs,
        "readings": readings,
        "peak": mx.get_peak_memory() - start,
        "end": mx.get_active_memory() - start,
    }


@pytest.mark.skipif(not mx.metal.is_available(), reason="measures the Metal allocator")
def test_the_resize_allocates_its_bill_and_no_more():
    """One layer's new buffers beside the conversation at a time, never two.

    Measured in a fresh process: the allocator's counts are the whole
    process's, and in a long test run other tests' arrays come and go
    beside these.
    """

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (root, env.get("PYTHONPATH", "")) if item
    )
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--isolated", "_resize_transient"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    got = json.loads(done.stdout.strip().splitlines()[-1])
    print(got)
    new = 2 * (2 * 8192 * 16 * 2) + 8192 * 8 * 2 + 2048 * 8 * 2
    gain = new - 2 * (2 * 4096 * 16 * 2) - 4096 * 8 * 2 - 1024 * 8 * 2
    assert got["left"] == 0 and got["needs"] == [new] * 3
    assert got["bill"] == 2 * gain + new
    # Each layer is admitted with the layers before it resized and their old
    # buffers gone; at the end only the gained rows remain.
    assert got["readings"] == [0, gain, 2 * gain]
    assert got["end"] == 3 * gain
    # The peak is the bill: the third layer's new buffers beside its old
    # ones. What is over it is the zero each new buffer is filled from.
    assert 0 <= got["peak"] - got["bill"] < 4096


def test_a_resize_never_cuts_the_rows_a_layer_holds():
    entry = _layer(300, 512, seed=10)
    buffers = _buffers(entry)
    with pytest.raises(ValueError, match="would cut them"):
        resize_qsa_buffers([entry], 256)
    assert all(a is b for a, b in zip(_buffers(entry), buffers))


# -- through generate_mtpk on the tiny Flash-Next pack ------------------------------
#
# Two QSA layers (four layers, recurrent and QSA in turn), so layers can
# disagree. A 512-row bucket, rows-gather from 16 tokens. Turn 0 (440 tokens)
# and turn 1 (484) leave the conversation in 512-row buffers at 508 tokens;
# turn 2 adds a short suffix: the prompt fits the 512 rows, the bank's reserve
# (its answer's 24 tokens plus 4) does not, and the bank plans 1,024 rows
# (768 without the bucket).

gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="bfloat16 expert gathers need the GPU"
)
TINY_LIVE = 90 * GB
TINY_ROW = 2 * (2 * 2 * 32 * 2 + 32 * 2 + 32 * 2 // 4)  # 672 bytes a row over both layers


@pytest.fixture(scope="module")
def tiny():
    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp
    from test_qwen4_fixed_m4_verify_exactness import _runtime

    if not mx.metal.is_available():
        pytest.skip("bfloat16 expert gathers need the GPU")
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    rt = _runtime(num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2)
    assert generation._qwen4_qsa_layer_count(rt) == 2
    assert generation._qwen4_fixed_m4_promotion_bytes_per_token(rt) == TINY_ROW
    yield rt
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(previous_device)


@pytest.fixture()
def lane(monkeypatch):
    for name in tuple(os.environ):
        prefixes = ("MTPLX_QWEN4_", "MTPLX_QSA_", "MTPLX_CONTEXT_COPY", "MTPLX_RAMP_")
        if name.startswith(prefixes) or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
            "MTPLX_FIXED_M4_COPY_WINDOWS",
            "MTPLX_ONE_COPY",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    monkeypatch.setattr(runtime_options, "_QWEN4_VERIFY_GLUE", False)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "proven", False)
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "retired", None)
    monkeypatch.setitem(graphbank._FIXED_M4_COPY_WINDOWS, "proven", set())
    monkeypatch.setitem(graphbank._FIXED_M4_COPY_WINDOWS, "retired", None)
    demotions.reset()
    yield monkeypatch
    demotions.reset()


def _squeeze(patch, room: int) -> None:
    """The engine ``room`` bytes under the lane's line, with nothing cached to release."""

    fraction = generation._QWEN4_FIXED_M4_PRESSURE_FRACTION
    limit = math.ceil((TINY_LIVE + room) / fraction)
    assert 0 <= int(limit * fraction) - (TINY_LIVE + room) <= 1
    patch.setattr(generation, "_metal_memory_limit_bytes", lambda rt: limit)
    patch.setattr(generation, "_mlx_live_memory_bytes", lambda: TINY_LIVE)
    patch.setattr(generation, "_mlx_release_allocator_cache", lambda: 0)


def _parent(patch) -> None:
    """The parent path: buffers the bank cannot adopt are copied into a padded bank."""

    patch.setattr(generation, "resize_bill", lambda cache, rows: None)


def _leaf_bits(cache, *, valid_only: bool = False) -> dict[str, np.ndarray]:
    """Every leaf of a stock cache, whole buffers (or the rows they hold)."""

    out: dict[str, np.ndarray] = {}
    for index, entry in enumerate(cache or ()):
        if isinstance(entry, QSACache):
            end = int(entry.offset)
            out[f"{index}.offset"] = np.array(end)
            for name, leaf, axis, valid in (
                ("keys", entry.kv.keys, 2, end),
                ("values", entry.kv.values, 2, end),
                ("raw", entry.raw_keys, 1, end),
                ("pooled", entry.pooled, 1, end // entry.ratio),
            ):
                if valid_only:
                    leaf = leaf[:, :, :valid] if axis == 2 else leaf[:, :valid]
                out[f"{index}.{name}"] = _bits(leaf)
            continue
        for slot, leaf in enumerate(entry.state):
            if isinstance(leaf, mx.array):
                out[f"{index}.{slot}"] = _bits(leaf)
    return out


class _Session:
    """One conversation on one runtime, banked between turns as the server banks it."""

    def __init__(self, rt, lane, *, parent: bool = False, splice_for=None):
        self.rt = rt
        self.lane = lane
        self.parent = parent
        self.splice_for = splice_for
        self.bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
        self.tokens: list[int] = []
        self.hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)

    def turn(self, suffix, *, max_tokens=24, room=None, patches=(), seed=None, prompt=None):
        prompt = list(self.tokens) + list(suffix) if prompt is None else list(prompt)
        promotions: list[tuple[str, int]] = []
        with self.lane.context() as patch:
            real_from = TensorOffsetQSACache.from_qsa_cache.__func__

            def recording(cls, entry, **kwargs):
                made = real_from(cls, entry, **kwargs)
                promotions.append((made.promotion, made.capacity))
                return made

            patch.setattr(TensorOffsetQSACache, "from_qsa_cache", classmethod(recording))
            if self.parent:
                _parent(patch)
            if room is not None:
                _squeeze(patch, room)
            for apply in patches:
                apply(patch)
            rounds = _recorded_rounds(patch)
            result = generation.generate_mtpk(
                self.rt, prompt, max_tokens=max_tokens, sampler=FLASH, draft_sampler=FLASH,
                speculative_depth=3, seed=SEED + len(prompt) if seed is None else seed,
                mtp_cache_policy="persistent", mtp_history_policy="committed",
                verify_strategy="batched", stop_token_ids=set(), capture_final_state=True,
                session_bank=self.bank, session_id="s", session_restore_mode="reference",
                commit_prompt_state_to_bank=True,
                vision_splice=self.splice_for(prompt) if self.splice_for else None,
            )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        tokens = prompt + list(result.tokens)
        record = SimpleNamespace(
            prompt=prompt,
            tokens=list(result.tokens),
            stats=result.stats,
            admission=dict(result.stats.fixed_m4_admission or {}),
            bank=dict((result.stats.graphbank or {}).get("compiled_verify") or {}),
            promotions=promotions,
            rounds=rounds,
            leaves=_leaf_bits(final.final_trunk_cache),
            valid=_leaf_bits(final.final_trunk_cache, valid_only=True),
            draft=_leaf_bits(final.final_committed_mtp_cache),
            logits=_bits(final.final_logits),
            held=[
                (int(e.kv.keys.shape[2]), int(e.raw_keys.shape[1]), int(e.pooled.shape[1]))
                for e in final.final_trunk_cache
                if isinstance(e, QSACache)
            ],
        )
        key = tokens
        if self.splice_for is not None:
            from mtplx.vision.splice import vision_bank_key_ids

            key = vision_bank_key_ids(tokens, self.splice_for(tokens)) or tokens
        self.bank.put(
            runtime=self.rt, token_ids=list(key), cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=True,
            session_id="s", mtp_history_policy=final.mtp_history_policy,
            hidden_variant=self.hidden_variant, snapshot_epoch=len(tokens),
            mtp_snapshot_epoch=len(tokens),
            mtp_history_cache_ref=final.final_committed_mtp_cache,
        )
        self.tokens = tokens
        return record

    def opening(self):
        """Turns 0 and 1: the conversation at 508 tokens in 512-row buffers."""

        first = self.turn([(7 * i + 3) % 128 for i in range(440)])
        second = self.turn([5, 6, 7, 8] * 5)
        assert len(self.tokens) == 508
        assert second.held == [(512, 512, 128)] * 2
        assert first.admission["promotion"] == second.admission["promotion"] == "adopted"
        return first, second


def _same(got: dict, want: dict, what: str) -> None:
    assert got.keys() == want.keys(), what
    for name, value in want.items():
        assert got[name].shape == value.shape, (what, name)
        assert np.array_equal(got[name], value), (what, name)


def _same_rounds(got: list[dict], want: list[dict]) -> None:
    assert len(got) == len(want) > 0
    for index, (a, b) in enumerate(zip(got, want)):
        _same(
            {k: np.asarray(v) for k, v in a.items()},
            {k: np.asarray(v) for k, v in b.items()},
            f"round {index}",
        )


def _same_turn(got, want, *, capacity: bool = True) -> None:
    """The same tokens, final logits, draft history and cache leaves.

    ``capacity`` compares whole buffers (the same bank rows on both sides);
    otherwise the rows each layer holds.
    """

    assert any(name.endswith(".keys") for name in want.leaves)
    assert got.tokens == want.tokens
    assert np.array_equal(got.logits, want.logits)
    _same(got.draft, want.draft, "draft history")
    _same(got.leaves if capacity else got.valid, want.leaves if capacity else want.valid, "trunk")


def _compiled(record) -> None:
    assert record.admission["engaged"] is True and record.admission["reason"] == "admitted"
    assert record.bank["fixed_m4"]["installed"] is True
    assert record.bank["compiled_calls"] == len(record.rounds) > 0
    assert record.bank["fallback_calls"] == 0


def _eager(record) -> None:
    assert record.admission["engaged"] is False and record.admission["reason"] == "memory_gate"
    assert record.bank == {} and record.rounds == []


@gpu
def test_the_transition_turn_resizes_where_the_copy_was_refused(tiny, lane):
    """The 147K turn in miniature: 400,000 bytes of room under the line."""

    new, refused, roomy = (_Session(tiny, lane, parent=p) for p in (False, True, True))
    for session in (new, refused, roomy):
        session.opening()
    turn_new = new.turn([5, 6], room=400_000)
    turn_refused = refused.turn([5, 6], room=400_000)
    turn_roomy = roomy.turn([5, 6])

    # Now: the turn keeps the compiled verifier, and the bank adopts the
    # resized buffers.
    _compiled(turn_new)
    assert turn_new.promotions == [("adopted", 768)] * 2
    assert turn_new.held == [(768, 768, 192)] * 2
    # Today: the padded copy (1,024 rows, then 768 without the bucket) is
    # over the line, and the turn verifies eagerly.
    _eager(turn_refused)
    assert turn_refused.admission["promotion"] == "copied"
    assert turn_refused.admission["promotion_bytes"] == 768 * TINY_ROW == 516_096
    # The resize costs the rows the layers gain plus one layer beside its old
    # buffers.
    assert turn_new.admission["promotion"] == "resized"
    assert turn_new.admission["held_rows"] == 512
    assert turn_new.admission["promotion_rows"] == 768
    assert turn_new.admission["capacity_bucket"] == 0
    layer_new, layer_old = 768 * TINY_ROW // 2, 512 * TINY_ROW // 2
    assert turn_new.admission["promotion_bytes"] == (layer_new - layer_old) + layer_new == 344_064
    # The parent with room copies 1,024 rows; the rows-gather lane gives the
    # same tokens at both widths, and the same as the eager verifier.
    assert turn_roomy.promotions == [("copied", 1024)] * 2
    _same_turn(turn_new, turn_roomy, capacity=False)
    _same_turn(turn_new, turn_refused, capacity=False)
    # Not only tokens: every round's logits, hidden rows, captures and state.
    _same_rounds(
        [{k: v for k, v in r.items() if k != "capacity"} for r in turn_new.rounds],
        [{k: v for k, v in r.items() if k != "capacity"} for r in turn_roomy.rounds],
    )


@gpu
@pytest.mark.parametrize("suffix", [[], [5], [5, 6]], ids=["identical", "one", "two"])
def test_the_resized_bank_is_the_copied_bank(tiny, lane, suffix):
    """At the same capacity, every bit: tokens, rounds, leaves and the next turn."""

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    turn_new = new.turn(suffix, max_tokens=48)
    turn_parent = parent.turn(suffix, max_tokens=48)

    _compiled(turn_new)
    _compiled(turn_parent)
    assert turn_new.promotions == [("adopted", 1024)] * 2
    assert turn_parent.promotions == [("copied", 1024)] * 2
    assert turn_new.admission["promotion"] == "resized"
    assert turn_new.admission["promotion_rows"] == 1024
    assert turn_new.admission["capacity_bucket"] == 512
    if not suffix:
        # An identical prompt: no prefill at all, the lease taken as it is.
        assert turn_new.stats.cached_tokens == len(turn_new.prompt) == 508
    _same_rounds(turn_new.rounds, turn_parent.rounds)
    _same_turn(turn_new, turn_parent)
    # Full and partial rejects both went through the resized bank.
    rejected_at = [
        event.get("rejected_at_depth")
        for event in turn_new.stats.events
        if "accepted_depths" in event
    ]
    assert 1 in rejected_at  # every draft rejected
    assert any(depth is not None and depth > 1 for depth in rejected_at)  # some kept
    # The next turn restores the lease and adopts its buffers on both sides.
    next_new, next_parent = new.turn([9, 10, 11, 12]), parent.turn([9, 10, 11, 12])
    assert next_new.stats.session_restore_mode == "reference_lease"
    assert next_new.admission["promotion"] == next_parent.admission["promotion"] == "adopted"
    _same_rounds(next_new.rounds, next_parent.rounds)
    _same_turn(next_new, next_parent)


@gpu
def test_a_turn_refused_for_memory_does_not_keep_later_turns_eager(tiny, lane):
    """The founder's 39 later turns: unequal layers after an eager turn are resized."""

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    # Less room than one layer's new buffers: both refuse, both verify eagerly.
    refused_new = new.turn([5, 6], room=300_000)
    refused_parent = parent.turn([5, 6], room=300_000)
    _eager(refused_new)
    _eager(refused_parent)
    _same_turn(refused_new, refused_parent)
    # The eager decode grew the stock buffers its own way: its first write
    # past 512 rows cut the keys and values to the 510 the prompt held and
    # added the 256-row step; the index keys doubled.
    assert refused_new.held == [(766, 1024, 256)] * 2
    # The next turn resizes that layout and returns to the compiled lane.
    back_new, back_parent = new.turn([9, 10, 11, 12]), parent.turn([9, 10, 11, 12])
    _compiled(back_new)
    assert back_new.promotions == [("adopted", 1024)] * 2
    assert back_parent.promotions == [("copied", 1024)] * 2
    assert back_new.admission["promotion"] == "resized"
    assert back_new.admission["held_rows"] is None
    kv = 2 * (2 * 32 * 2)  # keys and values: bytes per row of one layer
    assert back_new.admission["promotion_bytes"] == (1024 - 766) * kv + 1024 * kv
    _same_rounds(back_new.rounds, back_parent.rounds)
    _same_turn(back_new, back_parent)


@gpu
def test_index_keys_an_eager_turn_doubled_are_cut_for_a_bank_without_the_bucket(tiny, lane):
    """After an eager turn the index keys can be longer than the next bank: they are cut."""

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
        refused = session.turn([5, 6], room=300_000)
        _eager(refused)
        assert refused.held == [(766, 1024, 256)] * 2
    # Room for the 768-row bank (the bucket dropped) and not the 1,024-row
    # one: the keys and values grow to 768 rows, the index keys are cut to
    # 768 and the pooled blocks to 192. The parent gets the same 768-row bank
    # by copying, with room for that copy.
    back_new = new.turn([9, 10, 11, 12], room=300_000)
    back_parent = parent.turn([9, 10, 11, 12], room=600_000)
    _compiled(back_new)
    _compiled(back_parent)
    assert back_new.promotions == [("adopted", 768)] * 2
    assert back_parent.promotions == [("copied", 768)] * 2
    assert back_new.admission["promotion"] == "resized"
    assert back_new.admission["promotion_rows"] == 768
    assert back_new.admission["capacity_bucket"] == back_parent.admission["capacity_bucket"] == 0
    # keys and values (2 x 2 heads x 32 x bf16 a row), raw (32 x bf16 a row),
    # pooled (32 x bf16 a block): the first layer's new buffers, as the
    # second layer's cut more than pays for its own.
    new_layer = 768 * 2 * (2 * 32 * 2) + 768 * 32 * 2 + 192 * 32 * 2
    old_layer = 766 * 2 * (2 * 32 * 2) + 1024 * 32 * 2 + 256 * 32 * 2
    assert new_layer < old_layer
    assert back_new.admission["promotion_bytes"] == new_layer == 258_048
    _same_rounds(back_new.rounds, back_parent.rounds)
    _same_turn(back_new, back_parent)


@gpu
def test_pooled_keys_a_suffix_left_short_are_resized_before_the_bank_adopts(tiny, lane):
    """The review's counterexample: 508 held tokens and five more reach 513.
    The prefill grew the keys, values and raw index keys to the bank's 1,024
    rows; the pooled index keys still held every completed block in 128 and
    stayed there, and the bank grew them as it was built, billed 0."""

    pools: list[tuple] = []

    def observe(patch):
        real = TensorOffsetQSACache.from_qsa_cache.__func__

        def made(cls, entry, **kwargs):
            shape = tuple(entry.pooled.shape)
            bank = real(cls, entry, **kwargs)
            pools.append((shape, tuple(bank.pooled.shape), bank.pooled is entry.pooled))
            return bank

        patch.setattr(TensorOffsetQSACache, "from_qsa_cache", classmethod(made))

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    turn_new = new.turn([5, 6, 7, 8, 9], patches=[observe])
    turn_parent = parent.turn([5, 6, 7, 8, 9])

    _compiled(turn_new)
    _compiled(turn_parent)
    assert turn_new.admission["promotion"] == "resized"
    assert turn_new.admission["held_rows"] == 1024
    # Pooled blocks are 32 x bf16: 256 new against 128 old, layer by layer.
    assert turn_new.admission["promotion_bytes"] == 256 * 64 + (256 - 128) * 64 == 24_576
    # The bank takes the resized pooled buffers as they are.
    assert pools == [((1, 256, 32), (1, 256, 32), True)] * 2
    assert turn_new.promotions == [("adopted", 1024)] * 2
    # The parent's bank grew them itself: the same bits everywhere.
    assert turn_parent.promotions == [("adopted", 1024)] * 2
    _same_rounds(turn_new.rounds, turn_parent.rounds)
    _same_turn(turn_new, turn_parent)


@gpu
def test_a_resize_stopped_between_layers_keeps_the_conversation_and_finishes_next_turn(tiny, lane):
    """Memory taken between two layers: the turn falls back, counted, and the next resumes."""

    calls = {"n": 0}

    def second_layer_refused(patch):
        real = generation._qwen4_fixed_m4_layer_fits

        def layer_fits(rt, need, **kwargs):
            calls["n"] += 1
            return calls["n"] < 2 and real(rt, need, **kwargs)

        patch.setattr(generation, "_qwen4_fixed_m4_layer_fits", layer_fits)

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    stopped = new.turn([5, 6], patches=[second_layer_refused])
    snapshot = demotions.snapshot()
    refused = parent.turn([5, 6], room=300_000)

    assert calls["n"] == 2
    _eager(stopped)
    assert stopped.admission["promotion"] == "resized"
    assert stopped.admission["resize_layers_left"] == 1
    assert snapshot["counts"]["fixed_m4_lane_skipped"] == 1
    assert "layers left to resize (1)" in snapshot["reasons"]["fixed_m4_lane_skipped"]
    # Layer one was resized, layer two kept its buffers; the eager decode then
    # grew each its own way. The tokens are the refused turn's.
    assert stopped.held == [(1024, 1024, 256), (766, 1024, 256)]
    assert refused.held == [(766, 1024, 256)] * 2
    _same_turn(stopped, refused, capacity=False)

    back_new, back_parent = new.turn([9, 10, 11, 12]), parent.turn([9, 10, 11, 12])
    _compiled(back_new)
    assert back_new.promotions == [("adopted", 1024)] * 2
    assert back_new.admission["promotion"] == "resized"
    _same_rounds(back_new.rounds, back_parent.rounds)
    _same_turn(back_new, back_parent)


class _Interrupted(Exception):
    pass


@gpu
def test_a_resize_that_fails_keeps_the_only_copy_and_the_retry_is_the_same_turn(tiny, lane):
    """An exception during the resize (a failed allocation, an interrupt) reaches the caller."""

    calls = {"n": 0}

    def fails_on_the_second_layer(patch):
        def layer_fits(rt, need, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise _Interrupted()
            return True

        patch.setattr(generation, "_qwen4_fixed_m4_layer_fits", layer_fits)

    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    prompt = list(new.tokens) + [5, 6]
    with pytest.raises(_Interrupted):
        new.turn([5, 6], patches=[fails_on_the_second_layer])
    # The prompt's lease (banked before decode) is the conversation's only
    # copy: one layer resized, the other as it was, both holding the prompt.
    lease = new.bank.longest_prefix(prompt)
    assert lease is not None and lease.prefix_len == len(prompt) and lease.cache_ref is not None
    layers = [e for e in lease.cache_ref if isinstance(e, QSACache)]
    assert [TensorOffsetQSACache.held_rows(e) for e in layers] == [1024, 512]
    assert [e.offset for e in layers] == [len(prompt)] * 2

    retry = new.turn([], prompt=prompt, seed=SEED + len(prompt))
    uninterrupted = parent.turn([5, 6])
    assert retry.stats.session_restore_mode == "reference_lease"
    assert retry.stats.cached_tokens == len(prompt)
    _compiled(retry)
    assert retry.promotions == [("adopted", 1024)] * 2
    assert retry.admission["promotion"] == "resized" and retry.admission["held_rows"] is None
    _same_rounds(retry.rounds, uninterrupted.rounds)
    _same_turn(retry, uninterrupted)
    after_new, after_parent = new.turn([9, 10, 11, 12]), parent.turn([9, 10, 11, 12])
    _same_turn(after_new, after_parent)


def _second_layer_write_raises(error, calls):
    """A patch: writing the second layer's new buffers raises ``error``
    (after that layer was admitted, where an allocation fails)."""

    def apply(patch):
        real = one_copy._resize_layer

        def resizing(entry, rows):
            calls.append(rows)
            if len(calls) != 2:
                return real(entry, rows)

            def failing(*arrays):
                raise error

            with patch.context() as inner:
                inner.setattr(mx, "eval", failing)
                return real(entry, rows)

        patch.setattr(one_copy, "_resize_layer", resizing)

    return apply


@gpu
def test_a_resize_the_memory_refuses_answers_on_the_eager_verifier(tiny, lane):
    """The 2026-10-01 review: an allocation failure while the second layer's
    buffers were written failed a request whose eager answer fits."""

    calls: list[int] = []
    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    failed = new.turn([5, 6], patches=[_second_layer_write_raises(
        RuntimeError("[metal::malloc] Unable to allocate 172032 bytes"), calls,
    )])
    snapshot = demotions.snapshot()
    eager = parent.turn([5, 6], room=300_000)

    assert calls == [1024, 1024]
    _eager(failed)
    assert failed.admission["promotion"] == "resized"
    assert failed.admission["resize_layers_left"] == 1
    assert "Unable to allocate" in failed.admission["resize_error"]
    assert snapshot["counts"]["fixed_m4_lane_skipped"] == 1
    assert "failed to allocate a layer (1 left to resize)" in (
        snapshot["reasons"]["fixed_m4_lane_skipped"]
    )
    # The first layer kept its new buffers, the second its old ones, which
    # the eager decode then grew its own way. The answer is the eager one.
    assert failed.held == [(1024, 1024, 256), (766, 1024, 256)]
    _same_turn(failed, eager, capacity=False)
    # The next turn resizes the second layer and is compiled again.
    back_new, back_parent = new.turn([9, 10, 11, 12]), parent.turn([9, 10, 11, 12])
    _compiled(back_new)
    assert back_new.promotions == [("adopted", 1024)] * 2
    assert back_new.admission["promotion"] == "resized"
    _same_rounds(back_new.rounds, back_parent.rounds)
    _same_turn(back_new, back_parent)


class _Cancelled(BaseException):
    pass


@gpu
@pytest.mark.parametrize(
    "error",
    [RuntimeError("[metal] shape mismatch"), ValueError("bad rows"), _Cancelled()],
    ids=["runtime_error", "value_error", "cancellation"],
)
def test_other_failures_in_a_resize_reach_the_caller_and_the_retry_is_the_same_turn(
    tiny, lane, error,
):
    calls: list[int] = []
    new, parent = _Session(tiny, lane), _Session(tiny, lane, parent=True)
    for session in (new, parent):
        session.opening()
    prompt = list(new.tokens) + [5, 6]
    with pytest.raises(type(error)):
        new.turn([5, 6], patches=[_second_layer_write_raises(error, calls)])
    assert calls == [1024, 1024]
    lease = new.bank.longest_prefix(prompt)
    assert lease is not None and lease.prefix_len == len(prompt)
    layers = [e for e in lease.cache_ref if isinstance(e, QSACache)]
    assert [TensorOffsetQSACache.held_rows(e) for e in layers] == [1024, 512]
    retry = new.turn([], prompt=prompt, seed=SEED + len(prompt))
    uninterrupted = parent.turn([5, 6])
    assert retry.stats.cached_tokens == len(prompt)
    _compiled(retry)
    _same_rounds(retry.rounds, uninterrupted.rounds)
    _same_turn(retry, uninterrupted)


PAD, IMAGE_TOKENS, GRID = 120, 16, (1, 8, 8)


@gpu
def test_an_image_conversation_keeps_its_positions_through_the_resize(tiny, lane):
    """A screenshot in the history: the rotary delta and every bit are the copied bank's."""

    from mtplx.vision.mrope import build_mrope_positions
    from mtplx.vision.splice import VisionSplice

    mx.random.seed(99)
    rows = mx.random.normal((IMAGE_TOKENS, 64)).astype(mx.bfloat16)
    mx.eval(rows)

    def splice_for(ids):
        table, delta = build_mrope_positions(
            ids, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
        )
        return VisionSplice(
            image_pad_token_id=PAD, embeddings=rows, image_digests=(1,),
            pad_counts=(IMAGE_TOKENS,), image_grids=(GRID,),
            mrope_table=mx.array(table), mrope_delta=int(delta),
        )

    opening = [3, 5, 7, 9, 11, 13] + [PAD] * IMAGE_TOKENS + [(7 * i + 3) % 100 for i in range(462)]
    new = _Session(tiny, lane, splice_for=splice_for)
    parent = _Session(tiny, lane, parent=True, splice_for=splice_for)
    # Seeds whose answers hold no image placeholder, so the next turn's
    # position table describes one image (a chat template never emits one).
    for session in (new, parent):
        first = session.turn(opening, seed=SEED)
        assert PAD not in first.tokens and first.held == [(512, 512, 128)] * 2
    turn_new, turn_parent = new.turn([5, 6], seed=SEED + 1), parent.turn([5, 6], seed=SEED + 1)
    assert PAD not in turn_new.tokens

    _compiled(turn_new)
    assert turn_new.promotions == [("adopted", 1024)] * 2
    assert turn_parent.promotions == [("copied", 1024)] * 2
    for record in (turn_new, turn_parent):
        assert record.admission["positions"] == "vision_delta"
        assert record.admission["rope_delta"] == 4 - IMAGE_TOKENS
        assert record.bank["fixed_m4"]["rope_delta"] == 4 - IMAGE_TOKENS
        assert record.stats.session_restore_mode == "reference_lease"
    assert turn_new.admission["promotion"] == "resized"
    _same_rounds(turn_new.rounds, turn_parent.rounds)
    _same_turn(turn_new, turn_parent)


def _walk():
    rng = np.random.default_rng(7)
    return [int(token) for token in rng.integers(1, 128, size=300)]


@gpu
def test_compiled_copy_windows_run_on_the_resized_bank(tiny, lane):
    """MTPLX_FIXED_M4_COPY_WINDOWS=1 on the transition turn, every round a copy round.

    Turn 0 leaves the walk in 512-row buffers. Turn 1 quotes walk[45:50] and
    asks for 200 tokens: its prompt fits the 512 rows, its 204-row reserve
    does not. The primary and each copy round's acceptance are forced to
    continue the walk after the real samplers ran, as in
    tests/test_qwen4_fixed_m4_copy_windows.py: odd rounds accept the whole
    block, even rounds half of it and take the next walk token.
    """

    walk = _walk()
    lane.setenv("MTPLX_CONTEXT_COPY", "1")
    lane.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    lane.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "32")
    lane.setenv("MTPLX_FIXED_M4_COPY_WINDOWS", "1")
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)

    def forced(cursor, copy_rounds):
        def apply(patch):
            real_sample = generation._sample_from_logits
            real_accept = generation._point_mass_block_accept

            def primary(*args, **kwargs):
                _token, distribution = real_sample(*args, **kwargs)
                token = walk[cursor[0]]
                cursor[0] += 1
                return token, distribution

            def accept(block_logits, block, sampler, rng):
                real_accept(block_logits, block, sampler, rng)
                assert list(block) == walk[cursor[0] : cursor[0] + len(block)]
                copy_rounds.append((len(block) + 1, _bits(block_logits)))
                keep = len(block) if len(copy_rounds) % 2 else len(block) // 2
                cursor[0] += keep
                if keep == len(block):
                    return keep, None
                cursor[0] += 1
                return keep, walk[cursor[0] - 1]

            patch.setattr(generation, "_sample_from_logits", primary)
            patch.setattr(generation, "_point_mass_block_accept", accept)

        return apply

    results = []
    for parent in (False, True):
        # Each arm proves its own windows, as a fresh process would.
        lane.setitem(graphbank._FIXED_M4_COPY_WINDOWS, "proven", set())
        session = _Session(tiny, lane, parent=parent)
        first = session.turn(walk, max_tokens=8)
        assert first.held == [(512, 512, 128)] * 2
        cursor, copy_rounds = [50], []
        turn = session.turn(walk[45:50], max_tokens=200, patches=[forced(cursor, copy_rounds)])
        results.append((turn, copy_rounds))
    (turn_new, rounds_new), (turn_parent, rounds_parent) = results

    assert turn_new.admission["engaged"] is True and turn_new.bank["fixed_m4"]["installed"]
    assert turn_new.promotions == [("adopted", 1024)] * 2
    assert turn_parent.promotions == [("copied", 1024)] * 2
    assert turn_new.admission["promotion"] == "resized"
    assert turn_new.admission["promotion_rows"] == 1024
    windows = turn_new.bank["fixed_m4_copy_windows"]
    assert sum(windows["compiled"].values()) > 0
    parent_windows = turn_parent.bank["fixed_m4_copy_windows"]
    assert windows["compiled"] == parent_windows["compiled"]
    assert windows["eager"] == parent_windows["eager"]
    assert [w for w, _ in rounds_new] == [w for w, _ in rounds_parent] and len(rounds_new) >= 4
    for (width, got), (_w, want) in zip(rounds_new, rounds_parent):
        assert got.shape == want.shape and np.array_equal(got, want), width
    _same_rounds(turn_new.rounds, turn_parent.rounds)
    _same_turn(turn_new, turn_parent)


if __name__ == "__main__" and sys.argv[1:] == ["--isolated", "_resize_transient"]:
    print(json.dumps(_resize_transient()))
