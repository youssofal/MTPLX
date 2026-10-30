"""One copy of the conversation (mtplx/one_copy.py).

On 2026-09-29 a 138K-token Pi session held its conversation two or three
times: the session bank's snapshot views, the live cache the next request
copied on its first write (a view blocks MLX's in-place write), and the
fixed-M4 verifier's padded bank. These tests pin the one-copy store:

- the prefill sizes every QSA buffer once to the rows the verifier's bank
  plans, so the bank adopts those very buffers;
- the bank hands them back whole at the end of the request;
- the session bank keeps the conversation as a lease with recurrent anchors
  and never as views; a lease an answer decoded past rewinds to its anchor;
- a restore validates before it takes a lease, so a failed restore can never
  drop the only live copy;
- and generation produces exactly the tokens the copying path produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.cache_state import snapshot_untrimmable_cache
from mtplx.graphbank import TensorOffsetQSACache
from mtplx.models.qwen4_exp import QSACache, qsa_rows_target
from mtplx.one_copy import held_qsa_rows, prompt_lease_fields
from mtplx.session_bank import SessionBank, _exact_restore_serves
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, _bits, lane, pack

RUNTIME = SimpleNamespace(
    model_path="tiny",
    mtp_enabled=True,
    make_cache=lambda: [ArraysCache(size=2), QSACache(4), ArraysCache(size=2)],
    make_mtp_cache=lambda: [QSACache(4)],
)


def _address(value: mx.array) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint16), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _qsa(offset: int, rows: int | None = None, *, dtype=mx.bfloat16, seed: int = 0) -> QSACache:
    """A stock QSA layer holding ``offset`` tokens in buffers of ``rows`` rows."""

    rows = offset if rows is None else rows
    mx.random.seed(seed)
    entry = QSACache(4)
    entry.indexer_budget = 8
    entry.kv.keys = mx.random.normal((1, 2, rows, 16)).astype(dtype)
    entry.kv.values = mx.random.normal((1, 2, rows, 16)).astype(dtype)
    entry.kv.offset = offset
    entry.raw_keys = mx.random.normal((1, rows, 8)).astype(dtype)
    entry.pooled = mx.random.normal((1, rows // 4, 8)).astype(dtype)
    entry.pooled_len = offset // 4
    mx.eval(entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled)
    return entry


def _recurrent(seed: int) -> ArraysCache:
    mx.random.seed(seed)
    entry = ArraysCache(size=2)
    entry[0] = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    entry[1] = mx.random.normal((1, 2, 4, 4)).astype(mx.float32)
    mx.eval(*entry.cache)
    return entry


@pytest.fixture()
def gather(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    monkeypatch.delenv("MTPLX_QSA_M4_FUSED_KV_GATHER", raising=False)
    monkeypatch.delenv("MTPLX_QSA_SCORE_TILE_ROWS", raising=False)
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    return monkeypatch


# -- sizing ---------------------------------------------------------------------


def test_rows_target_sizes_every_qsa_buffer_once():
    entry = QSACache(4)
    keys = mx.ones((1, 2, 8, 16), dtype=mx.bfloat16)
    raw = mx.full((1, 8, 8), 2.0, dtype=mx.bfloat16)
    with qsa_rows_target(512):
        entry.write_raw(raw)
        entry.write_pooled(mx.full((1, 2, 8), 3.0, dtype=mx.bfloat16), 0, 2)
        entry.kv.update_and_fetch(keys, keys)
        assert entry.kv.keys.shape[2] == entry.raw_keys.shape[1] == 512
        assert entry.pooled.shape[1] == 128
        # A second write inside the target does not grow anything.
        entry.write_raw(raw)
        entry.kv.update_and_fetch(keys * 2, keys * 2)
        assert entry.kv.keys.shape[2] == 512 and entry.kv.offset == 16
    assert TensorOffsetQSACache.held_rows(entry) == 512
    assert bool(mx.all(entry.kv.keys[..., :8, :] == 1).item())
    assert bool(mx.all(entry.kv.keys[..., 8:16, :] == 2).item())
    # Outside the scope the stock growth runs, one step at a time.
    plain = QSACache(4)
    plain.kv.update_and_fetch(keys, keys)
    assert plain.kv.keys.shape[2] == 256


def test_rows_target_grows_a_restored_buffer_once_and_keeps_its_history():
    entry = _qsa(300, 300, seed=3)
    before = _bits(entry.kv.keys[..., :300, :])
    new = mx.zeros((1, 2, 4, 16), dtype=mx.bfloat16)
    with qsa_rows_target(1024):
        entry.kv.update_and_fetch(new, new)
    assert entry.kv.keys.shape[2] == 1024 and entry.kv.offset == 304
    assert np.array_equal(_bits(entry.kv.keys[..., :300, :]), before)


# -- adoption -------------------------------------------------------------------


def _plan_rows(offset: int, reserve: int, bucket: int) -> int:
    return TensorOffsetQSACache._bank_capacity(
        offset + reserve, 4, 256, rows_gather=True, bucket=bucket
    )


def test_bank_adopts_buffers_sized_for_it(gather):
    rows = _plan_rows(300, 100, 512)
    assert rows == 512
    entry = _qsa(300, rows)
    arrays = (entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled)
    bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
    assert bank.promotion == "adopted"
    assert bank.capacity == rows
    assert all(a is b for a, b in zip(arrays, (bank.kv.keys, bank.kv.values, bank.raw_keys, bank.pooled)))


def test_bank_copies_buffers_that_do_not_fit_its_plan(gather):
    entry = _qsa(300, 304)
    bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
    assert bank.promotion == "copied"
    assert bank.capacity == 512
    assert bank.kv.keys is not entry.kv.keys


@pytest.mark.parametrize(
    "held, planned, rows_gather, bucket, adopted",
    [
        (512, 512, True, 512, 512),
        (1024, 512, True, 512, 1024),  # one bucket larger: values unchanged
        (1536, 512, True, 512, None),  # two buckets: the selector's work would grow
        (1024, 512, True, 0, None),  # unbucketed: the exact width is the arithmetic
        (768, 512, False, 512, None),  # dense lane: the exact width is the arithmetic
        (768, 768, False, 512, 768),
        (1000, 512, True, 512, None),  # not page aligned
        (256, 512, True, 512, None),  # too small
    ],
)
def test_adoption_rule(held, planned, rows_gather, bucket, adopted):
    assert TensorOffsetQSACache.adoptable_capacity(
        held, planned, ratio=4, kv_step=256, rows_gather=rows_gather, bucket=bucket,
    ) == adopted


def test_adopted_bank_writes_in_place(gather):
    entry = _qsa(300, 512)
    bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
    del entry
    before = _address(bank.kv.keys)
    rows = mx.ones((1, 2, 4, 16), dtype=mx.bfloat16)
    bank.kv.cache[0] = mx.slice_update(bank.kv.cache[0], rows, bank.kv.cache[2], axes=(2,))
    assert _address(bank.kv.keys) == before


def test_a_view_held_elsewhere_forces_the_copy_the_lease_avoids(gather):
    """Why the session bank may not keep views of the live buffers."""

    entry = _qsa(300, 512)
    bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
    del entry
    before = _address(bank.kv.keys)
    snapshot_view = bank.kv.keys[..., :300, :]
    rows = mx.ones((1, 2, 4, 16), dtype=mx.bfloat16)
    bank.kv.cache[0] = mx.slice_update(bank.kv.cache[0], rows, bank.kv.cache[2], axes=(2,))
    assert _address(bank.kv.keys) != before
    del snapshot_view


def test_keep_capacity_demote_hands_back_the_same_buffers(gather):
    entry = _qsa(300, 512)
    bank = TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=100, capacity_bucket=512)
    arrays = (bank.kv.keys, bank.kv.values, bank.raw_keys, bank.pooled)
    stock = bank.demote(keep_capacity=True)
    assert isinstance(stock, QSACache)
    assert stock.offset == 300 and stock.pooled_len == 75
    assert all(a is b for a, b in zip(arrays, (stock.kv.keys, stock.kv.values, stock.raw_keys, stock.pooled)))
    keys, values, raw, pooled = stock.state
    assert keys.shape[2] == 300 and raw.shape[1] == 300 and pooled.shape[1] == 75
    # The next request's bank adopts them again.
    again = TensorOffsetQSACache.from_qsa_cache(stock, reserve_tokens=100, capacity_bucket=512)
    assert again.promotion == "adopted" and again.kv.keys is arrays[0]


def test_held_rows_needs_every_layer_to_agree():
    assert held_qsa_rows([_qsa(20, 64), _recurrent(1), _qsa(20, 64)]) == 64
    assert held_qsa_rows([_qsa(20, 64), _qsa(20, 128)]) is None
    assert held_qsa_rows([_recurrent(1)]) is None


# -- the session bank ---------------------------------------------------------------


def _conversation(offset: int, rows: int, *, seed: int = 0):
    trunk = [_recurrent(seed), _qsa(offset, rows, seed=seed + 1), _recurrent(seed + 2)]
    mtp = [_qsa(offset - 1, rows, seed=seed + 3)]
    return trunk, mtp


def _advance(trunk, mtp, tokens: int, *, seed: int = 9) -> None:
    """What an answer decoding into the leased cache does to it."""

    mx.random.seed(seed)
    for entry in trunk + mtp:
        if isinstance(entry, QSACache):
            rows = mx.random.normal((1, 2, tokens, 16)).astype(mx.bfloat16)
            entry.kv.update_and_fetch(rows, rows)
        else:
            entry[0] = mx.random.normal(entry[0].shape).astype(entry[0].dtype)
            entry[1] = mx.random.normal(entry[1].shape).astype(entry[1].dtype)
    mx.eval(*[leaf for entry in trunk + mtp for leaf in _leaves(entry)])


def _leaves(entry) -> list[mx.array]:
    if isinstance(entry, QSACache):
        return [entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled]
    return [leaf for leaf in entry.cache if leaf is not None]


def _prompt_lease(bank: SessionBank, trunk, mtp, tokens: list[int], **extra):
    fields = prompt_lease_fields(
        trunk, committed_mtp_cache=mtp, hidden=mx.ones((1, 1, 8)),
        prompt_len=len(tokens), boundaries=extra.pop("boundaries", []),
    )
    return bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), session_id="s", snapshot_epoch=len(tokens),
        mtp_snapshot_epoch=len(tokens), **fields, **extra,
    )


def test_one_copy_put_is_a_lease_without_views(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    entry = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)), cache=trunk, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s",
        mtp_history_cache_ref=mtp, snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    assert entry.live_ref_only and entry.cache_ref is trunk and entry.mtp_history_cache_ref is mtp
    assert all(state is None for state in entry.cache_snapshot.states)
    assert entry.mtp_history_snapshot is None
    assert entry.lease_kv_offset == 40 and entry.lease_mtp_offset == 39
    assert entry.held_nbytes >= sum(int(leaf.nbytes) for layer in trunk for leaf in _leaves(layer))


def test_the_copying_store_is_kept_behind_its_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, _mtp = _conversation(40, 64)
    entry = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)), cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="s", snapshot_epoch=40,
    )
    assert not entry.live_ref_only
    assert entry.cache_snapshot.states[1] is not None


@pytest.mark.parametrize("extends", [False, True])
def test_an_advanced_prompt_lease_restores_through_its_anchor(monkeypatch, extends):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    anchor = snapshot_untrimmable_cache(trunk)
    anchor_bits = [
        [_bits(leaf) for leaf in state] if state is not None else None
        for state in anchor.states
    ]
    prompt = list(range(40))
    entry = _prompt_lease(bank, trunk, mtp, prompt)
    _advance(trunk, mtp, 6)  # the answer decodes past the lease
    assert trunk[1].offset == 46
    lookup = prompt + [900, 901] if extends else prompt
    restored = bank.restore(RUNTIME, lookup, mode="reference", session_id="s")
    assert restored is not None and restored.restore_mode == "reference_lease"
    assert restored.cache is trunk and entry.cache_ref is None
    # Both lookup shapes land at the entry's end, where a clone lands: an
    # identical prompt decodes from the entry's stored logits
    # (tests/test_one_copy_cancelled_answer.py).
    assert trunk[1].offset == 40
    for state, bits in zip((layer.cache if not isinstance(layer, QSACache) else None for layer in trunk), anchor_bits):
        if bits is None:
            continue
        assert [np.array_equal(_bits(leaf), want) for leaf, want in zip(state, bits)] == [True, True]
    assert restored.mtp_history_cache is mtp and mtp[0].offset == 39


def test_an_advanced_lease_without_its_anchor_is_not_served_and_stays(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    prompt = list(range(40))
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="s", mtp_history_cache_ref=mtp,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    _advance(trunk, mtp, 3)
    assert not _exact_restore_serves(entry, {})
    assert bank.restore(RUNTIME, prompt + [7], mode="reference", session_id="s") is None
    assert entry.cache_ref is trunk and trunk[1].offset == 43


def test_restore_validates_before_it_takes_the_lease(monkeypatch):
    """Stage 0 item 5: a failed trim used to drop the only live copy."""

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    prompt = list(range(40))
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="s", mtp_history_cache_ref=mtp,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    mtp[0].kv.offset = 10  # a draft history that cannot reach the entry
    assert bank.restore(RUNTIME, prompt + [7], mode="reference", session_id="s") is None
    assert entry.cache_ref is trunk and entry.mtp_history_cache_ref is mtp
    assert trunk[1].offset == 40
    served = bank.restore_entry_prefix_cache(RUNTIME, entry, 40, mode="reference")
    assert served is None
    assert entry.cache_ref is trunk and trunk[1].offset == 40


def test_a_boundary_restore_rewinds_an_advanced_lease(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(24, 64)
    early = (24, snapshot_untrimmable_cache(trunk), mx.ones((1, 1, 8)))
    early_bits = _bits(early[1].states[0][0])
    _advance(trunk, mtp, 16)  # the prompt continues to 40
    prompt = list(range(40))
    entry = _prompt_lease(bank, trunk, mtp, prompt, boundaries=[early])
    _advance(trunk, mtp, 5, seed=11)  # and an answer decodes past it
    served = bank.restore_entry_prefix_cache(RUNTIME, entry, 30, mode="reference")
    assert served is not None
    cache, mtp_cache, mode, restore_point, _hidden = served
    assert mode == "reference_lease" and restore_point == 24
    assert cache[1].offset == 24
    assert np.array_equal(_bits(cache[0][0]), early_bits)
    # The draft history went back to the lease (39 rows), then by the gap.
    assert mtp_cache[0].offset == 39 - (40 - 24)


def test_the_answer_lease_keeps_the_prompt_logits_for_an_identical_retry(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    prompt = list(range(40))
    prompt_entry = _prompt_lease(bank, trunk, mtp, prompt)
    _advance(trunk, mtp, 6)
    answer = bank.put(
        runtime=RUNTIME, token_ids=prompt + list(range(100, 106)), cache=trunk,
        logits=mx.zeros((1, 16)), hidden=mx.ones((1, 1, 8)), keep_live_ref=True,
        session_id="s", mtp_history_cache_ref=mtp, snapshot_epoch=46, mtp_snapshot_epoch=46,
    )
    assert len(bank) == 1
    position, logits = answer.boundary_logits
    assert position == 40 and logits is prompt_entry.logits
    served = bank.restore_entry_prefix_cache(RUNTIME, answer, 40, mode="reference")
    _cache, _mtp, _mode, restore_point, _hidden = served
    assert restore_point == 40 and trunk[1].offset == 40


def test_the_spill_skips_a_lease_the_answer_ran_past(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    spilled = []
    bank.cold_tier = SimpleNamespace(spill_entry=lambda view, **kw: spilled.append(view) or True)
    trunk, mtp = _conversation(40, 64)
    entry = _prompt_lease(bank, trunk, mtp, list(range(40)))
    _advance(trunk, mtp, 2)
    assert bank.run_live_ref_spill(entry.token_ids, entry.snapshot_epoch) is False
    assert spilled == []
    assert bank.eviction_log[-1]["reason"] == "ssd_spill_lease_advanced"


def test_another_session_gets_a_copy_and_the_owner_keeps_its_lease(monkeypatch):
    """A subagent or a fork sharing a prefix must not take a conversation's only copy."""

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    keys_bits = _bits(trunk[1].kv.keys[..., :40, :])
    prompt = list(range(40))
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="owner", mtp_history_cache_ref=mtp,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    other = bank.restore(RUNTIME, prompt + [5], mode="reference", session_id="other")
    assert other is not None and other.restore_mode == "clone"
    assert other.cache is not trunk and other.mtp_history_cache is not mtp
    assert other.cache[1].offset == 40 and other.mtp_history_cache[0].offset == 39
    assert np.array_equal(_bits(other.cache[1].kv.keys[..., :40, :]), keys_bits)
    assert entry.cache_ref is trunk and entry.mtp_history_cache_ref is mtp
    # The borrower writes into buffers of its own; the owner's stay untouched.
    rows = mx.ones((1, 2, 1, 16), dtype=mx.bfloat16)
    other.cache[1].kv.update_and_fetch(rows, rows)
    mx.eval(other.cache[1].kv.keys)
    assert np.array_equal(_bits(trunk[1].kv.keys[..., :40, :]), keys_bits)
    owner = bank.restore(RUNTIME, prompt + [6], mode="reference", session_id="owner")
    assert owner is not None and owner.restore_mode == "reference_lease"
    assert owner.cache is trunk and entry.cache_ref is None
    # The same through the boundary path.
    trunk2, mtp2 = _conversation(40, 64, seed=20)
    anchored = _prompt_lease(bank, trunk2, mtp2, list(range(100, 140)),
                             boundaries=[(24, snapshot_untrimmable_cache(trunk2), None)])
    anchored.session_id = "owner2"
    served = bank.restore_entry_prefix_cache(
        RUNTIME, anchored, 30, mode="reference", session_id="other",
    )
    assert served is not None and served[2] == "clone" and served[0] is not trunk2
    assert anchored.cache_ref is trunk2


def test_taking_a_lease_voids_every_other_lease_on_the_same_cache(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    trunk, mtp = _conversation(40, 64)
    first = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)), cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, mtp_history_cache_ref=mtp, snapshot_epoch=40,
        mtp_snapshot_epoch=40,
    )
    trunk[1].kv.offset = 41
    mtp[0].kv.offset = 40
    second = bank.put(
        runtime=RUNTIME, token_ids=list(range(40)) + [1], cache=trunk,
        logits=mx.zeros((1, 16)), hidden=None, keep_live_ref=True,
        mtp_history_cache_ref=mtp, snapshot_epoch=41, mtp_snapshot_epoch=41,
    )
    assert first.cache_ref is trunk and second.cache_ref is trunk
    restored = bank.restore(RUNTIME, list(range(40)) + [1, 2], mode="reference")
    assert restored is not None and restored.entry is second
    assert first.cache_ref is None and first.mtp_history_cache_ref is None


# -- generation -----------------------------------------------------------------------


def _session_turns(pack, monkeypatch, *, one_copy: bool):
    """Two turns of one session on the compiled fixed-M4 lane, banked like the server does."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    monkeypatch.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    promotions: list[str] = []
    real_from = TensorOffsetQSACache.from_qsa_cache.__func__

    def recording(cls, entry, **kwargs):
        bank = real_from(cls, entry, **kwargs)
        promotions.append(bank.promotion)
        return bank

    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    prompt = [(7 * i + 3) % 128 for i in range(48)]
    turns = []
    with monkeypatch.context() as patch:
        patch.setattr(TensorOffsetQSACache, "from_qsa_cache", classmethod(recording))
        for turn in range(2):
            turns.append(_one_turn(rt, bank, prompt, turn, one_copy, hidden_variant))
            prompt = list(prompt) + list(turns[-1][0]) + [5, 6, 7, 8]
    return turns, promotions


def _one_turn(rt, bank, prompt, turn, one_copy, hidden_variant):
    if True:
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED + turn, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True, session_bank=bank,
            session_id="s", session_restore_mode="reference",
            commit_prompt_state_to_bank=True,
        )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        tokens = list(prompt) + list(result.tokens)
        metadata = (
            {"mtp_history_cache_ref": final.final_committed_mtp_cache}
            if one_copy
            else {"mtp_history_snapshot": generation.snapshot_cache(final.final_committed_mtp_cache)}
        )
        entry = bank.put(
            runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=True,
            session_id="s", mtp_history_policy=final.mtp_history_policy,
            hidden_variant=hidden_variant,
            snapshot_epoch=len(tokens), mtp_snapshot_epoch=len(tokens), **metadata,
        )
        return list(result.tokens), entry, result


def test_one_copy_generation_matches_the_copying_path_and_holds_one_copy(pack, lane):
    one, one_promotions = _session_turns(pack, lane, one_copy=True)
    two, _ = _session_turns(pack, lane, one_copy=False)
    assert [tokens for tokens, _e, _r in one] == [tokens for tokens, _e, _r in two]
    # Every promotion of the one-copy run took the prefill's own buffers.
    assert one_promotions and set(one_promotions) == {"adopted"}
    for _tokens, entry, _result in one:
        assert entry.live_ref_only
        assert all(state is None for state in entry.cache_snapshot.states)
        assert entry.mtp_history_snapshot is None
    second = one[1][2]
    assert second.stats.graphbank["compiled_verify"]["fixed_m4"]["installed"]
    # The second turn resumed the first turn's lease: no prefill of its history.
    for run in (one, two):
        assert run[1][2].stats.cached_tokens == len(run[0][1].token_ids)
    assert second.stats.session_restore_mode == "reference_lease"
