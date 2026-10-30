"""A conversation restored from the SSD tier below an entry's end uses its own block keys (#635).

Flash-Next's full-attention layers keep three streams in one ``QSACache``: the
attention keys and values, the indexer's raw key per token, and one pooled
key per four-token block that the indexer scores to pick the blocks a query
attends. The cold tier stores the keys and values in 256-token blocks, so a
restore below an entry's end (a new conversation sharing an older one's
system prompt and tools, a compacted or edited history) decodes them only up
to the restore point. The raw and pooled streams are stored whole and come
back at the stored entry's full length.

Before 2.12.3 the cache took its count of valid pooled blocks from that
table's length: the next forward found every block of the new tokens already
pooled, computed none, and the indexer scored the new conversation's queries
against the stored conversation's block keys. The answer could lose the
request and cite facts from another conversation, and the next commit stored
that table again, so every descendant kept it. The lookup prefers the
longest stored row among equal matches, so the donor was usually a much
longer conversation, which is also what made the one-copy resize refuse the
turn (#592, ``test_one_copy_stale_pooled_frontier.py``).

Everything below runs the shipped restore: the session bank, the cold tier
on disk, the tensor codec at its 256-token block, and a real Flash-Next
attention layer on the CPU (the bit-exact surface). The recurrent layers
are a stand-in restored from their boundary, as a GDN layer is.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mtplx import one_copy
from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot, restore_cache
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs
from mtplx.session_bank import SessionBank

RATIO = 4  # Flash-Next's indexer_compress_ratio
BLOCK = 256  # the cold tier's KV block and the bank's block-prefix quantum, as shipped
SHARED = 2 * BLOCK  # tokens the new conversation shares with the stored one
DONOR_TAIL = 2 * BLOCK  # the stored conversation's tokens past the shared prefix
SUFFIX = 64  # the new conversation's own tokens after the shared prefix
TEMPLATE = "template"
POLICY = "policy"


def _args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        # Under the codec's split width (2 x 256), as Flash-Next's 128 is:
        # the raw and pooled streams are stored as whole tensors.
        indexer_head_dim=16,
        indexer_budget=16,
        indexer_compress_ratio=RATIO,
    )


@pytest.fixture()
def attn():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(previous)


class _Recurrent:
    """A linear-attention layer's slot in the trunk cache: not trimmable, so a
    restore below an entry's end takes its state from the boundary there."""

    def __init__(self, value: float = 0.0):
        self.state = [mx.full((2, 2), value), None]
        self.meta_state = ("owned_recurrent_state", "persistent_eval")

    def is_trimmable(self) -> bool:
        return False

    def replace_state(self, value) -> None:
        self.state = list(value)


class _Runtime:
    model_path = Path("/tmp/flash-next-test-model")
    mtp_enabled = False

    def make_cache(self):
        return [QSACache(RATIO), _Recurrent()]

    def make_mtp_cache(self):
        return []


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


# One hidden row per token id: the same ids reach the layer as the same rows.
SHARED_IDS = list(range(SHARED))
DONOR_IDS = [10_000 + i for i in range(DONOR_TAIL)]
SUFFIX_IDS = [20_000 + i for i in range(SUFFIX)]
NEXT_IDS = [30_000 + i for i in range(8)]


@pytest.fixture()
def rows():
    return {
        "shared": _hidden(SHARED, seed=1),
        "donor": _hidden(DONOR_TAIL, seed=2),
        "suffix": _hidden(SUFFIX, seed=3),
        "next": _hidden(len(NEXT_IDS), seed=4),
    }


def _boundary(value: float) -> CacheSnapshot:
    return CacheSnapshot(
        states=(None, [mx.full((2, 2), value), None]),
        meta_states=(None, ("owned_recurrent_state", "persistent_eval")),
    )


def _tier(tmp_path) -> SessionBankColdTier:
    tier = SessionBankColdTier(base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=1)
    # The configured cap alone sizes the store, whatever this machine's free
    # disk is (the tier's own fallback when free space cannot be read).
    tier._free_disk_bytes = lambda: None
    return tier


def _bank(tier) -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30, cold_tier=tier,
    )


def _persist(bank, tier, *, tokens, cache, boundaries, session_id, policy=POLICY) -> None:
    """Bank ``cache`` for ``tokens`` and let the cold tier write it, as a commit does."""

    entry = bank.put(
        runtime=_Runtime(),
        token_ids=list(tokens),
        cache=cache,
        logits=mx.zeros((1, 8)),
        hidden=None,
        session_id=session_id,
        template_hash=TEMPLATE,
        policy_fingerprint=policy,
        snapshot_epoch=len(tokens),
        gdn_boundaries=boundaries,
    )
    assert entry is not None
    assert tier.flush(timeout_s=10.0) is True
    bank.clear()


def _stored_conversation(attn, rows, bank, tier, *, policy=POLICY) -> None:
    """An older, longer conversation on the SSD tier, with a boundary where the new one diverges."""

    donor = QSACache(RATIO)
    attn(rows["shared"], donor)
    attn(rows["donor"], donor)
    assert donor.offset == SHARED + DONOR_TAIL
    _persist(
        bank, tier,
        tokens=SHARED_IDS + DONOR_IDS,
        cache=[donor, _Recurrent(9.0)],
        boundaries=[(SHARED, _boundary(7.0), mx.zeros((1, 1, 4)))],
        session_id="older-conversation",
        policy=policy,
    )


def _candidates(bank, *, policy=POLICY):
    runtime = _Runtime()
    return bank.near_prefix_candidates(
        SHARED_IDS + SUFFIX_IDS,
        block_size=BLOCK,
        block_min_matched_tokens=SHARED,
        allow_block_prefix=True,
        model_path=str(runtime.model_path),
        mtp_enabled=runtime.mtp_enabled,
        template_hash=TEMPLATE,
        policy_fingerprint=policy,
    )


def _restore_below_the_end(bank):
    """The new conversation's prompt restored from the SSD tier at the shared prefix."""

    runtime = _Runtime()
    candidates = _candidates(bank)
    assert candidates, bank.last_prefix_diagnostic
    entry, matched = candidates[0]
    assert getattr(entry, "cache_source", None) == "ssd"
    assert entry.prefix_len == SHARED + DONOR_TAIL
    # The cold decoder materialized the keys and values only to the boundary.
    assert entry.cache_snapshot_prefix_len == SHARED
    restored = bank.restore_entry_prefix_cache(runtime, entry, matched, mode="clone")
    assert restored is not None
    cache, _mtp, _mode, restore_point, _hidden_last = restored
    assert restore_point == SHARED
    return cache


def _fresh(attn, rows, *chunks):
    cache = QSACache(RATIO)
    out = None
    for name in chunks:
        out = attn(rows[name], cache)
    return cache, out


def _pooled(cache: QSACache) -> mx.array:
    """The pooled blocks the cache holds for its own tokens (what a commit stores)."""

    return cache.state[3]


def test_an_ssd_restore_below_an_entrys_end_counts_only_the_blocks_it_restored(
    tmp_path, attn, rows,
):
    tier = _tier(tmp_path)
    try:
        bank = _bank(tier)
        _stored_conversation(attn, rows, bank, tier)
        cache = _restore_below_the_end(bank)
        qsa = cache[0]
        # The streams the codec cannot cut arrive at the stored entry's length.
        assert qsa.offset == SHARED
        assert int(qsa.pooled.shape[1]) == (SHARED + DONOR_TAIL) // RATIO
        assert int(qsa.raw_keys.shape[1]) == SHARED + DONOR_TAIL
        # The recurrent layer came from the boundary, not the entry's end.
        assert mx.array_equal(cache[1].state[0], mx.full((2, 2), 7.0)).item()

        out = attn(rows["suffix"], qsa)
        fresh, golden = _fresh(attn, rows, "shared", "suffix")
        # The layer answers the new tokens exactly as an unrestored
        # conversation does: their blocks were pooled from their own keys,
        # and the indexer selected what it selects without the restore.
        assert mx.array_equal(out, golden).item()
        assert mx.array_equal(_pooled(qsa), _pooled(fresh)).item()
        assert qsa.pooled_len == (SHARED + SUFFIX) // RATIO
    finally:
        tier.close()


def test_the_next_commit_and_its_descendants_hold_the_conversations_own_blocks(
    tmp_path, attn, rows,
):
    tier = _tier(tmp_path)
    try:
        bank = _bank(tier)
        _stored_conversation(attn, rows, bank, tier)
        cache = _restore_below_the_end(bank)
        attn(rows["suffix"], cache[0])
        fresh, _ = _fresh(attn, rows, "shared", "suffix")
        # The commit after the restored turn: what it stores is this
        # conversation's table, not the older conversation's.
        tokens = SHARED_IDS + SUFFIX_IDS
        _persist(
            bank, tier, tokens=tokens, cache=cache,
            boundaries=[(SHARED, _boundary(7.0), mx.zeros((1, 1, 4)))],
            session_id="new-conversation",
        )
        runtime = _Runtime()
        # A descendant: the next turn restores that entry whole from disk.
        descendant = bank.restore(
            runtime, tokens + NEXT_IDS, template_hash=TEMPLATE, policy_fingerprint=POLICY,
        )
        assert descendant is not None and descendant.cache_source == "ssd"
        qsa = descendant.cache[0]
        assert qsa.offset == len(tokens)
        assert mx.array_equal(_pooled(qsa), _pooled(fresh)).item()
        out = attn(rows["next"], qsa)
        _, golden = _fresh(attn, rows, "shared", "suffix", "next")
        assert mx.array_equal(out, golden).item()
    finally:
        tier.close()


def test_a_turn_restored_below_a_longer_entry_resizes_for_the_compiled_verifier(
    tmp_path, attn, rows,
):
    """#592: the same restore, then the one-copy store's resize before the
    verifier's bank adopts the buffers. The bank holds this prompt and its
    reserve, fewer rows than the stored conversation: on 2.12.2 the resize
    read that conversation's pooled count as blocks it must keep and the turn
    failed with "QSA pooled hold 256 rows; resizing to 192 would cut them"."""

    bank_rows = 3 * BLOCK  # the prompt's 576 tokens and their reserve, on the K/V step
    tier = _tier(tmp_path)
    try:
        bank = _bank(tier)
        _stored_conversation(attn, rows, bank, tier)
        cache = _restore_below_the_end(bank)
        # The suffix prefill sizes the buffers for the bank, as the one-copy
        # prefill does (one_copy.prefill_rows_scope).
        with one_copy.prefill_rows_scope(bank_rows):
            attn(rows["suffix"], cache[0])
        assert one_copy.layers_to_resize(cache, bank_rows) == 1
        assert one_copy.resize_bill(cache, bank_rows) is not None
        assert one_copy.resize_qsa_buffers(cache, bank_rows) == 0
        qsa = cache[0]
        assert int(qsa.kv.keys.shape[2]) == int(qsa.raw_keys.shape[1]) == bank_rows
        assert int(qsa.pooled.shape[1]) == bank_rows // RATIO
        # Nothing the conversation holds moved; the next tokens attend as in
        # a conversation that was never restored.
        fresh, _ = _fresh(attn, rows, "shared", "suffix")
        for leaf, want in zip(qsa.state, fresh.state):
            assert mx.array_equal(leaf, want).item()
        out = attn(rows["next"], qsa)
        _, golden = _fresh(attn, rows, "shared", "suffix", "next")
        assert mx.array_equal(out, golden).item()
    finally:
        tier.close()


def test_a_restored_state_installs_its_count_from_the_keys_it_restored(attn, rows):
    """The setter alone, on the shapes the cold decoder hands it."""

    donor, _ = _fresh(attn, rows, "shared", "donor")
    keys, values, raw, pooled = donor.state
    restored = QSACache(RATIO)
    restore_cache(
        [restored],
        CacheSnapshot(
            states=((keys[:, :, :SHARED], values[:, :, :SHARED], raw, pooled),),
            meta_states=(None,),
        ),
    )
    assert restored.offset == SHARED
    assert restored.pooled_len == SHARED // RATIO
    # A consistent state still keeps every block it holds.
    whole = QSACache(RATIO)
    whole.state = donor.state
    assert whole.pooled_len == (SHARED + DONOR_TAIL) // RATIO
