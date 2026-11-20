"""A new session resumes at the end of the system prompt it shares (12 Nov).

Two sessions of one agent client in the same project send the same system
prompt and tools, then different first messages: in the founder's bank a Pi
pair and an OpenCode pair matched exactly to the end of that segment plus
the next turn's 3-token header. A hybrid model resumes only at a recurrent
checkpoint at or below the match, and the checkpoints sat on the chunk grid,
so such a session resumed at 4,096 and read the rest of an 8K system prompt
again. The SSD lane also offered only the 256-token block edge below the
match, which no checkpoint at the segment's end could serve.

Now a prefill that passes through the end of a ChatML system turn of at least
2,048 tokens records a checkpoint there, every retention keeps it beside the
grid, and the SSD lane offers the token-exact match beside the block edge: a
hybrid row restores at the exact point only when it holds a newer checkpoint
there than at the edge, never at the prompt's last token, and otherwise
exactly as before (the first version replaced the edge, and a first turn
asked again after a restart went cold: tests/engine/test_e2e_server_tiny.py).
"""

from __future__ import annotations

import array
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import generation
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.checkpoint_anchors import (
    SYSTEM_SEGMENT_MIN_TOKENS,
    AnchorPlan,
    record_nbytes,
    retain_checkpoints,
    system_segment_end,
)
from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _KV, _Recurrent, _boundary, _runtime
from test_qwen4_fixed_m4_capacity_bucket import SEED, lane, pack  # noqa: F401

IM_START = 5
SYSTEM = 6
USER = 7
EDGE = 3000  # the system segment's length, in tokens
QWEN38 = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)


def _system_segment(length: int = EDGE) -> list[int]:
    return [IM_START, SYSTEM] + [10 + (7 * i) % 100 for i in range(length - 2)]


def _prompt(user_seed: int, length: int = EDGE) -> list[int]:
    user = [10 + (user_seed * 13 + 11 * i) % 100 for i in range(180)]
    return _system_segment(length) + [IM_START, USER] + user + [IM_START, 8]


class _ChatMLTokenizer:
    def convert_tokens_to_ids(self, token):
        return IM_START if token == "<|im_start|>" else 0

    def convert_ids_to_tokens(self, token_id):
        return "<|im_start|>" if token_id == IM_START else "<unk>"

    def encode(self, text, add_special_tokens=False):
        return [SYSTEM] if text == "system" else [9]


def _record(position: int):
    return (position, SimpleNamespace(nbytes=100), None)


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


def test_the_system_segment_ends_where_the_first_turn_ends(monkeypatch):
    monkeypatch.delenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", raising=False)
    prompt = _prompt(1)
    for tokens in (prompt, tuple(prompt), array.array("I", prompt)):
        assert system_segment_end(tokens) == EDGE
    # Too short to be worth a checkpoint of its own.
    short = _prompt(1, length=SYSTEM_SEGMENT_MIN_TOKENS - 1)
    assert system_segment_end(short) is None
    assert system_segment_end(_prompt(1, length=SYSTEM_SEGMENT_MIN_TOKENS)) == 2048
    # A template that opens with a token that never recurs (a BOS) has none.
    assert system_segment_end([1] + prompt) is None
    assert system_segment_end(_system_segment()) is None
    assert system_segment_end([IM_START]) is None
    monkeypatch.setenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", "0")
    assert system_segment_end(prompt) is None


def test_only_a_chatml_system_turn_records_one():
    rt = SimpleNamespace(tokenizer=_ChatMLTokenizer())
    assert generation._system_prompt_end(rt, _prompt(1)) == EDGE
    # A first turn that is a user message: no checkpoint for it.
    user_first = [IM_START, USER] + _prompt(1)[2:]
    assert generation._system_prompt_end(rt, user_first) is None
    # A tokenizer without ChatML markers has no system turn to find.
    other = SimpleNamespace(tokenizer=SimpleNamespace())
    assert generation._system_prompt_end(other, _prompt(1)) is None


def test_retention_keeps_it_beside_the_grid():
    # The checkpoint at the system prompt's end shares the grid's first cell
    # with the chunk-grid checkpoint below it, which the grid keeps first.
    records = [_record(p) for p in (4096, 8274, 16384, 40000, 80000, 120000, 155000)]
    plan = dict(record_count=4, prompt_end=156000, tail_backoff=64)
    assert 8274 not in _positions(retain_checkpoints(records, AnchorPlan(**plan)))
    kept = _positions(retain_checkpoints(records, AnchorPlan(**plan, system_end=8274)))
    assert 8274 in kept and 4096 in kept and len(kept) == 4


def test_the_prefill_records_it_only_when_it_passes_through_it():
    bank = SimpleNamespace(checkpoint_budget_bytes=None)
    cold = generation._checkpoint_sink(
        [], session_bank=bank, prompt_len=4000, system_end=EDGE
    )
    assert cold.plan.system_end == EDGE
    assert generation._mandatory_prefill_edges(None, cold, length=3999) == (EDGE,)
    # Resumed below it (another session's grid checkpoint): relative edge.
    below = generation._checkpoint_sink(
        [_record(2048)], session_bank=bank, prompt_len=4000, restore_point=2048,
        system_end=EDGE,
    )
    assert generation._mandatory_prefill_edges(
        1024, below, cached_tokens=2048, length=4000 - 2048 - 1
    ) == (EDGE - 2048,)
    # Resumed past it: nothing to record, and the inherited one is kept.
    past = generation._checkpoint_sink(
        [_record(EDGE)], session_bank=bank, prompt_len=9000, restore_point=8000,
        system_end=EDGE,
    )
    assert generation._mandatory_prefill_edges(
        None, past, cached_tokens=8000, length=999
    ) == ()
    assert EDGE in _positions(past)
    # An image prompt's prefill is not cut for it; retention still keeps it.
    image = generation._checkpoint_sink(
        [], session_bank=bank, prompt_len=4000, system_end=EDGE, cuts_forwards=False
    )
    assert generation._mandatory_prefill_edges(None, image, length=3999) == ()
    assert image.plan.system_end == EDGE


def _entry_with(tokens, positions):
    return SimpleNamespace(
        token_ids=tuple(tokens), gdn_boundaries=[_record(p) for p in positions]
    )


def test_an_inherited_set_keeps_it(monkeypatch):
    monkeypatch.delenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", raising=False)
    tokens = _system_segment(8274) + [IM_START, USER] + [11] * 150000
    entry = _entry_with(tokens, (4096, 8274, 16384, 40000, 80000, 120000, 150000))
    budget = 4 * 100
    kept = _positions(
        generation._inherited_gdn_boundaries(entry, 150000, budget_bytes=budget)
    )
    assert 8274 in kept and 4096 in kept and 150000 in kept
    monkeypatch.setenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", "0")
    off = _positions(
        generation._inherited_gdn_boundaries(entry, 150000, budget_bytes=budget)
    )
    assert 8274 not in off


def test_the_bank_keeps_it_when_it_thins_a_set():
    one = record_nbytes(_boundary(1))
    bank = SessionBank(checkpoint_budget_bytes=4 * one)
    tokens = _system_segment(8274) + [IM_START, USER] + list(range(20, 30020))
    n = len(tokens)
    positions = (4096, 8192, 8274, 12288, 16384, 20480, 24576, 28672, n - 64)
    entry = bank.put(
        runtime=_runtime(),
        token_ids=tokens,
        cache=[_KV(n), _Recurrent(n)],
        logits=mx.zeros((1, 4)),
        hidden=None,
        mtp_history_policy="cycle",
        gdn_boundaries=[_boundary(p) for p in positions],
    )
    kept = _positions(entry.gdn_boundaries)
    assert 8274 in kept and 4096 in kept and len(kept) == 4, kept


def _ssd_entry(tokens, *, boundaries):
    class Entry:
        token_ids = tuple(tokens)
        nbytes = 2048
        cache_snapshot = CacheSnapshot(
            states=[
                mx.zeros((1, 2, len(tokens), 4), dtype=mx.float16),
                mx.zeros((1, 4), dtype=mx.float16),
            ],
            meta_states=[{"offset": len(tokens)}, None],
        )
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = tuple(
            (
                position,
                CacheSnapshot(
                    states=(None, mx.full((1, 4), float(position), dtype=mx.float16)),
                    meta_states=(None, None),
                ),
                None,
            )
            for position in boundaries
        )
        has_recurrent = True
        session_id = "pi-a"
        token_hash = f"hash-{len(tokens):05d}" * 2
        prefix_len = len(tokens)
        model_path = "model"
        mtp_enabled = False
        hidden_variant = None
        template_hash = None
        mtp_history_policy = None
        draft_head_identity = None
        policy_fingerprint = None

    return Entry()


def test_the_ssd_lane_resumes_a_new_session_at_it(tmp_path):
    """The founder's restart case: the other session's conversation is only
    on SSD. Its rows were matched to the 256-token block edge below the
    match, under the checkpoint at the end of the system prompt."""

    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        stored = _prompt(1, length=8274)
        assert tier.put_entry(
            _ssd_entry(stored, boundaries=(4096, 8192, 8274, len(stored) - 64)),
            capabilities=["ar_insert"],
        )
        deadline = time.time() + 5.0
        while tier.stats()["writes_completed"] < 1:
            assert time.time() < deadline, "writer did not complete"
            time.sleep(0.05)
        query = tuple(_prompt(2, length=8274))
        hit = tier.lookup_prefix_boundary(
            query,
            model_path="model",
            mtp_enabled=False,
            max_token_gap=8,
            min_matched_tokens=64,
            block_size=256,
            block_min_matched_tokens=512,
        )
        assert hit is not None and hit.restore_kind == "block_prefix"
        # The system prompt plus the next turn's header; block edge 8,192.
        assert hit.matched_tokens == 8274 + 2
        assert hit.record.cache_snapshot_prefix_len == 8274
        assert tier.stats()["prefix_restores_at_exact_point"] == 1
    finally:
        tier.close()


def _ssd_lookup(tmp_path, stored, boundaries, query):
    """One row on SSD, one prefix lookup against it: (hit, the tier's stats)."""

    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.put_entry(
            _ssd_entry(stored, boundaries=boundaries), capabilities=["ar_insert"]
        )
        deadline = time.time() + 5.0
        while tier.stats()["writes_completed"] < 1:
            assert time.time() < deadline, "writer did not complete"
            time.sleep(0.05)
        hit = tier.lookup_prefix_boundary(
            tuple(query),
            model_path="model",
            mtp_enabled=False,
            max_token_gap=8,
            min_matched_tokens=64,
            block_size=256,
            block_min_matched_tokens=512,
        )
        return hit, tier.stats()
    finally:
        tier.close()


def test_the_exact_point_is_an_addition_to_the_block_edge(tmp_path):
    """A row written without a checkpoint at the system prompt's end (by a
    build before this one, or a conversation without one): the block edge
    restores exactly as it always did."""

    stored = _prompt(1, length=8274)
    hit, stats = _ssd_lookup(
        tmp_path, stored, (4096, 8192, len(stored) - 64), _prompt(2, length=8274)
    )
    assert hit is not None and hit.restore_kind == "block_prefix"
    assert hit.matched_tokens == 8192
    assert hit.record.cache_snapshot_prefix_len == 8192
    assert stats.get("prefix_restores_at_exact_point", 0) == 0


def test_a_first_turn_asked_again_after_a_restart_restores_as_before(tmp_path):
    """tests/engine/test_e2e_server_tiny.py's restart case: a first turn of
    1,025 tokens on SSD with its answer, checkpoints at 960, 1,024 and the
    prompt's end (1,025). Asked again by a new process, it restores 1,024
    tokens and reads one; a restore at the prompt's end leaves nothing to
    read, and the turn went cold."""

    prompt = list(range(1000, 2025))
    stored = prompt + list(range(5000, 5096))
    hit, stats = _ssd_lookup(tmp_path, stored, (960, 1024, 1025), prompt)
    assert hit is not None and hit.restore_kind == "block_prefix"
    assert hit.matched_tokens == 1024
    assert hit.record.cache_snapshot_prefix_len == 1024
    assert stats.get("prefix_restores_at_exact_point", 0) == 0


def test_an_image_between_the_edge_and_the_match_keeps_the_edge(tmp_path):
    """An image conversation on SSD (keyed ids): when the stretch between the
    block edge and the exact match holds image rows, a checkpoint there may
    sit inside the image, where a restore is cut back to the image's start
    and this row holds no state. The block edge restores as it always did."""

    from mtplx.vision.splice import _BANK_KEY_FLAG

    text = [10 + (7 * i) % 100 for i in range(8200)]
    image = [_BANK_KEY_FLAG | (0xABC00 + i) for i in range(100)]
    stored = text + image + [10 + (3 * i) % 100 for i in range(200)]
    query = stored[:8350] + [7, 7, 7] + [11] * 60
    hit, stats = _ssd_lookup(
        tmp_path, stored, (4096, 8192, 8250, len(stored) - 64), query
    )
    assert hit is not None and hit.restore_kind == "block_prefix"
    assert hit.matched_tokens == 8192
    assert hit.record.cache_snapshot_prefix_len == 8192
    assert stats.get("prefix_restores_at_exact_point", 0) == 0


def test_the_switch_turns_the_exact_point_off_too(tmp_path, monkeypatch):
    """MTPLX_SYSTEM_PROMPT_CHECKPOINT=0 is the whole rollback: the SSD lane
    restores at the block edge even where a newer checkpoint lies above it."""

    monkeypatch.setenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", "0")
    stored = _prompt(1, length=8274)
    hit, stats = _ssd_lookup(
        tmp_path, stored, (4096, 8192, 8274, len(stored) - 64), _prompt(2, length=8274)
    )
    assert hit is not None and hit.matched_tokens == 8192
    assert hit.record.cache_snapshot_prefix_len == 8192
    assert stats.get("prefix_restores_at_exact_point", 0) == 0


def test_the_exact_point_serves_a_row_the_block_edge_could_not(tmp_path):
    """No checkpoint at or below the block edge, one at the system prompt's
    end: the block edge refused this row; the exact point restores it."""

    stored = _prompt(1, length=8274)
    hit, stats = _ssd_lookup(
        tmp_path, stored, (8274, len(stored) - 64), _prompt(2, length=8274)
    )
    assert hit is not None and hit.matched_tokens == 8274 + 2
    assert hit.record.cache_snapshot_prefix_len == 8274
    assert stats["prefix_restores_at_exact_point"] == 1


def _first_turn(rt, bank, prompt, session_id):
    return generation.generate_mtpk(
        rt, list(prompt), max_tokens=8, sampler=QWEN38, draft_sampler=QWEN38,
        speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), capture_final_state=True, session_bank=bank,
        session_id=session_id, session_restore_mode="reference",
        commit_prompt_state_to_bank=True,
    )


@pytest.mark.parametrize("switch", ["1", "0"])
def test_a_new_session_resumes_at_the_end_of_the_system_prompt(pack, lane, switch):
    smoke, model = pack
    lane.setenv("MTPLX_ONE_COPY", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    # The product prefill (profiles.py) on a chunk grid the toy prompt spans.
    lane.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    lane.setenv("MTPLX_PREFILL_CHUNK_SIZE", "1024")
    lane.setenv("MTPLX_SYSTEM_PROMPT_CHECKPOINT", switch)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    rt._mtplx_chatml_system_markers = (IM_START, (SYSTEM,))
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)

    first = _first_turn(rt, bank, _prompt(1), "session-a")
    assert first.stats.cached_tokens == 0
    anchors = sorted(
        {
            int(record[0])
            for entry in bank._entries.values()
            for record in (entry.gdn_boundaries or [])
        }
    )
    second = _first_turn(rt, bank, _prompt(2), "session-b")
    cold_rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(cold_rt)
    cold = _first_turn(cold_rt, SessionBank(max_bytes=1 << 32), _prompt(2), "cold-b")

    # The chunk grid below the segment's end is kept either way.
    assert 2048 in anchors, anchors
    if switch == "1":
        assert EDGE in anchors, anchors
        # Resumed from session A's state at the end of the system prompt.
        assert second.stats.cached_tokens == EDGE
    else:
        # What every new session got before: the grid checkpoint below.
        assert EDGE not in anchors, anchors
        assert second.stats.cached_tokens == 2048
    # Resuming at a checkpoint gives the same tokens as a cold prefill.
    assert list(second.tokens) == list(cold.tokens)
