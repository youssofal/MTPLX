"""The admission guard, priced honestly and able to make room.

Scenarios from the field, on the real SessionBank and EngineSessionManager
with MLX's allocator and the process footprint mocked (no model, no GPU):

* the 2026-09-26 report (M5 Max 128 GB, Flash-Next, pi): the compaction is a
  74K-token full miss on a new session while the 114K conversation it
  summarizes sits in the bank twice (a generation-final and a postcommit
  entry, each holding a live cache), the engine at its 96 GiB limit. The old
  shed could not reach that conversation and refused 13 times in a row;
* #499 (M5 Pro 48 GB, 27B Optimized Speed, OpenCode): a 3,185-token turn on a
  96,170-token conversation, under the old 4,096-token floor, was admitted
  unexamined while its restore and its banked prompt copied the conversation
  twice more (38.5 GiB active, 39.5 GiB peak, against a 36 GiB limit);
* the warm 114K turn with a clone restore in the same report (94.9 to 100.2
  GiB during ordinary turns).

Plus the per-chunk supply check, the pressure loop's death signature and
#525's three points.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.engine_session import EngineSessionManager, per_session_play_ceiling_bytes
from mtplx.session_bank import SessionBank

GIB = 1024**3

# Flash-Next (Qwen3.8-Flash-Next config.json): 12 QSA layers x 2 KV heads x
# 256 x K+V x bf16 = 24,576 B a token of KV, 7,872 of QSA streams and MTP KV.
FN_KV = 24_576
FN_AUX = 7_872
FN_ROW = FN_KV + FN_AUX
FN_WEIGHTS = int(77.3 * GIB)
# Qwen3.8-27B: 16 full-attention layers x 4 KV heads x 256 x K+V x bf16.
Q27_KV = 65_536
Q27_WEIGHTS = 21_313_949_792  # the Speed pack fixture in test_memory_plan
# Qwen3.8-27B text_config (the Speed pack's config.json), read through the
# args class the runtime builds, so the fields the guard reads are the ones
# production has.
Q27_TEXT_CONFIG = dict(
    model_type="qwen3_5_text",
    hidden_size=5120,
    intermediate_size=17408,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262_144,
    vocab_size=248_320,
    rms_norm_eps=1e-6,
)


def _q27_text_args():
    from mlx_lm.models.qwen3_5 import TextModelArgs

    return TextModelArgs.from_dict(Q27_TEXT_CONFIG)


def _q27_runtime():
    return SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(args=_q27_text_args())),
        mtp_enabled=True,
        model_path=Path("models/qwen3.8-27b"),
    )

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    # What the sustained and turbo profiles stamp: chunked prefill, the auto
    # layout (dense decode up to the ceiling, repaged past it or with
    # quantized KV), the shared 2,048-row chunk.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    # The sparse prefill lane serves Flash-Next on an M5 (tensor units).
    import mtplx.models.qwen4_exp as qwen4

    monkeypatch.setattr(qwen4, "_qsa_prefill_enabled", lambda: True)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _entries_retained_by(obj, *, depth: int = 0) -> list:
    """The bank entries a queued job keeps alive: what its closure cells,
    default arguments and bound partial arguments hold, followed through
    nested functions and containers. Other objects (the bank a job's method
    belongs to) are not descended into: their references are not the job's."""

    import functools
    import inspect

    from mtplx.session_bank import SessionBankEntry

    if isinstance(obj, SessionBankEntry):
        return [obj]
    if depth > 4:
        return []
    children: list = []
    if isinstance(obj, functools.partial):
        children = [obj.func, *obj.args, *obj.keywords.values()]
    elif inspect.isfunction(obj):
        for cell in obj.__closure__ or ():
            try:
                children.append(cell.cell_contents)
            except ValueError:
                continue
        children.extend(obj.__defaults__ or ())
        children.extend((obj.__kwdefaults__ or {}).values())
    elif inspect.ismethod(obj):
        children = [obj.__func__]
    elif isinstance(obj, (tuple, list, set, frozenset)):
        children = list(obj)
    elif isinstance(obj, dict):
        children = list(obj.values())
    found: list = []
    for child in children:
        found.extend(_entries_retained_by(child, depth=depth + 1))
    return found


class _Machine:
    """MLX's allocator account and the process footprint, as the guard reads
    them: active = a fixed base, plus what the bank holds, plus what queued
    idle-lane jobs still hold (``lane``); clearing the cache empties the
    allocator pool; the footprint adds host memory outside MLX.

    The queued part is the review of 9c96dd9c: a queued settle or SSD encode
    keeps its entry's arrays until the job runs or is cancelled, so an entry
    evicted from the bank with its job still queued frees nothing. Counting
    only the bank's entries hid that."""

    def __init__(
        self,
        bank,
        *,
        base_gib: float,
        cache_gib: float,
        host_gib: float,
        lane=None,
    ):
        self.bank = bank
        self.base = int(base_gib * GIB)
        self.cache = int(cache_gib * GIB)
        self.host = int(host_gib * GIB)
        self.lane = lane

    def queued(self) -> int:
        """Snapshots out of the bank that a queued job still holds, found by
        what each job in the lane retains (its closure), never by the bank's
        own tracking map: a bank that drops its tracking while the job stays
        queued must read as holding the memory, because it does (the review
        of 23a94abf)."""

        if self.lane is None:
            return 0
        held: dict[int, int] = {}
        for job in list(self.lane.pending.values()):
            for entry in _entries_retained_by(job):
                if self.bank._entries.get(entry.token_ids) is entry:
                    continue
                held[id(entry)] = int(entry.nbytes)
        return sum(held.values())

    def active(self) -> int:
        return self.base + int(self.bank.total_nbytes) + self.queued()

    def stats(self) -> dict:
        return {
            "ok": True,
            "active_memory_bytes": self.active(),
            "cache_memory_bytes": self.cache,
        }

    def clear_cache(self) -> None:
        self.cache = 0

    def footprint(self, *args, **kwargs) -> int:
        return self.active() + self.cache + self.host


def _install(monkeypatch, machine: _Machine) -> None:
    import mlx.core as mx

    monkeypatch.setattr(srv, "_mlx_memory_stats_live", machine.stats)
    monkeypatch.setattr(mx, "clear_cache", machine.clear_cache)
    monkeypatch.setattr(srv, "phys_footprint_bytes", machine.footprint)


def _admit(state, **kwargs):
    """The admission as the build under test takes it.

    With MTPLX_TEST_BASELINE=1 the keywords an older build does not have are
    left out, so the field reproductions (#499, #525) run on that build and
    fail by what it does: the review of 9c96dd9c found them failing on
    1de2b1c0 with a TypeError, which proves nothing. Without it, every
    keyword is passed and an unknown one raises as usual."""

    import inspect
    import os

    if os.environ.get("MTPLX_TEST_BASELINE") == "1":
        params = inspect.signature(srv._prefill_admission_shed).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return srv._prefill_admission_shed(state, **kwargs)


def _flash_next_runtime():
    args = SimpleNamespace(
        layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
        num_attention_heads=24,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        hidden_size=2560,
        hc_count=4,
        ple_layer_ids=[2],
        indexer_n_heads=4,
    )
    return SimpleNamespace(
        model=SimpleNamespace(args=args),
        mtp_enabled=True,
        model_path=Path("models/flash-next"),
    )


def _state(manager, *, plan, runtime, limit_gib: float, total_gib: float):
    return SimpleNamespace(
        metal_memory_caps={
            "memory_limit_bytes": int(limit_gib * GIB),
            "total_ram_bytes": int(total_gib * GIB),
        },
        memory_plan=plan,
        runtime=runtime,
        sessions=manager,
        dashboard=SimpleNamespace(),
        allow_swap=False,
        memory_budget_bytes=None,
    )


def _flash_next_state(manager, *, limit_gib: float = 96):
    plan = SimpleNamespace(
        available=True,
        kv_bytes_per_token=FN_KV,
        kv_bytes_per_token_effective=FN_KV,
        aux_bytes_per_token=FN_AUX,
        prefill_transient_bytes_per_token=0,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=FN_WEIGHTS,
    )
    return _state(
        manager, plan=plan, runtime=_flash_next_runtime(), limit_gib=limit_gib, total_gib=128
    )


def _manager(**bank_kwargs) -> EngineSessionManager:
    defaults = dict(max_entries=64, max_bytes=60 * GIB, per_session_max_bytes=30 * GIB)
    defaults.update(bank_kwargs)
    return EngineSessionManager(bank=SessionBank(**defaults), idle_ttl_s=3600)


def _put(bank, tokens, *, session_id, row_bytes, live_cache=False):
    tokens = tuple(tokens)
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=len(tokens) * row_bytes,
    )
    assert entry is not None
    if live_cache:
        # keep_live_ref: the generation-final commit of a coding-agent turn
        # keeps the live cache, and its lazy snapshot aliases it.
        entry.cache_ref = object()
        entry.lazy_kv = True
    return entry


CONV = tuple(range(114_191))


def _julian_conversation(bank):
    """The 114K conversation resident twice: generation-final and postcommit
    entries diverging at the last token, each holding a live cache."""

    final = _put(bank, CONV, session_id="anon-conv", row_bytes=FN_ROW, live_cache=True)
    post = _put(
        bank, CONV[:-1] + (999_999,), session_id="anon-conv", row_bytes=FN_ROW, live_cache=True
    )
    return final, post


COMPACTION = list(range(1_000_000, 1_073_663))  # 73,663 tokens, nothing shared


class TestJulianCompaction:
    def _setup(self, monkeypatch, *, host_gib: float):
        manager = _manager()
        _julian_conversation(manager.bank)
        # The compaction request holds its own session's slot.
        incoming = manager.get_or_create("anon-compaction")
        assert incoming.try_begin_generation()
        # 96.0 GiB in MLX's account: weights and the rest of the process's
        # Metal allocations (88.6 GiB) plus the conversation twice (6.9).
        machine = _Machine(manager.bank, base_gib=88.6, cache_gib=0.5, host_gib=host_gib)
        _install(monkeypatch, machine)
        return manager, incoming, machine

    def test_the_idle_conversation_is_released_and_the_compaction_admitted(
        self, monkeypatch
    ):
        """At the base commit this returns refused=True: projected 100.7 GiB
        after the pool clear, nothing reachable (the conversation is another
        session, active-pinned, and both its entries hold a live cache)."""

        manager, incoming, machine = self._setup(monkeypatch, host_gib=6.0)
        state = _flash_next_state(manager)
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=COMPACTION,
                session_bank=manager.bank,
                session_id="anon-compaction",
            )
        finally:
            incoming.end_generation()
        assert receipt is not None
        assert receipt.get("refused") is not True
        released = receipt["idle_release"]
        assert [row["session_id"] for row in released["sessions"]] == ["anon-conv"]
        assert released["held_bytes"] == 2 * len(CONV) * FN_ROW
        assert not manager.bank.has_session_entries("anon-conv")
        # The growth counts the Flash-Next prefill scratch, never zero.
        assert receipt["growth"]["scratch_source"] == "qsa_itemized"
        assert receipt["growth"]["scratch_bytes"] > 3 * GIB
        assert receipt["projected_bytes_after"] <= 96 * GIB

    def test_a_host_leak_is_named_in_the_refusal(self, monkeypatch):
        """14.8 GiB outside MLX (the report's #546 figure): 6.8 GiB past the
        allowance is charged. The conversation is released and the prompt
        still does not fit; the refusal says that memory has to come down
        (queued writes finishing, or a restart), not that only a restart
        frees it (the review of 9c96dd9c: queued encodes hold host memory
        too)."""

        manager, incoming, machine = self._setup(monkeypatch, host_gib=14.8)
        state = _flash_next_state(manager)
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=COMPACTION,
                session_bank=manager.bank,
                session_id="anon-compaction",
            )
        finally:
            incoming.end_generation()
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["host_overhang_charged_bytes_after"] == int(6.8 * GIB)
        assert receipt["retry_can_succeed"] is False
        assert receipt["retry_when"] == "after_host_memory_returns"
        assert not manager.bank.has_session_entries("anon-conv")

    def test_a_conversation_in_flight_is_kept_and_the_507_says_so(self, monkeypatch):
        manager, incoming, machine = self._setup(monkeypatch, host_gib=6.0)
        conversation = manager.get_or_create("anon-conv")
        assert conversation.try_begin_generation()
        state = _flash_next_state(manager)
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=COMPACTION,
                session_bank=manager.bank,
                session_id="anon-compaction",
            )
        finally:
            incoming.end_generation()
            conversation.end_generation()
        assert receipt["refused"] is True
        assert manager.bank.has_session_entries("anon-conv")
        holders = receipt["holders"]
        [row] = [r for r in holders["sessions"] if r["session_id"] == "anon-conv"]
        assert row["held_because"] == "in_flight"
        assert holders["in_flight_bytes"] == 2 * len(CONV) * FN_ROW
        assert receipt["retry_can_succeed"] is True
        assert receipt["retry_when"] == "after_in_flight_requests_finish"

        error = srv._prefill_admission_refusal(state, receipt)
        assert error.status_code == 507
        detail = error.detail
        assert detail["code"] == "insufficient_memory"
        assert "refused before prefill" in detail["message"]
        assert "requests in flight finish" in detail["message"]
        memory = detail["memory"]
        assert memory["retry_can_succeed"] is True
        assert memory["retry_when"] == "after_in_flight_requests_finish"
        assert memory["holders"]["sessions"][0]["held_because"] == "in_flight"
        assert srv._http_exception_message(error) == detail["message"]


class TestWarmTurns:
    """Warm turns used to skip the projection under 4,096 new tokens."""

    def test_a_warm_114k_clone_restore_projects_its_copy_and_sheds_first(
        self, monkeypatch
    ):
        manager = _manager()
        final, post = _julian_conversation(manager.bank)
        # An idle subagent holds a live cache too; nothing but the idle
        # release reaches it.
        sub = _put(
            manager.bank,
            range(500_000, 500_000 + 110_000),
            session_id="sub",
            row_bytes=FN_ROW,
            live_cache=True,
        )
        conversation = manager.get_or_create("anon-conv")
        assert conversation.try_begin_generation()
        machine = _Machine(manager.bank, base_gib=81.5, cache_gib=0.5, host_gib=6.0)
        _install(monkeypatch, machine)
        state = _flash_next_state(manager)
        prompt = list(CONV) + list(range(2_000_000, 2_000_195))
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="anon-conv",
                restore_mode="clone",
            )
        finally:
            conversation.end_generation()
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == len(CONV)
        assert receipt["miss_tokens"] == 195
        assert receipt["restore_copies_prefix"] is True
        assert receipt["growth"]["restore_copy_bytes"] == len(CONV) * FN_ROW
        # Without the shed the turn crosses the limit itself.
        assert receipt["projected_bytes"] > 96 * GIB
        assert [row["session_id"] for row in receipt["idle_release"]["sessions"]] == ["sub"]
        assert "refused" not in receipt
        assert receipt["projected_bytes_after"] <= int(96 * GIB * 0.97)
        assert final.token_ids in manager.bank._entries
        assert post.token_ids in manager.bank._entries
        assert sub.token_ids not in manager.bank._entries

    def test_the_conversations_own_sibling_goes_only_to_avoid_a_refusal(
        self, monkeypatch
    ):
        manager = _manager()
        final, post = _julian_conversation(manager.bank)
        conversation = manager.get_or_create("anon-conv")
        assert conversation.try_begin_generation()
        # 9 GiB outside MLX: 1 GiB past the allowance is charged.
        machine = _Machine(manager.bank, base_gib=84.2, cache_gib=0.5, host_gib=9.0)
        _install(monkeypatch, machine)
        state = _flash_next_state(manager)
        prompt = list(CONV) + list(range(2_000_000, 2_000_195))
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="anon-conv",
                restore_mode="clone",
            )
        finally:
            conversation.end_generation()
        assert receipt.get("refused") is not True
        own = receipt["own_session_release"]
        assert own["entries"] == 1
        assert final.token_ids in manager.bank._entries  # the restore source
        assert post.token_ids not in manager.bank._entries


class Test499FortyEightGigSeat:
    """M5 Pro 48 GB, 27B Optimized Speed, OpenCode: a 99,355-token prompt on
    a committed 96,170-token conversation (3,185 new). Measured active 38.5
    GiB and peak 39.5 GiB against the plan's 36 GiB limit. By mechanism: the
    lease restore of a generation-final entry whose lazy snapshot aliases the
    live cache copies the conversation on the first write, and the banked
    prompt copies it again at decode."""

    def _plan(self):
        return SimpleNamespace(
            available=True,
            usable_bytes=36 * GIB,
            kv_bytes_per_token=Q27_KV,
            kv_bytes_per_token_effective=Q27_KV,
            aux_bytes_per_token=0,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
        )

    def test_the_turn_is_priced_with_both_copies_and_made_to_fit(self, monkeypatch):
        plan = self._plan()
        # The two-copy cap the engine sizes its per-session budget with:
        # (36 - 19.85 - 3) / 2 = 6.6 GiB, one 96K snapshot (5.9 GiB) per
        # session. The generation-final entry is that snapshot, and its
        # lazy views alias the live cache it keeps.
        cap = per_session_play_ceiling_bytes(plan)
        manager = _manager(max_bytes=14 * GIB, per_session_max_bytes=cap)
        conversation = tuple(range(96_170))
        final = _put(
            manager.bank, conversation, session_id="opencode", row_bytes=Q27_KV, live_cache=True
        )
        session = manager.get_or_create("opencode")
        assert session.try_begin_generation()
        machine = _Machine(
            manager.bank, base_gib=Q27_WEIGHTS / GIB + 0.5, cache_gib=0.3, host_gib=3.0
        )
        _install(monkeypatch, machine)
        runtime = SimpleNamespace(model=SimpleNamespace(args=SimpleNamespace()), mtp_enabled=True)
        state = _state(manager, plan=plan, runtime=runtime, limit_gib=36, total_gib=48)
        prompt = list(conversation) + list(range(3_000_000, 3_003_185))
        assert len(prompt) == 99_355
        try:
            receipt = _admit(
                state,
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="opencode",
                commit_prompt_prefix=True,
            )
        finally:
            session.end_generation()
        # The old 4,096-token floor returned None here: admitted unexamined.
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 96_170
        assert receipt["restore_copies_prefix"] is True
        # Before the shed, both copies were priced: the restore's (the
        # conversation again, next to its banked snapshot) and decode's copy
        # of the banked prompt, on what is in use (the 0.3 GiB allocator
        # pool is left out: MLX releases it before it passes its own
        # limit). That is 38.4 GiB: the report measured 38.5 GiB active and
        # a 39.5 GiB peak.
        in_use = machine.base + 96_170 * Q27_KV
        assert receipt["projected_bytes"] == in_use + 2 * 99_355 * Q27_KV
        assert 38 * GIB < receipt["projected_bytes"] < 40 * GIB
        # The banked prompt copy is what crossed the line: it is skipped
        # (never replaced by a live reference to a cache decode mutates),
        # and the turn fits the 36 GiB limit with the restore's copy alone.
        assert receipt["prompt_publish_skipped"] is True
        assert receipt["growth"]["publish_copy_bytes"] == 0
        assert receipt["growth"]["restore_copy_bytes"] == 96_170 * Q27_KV
        assert final.token_ids in manager.bank._entries
        assert receipt.get("refused") is not True
        assert receipt["projected_bytes_after"] <= 36 * GIB

    def test_with_the_heads_history_the_turn_runs_at_a_narrower_chunk(self, monkeypatch):
        """The plan now carries the MTP head's committed history (4,096 B a
        token on the 27B). Priced with it the turn projects 39.47 GiB, the
        trace's measured 39.5 GiB peak, where 2.12.0 ran it past the limit and
        produced no tokens before the client gave up. Without the banked
        prompt copy it needed 36.03 GiB at 2,048 rows and was refused (2.12.1);
        at 1,024 rows, whose measured working memory is 1.1 GiB less
        (2026-10-02: 1.52 against 2.55 GiB on the 27B), it needs 34.87 GiB,
        under the 0.97 line of the 36 GiB limit, and runs."""

        plan = self._plan()
        plan.mtp_history_bytes_per_token = 4_096
        row = Q27_KV + 4_096
        cap = per_session_play_ceiling_bytes(plan)
        manager = _manager(max_bytes=14 * GIB, per_session_max_bytes=cap)
        conversation = tuple(range(96_170))
        _put(manager.bank, conversation, session_id="opencode", row_bytes=row, live_cache=True)
        session = manager.get_or_create("opencode")
        assert session.try_begin_generation()
        machine = _Machine(
            manager.bank, base_gib=Q27_WEIGHTS / GIB + 0.5, cache_gib=0.3, host_gib=3.0
        )
        _install(monkeypatch, machine)
        state = _state(manager, plan=plan, runtime=_q27_runtime(), limit_gib=36, total_gib=48)
        prompt = list(conversation) + list(range(3_000_000, 3_003_185))
        try:
            receipt = _admit(
                state,
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="opencode",
                commit_prompt_prefix=True,
            )
        finally:
            session.end_generation()
        assert 39.4 * GIB < receipt["projected_bytes"] < 39.5 * GIB
        assert receipt["prompt_publish_skipped"] is True
        assert "refused" not in receipt
        assert receipt["prefill_chunk_requested"] == 2048
        assert receipt["prefill_chunk_tokens"] == 1024
        assert 34.8 * GIB < receipt["projected_bytes_after"] < 0.97 * 36 * GIB

    def test_the_banked_prompt_copy_is_priced_while_it_fits_the_cap(self, monkeypatch):
        plan = self._plan()
        cap = per_session_play_ceiling_bytes(plan)
        geometry = srv._admission_geometry(SimpleNamespace(memory_plan=plan))
        growth = srv._admission_growth(
            geometry,
            prompt_tokens=99_355,
            reused_tokens=96_170,
            restore_copies_prefix=True,
            layout="contiguous_dense_decode",
            source_layout=None,
            output_tokens=0,
            publish=True,
            scratch_bytes=3 * GIB,
        )
        # Rows written: the restore's copy and the new tokens.
        assert growth["live_prefill_bytes"] == 99_355 * Q27_KV
        # Decode start: the live cache and decode's copy of the banked prompt.
        assert growth["decode_start_bytes"] == 2 * 99_355 * Q27_KV
        assert growth["growth_bytes"] == growth["decode_start_bytes"]
        # 99,355 x 64 KiB = 6.06 GiB fits the seat's two-copy cap, so the
        # snapshot is banked and its copy is real.
        assert 99_355 * Q27_KV <= cap


class TestGrowthModel:
    def _geometry(self, **overrides):
        values = dict(
            live_bytes_per_token=FN_ROW,
            paged_bytes_per_token=FN_ROW,
            context_transient_bytes_per_token=0,
            flat_transient_bytes=3 * GIB,
            weights_bytes=FN_WEIGHTS,
        )
        values.update(overrides)
        return srv._AdmissionGeometry(**values)

    def test_the_largest_moment_not_the_sum(self):
        growth = srv._admission_growth(
            self._geometry(),
            prompt_tokens=60_000,
            reused_tokens=0,
            restore_copies_prefix=True,
            layout="contiguous_dense_decode",
            source_layout=None,
            output_tokens=0,
            publish=True,
            scratch_bytes=3 * GIB,
        )
        rows = 60_000 * FN_ROW
        assert growth["prefill_end_bytes"] == rows + 3 * GIB
        assert growth["decode_start_bytes"] == 2 * rows
        assert growth["growth_bytes"] == max(rows + 3 * GIB, 2 * rows)

    def test_a_pure_lease_adds_only_the_new_rows(self):
        growth = srv._admission_growth(
            self._geometry(),
            prompt_tokens=100_000,
            reused_tokens=97_000,
            restore_copies_prefix=False,
            layout="contiguous_dense_decode",
            source_layout="contiguous_dense_decode",
            output_tokens=0,
            publish=False,
            scratch_bytes=GIB,
        )
        assert growth["restore_copy_bytes"] == 0
        assert growth["live_prefill_bytes"] == 3_000 * FN_ROW

    def test_the_scratch_is_never_zero(self):
        state = SimpleNamespace(runtime=None)
        geometry = self._geometry()
        scratch, source = srv._admission_scratch_bytes(
            state, rows=1, prompt_tokens=10, geometry=geometry
        )
        assert scratch > 0 and source == "flat_per_row"
        state = SimpleNamespace(runtime=_flash_next_runtime())
        scratch, source = srv._admission_scratch_bytes(
            state, rows=2048, prompt_tokens=73_663, geometry=geometry
        )
        assert source == "qsa_itemized"
        # The itemized bill at 2,048 rows for a 74K prompt: forward 1 GiB,
        # dense scores to the 32K crossover, the fixed part.
        assert 3 * GIB < scratch < 5 * GIB

    def test_the_27b_is_priced_from_its_geometry(self):
        """#525 point A: a family without a QSA indexer is billed rows times
        what a row holds plus a fixed part, never 0 and never a flat figure
        that ignores the rows. The 27B is a dense family, billed what was
        measured on it (2026-10-02, the 27B and Bonsai 2 27B through mtplx
        serve: 0.45 to 0.74 GiB for 98 to 195 rows, 1.52 to 1.72 at 1,024,
        2.52 to 2.72 at 2,048): 0.75 GiB plus ten MLP rows a row (1.09 MiB).
        2.12.1 charged 3 GiB at 2,048 rows and 2.04 GiB for a 195-token turn."""

        args = _q27_text_args()
        assert srv._forward_row_bytes(args) == 139_264
        mlp_row = 2 * (5_120 + 3 * 17_408)
        state = SimpleNamespace(runtime=_q27_runtime())
        geometry = self._geometry(live_bytes_per_token=Q27_KV)

        def bill(rows, prompt_tokens=99_355):
            scratch, source = srv._admission_scratch_bytes(
                state, rows=rows, prompt_tokens=prompt_tokens, geometry=geometry
            )
            assert source == "geometry_measured"
            return scratch

        assert bill(2048) == 0.75 * GIB + 10 * mlp_row * 2048
        assert 2.72 * GIB < bill(2048) < 2.95 * GIB
        assert 1.72 * GIB < bill(1024) < 1.85 * GIB
        # Short prompts are on the steeper line: 0.25 GiB plus 36 MLP rows.
        assert bill(98) == 0.25 * GIB + 36 * mlp_row * 98
        assert 0.45 * GIB < bill(98) < 0.65 * GIB
        assert 0.74 * GIB < bill(195) < 0.97 * GIB
        assert bill(4096) == 0.75 * GIB + 10 * mlp_row * 4096
        # Fused attention: no term grows with the keys (the QSA bill would
        # charge 11 GB of scores at 99K keys).
        assert bill(2048, prompt_tokens=1_000) == bill(2048, prompt_tokens=262_144)

    def test_an_unreadable_config_is_charged_the_reserve_per_row(self):
        scratch, source = srv._admission_scratch_bytes(
            SimpleNamespace(runtime=SimpleNamespace(model=SimpleNamespace())),
            rows=1024,
            prompt_tokens=50_000,
            geometry=self._geometry(),
        )
        assert source == "flat_per_row"
        assert scratch == 3 * GIB // 2

    def test_quantized_kv_is_priced_at_full_width_through_the_repage(self):
        """#525 point A on q8: the prefill writes bf16 rows into the
        contiguous cache and the repage fills the quantized pages while
        those rows are still live. The old projection charged the q8 width
        alone: 150K x 36,044 + 3 GiB = 8.0 GiB."""

        q8 = int(Q27_KV * 0.55)
        growth = srv._admission_growth(
            self._geometry(live_bytes_per_token=Q27_KV, paged_bytes_per_token=q8),
            prompt_tokens=150_000,
            reused_tokens=0,
            restore_copies_prefix=True,
            layout="contiguous_then_repage",
            source_layout=None,
            output_tokens=16_384,
            publish=False,
            scratch_bytes=3 * GIB,
        )
        assert growth["repage_bytes"] == 150_000 * Q27_KV + (150_000 + 16_384) * q8
        assert growth["growth_bytes"] == growth["repage_bytes"]
        assert growth["growth_bytes"] > 150_000 * q8 + 3 * GIB


class TestTheLimitIsTheLimit:
    def test_lowering_the_limit_lowers_what_is_admitted(self, monkeypatch):
        """80 GiB in MLX's account and 14 GiB of host memory. At 90 GiB the
        old allowance (22 GiB there) charged none of it and admitted; the
        seat's 8 GiB allowance charges 6 and refuses."""

        manager = _manager()
        machine = _Machine(manager.bank, base_gib=80, cache_gib=0, host_gib=14)
        _install(monkeypatch, machine)
        prompt = list(range(40_000))
        at_96 = srv._prefill_admission_shed(
            _flash_next_state(manager, limit_gib=96),
            prompt_ids=prompt,
            session_bank=manager.bank,
            session_id="s",
        )
        at_90 = srv._prefill_admission_shed(
            _flash_next_state(manager, limit_gib=90),
            prompt_ids=prompt,
            session_bank=manager.bank,
            session_id="s",
        )
        assert at_96 is None or at_96.get("refused") is not True
        assert at_90["refused"] is True


# Free pages under the kernel's own free-page target (4,000 x 16 KiB = 62.5 MiB;
# system_memory.starved_free_bytes): the recorded deaths sat there (09-03 at
# 0.0 GB free, the 09-23 panic at 878 pages). At 0.3 GiB with 20 GiB of file
# cache the Mac is healthy on macOS 27: the 2026-09-29 agent replay ran 0.1 to
# 0.6 GB free with the compressor stepping 0.3 to 0.5 GB/s and swap flat
# (b15ddcfb), so these mechanism tests use a starved reading.
STARVED_FREE_GIB = 16 / 1024


class TestPerChunkSupplyCheck:
    def _reading(self, *, available_gib, free_gib, compressor_gib, at_s, wired_gib=88):
        return sm.SystemMemory(
            available_bytes=int(available_gib * GIB),
            total_bytes=128 * GIB,
            level_percent=20,
            free_bytes=int(free_gib * GIB),
            file_backed_bytes=int((available_gib - free_gib) * GIB),
            wired_bytes=int(wired_gib * GIB),
            compressor_bytes=int(compressor_gib * GIB),
            swap_used_bytes=0,
            monotonic_s=at_s,
        )

    def _guard(
        self,
        monkeypatch,
        readings,
        *,
        pool_gib=0.0,
        active_gib=80.0,
        limit_gib=None,
        allow_swap=False,
        **guard_kwargs,
    ):
        sequence = iter(readings)
        last = [None]

        def reader():
            try:
                last[0] = next(sequence)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        monkeypatch.setattr(
            srv,
            "_mlx_memory_stats_live",
            lambda: {"ok": True, "active_memory_bytes": int(active_gib * GIB),
                     "cache_memory_bytes": int(pool_gib * GIB)},
        )
        # No host memory past the allowance: the engine line is MLX's own.
        monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
        state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=allow_swap)
        if limit_gib is not None:
            state.metal_memory_caps = {"memory_limit_bytes": int(limit_gib * GIB)}
        return srv._PrefillSystemGuard(state, **guard_kwargs)

    def test_a_supply_drop_stops_the_prefill_before_the_next_chunk(self, monkeypatch):
        guard = self._guard(
            monkeypatch,
            [
                self._reading(available_gib=20, free_gib=6, compressor_gib=4, at_s=0.0),
                self._reading(available_gib=3, free_gib=2, compressor_gib=4, at_s=1.0),
            ],
        )
        assert guard() is False
        assert guard() is True
        assert guard.tripped["reason"] == "under_abort_floor"
        assert guard.tripped["abort_floor_bytes"] == 88 * GIB // 16
        # The trip belongs to this request and stays tripped.
        assert guard() is True

    def test_the_engines_own_pool_counts_as_supply(self, monkeypatch):
        guard = self._guard(
            monkeypatch,
            [self._reading(available_gib=3, free_gib=2, compressor_gib=4, at_s=0.0)],
            pool_gib=3.0,
        )
        assert guard() is False

    def test_the_death_signature_stops_it_with_file_cache_to_spare(self, monkeypatch):
        guard = self._guard(
            monkeypatch,
            [
                self._reading(available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=10, at_s=0.0),
                self._reading(available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=12, at_s=1.0),
            ],
        )
        assert guard() is False
        assert guard() is True
        assert guard.tripped["reason"] == "death_signature"

    def test_an_unreadable_machine_never_trips(self, monkeypatch):
        guard = self._guard(monkeypatch, [None])
        assert guard() is False

    def test_steady_compression_in_small_steps_trips_over_the_window(self, monkeypatch):
        """Review of 9c96dd9c: the check replaced its previous reading every
        time, so the compressor growing 320 MiB/s for five seconds in 80 MiB
        steps never grew 256 MiB between two readings and never tripped.
        Measured from every reading of the last ten seconds, it trips once
        320 MiB have accumulated, a second in."""

        step = 80 / 1024
        readings = [
            self._reading(
                available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=10 + i * step, at_s=i * 0.25
            )
            for i in range(21)
        ]
        guard = self._guard(monkeypatch, readings)
        results = [guard() for _ in readings]
        assert True in results
        assert results.index(True) == 4
        assert guard.tripped["reason"] == "death_signature"
        assert guard.tripped["interval_s"] == 1.0

    def test_a_mac_at_its_free_page_floor_at_rest_never_trips(self, monkeypatch):
        """The validation rerun of 54e01d1f (128 GB, 12 GB of other apps):
        free pages sat at the kernel's floor, under the abort floor, for the
        whole run while the compressor moved about 1 GB. Twenty seconds of
        that, with a 100 MiB burst in one step, is not the signature."""

        readings = []
        compressor = 2.84
        for i in range(100):
            compressor += 20 / 1024 * 0.2 + (100 / 1024 if i == 50 else 0.0)
            readings.append(
                self._reading(
                    available_gib=16, free_gib=3.8, compressor_gib=compressor, at_s=i * 0.2
                )
            )
        guard = self._guard(monkeypatch, readings)
        assert [guard() for _ in readings] == [False] * len(readings)

    def test_the_next_chunk_is_reserved_before_it_allocates(self, monkeypatch):
        """10 GiB free and reclaimable against a 5.5 GiB floor (88 GiB
        wired): a chunk that allocates 4.6 GiB would leave 5.4 GiB, so the
        prefill stops before it; a 4.4 GiB chunk leaves 5.6 GiB and runs."""

        reading = self._reading(available_gib=10, free_gib=3, compressor_gib=3, at_s=0.0)
        tight = self._guard(monkeypatch, [reading], chunk_reserve_bytes=int(4.6 * GIB))
        assert tight() is True
        assert tight.tripped["reason"] == "under_abort_floor"
        assert tight.tripped["chunk_reserve_bytes"] == int(4.6 * GIB)
        roomy = self._guard(monkeypatch, [reading], chunk_reserve_bytes=int(4.4 * GIB))
        assert roomy() is False

    def test_an_under_priced_request_stops_at_the_engines_limit(self, monkeypatch):
        """The engine at 90 GiB of a 96 GiB limit with the Mac roomy: a
        7 GiB chunk would take it past the limit, where MLX allocates anyway
        and macOS compresses for minutes (#450). An admission that
        under-priced the request stops here, before the chunk."""

        reading = self._reading(available_gib=30, free_gib=20, compressor_gib=3, at_s=0.0)
        guard = self._guard(
            monkeypatch,
            [reading],
            active_gib=90,
            limit_gib=96,
            chunk_reserve_bytes=7 * GIB,
        )
        assert guard() is True
        assert guard.tripped["reason"] == "engine_limit"
        assert guard.tripped["engine_bytes"] == 90 * GIB
        assert guard.tripped["limit_bytes"] == 96 * GIB
        fits = self._guard(
            monkeypatch,
            [reading],
            active_gib=90,
            limit_gib=96,
            chunk_reserve_bytes=5 * GIB,
        )
        assert fits() is False

    def test_the_engine_line_leaves_the_pool_out(self, monkeypatch):
        """MLX hands pooled buffers back before it allocates past its limit,
        so 8 GiB of pool on 90 GiB active is not over a 96 GiB limit."""

        reading = self._reading(available_gib=30, free_gib=20, compressor_gib=3, at_s=0.0)
        guard = self._guard(
            monkeypatch,
            [reading],
            active_gib=90,
            pool_gib=8,
            limit_gib=96,
            chunk_reserve_bytes=5 * GIB,
        )
        assert guard() is False

    def test_the_engine_line_holds_without_a_machine_reading(self, monkeypatch):
        guard = self._guard(monkeypatch, [None], active_gib=97, limit_gib=96)
        assert guard() is True
        assert guard.tripped["reason"] == "engine_limit"
        assert guard.tripped["system_memory"] is None

    def test_allow_swap_is_the_operators_choice_here_too(self, monkeypatch):
        """--allow-swap admits past every line at admission; the per-chunk
        check used to stop the same request a chunk later."""

        guard = self._guard(
            monkeypatch,
            [self._reading(available_gib=1, free_gib=0.5, compressor_gib=4, at_s=0.0)],
            active_gib=97,
            limit_gib=96,
            allow_swap=True,
        )
        assert guard() is False
        assert guard.tripped is None

    def test_it_plugs_into_the_prefills_abort_site(self, monkeypatch):
        from mtplx.generation import PostcommitAbort, _check_postcommit_abort

        guard = self._guard(
            monkeypatch,
            [self._reading(available_gib=1, free_gib=0.5, compressor_gib=4, at_s=0.0)],
        )
        with pytest.raises(PostcommitAbort):
            _check_postcommit_abort(guard)

    def test_the_abort_is_a_structured_507(self, monkeypatch):
        guard = self._guard(
            monkeypatch,
            [
                self._reading(available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=10, at_s=0.0),
                self._reading(available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=12, at_s=1.0),
            ],
        )
        guard()
        guard()
        monkeypatch.setattr(srv, "_shed_after_allocation_failure", lambda state: {})
        error = srv._prefill_system_abort_exception(SimpleNamespace(), guard.tripped)
        assert error.status_code == 507
        assert error.detail["code"] == "insufficient_memory"
        assert "stopped before its next chunk" in error.detail["message"]
        assert "compressor grew 2.0 GiB" in error.detail["message"]
        assert error.detail["memory"]["reason"] == "death_signature"

    def test_the_engine_limit_abort_says_what_it_needed(self, monkeypatch):
        guard = self._guard(
            monkeypatch,
            [self._reading(available_gib=30, free_gib=20, compressor_gib=3, at_s=0.0)],
            active_gib=90,
            limit_gib=96,
            chunk_reserve_bytes=7 * GIB,
        )
        guard()
        monkeypatch.setattr(srv, "_shed_after_allocation_failure", lambda state: {})
        error = srv._prefill_system_abort_exception(SimpleNamespace(), guard.tripped)
        assert error.status_code == 507
        assert "the engine held 90.0 GiB" in error.detail["message"]
        assert "needs 7.0 GiB, past its 96.0 GiB limit" in error.detail["message"]
        assert "other apps" not in error.detail["message"]
        # It says what it gave back (2026-10-01: "released half of its
        # session cache" with 0 bytes released), and that a retry is priced
        # again before it starts rather than promising either outcome.
        assert "after finding nothing else it could give back" in error.detail["message"]
        assert "A retry is priced again before it starts" in error.detail["message"]
        assert "half of its session cache" not in error.detail["message"]
        assert error.detail["memory"]["retry_can_succeed"] is True
        assert error.detail["memory"]["retry_when"] == "after_background_work_finishes"

    # The founder's 2026-10-01 refusals, in his engine's own numbers: a 32K Pi
    # turn after a compaction held 93.46 GB with a 2.79 GB lease of the
    # compaction prompt (same Pi session id, shares 41 tokens), the chunk
    # reserve was 3.19 GB and the line 96.64 GB.
    _HELD = 93_460_615_364
    _CHUNK = 3_190_364_976
    _LIMIT = 96_636_764_160

    def _founder_guard(self, monkeypatch, active, **guard_kwargs):
        box = {"active": int(active)}
        guard = self._guard(
            monkeypatch,
            [self._reading(available_gib=26, free_gib=3, compressor_gib=2.5, at_s=0.0)],
            limit_gib=self._LIMIT / GIB,
            **guard_kwargs,
        )
        monkeypatch.setattr(
            srv,
            "_mlx_memory_stats_live",
            lambda: {"ok": True, "active_memory_bytes": box["active"], "cache_memory_bytes": 0},
        )
        monkeypatch.setattr(srv, "_shed_after_allocation_failure", lambda state: {})
        return guard, box

    def test_the_last_chunks_progress_keeps_the_forward_reserved(self, monkeypatch):
        # 32,408 of 32,409 is the last chunk's trunk, not the end: its draft
        # history pass, its checkpoint and the final token's forward follow
        # (the review of 425ffc58), so the forward stays reserved until the
        # prefill completes. The founder's reading is 13 MB over with it.
        guard, _box = self._founder_guard(
            monkeypatch,
            self._HELD,
            chunk_reserve_bytes=self._CHUNK,
            after_prefill_reserve_bytes=115_642_384,
        )
        guard.note_prefill_progress(
            {"phase": "chunk", "tokens_done": 32_408, "tokens_total": 32_409}
        )
        assert guard.prefill_done_by is None
        assert guard() is True
        assert guard.tripped["reason"] == "engine_limit"
        assert guard.tripped["chunk_reserve_bytes"] == self._CHUNK

    def test_completion_hands_over_to_the_after_prefill_reserve(self, monkeypatch):
        guard, _box = self._founder_guard(
            monkeypatch,
            self._HELD,
            chunk_reserve_bytes=self._CHUNK,
            after_prefill_reserve_bytes=115_642_384,
        )
        guard.note_prefill_progress({"phase": "completed", "tokens_total": 32_409})
        assert guard.prefill_done_by == "prefill_progress"
        assert guard() is False
        assert guard.tripped is None

    def test_its_own_conversations_unusable_state_goes_before_a_refusal(self, monkeypatch):
        calls = []

        def own_session_shed(reason):
            calls.append(reason)
            box["active"] -= 2_793_632_912
            return {"held_bytes": 2_793_632_912, "entries": 1}

        guard, box = self._founder_guard(
            monkeypatch,
            self._HELD,
            chunk_reserve_bytes=self._CHUNK,
            own_session_shed=own_session_shed,
        )
        assert guard() is False
        assert guard.tripped is None
        assert calls == ["prefill_shed_before_abort_own_session"]
        assert guard.shed["own_session_released_bytes"] == 2_793_632_912
        assert guard.shed["request_continued"] is True

    def test_a_refusal_after_giving_back_says_how_much(self, monkeypatch):
        def own_session_shed(reason):
            return {"held_bytes": 1 * GIB, "entries": 1}

        guard, _box = self._founder_guard(
            monkeypatch,
            self._HELD + 4 * GIB,
            chunk_reserve_bytes=self._CHUNK,
            own_session_shed=own_session_shed,
        )
        assert guard() is True
        error = srv._prefill_system_abort_exception(SimpleNamespace(), guard.tripped)
        assert "after giving back 1.0 GiB of saved conversation state" in error.detail["message"]

    def test_the_receipt_names_the_closest_reading_to_the_line(self, monkeypatch):
        guard, box = self._founder_guard(
            monkeypatch, 86_000_000_000, chunk_reserve_bytes=self._CHUNK
        )
        guard.note_prefill_progress(
            {"phase": "chunk", "tokens_done": 2_048, "tokens_total": 32_409}
        )
        assert guard() is False
        box["active"] = 92_000_000_000
        guard.note_prefill_progress(
            {"phase": "chunk", "tokens_done": 14_336, "tokens_total": 32_409}
        )
        assert guard() is False
        box["active"] = 88_000_000_000
        assert guard() is False
        closest = guard.trajectory()["engine_margin_min"]
        assert closest["tokens_done"] == 14_336
        assert closest["engine_bytes"] == 92_000_000_000
        assert closest["margin_bytes"] == self._LIMIT - 92_000_000_000 - self._CHUNK
        assert closest["check"] == 2


def test_a_refusal_queued_for_the_stream_holds_no_prefill_frames():
    """The 507 raised while handling the prefill's abort chains that abort,
    and the abort's traceback holds the prefill's frames and caches: queued
    as raised, 4.95 GB stayed active after a refusal (2026-10-01). Queued
    detached, the prefill's arrays go with the frames, no collection
    needed."""

    import gc
    import weakref

    from fastapi import HTTPException

    from mtplx.generation import PostcommitAbort

    class Cache:
        pass

    held = {}

    def prefill():
        cache = Cache()
        held["ref"] = weakref.ref(cache)
        raise PostcommitAbort("foreground_preempted_postcommit")

    def run_generation():
        try:
            prefill()
        except PostcommitAbort:
            raise HTTPException(status_code=507, detail={"message": "refused"})

    gc.disable()
    try:
        try:
            run_generation()
        except HTTPException as exc:
            item = srv._stream_error_queue_item(exc)
        assert held["ref"]() is None
        kind, error = item
        assert kind == "error" and error.status_code == 507
        assert error.__traceback__ is None and error.__context__ is None
    finally:
        gc.enable()


def test_a_detached_refusal_drops_both_of_its_chained_exceptions():
    """``raise ... from`` sets the cause while the exception being handled
    stays the context: two branches, and each one's frames go."""

    first = RuntimeError("context")
    second = ValueError("cause")
    top = KeyError("top")
    top.__context__ = first
    top.__cause__ = second
    first.__context__ = OSError("deeper")

    srv._detach_exception_frames(top)

    assert top.__cause__ is None and top.__context__ is None
    assert first.__context__ is None


class _LoopBank:
    def __init__(self, total, max_bytes):
        self.total_nbytes = total
        self.max_bytes = max_bytes
        self.calls = []
        self.protected_sessions = []

    def shrink_to_bytes(
        self,
        target,
        *,
        reason,
        protect_active=False,
        protect_session_ids=None,
        protect_unwritten=False,
        receipt_out=None,
    ):
        # The real bank's keywords: the trim names the sessions generating.
        self.calls.append((target, reason))
        self.protected_sessions.append(set(protect_session_ids or ()))
        self.total_nbytes = min(self.total_nbytes, target)
        return 1


def _run_loop(
    state, monkeypatch, *, seconds: float, interval_s: float = 3600, macos_level: int = 1
):
    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: macos_level)

    async def run():
        task = asyncio.ensure_future(srv._memory_pressure_loop(state, interval_s=interval_s))
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def _loop_state(bank):
    return SimpleNamespace(
        sessions=SimpleNamespace(bank=bank),
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


class TestPressureLoop:
    def test_the_death_signature_between_two_ticks_is_critical(self, monkeypatch):
        events: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_guard_event", lambda state, payload: events.append(payload)
        )
        readings = [
            TestPerChunkSupplyCheck._reading(
                None, available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=10, at_s=0.0
            ),
            TestPerChunkSupplyCheck._reading(
                None, available_gib=20, free_gib=STARVED_FREE_GIB, compressor_gib=12, at_s=1.0
            ),
        ]
        sequence = iter(readings)
        last = [None]

        def reader():
            try:
                last[0] = next(sequence)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.2, interval_s=0.01)
        # Each reading alone reads normal (20 GiB of file cache); the second
        # against the first is the crash receipts' shape.
        assert (0, "memory_pressure_critical") in bank.calls
        trims = [e for e in events if e["action"] == "pressure_trim"]
        assert trims and trims[0]["system_thrashing"] is True

    def test_steady_compression_between_ticks_is_critical_over_the_window(
        self, monkeypatch
    ):
        """80 MiB of compressor growth per reading never reached 256 MiB
        between two ticks; over the loop's ten-second window it is 320 MiB/s
        and reads CRITICAL."""

        events: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_guard_event", lambda state, payload: events.append(payload)
        )
        step = 80 / 1024
        readings = [
            TestPerChunkSupplyCheck._reading(
                None,
                available_gib=20,
                free_gib=STARVED_FREE_GIB,
                compressor_gib=10 + i * step,
                at_s=i * 0.25,
            )
            for i in range(12)
        ]
        sequence = iter(readings)
        last = [None]

        def reader():
            try:
                last[0] = next(sequence)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.3, interval_s=0.01)
        assert (0, "memory_pressure_critical") in bank.calls
        trims = [e for e in events if e["action"] == "pressure_trim"]
        assert trims and trims[0]["system_thrashing"] is True

    def test_a_warning_trim_halves_what_is_resident(self, monkeypatch):
        """#525 point C: the WARNING target was half the budget, a no-op
        for a bank already under it (421 trims in a row evicted nothing)."""

        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: TestPerChunkSupplyCheck._reading(
                None, available_gib=5, free_gib=2, compressor_gib=4, at_s=0.0, wired_gib=0
            ),
        )
        bank = _LoopBank(total=3 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert bank.calls == [(3 * GIB // 2, "memory_pressure_warning")]

    def test_a_warning_waits_for_an_idle_engine_and_critical_does_not(self, monkeypatch):
        """The Mac's own WARNING used to act at once on a busy engine. The
        admission already priced the running request to stay above the
        abort floor, so a dip under the shed floor while it runs is
        expected; trimming then only disturbs the request. Under the abort
        floor it acts at once."""

        monkeypatch.setattr(srv, "_engine_busy_signal", lambda state: True)
        warning = TestPerChunkSupplyCheck._reading(
            None, available_gib=5, free_gib=2, compressor_gib=4, at_s=0.0, wired_gib=0
        )
        monkeypatch.setattr(sm, "_reader", lambda: warning)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        _run_loop(_loop_state(bank), monkeypatch, seconds=0.05)
        assert bank.calls == []
        critical = TestPerChunkSupplyCheck._reading(
            None, available_gib=2, free_gib=1, compressor_gib=4, at_s=0.0, wired_gib=0
        )
        monkeypatch.setattr(sm, "_reader", lambda: critical)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        _run_loop(_loop_state(bank), monkeypatch, seconds=0.05)
        assert bank.calls == [(0, "memory_pressure_critical")]


class TestPlannerReserve:
    def test_the_steady_reserve_covers_the_committed_window_with_aux(self):
        """#525 point B: the reserve stopped at the dense ceiling and counted
        KV alone."""

        from mtplx.memory_plan import plan_memory

        plan = plan_memory(
            total_ram_bytes=128 * GIB,
            model_weights_bytes=FN_WEIGHTS,
            kv_bytes_per_token=FN_KV,
            aux_bytes_per_token=FN_AUX,
            model_max_context=262_144,
            requested_context=262_144,
            dense_decode_ceiling=131_072,
            usable_bytes_override=96 * GIB,
        )
        assert plan.kv_reserve_tokens == 262_144
        assert plan.kv_reserve_bytes == 262_144 * (FN_KV + FN_AUX)


# #525 (M5 Max 64 GiB, Qwen3.8-27B family, --context-window 262144,
# --paged-kv-quantization q4): 65,536 B a token of KV at full width, the
# lighter 14.9 GiB pack, a 14.2 GiB bank the WARNING trims never touched.
R525_RAM = 64 * GIB
R525_WINDOW = 262_144
R525_WEIGHTS = int(14.9 * GIB)
R525_BANK = int(14.2 * GIB)
R525_Q4 = int(Q27_KV * 0.30)


def _r525_plan():
    from mtplx.memory_plan import plan_memory

    return plan_memory(
        total_ram_bytes=R525_RAM,
        model_weights_bytes=R525_WEIGHTS,
        kv_bytes_per_token=Q27_KV,
        kv_quantization="q4",
        model_max_context=R525_WINDOW,
        requested_context=R525_WINDOW,
        dense_decode_ceiling=157_286,
    )


class Test525SixtyFourGigSeat:
    def test_the_plan_is_the_reporters(self):
        plan = _r525_plan()
        assert plan.usable_bytes == 48 * GIB
        assert plan.kv_bytes_per_token_effective == R525_Q4 == 19_660
        # "session bank up to 30.1G" for the 14.9 G pack.
        assert round(plan.bank_idle_max_bytes / GIB, 1) == 30.1

    def test_a_deep_cold_prompt_is_priced_at_full_width_through_the_q4_repage(
        self, monkeypatch
    ):
        """Point A. At the base commit this returns None: the projection
        priced the q4 width and the flat 3 GiB (33.55 + 4.58 + 3 = 41.1 GiB
        against the 46.6 GiB line), and the prefill wrote 250K bf16 rows
        into the contiguous cache (15.3 GiB) and filled the q4 pages (4.9
        GiB) while they were live, with the 14.2 GiB bank still resident:
        53.7 GiB against a 48 GiB limit."""

        monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q4")
        plan = _r525_plan()
        manager = _manager(
            max_bytes=plan.bank_idle_max_bytes, per_session_max_bytes=16 * GIB
        )
        # Two idle conversations, 14.2 GiB between them.
        _put(manager.bank, range(0, 1_000), session_id="idle-a", row_bytes=R525_BANK // 2000)
        _put(
            manager.bank, range(5_000, 6_000), session_id="idle-b", row_bytes=R525_BANK // 2000
        )
        incoming = manager.get_or_create("deep")
        assert incoming.try_begin_generation()
        # Weights and the rest of the process's Metal allocations, the bank,
        # and the 3.45 GiB pool of the report's steady-decode receipt.
        machine = _Machine(
            manager.bank, base_gib=R525_WEIGHTS / GIB + 1.0, cache_gib=3.45, host_gib=2.0
        )
        _install(monkeypatch, machine)
        state = _state(
            manager, plan=plan, runtime=_q27_runtime(), limit_gib=48, total_gib=64
        )
        prompt = list(range(10_000_000, 10_250_000))
        try:
            receipt = _admit(
                state,
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="deep",
                max_new_tokens=16_384,
                mtp_depth=2,
            )
        finally:
            incoming.end_generation()
        assert receipt is not None
        growth = receipt["growth"]
        assert growth["layout"] == "contiguous_then_repage"
        assert growth["live_prefill_bytes"] == 250_000 * Q27_KV
        assert growth["repage_copy_bytes"] == (250_000 + 16_386) * R525_Q4
        assert growth["growth_bytes"] == growth["repage_bytes"]
        assert growth["scratch_source"] == "geometry_measured"
        # The 27B's measured 2,048-row bill (2.52 to 2.72 GiB measured).
        assert growth["scratch_bytes"] == srv._dense_prefill_bill(_q27_text_args(), 2048)
        assert 2.72 * GIB < growth["scratch_bytes"] < 2.95 * GIB
        # The old projection, the q4 width and the flat reserve (7.6 GiB),
        # missed 12.6 GiB of it.
        old_growth = 250_000 * R525_Q4 + 3 * GIB
        assert growth["growth_bytes"] - old_growth > 12.5 * GIB
        # Without a shed the prefill crosses the limit itself.
        assert receipt["projected_bytes"] > 48 * GIB
        # The oldest idle conversation makes room; nothing is refused.
        assert receipt.get("refused") is not True
        assert not manager.bank.has_session_entries("idle-a")
        assert manager.bank.has_session_entries("idle-b")
        assert receipt["projected_bytes_after"] <= int(48 * GIB * 0.97)

    def test_the_steady_reserve_covers_the_committed_window(self, monkeypatch):
        """Point B. At the base commit the reserve stopped at the dense
        ceiling: 157,286 tokens (15% of 64 GiB over 65,536 B a token), 2.9
        GiB of q4 KV, and the last 105K committed tokens rode the paged lane
        unpriced."""

        from mtplx.generation import _dense_decode_max_context

        monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "auto")
        monkeypatch.setenv("MTPLX_MEMORY_BUDGET", str(R525_RAM))
        monkeypatch.delenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN", raising=False)
        monkeypatch.delenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED", raising=False)
        monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)
        assert _dense_decode_max_context() == 157_286
        plan = _r525_plan()
        assert plan.kv_reserve_tokens == R525_WINDOW
        assert plan.kv_reserve_bytes == R525_WINDOW * R525_Q4
        assert plan.bank_steady_bytes == (
            48 * GIB - R525_WEIGHTS - 3 * GIB - R525_WINDOW * R525_Q4
        )

    def test_a_warning_trim_reaches_a_bank_under_half_its_budget(self, monkeypatch):
        """Point C. macOS WARNING, the engine under its own line: at the
        base commit the target was half the 30.1 GiB budget (15.05 GiB), so
        a 14.2 GiB bank evicted nothing (421 receipts in a row). Half of what
        is resident is 7.1 GiB."""

        events: list[dict] = []
        monkeypatch.setattr(
            srv, "_record_guard_event", lambda state, payload: events.append(payload)
        )
        plan = _r525_plan()
        bank = SessionBank(
            max_entries=64,
            max_bytes=plan.bank_idle_max_bytes,
            per_session_max_bytes=16 * GIB,
        )
        for index in range(4):
            _put(
                bank,
                range(index * 10_000, index * 10_000 + 1_000),
                session_id=f"s{index}",
                row_bytes=R525_BANK // 4000,
            )
        assert bank.total_nbytes == 4 * 1_000 * (R525_BANK // 4000)
        monkeypatch.setattr(
            srv,
            "_mlx_memory_stats_live",
            lambda: {
                "ok": True,
                "active_memory_bytes": R525_WEIGHTS + GIB + int(bank.total_nbytes),
                "cache_memory_bytes": int(3.45 * GIB),
            },
        )
        monkeypatch.setattr(
            srv,
            "phys_footprint_bytes",
            lambda *a, **k: R525_WEIGHTS + GIB + int(bank.total_nbytes) + int(5.45 * GIB),
        )
        state = SimpleNamespace(
            sessions=SimpleNamespace(bank=bank),
            dashboard=SimpleNamespace(last_memory_pressure_level=0),
            metal_memory_caps={
                "memory_limit_bytes": plan.usable_bytes,
                "total_ram_bytes": R525_RAM,
            },
            memory_budget_bytes=None,
        )
        _run_loop(state, monkeypatch, seconds=0.05, macos_level=2)
        [trim] = [e for e in events if e["action"] == "pressure_trim"]
        assert trim["level"] == 2
        assert trim["level_source"] == "macos"
        assert trim["bank_entries_evicted"] == 2
        assert trim["bank_bytes_after"] <= R525_BANK // 2



# 2026-09-27 validation of c7c3c6f2 (M5 Max 128 GB, Flash-Next Optimized
# Speed, 96 GiB limit, a 12 GB desktop load): the third turn of the session
# build-up, 18,113 prompt tokens with 6,022 uncached, was refused by the
# whole-Mac rule alone. The receipt, in bytes as the guard printed them.
V_WIRED = 95_520_000_000
V_FREE = 4_040_000_000
V_FILE_BACKED = 12_150_000_000
V_COMPRESSOR = 2_740_000_000
V_SCRATCH_WIDE = 6_230_000_000  # the itemized bill at 4,096 rows
V_SCRATCH_NARROW = 3_500_000_000  # the 2,048-row bill's class
V_PROMPT = 18_113
V_CACHED = 12_091
V_ROWS = V_PROMPT * FN_ROW  # the rows the prefill writes, with the copy
V_ABORT = V_WIRED // 16
# The abort floor once the 2,048-row turn's growth is wired.
V_ABORT_NARROW = (V_WIRED + V_ROWS + V_SCRATCH_NARROW) // 16


def _v_reading(*, free: int, file_backed: int) -> sm.SystemMemory:
    total = 128 * GIB
    return sm.SystemMemory(
        available_bytes=free + file_backed,
        total_bytes=total,
        level_percent=int((total - V_WIRED - V_COMPRESSOR) * 100 // total),
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=V_WIRED,
        compressor_bytes=V_COMPRESSOR,
        swap_used_bytes=0,
        monotonic_s=0.0,
    )


class TestValidationTurn:
    """The rule the refusal broke: admit when the supply after the growth
    stays over the abort floor; reclaim first when it would fall under the
    shed floor; refuse only when it still falls under the abort floor after
    that. The chunk width is the first lever: a narrower forward is cheaper
    than anyone's state."""

    def _admit(self, monkeypatch, *, free: int, file_backed: int, **admission_kwargs):
        monkeypatch.setattr(
            srv,
            "_admission_scratch_bytes",
            lambda state, *, rows, prompt_tokens, geometry: (
                V_SCRATCH_WIDE if rows > 2048 else V_SCRATCH_NARROW,
                "qsa_itemized",
            ),
        )
        monkeypatch.setattr(sm, "_reader", lambda: _v_reading(free=free, file_backed=file_backed))
        manager = _manager()
        prompt = list(range(V_PROMPT))
        source = _put(manager.bank, prompt[:V_CACHED], session_id="anon-julian", row_bytes=FN_ROW)
        session = manager.get_or_create("anon-julian")
        assert session.try_begin_generation()
        # The engine line is fine throughout (projected 92 GB of a 100 GB line).
        machine = _Machine(manager.bank, base_gib=84.0, cache_gib=1.0, host_gib=6.0)
        _install(monkeypatch, machine)
        try:
            receipt = srv._prefill_admission_shed(
                _flash_next_state(manager),
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="anon-julian",
                prefill_chunk_tokens=4096,
                restore_mode="clone",
                **admission_kwargs,
            )
        finally:
            session.end_generation()
        return receipt, manager, source

    def test_the_refused_turn_runs_at_the_narrower_chunk(self, monkeypatch):
        """At c7c3c6f2 this is refused: 6.82 GB of growth plus the 5.97 GB
        abort floor plus a 5.97 GB request-sized margin against 16.19 GB.
        The floors are the Mac's once the growth is wired. At 4,096 rows it
        would leave 9.37 GB, under a 12.79 GB shed floor; at 2,048 rows it
        leaves 12.10 GB, over the 6.23 GB abort floor and just under the
        12.45 GB shed floor, so the engine gives back its allocator pool and
        runs it at 2,048 rows. The conversation's banked state stays."""

        receipt, manager, source = self._admit(
            monkeypatch, free=V_FREE, file_backed=V_FILE_BACKED
        )
        assert receipt.get("refused") is not True
        assert receipt["prefill_chunk_requested"] == 4096
        assert receipt["prefill_chunk_tokens"] == 2048
        assert receipt["reclamation_steps"][0] == "allocator_pool"
        assert receipt["growth_by_chunk"] == {
            "4096": V_ROWS + V_SCRATCH_WIDE,
            "2048": V_ROWS + V_SCRATCH_NARROW,
        }
        # The floors are computed after the growth: at the requested chunk,
        # then at the one it runs.
        wide_abort = (V_WIRED + V_ROWS + V_SCRATCH_WIDE) // 16
        assert receipt["system_abort_floor_bytes"] == wide_abort
        assert receipt["system_shed_floor_bytes"] == 2 * wide_abort
        assert receipt["system_abort_floor_bytes_after"] == V_ABORT_NARROW
        assert receipt["system_shed_floor_bytes_after"] == 2 * V_ABORT_NARROW
        assert receipt["system_shortfall_bytes_after"] == 0
        assert receipt["growth"]["restore_copy_bytes"] == V_CACHED * FN_ROW
        assert source.token_ids in manager.bank._entries

    def test_a_turn_clear_of_both_floors_only_narrows(self, monkeypatch):
        """With 0.5 GB more supply the 2,048-row chunk clears the shed floor
        too: the engine's free pool goes first (it takes nothing from
        anyone), then the narrower chunk is the whole answer and no state
        is taken. The receipt names both steps in the order they ran."""

        receipt, manager, source = self._admit(
            monkeypatch, free=V_FREE + 500_000_000, file_backed=V_FILE_BACKED
        )
        assert receipt["prefill_chunk_tokens"] == 2048
        assert receipt["reclamation_steps"] == ["allocator_pool", "narrower_prefill_chunk"]
        assert receipt["early_pool_clear"]["cache_bytes_after"] == 0
        assert "cache_cleared" not in receipt
        assert source.token_ids in manager.bank._entries

    def test_between_the_floors_it_runs_narrow_after_reclamation(self, monkeypatch):
        """11 GB of supply: the 2,048-row chunk leaves 6.9 GB, over the
        abort floor and under the shed floor. The engine gives back its
        pool first, then runs the request at the narrower chunk."""

        receipt, _manager_, _source = self._admit(
            monkeypatch, free=2_000_000_000, file_backed=9_000_000_000
        )
        assert receipt.get("refused") is not True
        assert receipt["prefill_chunk_tokens"] == 2048
        assert receipt["reclamation_steps"][0] == "allocator_pool"
        assert receipt["system_shortfall_bytes_after"] == 0

    def test_under_the_abort_floor_after_reclamation_it_is_refused(self, monkeypatch):
        """0.1 GB under the abort floor at the narrowest chunk, with nothing
        left to give back: refused, and the shortfall is the request's own
        growth against the abort floor (c7c3c6f2 reported 8.8 GB: the wide
        chunk, the floor and a margin the size of the request)."""

        supply = V_ROWS + V_SCRATCH_NARROW + V_ABORT_NARROW - 100_000_000
        receipt, _manager_, _source = self._admit(
            monkeypatch, free=2_000_000_000, file_backed=supply - 2_000_000_000
        )
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "system_memory_short_after_reclamation"
        assert receipt["system_shortfall_bytes_after"] == 100_000_000
        assert receipt["prefill_chunk_tokens"] == 2048
        assert receipt["system_abort_floor_bytes_after"] == V_ABORT_NARROW

    def test_a_turn_that_would_trip_late_is_refused_before_it_starts(self, monkeypatch):
        """0.1 GB over the abort floor of the Mac as it is, and 0.16 GB under
        the one it will have once the 4.09 GB of growth is wired (the floor
        the per-chunk check reads by the end of the prefill): admitted by
        54e01d1f to trip late in its own prefill, refused now before any
        work is done."""

        supply = V_ROWS + V_SCRATCH_NARROW + V_ABORT + 100_000_000
        receipt, _manager_, _source = self._admit(
            monkeypatch, free=2_000_000_000, file_backed=supply - 2_000_000_000
        )
        assert receipt["refused"] is True
        assert receipt["system_shortfall_bytes_after"] == (
            V_ABORT_NARROW - V_ABORT - 100_000_000
        )

    def test_the_admission_hands_its_chunk_to_the_per_chunk_check(self, monkeypatch):
        """The per-chunk check reserves what the admission priced for one
        forward: the 2,048-row chunk it narrowed to, with that chunk's
        scratch, not the 4,096-row one it was asked for."""

        pricing: dict = {}
        receipt, manager, _source = self._admit(
            monkeypatch, free=V_FREE, file_backed=V_FILE_BACKED, pricing=pricing
        )
        assert receipt["prefill_chunk_tokens"] == 2048
        narrow = 2048 * FN_ROW + V_SCRATCH_NARROW
        assert pricing["growth"]["chunk_bytes"] == narrow
        assert pricing["growth"] == receipt["growth"]
        assert (
            srv._prefill_chunk_reserve_bytes(
                _flash_next_state(manager),
                prompt_tokens=V_PROMPT,
                chunk_tokens=4096,
                priced=pricing["growth"],
            )
            == narrow
        )

    def test_an_admission_with_nothing_to_do_still_hands_its_bill(self, monkeypatch):
        """A roomy Mac: nothing is probed or reclaimed, and the bill is the
        cheap worst case the admission cleared (the whole prompt new, at the
        requested chunk)."""

        pricing: dict = {}
        monkeypatch.setattr(
            srv,
            "_admission_scratch_bytes",
            lambda state, *, rows, prompt_tokens, geometry: (
                V_SCRATCH_WIDE if rows > 2048 else V_SCRATCH_NARROW,
                "qsa_itemized",
            ),
        )
        monkeypatch.setattr(
            sm, "_reader", lambda: _v_reading(free=20_000_000_000, file_backed=V_FILE_BACKED)
        )
        manager = _manager()
        machine = _Machine(manager.bank, base_gib=84.0, cache_gib=1.0, host_gib=6.0)
        _install(monkeypatch, machine)
        receipt = srv._prefill_admission_shed(
            _flash_next_state(manager),
            prompt_ids=list(range(V_PROMPT)),
            session_bank=manager.bank,
            session_id="anon-julian",
            prefill_chunk_tokens=4096,
            restore_mode="clone",
            pricing=pricing,
        )
        assert receipt is None
        assert pricing["growth"]["prefill_chunk_tokens"] == 4096
        assert pricing["growth"]["chunk_bytes"] == 4096 * FN_ROW + V_SCRATCH_WIDE

    def test_without_the_admissions_bill_the_widest_forward_is_reserved(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            srv,
            "_admission_scratch_bytes",
            lambda state, *, rows, prompt_tokens, geometry: (
                V_SCRATCH_WIDE if rows > 2048 else V_SCRATCH_NARROW,
                "qsa_itemized",
            ),
        )
        state = _flash_next_state(_manager())
        assert (
            srv._prefill_chunk_reserve_bytes(
                state, prompt_tokens=V_PROMPT, chunk_tokens=4096, priced=None
            )
            == 4096 * FN_ROW + V_SCRATCH_WIDE
        )
        # A prompt shorter than the chunk is one forward of its own rows.
        assert (
            srv._prefill_chunk_reserve_bytes(
                state, prompt_tokens=1000, chunk_tokens=4096, priced=None
            )
            == 1000 * FN_ROW + V_SCRATCH_NARROW
        )

    def test_the_request_arms_its_check_with_that_bill(self, monkeypatch):
        import inspect

        src = inspect.getsource(srv._run_generation)
        admitted = src.index("admission_shed = _prefill_admission_shed(")
        handed = src.index("pricing=admission_pricing,")
        narrowed = src.index(
            'prefill_chunk_tokens = int(admission_shed["prefill_chunk_tokens"])'
        )
        armed = src.index("prefill_system_guard = make_prefill_system_guard(")
        reserved = src.index('priced=admission_pricing.get("growth"),')
        assert admitted < handed < narrowed < armed < reserved
        # Both generation and scoring construct the guard through the same
        # helper. Check the actual bill and post-forward plan handed to it.
        bill = {"chunk_bytes": 123}
        seen = []

        def reserve(_state, **kwargs):
            seen.append(("reserve", kwargs))
            return bill["chunk_bytes"]

        def after(_state, **kwargs):
            seen.append(("after", kwargs))
            return {"after_prefill_reserve_bytes": 23, "restore_bytes": 31}

        monkeypatch.setattr(srv, "_prefill_chunk_reserve_bytes", reserve)
        monkeypatch.setattr(srv, "_prefill_after_forward_plan", after)
        guard = srv.make_prefill_system_guard(
            _flash_next_state(_manager()), prompt_tokens=4096, chunk_tokens=256, priced=bill
        )
        assert [name for name, _kwargs in seen] == ["reserve", "after"]
        assert all(kwargs["priced"] is bill for _name, kwargs in seen)
        assert all(kwargs["chunk_tokens"] == 256 for _name, kwargs in seen)
        assert guard.chunk_reserve_bytes == 123
        assert guard.after_prefill_reserve_bytes == 23
        assert guard.restore_bytes == 31

    def test_the_chosen_width_reaches_the_prefill(self):
        """The admission's width is the one the prefill runs: generation
        reads it back before it enters the chunk override."""

        import inspect

        src = inspect.getsource(srv._run_generation)
        admitted = src.index("admission_shed = _prefill_admission_shed(")
        narrowed = src.index(
            'prefill_chunk_tokens = int(admission_shed["prefill_chunk_tokens"])'
        )
        override = src.index("prefill_chunk_size_override(prefill_chunk_tokens)")
        assert admitted < narrowed < override


class TestRepeatedWarnings:
    """The review of 9c96dd9c: repeated WARNINGs halve the bank again and
    again, with no test of when that stops. It stops when the Mac does: one
    trim per re-arm period (120 s) while the level stays elevated, none once
    it reads normal, so the bank empties only if the Mac stays short for
    that many periods."""

    def test_one_trim_per_period_while_short_and_none_after(self):
        guard = srv._MemoryPressureGuard()
        trims = [t for t in range(0, 600, 10) if guard.decide(2, float(t), False)]
        assert trims == [0, 120, 240, 360, 480]
        # The Mac recovers: no trim, however long it stays normal.
        assert not any(guard.decide(1, float(t), False) for t in range(600, 1200, 10))

    def test_the_bank_halves_once_per_trim_and_stops_with_the_mac(self, monkeypatch):
        bank = _LoopBank(total=8 * GIB, max_bytes=16 * GIB)
        short = [True]

        def reading():
            available = 5 if short[0] else 40
            return TestPerChunkSupplyCheck._reading(
                None,
                available_gib=available,
                free_gib=2,
                compressor_gib=4,
                at_s=0.0,
                wired_gib=0,
            )

        monkeypatch.setattr(sm, "_reader", reading)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert bank.calls == [(4 * GIB, "memory_pressure_warning")]
        short[0] = False
        _run_loop(state, monkeypatch, seconds=0.05)
        # A fresh loop on a Mac with room trims nothing.
        assert bank.calls == [(4 * GIB, "memory_pressure_warning")]
        assert bank.total_nbytes == 4 * GIB


def _flash_next_speed_lane(monkeypatch):
    """The Flash-Next speed profile's sparse prefill: forwards of 2,048 rows
    or more go sparse from 16K of history, narrower ones from 32K."""

    import mtplx.models.qwen4_exp as qwen4

    monkeypatch.setattr(qwen4, "_qsa_prefill_enabled", lambda: True)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "16384")
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_MIDLOOP_EVAL", "4")


def _chunk_bill(state, rows: int, prompt_tokens: int) -> int:
    geometry = srv._admission_geometry(state, prefill_width=rows)
    scratch, _source = srv._admission_scratch_bytes(
        state, rows=rows, prompt_tokens=prompt_tokens, geometry=geometry
    )
    return srv._admission_chunk_bytes(geometry, rows, scratch)


def test_a_narrower_last_chunk_that_costs_more_is_what_the_check_reserves(monkeypatch):
    # The review of 425ffc58: 32,409 tokens at 2,048 rows leave a 1,688-row
    # last chunk, which stays on the dense lane at 30,720 of history and is
    # billed more than a full chunk. The 2026-10-01 replay measured it: the
    # 1,536-row last chunk of a 32,257-token turn took the peak 0.95 GB past
    # the full chunks' while the check reserved a full chunk.
    manager = _manager()
    _install(monkeypatch, _Machine(manager.bank, base_gib=84, cache_gib=0, host_gib=0))
    state = _flash_next_state(manager)
    _flash_next_speed_lane(monkeypatch)
    full = _chunk_bill(state, 2048, 32_409)
    last = _chunk_bill(state, 1688, 32_409)
    assert last > full

    bill = srv._prefill_forward_bill(
        state, new_tokens=32_409, width=2048, prompt_tokens=32_409,
        geometry=srv._admission_geometry(state, prefill_width=2048),
    )
    assert bill["rows"] == 1688 and bill["chunk_bytes"] == last

    guard = srv.make_prefill_system_guard(state, prompt_tokens=32_409, chunk_tokens=2048, priced=None)
    assert guard.chunk_reserve_bytes == last
    guard.note_prefill_progress({"phase": "chunk", "tokens_done": 30_720, "tokens_total": 32_409})
    assert guard._evaluate()["reserve"] == last


def test_a_wide_chunk_keeps_its_own_bill_over_a_cheaper_last_chunk(monkeypatch):
    # The founder's turn at 4,096 rows: its 3,585-row last chunk goes sparse
    # like the others and costs less, so nothing changes for it.
    manager = _manager()
    _install(monkeypatch, _Machine(manager.bank, base_gib=84, cache_gib=0, host_gib=0))
    state = _flash_next_state(manager)
    _flash_next_speed_lane(monkeypatch)
    bill = srv._prefill_forward_bill(
        state, new_tokens=32_258, width=4096, prompt_tokens=32_258,
        geometry=srv._admission_geometry(state, prefill_width=4096),
    )
    assert bill["rows"] == 4096
    assert bill["chunk_bytes"] == _chunk_bill(state, 4096, 32_258)


def test_prompt_scoring_keeps_its_reserve_until_the_last_token_is_scored(monkeypatch):
    # The review of 425ffc58: 257 tokens score as 256 then 1; the first
    # chunk's progress is not the end of the forwards.
    manager = _manager()
    _install(monkeypatch, _Machine(manager.bank, base_gib=20, cache_gib=0, host_gib=0))
    state = _flash_next_state(manager)
    guard = srv.make_prefill_system_guard(
        state, prompt_tokens=257, chunk_tokens=256, priced=None, prompt_scoring=True,
    )
    guard.note_prefill_progress({"phase": "chunk", "tokens_done": 256, "tokens_total": 257})
    assert guard.prefill_done_by is None
    assert guard._evaluate()["reserve"] > 0
    guard.note_prefill_progress({"phase": "chunk", "tokens_done": 257, "tokens_total": 257})
    assert guard.prefill_done_by == "prefill_progress"


def test_the_early_move_to_ssd_needs_an_ssd_cache_that_reads_back(tmp_path, monkeypatch):
    # The review of 425ffc58: a write-only SSD cache publishes but never
    # restores, so moving a conversation's entry there loses its only
    # usable copy.
    from mtplx.cache_bank.cold_tier import SessionBankColdTier

    manager = _manager()
    bank = manager.bank
    entry = _put(bank, range(2000), session_id="pi", row_bytes=32, live_cache=True)
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="write-only", min_prefix_tokens=1)
    bank.cold_tier = tier
    monkeypatch.setattr(tier, "_entry_in_manifest", lambda entry_id: True)
    try:
        assert bank.entry_is_durable(entry)
        moved = srv._admission_move_own_durable_entries(
            bank, session_id="pi", probe_ids=[9999] * 100,
            restore_keys=set(), reused_tokens=0,
        )
        assert moved is None
        assert entry.token_ids in bank._entries
    finally:
        tier.close()
