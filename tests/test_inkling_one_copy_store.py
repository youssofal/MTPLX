"""Inkling 2 Mini's conversation is kept once (mtplx/one_copy.py, ``one_copy_store``).

Its decode writes the global banks in place, and a banked snapshot held views of the live
buffers: MLX cannot write a buffer in place while a view shares it, so the first write after
the prompt's publish (and every follow-up turn's first write after the answer's commit)
copied every global visit's whole history, 32 KB a token held twice (Bare Speed through
`mtplx serve`, 21 November: 2.28 GB of a 64K prompt's decode start). These tests pin the
one-copy store on Inkling's caches: a window ring, its convolution tails and a global visit,
the same cache classes the model makes.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.cache_state import snapshot_untrimmable_cache
from mtplx.models.inkling.cache import WindowCache
from mtplx.models.inkling.runtime import InklingRuntime
from mtplx.one_copy import (
    one_copy_runtime,
    one_copy_store,
    one_copy_store_cache,
    prompt_lease_fields,
)
from mtplx.session_bank import SessionBank

WINDOW, HEADS, DIM = 8, 2, 16

RUNTIME = SimpleNamespace(
    model_path="tiny-inkling",
    mtp_enabled=True,
    make_cache=lambda: [WindowCache(max_size=WINDOW), ArraysCache(4), KVCache(), ArraysCache(4)],
    make_mtp_cache=lambda: [],
)


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _rows(count: int, seed: int) -> mx.array:
    return mx.random.normal((1, HEADS, count, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)


def _tails(seed: int) -> ArraysCache:
    tails = ArraysCache(4)
    for slot in range(4):
        tails[slot] = mx.random.normal((1, 3, 12), key=mx.random.key(seed + slot))
    return tails


def _conversation(tokens: int, *, seed: int = 0) -> list:
    """One window visit and one global visit after a ``tokens``-token prompt read."""

    window, global_ = WindowCache(max_size=WINDOW), KVCache()
    window.update_and_fetch(_rows(tokens, seed), _rows(tokens, seed + 1))
    global_.update_and_fetch(_rows(tokens, seed + 2), _rows(tokens, seed + 3))
    cache = [window, _tails(seed + 10), global_, _tails(seed + 20)]
    mx.eval(*_leaves(cache))
    return cache


def _leaves(cache) -> list[mx.array]:
    out = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out += [leaf for leaf in entry.cache if leaf is not None]
        else:
            out += [entry.keys, entry.values]
    return out


def _decode(cache, tokens: int, *, seed: int = 50) -> None:
    """What an answer decoding into the live cache does: one row at a time into the ring and
    the global visit, new convolution tails."""

    for step in range(tokens):
        cache[0].update_and_fetch(_rows(1, seed + 4 * step), _rows(1, seed + 4 * step + 1))
        cache[2].update_and_fetch(_rows(1, seed + 4 * step + 2), _rows(1, seed + 4 * step + 3))
        for tails in (cache[1], cache[3]):
            for slot in range(4):
                tails[slot] = tails[slot] + 1.0
    mx.eval(*_leaves(cache))


def _address(value: mx.array) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint16), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _bits(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value).copy()


def _bank() -> SessionBank:
    return SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)


def _put_answer(bank, cache, tokens, **extra):
    return bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=cache, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s",
        snapshot_epoch=len(tokens), **extra,
    )


def test_inkling_caches_and_runtime_are_kept_by_the_one_copy_store(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    assert one_copy_store_cache(RUNTIME.make_cache())
    assert not one_copy_store_cache([KVCache(), ArraysCache(4)])
    assert not one_copy_store_cache([])
    assert one_copy_store(InklingRuntime) and not one_copy_runtime(InklingRuntime)
    assert not one_copy_store(SimpleNamespace())
    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    assert not one_copy_store_cache(RUNTIME.make_cache())
    assert not one_copy_store(InklingRuntime)


def test_the_answer_is_banked_as_a_lease_and_the_next_write_is_in_place(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    entry = _put_answer(bank, cache, list(range(20)))
    assert entry.live_ref_only and entry.cache_ref is cache
    assert all(state is None for state in entry.cache_snapshot.states)
    assert entry.lease_kv_offset == 20
    before = _address(cache[2].keys)
    _decode(cache, 1)
    # No view shares the global visit's buffer: the row is written where the history is.
    assert _address(cache[2].keys) == before


def test_the_copying_store_behind_the_switch_holds_views_and_copies_at_the_next_write(monkeypatch):
    """What the one-copy store removes, measured on the same caches with it switched off."""

    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    bank = _bank()
    cache = _conversation(20)
    entry = _put_answer(bank, cache, list(range(20)))
    assert not entry.live_ref_only and entry.cache_snapshot.states[2] is not None
    before = _address(cache[2].keys)
    _decode(cache, 1)
    assert _address(cache[2].keys) != before


@pytest.mark.parametrize("extends", [False, True])
def test_a_prompt_lease_the_answer_ran_past_restores_through_its_anchor(monkeypatch, extends):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    anchor = snapshot_untrimmable_cache(cache)
    want = {i: [_bits(leaf) for leaf in state] for i, state in enumerate(anchor.states) if state is not None}
    prompt = list(range(20))
    fields = prompt_lease_fields(
        cache, committed_mtp_cache=None, hidden=mx.ones((1, 1, 8)), prompt_len=len(prompt), boundaries=[],
    )
    entry = bank.put(
        runtime=RUNTIME, token_ids=prompt, cache=cache, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), session_id="s", snapshot_epoch=len(prompt), **fields,
    )
    assert entry.live_ref_only and entry.lease_kv_offset == 20
    _decode(cache, 6)  # the answer decodes into the leased cache and is never committed
    assert cache[2].offset == 26 and cache[0].offset == 26
    lookup = prompt + [900, 901] if extends else prompt
    restored = bank.restore(RUNTIME, lookup, mode="reference", session_id="s")
    assert restored is not None and restored.restore_mode == "reference_lease"
    assert restored.cache is cache and entry.cache_ref is None
    assert cache[2].offset == 20 and cache[0].offset == 20
    for index, bits in want.items():
        state = cache[index].state
        leaves = list(state) if isinstance(state, (list, tuple)) else [state]
        assert all(np.array_equal(_bits(leaf), expect) for leaf, expect in zip(leaves, bits)), index


def test_another_session_gets_a_copy_and_the_owner_keeps_its_lease(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    cache = _conversation(20)
    prompt = list(range(20))
    entry = _put_answer(bank, cache, prompt)
    other = bank.restore(RUNTIME, prompt + [5], mode="reference", session_id="other")
    assert other is not None and other.cache is not cache
    assert entry.cache_ref is cache
    owner = bank.restore(RUNTIME, prompt + [6], mode="reference", session_id="s")
    assert owner is not None and owner.cache is cache and owner.restore_mode == "reference_lease"


def test_the_prompt_anchor_is_the_reads_own_checkpoint_at_the_prompts_end(monkeypatch):
    """The read keeps a checkpoint at the prompt's end (copied and evaluated at capture); the lease's
    anchor is that record rather than a second 180 MB copy beside it for the whole answer."""

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    cache = _conversation(20)
    hidden = mx.ones((1, 1, 8))
    early = (12, snapshot_untrimmable_cache(cache), None)
    at_end = (20, snapshot_untrimmable_cache(cache), None)
    inkling = SimpleNamespace(one_copy_store=True)  # opted in; not the fixed-M4 verifier's runtime
    fields = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                                 boundaries=[early, at_end], runtime=inkling)
    records = fields["gdn_boundaries"]
    assert [int(r[0]) for r in records] == [12, 20]
    assert records[-1][1] is at_end[1] and records[-1][2] is hidden
    # Without one at the end, the anchor is copied from the cache as before.
    fresh = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                                boundaries=[early], runtime=inkling)
    anchor = fresh["gdn_boundaries"][-1][1]
    assert anchor is not at_end[1] and anchor.states[2] is None and anchor.states[0] is not None
    # The fixed-M4 verifier's runtime keeps its own copy whatever the read kept: its path is the base's.
    verifier = SimpleNamespace(qwen4_fixed_m4_compiled_verify=True)
    assert one_copy_runtime(verifier)
    own = prompt_lease_fields(cache, committed_mtp_cache=None, hidden=hidden, prompt_len=20,
                              boundaries=[early, at_end], runtime=verifier)
    assert [int(r[0]) for r in own["gdn_boundaries"]] == [12, 20]
    assert own["gdn_boundaries"][-1][1] is not at_end[1]


@pytest.mark.parametrize("inkling", [True, False], ids=["inkling", "runtime_without_hook"])
def test_partial_restore_consumes_the_seed_according_to_the_runtime(inkling):
    from mtplx.generation import _restore_near_prefix_prompt_state

    prefix = list(range(64))
    cache = _conversation(len(prefix))
    boundary = snapshot_untrimmable_cache(cache)
    _decode(cache, 1, seed=100)
    bank = _bank()
    bank.put(
        runtime=RUNTIME, token_ids=prefix + [100], cache=cache, logits=mx.zeros((1, 16)),
        hidden=None, hidden_variant="post_norm", mtp_history_policy="cycle",
        gdn_boundaries=[(len(prefix), boundary, None)],
    )
    forwarded = []

    def forward(ids, *, cache, **kwargs):
        for token in ids.tolist()[0]:
            forwarded.append(token)
            _decode(cache, 1, seed=token)
        return mx.zeros((1, 1, 16)), cache[1][0][:, -1:, :]

    runtime = SimpleNamespace(**vars(RUNTIME), forward_ar=forward)
    if inkling:
        runtime.session_restore_uses_bank_boundary = InklingRuntime.session_restore_uses_bank_boundary
        runtime.note_session_restore_decline = InklingRuntime.note_session_restore_decline
    prompt = prefix + [101]
    state = _restore_near_prefix_prompt_state(
        runtime, prompt, session_bank=bank, base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm", mtp_history_policy="cycle", template_hash=None,
        draft_head_identity=None, policy_fingerprint=None,
    )
    assert state is not None and state.cached_tokens == len(prefix)
    expected_ids = [101] if inkling else [63, 101]
    assert forwarded == expected_ids
    assert state.restore_served["boundary_restore"] is inkling
    expected = _conversation(len(prefix))
    for token in expected_ids:
        _decode(expected, 1, seed=token)
    for actual, wanted in zip(state.trunk_cache, expected, strict=True):
        if hasattr(wanted, "offset"):
            assert actual.offset == wanted.offset == len(prefix) + len(expected_ids)
        for a, b in zip(actual.state, wanted.state, strict=True):
            np.testing.assert_array_equal(_bits(a), _bits(b))


def test_mislabeled_saved_rows_are_refused_but_an_earlier_checkpoint_can_restore():
    from mtplx.models.inkling.fast.routes import capture_counts

    runtime = object.__new__(InklingRuntime)
    runtime.model_path = RUNTIME.model_path
    runtime.mtp_enabled = False
    runtime.trunk_entries = 4
    runtime.drafter = None
    runtime.model = SimpleNamespace(
        config={"sliding_window_size": WINDOW}, visit_order=(0, 1), make_cache=RUNTIME.make_cache,
        layers=[SimpleNamespace(sliding=True), SimpleNamespace(sliding=False)],
    )
    cache = _conversation(63)
    good = snapshot_untrimmable_cache(cache)
    _decode(cache, 2)
    bad = snapshot_untrimmable_cache(cache)  # The released writer labeled 65 rows as 64.
    _decode(cache, 1)
    bank = _bank()
    tokens = list(range(65))
    entry = bank.put(
        runtime=runtime, token_ids=tokens, cache=cache, logits=mx.zeros((1, 16)), hidden=None,
        gdn_boundaries=[(63, good, None), (64, bad, None)],
    )
    with capture_counts() as counts:
        assert bank.restore(runtime, tokens) is None
        assert bank.restore_entry_prefix_cache(runtime, entry, 64) is None
        restored = bank.restore_entry_prefix_cache(runtime, entry, 63)
        assert restored is not None
        assert restored[0][0].offset == restored[0][2].offset == 63
    assert counts["restore.decline.snapshot_length"] == 2


@pytest.mark.parametrize(
    ("policy", "reason"),
    [("cycle", "missing_boundary_logits"), ("committed", "missing_boundary_hidden")],
)
def test_boundary_refusal_is_counted_in_the_request_log(policy, reason):
    from mtplx.cache_state import snapshot_cache
    from mtplx.generation import _restore_near_prefix_prompt_state
    from mtplx.models.inkling.fast.decoder import observe_fast_request

    runtime = SimpleNamespace(
        **vars(RUNTIME), session_restore_uses_bank_boundary=True,
        note_session_restore_decline=InklingRuntime.note_session_restore_decline,
    )
    prefix = list(range(512))
    cache = _conversation(len(prefix))
    boundary = snapshot_untrimmable_cache(cache)
    _decode(cache, 1)
    bank = _bank()
    bank.put(
        runtime=runtime, token_ids=prefix + [999], cache=cache, logits=mx.zeros((1, 16)),
        hidden=None, hidden_variant="post_norm", mtp_history_policy=policy,
        mtp_history_snapshot=snapshot_cache([]) if policy == "committed" else None,
        gdn_boundaries=[(len(prefix), boundary, None)],
    )

    def restore(rt):
        state = _restore_near_prefix_prompt_state(
            rt, prefix, session_bank=bank, base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm", mtp_history_policy=policy, template_hash=None,
            draft_head_identity=None, policy_fingerprint=None,
        )
        assert state is None
        return SimpleNamespace(stats=SimpleNamespace())

    assert observe_fast_request(restore, runtime).stats.inkling_fast[f"restore.decline.{reason}"] == 1


def test_clone_restored_inkling_turn_preserves_prompt_boundary_logits_for_repeat(monkeypatch):
    """A clone-restored Inkling turn with zero suffix tokens skips
    ``_commit_ar_prompt_state`` before decode, so no prompt lease sits beside
    the answer lease. Carrying ``prompt_boundary_logits`` on
    ``GenerationFinalState`` and banking it on the generation-final entry keeps
    a follow-up repeat of the same prompt from declining with
    ``restore.decline.missing_boundary_logits``."""
    from mtplx.generation import _restore_near_prefix_prompt_state
    from mtplx.models.inkling import generation as inkling_gen
    from mtplx.models.inkling.fast.decoder import observe_fast_request
    from mtplx.sampling import SamplerConfig
    from mtplx.server import openai as server

    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    monkeypatch.setenv("MTPLX_INKLING_FAST", "off")
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")

    vocab = 16
    prompt = list(range(20))
    prompt_logits = mx.linspace(-1.0, 2.0, vocab, dtype=mx.float32).reshape(1, vocab)
    mx.eval(prompt_logits)
    first_token = int(mx.argmax(prompt_logits[0]).item())
    next_token = (first_token + 3) % vocab

    # Seed the bank as if Turn 1 (session "s1") banked a prompt lease at len(prompt)
    # and then committed its answer lease (prompt + [101, 102]), keeping both the
    # prompt-end boundary snapshot and boundary_logits at len(prompt).
    bank = _bank()
    seed_cache = _conversation(len(prompt), seed=7)
    prompt_fields = prompt_lease_fields(
        seed_cache,
        committed_mtp_cache=None,
        hidden=mx.ones((1, 1, 8)),
        prompt_len=len(prompt),
        boundaries=[],
    )
    bank.put(
        runtime=RUNTIME,
        token_ids=prompt,
        cache=seed_cache,
        logits=prompt_logits,
        hidden=mx.ones((1, 1, 8)),
        hidden_variant="post_norm",
        session_id="s1",
        mtp_history_policy="cycle",
        snapshot_epoch=len(prompt),
        **prompt_fields,
    )
    _decode(seed_cache, 2, seed=200)
    seed_entry = bank.put(
        runtime=RUNTIME,
        token_ids=prompt + [101, 102],
        cache=seed_cache,
        logits=mx.zeros((1, vocab), dtype=mx.float32),
        hidden=mx.ones((1, 1, 8)),
        hidden_variant="post_norm",
        keep_live_ref=True,
        session_id="s1",
        mtp_history_policy="cycle",
        snapshot_epoch=len(prompt) + 2,
    )
    assert seed_entry is not None
    assert seed_entry.boundary_logits is not None
    assert seed_entry.boundary_logits[0] == len(prompt)

    class _ToyTokenizer:
        eos_token_id = None

        def __init__(self):
            self._specials: dict[str, int] = {}

        def convert_tokens_to_ids(self, text):
            return self._specials.setdefault(text, 900 + len(self._specials))

        def decode(self, ids, **_kwargs):
            return ",".join(str(int(t)) for t in ids)

    class _ToyDrafter:
        width = 4

        def propose(self, _model, _anchor, _cache, count, **_kwargs):
            return mx.array([next_token] * count, dtype=mx.int32)

    class _ToyInklingRuntime:
        model_path = RUNTIME.model_path
        mtp_enabled = True
        one_copy_store = True
        session_restore_uses_bank_boundary = True
        trunk_entries = 4
        tokenizer = _ToyTokenizer()
        drafter = _ToyDrafter()
        model = SimpleNamespace()
        contract = SimpleNamespace(
            base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm",
            mtp_position_mode="cache",
        )
        fast = None

        @staticmethod
        def make_cache():
            return RUNTIME.make_cache()

        @staticmethod
        def make_mtp_cache():
            return []

        @staticmethod
        def note_session_restore_decline(reason):
            InklingRuntime.note_session_restore_decline(reason)

        def verify(self, block, _cache):
            width = len(block) if isinstance(block, list) else int(block.shape[0])
            rows = mx.full((width, vocab), -10.0, dtype=mx.float32)
            rows[:, next_token] = 5.0
            features = [mx.ones((1, width, 8), dtype=mx.float32)]
            return rows, features, [{}]

        def commit(self, cache, _features, _tapes, kept, _total):
            _decode(cache, kept, seed=300)

        def forward_ar(self, ids, *, cache, return_hidden=False, **_kwargs):
            _decode(cache, int(ids.shape[-1]), seed=400)
            row = mx.full((1, 1, vocab), -10.0, dtype=mx.float32)
            row[:, :, next_token] = 5.0
            if return_hidden:
                return row, mx.ones((1, 1, 8), dtype=mx.float32)
            return row

    rt = _ToyInklingRuntime()
    prompt_commits = []
    real_commit = inkling_gen._commit_ar_prompt_state

    def spy_commit(*args, **kwargs):
        prompt_commits.append(True)
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(inkling_gen, "_commit_ar_prompt_state", spy_commit)

    # Turn 2: different session ("s2") repeats the identical prompt -> restores via
    # clone with suffix_tokens == 0, skipping _commit_ar_prompt_state.
    out = inkling_gen.generate(
        rt,
        list(prompt),
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        depth=1,
        seed=0,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        session_bank=bank,
        session_id="s2",
        session_restore_mode="reference",
        session_template_hash=None,
        session_draft_head_identity=None,
        session_policy_fingerprint=None,
        capture_final_state=True,
        commit_prompt_state_to_bank=True,
        vision_splice=None,
        constraint=None,
        abort_check=None,
        first_token_logprobs_top_k=None,
        store_prefix_snapshot=False,
    )

    assert out.stats.session_cache_hit is True
    assert out.stats.new_prefill_tokens == 0
    assert str(out.stats.session_restore_mode).endswith("clone")
    assert prompt_commits == []
    assert out.tokens == [first_token, next_token]
    assert out.final_state is not None
    assert out.final_state.prompt_boundary_logits is not None
    assert np.array_equal(_bits(out.final_state.prompt_boundary_logits), _bits(prompt_logits))

    # Bank Turn 2's generation-final state through the server's values helper.
    srv_state = SimpleNamespace(backend_descriptor=SimpleNamespace(backend_id="inkling2_loop"))
    values = server._generation_final_bank_values(
        srv_state,
        out.final_state,
        prompt_ids=prompt,
        final_token_ids=prompt + list(out.tokens),
    )
    assert "boundary_logits" in values
    assert values["boundary_logits"][0] == len(prompt)
    token_ids = values.pop("token_ids")
    answer_entry = bank.put(
        runtime=rt,
        token_ids=token_ids,
        keep_live_ref=True,
        session_id="s2",
        hidden_variant="post_norm",
        mtp_history_policy="cycle",
        snapshot_epoch=len(token_ids),
        **values,
    )
    assert answer_entry is not None
    assert answer_entry.boundary_logits is not None
    assert answer_entry.boundary_logits[0] == len(prompt)
    assert np.array_equal(_bits(answer_entry.boundary_logits[1]), _bits(prompt_logits))

    # Drop Turn 1's entry so Turn 3 MUST restore from Turn 2's clone-restored entry.
    bank._evict_entry(seed_entry, reason="test_force_turn2_entry")

    def repeat_restore(runtime):
        state = _restore_near_prefix_prompt_state(
            runtime,
            prompt,
            session_bank=bank,
            base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm",
            mtp_history_policy="cycle",
            template_hash=None,
            draft_head_identity=None,
            policy_fingerprint=None,
            session_id="s3",
        )
        assert state is not None
        assert state.cached_tokens == len(prompt)
        assert state.suffix_tokens == 0
        assert np.array_equal(_bits(state.logits), _bits(prompt_logits))
        return SimpleNamespace(stats=SimpleNamespace())

    observed = observe_fast_request(repeat_restore, rt)
    assert observed.stats.inkling_fast.get("restore.decline.missing_boundary_logits", 0) == 0


def test_boundary_logits_contract_on_durable_snapshot_and_superseded_lease(monkeypatch):
    """``SessionBank.put`` keeps supplied ``boundary_logits`` on durable entries,
    ignores positions without a stored boundary, and never double-counts
    ``lease_aux_nbytes`` when superseding a matching prompt lease."""
    from mtplx.server import openai as server

    prompt = list(range(20))
    answer = prompt + [101, 102]
    prompt_logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
    mx.eval(prompt_logits)

    # Diverged or non-extended histories produce no boundary_logits.
    fs = SimpleNamespace(prompt_boundary_logits=prompt_logits)
    assert server._generation_final_prompt_boundary_logits(
        fs, prompt_ids=prompt, final_token_ids=prompt
    ) is None
    assert server._generation_final_prompt_boundary_logits(
        fs, prompt_ids=prompt, final_token_ids=[999] + answer[1:]
    ) is None
    assert server._generation_final_prompt_boundary_logits(
        fs, prompt_ids=prompt, final_token_ids=answer
    ) == (len(prompt), prompt_logits)

    # Durable snapshot entry (copying store): keeps boundary_logits when boundary exists.
    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    bank = _bank()
    cache = _conversation(len(prompt), seed=11)
    boundary = snapshot_untrimmable_cache(cache)
    _decode(cache, 2, seed=21)
    durable = bank.put(
        runtime=RUNTIME,
        token_ids=answer,
        cache=cache,
        logits=mx.zeros((1, 16), dtype=mx.float32),
        hidden=mx.ones((1, 1, 8)),
        session_id="dur",
        snapshot_epoch=len(answer),
        gdn_boundaries=[(len(prompt), boundary, None)],
        boundary_logits=(len(prompt), prompt_logits),
    )
    assert durable is not None and not durable.live_ref_only
    assert durable.boundary_logits is not None
    assert durable.boundary_logits[0] == len(prompt)
    assert np.array_equal(_bits(durable.boundary_logits[1]), _bits(prompt_logits))

    # Without a boundary at that position (unrelated tokens so no donor inherits one),
    # boundary_logits is ignored.
    unrelated = [500 + i for i in range(len(answer))]
    without_boundary = bank.put(
        runtime=RUNTIME,
        token_ids=unrelated,
        cache=cache,
        logits=mx.zeros((1, 16), dtype=mx.float32),
        hidden=mx.ones((1, 1, 8)),
        session_id="dur2",
        snapshot_epoch=len(unrelated),
        gdn_boundaries=[],
        boundary_logits=(len(prompt), prompt_logits),
    )
    assert without_boundary is not None and without_boundary.boundary_logits is None

    # One-copy lease superseding a prompt lease: aux bytes count boundary_logits once.
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank_a = _bank()
    bank_b = _bank()
    cache_a = _conversation(len(prompt), seed=31)
    cache_b = _conversation(len(prompt), seed=31)
    for b, c in ((bank_a, cache_a), (bank_b, cache_b)):
        fields = prompt_lease_fields(
            c, committed_mtp_cache=None, hidden=mx.ones((1, 1, 8)), prompt_len=len(prompt), boundaries=[]
        )
        b.put(
            runtime=RUNTIME, token_ids=prompt, cache=c, logits=prompt_logits,
            hidden=mx.ones((1, 1, 8)), session_id="s", snapshot_epoch=len(prompt), **fields,
        )
        _decode(c, 2, seed=41)
    entry_a = bank_a.put(
        runtime=RUNTIME, token_ids=answer, cache=cache_a, logits=mx.zeros((1, 16), dtype=mx.float32),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s", snapshot_epoch=len(answer),
    )
    entry_b = bank_b.put(
        runtime=RUNTIME, token_ids=answer, cache=cache_b, logits=mx.zeros((1, 16), dtype=mx.float32),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s", snapshot_epoch=len(answer),
        boundary_logits=(len(prompt), prompt_logits),
    )
    assert entry_a is not None and entry_b is not None
    assert entry_a.boundary_logits is not None and entry_b.boundary_logits is not None
    assert entry_a.lease_aux_nbytes == entry_b.lease_aux_nbytes
