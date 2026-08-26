"""Regression tests for the postcommit prefix-reuse + observability plumbing.

`_store_retokenized_history_snapshot` builds a SessionBank entry for the
turn that just generated. Before this fix that build re-prefilled the FULL
history from cold, which on multi-turn tool-calling agents costs ~27-29 s
per postcommit on Qwen3.6-27B at 18-25K-token contexts. The
`ModelWorkScheduler` admits the next foreground request only after the
in-flight idle task completes, so the user-visible symptom was a 30 s
stream-silence between turns.

The fix is to pass `session_bank` (and the matching `template_hash`,
`draft_head_identity`, `policy_fingerprint`) to
`restore_or_prefill_prompt_state`. The bank already contains an entry for
the previous turn's history, the new turn's history starts with that
verbatim (chat-template encoding is deterministic), so the longest-prefix
lookup matches and only the suffix needs to be forward-AR'd. Postcommit
cost collapses to roughly suffix-forward time (~1 s for ~120 new tokens).

These tests pin two invariants:

1. The plumbing: `restore_or_prefill_prompt_state` MUST receive
   `session_bank`, `template_hash`, `draft_head_identity`, and
   `policy_fingerprint`. Without ALL of them the bank lookup either does
   not run (no `session_bank`) or rejects the entry as a mismatch (one of
   the identity fields missing).
2. The observability: the result dict MUST surface
   `cache_hit`, `cached_tokens`, `suffix_tokens`, and `cache_miss_reason`
   from `prompt_state`. Operators rely on these fields in the
   `[mtplx] idle async session postcommit ...` log to debug regressions
   where the prefix shortcut stops firing.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from mtplx.server import openai


def _make_state(*, bank: object) -> SimpleNamespace:
    """Minimum stub of `ServerState` for `_store_retokenized_history_snapshot`.

    The function only reaches `state.runtime.tokenizer`,
    `state.runtime.mtp_enabled` (which picks the MTP-history policy the
    snapshot banks under), `state.sessions.bank`, `state.template_hash`,
    `state.draft_head_identity`, `state.lock`, `state.begin_foreground`,
    `state.end_foreground`, and
    `state.args.strip_assistant_reasoning_history`. Everything else is
    bypassed by the monkeypatched `restore_or_prefill_prompt_state` and
    `_encode_messages`.
    """

    class _Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return [10, 11, 12, 13, 14]

    args = SimpleNamespace(strip_assistant_reasoning_history=False)
    return SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        runtime=SimpleNamespace(tokenizer=_Tokenizer(), mtp_enabled=True),
        sessions=SimpleNamespace(bank=bank),
        template_hash="tmpl-abc",
        draft_head_identity="draft-xyz",
        lock=threading.Lock(),
        begin_foreground=lambda: None,
        end_foreground=lambda: None,
        args=args,
    )


def test_postcommit_passes_session_bank_and_identities_to_restore(monkeypatch):
    """Tier 1.2 / Option B plumbing.

    Without `session_bank`, the bank lookup never runs and every postcommit
    pays a full re-prefill. Without the identity / policy fingerprints, the
    bank lookup runs but rejects the entry with a TEMPLATE_MISMATCH /
    POLICY_MISMATCH miss. All four kwargs MUST be present.
    """
    captured: dict = {}

    class _PromptState:
        trunk_cache = "cache"
        logits = "logits"
        hidden = "hidden"
        committed_mtp_cache = None
        cache_hit = True
        cached_tokens = 17_500
        suffix_tokens = 320
        cache_miss_reason = None

    def fake_restore_or_prefill(*args, **kwargs):
        captured.update(kwargs)
        return _PromptState()

    class _Bank:
        def put(self, **_kwargs):
            return SimpleNamespace(
                prefix_len=5,
                nbytes=42,
                token_hash="hash",
            )

    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", fake_restore_or_prefill
    )
    monkeypatch.setattr(openai, "snapshot_cache", lambda c: c)
    monkeypatch.setattr(openai, "_encode_messages", lambda *a, **k: [10, 11, 12, 13, 14])
    # _history_ids_for_postcommit goes through the upstream sentinel-based
    # `_postcommit_next_turn_prefix_ids` helper; bypass it so this test
    # focuses purely on the restore-call plumbing.
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *a, **k: ([10, 11, 12, 13, 14], None),
    )

    bank = _Bank()
    state = _make_state(bank=bank)

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="session-A",
        messages=[],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy-fp",
    )

    assert result["stored"] is True, result
    assert captured.get("session_bank") is bank, (
        "restore_or_prefill_prompt_state must receive `session_bank` so the "
        "longest_prefix lookup can shortcut a full re-prefill."
    )
    assert captured.get("template_hash") == "tmpl-abc", (
        "template_hash MUST be passed - without it the bank rejects the "
        "entry as TEMPLATE_MISMATCH."
    )
    assert captured.get("draft_head_identity") == "draft-xyz", (
        "draft_head_identity MUST be passed - without it the bank rejects "
        "the entry as a draft-head identity mismatch."
    )
    assert captured.get("policy_fingerprint") == "policy-fp", (
        "policy_fingerprint MUST be passed - without it the bank rejects "
        "the entry as POLICY_MISMATCH."
    )


def test_postcommit_propagates_observability_fields(monkeypatch):
    """`cache_hit`, `cached_tokens`, `suffix_tokens`, `cache_miss_reason`
    must propagate from `prompt_state` into the result dict so the
    `[mtplx] idle async session postcommit ...` log surfaces them.
    """

    class _PromptState:
        trunk_cache = "cache"
        logits = "logits"
        hidden = "hidden"
        committed_mtp_cache = None
        cache_hit = True
        cached_tokens = 20_594
        suffix_tokens = 122
        cache_miss_reason = None

    class _Bank:
        def put(self, **_kwargs):
            return SimpleNamespace(prefix_len=20_716, nbytes=4242, token_hash="h")

    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", lambda *a, **k: _PromptState()
    )
    monkeypatch.setattr(openai, "snapshot_cache", lambda c: c)
    monkeypatch.setattr(openai, "_encode_messages", lambda *a, **k: [10, 11, 12, 13, 14])
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *a, **k: ([10, 11, 12, 13, 14], None),
    )

    state = _make_state(bank=_Bank())

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="session-A",
        messages=[],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy-fp",
    )

    assert result["stored"] is True
    assert result["cache_hit"] is True
    assert result["cached_tokens"] == 20_594
    assert result["suffix_tokens"] == 122
    assert "cache_miss_reason" in result, (
        "cache_miss_reason must always be present in the result so the "
        "log line records it (None on hit, a reason string on miss)."
    )
    assert result["cache_miss_reason"] is None


def test_postcommit_cache_miss_reason_records_miss(monkeypatch):
    """When the prefix shortcut does NOT fire (e.g. cold session, or
    template_hash drift), `cache_miss_reason` must record why so operators
    can debug from logs alone.
    """

    class _PromptState:
        trunk_cache = "cache"
        logits = "logits"
        hidden = "hidden"
        committed_mtp_cache = None
        cache_hit = False
        cached_tokens = 0
        suffix_tokens = 18_400
        cache_miss_reason = "prefix_divergence_at_token"

    class _Bank:
        def put(self, **_kwargs):
            return SimpleNamespace(prefix_len=18_400, nbytes=4242, token_hash="h")

    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", lambda *a, **k: _PromptState()
    )
    monkeypatch.setattr(openai, "snapshot_cache", lambda c: c)
    monkeypatch.setattr(openai, "_encode_messages", lambda *a, **k: [10, 11, 12, 13, 14])
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *a, **k: ([10, 11, 12, 13, 14], None),
    )

    state = _make_state(bank=_Bank())

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="session-A",
        messages=[],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy-fp",
    )

    assert result["stored"] is True
    assert result["cache_hit"] is False
    assert result["cache_miss_reason"] == "prefix_divergence_at_token"


def test_public_stats_include_postcommit_cache_observability():
    """The API-visible stats must not hide the prefix-reuse fields."""
    public = openai._public_mtplx_stats(
        {
            "stats": {
                "session_postcommit_snapshot": {
                    "stored": True,
                    "mode": "retokenized_history",
                    "cache_hit": True,
                    "cached_tokens": 20_594,
                    "suffix_tokens": 122,
                    "cache_miss_reason": None,
                    "token_hash": "internal-only",
                }
            }
        }
    )

    assert public["session_postcommit_snapshot"] == {
        "stored": True,
        "mode": "retokenized_history",
        "cache_hit": True,
        "cached_tokens": 20_594,
        "suffix_tokens": 122,
        "cache_miss_reason": None,
    }
