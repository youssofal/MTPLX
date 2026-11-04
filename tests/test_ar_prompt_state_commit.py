"""Plain decoding banks each prompt before its answer, as speculative decoding does.

A conversation's next turn shares the prompt and then differs where the
answer starts: the client sends the answer back without its reasoning, or
renders a tool call its own way. On a hybrid model (every Qwen pack) a turn
can only resume at a recurrent checkpoint. The speculative lane banks every
prompt with one at its end before decoding (``commit_prompt_state_to_bank``);
the plain-decoding lane never did, and its prefill store skips suffixes under
1,024 tokens, so a follow-up resumed at the last turn that reached that floor.
On the tiny Flash-Next harness with ``--generation-mode ar --no-load-mtp``
turns 3 to 5 resumed at 3,708 tokens instead of 3,742, 3,776 and 3,808; by
the 1,024-token floor a follow-up could re-read up to about 1,000 tokens.
2.12.2 was the same; tonight's ``--no-load-mtp`` boot (#400) sends M1 and M2 owners there.

The engine tests drive the real ``generate_ar`` and ``SessionBank`` with a
toy hybrid (an attention layer and a recurrent running sum, from
tests/test_checkpoint_pre_image.py) on the CPU, and bank each answer with the
server's own generation-final helpers. The server tests pin what
``_run_generation`` hands the two lanes: the plain-decoding call now gets the
commit and the admission prices it; the speculative call is unchanged, all 43
arguments, as captured on the release head before this change (9724538b).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.server import openai
from mtplx.session_bank import SessionBank
from tests.test_checkpoint_pre_image import HybridToy
from tests.test_server_openai import _fake_streaming_session_state

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
SESSION = "plain-decoding"
TEMPLATE = "template"
FINGERPRINT = "policy"
# 21 tokens: off the 8-token chunk grid, so no prefill checkpoint lands on the
# prompt's end by accident.
FIRST_PROMPT = [(5 * i + 2) % 14 for i in range(21)]
USER_TURNS = ([3, 7, 1, 9], [2, 2, 8, 4, 6], [11, 5, 0])
ANSWER_TOKENS = 4


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


@pytest.fixture
def toy(monkeypatch):
    """The CPU, the toy's chunk grid, and production floors otherwise."""

    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "8")
    # The toy prompts are far below the production block and match floors.
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    for name in (
        "MTPLX_GDN_BOUNDARY_CAPTURE",
        "MTPLX_GDN_BOUNDARY_MAX",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_SESSION_NEAR_PREFIX_RESTORE",
        "MTPLX_SESSION_STORE_ON_PREFILL",
        # The production 1,024-token floor: no toy suffix reaches it.
        "MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _runtime(*, head: bool = False) -> MTPLXRuntime:
    # ``head``: a server that loaded the draft head and serves this request with
    # plain decoding (--generation-mode ar without --no-load-mtp). Its saved
    # answers carry the speculative history label (_bank_history_policy).
    return MTPLXRuntime(
        model=HybridToy(),
        tokenizer=_Tokenizer(),
        model_path=Path("models/plain-decoding-prompt-commit"),
        mtp_enabled=head,
        contract=MTPContract(),
    )


def _bank() -> SessionBank:
    return SessionBank(max_entries=32, max_bytes=1 << 26, per_session_max_bytes=1 << 26)


def _turn(rt, bank, prompt, *, commit: bool):
    out = generate_ar(
        rt,
        list(prompt),
        max_tokens=ANSWER_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        session_bank=bank,
        session_id=SESSION,
        session_template_hash=TEMPLATE,
        session_policy_fingerprint=FINGERPRINT,
        capture_final_state=True,
        commit_prompt_state_to_bank=commit,
    )
    # The answer, banked as the server banks it after every turn.
    state = SimpleNamespace(
        runtime=rt,
        backend_descriptor=SimpleNamespace(backend_id="native_mtp", mtp_history_policy=""),
    )
    values = openai._generation_final_bank_values(
        state,
        out.final_state,
        prompt_ids=list(prompt),
        final_token_ids=list(prompt) + list(out.tokens),
    )
    token_ids = values.pop("token_ids")
    entry = bank.put(
        runtime=rt,
        token_ids=token_ids,
        keep_live_ref=True,
        session_id=SESSION,
        template_hash=TEMPLATE,
        draft_head_identity=None,
        policy_fingerprint=FINGERPRINT,
        snapshot_epoch=len(token_ids),
        **values,
        **openai._generation_final_bank_metadata(
            state, out.final_state, token_count=len(token_ids)
        ),
    )
    assert entry is not None
    return out


def _resent(prompt, answer, user):
    """The next prompt: the answer comes back changed where it starts (its
    reasoning dropped), so the history diverges at the assistant turn."""

    changed = [(int(answer[0]) + 1) % 14, *(int(token) for token in answer[1:])]
    return [*prompt, *changed, *user]


def _conversation(*, commit: bool, head: bool = False):
    rt, bank = _runtime(head=head), _bank()
    prompts = [list(FIRST_PROMPT)]
    outs = [_turn(rt, bank, FIRST_PROMPT, commit=commit)]
    for user in USER_TURNS:
        prompts.append(_resent(prompts[-1], outs[-1].tokens, user))
        outs.append(_turn(rt, bank, prompts[-1], commit=commit))
    return prompts, outs


@pytest.mark.parametrize("head", [False, True], ids=["no-draft-head", "draft-head-loaded"])
def test_each_follow_up_resumes_at_the_previous_prompt_end(toy, head):
    # A copying store: the prompt's snapshot stays banked beside the answer,
    # whatever label the server gives the answer.
    prompts, outs = _conversation(commit=True, head=head)

    for turn in range(1, len(outs)):
        stats = outs[turn].stats
        assert stats.session_cache_hit is True
        assert stats.cached_tokens == len(prompts[turn - 1])
        assert stats.new_prefill_tokens == len(prompts[turn]) - len(prompts[turn - 1])
    commit = outs[-1].stats.session_prompt_prefix_bank_commit
    assert commit["stored"] is True
    assert commit["mode"] == "prompt_prefix"
    assert commit["prefix_len"] == len(prompts[-1])


def test_without_the_commit_follow_ups_resume_below_the_previous_prompt(toy):
    """Teeth: the same conversation without the commit (the lane before this
    change never made one) resumes every follow-up short of the previous
    prompt's end, at a checkpoint the prefill happened to leave."""

    prompts, outs = _conversation(commit=False)

    for previous, out in zip(prompts, outs[1:]):
        assert out.stats.cached_tokens < len(previous)
        assert out.stats.session_prompt_prefix_bank_commit == {}


def test_a_resumed_turn_decodes_what_a_cold_prefill_decodes(toy):
    prompts, outs = _conversation(commit=True)

    for prompt, out in zip(prompts[1:], outs[1:]):
        cold = generate_ar(
            _runtime(),
            list(prompt),
            max_tokens=ANSWER_TOKENS,
            sampler=GREEDY,
            seed=0,
            stop_token_ids=set(),
        )
        assert out.stats.cached_tokens > 0
        assert list(out.tokens) == list(cold.tokens)


def test_no_bank_or_no_session_commits_nothing(toy):
    alone = generate_ar(
        _runtime(),
        list(FIRST_PROMPT),
        max_tokens=ANSWER_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        commit_prompt_state_to_bank=True,
    )
    bank = _bank()
    anonymous = generate_ar(
        _runtime(),
        list(FIRST_PROMPT),
        max_tokens=ANSWER_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        session_bank=bank,
        commit_prompt_state_to_bank=True,
    )

    assert alone.stats.session_prompt_prefix_bank_commit == {}
    assert anonymous.stats.session_prompt_prefix_bank_commit == {}
    assert list(alone.tokens) == list(anonymous.tokens)


# ----------------------------------------------------------------- the server

# The admission itself, held before any test wraps it.
_ADMISSION = openai._prefill_admission_shed

# What _run_generation handed generate_mtpk on 9724538b, before this change.
MTP_CALL_ARGUMENTS = frozenset(
    {
        "abort_check", "adaptive_policy", "capture_final_state",
        "commit_prompt_state_keep_live_ref", "commit_prompt_state_to_bank",
        "constraint", "draft_core", "draft_sampler", "first_token_logprobs_top_k",
        "loop_guard", "max_tokens", "mtp_cache_policy", "mtp_hidden_variant",
        "mtp_history_policy", "online_correction_cache", "online_correction_cache_key",
        "online_correction_cache_min_depth", "online_hidden_corrector_alpha",
        "online_hidden_corrector_decay", "online_hidden_corrector_key",
        "online_hidden_corrector_max_feed_depth", "online_hidden_corrector_warmup",
        "prefill_callback", "prompt_correction_cache", "prompt_correction_cache_min_depth",
        "repetition_stop", "sampler", "seed", "session_bank",
        "session_draft_head_identity", "session_id", "session_policy_fingerprint",
        "session_restore_mode", "session_template_hash", "speculative_depth",
        "store_prefix_snapshot", "thinking_guard", "token_callback", "trace_label",
        "trace_metadata", "verify_core", "verify_strategy", "vision_splice",
    }
)


def _served(monkeypatch, mode: str, *, commit_prompt_prefix: bool = True, bank: bool = True):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    calls: dict[str, dict] = {}

    def lane(name):
        def fake(*_args, **kwargs):
            calls[name] = kwargs
            return SimpleNamespace(
                tokens=[],
                text="",
                stats=SimpleNamespace(
                    to_dict=lambda: {
                        "prompt_eval_time_s": 0.0,
                        "generated_tokens": 0,
                        "elapsed_s": 0.0,
                        "tok_s": 0.0,
                    }
                ),
                final_state=None,
            )

        return fake

    def admission(*args, **kwargs):
        calls["admission"] = kwargs
        return _ADMISSION(*args, **kwargs)

    monkeypatch.setattr(openai, "generate_mtpk", lane("mtp"))
    monkeypatch.setattr(openai, "generate_ar", lane("ar"))
    monkeypatch.setattr(openai, "_prefill_admission_shed", admission)
    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode=mode,
        depth=3,
        session_id="sess-tool",
        session_bank=state.sessions.bank if bank else None,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        commit_prompt_prefix_to_bank=commit_prompt_prefix,
    )
    return calls


def test_the_plain_decoding_call_gets_the_commit_the_speculative_call_gets(monkeypatch):
    ar = _served(monkeypatch, "ar")
    mtp = _served(monkeypatch, "mtp")

    assert ar["ar"]["commit_prompt_state_to_bank"] is True
    assert mtp["mtp"]["commit_prompt_state_to_bank"] is True
    # The admission prices the commit for both lanes.
    assert ar["admission"]["commit_prompt_prefix"] is True
    assert mtp["admission"]["commit_prompt_prefix"] is True


@pytest.mark.parametrize("mode, lane", [("ar", "ar"), ("mtp", "mtp")])
def test_no_commit_when_the_request_does_not_ask_or_has_no_bank(monkeypatch, mode, lane):
    unasked = _served(monkeypatch, mode, commit_prompt_prefix=False)
    bankless = _served(monkeypatch, mode, bank=False)

    assert unasked[lane]["commit_prompt_state_to_bank"] is False
    assert unasked["admission"]["commit_prompt_prefix"] is False
    assert bankless[lane]["commit_prompt_state_to_bank"] is False


def test_the_speculative_call_is_unchanged(monkeypatch):
    mtp = _served(monkeypatch, "mtp")["mtp"]

    assert frozenset(mtp) == MTP_CALL_ARGUMENTS
    assert mtp["commit_prompt_state_to_bank"] is True
    assert mtp["commit_prompt_state_keep_live_ref"] is False
    assert mtp["mtp_history_policy"] == "committed"
    assert mtp["capture_final_state"] is True
    assert mtp["store_prefix_snapshot"] is None
