"""Typed registry for the Qwen3.8 Flash-Next full-stack decode env keys.

Why this module exists
----------------------
The in-process benchmark drivers arm a stack of decode switches. ``mtplx
serve`` reaches most of them on its own -- ``mtplx/server/openai.py``
auto-arms them for the served pack -- but not all, and the ones it misses are
env-gated and default-off, so they simply do not happen. The visible symptom
is ``[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`` in the server log while the
same code measures faster in-process.

Where the reference stack comes from
------------------------------------
The measured stack has two halves, and this registry carries both.

:data:`CONTROL_ARM_KEYS` is the RETAINED-STACK CONTROL ARM -- the configuration
every A/B window measures its candidate against: the 19 family keys the
benchmark harness derives from its own flags plus the two FR-Spec keys its
``--full-frspec`` block adds, 21 in all.

Two consequences of deriving it from the real control arm rather than from a
transcribed block:

* ``MTPLX_QWEN4_RELAXED_DRAFT_TIES`` is NOT in it. The harness never sets it
  and ``--compiled-mtp-prepare`` does not imply it, so it is a
  CANDIDATE-arm flag, never part of the measured control. An earlier draft of
  this registry carried it; shipping it would have armed a lane the control
  never measured.
* ``MTPLX_COMPILED_VERIFY=on`` and ``MTPLX_NAX_VERIFY=0`` ARE in it. Both are
  already supplied by the server for this family, so neither changes what the
  profile stamps -- but leaving them out understated the stack the check is
  against.

The other half is the RETAINED FABLE SET: the fifteen decode keys and the
eight prefill keys the PR-391 battery ran its branch arms with. Those
files are now committed as ``docs/perf/pr391-stack.flags`` and
``docs/perf/pr391-prefill.flags`` (see :data:`GROUP_FLAG_FILES`), and
``tests/test_full_stack_profile.py`` asserts that this registry and those
files carry byte-identical key/value sets, so the record and the code cannot
drift apart. Nothing else in the tree sets any of the twenty-three -- not the
server's auto-arm, not a model pack's runtime contract, not another profile
-- which is why they are all :data:`OWNER_PROFILE`, and why serving them
before this change meant exporting two files by hand.

:data:`FULL_STACK_KEYS` is the union: the 44 keys a fully armed serve must
resolve to, whoever supplies each one.

WHEN a key is stamped, not just whether
---------------------------------------
Nine of the profile's own keys are read ONCE at module import
(:data:`BIND_IMPORT`; ``mtplx.runtime_options`` and friends freeze them in
module constants so the hot path never touches ``os.environ``).
``apply_profile_env`` runs inside ``mtplx/server/openai.py:_load`` -- and by
then that module's own import block has already pulled every one of those
readers. For those eleven keys a profile stamp there arms NOTHING: the
environment changes and no reader ever looks again.

So :func:`stamp_import_time_defaults` puts exactly that subset in place
from ``mtplx/server/__init__.py``, which Python executes while it resolves
``python -m mtplx.server.openai`` and therefore before the first line of
``openai.py``. It stamps nothing else, it yields to an operator export the
same way ``apply_profile_env`` does, and ``EARLY_STAMP_ENV`` turns it off.
Every other profile key stays ``apply_profile_env``'s to set.

The second half of the problem is spelling. Most of these keys are read by a
bare ``os.environ.get(name, default)`` at one call site, so a misspelled key
is not an error -- it is silence, and the lane it was meant to arm stays off
while every receipt still says "ok".

So this module is the ONE place that names each key of the stack, its type,
the value a reader sees when it is unset, the call site that reads it, and --
critically -- **who sets it**. Four things hang off that:

* :data:`FULL_STACK_PROFILE_ENV`, the 26 keys the ``turbo-full-stack``
  profile stamps: exactly the ones nothing else sets -- the control arm's
  three-key gap plus both retained Fable sets in full. The profile does not
  restate a key the server already arms, because restating it would also
  STOMP an operator's explicit export (a profile-owned key beats the
  environment unless it is in ``PROFILE_ENV_USER_OVERRIDE_KEYS``, while the
  server's auto-arm deliberately yields to one). All 26 ARE in that set, so
  an operator export still beats every one of them -- ``MTPLX_FRSPEC_DRAFT=0``
  is the kill switch for a lane whose installer raises rather than falling
  back;
* :data:`FULL_STACK_RESTACK_ENV`, the full 44-key block, kept as the
  reference the whole stack is checked against, and :data:`CONTROL_ARM_ENV`,
  its ABBA-derived half;
* :func:`resolved_stack`, which answers "is the stack actually armed, and by
  whom" against a live environment; and
* :func:`warn_unknown_family_keys`, which says at startup that an
  ``MTPLX_QWEN4_*`` / ``MTPLX_QSA_*`` / ``MTPLX_FRSPEC_*`` key in the
  environment is read by nothing in this package -- a WARNING, never a raise,
  and never a change to any default. ``MTPLX_FABLE_*`` is deliberately NOT a
  family prefix: the tree carries a long tail of experiment flags under it
  (``_HOST_TRIMS``, ...) that this registry has no reason to enumerate,
  and warning about each one would be noise, not a typo check.

Who sets what (verified against mtplx/server/openai.py:730-890, 2026-09-02)
--------------------------------------------------------------------------
``_server_runtime_env_overrides`` builds runtime overrides BEFORE
``apply_profile_env`` runs and they are applied AFTER the profile env, so a
server override beats a profile value, and the server's own
``if os.environ.get(key) is None`` / ``pop`` guards are what let an operator
export beat the server. Precedence, weakest to strongest:

    reader default  <  profile env  <  server auto-arm  <  operator export
                                                        <  server FORCED

Hence :data:`OWNER_SERVER_AUTO` keys must NOT appear in the profile env: the
profile would win over the operator on exactly the keys the server chose to
let the operator own.

Parse fidelity
--------------
These call sites do NOT agree on how to parse a boolean, and this registry
does not "fix" that: routing a read through here must be behaviour-preserving
to the byte. Each entry records the parse its call site actually performs
(:class:`EnvKeySpec.parse`), and :func:`flag_enabled` reproduces it:

``lenient``          ``(env.get(name) or default).strip().lower() in TRUE_TOKENS``
``lenient_nostrip``  ``env.get(name, default).lower() in TRUE_TOKENS``
``lenient_raising``  the same vocabulary as ``lenient``, but a spelling in
                     neither ``TRUE_TOKENS`` nor
                     :data:`RAISING_FALSE_TOKENS` RAISES instead of reading
                     as off (``graph_build_overlap.enabled``,
                     ``ple_prefill_lookahead.enabled``,
                     ``ple_row_gather.enabled``)
``strict``           ``mtplx.runtime_options.env_bool`` (raises on an unknown
                     spelling; accepts ``enable``/``enabled``)
``text``             the raw string, read with :func:`text_value`
``mode``             a multi-valued string knob, compared on its resolved
                     mode rather than its spelling (``MTPLX_COMPILED_VERIFY``
                     reads ``1`` and ``on`` as the same mode)

``tests/test_full_stack_profile.py`` pins EVERY routed key's parse against
the expression its call site used before it was routed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

#: The boolean vocabulary the bare ``os.environ.get`` call sites accept.
#: Deliberately NARROWER than ``mtplx.runtime_options.ENV_TRUE_VALUES``
#: (which also takes ``enable``/``enabled``): these sites never accepted the
#: wider spelling and this registry does not widen them.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: The false vocabulary of the ``lenient_raising`` call sites
#: (``mtplx/graph_build_overlap.py``, ``mtplx/ple_prefill_lookahead.py``,
#: ``mtplx/ple_row_gather.py``). Those readers accept exactly
#: ``TRUE_TOKENS`` and this set, and RAISE on anything else rather than
#: reading an unknown spelling as off.
RAISING_FALSE_TOKENS = frozenset({"", "0", "false", "no", "off"})

#: Env-key prefixes this registry is responsible for. An unregistered key
#: under one of these prefixes is what :func:`warn_unknown_family_keys`
#: reports.
FAMILY_PREFIXES = ("MTPLX_QWEN4_", "MTPLX_QSA_", "MTPLX_FRSPEC_")

#: Name of the opt-in profile that closes the gap. Imported by
#: ``mtplx.profiles``; kept here so the registry, not the profile table, is
#: the single place the stack is described.
FULL_STACK_PROFILE_NAME = "turbo-full-stack"

PARSE_KINDS = (
    "lenient",
    "lenient_nostrip",
    "lenient_raising",
    "strict",
    "text",
    "mode",
)

#: Which side of the stack a key belongs to, and therefore which committed
#: record is its canonical spelling (see :data:`GROUP_FLAG_FILES`).
GROUP_CONTROL_ARM = "abba_control_arm"
GROUP_FABLE_DECODE = "fable_decode"
GROUP_FABLE_PREFILL = "fable_prefill"
GROUPS = (GROUP_CONTROL_ARM, GROUP_FABLE_DECODE, GROUP_FABLE_PREFILL)

#: When the READER of a key stops being able to see a change to it. This is
#: not a style note: it decides whether stamping the key from
#: ``profiles.apply_profile_env`` is early enough to arm anything at all.
#:
#: ``import``      the reader froze the value in a module-level constant at
#:                 ``import`` time (``mtplx.runtime_options``,
#:                 ``mtplx.fable_block_verify``,
#:                 ``mtplx.fable_draft_k20_prescatter``). ``mtplx serve``
#:                 re-execs ``python -m mtplx.server.openai``, whose
#:                 module-level imports pull all three BEFORE
#:                 ``apply_profile_env`` runs at
#:                 ``mtplx/server/openai.py:_load``. A profile stamp alone is
#:                 therefore TOO LATE for these keys, which is what
#:                 :func:`stamp_import_time_defaults` exists to fix.
#: ``first_read``  ``lru_cache``/lazy-global: fixed by the first call, which
#:                 happens at model install, after the profile env.
#: ``call``        re-read on every call; a profile stamp always reaches it.
BIND_IMPORT = "import"
BIND_FIRST_READ = "first_read"
BIND_CALL = "call"
BINDS = (BIND_IMPORT, BIND_FIRST_READ, BIND_CALL)

#: Nothing sets it. Serving gets the reader's own unset default.
OWNER_DEFAULT = "default"
#: mtplx/server/openai.py:_server_runtime_env_overrides arms it for the
#: served pack, but only when the operator has not exported it. Yields to an
#: operator export -- which is why the profile must not restate these.
OWNER_SERVER_AUTO = "server_auto_arm"
#: Same function, but assigned unconditionally: it beats an operator export
#: too, because the value is a correctness requirement rather than a knob.
OWNER_SERVER_FORCED = "server_forced"
#: The ``turbo-full-stack`` profile stamps it, because nothing else does.
OWNER_PROFILE = "profile"

OWNERS = (OWNER_DEFAULT, OWNER_SERVER_AUTO, OWNER_SERVER_FORCED, OWNER_PROFILE)


@dataclass(frozen=True)
class EnvKeySpec:
    """One env key of the full-stack decode lane.

    ``default`` is the value the READER sees when the key is unset -- i.e.
    the literal already written at the call site, not an aspiration.
    ``stack_value`` is the value the in-process drivers set, i.e. what a
    fully armed stack must resolve to, whoever supplies it. Nothing in this
    module changes a default.
    """

    name: str
    kind: str  # "bool" | "str"
    parse: str  # one of PARSE_KINDS
    default: str
    stack_value: str
    owner: str
    owner_site: str
    owner_predicate: str
    reader: str
    note: str
    routed: bool = False
    #: Part of the measured stack. A key can be registered (so its reads are
    #: typed and it is a known spelling) without being part of the stack the
    #: self-check scores -- see MTPLX_QWEN4_RELAXED_DRAFT_TIES.
    in_stack: bool = True
    #: Which half of the measured stack this key belongs to.
    group: str = GROUP_CONTROL_ARM
    #: When the reader stops seeing changes to it. See :data:`BINDS`.
    binds_at: str = BIND_CALL
    #: The OPTIMIZATION LANE this key belongs to: the unit an operator turns
    #: off with ``MTPLX_FABLE_DISABLE=<lane>`` or
    #: ``--disable-optimization <lane>``. Several keys can share one lane
    #: when turning half of it off would be incoherent (the prefill chunk
    #: width and the QSA compile rows are one such pair). Where the lane
    #: also has an ``mtplx.fable_install_receipts`` verdict, the two names
    #: are deliberately the SAME string, so the name an operator types is
    #: the name the receipt prints. Empty for a key nothing defaults on.
    lane: str = ""
    #: WHERE this key's install-time verdict is printed, when the receipt is
    #: NOT an ``mtplx.fable_install_receipts`` lane. The program's rule is
    #: that every armed flag proves itself at install, so a key defaulted on
    #: with neither a lane nor an entry here is an unprovable arm --
    #: ``tests/test_fable_defaults.py`` fails on one.
    receipt: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("bool", "str"):
            raise ValueError(f"{self.name}: kind must be 'bool' or 'str'")
        if self.parse not in PARSE_KINDS:
            raise ValueError(f"{self.name}: parse must be one of {PARSE_KINDS}")
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: group must be one of {GROUPS}")
        if self.binds_at not in BINDS:
            raise ValueError(f"{self.name}: binds_at must be one of {BINDS}")
        if self.kind == "str" and self.parse not in ("text", "mode"):
            raise ValueError(f"{self.name}: str keys must use parse='text'/'mode'")
        if self.owner not in OWNERS:
            raise ValueError(f"{self.name}: owner must be one of {OWNERS}")
        if not self.reader.strip():
            raise ValueError(f"{self.name}: reader must name the call site")
        if not self.owner_site.strip():
            raise ValueError(f"{self.name}: owner_site must name who sets it")
        if not self.note.strip():
            raise ValueError(f"{self.name}: note must say what the key arms")

    @property
    def stamped_by_profile(self) -> bool:
        return self.owner == OWNER_PROFILE


_SERVER = "mtplx/server/openai.py:_server_runtime_env_overrides"
_QWEN4_EXP = "_served_model_type_is_qwen4_exp(args)"
_FIXED_M4 = "_served_model_is_qwen4_fixed_m4(args)"
_MTP_QWEN4 = 'generation_mode == "mtp" and _served_model_type_is_qwen4_exp(args)'
_PROFILE_SITE = "mtplx/profiles.py:TURBO_FULL_STACK_PROFILE"
_NOBODY = "nobody"

#: The stack, in the order the benchmark harness lists it. Ownership is
#: verified against ``mtplx/server/openai.py``'s runtime overrides.
REGISTERED_KEYS: tuple[EnvKeySpec, ...] = (
    EnvKeySpec(
        name="MTPLX_QWEN4_FIXED_M4_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/qwen4_fixed_verify.py:qwen4_fixed_verify_enabled",
        note=(
            "Construction-bound fixed-M4 (4-row) compiled verifier. The "
            "server arms it for a fixed-M4 pack; an operator export makes the "
            "server drop its override entirely (pop, not setdefault)."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_STAGE3",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/qwen4_m4_stage3.py:qwen4_m4_stage3_flags",
        note=(
            "Stage-3 M4 MoE combine tail. Already goes through "
            "runtime_options.env_bool, so it is registered but not rerouted."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QSA_M4_FUSED_KV_GATHER",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/graphbank.py:_env_enabled",
        note="One-dispatch QSA selected-K/V gather for the fixed-M4 rows.",
    ),
    EnvKeySpec(
        name="MTPLX_QSA_GATHER",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_qsa_gather_enabled",
        note="QSA rows-gather decode lane (self-fenced to S 2..8 at KV>=16384).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_COMPILED_GDN",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:Qwen4ExpTextModel.__init__",
        note="Compiled GDN decode runs (paired with MTPLX_AR_PIPELINE).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_AR_PIPELINE",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_env_truthy",
        note="Pipelined AR decode lane. Read through generation's _env_truthy.",
    ),
    EnvKeySpec(
        name="MTPLX_FAMILY_CAPTURE_COMMIT",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_family_capture_commit_enabled",
        note=(
            "Layer-owned capture-commit (repair-free speculative rollback). "
            "Already reads through runtime_options.env_bool."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_HC_V3",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_hc_v3_enabled",
        note="Fused hyper-connection read v3.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_INPROJ",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_in_proj_enabled",
        note="GDN in_proj fusion (four input GEMVs to one).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GATE_UP",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gate_up_enabled",
        note="Sanitize-time MoE gate+up library merge.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_CONVNORM",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_conv_norm_enabled",
        note="Fused GDN conv+silu+l2norm between the GEMVs.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_STEP",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_step_enabled",
        note="One-dispatch GDN decode step (supersedes CONVNORM at decode).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_CONVNORM_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_conv_norm_rows_enabled",
        note="Verify-width conv+silu+l2norm rows kernel (S<=6).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime.py:load",
        lane="compiled_mtp_prepare",
        receipt=(
            "mtplx/runtime.py '[qwen4-compiled-MTP-prepare]' + "
            "full_stack_selfcheck marker qwen4_compiled_mtp_prepare"
        ),
        note=(
            "Compiled Qwen4 MTP preparation (driver flag: "
            "--compiled-mtp-prepare). GAP: no server path sets it."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_BATCH_TARGET_ARRAYS",
        kind="bool",
        parse="lenient_nostrip",
        default="",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_batch_target_arrays_enabled",
        note=(
            "Batched target-distribution precompute. Turbo sets 0 and the "
            "server's auto-arm already overrides it to 1 for this family "
            "(the override is applied AFTER the profile env), so the "
            "turbo-vs-driver conflict is resolved by the server, not here. "
            "NOTE: this call site lowercases WITHOUT stripping, hence "
            "parse='lenient_nostrip'."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_lazy_target_distributions_enabled",
        note=(
            "Lazy per-row target distributions. Turbo sets 1, which would "
            "make MTPLX_BATCH_TARGET_ARRAYS runtime-dead "
            "(profiles.RUNTIME_GATED_ENV_PAIRS); the server's auto-arm "
            "already overrides it to 0 for this family."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_SKIP_VERIFY_SNAPSHOT",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="0",
        owner=OWNER_SERVER_FORCED,
        owner_site=_SERVER,
        owner_predicate=_MTP_QWEN4,
        reader="mtplx/generation.py:_skip_verify_snapshot",
        note=(
            "Turbo sets 1; the server assigns 0 UNCONDITIONALLY for mtp on "
            "this family (plain assignment, not setdefault), so it beats both "
            "the profile and an operator export. Flash-Next rejection "
            "rollback replays from the recurrent-state snapshot -- this is a "
            "correctness requirement, not a speed knob."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_DRAFT",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/frspec_draft.py:frspec_enabled",
        lane="frspec",
        receipt=(
            "mtplx/draft_lm_head.py '[frspec] install report' + "
            "full_stack_selfcheck marker frspec_installed"
        ),
        note=(
            "FR-Spec row-pruned draft head. This is the switch whose absence "
            "the log reports as '[frspec] disabled "
            "(MTPLX_FRSPEC_DRAFT=None)'. It was the original GAP key: until "
            "2026-09-03 no server path set it AND it was not in "
            "profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS, so neither a serve nor "
            "a model pack could arm it. The defaults arm it now, and the "
            "allowlist spreads FULL_STACK_PROFILE_ENV, so a pack can pin or "
            "override it like any other stack key."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_VOCAB",
        kind="str",
        parse="text",
        default="",
        stack_value="builtin:qwen38-code-64k",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/frspec_draft.py:_vocab_path",
        lane="frspec",
        receipt=(
            "mtplx/draft_lm_head.py '[frspec] install report' (n=65536 "
            "is this vocabulary) + marker frspec_installed"
        ),
        note=(
            "FR-Spec vocabulary; 'builtin:qwen38-code-64k' is the 65,536-row "
            "table the engagement marker reports as n=65536. Same history as "
            "MTPLX_FRSPEC_DRAFT: the original GAP pair, armed by the defaults "
            "and pack-overridable since 2026-09-03."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_COMPILED_VERIFY",
        kind="str",
        parse="mode",
        default="",
        stack_value="on",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/graphbank.py:_compiled_verify_mode",
        note=(
            "Compiled verify. The control arm sets the string 'on' and turbo "
            "sets '1'; graphbank resolves anything outside "
            "{'', 0, false, no, off, parity, parity2} to the SAME 'on' mode, "
            "so the two spellings agree and this key is compared on the "
            "resolved mode, not the literal."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_NAX_VERIFY",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/nax_verify.py:nax_verify_enabled",
        note=(
            "27B NAX verify patch. Turbo sets 1; the control arm and the "
            "server's family override both set 0 (unmeasured and mostly "
            "bypassed on this family), and the server's override is applied "
            "after the profile env, so turbo already resolves to 0 here."
        ),
    ),
    # --- the retained Fable DECODE set -----------------------------------
    # docs/perf/pr391-stack.flags, verbatim. Every key here is owned by the
    # profile: no server path, no model contract and no other profile sets
    # any of them, which is why the battery had to export the file.
    EnvKeySpec(
        name="MTPLX_FABLE_HC_M4",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_hc_m4_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="hc_m4",
        receipt=(
            "mtplx/models/qwen4_exp.py:install_hc_m4_pack_validation -> "
            "runtime.qwen4_hc_m4_report, at /health "
            "engagement_reports.qwen4_hc_m4"
        ),
        note=(
            "Verify-width fused hyper-connection read "
            "(mtplx/kernels/qwen4_m4_hyper_read). RAISES on a family-contract "
            "miss rather than falling back, so an armed-but-inert lane is "
            "unreachable. Its install-time pack validation is published as "
            "the runtime's qwen4_hc_m4 engagement report."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_OPDIET",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_opdiet_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="opdiet",
        note=(
            "Exact-preserving op diet for the compiled fixed-M4 verify graph "
            "(items bank/rope/resid/k20). MTPLX_FABLE_OPDIET_ITEMS narrows it "
            "and is NOT part of the stack: unset selects everything, which is "
            "what the battery measured."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_BLOCK_VERIFY",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/fable_block_verify.py:_env_truthy",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="block_verify",
        note=(
            "Block speculative verification (Sun et al. 2024) in place of the "
            "per-token Leviathan-Chen accept law. mtplx/generation.py caches "
            "the module's decision in _FABLE_BLOCK_VERIFY at ITS import too, "
            "so both readers freeze at import."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_ROUTE_KERNEL",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/qwen4_m4_stage3.py:fable_route_kernel_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_FIRST_READ,
        lane="route_kernel",
        note=(
            "M4 MoE route GEMV+top-k kernel: the stage-3 tail's ten "
            "dispatches in two. Cached on first call (a lazy global), which "
            "is the stage-3 install, after the profile env."
        ),
    ),
    # ---- the M4 routed-GLU MoE tail -- three keys, one lane ----
    #
    # These three were armed on EVERY branch arm the PR-391 battery and
    # sanity runs measured: the benchmark harness applies them from its own
    # base environment whatever flag file is passed, and its A/B window puts
    # them on BOTH arms. Nothing under mtplx/ set them and upstream 2.10.2 has no
    # reader for any of them, so before this entry the served stack was three
    # lanes short of the measured one.
    #
    # They are also a HARD requirement of the already-defaulted
    # MTPLX_FABLE_ROUTE_KERNEL. qwen4_m4_stage3._validate_feature_combination
    # raises "requires MTPLX_QWEN4_M4_ROUTED_GLU" when the route kernel is
    # armed without the chain, and mtplx/runtime.py:1182 calls
    # qwen4_m4_stage3_flags() unguarded -- so a stock `mtplx serve` against a
    # Flash-Next pack could not finish loading the model at all.
    #
    # ONE LANE, deliberately: the chain is reduce -> residual tail -> GLU ->
    # route kernel, each member validating the one before it, so disabling a
    # prefix while a later member stays armed is precisely what that
    # validator refuses. Sharing the route_kernel lane is the same doctrine
    # MTPLX_PREFILL_CHUNK_SIZE / MTPLX_QSA_PREFILL_COMPILE_ROWS follow: a
    # lane is the coherent unit, and --disable-optimization route_kernel
    # returns all four keys to their shipped defaults at once. (An operator
    # who exports only MTPLX_QWEN4_M4_ROUTED_GLU=0 still gets the validator's
    # raise, by design -- it names the key to unset alongside it.)
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/qwen4_m4_stage3.py:qwen4_m4_routed_down_reduce_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_CALL,
        lane="route_kernel",
        receipt=(
            "mtplx/qwen4_m4_stage3.py:install_qwen4_m4_stage3 -> "
            "runtime.qwen4_m4_stage3_report['routed_down_reduce'], logged as "
            "[qwen4-M4-stage3] and published at /health "
            "engagement_reports.qwen4_m4_stage3"
        ),
        note=(
            "Routed q4/group-32 down projection emits the already "
            "route-weighted, already reduced BF16 [4, 2560] instead of a "
            "[4, 10, 2560] materialization plus a stock combine tail. Exact: "
            "it reproduces MLX 0.32.2's per-dot BF16 narrowing and the "
            "0+8, 1+9, then 2..7 reduction order "
            "(docs/perf/pr391-m4-routed-down-reduce-result.md). "
            "Construction-bound to the physical-M4 class; non-M4 rows keep "
            "the parent path."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/qwen4_m4_stage3.py:qwen4_m4_routed_down_residual_tail_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_CALL,
        lane="route_kernel",
        receipt=(
            "mtplx/qwen4_m4_stage3.py:install_qwen4_m4_stage3 -> "
            "runtime.qwen4_m4_stage3_report['routed_down_residual_tail'], logged as "
            "[qwen4-M4-stage3] and published at /health "
            "engagement_reports.qwen4_m4_stage3"
        ),
        note=(
            "Folds the MLP hyper-residual write into a second dispatch, "
            "giving the boundary "
            "routed_q4g32_reduce_shared_add_mlp_residual. Requires "
            "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE (the validator raises "
            "otherwise). Exact -- BF16 shared product, block add, inject "
            "product and hyper add, no eligibility or arithmetic fallback; "
            "+0.822% decode tok/s "
            "(docs/perf/pr391-m4-routed-down-residual-tail-result.md)."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_ROUTED_GLU",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/qwen4_m4_stage3.py:qwen4_m4_routed_glu_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_CALL,
        lane="route_kernel",
        receipt=(
            "mtplx/qwen4_m4_stage3.py:install_qwen4_m4_stage3 -> "
            "runtime.qwen4_m4_stage3_report['paired_routed_glu'], logged as "
            "[qwen4-M4-stage3] and published at /health "
            "engagement_reports.qwen4_m4_stage3"
        ),
        note=(
            "Pairs fused-pack rows j and 640+j in the routed gate/up "
            "projection and its SiLU*up producer, reusing one hidden-input "
            "tile per pair and emitting only [4, 10, 640]. Requires the "
            "residual tail. Exact -- same 64-thread ownership, five "
            "512-value K blocks, stock BF16 sigmoid/SiLU/product "
            "boundaries; +1.6925% decode tok/s "
            "(docs/perf/pr391-m4-paired-routed-glu-result.md). This is the "
            "key MTPLX_FABLE_ROUTE_KERNEL, MTPLX_FABLE_MOE_EXPERT_MAJOR and "
            "MTPLX_FABLE_SHARED_LANE all name as their prerequisite."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_DRAFT_K20_PRESCATTER",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/fable_draft_k20_prescatter.py:_env_truthy",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="draft_k20_prescatter",
        note=(
            "Draft K20 support built from the FR-Spec head's compact 65,536 "
            "row instead of the 248,320 scattered one. Requires "
            "MTPLX_FRSPEC_DRAFT: with no FR-Spec head there is no compact row "
            "to read and the claim declines to the stock draft read."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_GRAPH_BUILD_OVERLAP",
        kind="bool",
        parse="lenient_raising",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/graph_build_overlap.py:enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="graph_build_overlap",
        note=(
            "Host graph build overlapped behind a compiled leading-layer "
            "prefix. Its reader RAISES on a spelling outside the true/false "
            "vocabulary rather than reading it as off, hence "
            "parse='lenient_raising'."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS",
        kind="str",
        parse="text",
        default="1",
        stack_value="3",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/graph_build_overlap.py:layers",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="graph_build_overlap",
        note=(
            "How many leading target layers the prefix graph carries. The "
            "reader's own default is 1 (the shipped layer-0 prefix); the "
            "measured stack runs 3. Values outside [1, 8] raise at the flag."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_VERIFY_GLUE",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_verify_glue_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="verify_glue",
        receipt=(
            "mtplx/fable_verify_glue.py:install() -- one line per "
            "selected item, to logger and stderr"
        ),
        note="Fused verify-width glue kernels, item-selected by the key below.",
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_VERIFY_GLUE_ITEMS",
        kind="str",
        parse="text",
        default="",
        stack_value="qsa_rope,qsa_rope_idx",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:parse_verify_glue_items",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="verify_glue",
        receipt=(
            "mtplx/fable_verify_glue.py:install() -- the per-item lines "
            "name the selected set"
        ),
        note=(
            "The two instantiated glue items. Unset means 'all', which is the "
            "SAME set today -- but the battery named them explicitly and the "
            "registry records the literal it measured, so a third item landing "
            "later cannot silently join the arm. An unknown name raises."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_QSA_SPARSE_DECODE",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_qsa_sparse_decode_enabled",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="qsa_sparse_decode",
        receipt=(
            "mtplx/kernels/qsa_sparse_decode.py:engagement_line()"
        ),
        note=(
            "Split-K native sparse-GQA attention for the M=4 fixed verify. "
            "The M=1 width is deliberately NOT in the stack: the retained "
            "dispatch census shows zero QSA attention dispatches at M=1."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_QSA_SPARSE_DECODE_TILE",
        kind="str",
        parse="text",
        default="128:32",
        stack_value="128:32",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_qsa_sparse_decode_tile",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="qsa_sparse_decode",
        receipt=(
            "mtplx/kernels/qsa_sparse_decode.py:engagement_line() -- "
            "the armed tile is in the line"
        ),
        note=(
            "The (BK, DC) metallib tile. The reader's unset default is already "
            "128:32, so an unset environment satisfies the stack -- but "
            "nothing HOLDS it there, which is why the battery pinned it and "
            "resolved_stack reports present=False separately from ok=True. "
            "An uninstantiated pair raises."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS",
        kind="str",
        parse="text",
        default="17",
        stack_value="17",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime_options.py:fable_qsa_sparse_decode_splits",
        group=GROUP_FABLE_DECODE,
        binds_at=BIND_IMPORT,
        lane="qsa_sparse_decode",
        receipt=(
            "mtplx/kernels/qsa_sparse_decode.py:engagement_line() -- "
            "the armed split target is in the line"
        ),
        note=(
            "KV-split target. 17 is the measured optimum AND the reader's "
            "current default (raised from a placeholder 8, which measured "
            "2.2x slower), so the same present/ok distinction as the tile "
            "above applies. Outside [1, 64] raises."
        ),
    ),
    # --- the retained Fable PREFILL set -----------------------------------
    # docs/perf/pr391-prefill.flags, verbatim.
    EnvKeySpec(
        name="MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD",
        kind="bool",
        parse="lenient_raising",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/ple_prefill_lookahead.py:enabled",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_FIRST_READ,
        lane="ple_prefill_lookahead",
        note=(
            "Chunk k+1's PLE n-gram rows prepared on a worker thread during "
            "chunk k's forward. Armed on a model with no PLE stage it is "
            "inert, and the receipt says 'refused' rather than leaving the "
            "operator to find a missing delta."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_PLE_FIRST_GATHER_EARLY",
        kind="bool",
        parse="lenient_raising",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/ple_row_gather.py:enabled",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_FIRST_READ,
        lane="ple_first_gather_early",
        note=(
            "Starts the first chunk's PLE gather at request arrival -- the one "
            "gather the lookahead above cannot hide, because it has no "
            "previous chunk to run behind. Read through mtplx.ple_row_gather, "
            "not the lookahead module."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_PREFILL_CHUNK_SIZE",
        kind="str",
        parse="text",
        default="2048",
        stack_value="4096",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/generation.py:_prefill_chunk_size",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="prefill_chunk",
        note=(
            "THE ONE TURBO VALUE THIS PROFILE REPLACES: turbo ships 'auto' "
            "(per KV layout, 2048 either way) and the measured prefill stack "
            "runs a fixed 4096. It must stay coherent with "
            "MTPLX_QSA_PREFILL_COMPILE_ROWS -- the QSA prefill graph bank "
            "captures one width, and fable_prefill_chunk."
            "assert_prefill_chunk_coherent refuses the mismatched pair at the "
            "request boundary. Already an operator-overridable key before "
            "this profile existed."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QSA_PREFILL_COMPILE_ROWS",
        kind="str",
        parse="text",
        default="2048",
        stack_value="4096",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/models/qwen4_exp.py:_qsa_prefill_compile_rows",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="prefill_chunk",
        note=(
            "The single row width the QSA prefill graph bank captures. Paired "
            "with MTPLX_PREFILL_CHUNK_SIZE above; changing one without the "
            "other is the incoherence the guard refuses."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_PREFILL_QSA_QUERY_TILE",
        kind="str",
        parse="text",
        default="0",
        stack_value="2048",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/fable_prefill_chunk.py:resolve_query_tile_rows",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="prefill_qsa_query_tile",
        note=(
            "Rows per QSA attention query tile; 0 (the reader's default) means "
            "whole-chunk attention. 2048 is HALF the 4096 chunk above -- a "
            "tile that is not narrower than the widest configured chunk "
            "cannot change a single attention call, and the receipt "
            "reports exactly that as 'refused'."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_GDN_BLOCKED_PREFILL",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/kernels/gdn_blocked_prefill.py:blocked_prefill_env_enabled",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="gdn_blocked_prefill",
        note="Blocked (chunked-scan) GDN prefill route.",
    ),
    EnvKeySpec(
        name="MTPLX_FABLE_PREFILL_MASK_FUSE",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/models/qwen4_exp.py:_prefill_mask_fuse_enabled",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="prefill_mask_fuse",
        receipt=(
            "mtplx/models/qwen4_exp.py -- one 'engaged:' line per fused "
            "shape class, and a refusal line per class MLX cannot fuse"
        ),
        note=(
            "Hands the prefill attention mask to MLX's fused SDPA instead of "
            "materialising it. Per-shape-class: a class this MLX cannot fuse "
            "takes the dense route without disarming the rest."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_SESSION_BANK_MAX_BYTES",
        kind="str",
        parse="text",
        default="",
        stack_value="8G",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/engine_session.py:resolve_session_bank_max_bytes",
        group=GROUP_FABLE_PREFILL,
        binds_at=BIND_CALL,
        lane="session_bank_max_bytes",
        note=(
            "NOT A SPEED KEY -- a serving MEMORY BUDGET, and the one key here "
            "an operator should expect to retune per machine. Unset, the bank "
            "auto-sizes from the machine memory plan; an explicit value pins "
            "it, so that two arms cannot see different banks. Override with "
            "MTPLX_SESSION_BANK_MAX_BYTES=<n>G, or "
            "MTPLX_SESSION_BANK_MAX_BYTES=auto to hand it back to the "
            "auto-sizer; either wins over the default. Oversized boundaries "
            "against too small a budget drop whole entries and re-prefill. "
            "A harness that exports MTPLX_SESSION_BANK_MAX_BYTES=auto gets "
            "the auto-sizer rather than this value -- an export wins, which "
            "is the mechanism working, not a conflict."
        ),
    ),
    # --- registered, but NOT part of the measured control arm -------------
    EnvKeySpec(
        name="MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_DEFAULT,
        owner_site=_NOBODY,
        owner_predicate="never armed by default",
        reader="mtplx/runtime.py:load",
        note=(
            "Relaxed Qwen4 draft ties. Registered so its read is typed and "
            "its spelling is known, but NOT in the stack: the measured "
            "control arm never set it, so it stays a candidate-arm flag."
        ),
        routed=True,
        in_stack=False,
    ),
)

#: The whole measured stack, in registry order: the ABBA control arm plus
#: the retained Fable decode and prefill sets. This is what the startup
#: stack line scores and what ``/health`` publishes under
#: ``engagement_reports.full_stack_selfcheck.stack``.
FULL_STACK_KEYS: tuple[EnvKeySpec, ...] = tuple(
    entry for entry in REGISTERED_KEYS if entry.in_stack
)

#: Just the A/B control arm -- 21 keys. Kept separate from
#: :data:`FULL_STACK_KEYS` because the retained decode and prefill sets have
#: their own second side (the committed flag files, see
#: :data:`GROUP_FLAG_FILES`).
CONTROL_ARM_KEYS: tuple[EnvKeySpec, ...] = tuple(
    entry for entry in FULL_STACK_KEYS if entry.group == GROUP_CONTROL_ARM
)

_BY_NAME: dict[str, EnvKeySpec] = {spec.name: spec for spec in REGISTERED_KEYS}

if len(_BY_NAME) != len(REGISTERED_KEYS):  # pragma: no cover - construction check
    raise RuntimeError("full-stack env registry has duplicate key names")

#: The complete driver stack: what a fully armed serve must resolve to,
#: whoever supplies each value. This is the reference :func:`resolved_stack`
#: checks a live environment against; it is NOT what the profile stamps.
FULL_STACK_RESTACK_ENV: dict[str, str] = {
    spec.name: spec.stack_value for spec in FULL_STACK_KEYS
}

#: The A/B control arm alone.
CONTROL_ARM_ENV: dict[str, str] = {
    spec.name: spec.stack_value for spec in CONTROL_ARM_KEYS
}

#: What the ``turbo-full-stack`` profile stamps: every key of the measured
#: stack that nothing else sets. Restating a SERVER-armed key here would
#: stomp an operator's explicit export, because a profile-owned key beats
#: the environment while the server's auto-arm deliberately yields to it --
#: so the control arm contributes only its three gap keys, while both Fable
#: sets are the profile's in full (no server path, no model contract and no
#: other profile sets any of them).
FULL_STACK_PROFILE_ENV: dict[str, str] = {
    spec.name: spec.stack_value for spec in FULL_STACK_KEYS if spec.stamped_by_profile
}

#: The subset of the profile's own keys whose readers freeze at IMPORT time
#: (:data:`BIND_IMPORT`). ``apply_profile_env`` runs inside
#: ``mtplx/server/openai.py:_load``, long after that module's own imports
#: pulled ``mtplx.runtime_options``, ``mtplx.generation``,
#: ``mtplx.fable_block_verify`` and ``mtplx.fable_draft_k20_prescatter`` --
#: so for these keys, and only these, a profile stamp at that point arms
#: nothing at all. :func:`stamp_import_time_defaults` puts them in the
#: environment from ``mtplx/server/__init__.py``, which Python executes
#: before the first line of ``openai.py``.
IMPORT_TIME_PROFILE_ENV: dict[str, str] = {
    spec.name: spec.stack_value
    for spec in FULL_STACK_KEYS
    if spec.stamped_by_profile and spec.binds_at == BIND_IMPORT
}

#: group -> the committed file that is that group's canonical record, as a
#: repo-relative path. The files are the RECORD; this registry is the source
#: of truth, and ``tests/test_full_stack_profile.py`` asserts they agree so
#: neither can drift.
GROUP_FLAG_FILES: dict[str, str] = {
    GROUP_FABLE_DECODE: "docs/perf/pr391-stack.flags",
    GROUP_FABLE_PREFILL: "docs/perf/pr391-prefill.flags",
}

#: Every spelling that selects the full-stack profile. ``mtplx.profiles``
#: builds its alias table from this, so the early stamp below and the
#: profile registry cannot disagree about what the operator typed.
FULL_STACK_PROFILE_ALIASES: tuple[str, ...] = (
    FULL_STACK_PROFILE_NAME,
    "full-stack",
    "full_stack",
    "turbo_full_stack",
)

#: Set it to a false value to suppress :func:`stamp_import_time_defaults`.
#: The escape hatch for an operator who wants the import-time lanes to keep
#: the values their readers would freeze without a profile.
EARLY_STAMP_ENV = "MTPLX_PROFILE_EARLY_ENV"

#: Family-prefixed keys that ARE read somewhere in the package but are not
#: part of this stack and are not registered in ``mtplx.profiles``. Kept
#: explicit (rather than discovered at import) so the known-set is reviewable
#: and cheap; ``tests/test_full_stack_profile.py`` re-derives it from the
#: source and fails if a new reader appears without landing here.
OTHER_KNOWN_FAMILY_KEYS: tuple[str, ...] = (
    "MTPLX_FRSPEC_LEGACY",
    "MTPLX_FRSPEC_N",
    "MTPLX_QSA_GATHER_DECODE",
    "MTPLX_QSA_SCORE_TILE_ROWS",
)



# ---------------------------------------------------------------------------
# Defaults-on: the retained stack IS the served configuration for this family
# ---------------------------------------------------------------------------
#
# The retained stack stopped being an opt-in on 2026-09-03. The 26 keys above
# whose owner is OWNER_PROFILE are now DEFAULTED ON for a served Qwen3.8
# Flash-Next pack, by exactly the mechanism the server already used for its
# sixteen M4 keys: a ``setdefault`` behind a served-config predicate, which
# every explicit operator export beats. Turning one optimization off is
# therefore a normal env export -- ``MTPLX_FABLE_QSA_SPARSE_DECODE=0`` -- and
# the lane's install verdict says who won.
#
# Two things this section owns, because a key list that lives in two places
# drifts:
#
#   * LANE_KEYS, the operator-facing name of each optimization, grouped so
#     that a lane is always a coherent unit. MTPLX_PREFILL_CHUNK_SIZE and
#     MTPLX_QSA_PREFILL_COMPILE_ROWS share the ``prefill_chunk`` lane because
#     the QSA prefill graph bank captures ONE width: defaulting one on
#     without the other is precisely the pair
#     ``fable_prefill_chunk.assert_prefill_chunk_coherent`` refuses.
#   * DISABLING A LANE MEANS NOT STAMPING ITS KEYS -- never stamping a "0".
#     Each of these readers has its own shipped default (whole-chunk
#     attention, a 2048 chunk, a 1-layer prefix, an auto-sized session bank),
#     and leaving the key unset is what restores it. Stamping "0" would be
#     wrong for the five value knobs and would also make the key look armed.

#: The lane every operator-facing switch names, in registry order.
LANE_KEYS: dict[str, tuple[str, ...]] = {}
for _entry in FULL_STACK_KEYS:
    if _entry.lane:
        LANE_KEYS[_entry.lane] = LANE_KEYS.get(_entry.lane, ()) + (_entry.name,)
del _entry

#: Lane names in registry order.
LANES: tuple[str, ...] = tuple(LANE_KEYS)

#: key -> lane.
KEY_LANE: dict[str, str] = {
    key: lane for lane, keys in LANE_KEYS.items() for key in keys
}

#: Lanes that are armed OUTSIDE this registry's measured 44-key stack but that
#: the operator turns off with the SAME two switches (``--disable-optimization``
#: and ``MTPLX_FABLE_DISABLE``). A stacked lane (e.g. the cached-PLE and pooled-rowsel auxiliary
#: lanes in ``mtplx.qwen4_aux_lanes``) registers its
#: lane->keys here at import so :func:`parse_disable_lanes` accepts its name and
#: ``all`` expands to it -- WITHOUT joining ``FULL_STACK_KEYS``, so the measured
#: stack counts, the committed flag files and the full-stack self-check are
#: unchanged. ``fable_default_env`` still stamps only ``FULL_STACK_PROFILE_ENV``,
#: so a stacked lane owns its own defaults; this registry only shares the
#: operator's off switch.
_EXTRA_LANE_KEYS: dict[str, tuple[str, ...]] = {}


def register_extra_lanes(mapping: Mapping[str, Iterable[str]]) -> None:
    """Register stacked-lane names for the shared opt-out switches.

    Idempotent: re-registering the same lane with the same keys is a no-op.
    A name that collides with a measured-stack lane, or a re-registration with
    different keys, raises -- the two registries must not disagree.
    """

    for lane, keys in mapping.items():
        keys_tuple = tuple(keys)
        if lane in LANE_KEYS:
            raise ValueError(
                f"lane {lane!r} is already a measured-stack lane; "
                "extra lanes must use a distinct name"
            )
        existing = _EXTRA_LANE_KEYS.get(lane)
        if existing is not None and existing != keys_tuple:
            raise ValueError(
                f"lane {lane!r} is already registered with different keys"
            )
        _EXTRA_LANE_KEYS[lane] = keys_tuple


def _all_lane_keys() -> dict[str, tuple[str, ...]]:
    """Measured-stack lanes plus any registered stacked lanes."""

    merged = dict(LANE_KEYS)
    merged.update(_EXTRA_LANE_KEYS)
    return merged


def all_lanes() -> tuple[str, ...]:
    """Every lane name the opt-out switches accept, in registration order."""

    return tuple(LANE_KEYS) + tuple(_EXTRA_LANE_KEYS)

#: The one reserved lane name: every lane at once, i.e. the stock path.
DISABLE_ALL = "all"

#: Comma list of lanes to leave unarmed. Read once per serve, never in a
#: request path.
DISABLE_ENV = "MTPLX_FABLE_DISABLE"

#: The server flag that says the same thing (repeatable).
DISABLE_FLAG = "--disable-optimization"

#: What the defaults actually armed in THIS process, key -> value. Written
#: by :func:`record_defaults_applied` from the two places that arm them (the
#: pre-import stamp and the server's runtime overrides) and read by
#: ``mtplx.fable_install_receipts`` so a verdict can say whether the value it
#: sees came from the defaults or from the operator. Empty in any process
#: that did not arm them -- a driver, a test, another model family.
DEFAULTS_APPLIED: dict[str, str] = {}

#: Value source names for :func:`value_source`.
SOURCE_DEFAULT = "default"
SOURCE_OPERATOR = "operator"


def parse_disable_lanes(raw: str | Iterable[str] | None) -> frozenset[str]:
    """Parse a lane list. Unknown names RAISE; they never silently do nothing.

    Accepts a comma string (the env spelling) or an iterable of tokens (the
    repeated-flag spelling). ``all`` expands to every lane.
    """

    if raw is None:
        return frozenset()
    tokens: list[str] = []
    if isinstance(raw, str):
        tokens = [token.strip().lower() for token in raw.split(",")]
    else:
        for item in raw:
            tokens.extend(token.strip().lower() for token in str(item).split(","))
    tokens = [token for token in tokens if token]
    if not tokens:
        return frozenset()
    measured = set(LANES)
    extra = set(_EXTRA_LANE_KEYS)
    if DISABLE_ALL in tokens:
        # ``all`` here means every MEASURED lane, i.e. the retained stock path.
        # A registered stacked lane owns its own ``all`` handling (it resolves
        # the same token against its own registry), so this function stays
        # measured-only and ``parse_disable_lanes("all") == frozenset(LANES)``
        # holds however many stacked lanes are registered.
        return frozenset(LANES)
    unknown = sorted(set(tokens) - measured - extra)
    if unknown:
        raise ValueError(
            f"unknown optimization lane(s) {', '.join(unknown)}; expected a "
            f"comma list from: {', '.join(all_lanes())}, or {DISABLE_ALL!r}"
        )
    # A registered stacked-lane token is accepted (so the shared switch does not
    # raise) but is NOT returned here: its keys are not in FULL_STACK_PROFILE_ENV
    # and its owner turns it off from its own registry.
    return frozenset(token for token in tokens if token in measured)


def disabled_keys(lanes: Iterable[str]) -> frozenset[str]:
    """Every measured-stack key belonging to the named lanes."""

    resolved = parse_disable_lanes(list(lanes))
    return frozenset(
        key for lane in resolved for key in LANE_KEYS[lane]
    )


def disable_lanes_from_argv(argv: Sequence[str] | None = None) -> list[str]:
    """Every ``--disable-optimization`` value in ``argv``, unparsed.

    Deliberately tiny and total, for the same reason as
    :func:`profile_name_from_argv`: it runs before the server's parser --
    before the imports that build it -- so it cannot use argparse.
    """

    source = list(sys.argv if argv is None else argv)
    found: list[str] = []
    for index, token in enumerate(source):
        if token == DISABLE_FLAG and index + 1 < len(source):
            found.append(source[index + 1].strip())
        elif token.startswith(DISABLE_FLAG + "="):
            found.append(token.partition("=")[2].strip())
    return found


def resolve_disable_lanes(
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    *,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """The union of the env list, the argv flags, and any caller-supplied set.

    All three spellings mean the same thing and compose, so a launcher can
    pin one lane off in the environment while a single run adds another on
    the command line.
    """

    source = os.environ if environ is None else environ
    tokens: list[str] = []
    raw_env = source.get(DISABLE_ENV)
    if raw_env:
        tokens.append(str(raw_env))
    tokens.extend(disable_lanes_from_argv(argv))
    tokens.extend(str(item) for item in extra)
    return parse_disable_lanes(tokens)


def fable_default_env(
    environ: Mapping[str, str] | None = None,
    *,
    disabled_lanes: Iterable[str] = (),
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """The keys the defaults would arm in ``environ``, and their values.

    Excludes a key the operator has already exported (any non-empty value,
    ``0`` included -- that IS the off switch) and every key of a disabled
    lane. ``keys`` narrows the answer to a subset, which is how the
    pre-import stamp asks for only its eleven.
    """

    source = os.environ if environ is None else environ
    off = disabled_keys(disabled_lanes)
    wanted = FULL_STACK_PROFILE_ENV if keys is None else {
        key: FULL_STACK_PROFILE_ENV[key]
        for key in keys
        if key in FULL_STACK_PROFILE_ENV
    }
    return {
        key: value
        for key, value in wanted.items()
        if key not in off and not str(source.get(key) or "").strip()
    }


def record_defaults_applied(applied: Mapping[str, str]) -> None:
    """Remember what the defaults armed, for the install verdicts to read."""

    DEFAULTS_APPLIED.update({str(k): str(v) for k, v in applied.items()})


def value_source(name: str, env: Mapping[str, str] | None = None) -> str:
    """Who put the CURRENT value of ``name`` in the environment.

    ``"default"`` when this process armed it and the value is still the one
    it armed, ``"operator"`` when the key is one of ours and carries a value
    the defaults did not put there, and ``""`` when the answer is not known
    (nothing armed defaults in this process, or the key is not ours).
    """

    if name not in FULL_STACK_PROFILE_ENV:
        return ""
    source = os.environ if env is None else env
    present = str(source.get(name) or "").strip()
    if not present:
        return ""
    if DEFAULTS_APPLIED.get(name) == present:
        return SOURCE_DEFAULT
    if not DEFAULTS_APPLIED:
        return ""
    return SOURCE_OPERATOR


def fable_defaults_report(
    environ: Mapping[str, str] | None = None,
    *,
    disabled_lanes: Iterable[str] = (),
    model_gate: str = "",
) -> dict[str, Any]:
    """What the defaults did, for ``GET /health`` and the startup line.

    ``armed_by_default`` is what this process armed; ``operator_off`` is
    every key of ours the operator exported to something the reader will not
    read as the stack value -- the deliberate opt-outs. ``operator_pinned``
    is an export that agrees with the default, which changes nothing but is
    worth showing so an operator can see their export landed.
    """

    source = os.environ if environ is None else environ
    lanes = frozenset(parse_disable_lanes(list(disabled_lanes)))
    armed = sorted(key for key in DEFAULTS_APPLIED if key in FULL_STACK_PROFILE_ENV)
    operator_off: list[dict[str, str]] = []
    operator_pinned: list[dict[str, str]] = []
    for key, wanted in FULL_STACK_PROFILE_ENV.items():
        if key in DEFAULTS_APPLIED:
            continue
        present = str(source.get(key) or "").strip()
        if not present:
            continue
        row = {"key": key, "lane": KEY_LANE.get(key, ""), "value": present}
        (operator_pinned if present == wanted else operator_off).append(row)
    return {
        "model_gate": model_gate,
        "lanes": list(LANES),
        "armed_by_default": armed,
        "operator_off": operator_off,
        "operator_pinned": operator_pinned,
        "disabled_lanes": sorted(lanes),
        "disabled_keys": sorted(disabled_keys(lanes)),
    }


def defaults_summary_line(
    environ: Mapping[str, str] | None = None,
    *,
    disabled_lanes: Iterable[str] = (),
    model_gate: str = "",
) -> str:
    """One line saying what the defaults armed and what the operator turned off."""

    report = fable_defaults_report(
        environ, disabled_lanes=disabled_lanes, model_gate=model_gate
    )
    head = (
        f"{len(report['armed_by_default'])}/{len(FULL_STACK_PROFILE_ENV)} "
        "retained-stack keys armed by default"
    )
    if model_gate:
        head = f"{head} ({model_gate})"
    parts = [head]
    if report["disabled_lanes"]:
        parts.append("lanes off: " + ", ".join(report["disabled_lanes"]))
    if report["operator_off"]:
        parts.append(
            "operator off: "
            + ", ".join(f"{row['key']}={row['value']}" for row in report["operator_off"])
        )
    if report["operator_pinned"]:
        parts.append(
            "operator pinned: "
            + ", ".join(row["key"] for row in report["operator_pinned"])
        )
    return "; ".join(parts)


#: ``model_type`` spellings that identify a Qwen3.8 Flash-Next pack. Mirrors
#: ``mtplx/server/openai.py:_served_model_type_is_qwen4_exp`` so the
#: pre-import stamp can apply the SAME gate the server applies later,
#: without importing the server (which is the whole point of running early).
FLASH_NEXT_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})


def model_path_from_argv(argv: Sequence[str] | None = None) -> str | None:
    """The ``--model`` value in ``argv``, or ``None``."""

    source = list(sys.argv if argv is None else argv)
    for index, token in enumerate(source):
        if token == "--model" and index + 1 < len(source):
            return source[index + 1].strip()
        if token.startswith("--model="):
            return token.partition("=")[2].strip()
    return None


def is_flash_next_model_dir(model: str | None) -> bool:
    """Does ``<model>/config.json`` name the Qwen3.8 Flash-Next family?

    The same read ``_served_model_type_is_qwen4_exp`` performs, and the same
    total contract: any failure -- no path, no file, unreadable JSON -- is
    ``False``, never an exception.
    """

    if not model:
        return False
    try:
        import json

        with open(os.path.join(str(model), "config.json"), "rb") as handle:
            config = json.load(handle)
    except Exception:
        return False
    text = config.get("text_config") or {}
    spellings = {
        str(config.get("model_type") or "").lower(),
        str(text.get("model_type") or "").lower() if isinstance(text, Mapping) else "",
    }
    return bool(spellings & FLASH_NEXT_MODEL_TYPES)

def group_env(group: str) -> dict[str, str]:
    """The measured values of one group, in registry order."""

    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
    return {
        entry.name: entry.stack_value
        for entry in FULL_STACK_KEYS
        if entry.group == group
    }


def parse_flag_file(text: str) -> dict[str, str]:
    """Parse a ``KEY=value`` flag file: ``#`` comments and blanks ignored.

    The same shape ``set -a; . docs/perf/pr391-stack.flags`` sources, so the
    committed record can be exported by hand for an A/B without the profile.
    """

    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"flag file line is not KEY=value: {raw!r}")
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def render_flag_file(group: str, *, header: str = "") -> str:
    """One group as a ``KEY=value`` flag file, for regenerating the record."""

    lines = [f"# {header}"] if header else []
    lines.extend(f"{key}={value}" for key, value in group_env(group).items())
    return "\n".join(lines) + "\n"


def profile_name_from_argv(argv: Sequence[str] | None = None) -> str | None:
    """The ``--profile`` value in ``argv``, or ``None``.

    Deliberately tiny and total: no argparse (which would need the server's
    whole parser, i.e. the imports this exists to precede), and any malformed
    spelling simply yields ``None``.
    """

    source = list(sys.argv if argv is None else argv)
    for index, token in enumerate(source):
        if token == "--profile" and index + 1 < len(source):
            return source[index + 1].strip()
        if token.startswith("--profile="):
            return token.partition("=")[2].strip()
    return None


def stamp_import_time_defaults(
    argv: Sequence[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Arm the IMPORT-BOUND defaults before their readers can freeze them.

    Called from ``mtplx/server/__init__.py``, which Python executes while
    resolving ``python -m mtplx.server.openai`` -- before a single one of
    ``openai.py``'s module-level imports runs. That is the only window in
    which these keys can still arm anything: their readers freeze the value
    in a module constant at import (:data:`BIND_IMPORT`), and the server's
    own ``setdefault`` block does not run until ``_load``, several thousand
    imported lines later. Everything else the defaults arm is left to that
    block, so this function can never become a second, competing definition
    of what the defaults are -- it only moves NINE of them earlier.

    Same gate, same precedence, same escape hatches as the server block:

    * the served model must be a Flash-Next pack (read from
      ``<--model>/config.json``, the same check
      ``_served_model_type_is_qwen4_exp`` makes), or argv must explicitly
      select the full-stack profile;
    * a key already present in the environment is left alone, so an operator
      export -- ``MTPLX_FABLE_OPDIET=0`` included -- still wins;
    * ``MTPLX_FABLE_DISABLE`` and ``--disable-optimization`` leave a lane's
      keys unstamped, which restores each reader's shipped default;
    * ``EARLY_STAMP_ENV=0`` turns the whole hook off.

    Returns the keys it actually stamped. Never raises: this runs at package
    import, and a startup convenience must not be able to break a load.
    """

    target = os.environ if environ is None else environ
    try:
        if str(target.get(EARLY_STAMP_ENV) or "1").strip().lower() not in TRUE_TOKENS:
            return {}
        family = is_flash_next_model_dir(model_path_from_argv(argv))
        selected = profile_name_from_argv(argv) in FULL_STACK_PROFILE_ALIASES
        if not (family or selected):
            return {}
        lanes = resolve_disable_lanes(target, argv)
        stamped = fable_default_env(
            target,
            disabled_lanes=lanes,
            keys=IMPORT_TIME_PROFILE_ENV,
        )
        for key, value in stamped.items():
            target[key] = value
        record_defaults_applied(stamped)
        return stamped
    except Exception:  # pragma: no cover - an import-time hook must not raise
        return {}


def spec(name: str) -> EnvKeySpec | None:
    """The registry entry for ``name``, or ``None`` when unregistered."""

    return _BY_NAME.get(name)


def registered_names() -> tuple[str, ...]:
    """Every key of the full-stack decode lane, in registry order."""

    return tuple(entry.name for entry in FULL_STACK_KEYS)


def keys_owned_by(owner: str, *, group: str | None = None) -> tuple[str, ...]:
    """Every key a given setter is responsible for, optionally in one group."""

    if owner not in OWNERS:
        raise ValueError(f"unknown owner {owner!r}; expected one of {OWNERS}")
    if group is not None and group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUPS}")
    return tuple(
        entry.name
        for entry in FULL_STACK_KEYS
        if entry.owner == owner and (group is None or entry.group == group)
    )


def _require(name: str) -> EnvKeySpec:
    found = _BY_NAME.get(name)
    if found is None:
        raise KeyError(
            f"{name} is not in the full-stack env registry; add an EnvKeySpec "
            "for it rather than reading os.environ directly"
        )
    return found


def flag_enabled(name: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Read a registered boolean key exactly the way its call site does.

    Reproduces the call site's own parse (see the module docstring): this is
    a routing helper, not a normalization pass. ``strict`` keys defer to
    ``mtplx.runtime_options.env_bool``, imported lazily so this module keeps
    no import-time coupling (``runtime_options`` parses flags at import and
    can raise on a bad spelling).
    """

    entry = _require(name)
    if entry.kind != "bool":
        raise TypeError(f"{name} is a {entry.kind} key; use text_value()")
    source = os.environ if env is None else env
    if entry.parse == "strict":
        from .runtime_options import env_bool

        return env_bool(
            name,
            default=entry.default.strip().lower() in TRUE_TOKENS,
            env=source,
        )
    if entry.parse == "lenient_nostrip":
        return str(source.get(name, entry.default)).lower() in TRUE_TOKENS
    if entry.parse == "lenient_raising":
        raw = str(source.get(name) or entry.default).strip().lower()
        if raw in RAISING_FALSE_TOKENS:
            return False
        if raw in TRUE_TOKENS:
            return True
        accepted = sorted((TRUE_TOKENS | RAISING_FALSE_TOKENS) - {""})
        raise ValueError(
            f"{name} must be one of {accepted}, got {source.get(name)!r}"
        )
    return str(source.get(name) or entry.default).strip().lower() in TRUE_TOKENS


def flag_reader(name: str) -> Callable[[], bool]:
    """A zero-overhead per-call reader for a HOT-PATH gate.

    :func:`flag_enabled` is the right call almost everywhere, but it costs an
    extra stack frame plus a registry lookup and a parse branch per call --
    measured +53 ns (206 vs 153) against the bare expression it replaced,
    which is ~0.08% of a token across 48 layers x 4 per-forward reads. That is
    small, but it is a control-vs-candidate delta in a decode measurement, and
    a measurement tool must not move the thing it measures.

    So the per-forward gates bind this instead: the key, the default and the
    token set are baked in at import, and the closure body is the same work
    the original in-line expression did, in the same single frame.

    The env is still read on EVERY call, deliberately. These gates live on
    modules that are constructed once and then A/B'd by flipping the env
    between arms in one process (tests/test_gdn_step_fused.py and friends do
    exactly that on a shared fixture), so caching at construction -- the
    pattern Qwen4ExpTextModel.__init__ uses for MTPLX_COMPILED_GDN, which is
    read once per model -- would silently change that behaviour.
    """

    entry = _require(name)
    if entry.kind != "bool":
        raise TypeError(f"{name} is a {entry.kind} key; use text_value()")
    default = entry.default
    tokens = TRUE_TOKENS
    environ = os.environ  # the mapping itself: setenv/delenv mutate in place

    if entry.parse == "lenient":

        def _read() -> bool:
            return str(environ.get(name) or default).strip().lower() in tokens

    elif entry.parse == "lenient_nostrip":

        def _read() -> bool:
            return str(environ.get(name, default)).lower() in tokens

    else:  # "strict"/"lenient_raising" -- keep the raising parse as-is

        def _read() -> bool:
            return flag_enabled(name)

    _read.__name__ = f"read_{name.lower()}"
    _read.__qualname__ = _read.__name__
    _read.__doc__ = f"Read {name} exactly as {entry.reader} always did."
    return _read


def text_value(name: str, *, env: Mapping[str, str] | None = None) -> str:
    """Read a registered string key, stripped, falling back to its default."""

    entry = _require(name)
    source = os.environ if env is None else env
    return str(source.get(name) or entry.default).strip()


#: ``mtplx/graphbank.py:_compiled_verify_mode`` off-spellings. Mirrored here
#: so this module compares the resolved mode, not the literal, without
#: importing graphbank (which pulls MLX).
_MODE_OFF_TOKENS = frozenset({"", "0", "false", "no", "off"})
_MODE_NAMED = frozenset({"parity", "parity2"})


def resolved_mode(name: str, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve a ``parse="mode"`` key the way its reader does."""

    entry = _require(name)
    if entry.parse != "mode":
        raise TypeError(f"{name} is not a mode key")
    source = os.environ if env is None else env
    raw = str(source.get(name) or entry.default).strip().lower()
    if raw in _MODE_OFF_TOKENS:
        return "off"
    if raw in _MODE_NAMED:
        return raw
    return "on"


def resolved_stack(env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Is the whole driver stack armed in ``env``, and who was responsible?

    One row per key: the value the stack needs, the value present, whether
    the runtime will behave as the stack requires, and who was supposed to
    supply it. Comparison is on the PARSED value, so ``"true"`` and ``"1"``
    agree and ``MTPLX_COMPILED_VERIFY`` reads ``1`` and ``on`` as one mode.

    ``present``/``supplied_by`` are reported separately from ``ok`` because a
    want-"0" key (LAZY_TARGET_DISTRIBUTIONS, NAX_VERIFY, SKIP_VERIFY_SNAPSHOT)
    also reads as satisfied when it is simply ABSENT and the reader's own
    default happens to match. That is a real distinction: nobody armed it, so
    nothing holds it there if a default or a launcher lane later moves.
    """

    source = os.environ if env is None else env
    rows: list[dict[str, Any]] = []
    for entry in FULL_STACK_KEYS:
        observed = source.get(entry.name)
        present = bool(str(observed or "").strip())
        if entry.parse == "mode":
            wanted_mode = entry.stack_value.strip().lower()
            ok = resolved_mode(entry.name, env=source) == (
                wanted_mode if wanted_mode not in _MODE_OFF_TOKENS else "off"
            )
        elif entry.kind == "bool":
            try:
                actual = flag_enabled(entry.name, env=source)
            except ValueError:  # an unparseable operator spelling
                actual = None
            ok = actual is (entry.stack_value.strip().lower() in TRUE_TOKENS)
        else:
            ok = text_value(entry.name, env=source) == entry.stack_value
        rows.append(
            {
                "name": entry.name,
                "wanted": entry.stack_value,
                "observed": observed,
                "present": present,
                "ok": bool(ok),
                "supplied_by": entry.owner if present else "reader default",
                "owner": entry.owner,
                "owner_predicate": entry.owner_predicate,
            }
        )
    return rows


def stack_summary_line(
    env: Mapping[str, str] | None = None,
    *,
    shape: str | None = None,
) -> str:
    """One line saying how much of the driver stack is armed, and by whom.

    Not a receipt for any lane -- the lanes have their own install reports
    (see mtplx/full_stack_selfcheck.py). This is the env-level answer, which
    is the one that explains a missing lane: a key the server's auto-arm
    skipped because its model predicate did not hold reads as unarmed here.
    ``shape`` is the serve shape the caller knows and this module does not
    (generation mode, model family), so a partial stack can be read against
    the predicates that produced it.
    """

    rows = resolved_stack(env)
    missing = [row for row in rows if not row["ok"]]
    by_source: dict[str, int] = {}
    for row in rows:
        if row["ok"]:
            key = row["supplied_by"]
            by_source[key] = by_source.get(key, 0) + 1
    supplied = ", ".join(f"{name} {count}" for name, count in sorted(by_source.items()))
    head = f"{len(rows) - len(missing)}/{len(rows)} driver-stack keys armed"
    if shape:
        head = f"{head} ({shape})"
    if not missing:
        return f"{head} [{supplied}]"
    detail = ", ".join(
        f"{row['name']}={row['observed']!r} want {row['wanted']!r} "
        f"[{row['owner']}: {row['owner_predicate']}]"
        for row in missing
    )
    return f"{head} [{supplied}]; NOT armed: {detail}"


def known_family_keys() -> frozenset[str]:
    """Every family-prefixed key this package knows how to read or stamp.

    The union of the registry, the keys ``mtplx.profiles`` validates
    (``MODEL_RUNTIME_ENV_OVERRIDE_KEYS`` and
    ``PROFILE_ENV_USER_OVERRIDE_KEYS``), and
    :data:`OTHER_KNOWN_FAMILY_KEYS`. Imported lazily: ``mtplx.profiles``
    imports this module for the profile env block.
    """

    from .profiles import (
        MODEL_RUNTIME_ENV_OVERRIDE_KEYS,
        PROFILE_ENV_USER_OVERRIDE_KEYS,
    )

    known = set(_BY_NAME)
    known.update(OTHER_KNOWN_FAMILY_KEYS)
    for key in (*MODEL_RUNTIME_ENV_OVERRIDE_KEYS, *PROFILE_ENV_USER_OVERRIDE_KEYS):
        if key.startswith(FAMILY_PREFIXES):
            known.add(key)
    return frozenset(known)


def unknown_family_keys(env: Mapping[str, str] | None = None) -> list[str]:
    """Family-prefixed keys present in ``env`` that nothing in mtplx reads."""

    source = os.environ if env is None else env
    known = known_family_keys()
    return sorted(
        key for key in source if key.startswith(FAMILY_PREFIXES) and key not in known
    )


def _nearest_known(name: str, candidates: Iterable[str]) -> str | None:
    """Cheap 'did you mean' by longest shared prefix under the same family."""

    best: str | None = None
    best_score = 0
    for candidate in candidates:
        score = 0
        for left, right in zip(name, candidate):
            if left != right:
                break
            score += 1
        if score > best_score and score > len("MTPLX_QWEN4_"):
            best, best_score = candidate, score
    return best


def warn_unknown_family_keys(
    env: Mapping[str, str] | None = None,
    *,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """Log one WARNING per unreadable family key. Never raises, never mutates.

    Returns the unknown keys so a health surface can carry them. A key here
    is not an error -- it is a key that will silently do nothing, which is
    the failure this registry exists to make visible.
    """

    unknown = unknown_family_keys(env)
    if not unknown:
        return unknown
    source = os.environ if env is None else env
    emit = warn if warn is not None else _default_warn
    known = known_family_keys()
    for key in unknown:
        suggestion = _nearest_known(key, known)
        hint = f"; did you mean {suggestion}?" if suggestion else ""
        try:
            emit(
                f"[mtplx] WARNING: {key}={source.get(key)!r} is set but no "
                f"reader in mtplx consults it -- it will silently do "
                f"nothing{hint}"
            )
        except Exception:  # pragma: no cover - a warning must not break boot
            pass
    return unknown


def _default_warn(line: str) -> None:  # pragma: no cover - trivial sink
    import logging

    logging.getLogger("mtplx.full_stack_env").warning("%s", line)
    print(line, flush=True)


def registry_rows() -> list[dict[str, object]]:
    """The registry as plain data, for ``/health`` and docs generation."""

    return [
        {
            "name": entry.name,
            "kind": entry.kind,
            "parse": entry.parse,
            "default": entry.default,
            "stack_value": entry.stack_value,
            "owner": entry.owner,
            "owner_site": entry.owner_site,
            "owner_predicate": entry.owner_predicate,
            "reader": entry.reader,
            "routed": entry.routed,
            "in_stack": entry.in_stack,
            "group": entry.group,
            "binds_at": entry.binds_at,
            "lane": entry.lane,
            "receipt": entry.receipt,
            "note": entry.note,
        }
        for entry in REGISTERED_KEYS
    ]


__all__ = [
    "BINDS",
    "BIND_CALL",
    "BIND_FIRST_READ",
    "BIND_IMPORT",
    "CONTROL_ARM_ENV",
    "CONTROL_ARM_KEYS",
    "EARLY_STAMP_ENV",
    "EnvKeySpec",
    "FAMILY_PREFIXES",
    "FULL_STACK_KEYS",
    "GROUPS",
    "GROUP_CONTROL_ARM",
    "GROUP_FABLE_DECODE",
    "GROUP_FABLE_PREFILL",
    "GROUP_FLAG_FILES",
    "IMPORT_TIME_PROFILE_ENV",
    "REGISTERED_KEYS",
    "FULL_STACK_PROFILE_ALIASES",
    "FULL_STACK_PROFILE_ENV",
    "FULL_STACK_PROFILE_NAME",
    "FULL_STACK_RESTACK_ENV",
    "OTHER_KNOWN_FAMILY_KEYS",
    "OWNERS",
    "OWNER_DEFAULT",
    "OWNER_PROFILE",
    "OWNER_SERVER_AUTO",
    "OWNER_SERVER_FORCED",
    "RAISING_FALSE_TOKENS",
    "TRUE_TOKENS",
    "flag_enabled",
    "DEFAULTS_APPLIED",
    "DISABLE_ALL",
    "DISABLE_ENV",
    "DISABLE_FLAG",
    "KEY_LANE",
    "LANES",
    "LANE_KEYS",
    "SOURCE_DEFAULT",
    "SOURCE_OPERATOR",
    "defaults_summary_line",
    "disable_lanes_from_argv",
    "disabled_keys",
    "fable_default_env",
    "fable_defaults_report",
    "is_flash_next_model_dir",
    "model_path_from_argv",
    "parse_disable_lanes",
    "record_defaults_applied",
    "resolve_disable_lanes",
    "value_source",
    "group_env",
    "parse_flag_file",
    "profile_name_from_argv",
    "render_flag_file",
    "stamp_import_time_defaults",
    "flag_reader",
    "keys_owned_by",
    "known_family_keys",
    "registered_names",
    "registry_rows",
    "resolved_mode",
    "resolved_stack",
    "spec",
    "stack_summary_line",
    "text_value",
    "unknown_family_keys",
    "warn_unknown_family_keys",
]
