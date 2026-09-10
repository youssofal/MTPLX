from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


KV_QUANT_MODES = ("off", "q8", "q4")

#: The one boolean vocabulary for MTPLX env flags.
#:
#: Every reader of a boolean ``MTPLX_*`` var should go through
#: :func:`env_bool` so a spelling means the same thing everywhere. Values
#: outside these sets raise rather than being silently read as "off" by one
#: reader and "on" by another — the failure mode catalogued in
#: docs/AUDIT_2026-07-18.md, where ``=enabled`` disabled a feature in the
#: server while enabling it in the generation loop.
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})


def env_bool(
    name: str,
    *,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean ``MTPLX_*`` env var, or raise on an unknown spelling.

    Unset (and set-but-empty) yields ``default``. Anything that is neither
    a recognized true nor a recognized false value is a configuration
    error: guessing is what let one variable mean three things.
    """

    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if not token:
        return bool(default)
    if token in ENV_TRUE_VALUES:
        return True
    if token in ENV_FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(ENV_TRUE_VALUES | ENV_FALSE_VALUES))
    raise ValueError(
        f"{name}={raw!r} is not a boolean; expected one of: {accepted}"
    )


#: Exact-preserving op diet for the compiled fixed-M4 verify graph.
#:
#: Read ONCE at import so the hot path never touches ``os.environ`` and so a
#: mid-run env change cannot make two traces of the same graph disagree. With
#: the flag off every gated site executes the pre-diet expression verbatim;
#: with it on the rewritten sites are value-identical by construction (see
#: tests/test_qwen4_opdiet.py, which proves each rewrite against its original
#: on random inputs).
#: ``None`` = resolve from the environment on each read; a test may force a
#: bool. Read at USE, not frozen at import: the server's fixed-M4 auto-arm
#: stamps MTPLX_QWEN4_OPDIET into the environment AFTER this module is imported
#: (via the generation import), so an import-time read froze the default and
#: the served compiled verifier ran without the op diet (arming audit
#: 2026-09-07). The env is frozen once serving starts, so two traces of the
#: same graph still read the same value.
_QWEN4_OPDIET = None

#: The independently selectable rewrites behind the master switch.
#:
#: ``bank``  QSA fixed pooled-bank conditional write (_extend_pooled_fixed)
#: ``rope``  half-width RoPE tables, shared per forward, split-half rotation
#: ``resid`` hyper-connection residual write fused into one kernel
#: ``k20``   eager K20 target/draft support (fused deterministic+ordered pair)
#:
#: Removing dispatches is not the same as removing GPU time: a rewrite can
#: trade contiguous vectorized kernels for broadcast/general ones and lose.
#: ``MTPLX_QWEN4_OPDIET_ITEMS`` exists so a result can be attributed to ONE
#: item instead of the whole flag -- which is how the first ``bank`` spelling
#: was caught (2026-09-01: fewest dispatches of three, slowest of three;
#: the PR #391 harness micro_opdiet.py).
QWEN4_OPDIET_ITEMS = ("bank", "rope", "resid", "k20")


def parse_opdiet_items(
    raw: str | None,
    *,
    known: tuple[str, ...] = QWEN4_OPDIET_ITEMS,
) -> frozenset[str]:
    """Parse ``MTPLX_QWEN4_OPDIET_ITEMS``; unset/empty selects everything.

    An unknown name raises rather than being dropped: a typo that silently
    disables the item under test would make the A/B measure the wrong thing.
    """

    if raw is None:
        return frozenset(known)
    tokens = {token.strip().lower() for token in str(raw).split(",")}
    tokens.discard("")
    if not tokens:
        return frozenset(known)
    if tokens == {"all"}:
        return frozenset(known)
    unknown = sorted(tokens - set(known))
    if unknown:
        raise ValueError(
            f"MTPLX_QWEN4_OPDIET_ITEMS={raw!r} has unknown item(s) "
            f"{', '.join(unknown)}; expected a comma list from: "
            f"{', '.join(known)}"
        )
    return frozenset(tokens)


#: ``None`` = resolve from the environment on each read; a test may force a
#: frozenset. Read at use, not frozen at import (same server-arming reason).
_QWEN4_OPDIET_SELECTED = None

#: First-use latch: the op-diet items actually applied at a gated site in the
#: compiled fixed-M4 verify graph. The graph has no other per-window observable,
#: so /health surfaces this so the battery can confirm which rewrites ran
#: instead of trusting the env. Read-only reporting.
_QWEN4_OPDIET_APPLIED: set[str] = set()


def qwen4_opdiet_enabled(item: str | None = None) -> bool:
    """True when the op diet is armed, and this item is selected.

    ``item=None`` answers only the master switch. Every gated call site names
    its item so ``MTPLX_QWEN4_OPDIET_ITEMS`` can isolate one rewrite. Read at
    use, not frozen at import; a test may set :data:`_QWEN4_OPDIET` /
    :data:`_QWEN4_OPDIET_SELECTED` to force the answer.
    """

    master = (
        _QWEN4_OPDIET
        if _QWEN4_OPDIET is not None
        else env_bool("MTPLX_QWEN4_OPDIET", default=False)
    )
    if not master:
        return False
    if item is None:
        return True
    if item not in QWEN4_OPDIET_ITEMS:
        raise ValueError(f"unknown op-diet item {item!r}")
    selected = (
        _QWEN4_OPDIET_SELECTED
        if _QWEN4_OPDIET_SELECTED is not None
        else parse_opdiet_items(os.environ.get("MTPLX_QWEN4_OPDIET_ITEMS"))
    )
    applied = item in selected
    if applied:
        _QWEN4_OPDIET_APPLIED.add(item)
    return applied


def qwen4_opdiet_report() -> dict:
    """Install/first-use verdict for ``/health qwen4_install_reports.opdiet``.

    ``armed`` is read at use (reflects the served auto-arm stamp, gate-able
    without a request); ``items`` is the configured selection; ``applied`` is
    the first-use latch of items that actually ran at a gated site.
    """

    if not qwen4_opdiet_enabled():
        return {"armed": False, "items": [], "applied": sorted(_QWEN4_OPDIET_APPLIED)}
    selected = (
        _QWEN4_OPDIET_SELECTED
        if _QWEN4_OPDIET_SELECTED is not None
        else parse_opdiet_items(os.environ.get("MTPLX_QWEN4_OPDIET_ITEMS"))
    )
    return {
        "armed": True,
        "items": sorted(selected),
        "applied": sorted(_QWEN4_OPDIET_APPLIED),
    }


def reset_qwen4_opdiet_applied_for_test() -> None:
    """Clear the applied-items latch (tests only)."""

    _QWEN4_OPDIET_APPLIED.clear()


#: W70 -- fused glue inside the compiled fixed-M4 verify body.
#:
#: One flag, per-item selection, same shape as ``MTPLX_QWEN4_OPDIET``: a
#: result must be attributable to ONE rewrite, because "fewer dispatches" and
#: "faster" are different claims (the op diet's first ``bank`` spelling issued
#: the fewest dispatches of three and was the slowest of three).
#:
#: ``qsa_rope``      the attention query/key rotation of a QSA layer -- the
#:                   RoPE table build plus two 5-dispatch rotations -- as ONE
#:                   ``mtplx/kernels/qwen4_m4_rope`` dispatch per layer.
#: ``qsa_rope_idx``  the indexer's query preparation (RMSNorm + partial RoPE)
#:                   through the SHIPPED ``qsa_indexer_prepare_queries_metal``,
#:                   which the fixed-M4 lane never called because
#:                   ``_prepare_queries`` gates on MTPLX_QSA_FUSED_INDEXER.
#:
#: Read ONCE at import, same reasoning as the flags above: the hot path must
#: not touch ``os.environ``, and two traces of the same compiled verify graph
#: must not disagree about which chain they contain.  Off by default.
#:
#: NOT AN ITEM, and the reason is structural rather than a matter of effort:
#: ``hc_triple`` (W69 §4, 194 nodes).  ``hc_norm -> hc_down -> hc_up`` cannot
#: become one dispatch.  ``hc_down`` produces ``mixv[R, 320]`` across 81
#: cooperating threadgroups and every one of ``hc_up``'s 320 threadgroups
#: reads the WHOLE vector; ``hc_norm``'s ``normed[R, 10240]`` is likewise read
#: in full by every ``hc_down`` threadgroup.  Those are grid-wide
#: read-after-write edges, and Metal has no grid-wide barrier inside a
#: dispatch.  The single-threadgroup spelling that would avoid them is the one
#: ``kernels/qwen4_m4_hyper_read`` already measured at 13.2 tok/s against 67.8.
QWEN4_VERIFY_GLUE_ITEMS = ("qsa_rope", "qsa_rope_idx")

#: ``None`` = resolve from the environment on each read; a test may force a
#: bool (directly or via :func:`reset_qwen4_verify_glue_cache`). Read at USE,
#: not frozen at import: the server's fixed-M4 auto-arm stamps
#: MTPLX_QWEN4_VERIFY_GLUE into the environment AFTER this module is imported,
#: so an import-time read froze the default and the served verify body ran
#: without the fused glue (arming audit 2026-09-07).
_QWEN4_VERIFY_GLUE = None


def parse_verify_glue_items(
    raw: str | None,
    *,
    known: tuple[str, ...] = QWEN4_VERIFY_GLUE_ITEMS,
) -> frozenset[str]:
    """Parse ``MTPLX_QWEN4_VERIFY_GLUE_ITEMS``; unset/empty selects everything.

    An unknown name raises rather than being dropped: a typo that silently
    disabled the item under test would make the arm measure the control twice.
    """

    if raw is None:
        return frozenset(known)
    tokens = {token.strip().lower() for token in str(raw).split(",")}
    tokens.discard("")
    if not tokens:
        return frozenset(known)
    if tokens == {"all"}:
        return frozenset(known)
    unknown = sorted(tokens - set(known))
    if unknown:
        raise ValueError(
            f"MTPLX_QWEN4_VERIFY_GLUE_ITEMS={raw!r} has unknown item(s) "
            f"{', '.join(unknown)}; expected a comma list from: "
            f"{', '.join(known)}"
        )
    return frozenset(tokens)


#: ``None`` = resolve from the environment on each read; a test may force a
#: frozenset. Read at use, not frozen at import (same server-arming reason).
_QWEN4_VERIFY_GLUE_SELECTED = None


def qwen4_verify_glue_enabled(item: str | None = None) -> bool:
    """True when the verify-glue flag is armed, and this item is selected.

    Read at use, not frozen at import; a test may set :data:`_QWEN4_VERIFY_GLUE`
    / :data:`_QWEN4_VERIFY_GLUE_SELECTED` (directly or via
    :func:`reset_qwen4_verify_glue_cache`) to force the answer.
    """

    master = (
        _QWEN4_VERIFY_GLUE
        if _QWEN4_VERIFY_GLUE is not None
        else env_bool("MTPLX_QWEN4_VERIFY_GLUE", default=False)
    )
    if not master:
        return False
    if item is None:
        return True
    if item not in QWEN4_VERIFY_GLUE_ITEMS:
        raise ValueError(f"unknown verify-glue item {item!r}")
    selected = (
        _QWEN4_VERIFY_GLUE_SELECTED
        if _QWEN4_VERIFY_GLUE_SELECTED is not None
        else parse_verify_glue_items(
            os.environ.get("MTPLX_QWEN4_VERIFY_GLUE_ITEMS")
        )
    )
    return item in selected


def reset_qwen4_verify_glue_cache(env: Mapping[str, str] | None = None) -> None:
    """Force the verify-glue gates from a given environment.  Tests only.

    The runtime reads these at use and never calls this; it exists so a test
    can arm one item without a subprocess by FORCING the module globals (which
    then win over the environment until reset again).
    """

    global _QWEN4_VERIFY_GLUE, _QWEN4_VERIFY_GLUE_SELECTED
    source = os.environ if env is None else env
    _QWEN4_VERIFY_GLUE = env_bool(
        "MTPLX_QWEN4_VERIFY_GLUE", default=False, env=source
    )
    _QWEN4_VERIFY_GLUE_SELECTED = parse_verify_glue_items(
        source.get("MTPLX_QWEN4_VERIFY_GLUE_ITEMS")
    )


#: Verify-width fused hyper-connection read (mtplx/kernels/qwen4_m4_hyper_read).
#:
#: Read at USE, never frozen at import. ``MTPLX_QWEN4_HC_M4`` is the key; the
#: old ``MTPLX_FABLE_HC_M4`` name is honoured as an alias only when the new key
#: is unset (the new key wins for any non-empty value, including ``0`` for the
#: per-key opt-out). The kernel RAISES on a family-contract miss rather than
#: falling back, so an armed-but-inert lane is unreachable.
#:
#: ``None`` = resolve from the environment on each read; a test may force a
#: bool. This must NOT freeze at import: the server's fixed-M4 auto-arm stamps
#: ``MTPLX_QWEN4_HC_M4`` into the environment AFTER this module is imported, so
#: an import-time read froze the default (False) and the served lane never
#: armed -- the second arming failure the battery caught 2026-09-07 (the first
#: was the sibling QSA sparse-decode reader). The install check runs after the
#: overrides are applied, so reading the environment there sees the stamp.
_QWEN4_HC_M4 = None


def _resolve_qwen4_hc_m4() -> bool:
    raw = os.environ.get("MTPLX_QWEN4_HC_M4")
    if raw is None or not str(raw).strip():
        return env_bool("MTPLX_FABLE_HC_M4", default=False)
    return env_bool("MTPLX_QWEN4_HC_M4", default=False)


def qwen4_hc_m4_enabled() -> bool:
    """True when the HC_M4 flag is armed for this process.

    Armed by ``MTPLX_QWEN4_HC_M4`` (or the old ``MTPLX_FABLE_HC_M4`` alias).
    Read at use, not frozen at import; a test may set :data:`_QWEN4_HC_M4` to
    a bool to force the answer.
    """

    if _QWEN4_HC_M4 is not None:
        return bool(_QWEN4_HC_M4)
    return _resolve_qwen4_hc_m4()



#: Split-K (KV-split) native sparse-GQA attention for the DECODE geometries
#: (native_extensions/qsa_sparse_gqa, mtplx/kernels/qsa_sparse_decode.py).
#:
#: ``MTPLX_QSA_SPARSE_DECODE`` serves the M=4 fixed verify, all 12 QSA
#: layers, once per verify cycle.  This is where the bytes are: the shipped
#: lane materialises a [1, 2, 4, 2052, 256] gathered K/V pair per layer
#: (16.8 MB written, then re-read by the score and P@V GEMMs), plus MLX's own
#: 8.4 MB contiguous copy of the transposed key view.  The kernel reads the
#: cache rows once and never writes them.
#:
#: Off by default.  It RAISES on a contract failure rather than silently
#: reverting -- a silently inert flag is how MTPLX_FUSED_HC_V3 came to be
#: armed-but-dead at M=4.  The one thing that does NOT raise is a PARITY
#: failure at install: this kernel is rounding-class, so a parity miss is a
#: numerical verdict, and the lane disables itself for the process and
#: reports the measured deltas.
def _resolve_qsa_sparse_decode() -> bool:
    # New key wins for any non-empty value (including "0" for the per-key
    # opt-out); the old MTPLX_FABLE_QSA_SPARSE_DECODE name is honoured as an
    # alias only when the new key is unset.
    raw = os.environ.get("MTPLX_QSA_SPARSE_DECODE")
    if raw is None or not str(raw).strip():
        return env_bool("MTPLX_FABLE_QSA_SPARSE_DECODE", default=False)
    return env_bool("MTPLX_QSA_SPARSE_DECODE", default=False)


#: ``None`` = resolve from the environment on every read; a test may force a
#: bool (the fixtures set this directly to arm/disarm the lane).
#:
#: The flag is read at USE, never frozen at import. The server's fixed-M4
#: auto-arm stamps ``MTPLX_QSA_SPARSE_DECODE`` into the environment AFTER this
#: module is imported (``openai.py:_server_runtime_env_overrides``, gated on
#: the built native extension). An import-time read (or a first-use read that
#: happened to fire before the stamp) froze the default (False) before the
#: stamp landed, so the served lane never engaged even with the extension
#: built -- the bug the battery caught 2026-09-07. Reading the environment on
#: each call means the graphbank cache install (after the overrides are
#: applied) always sees the resolved value; the read is a dict lookup and the
#: env is frozen once serving starts.
_QSA_SPARSE_DECODE = None


def qsa_sparse_decode_enabled() -> bool:
    """True when the QSA split-K decode flag is armed for this process.

    Armed by ``MTPLX_QSA_SPARSE_DECODE`` (or the old
    ``MTPLX_FABLE_QSA_SPARSE_DECODE`` alias). Read at use, not frozen at
    import -- see the note on :data:`_QSA_SPARSE_DECODE`. A test may set that
    global to a bool to force the answer.
    """

    if _QSA_SPARSE_DECODE is not None:
        return bool(_QSA_SPARSE_DECODE)
    return _resolve_qsa_sparse_decode()


def _parse_sparse_decode_tile(raw: str | None) -> tuple[int, int]:
    """``"BK:DC"`` -> the compiled tile pair; unset means the default."""

    if raw is None or not str(raw).strip():
        return (128, 32)
    token = str(raw).strip()
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"MTPLX_QSA_SPARSE_DECODE_TILE={raw!r} must be 'BK:DC'"
        )
    try:
        tile = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ValueError(
            f"MTPLX_QSA_SPARSE_DECODE_TILE={raw!r} must be 'BK:DC'"
        ) from exc
    if tile not in QSA_SPARSE_DECODE_TILES:
        accepted = ", ".join(f"{a}:{b}" for a, b in QSA_SPARSE_DECODE_TILES)
        raise ValueError(
            f"MTPLX_QSA_SPARSE_DECODE_TILE={raw!r} is not instantiated; "
            f"expected one of: {accepted}"
        )
    return tile


#: The (BK, DC) pairs the metallib instantiates.  Anything else raises rather
#: than falling back, so a typo in a sweep cannot quietly measure the default.
QSA_SPARSE_DECODE_TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
QSA_SPARSE_DECODE_MAX_SPLITS = 64

#: ``None`` = resolve from the environment on each read; a test may force a
#: (key_tile, dim_tile) tuple. Read at use, not frozen at import (same
#: server-arming ordering as the master flag above).
_QSA_SPARSE_DECODE_TILE = None


#: MEASURED default (2026-09-02, guarded micro, M=4, 16K, 12 layers).  The
#: kernel is occupancy-bound, and at the shipped tile (BK=128) there are 17
#: BK-tiles over the 2,051 selected keys, so 17 is the smallest split target
#: that reaches one tile per threadgroup -- a 4 x 2 x 17 = 136-threadgroup
#: grid on a 40-core M5 Max.  Everything below it leaves cores idle:
#:
#:     splits   n_splits   threadgroups   ms/layer   x baseline
#:          4          4             32      0.325         0.70
#:          8          6             48      0.210         1.08
#:         16          9             72      0.149         1.52
#:         17         17            136      0.094-0.099   2.3-2.4
#:
#: Larger values clamp to the same 17 at BK=128, so 17 is also the point past
#: which the knob stops doing anything -- which is why the first sweep's s17
#: and s32 rows are the SAME configuration measured twice, and their 5.3%
#: spread is the bench's noise floor rather than a result.
#:
#: The previous default of 8 was a placeholder, and it measured 2.2x slower.
QSA_SPARSE_DECODE_DEFAULT_SPLITS = 17


def _parse_sparse_decode_splits(raw: str | None) -> int:
    """``MTPLX_QSA_SPARSE_DECODE_SPLITS`` -- the KV-split target."""

    if raw is None or not str(raw).strip():
        return QSA_SPARSE_DECODE_DEFAULT_SPLITS
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(
            f"MTPLX_QSA_SPARSE_DECODE_SPLITS={raw!r} must be an integer"
        ) from exc
    if not 1 <= value <= QSA_SPARSE_DECODE_MAX_SPLITS:
        raise ValueError(
            f"MTPLX_QSA_SPARSE_DECODE_SPLITS={raw!r} must be in "
            f"[1, {QSA_SPARSE_DECODE_MAX_SPLITS}]"
        )
    return value


#: ``None`` = resolve from the environment on each read; a test may force an
#: int. Read at use, not frozen at import.
_QSA_SPARSE_DECODE_SPLITS = None


def qsa_sparse_decode_tile() -> tuple[int, int]:
    """The armed ``(key_tile, dimension_tile)`` for the decode kernel.

    Read at use, not frozen at import; a test may set
    :data:`_QSA_SPARSE_DECODE_TILE` to force the answer.
    """

    if _QSA_SPARSE_DECODE_TILE is not None:
        return _QSA_SPARSE_DECODE_TILE
    return _parse_sparse_decode_tile(
        os.environ.get("MTPLX_QSA_SPARSE_DECODE_TILE")
        or os.environ.get("MTPLX_FABLE_QSA_SPARSE_DECODE_TILE")
    )


def qsa_sparse_decode_splits() -> int:
    """The armed KV-split target for the decode kernel.

    Read at use, not frozen at import; a test may set
    :data:`_QSA_SPARSE_DECODE_SPLITS` to force the answer.
    """

    if _QSA_SPARSE_DECODE_SPLITS is not None:
        return _QSA_SPARSE_DECODE_SPLITS
    return _parse_sparse_decode_splits(
        os.environ.get("MTPLX_QSA_SPARSE_DECODE_SPLITS")
        or os.environ.get("MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS")
    )


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def parser_option_names(parser: object, namespace: object = None) -> set[str]:
    """Every option name reachable in the parse context, without dashes.

    Walks the root parser plus whichever subparsers the parse actually
    descended into (using ``namespace`` to pick the branch), which is the
    same scope argparse resolves abbreviations against.
    """

    names: set[str] = set()
    seen: set[int] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for action in getattr(current, "_actions", ()):
            for option in getattr(action, "option_strings", ()) or ():
                names.add(str(option).lstrip("-"))
            choices = getattr(action, "choices", None)
            dest = getattr(action, "dest", None)
            if not isinstance(choices, dict) or not dest:
                continue
            # A subparsers action: follow only the branch that was taken.
            picked = getattr(namespace, str(dest), None) if namespace else None
            if picked is not None and picked in choices:
                pending.append(choices[picked])
            elif namespace is None:
                pending.extend(choices.values())
    return names


def canonicalize_flag_tokens(
    tokens: set[str],
    parser: object,
    namespace: object = None,
) -> set[str]:
    """Expand argparse abbreviations to the flag names they resolved to.

    ``--temp 0.9`` sets ``args.temperature`` but was recorded as ``temp``,
    so every ``"temperature" in cli_flags`` check read it as *not typed*
    and the config file happily overwrote the user's value. Resolving here
    (rather than setting ``allow_abbrev=False``) keeps abbreviations
    working while making the explicit-flag signal true.

    The raw token is kept alongside the expansion, so checks written
    against either spelling keep working. Ambiguous prefixes expand to
    nothing — argparse would have rejected the command anyway.
    """

    known = parser_option_names(parser, namespace)
    resolved = set(tokens)
    for token in tokens:
        if token in known:
            continue
        matches = {name for name in known if name.startswith(token)}
        if len(matches) == 1:
            resolved |= matches
    return resolved


def block_prefix_restore_enabled() -> bool:
    """The single parse of ``MTPLX_SESSION_BLOCK_PREFIX_RESTORE``.

    Default ON. Every reader — the decode loop, the engine session, the
    session bank's cold tier, and the server's settings view — goes through
    this, so one spelling cannot mean ON in one and OFF in another. Unset
    used to mean OFF in the cold tier and ON everywhere else, which
    silently disabled cold-tier block-prefix restore for library embedders
    (the CLI path masks it by force-setting "1"); and the server's
    allowlist-only read reported "off" for spellings the runtime honours as
    on, e.g. ``=enabled``.

    Lives here rather than in :mod:`mtplx.session_bank` because this module
    has no heavy imports: the server reads the setting on paths where the
    mlx-backed runtime may be unavailable.
    """

    return env_bool("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", default=True)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def generate_api_key_file(api_key_file: str | os.PathLike[str]) -> str:
    """Create ``api_key_file`` holding a fresh random key and return the key.

    Server entrypoints call this when the user passed ``--api-key-file`` for a
    path that does not exist yet, so the recovery command our own non-localhost
    refusal prints is runnable as-is instead of dying on FileNotFoundError.
    Read-only consumers of key files (doctor, connect) must NOT call this — a
    missing file is a real error there. The file is created 0600.
    """
    import secrets

    path = Path(api_key_file).expanduser()
    key = "mtplx-" + secrets.token_hex(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
