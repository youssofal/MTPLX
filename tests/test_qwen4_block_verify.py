"""Pure-python coverage for block verification on the stock native-MTP lane.

No MLX import happens here.  ``mtplx.qwen4_block_verify``,
``mtplx.qwen4_k20_log``, ``mtplx.sampling`` and the offline scorer are all
plain NumPy, so the law can be driven directly; the generation-side wiring is
checked by source inspection, which is how the rest of this lane is guarded
(see ``tests/test_qwen4_k20_log.py`` and ``tests/test_pr391_float32_d3_core.py``).

The oracle is ``tests/block_verify_reference.py``.  Every
behavioural test below builds the SAME prepared rows for both implementations
and demands that the in-loop one reproduce the reference's decisions.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mtplx import qwen4_block_verify as bv_mod
from mtplx.qwen4_block_verify import (
    BlockVerifier,
    build_verifier,
    prepared_pair,
    water_fill_lambda,
)
from mtplx.sampling import (
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
    sample_from_distribution,
)

from block_verify_reference import (
    DEPTH,
    LAYOUT_STOCK_BV,
    Window,
    block_ladder_columns,
    decide_block,
    load_log,
    prepare_residual,
    score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Row helpers -- one set of prepared rows, shared by both implementations.
# ---------------------------------------------------------------------------


def _sparse(pairs: dict[int, float]) -> SparseDistribution:
    ids = np.asarray(sorted(pairs), dtype=np.int64)
    probs = np.asarray([pairs[int(token)] for token in ids], dtype=np.float64)
    return SparseDistribution(ids, probs, 100000)


def _rows(draft_pairs, target_pairs):
    draft = [_sparse(pairs) for pairs in draft_pairs]
    target = [_sparse(pairs) for pairs in target_pairs]
    return draft, target


def _window(draft, target, tokens, uniforms, *, bonus=True, stops=()) -> Window:
    """The reference's Window over the SAME preparation the lane applies."""

    return Window(
        draft_tokens=tokens,
        draft_rows=[prepared_pair(row) for row in draft],
        target_rows=[prepared_pair(row) for row in target],
        uniforms=np.asarray(uniforms, dtype=np.float64),
        bonus_allowed=bonus,
        stops=frozenset(int(token) for token in stops),
    )


def _verifier(draft, target, tokens) -> BlockVerifier:
    verifier = build_verifier(
        draft_tokens=tokens, draft_probs=draft, target_list=target
    )
    assert verifier is not None
    return verifier


# A ladder that drops under water at depth 1 (the target dislikes x1), then
# meets a depth-2 token the target likes far more than the draft did -- the
# shape block verification exists to exploit.
_UNDERWATER_DRAFT = [
    {1: 0.22, 2: 0.36, 3: 0.42},
    {4: 0.40, 5: 0.30, 6: 0.30},
    {7: 0.55, 8: 0.35, 9: 0.10},
]
_UNDERWATER_TARGET = [
    {1: 0.20, 2: 0.46, 3: 0.34},
    {4: 0.30, 5: 0.26, 6: 0.44},
    {7: 0.52, 8: 0.30, 9: 0.18},
    {10: 0.60, 11: 0.40},
]
_UNDERWATER_TOKENS = [1, 4, 7]

# A ladder that stays at 1 through depths 1-2 (rho >= 1 there, so the running
# product never dips) and only then meets a token the target likes less.  The
# block law must be the shipped law token for token on such a window: both the
# coin and the residual scale collapse.  The last depth is the ONLY place a
# rejection can happen on a unit ladder -- rho >= 1 everywhere else means
# a_d = 1 there -- which is exactly what makes this the partial parity check.
_UNIT_DRAFT = [
    {1: 0.50, 2: 0.50},
    {4: 0.60, 5: 0.40},
    {7: 0.30, 8: 0.70},
]
_UNIT_TARGET = [
    {1: 0.80, 2: 0.20},
    {4: 0.75, 5: 0.25},
    {7: 0.20, 8: 0.80},
    {10: 1.0},
]
_UNIT_TOKENS = [1, 4, 7]

# Five-wide rows whose depth-2 residual keeps more than one token, so the
# scale's effect on the correction's SHAPE (not just its support) is visible.
_WIDE_DRAFT = [
    {1: 0.32, 2: 0.01, 3: 0.33, 4: 0.06, 5: 0.29},
    {6: 0.28, 7: 0.05, 8: 0.23, 9: 0.10, 10: 0.34},
    {11: 0.23, 12: 0.20, 13: 0.24, 14: 0.06, 15: 0.26},
]
_WIDE_TARGET = [
    {1: 0.19, 2: 0.24, 3: 0.08, 4: 0.02, 5: 0.47},
    {6: 0.05, 7: 0.21, 8: 0.32, 9: 0.19, 10: 0.24},
    {11: 0.13, 12: 0.24, 13: 0.18, 14: 0.19, 15: 0.25},
    {20: 0.60, 21: 0.40},
]
_WIDE_TOKENS = [1, 6, 11]


# ---------------------------------------------------------------------------
# A faithful replica of the stock accept loop, so both laws can be driven and
# their RNG traffic counted without importing generation.py (which needs MLX).
# The AST contract below pins the real loop to this shape.
# ---------------------------------------------------------------------------


class _CountingRNG:
    """Wraps a Generator and counts the two draw kinds the loop can make."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.randoms = 0
        self.choices = 0

    def random(self) -> float:
        self.randoms += 1
        return float(self._rng.random())

    def choice(self, values, *, p):
        self.choices += 1
        return self._rng.choice(values, p=p)

    @property
    def draws(self) -> int:
        return self.randoms + self.choices


class _Result:
    __slots__ = (
        "accepted",
        "first_reject",
        "selected_kind",
        "selected_present",
        "draws_used",
        "accept_probability",
        "correction",
        "bonus",
    )

    def __init__(self) -> None:
        self.accepted = 0
        self.first_reject = -1
        self.selected_kind = 0
        self.selected_present = False
        self.draws_used = 0
        self.accept_probability = [0.0] * DEPTH
        self.correction: int | None = None
        self.bonus: int | None = None

    def key(self) -> tuple[int, ...]:
        return (
            self.accepted,
            self.first_reject,
            self.selected_kind,
            int(self.selected_present),
            self.draws_used,
        )


def _accept_loop(
    draft,
    target,
    tokens,
    coins,
    *,
    verifier: BlockVerifier | None,
    rng,
    bonus_allowed: bool = True,
    stops: frozenset[int] = frozenset(),
):
    """``generation.generate_mtpk``'s accept loop, host-side, both laws.

    Mirrors the real loop's structure exactly: one accept coin per depth
    reached, ``<=`` owns the tie, the correction is one ``rng.choice`` off the
    residual (scaled when armed), the bonus is one more.  ``coins`` stands in
    for ``float(rng.random())`` so a path can be forced; the rng is still
    handed to the correction and bonus draws so they are counted.
    """

    out = _Result()
    for depth, draft_token in enumerate(tokens):
        accept_prob = acceptance_probability(target[depth], draft[depth], draft_token)
        if verifier is not None:
            accept_prob = verifier.accept_probability[depth]
        coin = float(coins[depth])
        accepted_now = coin <= accept_prob
        if accepted_now:
            correction = draft_token
        elif verifier is not None:
            correction = sample_from_distribution(verifier.scaled_residual(depth), rng)
        else:
            correction = sample_from_distribution(
                residual_distribution(target[depth], draft[depth]), rng
            )
        out.accept_probability[depth] = float(accept_prob)
        if accepted_now:
            out.accepted = depth + 1
            out.draws_used = depth + 1
            if draft_token in stops:
                return out
            continue
        out.first_reject = depth
        out.correction = int(correction)
        out.selected_kind = 1
        out.selected_present = True
        out.draws_used = depth + 2
        return out
    if bonus_allowed:
        out.bonus = int(sample_from_distribution(target[len(tokens)], rng))
        out.selected_kind = 2
        out.selected_present = True
        out.draws_used = len(tokens) + 1
    return out


def _coins_for(path: list[bool], probabilities: list[float]) -> list[float]:
    """Uniforms that realise ``path`` under a law whose coins are given."""

    coins = []
    for accept, probability in zip(path, probabilities):
        if accept:
            coins.append(0.0)
        else:
            assert probability < 1.0, "cannot force a rejection at a=1"
            coins.append(min(1.0, probability + 1e-9))
    coins.extend([0.0] * (DEPTH - len(coins)))
    return coins


# ---------------------------------------------------------------------------
# (a) The flag-off contract on the real accept loop.
# ---------------------------------------------------------------------------


def _generation_source() -> str:
    return (REPO_ROOT / "mtplx" / "generation.py").read_text()


def _accept_loop_source(source: str) -> str:
    """The stock native-MTP accept loop, from its `for` to the round's end."""

    start = source.index(
        "for depth_index, draft_token in enumerate(_host_accept_drafts):"
    )
    end = source.index("elapsed_accept = max(", start)
    return source[start:end]


def test_the_gate_is_read_at_use_and_only_in_one_place():
    source = _generation_source()
    # generation.py consults the reader at USE, not a module constant frozen
    # at import: the served fixed-M4 auto-arm stamps the key after import, so a
    # frozen constant would leave the lane off as launched (arming audit).
    assert "_qwen4_block_verify_enabled()" in source
    assert "_QWEN4_BLOCK_VERIFY = " not in source
    # generation.py never reads the env var itself.
    assert 'os.environ.get("MTPLX_QWEN4_BLOCK_VERIFY"' not in source
    assert '_env_truthy("MTPLX_QWEN4_BLOCK_VERIFY")' not in source
    module = (REPO_ROOT / "mtplx" / "qwen4_block_verify.py").read_text()
    assert module.count('_ENV_VAR = "MTPLX_QWEN4_BLOCK_VERIFY"') == 1
    # the env is read in exactly one place (is_enabled), at use.
    assert module.count("_env_truthy(_ENV_VAR)") == 1
    assert "os.environ" not in inspect.getsource(bv_mod.BlockVerifier)
    assert "os.environ" not in inspect.getsource(bv_mod.build_verifier)


def test_the_shipped_law_survives_verbatim_in_the_accept_loop():
    """Flag off => the loop evaluates exactly what it evaluated before."""

    loop = _accept_loop_source(_generation_source())
    # Both sampled branches keep their original acceptance expression...
    assert (
        "accept_prob = (\n"
        "                    1.0 if q <= 0 and p > 0 else "
        "(0.0 if q <= 0 else min(1.0, p / q))\n"
        "                )" in loop
    )
    assert "accept_prob = compute_acceptance_probability(\n" in loop
    # ...and their original residual, reached whenever `_bv` is None.
    assert "residual_distribution(target_p, draft_q), rng" in loop
    assert "residual_distribution(\n" in loop
    assert "else target_distribution_batch.to_distribution(depth_index)," in loop
    # The accept coin is still drawn once per depth and compared with `<=`,
    # so arming block verification cannot shift the PCG64 stream.
    assert loop.count("accepted_now = float(rng.random()) <= accept_prob") == 2
    assert loop.count("rng.random()") == 2


def test_every_block_verification_read_is_guarded():
    loop = _accept_loop_source(_generation_source())
    # Each mention of the verifier is inside an `if/elif _bv is not None`.
    reads = [line.strip() for line in loop.splitlines() if "_bv." in line]
    assert reads, "the loop must read the ladder somewhere"
    for index, line in enumerate(loop.splitlines()):
        if "_bv." not in line:
            continue
        guard = "\n".join(loop.splitlines()[max(0, index - 6) : index])
        assert "_bv is not None" in guard, line
    # The two overrides and the two scaled residuals, and nothing else.
    assert loop.count("_bv.accept_probability[depth_index]") == 2
    assert loop.count("_bv.scaled_residual(depth_index)") == 2
    assert loop.count("elif _bv is not None:") == 2
    assert loop.count("if _bv is not None:") - loop.count(
        "elif _bv is not None:"
    ) == 2


def test_the_verifier_is_built_once_before_the_loop_and_is_gated():
    source = _generation_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_mtpk"
    )
    body = ast.get_source_segment(source, function)
    built = body.index("_bv = _qwen4_build_block_verifier(")
    loop = body.index(
        "for depth_index, draft_token in enumerate(_host_accept_drafts):"
    )
    assert built < loop
    assert body.count("_qwen4_build_block_verifier(") == 1
    gate = body.rindex("_qwen4_block_verify_enabled()", 0, built)
    # The construction sits under the at-use gate, and under the preconditions
    # that make a block law meaningful at all.
    guard = body[gate:built]
    assert "sampler.temperature > 0" in guard
    assert "target_prefix_tokens is None" in guard
    assert "_bv = None" in body[max(0, gate - 200) : gate]


def test_no_rng_reaches_the_block_verification_module():
    """The law draws nothing, so arming it cannot shift the PCG64 stream."""

    path = REPO_ROOT / "mtplx" / "qwen4_block_verify.py"
    module = path.read_text()
    assert "import mlx" not in module and "mlx.core" not in module
    assert "np.random" not in module
    tree = ast.parse(module)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [argument.arg for argument in node.args.args]
            names += [argument.arg for argument in node.args.kwonlyargs]
            assert "rng" not in names, node.name
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"random", "choice", "default_rng"}, node.attr
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "random" not in ast.dump(node)


# ---------------------------------------------------------------------------
# (b) The in-loop law equals the offline reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accept_length", [0, 1, 2, 3])
def test_in_loop_matches_the_reference_for_every_accept_length(accept_length):
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    path = [True] * accept_length + ([False] if accept_length < DEPTH else [])
    coins = _coins_for(path, verifier.accept_probability)

    window = _window(draft, target, _UNDERWATER_TOKENS, coins + [0.5])
    expected = decide_block(window, cap_mode="reach")
    got = _accept_loop(
        draft,
        target,
        _UNDERWATER_TOKENS,
        coins,
        verifier=verifier,
        rng=_CountingRNG(1),
    )
    assert got.key() == expected.key(with_token=False)
    assert got.accepted == accept_length
    np.testing.assert_array_equal(
        got.accept_probability, expected.accept_probability
    )


def test_the_ladder_is_bit_identical_to_the_reference_ladder():
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    window = _window(draft, target, _UNDERWATER_TOKENS, [0.5] * (DEPTH + 1))
    columns = block_ladder_columns(window, cap_mode="reach")
    np.testing.assert_array_equal(verifier.accept_probability, columns["coin"])
    np.testing.assert_array_equal(verifier.residual_scale, columns["scale"])
    np.testing.assert_array_equal(verifier.budget, columns["budget"])
    np.testing.assert_array_equal(verifier.realised, columns["realised"])
    np.testing.assert_array_equal(verifier.clipped, columns["clipped"])


def test_the_ladder_actually_leaves_one_and_buys_reach():
    """The fixture must exercise the lever, or the tests above prove nothing."""

    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    shipped = [
        acceptance_probability(target[d], draft[d], _UNDERWATER_TOKENS[d])
        for d in range(DEPTH)
    ]
    assert verifier.residual_scale[1] < 1.0  # under water from depth 2 on
    assert verifier.residual_scale[2] < 1.0
    # Depth 2's coin is strictly better than the shipped law's on this
    # realisation, which is the whole lever.  Depth 1's is NOT pinned to
    # min(1, rho_1): it is saturated only in EXPECTATION over x_2 (H §3.1),
    # which is what `test_the_water_fill_holds_the_budget_in_expectation`
    # checks; per realisation the look-ahead moves it either way.
    assert verifier.accept_probability[1] > shipped[1]
    assert sum(verifier.clipped) == 0


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_the_correction_is_the_scaled_residual_the_reference_builds(depth):
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    window = _window(draft, target, _UNDERWATER_TOKENS, [0.5] * (DEPTH + 1))

    got = verifier.scaled_residual(depth)
    target_ids, target_double = window.target_double[depth]
    draft_ids, draft_probs = window.draft[depth]
    expected = prepare_residual(
        target_ids,
        target_double,
        draft_ids,
        draft_probs,
        scale=np.float64(verifier.residual_scale[depth]),
    )
    if expected is None:
        # The reference falls back to the double-normalised target row when
        # nothing survives (c*p - q)+; so must the lane.
        expected = (target_ids, target_double)
    np.testing.assert_array_equal(got.token_ids, expected[0].astype(np.int64))
    np.testing.assert_array_equal(got.probs, expected[1])


def test_the_scaled_residual_reshapes_the_correction_under_water():
    """(c*p - q)+ is not (p - q)+ rescaled: the scale moves support AND shape."""

    draft, target = _rows(_WIDE_DRAFT, _WIDE_TARGET)
    verifier = _verifier(draft, target, _WIDE_TOKENS)
    assert verifier.residual_scale[1] < 1.0
    scaled = verifier.scaled_residual(1)
    shipped = residual_distribution(target[1], draft[1])
    assert scaled.token_ids.size >= 2
    assert set(scaled.token_ids.tolist()) < set(shipped.token_ids.tolist())
    assert not np.allclose(scaled.probs[:1], shipped.probs[:1])

    # ... and it IS the shipped residual where the ladder is still at 1.
    assert verifier.residual_scale[0] == 1.0
    at_depth_zero = verifier.scaled_residual(0)
    reference = residual_distribution(target[0], draft[0])
    np.testing.assert_array_equal(at_depth_zero.token_ids, reference.token_ids)
    np.testing.assert_allclose(at_depth_zero.probs, reference.probs, rtol=0, atol=1e-15)


@pytest.mark.parametrize("accept_length", [2, 3])
def test_the_c_equals_one_identity_holds_token_for_token(accept_length):
    """H §3.2: with the ladder at 1 the block law IS the shipped law."""

    draft, target = _rows(_UNIT_DRAFT, _UNIT_TARGET)
    verifier = _verifier(draft, target, _UNIT_TOKENS)
    shipped = [
        acceptance_probability(target[d], draft[d], _UNIT_TOKENS[d])
        for d in range(DEPTH)
    ]
    assert all(scale == 1.0 for scale in verifier.residual_scale)
    assert verifier.accept_probability[:2] == [1.0, 1.0]
    np.testing.assert_allclose(verifier.accept_probability, shipped, rtol=0, atol=0)

    path = [True] * accept_length + ([False] if accept_length < DEPTH else [])
    coins = _coins_for(path, shipped)
    block = _accept_loop(
        draft, target, _UNIT_TOKENS, coins, verifier=verifier, rng=_CountingRNG(5)
    )
    current = _accept_loop(
        draft, target, _UNIT_TOKENS, coins, verifier=None, rng=_CountingRNG(5)
    )
    assert block.key() == current.key()
    assert block.correction == current.correction
    assert block.bonus == current.bonus
    assert block.accept_probability == current.accept_probability


def test_a_zero_mass_drafted_token_is_an_unconditional_rejection():
    """alpha = 0 rows: the target zeroed the drafted token (H §1.2, 8-16%)."""

    draft, target = _rows(
        [{1: 0.5, 2: 0.5}, {4: 1.0}, {7: 1.0}],
        [{2: 1.0}, {4: 0.5, 5: 0.5}, {7: 1.0}, {9: 1.0}],
    )
    verifier = _verifier(draft, target, [1, 4, 7])
    assert verifier.accept_probability[0] == 0.0
    assert verifier.budget[0] == 0.0
    window = _window(draft, target, [1, 4, 7], [0.5, 0.5, 0.5, 0.5])
    expected = decide_block(window, cap_mode="reach")
    got = _accept_loop(
        draft, target, [1, 4, 7], [0.5, 0.5, 0.5], verifier=verifier, rng=_CountingRNG(2)
    )
    assert got.key() == expected.key(with_token=False)
    assert got.accepted == 0 and got.first_reject == 0


def test_a_uniform_equal_to_the_coin_accepts():
    """`<=` tie ownership, including the degenerate a_d = 0 coin."""

    draft, target = _rows(
        [{1: 0.5, 2: 0.5}, {4: 1.0}, {7: 1.0}],
        [{2: 1.0}, {4: 0.5, 5: 0.5}, {7: 1.0}, {9: 1.0}],
    )
    verifier = _verifier(draft, target, [1, 4, 7])
    window = _window(draft, target, [1, 4, 7], [0.0, 0.0, 0.0, 0.5])
    expected = decide_block(window, cap_mode="reach")
    got = _accept_loop(
        draft, target, [1, 4, 7], [0.0, 0.0, 0.0], verifier=verifier, rng=_CountingRNG(3)
    )
    assert got.key() == expected.key(with_token=False)
    assert got.accepted == DEPTH  # a coin of exactly 0.0 accepts a 0.0 probability


def test_an_accepted_stop_token_ends_the_window_without_a_bonus():
    draft, target = _rows(_UNIT_DRAFT, _UNIT_TARGET)
    verifier = _verifier(draft, target, _UNIT_TOKENS)
    window = _window(
        draft, target, _UNIT_TOKENS, [0.0] * (DEPTH + 1), stops=(_UNIT_TOKENS[1],)
    )
    expected = decide_block(window, cap_mode="reach")
    got = _accept_loop(
        draft,
        target,
        _UNIT_TOKENS,
        [0.0, 0.0, 0.0],
        verifier=verifier,
        rng=_CountingRNG(4),
        stops=frozenset({_UNIT_TOKENS[1]}),
    )
    assert got.key() == expected.key(with_token=False)
    assert got.accepted == 2 and got.bonus is None


def test_full_accept_emits_the_bonus_from_the_unchanged_row():
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    rng = _CountingRNG(11)
    got = _accept_loop(
        draft,
        target,
        _UNDERWATER_TOKENS,
        [0.0, 0.0, 0.0],
        verifier=verifier,
        rng=rng,
    )
    assert got.accepted == DEPTH and got.selected_kind == 2
    assert got.bonus in set(int(t) for t in target[DEPTH].token_ids)
    assert rng.choices == 1  # the bonus, and nothing else


def test_the_verifier_declines_when_a_row_is_missing():
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS,
            draft_probs=[draft[0], None, draft[2]],
            target_list=target,
        )
        is None
    )
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS,
            draft_probs=draft,
            target_list=target[:2],
        )
        is None
    )
    assert (
        build_verifier(
            draft_tokens=_UNDERWATER_TOKENS, draft_probs=draft, target_list=None
        )
        is None
    )


def test_the_verifier_reads_a_batched_support_the_same_way():
    class _Batch:
        vocab_size = 100000

        def __init__(self, rows):
            width = max(row.token_ids.size for row in rows)
            self.token_ids = np.zeros((len(rows), width), dtype=np.int64)
            self.probs = np.zeros((len(rows), width), dtype=np.float64)
            for index, row in enumerate(rows):
                size = int(row.token_ids.size)
                self.token_ids[index, :size] = row.token_ids
                self.probs[index, :size] = row.probs

    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    from_list = _verifier(draft, target, _UNDERWATER_TOKENS)
    from_batch = build_verifier(
        draft_tokens=_UNDERWATER_TOKENS,
        draft_probs=draft,
        target_batch=_Batch(target),
    )
    assert from_batch is not None
    np.testing.assert_array_equal(
        from_batch.accept_probability, from_list.accept_probability
    )
    np.testing.assert_array_equal(
        from_batch.residual_scale, from_list.residual_scale
    )


def test_the_water_fill_holds_the_budget_in_expectation():
    """E_{x_{d+1}~q}[w_d] = A_d -- the property that keeps the law exact."""

    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    ids, probs = prepared_pair(draft[1])
    realised = []
    for token in ids:
        tokens = [_UNDERWATER_TOKENS[0], int(token), _UNDERWATER_TOKENS[2]]
        verifier = _verifier(draft, target, tokens)
        realised.append(verifier.realised[0])
    expectation = float(np.sum(probs * np.asarray(realised, dtype=np.float64)))
    reference = _verifier(draft, target, _UNDERWATER_TOKENS)
    assert expectation == pytest.approx(reference.budget[0], abs=1e-12)


def test_water_fill_lambda_is_the_reference_solver():
    q = np.array([0.5, 0.3, 0.2])
    base = np.array([0.1, 0.4, 0.9])
    level = water_fill_lambda(q, base, np.float64(1.0), np.float64(0.6))
    assert float(np.sum(q * np.minimum(1.0, base + level))) == pytest.approx(0.6)
    # Already at or above the budget: no water is added.
    assert float(water_fill_lambda(q, base, np.float64(1.0), np.float64(0.2))) == 0.0


# ---------------------------------------------------------------------------
# (c) Draw accounting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accept_length", [0, 1, 2, 3])
def test_both_laws_draw_the_same_uniforms_on_the_same_outcome_path(accept_length):
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    shipped = [
        acceptance_probability(target[d], draft[d], _UNDERWATER_TOKENS[d])
        for d in range(DEPTH)
    ]
    path = [True] * accept_length + ([False] if accept_length < DEPTH else [])

    block_rng = _CountingRNG(7)
    current_rng = _CountingRNG(7)
    block = _accept_loop(
        draft,
        target,
        _UNDERWATER_TOKENS,
        _coins_for(path, verifier.accept_probability),
        verifier=verifier,
        rng=block_rng,
    )
    current = _accept_loop(
        draft,
        target,
        _UNDERWATER_TOKENS,
        _coins_for(path, shipped),
        verifier=None,
        rng=current_rng,
    )
    assert block.key() == current.key()
    # The accept coins are drawn by the caller in the real loop, so what is
    # counted here is the correction/bonus traffic -- the only draws the law
    # itself makes. `draws_used` accounts for both and must agree.
    assert block_rng.choices == current_rng.choices
    assert block_rng.draws == current_rng.draws
    assert block.draws_used == current.draws_used
    assert block.draws_used == accept_length + (1 if accept_length < DEPTH else 0) + 1


def test_building_the_ladder_consumes_no_randomness():
    draft, target = _rows(_UNDERWATER_DRAFT, _UNDERWATER_TARGET)
    rng = _CountingRNG(9)
    verifier = _verifier(draft, target, _UNDERWATER_TOKENS)
    for depth in range(DEPTH):
        verifier.scaled_residual(depth)
    assert rng.draws == 0
    # And the module cannot draw even if it wanted to: it takes no rng.
    assert "rng" not in inspect.signature(BlockVerifier.__init__).parameters
    assert "rng" not in inspect.signature(build_verifier).parameters


# ---------------------------------------------------------------------------
# (d) The K20 log round-trips through the offline replay.
# ---------------------------------------------------------------------------


def _random_window(source):
    supports = [source.choice(400, size=8, replace=False) for _ in range(DEPTH + 1)]

    def row(support, sharpness):
        weights = source.random(support.size) ** sharpness
        weights /= weights.sum()
        order = np.argsort(support)
        return SparseDistribution(support[order], weights[order], 50000)

    target = [row(supports[r], 2.0) for r in range(DEPTH + 1)]
    draft = [row(supports[d], 1.0) for d in range(DEPTH)]
    tokens = [int(source.choice(q.token_ids, p=q.probs)) for q in draft]
    return draft, target, tokens


