# Speculative-cascade acceptance for Qwen3.8 Flash-Next

Speculative-cascade acceptance is a second opt-in, lossy decode-verify rule
beside typical acceptance. Per draft position it decides whether the draft token
is good enough to keep, or whether to defer to the exact target law. It is OFF by
default, mutually exclusive with typical acceptance, and NOT distribution-exact.

Citation: Narasimhan, Jitkrittum, Rawat, Kim, Gupta, Menon, and Kumar, "Faster Cascades via
Speculative Decoding," arXiv:2405.19261 v2 (2024). This implements the
r-hat_OPT deferral rule of Section 4.3, Equation (10): the plug-in ESTIMATOR of
the optimal speculative-cascade deferral rule (Lemma 4, Equation (9)), replacing
that rule's ground-truth expected 0-1 losses with one minus each model's max
probability. It is not the optimal rule and not an oracle. The separate Diff rule
(Equation (5), max q < max p - alpha, no total-variation term) is the
SEQUENTIAL-cascade oracle and is not implemented here. Executed with the
speculative decoding of Algorithm 4.

Terms:

- p: the target distribution at a position (the truncated top-p / top-k target
  row already materialized for the exact rule).
- q: the draft distribution at a position (the native MTP head's scored rows).
- D_TV(p, q) = sum_v max(0, p(v) - q(v)) over the scored top-k support.
- defer (r = 1): use the large / target model here; accept (r = 0): keep the
  small / draft model's token.
- alpha: the deferral cost, the single operator knob.

## Problem

The exact speculative law commits the longest draft prefix the target would have
produced under the same coins, and no more. Even when the draft head closely
tracks the target it still spends a rejection whenever the coin `min(1, p/q)`
falls, which caps tokens per cycle. Typical acceptance loosens this with a
per-row typicality floor, but it is blind to how well the draft itself matched
the target: it can reject a token the draft was confidently and correctly
proposing, and it resamples from the target row rather than the residual.

## Change

A speculative-cascade verify rule that, at each draft position with target p and
draft q, applies Equation (10):

    defer (r = 1)  <=>  max_v q(v) < max_v p(v) - alpha * D_TV(p, q)

If it does NOT defer, the draft is "good enough" (the speculative-cascade target
is pi = q, so Algorithm 4's accept probability min(1, pi/q) = 1): accept the
draft token with no coin. If it DOES defer, pi = p and the code runs the exact
speculative law unchanged, accept with `min(1, p(x_t)/q(x_t))` and, on a coin
rejection, resample the residual `norm(max(0, p - q))`. So the deferral test uses
the total variation over the scored top-k, and the deferred path uses the token's
own target probability p(x_t); the cascade path is a strict superset of the exact
rule with a draft-accept shortcut. The decision is deterministic (it consumes no
uniform); the coin only appears on the deferred path, so with the lane off the
RNG stream and verify path are byte-identical.

What q is. On the served Turbo path the draft is the native MTP head. Its scored
rows reach the verify loop as `draft_probs[depth_index]`, a `SparseDistribution`
over the head's scored vocabulary, which for this pack is the FR-Spec
frequency-ranked subset (the same q the exact rule already uses in `min(1, p/q)`
and the residual). The rule reads q's peak `max_v q` and q's mass on the scored
top-k for D_TV from that object; it adds no draft distribution and no draft
forward. q is required, so the lane is a temperature > 0, MTP-on rule, like
typical acceptance.

The Lemma 3 note, in plain words. In Equation (10) the cost term `alpha * D_TV`
is SUBTRACTED from the target's confidence. So more disagreement between the
draft and the target LOWERS the bar for accepting the draft, which means the rule
defers LESS when they disagree, not more. The paper's reason (their Lemma 3) is
that a large disagreement makes the verification step itself expensive, so the
rule only pays that cost when the target is clearly better; it accepts a
draft that is confident, even on a token the target would not have picked. The
intuitive "reject when the draft diverges" behaviour still holds for the common
case, a diverging draft that has also lost its peak confidence, which defers; a
draft that stays confident on a wrong token is accepted. This is the paper's rule
as written; the tests pin it explicitly so a future sign change is caught.

## Effect

Higher acceptance when the draft distribution agrees with the target, including on
atypical tokens that typical acceptance would reject, at the cost of exactness
(the accepted draft token is sampled from q, not p). It is a speed / quality
dial, judged on task-quality evals like typical acceptance, never on
distribution-exactness.

## Exactness

NOT distribution-exact when on. The accepted-draft (non-defer) positions emit q's
token, which differs from the target law. The deferred positions ARE exact
(min(1, p/q) coin + residual). With the lane OFF (knob unset) the verify path is
byte-identical to the exact rule: the cascade branches are gated behind
`_cascade_active`, which is false, so neither the deterministic test nor any coin
runs.

## Files

- `mtplx/sampling.py`: `cascade_defer_decision`, `total_variation`,
  `_peak_probability`.
- `mtplx/generation.py`: `_cascade_accept_alpha` / `_cascade_accept_enabled`
  (env at use), `_assert_lossy_verify_rules_exclusive`, the two verify-loop
  decision branches (batched + lazy target), the `VerifyStats` cascade fields,
  and the `[cascade-accept]` verdict line.
- `mtplx/server/openai.py`: `--cascade-threshold`, the env stamp + fail-loud
  mutual exclusion with `--typical-threshold`, and the `/health`
  `cascade_acceptance` install report.
- `tests/test_cascade_acceptance.py`, `tests/test_cascade_threshold_cli_health_cpu.py`.

## Definitions: deferral, accept, and resample rates

Speculative cascade has no single "accept rate": the paper's quantity is the
DEFERRAL rate, and the served verdict's `accept_rate` is a different, kept-draft
quantity. Over the positions the cascade rule actually decided
(`cascade_positions`, one per drafted position in a cascade window), the three
rates are:

| quantity | formula | meaning | artifact |
| --- | --- | --- | --- |
| defer_rate | `cascade_deferred / cascade_positions` | the paper's deferral rate r: the fraction of decided positions where the rule deferred to the target (before the coin) | `[cascade-accept]` verdict line `defer_rate=`; `VerifyStats.cascade_defer_rate` |
| accept_rate | `cascade_accepted / cascade_positions` | kept-draft rate: no-defer accepts plus deferred tokens the exact coin then kept; NOT the deferral rate | `[cascade-accept]` verdict line `accept_rate=`; `VerifyStats.cascade_accepted` |
| resample_rate | `cascade_resamples / cascade_positions` | deferred tokens that lost the exact coin and were resampled from the residual | `VerifyStats.cascade_resamples` |

Counter definitions: `cascade_positions` counts every draft position the rule
decided; `cascade_deferred` counts positions where the rule deferred (`r = 1`,
incremented before the coin); `cascade_accepted` counts kept drafts (no-defer
accepts plus coin-accepted deferred tokens); `cascade_resamples` counts deferred
tokens that lost the coin. Two identities always hold:
`cascade_accepted + cascade_resamples == cascade_positions` and
`cascade_deferred == (coin-accepted deferred tokens) + cascade_resamples`, so
`accept_rate + resample_rate == 1` over decided positions, while `defer_rate` is
independent of them.

Note on the receipt: the perf harness records a separate `mtp_accept_rate` (the
fraction of drafted tokens accepted across the whole request, the throughput
lever). That is a decode-throughput measure over all MTP positions and is not
the cascade `defer_rate`; read `defer_rate` from the `[cascade-accept]` verdict
line or `VerifyStats.cascade_defer_rate`, not from `mtp_accept_rate`.

## Switch

- `--cascade-threshold ALPHA` (env `MTPLX_FABLE_CASCADE_THRESHOLD`): the deferral
  cost alpha. UNSET (or blank) = OFF = exact speculative sampling (the default).
  Any set value, including an explicit 0, turns the lane on, so the off switch is
  UNSETTING the key, not setting it to 0 (alpha = 0 still defers whenever the
  target is strictly more confident than the draft). Higher alpha widens the
  accept band, so fewer positions defer. Resolved at use, per request.
- `--cascade-rule opt|tokenv1|tokenv2|tokenv3` (env `MTPLX_FABLE_CASCADE_RULE`):
  which deferral rule `--cascade-threshold` applies. Default `tokenv3` (the
  token-specific multiplicative rule, Eq. 15). It is the default because it is
  the only rule measured at exact-level accuracy: at alpha 0.95 it holds
  HumanEval 0.9695 strict (equal to exact) while OPT loses quality at every alpha
  (0.9024 at alpha 0.0 down to 0.7073 at alpha 0.75). To use the original
  position-level peak rule, select `opt` (env `MTPLX_FABLE_CASCADE_RULE=opt` or
  `--cascade-rule opt`); `tokenv1`/`tokenv2` are the additive token-specific
  rules (Eq. 13/14). The default change does not turn the mode on: the lane is
  still OFF unless `--cascade-threshold` is set, in which case the exact
  speculative law runs unchanged. For `tokenv3`, alpha is a FRACTION of the
  target peak (`Top_alpha = {v : p(v) >= max_p * (1 - alpha)}`), so its useful
  range is alpha >= 0.9; a small alpha collapses `Top_alpha` toward the argmax
  and defers almost everything (near-exact, slow). All rules are resolved at use,
  per request.
- Mutually exclusive with `--typical-threshold` (`MTPLX_FABLE_TYPICAL_THRESHOLD`
  > 0). Setting both fails loud at serve startup (SystemExit) and in the verify
  setup (ValueError).
- Observability: `/health` -> `cascade_acceptance` reports `enabled`, `alpha`
  (= `threshold`), `rule_name` (the resolved rule, e.g. `tokenv3`), `rule` (its
  one-line definition), the divergence definition, and `conflict`. Each
  generation prints a per-request verdict line that names the rule:
  `[cascade-accept] NOT distribution-exact; rule=RULE threshold=A alpha=A
  positions=N accepted=N resamples=N accept_rate=R mean_divergence=D
  tokens_per_cycle=T accepted_by_depth=... generated=N verify_calls=N` (the same
  format as `[typical-accept]`, with the added `rule=` field).

## Recommended alpha grid for the 16K sweep

The paper varies alpha continuously as a "lenience parameter" to trace a
quality-versus-latency Pareto curve (Section 6, Figure 2); it fixes no grid, and
reports temperatures T in {0, 0.1, 0.5, 1.0} and block sizes gamma in {3, 5, 7}
(Appendix E.1). So the grid is set from Equation (10)'s structure and this
model's measured scale.

Scale on this model, from the saved #478 receipts and verdict lines at 16,384
tokens, temperature 1, depth 3:

- Exact MTP acceptance rate is 0.43 to 0.47. The expected exact speculative
  acceptance rate equals the overlap `sum_v min(p, q) = 1 - D_TV(p, q)`, so
  `D_TV ~= 0.53 to 0.57` per position on average.
- Target-row entropy on the `[typical-accept]` lines is 0.83 to 1.81 nats, so the
  target peak `max_v p` sits around 0.35 to 0.65.

The deferral boundary is `max_v p - max_v q = alpha * D_TV`, so the useful alpha
range is set by the plausible target-versus-draft peak gap (at most ~max_v p,
about 0.6) divided by D_TV (~0.55): alpha up to about 1 already reaches the point
where the cost term dominates any confidence edge, and alpha ~2 defers almost
never. The four-value grid, from most deferral (least lossy, closest to the exact
rule) to near all-accept (fastest, lossiest):

| alpha | alpha * D_TV (D_TV ~= 0.55) | expected behaviour |
| --- | --- | --- |
| 0.0 | 0.00 | defer whenever the target is strictly more confident than the draft; maximal deferral |
| 0.5 | ~0.28 | defer only when the target's peak leads the draft's by more than ~0.28; moderate |
| 1.0 | ~0.55 | defer only on a large target-confidence lead; light |
| 2.0 | ~1.10 | defer essentially never; near all-accept |

Single alpha for the HumanEval cell: **alpha = 0.5**. It is the moderate point,
accepting the draft where the draft and target peaks are close and deferring
where the target clearly leads, which is the "gain speed while holding quality"
operating point the sweep should confirm. If HumanEval drops, fall back toward
alpha = 0 (more deferral); if quality holds, push toward alpha = 1.0 for more
speed.

## Provenance and rule-diff proof (arm G)

The arm G 16K sweep and the pending HumanEval cell were measured on served code
`2eac2fee` (cascade stacked on the #478 typical head). This branch re-parents the
cascade mode onto the #475 served base `27d5ff6b`, the same base #478 is built on,
so #478 (typical) and this PR (cascade) are alternative-mode PEERS on one base
rather than a stack. This is the same provenance bridge #475 uses (docs commits
over the served base); the measurements transfer because the cascade RULE is
byte-identical on the new base.

Per-function sha256 (first 16 hex) of the cascade computation, this branch vs the
measured `2eac2fee`:

| symbol | sha256 | vs 2eac2fee |
| --- | --- | --- |
| `mtplx/sampling.py::_peak_probability` | `f3bba8b637a1ff9e` | identical |
| `mtplx/sampling.py::total_variation` | `1b960bfacfb37ce9` | identical |
| `mtplx/sampling.py::cascade_defer_decision` | `3ba053b7097fa91e` | identical |
| `mtplx/generation.py::_cascade_accept_alpha` | `b81b13739d2166b7` | identical |
| `mtplx/generation.py::_cascade_accept_enabled` | `1c5fae48d7f81e85` | identical |
| batched-target cascade verify branch | `6d2749c7d09e8cbb` | identical |
| lazy-target cascade verify body | `1ab16b485087dce2` | identical |
| `[cascade-accept]` verdict block | `78d7c3e84c84f6c1` | identical |
| `/health` `_cascade_acceptance_health_payload` (feature commit) | `7a9006f74a187d70` | identical |
| `--cascade-threshold` argparse block | (identical) | identical |

The accept/defer/coin/residual, the total-variation and peak-probability
computations, the RNG draws, and the `[cascade-accept]` telemetry are therefore
byte-identical to the measured code. Three forced glue deltas remain, none of
which changes what a cascade run computes (so arm G stands):

1. Lazy-target dispatch keyword: `elif _cascade_active:` on `2eac2fee` (it chained
   off typical's `if _typical_active:`) becomes `if _cascade_active:` here, because
   typical is dropped. The branch body is byte-identical.
2. The exact-path lazy block is re-indented one level under the new `else:` and
   `target_p_for_cache = target_p` is hoisted above the dispatch (both idempotent
   / non-behavioral; the cascade-off output is unchanged).
3. `_assert_lossy_verify_rules_exclusive` reads `MTPLX_FABLE_TYPICAL_THRESHOLD`
   from the environment directly instead of calling the (now-absent)
   `_typical_accept_threshold()`.

Mutual exclusion (option a): the guard reads the typical env name defensively. It
is INERT on this branch, retained defensively; #478 and cascade are alternative
modes and were never intended to be armed together, so nothing here arms a typical
threshold, but the guard still fails loud (ValueError in the verify setup,
SystemExit at serve start) if an operator ever exports both keys.

## Token-specific deferral rules (TokenV1 / TokenV3)

`r_OPT` (Eq. 10) decides between q and p by comparing only their peaks. Sec. 4.4
of the paper names the failure that costs us HumanEval at every alpha: the draft
token `x_t ~ q_t(.)` may not maximise `q_t`, so "even when `x_t` is of poor
quality, we may end up accepting it because `q_t` happens to be more peaked than
`p_t`." Their fix is a token-specific rule `r(x_<t, v)` that judges the specific
candidate token, with target distribution (Eq. 11)

    pi_Token(v) = q(v) * (1 - r(x_<t, v)) + p(v) * eta,
    eta = sum_{v'} r(x_<t, v') * q(v').

We implement two plug-ins behind `--cascade-rule` / `MTPLX_FABLE_CASCADE_RULE`
(`opt` default; `tokenv1`, `tokenv2`, `tokenv3`):

| rule | defer r(x_<t, v) = 1 iff | eq |
| --- | --- | --- |
| `opt` (default) | `max_v q(v) < max_v p(v) - alpha * D_TV(p,q)` (position-level) | 10 |
| `tokenv1` | `q(v) < max_v' p(v') - alpha` | 13 |
| `tokenv2` | `p(v) < max_v' p(v') - alpha` | 14 |
| `tokenv3` | `p(v) < max_v' p(v') * (1 - alpha)` | 15 |

For TokenV3 this yields the intuitive target (Sec. 4.4)

    pi_TokenV3(v) = q(v) * 1[v in Top_alpha] + p(v) * sum_{v' not in Top_alpha} q(v'),
    Top_alpha = { v : p(v) >= max_v' p(v') * (1 - alpha) }.

The `p(v)*eta` term is present for every v, so a token in `Top_alpha`
(`r = 0`) has `pi(v) = q(v) + p(v)*eta >= q(v)` and the generic speculative coin
accepts it with probability 1 (accept, no coin drawn); a deferred token
(`r = 1`) has `pi(v) = p(v)*eta` and takes the exact `min(1, pi/q)` coin plus
`norm(max(0, pi - q))` residual -- Algorithm 6 (Appendix D) is
`GenSpecSample(q, p, pi_Token)`, i.e. the shipped exact path with `pi_Token` as
the target instead of `p`. `sum_v pi_Token(v) = 1` by construction. A
confidently-wrong drafted token (`p(v)` small) is now DEFERRED even when
`max q > max p` -- exactly the case `r_OPT` accepts.

`r_OPT` stays the default and its executed code is unchanged; the token-specific
branches are added as a new `elif ... _cascade_rule != "opt" ...` above the OPT
branch at both verify sites, so with `--cascade-rule opt` (or unset) the OPT
path is byte-for-byte what it was.

### Provenance after the citation fix

The TokenV3 commit also corrects the stale citation in
`cascade_defer_decision`'s docstring (it read "Mreddy" / "ICLR 2025"; corrected
to "Narasimhan, Jitkrittum, Rawat, Kim, Gupta, Menon, Kumar, arXiv:2405.19261 v2
(2024)"). That edit is inside the sha256-hashed function body #485 cited as
byte-identical to `2eac2fee`, so the whole-function TEXT hash of
`cascade_defer_decision` necessarily changes. We therefore restate the proof two
ways -- whole-function text (docstring included) AND AST of the function body
with the docstring node stripped (its executable code):

Whole-function TEXT sha256 (first 16 hex; docstring included):

| function | 2eac2fee | d8efc3f5 | this branch |
| --- | --- | --- | --- |
| `cascade_defer_decision` | `1f5061f6caa3838d` | `1f5061f6caa3838d` | `895a81bd203accc5` (docstring changed) |
| `total_variation` | `804fe67757c30ea3` | `804fe67757c30ea3` | `804fe67757c30ea3` (same) |
| `_peak_probability` | `3eb10a6b9e83045a` | `3eb10a6b9e83045a` | `3eb10a6b9e83045a` (same) |

AST CODE sha256 (first 16 hex; docstring node removed):

| function | 2eac2fee | d8efc3f5 | this branch |
| --- | --- | --- | --- |
| `cascade_defer_decision` | `1e08468afd459849` | `1e08468afd459849` | `1e08468afd459849` (same) |
| `total_variation` | `d4832efa65b6a0e2` | `d4832efa65b6a0e2` | `d4832efa65b6a0e2` (same) |
| `_peak_probability` | `9c62b65e7dd51f9c` | `9c62b65e7dd51f9c` | `9c62b65e7dd51f9c` (same) |

The OPT rule's *code* is thus byte-identical to `2eac2fee`/`d8efc3f5`; only the
docstring text changed. The measured arm-G OPT numbers stand. (The whole-file
text hashes in the earlier table above use a different extractor and are
unaffected; the two tables here are self-consistent, produced by one
docstring-stripping AST script.)

## Measured results (16K HumanEval)

Optimized-Speed pack, 16,384-token context, HumanEval (164), temperature 1.0,
reasoning xhigh, sampled decode; decode tok/s is the fastest of the seeds.
"strict" pass@1 is the evalplus-sanitized score over all problems; "completed"
excludes output-cap truncations; "trunc" is the truncation rate.

Baselines:

- Exact speculative sampling (cascade off): HumanEval 0.9634 strict at 82.85 tok/s.
- Typical-0.09 (the #478 lane): 0.9634 strict at 104.65 tok/s.

OPT rule (`r_OPT`, Eq. 10) -- HumanEval strict pass@1 by alpha:

| alpha | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| HE strict | 0.9024 | 0.8537 | 0.7805 | 0.7073 | dnf | not run |

OPT alpha 0.25 ran at 104.78 tok/s (equal speed to typical-0.09) at 0.8537 strict.

TokenV3 rule (`r_TokenV3`, Eq. 15) -- decode tok/s by alpha:

| alpha | 0.0 | 0.25 | 0.5 | 0.75 | 0.95 |
| --- | --- | --- | --- | --- | --- |
| decode tok/s | 84.09 | 88.70 | 95.30 | 96.52 | 107.10 |

(TokenV3 alpha 0.0 sits at 84.09 tok/s, next to exact's 82.85, because
`Top_alpha` is then the argmax alone and almost every token defers; higher alpha
widens `Top_alpha` and accepts more, up to 107.10 tok/s at alpha 0.95.)

TokenV3 quality:

- alpha 0.95: HumanEval 0.9695 strict / 1.000 completed / 3.05% truncation, at
  107.10 tok/s.
- alpha 0.75: HumanEval 0.9695 strict (159/164) / 0.9876 completed / 1.83%
  truncation (mean 2,652 tokens), cascade acceptance 0.820, at 95.30 tok/s
  (+15.1% vs exact 82.85); wall 1 h 19 m.

Equal-speed comparison (~105 tok/s): TokenV3 alpha 0.95 = 0.9695 strict at
107.10 tok/s; typical-0.09 = 0.9634 strict at 104.65 tok/s; OPT alpha 0.25 =
0.8537 strict at 104.78 tok/s. Exact speculative sampling reaches 0.9634 strict
but at 82.85 tok/s.

TokenV3 held 0.9695 strict at both alpha 0.75 and alpha 0.95; OPT ran 0.9024 ->
0.7073 across alpha 0.0 -> 0.75.

### Why the peak rule loses and the token-specific rule does not (Sec. 4.4)

`r_OPT` (Eq. 10) decides between the draft `q` and the target `p` by comparing
only their maximum token probabilities. As Sec. 4.4 of the paper states, the
drafted token `x_t ~ q_t(.)` may not maximise `q_t`, so "even when `x_t` is of
poor quality, we may end up accepting it because `q_t` happens to be more peaked
than `p_t`." That is the mechanism behind the OPT grid above: as alpha rises the
accept band widens and more of these poor tokens are admitted, and strict pass@1
falls monotonically. The token-specific rules judge the drafted token `v`
itself: `r_TokenV3` defers `v` iff `p(v) < max_v' p(v') * (1 - alpha)`, i.e. iff
`v` is outside the target's top band `Top_alpha`. A confidently-wrong drafted
token has small `p(v)`, so it is deferred and re-drawn from the exact target
`pi_Token` (Eq. 11) even when `q` is more peaked than `p` -- exactly the case
`r_OPT` accepts. This is why TokenV3 holds strict pass@1 at exact's level
(0.9695 vs 0.9634) while still admitting the high-`p(v)` tokens that give the
speed-up.

