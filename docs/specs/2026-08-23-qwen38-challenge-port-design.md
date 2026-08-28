# Qwen 3.8 Challenge Optimization Port Design

Status: approved; implementation inventory started 2026-08-23

## Goal

Port the transferable, still-relevant optimizations from the 82 accepted
Qwen 3.8 MTP challenge submissions into MTPLX on a branch based on current
upstream `main`. Keep the work in one separate pull request. Do not copy
challenge-specific infrastructure, resamples, no-op changes, mechanisms that
were later removed or superseded, or behavior MTPLX already implements.

The source-selection threshold is a relative official-score improvement greater
than 0.10 percent:

```text
relative improvement = (submission score / previous promoted score - 1) * 100
```

This threshold selects source candidates. It does not waive MTPLX exactness or
performance gates. A source winner that is neutral or slower in a matched MTPLX
A/B is rejected from the final PR.

## Authorities and pins

- Yukon leaderboard: `https://www.yukon.org/mlxfast`, 82 accepted/promoted rows
  as rendered on 2026-08-23.
- Challenge source: `Layr-Labs/qwen-3.8-mtp-challenge@eb5eadc7a165047d4321ce883b9ff30894d8bd19`.
- MTPLX base: `youssofal/MTPLX@bd4421567f9e16ce957c6ef97708b072dcd73937`
  (`v2.9.1`).
- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-challenge-port`.
- Branch: `perf/qwen38-challenge-port`.

Yukon is the score and ordering authority. GitHub PR notes and the local
challenge commit graph are the mechanism and provenance authorities. Current
MTPLX source and tests are the authority for deciding whether a mechanism is
already present.

## Decision labels

- **PORT**: survives in the final challenge stack, clears the source threshold,
  is transferable, and is not already implemented in the Qwen 3.8 MTPLX lane.
- **ALREADY**: MTPLX already implements the behavior; add coverage only if the
  existing proof is incomplete.
- **DEPENDENCY**: below-threshold or superseded as a standalone submission, but
  required to implement a qualifying final mechanism. It receives no separate
  performance claim.
- **WEAK**: at or below 0.10 percent, a resample, or a no-op/noise-only result.
- **SUPERSEDED**: replaced or reversed by a later accepted mechanism. Only the
  final descendant is considered.
- **CHALLENGE-ONLY**: benchmark orchestration, Swift worker, declared-head
  staging, Metal command-buffer policy, or another mechanism without an MTPLX
  runtime analogue.

## Consolidated port set

The implementation is organized around final mechanisms, not 54 historical
patches. Historical submissions that converge on one final mechanism share one
candidate and one MTPLX A/B gate.

### 1. Exact target readout — closed as non-transferable

Do not port the hierarchical two-stage top-2 reduction. Challenge PR #29 won
because the trusted worker was required to materialize target top-2 values for
every row; PR #37 then reused top-1 and removed a second vocabulary-wide
argmax. MTPLX's production verifier has no target top-2 ledger consumer and
already launches only the target argmax. Adding the two-stage reducer would add
two Metal dispatches without removing duplicate work. MTPLX's existing
experimental dense-logit top-k kernel is likewise not routed on this greedy
target path. Rows 5 and 6 therefore remain qualifying source results but are
skipped as challenge-only work-binding optimizations.

### 2. Compact proposal head

Port the final proposal-only compact-vocabulary stack: coarse affine-2 cluster
selection, exact affine-4 reranking, the surviving E87 two-dispatch selector,
the 0.15 derived-cluster probe fraction, the 32-value-per-lane affine-2 QMV,
and lazy construction that does not compile the unused fallback probe sorter.
The proposal approximation may change acceptance but must never change target
token correctness.

Sources: submissions 10, 42, 47, 67, 69, 71, 79, and 82. Submission 46 is a
dependency for the final top-32 candidate contract but has no standalone claim.

### 3. Final wide affine-4 verification QMV — evaluated and rejected

The final cross-row affine-4 family was ported as one candidate rather than as
each intermediate
kernel: direct-nibble packing for the retained widths, reusable activation
chunk-sum tables where they amortize, width-specialized ownership for M=2..9,
and exact active-group launch geometry. The candidate must preserve real Qwen
3.8 shapes, BF16 activation order, packing, group size, and output layout.

The corrected group-aware C7 route preserved exact output and schedule and
engaged the intended kernels, but regressed matched 16K wall time by 0.7253%
on top of C3+C6. It is therefore absent from the production route; the raw
rejection receipt remains as evidence.

Sources: submissions 19, 34, 36, 39, 40, 41, 70, 78, and 80. Submissions 9,
15, and 44 are superseded ancestors.

### 4. Qwen projection, attention, GDN, and normalization fusions

Port only final retained fusions that MTPLX does not already bind for the Qwen
3.8 27B lane:

- fused affine-4 Q/K/V projection;
- fused GDN input projections and compiled GDN satellite expressions;
- one-forward SDPA width bridging;
- exact full-attention Q/K RMSNorm plus partial RoPE;
- boundary-fused residual/RMSNorm with attention-copy elimination;
- dual pre-fc RMSNorm, including the retained concatenation-free output form;
- schedule-neutral verify reuse and the compiled attention output gate.

Sources: submissions 3, 13, 16, 18, 21, 30, 45, 60, and 61.

### 5. MTP-head cache append and early submission

MTPLX already skipped the vocabulary head in `mtp_update_cache`, so that part
of the source mechanism is `ALREADY`. It still computed Q/gate, attention,
output projection, MoE, and final norm for committed-history rows whose outputs
the caller discards. The retained route performs only embedding/hidden
normalization, fusion, input normalization, K/V projection, RoPE, and the cache
write. Unlike the Swift worker, MTPLX appends history separately from the next
proposal step, so every row in this call is dead-output K/V-only work rather
than all-but-the-final row.

The source's early first-draft submission is also already implicit in MTPLX:
each proposal ID must be sampled on the host before the next chained MTP call
can be built, which submits the first step at that boundary. No extra async
dispatch is added. The optimized cache route remains limited to the validated
one-layer Qwen 3.8 artifact and preserves history, rollback, session-bank, and
prefix-reuse contracts.

Sources: the not-already-present portions of submissions 11 and 20.

### 6. Precision-island head artifact

Evaluate the final selected-BF16 Q/K/V precision islands together with the
compact Q2/Q4 proposal head. This is an artifact candidate, not a runtime
requantization. It may be included only with a reproducible manifest, digest,
license/provenance record, and unchanged target-token parity.

Sources: submissions 33 and 36. Submissions 37 and 64 are duplicate/restacked
or below-threshold variants.

### 7. Final depth and warmup policy

MTPLX already has committed history and cost-model/expected-value depth policy.
Do not add a second controller. Evaluate only the final challenge calibration
that survived to the tip: legal zero-draft escape, the retained depth ceiling
and segmented cap, per-position acceptance EMA, target-boundary prices, and the
specific `callWithHiddenAndNormed` warm shape. Bind selected values at model
construction and expose them in the route fingerprint.

Sources: the remaining portions of submissions 11, 38, and 59. Earlier streak
gate and width-floor changes are superseded.

## Existing MTPLX behavior retained unchanged

- committed MTP-head history and prompt streaming;
- capture/commit verification, per-row GDN state capture, rollback, and repair;
- cost-model and expected-value adaptive depth policy types;
- Q4/group-64 draft-only LM-head support;
- NAX/verify-kernel and compiled-verify routes;
- Qwen 3.5/3.6/3.8 MTP model injection and session/prefix cache identity;
- Turbo profile packed-GQA attention and bounded warmup ladder.

New candidates must compose with these mechanisms. A challenge mechanism is
not allowed to silently replace a stronger MTPLX route simply because it has a
similar name.

## Complete 82-submission disposition ledger

The delta column is computed from consecutive official Yukon scores. Row 1 is
the challenge bootstrap and has no preceding Yukon row.

| # | PR | Delta | Mechanism | Disposition |
| ---: | ---: | ---: | --- | --- |
| 1 | 2 | n/a | Initial benchmark bootstrap | CHALLENGE-ONLY |
| 2 | 13 | 23.2995% | Infrastructure resubmission | CHALLENGE-ONLY |
| 3 | 18 | 1.5662% | Fused affine-4 Q/K/V | PORT, fusion set |
| 4 | 24 | 1.2868% | K=1 fused verify/lazy boundary | ALREADY in generalized MTPLX verify |
| 5 | 29 | 6.7162% | Hierarchical exact top-2 | CHALLENGE-ONLY; MTPLX has no target top-2 ledger |
| 6 | 37 | 1.7673% | Reuse top-2 top-1 as argmax | CHALLENGE-ONLY; MTPLX has no duplicate target argmax |
| 7 | 38 | 20.4283% | Committed MTP-head history | ALREADY |
| 8 | 41 | 17.9183% | History/single-sync/fusions/checkpoints/depth bundle | ALREADY for history, sync, checkpoints; fusions tracked separately |
| 9 | 55 | 6.8764% | Paired affine QMV | SUPERSEDED by final C7 candidate; C7 rejected locally |
| 10 | 59 | 5.3467% | Warmed compact vocabulary | PORT, compact-head set |
| 11 | 63 | 5.7425% | Cost model, seed warm, early flush | ALREADY for controller, warm framework, and proposal submission boundary; final calibration tracked separately |
| 12 | 70 | 0.8266% | Lazy exact prefix replay | ALREADY |
| 13 | 71 | 2.0944% | Fused GDN input projections | PORT, fusion set |
| 14 | 77 | 0.9839% | Lazy exact prefix replay restack | SUPERSEDED/ALREADY |
| 15 | 95 | 3.7597% | Width-6..9 chunking | SUPERSEDED by final C7 candidate; C7 rejected locally |
| 16 | 103 | 0.5808% | Compiled GDN satellite expressions | PORT, fusion set |
| 17 | 126 | 7.5460% | Declared Q4/group-64 head | ALREADY; later superseded by compact Q2/Q4 artifact |
| 18 | 135 | 0.4535% | One-forward exact SDPA width bridge | PORT, fusion set |
| 19 | 160 | 2.5391% | Wider cross-row QMV and fused draft readout | Target-QMV part rejected through C7; proposal readout retained through C8 |
| 20 | 180 | 0.9180% | K/V-only committed-history flush | RETAINED at original request context >=16K; exact cache and +2.3680% clean conditioned 16K ABBA win |
| 21 | 186 | 1.5222% | Fused Q/K RMSNorm and partial RoPE | PORT, fusion set |
| 22 | 194 | 0.0911% | Packed GDN prework mixer | WEAK, below threshold |
| 23 | 215 | 0.2964% | Two-level compact selector | SUPERSEDED by E87 |
| 24 | 234 | 0.9658% | Confidence-aware scheduling refinements | ALREADY/superseded by current cost model |
| 25 | 270 | 0.5421% | Open deep cap one round sooner | SUPERSEDED by later policy |
| 26 | 276 | 0.1799% | Six-way composite/restack | SUPERSEDED or ALREADY; no isolated port claim |
| 27 | 301 | 0.0348% | Recurrent replay scheduling | WEAK, below threshold |
| 28 | 304 | 0.2525% | Q4 head requantization | ALREADY and superseded |
| 29 | 333 | 0.0229% | All-width compiled GDN/packed beta | WEAK, below threshold |
| 30 | 350 | 0.4202% | Verify reuse and compiled output gate | PORT, fusion set |
| 31 | 364 | 0.0533% | Row-cost adaptive depth | WEAK and ALREADY |
| 32 | 365 | 0.1764% | Streak-gate restore | SUPERSEDED by final policy |
| 33 | 401 | 1.7181% | Q4 head with BF16 Q/K/V precision islands | PORT, artifact set |
| 34 | 405 | 0.6815% | Direct-nibble QMV M=6,9 | SUPERSEDED by rejected C7 |
| 35 | 417 | 0.0537% | Warm `AndNormed` | WEAK, below threshold |
| 36 | 423 | 1.6826% | Precision islands plus direct-nibble M=8 | PORT through C8 islands; target-QMV part rejected through C7 |
| 37 | 428 | 0.1477% | Precision-island restack | SUPERSEDED duplicate |
| 38 | 430 | 0.6244% | Restore `callWithHiddenAndNormed` warm | PORT, final warm set |
| 39 | 437 | 0.3030% | Affine-4 QMV M=4 ownership | SUPERSEDED by rejected C7 |
| 40 | 438 | 0.6643% | Direct-nibble QMV M=7 | SUPERSEDED by rejected C7 |
| 41 | 450 | 0.4931% | Direct-nibble QMV M=3,4,5,7 | SUPERSEDED by rejected C7 |
| 42 | 472 | 1.4130% | Affine-2 shortlist plus affine-4 rerank | PORT, compact-head set |
| 43 | 493 | 0.0824% | M=1 coarse-readout QMV | WEAK, below threshold |
| 44 | 503 | 0.0718% | M=8 4+4 QMV combine | WEAK and superseded |
| 45 | 505 | 0.4408% | Boundary residual/RMSNorm fusion | PORT, fusion set |
| 46 | 509 | 0.0035% | Exact top-32 shortlist | DEPENDENCY only; below-threshold standalone result |
| 47 | 530 | 0.5687% | 32-value/lane affine-2 QMV | PORT, compact-head set |
| 48 | 543 | 0.1230% | Remove residency/command-buffer mechanisms | SUPERSEDED; later reversed |
| 49 | 553 | 0.0595% | Residual/RMSNorm restack | WEAK duplicate |
| 50 | 572 | 0.2736% | Restore wired residency/command buffers | CHALLENGE-ONLY |
| 51 | 580 | 0.0081% | Residency/command-buffer restack | WEAK and CHALLENGE-ONLY |
| 52 | 597 | 0.0283% | Verify-concat JIT warm | WEAK, below threshold |
| 53 | 600 | 0.1577% | Force 512 MiB command-buffer profile | CHALLENGE-ONLY |
| 54 | 693 | 0.0173% | Later-window SDPA compile | WEAK, below threshold |
| 55 | 763 | 0.0623% | Variance resample | WEAK/no-op |
| 56 | 788 | 0.0155% | SDPA warm restack | WEAK, below threshold |
| 57 | 833 | 0.0694% | Frontier resample | WEAK/no-op |
| 58 | 834 | 0.0394% | Width cap/floor change | WEAK and superseded |
| 59 | 843 | 1.4217% | Final deeper floor under cap 7 | PORT only as final-policy candidate |
| 60 | 846 | 0.2224% | Dual pre-fc RMSNorm | PORT, fusion set |
| 61 | 866 | 0.2836% | Concatenation-free dual RMSNorm output | PORT, fusion set |
| 62 | 895 | 0.0215% | SDPA warm resample | WEAK/no-op |
| 63 | 911 | 0.1698% | Controlled resample | WEAK/no-op |
| 64 | 914 | 0.0799% | Precision-island dead-work removal | WEAK standalone result |
| 65 | 958 | 0.0089% | E87 arm C/probe compaction | WEAK, below threshold |
| 66 | 965 | 0.3080% | Variance resample | WEAK/no-op |
| 67 | 968 | 0.3523% | Selected affine-4 rerank | PORT, compact-head set |
| 68 | 1028 | 0.0540% | Centroid-selector redraw | WEAK/no-op |
| 69 | 1031 | 0.2133% | E87 two-dispatch shortlist | PORT, compact-head set |
| 70 | 1063 | 3.9125% | Hoisted activation chunk sums | Evaluated through C7; rejected locally |
| 71 | 1066 | 0.7439% | E121 affine-2 cluster QMV | PORT, compact-head set |
| 72 | 1071 | 0.0524% | Restore qL2/3 SDPA warms | WEAK, below threshold |
| 73 | 1089 | 0.0227% | Three-mechanism restore | WEAK/restack |
| 74 | 1095 | 0.0455% | Width-specialized single-pass accumulator | WEAK, below threshold |
| 75 | 1105 | 0.0686% | qL1..5 warm restore | WEAK, below threshold |
| 76 | 1107 | 0.0518% | New-crown resample | WEAK/no-op |
| 77 | 1117 | 0.0503% | Probe fraction 0.15 | DEPENDENCY of qualifying row 79 |
| 78 | 1123 | 4.3907% | Tight active-group QMV launch | Evaluated through C7; rejected locally |
| 79 | 1130 | 0.2444% | Restore 0.15 probe fraction | PORT, compact-head set |
| 80 | 1139 | 0.3477% | Tight launch for width M=2 | C7 endpoint; rejected at -0.7253% locally |
| 81 | 1150 | 0.0597% | Flush-fold warm M=3..9 | WEAK, below threshold |
| 82 | 1153 | 0.3733% | Skip unused probe-sort JIT | PORT as compact-head construction invariant |

## Correctness boundaries

1. The target model remains the sole authority for emitted greedy tokens.
2. Target-side kernels must match the selected MTPLX control at the declared
   exactness level. A kernel may not silently fall back on a contract miss.
3. Proposal-only approximations may change proposal IDs and acceptance, but
   target output and stop handling must remain identical.
4. Committed MTP history receives only target-authoritative committed rows.
   Rejected speculative rows never persist.
5. Kernel construction validates exact Qwen 3.8 shape, dtype, bit packing,
   group size, ownership, and output layout before binding a route.
6. Candidate identity enters session/prefix-cache fingerprints. A session may
   not restore across incompatible head, policy, or kernel routes.

## Performance gates

- Acquire `/tmp/mtplx-gpu-exclusive.lock` before any real-model GPU run.
- Use the exact Optimized-Speed artifact with the Turbo Q4/group-64 draft head,
  depth 3, temperature 1.0, top-p 0.95, top-k 20, and seed 42.
- Build exactly 16,384 Python input tokens from `mtplx/generation.py`, keeping
  the intact `python_modules_long.jsonl` instruction at the tail, and require
  all 1,024 output tokens.
- Give each route one full-output conditioning generation, then run exactly
  four timed ABBA arms. Record prefill tok/s, decode tok/s, peak memory, wall
  time, route identity, counters, output hash, and acceptance schedule.
- Measure candidates chronologically and cumulatively. Candidate N is compared
  with all earlier retained winners; a passing >0.05% win becomes the next
  control. Never add or multiply isolated percentages.
- Exact output/schedule parity passes immediately. Deterministic full-length
  output with a tie-breaking token or schedule shift is recorded for numerical
  audit and is not rejected from hashes alone.
- Keep only independently measured wins. Source leaderboard improvement is
  evidence to test, not proof of an MTPLX win.

## Licensing and attribution

The challenge repository is MIT licensed while MTPLX is Apache-2.0. Any adapted
code must retain the applicable MIT notice and identify the challenge source in
`NOTICE` and adjacent source comments. Weight artifacts require their own
manifest, digest, source revision, and license/provenance record.

## Pull-request boundary

The final PR contains only retained MTPLX winners, their tests, benchmark
receipts, route/provenance documentation, and attribution. Rejected candidates
remain documented in the ledger and local benchmark artifacts but are removed
from production code. The branch is pushed to the user's MTPLX fork and opened
against `youssofal/MTPLX:main` only after the final verification gate.
