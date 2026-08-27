# Single stream: what continuous batching costs a lone request

Measured because the question "what did you measure for single stream
regression when batching is idle" had no answer in this repo. Every arm in
`dense-mtp-batch-results.md`, n = 1 included, runs through
`generate_dense_mtp_batch`. None of them compares against the path a lone
request takes when batching is off.

**In plain terms:** if you are the only person using the server, does this
change make you slower? Answer: no, you get faster, but not for the reason the
name suggests. The batching itself does almost nothing at one request. The
speedup comes from two other optimisations that live in the same driver.

## What is being compared

`DenseMTPBatchGenerationService._run_sealed` picks the path:

```python
if len(jobs) == 1 and not refill and not self.continuous:
    self._run_solo(jobs[0])      # -> generate_mtpk
else:
    self._run_cohort(jobs, ...)  # -> generate_dense_mtp_batch
```

With `continuous=True` (the shipping default) a lone request takes the second
branch. So the comparison is `generate_mtpk` against
`generate_dense_mtp_batch` at width 1. Both are library functions and are
called directly here, with no server in the loop.

## Result, greedy

Decode-only tok/s, median of 3, one warmup discarded. Every arm emitted exactly
512 tokens with `finish=length`, verified rather than assumed.

| arm | loop | capture backend | decode tok/s | vs solo |
|---|---|---|---|---|
| solo | n/a | n/a | 51.84 | baseline |
| cohort of 1 | serial | stock | 52.61 | +1.5% |
| cohort of 1 | pipelined | stock | 58.38 | +12.6% |
| cohort of 1 | serial | linear-gdn-from-conv-stream | 59.76 | +15.3% |
| cohort of 1 | pipelined | linear-gdn-from-conv-stream | 63.23 | **+22.0%** |

The last row is what ships. **A lone request is 22% faster through the batch
driver, not slower.**

## Where the 22% comes from

Reading down the table, with each optimisation isolated:

| contribution | effect |
|---|---|
| the cohort loop itself | **+1.5%** |
| pipelining (`loop_mode`) | +11.0% |
| capture backend | +13.6% |

The cohort loop on its own is break-even. That is the expected result: at width
1 there is nothing to batch. The gain is pipelining plus the capture backend,
neither of which is about batching, both of which exist only in this driver.

Pipelining is the driver building cycle N+1's graph before cycle N's host sync
completes, so the CPU prepares the next round while the GPU finishes the
current one. The solo loop does those strictly in order and leaves the GPU
idle while the CPU works.

The two do not add: 52.61 x 1.11 x 1.136 would be 66.3, and the measured
combination is 63.23. They overlap.

### Consequence for a comment in the code

`_run_sealed` says a one-row continuous cohort "is slower than the tuned solo
loop for a request that really is alone". As a claim about the *loop* that is
still roughly right, +1.5% is break-even. As a claim about what a lone request
experiences it is wrong by 22 points, because the driver carries two
optimisations the solo path does not. The comment should say which it means.

## Result, temperature 1.0

Six matched seeds, paired per seed. Sampled decoding draws a different token
sequence per seed, draft acceptance depends on the sequence drawn, and the two
engines do not draw identically from the same seed, so the pairing is
imperfect and the spread is wide.

| seed | solo tok/s | cohort tok/s | delta |
|---|---|---|---|
| 1234 | 53.57 | 64.36 | +20.13% |
| 1235 | 59.42 | 56.43 | **-5.03%** |
| 1236 | 52.35 | 58.42 | +11.59% |
| 1237 | 49.46 | 58.42 | +18.11% |
| 1238 | 50.29 | 56.42 | +12.18% |
| 1239 | 53.03 | 57.71 | +8.82% |

Median **+11.9%**, per-seed range -5.0% to +20.1%. All arms emitted 512 tokens.

**One seed is negative.** Six seeds is not enough to state a sampled headline
with the confidence the greedy number carries. Reported as measured rather than
averaged into something tidier. The greedy figure is the one to quote.

## Output identity

Separately, and already recorded: stock upstream `bd44215` with
`--scheduler-mode serial` against the fork, eight prompts run one at a time
with seeds fixed, **8 of 8 byte-identical**. The solo path is untouched by the
serving work.

## Scope

These call the two generate functions directly. They do not include the seal
wait a real request pays before dispatch (`batch_wait_s`, default 20 ms), which
adds to time to first token for a lone request. Decode then runs at the rates
above.

## Reproduce

```
Model     Qwen3.8-27B-MTPLX-Optimized-Speed (4-bit)
Machine   Apple M3 Ultra, 256 GB
Branch    this PR branch, measured at the tip of the serving commit
Prompt    mtplx/benchmarks/prompts/long_code.jsonl
          row long_warm_code_continuation, 172 tokens
Tokens    512, stop tokens disabled so both arms run to fixed length
MTP       depth 3
Solo      generate_mtpk, the server's own kwargs: mtp_hidden_variant=post_norm,
          mtp_cache_policy=persistent, mtp_history_policy=committed,
          draft_core=stock
Cohort    generate_dense_mtp_batch, one prompt, the bench runner's kwargs:
          head_history=committed, ragged_attention=True, draft_core=eager
```

### Confounds checked

The arms differ in more than one setting, because the two paths do not accept
the same options. Each difference was measured rather than assumed away.

* **Token count.** Both emit exactly 512, `finish=length`. Rates computed on
  elapsed time are therefore comparable. The first version of this benchmark
  divided an assumed 512 by elapsed without checking.
* **Prefill.** Decode-only says +22.1%, wall clock says +21.3%. They agree, so
  prefill is not producing the gap. Figures above are decode-only, matching
  what the existing ladder reports.
* **Draft core.** The option sets are disjoint: solo takes
  `stock`/`device-d2`/`device`, the driver takes `eager`/`compiled`. They
  cannot be equalised. Bounded instead: swapping cores within solo moves it
  4.4% (49.67 to 51.87) and the two driver cores agree within 0.08% (63.13,
  63.08). Both far smaller than the 22% gap, and best-solo against best-cohort
  is still +21.7%.
* **Capture backend.** Not negligible, so it is broken out above rather than
  folded into a "batching" number.
