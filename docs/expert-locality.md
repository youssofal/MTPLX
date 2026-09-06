# Expert locality telemetry

Expert locality telemetry samples the router-selected expert indexes that an
MoE model already produced. It reports per-layer and per-lane working-set
coverage, reuse distance, consecutive overlap, and simulated LRU hit rates.
It never changes routing or expert placement.

The feature is off by default. Enable it before server startup:

```bash
export MTPLX_EXPERT_LOCALITY=1
export MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY=16
```

The startup installer wraps discovered `switch_mlp` modules on the model owner
thread. When the feature is off, no wrapper is installed. The default enabled
sampling interval is 16 router calls. Set
`MTPLX_EXPERT_LOCALITY_MAX_EVENTS` to bound retained observations and
`MTPLX_EXPERT_LOCALITY_CACHE_SIZES` to change the simulated LRU capacities.

Call `mtplx.expert_locality.expert_locality_metrics()` to read a JSON-safe
snapshot. Lanes distinguish prefill, decode, MTP verify, MTP repair, and
postcommit work when the runtime attention context exposes them.

## Measurement

Run the committed no-model measurement with:

```bash
python3 scripts/bench_expert_locality.py \
  --iterations 50000 \
  --repeats 7 \
  --sample-every 16
```

The recorded Apple M5 Max receipt reports 1.130 microseconds of median overhead
per wrapped switch call at a 1-in-16 sampling interval, with output parity and
the expected eight-expert 90 percent working set. Full sampling measured 3.405
microseconds of median overhead per call. See
`docs/perf/receipts/expert-locality-no-model-20260826.json` for the raw values,
environment, and scope.

This is a CPU-only Python-list measurement of the instrumentation boundary. It
does not claim model throughput, Metal timing, or MLX materialization cost.
