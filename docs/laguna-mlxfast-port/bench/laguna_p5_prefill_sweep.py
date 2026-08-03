#!/usr/bin/env python3
"""P5 prefill size sweep: find where the MoE-combine fusion stops winning / breaks.

Loads Laguna-S-2.1 ONCE (install_from_env reference fusions), then for each context
size runs three lanes on the SAME model:
  - reference : eager LagunaModel forward + head (the shipped baseline)
  - alt[d1]   : alt_prefill_forward with D1 (the shipped decode+prefill fusion)
  - alt[d1,p5]: D1 + P5 (the prefill MoE-combine tail fusion under test)

Reports prefill tok/s per lane, the P5-vs-D1 speedup, first-token digest match
(P5 must be bit-exact -> same token), and peak GiB, at each size. The crossover is
where alt[d1,p5]/alt[d1] drops <= 1.0 (or a size errors) -> that becomes the
`prefill_max_tokens` gate. Runs under the flock via f04/f03 (qwen unloaded).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

WT = str(Path(__file__).resolve().parents[3])  # repo root
BENCH = os.environ.get(
    "MTPLX_LAGUNA_BENCH_DIR", str(Path(__file__).resolve().parent)
)
sys.path.insert(0, WT)
sys.path.insert(0, BENCH)

SIZES = [int(s) for s in os.environ.get("P5_SIZES", "1024,4096,16384,32768,65536").split(",")]
REPS = int(os.environ.get("P5_REPS", "2"))


def _first_token(logits, mx) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32).item())


def main() -> int:
    import mlx.core as mx
    from laguna_lane import build_prompts, guard_memory, resolve_model_dir

    from mtplx.laguna_alt_step import AltConfig, alt_prefill_forward
    from mtplx.models import laguna_fused
    from mtplx.runtime import load as runtime_load

    print(f"=== P5 prefill sweep | sizes={SIZES} reps={REPS} ===", flush=True)
    model_dir = resolve_model_dir()
    runtime = runtime_load(model_dir, mtp=False)
    mx.eval(runtime.model.parameters())
    report = laguna_fused.install_from_env(runtime.model)
    print(f"install_from_env: {report}", flush=True)
    model = runtime.model

    # D1-free P5 (the shippable prefill candidate) and D1+P5 (for comparison).
    cfg_p5 = AltConfig(p5_prefill_moe_tail=True)
    cfg_d1_p5 = AltConfig(d1_residual_router=True, p5_prefill_moe_tail=True)

    def run_reference(prompts, ctx):
        cache = runtime.make_cache()
        logits = runtime.model(prompts, cache=cache, logits_keep=1)
        mx.eval(logits); mx.synchronize(); del cache; gc.collect(); mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for _ in range(REPS):
            cache = runtime.make_cache()
            logits = runtime.model(prompts, cache=cache, logits_keep=1)
            mx.eval(logits); del cache
        mx.synchronize()
        dt = (time.perf_counter() - t0) / REPS
        return ctx / dt, _first_token(logits, mx), int(mx.get_peak_memory())

    def run_alt(prompts, ctx, config):
        def fwd():
            cache = runtime.make_cache()
            hidden = alt_prefill_forward(model, prompts, cache, config=config)
            return model.lm_head(hidden[:, -1:, :])
        logits = fwd(); mx.eval(logits); mx.synchronize()
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        t0 = time.perf_counter()
        for _ in range(REPS):
            logits = fwd(); mx.eval(logits)
        mx.synchronize()
        dt = (time.perf_counter() - t0) / REPS
        return ctx / dt, _first_token(logits, mx), int(mx.get_peak_memory())

    rows = []
    for ctx in SIZES:
        guard = guard_memory(1, ctx, 0)
        if guard.get("refused"):
            print(f"[ctx={ctx}] REFUSED by guard_memory: {guard}", flush=True)
            rows.append((ctx, None, None, None, None, None, "refused"))
            break
        try:
            prompts = build_prompts(runtime.tokenizer, 1, ctx)
            r_tps, r_tok, r_peak = run_reference(prompts, ctx)
            p5_tps, p5_tok, p5_peak = run_alt(prompts, ctx, cfg_p5)      # D1-free
            d1p5_tps, d1p5_tok, _ = run_alt(prompts, ctx, cfg_d1_p5)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[ctx={ctx}] ERROR {exc!r}", flush=True)
            rows.append((ctx, None, None, None, None, None, f"error:{exc!r}"))
            break
        digest_ok = (p5_tok == r_tok and d1p5_tok == r_tok)
        p5_effect = p5_tps / r_tps if r_tps else 0.0   # D1-free P5 vs reference
        rows.append((ctx, r_tps, p5_tps, d1p5_tps, p5_effect, digest_ok, p5_peak))
        print(
            f"[ctx={ctx}] ref={r_tps:.1f} p5={p5_tps:.1f} d1+p5={d1p5_tps:.1f} tok/s | "
            f"P5-vs-REF={p5_effect:.3f}x | digest={'MATCH' if digest_ok else 'MISMATCH!!'} "
            f"(r={r_tok} p5={p5_tok} d1p5={d1p5_tok}) | peak={p5_peak/1024**3:.1f}GiB",
            flush=True,
        )
        gc.collect(); mx.clear_cache()

    print("\n=== SWEEP SUMMARY (P5 prefill MoE-combine, D1-free vs reference) ===", flush=True)
    print(f"{'ctx':>7} {'ref':>9} {'p5':>9} {'d1+p5':>9} {'P5/ref':>7} {'digest':>8} {'peakGiB':>8}", flush=True)
    for ctx, r, p5, d1p5, eff, dg, peak in rows:
        if r is None:
            print(f"{ctx:>7} {'--':>9} {'--':>9} {'--':>9} {'--':>7} {str(peak):>8}", flush=True)
            continue
        print(f"{ctx:>7} {r:>9.1f} {p5:>9.1f} {d1p5:>9.1f} {eff:>7.3f} "
              f"{'MATCH' if dg else 'MISS!!':>8} {peak/1024**3:>8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
