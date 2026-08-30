"""Repro: 27B on a 16 GB Mac at long context, decode >= 10 tok/s out to 128K.

Three serving changes, no model edits (measured on keXjos/Qwen3.8-27B-mlx-2Bit,
base M4 / 16 GB, macOS 26.x, mlx-lm 0.31.3):

    1. backbone-only chunked prefill + lm_head on the LAST position only.
       Whole-sequence forwards materialize logits for every position
       ((ctx x 248320) — a single >9.5 GB alloc near ctx=16K that breaks
       Metal's per-buffer cap, with RAM thrash well below that).
    2. 2048-token prefill chunks to bound transient activations.
    3. rotating KV window on the full-attention layers only; the GDN
       linear-attention layers keep constant-size global state, so step
       time is flat vs context.

Run:
    PYTHONPATH=. python benchmarks/repro_long_ctx_16gb.py --model <path-to-kexjos>

First-time model download (~8 GB):
    PYTHONPATH=. python benchmarks/repro_long_ctx_16gb.py --model auto --download-model

Expected: every swept size decodes >= 10 tok/s with peak <= 12 GB;
step time flat (~90-98 ms) from 4K to 128K. Needle-recall beyond the window
is NOT expected to work on this checkpoint (misses identically with full
unwindowed KV — see the notes branch worklog).
"""
import argparse
import sys
import time

import mlx.core as mx

MODEL_REPO = "keXjos/Qwen3.8-27B-mlx-2Bit"


def get_model_path(path: str, allow_download: bool) -> str:
    import os

    if path != "auto" and os.path.isdir(path):
        return path
    default = os.environ.get(
        "MODELS_DIR", "/Volumes/medusa-1tb/models"
    ) + "/qwen38-27b-2bit/kexjos"
    if os.path.isdir(default):
        return default
    if not allow_download:
        sys.exit(
            f"model not found ({path=}, {default=}). Pass --model <dir> or "
            "add --download-model to fetch ~8 GB from the Hub."
        )
    from huggingface_hub import snapshot_download

    print(f"downloading {MODEL_REPO} (~8 GB)...", flush=True)
    return snapshot_download(MODEL_REPO)


def build_cache(model, window):
    from mlx_lm.models.cache import ArraysCache, RotatingKVCache

    layers = model.language_model.model.layers
    return [
        ArraysCache(size=2) if layer.is_linear else RotatingKVCache(max_size=window, keep=4)
        for layer in layers
    ]


def run_target(model, tokenizer, target, *, chunk, window, gen, min_tps) -> bool:
    lm = model.language_model
    filler_ids = tokenizer.encode(
        "The theory of general relativity describes gravity as the curvature of "
        "spacetime produced by mass and energy. Light bends around massive "
        "objects, clocks run slower in stronger fields, and the universe expands. "
    )[5:]
    ids = (filler_ids * (target // len(filler_ids) + 1))[:target]

    cache = build_cache(model, window)
    mx.clear_cache()
    mx.reset_peak_memory()
    try:
        t0 = time.perf_counter()
        h = None
        for i in range(0, len(ids), chunk):
            h = lm.model(mx.array([ids[i : i + chunk]]), cache)
            mx.eval(h)
        logits = lm.lm_head(h[:, -1:, :])
        mx.eval(logits)
        pf = time.perf_counter() - t0

        steps = []
        for i in range(gen):
            t0 = time.perf_counter()
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            _ = int(nid)
            logits = model(nid[:, None], cache=cache)
            mx.eval(logits)
            steps.append(time.perf_counter() - t0)
        steps = sorted(steps)[len(steps) // 2]
        ms = steps * 1e3
        tps = 1000.0 / ms
        peak = mx.get_peak_memory() / 1e9
        ok = tps >= min_tps
        print(
            f"ctx={len(ids):6d} win={window}: prefill {len(ids)/pf:5.0f} tok/s | "
            f"decode {ms:6.1f} ms/step ({tps:5.2f} tok/s) | peak {peak:4.1f} GB "
            f"{'OK' if ok else 'BELOW TARGET'}",
            flush=True,
        )
        return ok
    except Exception as e:
        print(f"ctx={target} win={window}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
        return False
    finally:
        mx.clear_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto")
    ap.add_argument("--download-model", action="store_true")
    ap.add_argument("--targets", default="4096,32768,65536,131072")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--min-tps", type=float, default=10.0)
    args = ap.parse_args()

    from mlx_lm import load

    path = get_model_path(args.model, args.download_model)
    model, tokenizer = load(path)
    print(f"loaded {path}")

    targets = [int(t) for t in args.targets.split(",")]
    ok = True
    for tgt in targets:
        ok &= run_target(
            model, tokenizer, tgt,
            chunk=args.chunk, window=args.window, gen=args.gen, min_tps=args.min_tps,
        )
    print()
    print("LONG-CTX TARGET MET" if ok else "TARGET MISSED ON ONE OR MORE SIZES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
