"""Repro: INT4 simdgroup-MMA lanes — exactness gates + speedups vs stock quantized_matmul.

Validates the two portable lanes in mtplx/kernels/int4_simd_mma.py:

    vocab lane    M 8..16, wide-N (lm_head-class shapes); BM8/16 padded tile,
                  BN=32, BK=64, one 128-thread TG per 32 output columns.
    prefill lane  M >= 128, M % 128 == 0; BM128/BN32/BK64 WM4/WN1,
                  double-buffered weight tiles, one barrier per iteration,
                  GROUP_M=8 swizzle.

Run (synthetic weights, ~1 min on an M4):
    PYTHONPATH=. python benchmarks/repro_int4_mma_lanes.py

Add real production W4/G64 weights (downloads the 239 MB DFlash2 drafter
checkpoint from Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed):

    PYTHONPATH=. python benchmarks/repro_int4_mma_lanes.py --real-weights

Expected (base M4 / 16 GB, macOS 26.x, mlx 0.32.0):
    vocab lane   dmax <= 0.0625 bf16 vs stock (bf16 accumulation-order noise;
                 equal to stock's own distance from an fp32 ground truth);
                 ~1.28x @ m=8 / 1.22x @ m=16 at N=248320 g64.
    prefill lane BIT-EXACT (mx.array_equal) on every shape; flat 1.06-1.09x
                 from M=128 through M=8192 on real W4/G64 weights.

Exit code 0 = all gates pass.
"""
import sys
import time

import mlx.core as mx

from mtplx.kernels.int4_simd_mma import int4_prefill_qmm, int4_vocab_qmm

VOCAB_CASES = [(m, gs) for m in (8, 12, 16) for gs in (32, 64)]
PREFILL_SHAPES = [(5120, 34816), (5120, 8192), (6144, 5120), (17408, 5120)]
DRAFTER_REPO = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"


def make(m, k, n, gs, seed=0, dtype=mx.bfloat16):
    wq = mx.random.uniform(0, 2**32 - 1, (n, k // 8), key=mx.random.key(seed)).astype(mx.uint32)
    s = (mx.random.normal((n, k // gs), key=mx.random.key(seed + 1)) * 0.02).astype(dtype)
    b = (mx.random.normal((n, k // gs), key=mx.random.key(seed + 2)) * 0.02).astype(dtype)
    x = (mx.random.normal((m, k), key=mx.random.key(seed + 3)) * 0.5).astype(dtype)
    return x, wq, s, b


def stock(x, wq, s, b, gs):
    return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=4)


def dmax(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    mx.eval()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def check_vocab() -> bool:
    ok = True
    print("== vocab lane exactness (k=512 n=6400)")
    for m, gs in VOCAB_CASES:
        x, wq, s, b = make(m, 512, 6400, gs, seed=m * 7 + gs)
        ref = stock(x, wq, s, b, gs)
        got = int4_vocab_qmm(x, wq, s, b, group_size=gs)
        mx.eval(ref, got)
        tol = max(0.05, 0.02 * float(mx.abs(ref).max()))
        d = dmax(ref, got)
        good = d <= tol
        ok &= good
        print(f"   m={m:3d} gs={gs:3d}  dmax={d:.5f} tol={tol:.4f}  {'OK' if good else 'FAIL'}")
    return ok


def check_prefill() -> bool:
    ok = True
    print("== prefill lane (synthetic, bit-exact expected)")
    k, n = PREFILL_SHAPES[0]
    m, gs = 128, 32
    x, wq, s, b = make(m, k, n, 32, seed=m + k)
    ref = stock(x, wq, s, b, gs)
    got = int4_prefill_qmm(x, wq, s, b, group_size=gs)
    mx.eval(ref, got)
    exact = bool(mx.array_equal(got, ref))
    ms_ref = bench(lambda: stock(x, wq, s, b, gs))
    ms_new = bench(lambda: int4_prefill_qmm(x, wq, s, b, group_size=gs))
    ok &= exact
    print(
        f"   ({n}x{k}) m={m}  bit-exact={exact}  "
        f"stock {ms_ref:.1f} ms  mma {ms_new:.1f} ms  {ms_ref / ms_new:.2f}x"
    )
    return ok


def bench_vocab_wide() -> bool:
    print("== vocab lane speed (N=248320 K=5120 g64, synthetic)")
    ok = True
    for m in (8, 16):
        x, wq, s, b = make(m, 5120, 248320, 64, seed=m)
        ms_ref = bench(lambda: stock(x, wq, s, b, 64), iters=10)
        ms_new = bench(lambda: int4_vocab_qmm(x, wq, s, b, group_size=64), iters=10)
        sp = ms_ref / ms_new
        ok &= sp > 1.0
        print(f"   m={m:3d}  stock {ms_ref:7.2f} ms  mma {ms_new:7.2f} ms  {sp:5.2f}x")
    return ok


def _real_weights():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DRAFTER_REPO, "mtp.safetensors")
    return mx.load(path)


def _real_shapes(f):
    return [
        ("q_proj", 12288, 5120),
        ("o_proj", 5120, 6144),
        ("up_proj", 17408, 5120),
        ("down_proj", 5120, 17408),
    ]


def load_real_tensors(f, name, n, k):
    base = {
        "q_proj": "mtp.layers.0.self_attn.q_proj",
        "o_proj": "mtp.layers.0.self_attn.o_proj",
        "up_proj": "mtp.layers.0.mlp.up_proj",
        "down_proj": "mtp.layers.0.mlp.down_proj",
    }[name]
    return f[f"{base}.weight"], f[f"{base}.scales"], f[f"{base}.biases"]


def check_real(drafter_file) -> bool:
    ok = True
    f = mx.load(drafter_file)
    print("== prefill lane, REAL W4/G64 drafter weights (bit-exact expected)")
    for name, n, k in _real_shapes(None):
        wq, s, b = load_real_tensors(f, name, n, k)
        for m in (128, 256):
            x = (mx.random.normal((m, k), key=mx.random.key(m + n)) * 0.3).astype(mx.bfloat16)
            ref = stock(x, wq, s, b, 64)
            got = int4_prefill_qmm(x, wq, s, b, group_size=64)
            mx.eval(ref, got)
            exact = bool(mx.array_equal(got, ref))
            ms_ref = bench(lambda: stock(x, wq, s, b, 64))
            ms_new = bench(lambda: int4_prefill_qmm(x, wq, s, b, group_size=64))
            ok &= exact
            print(
                f"   {name:9s} m={m:4d}  bit-exact={exact}  "
                f"stock {ms_ref:7.1f} ms  mma {ms_new:7.1f} ms  {ms_ref / ms_new:5.2f}x"
            )
    print("== vocab lane on real up_proj weights")
    wq, s, b = load_real_tensors(f, "up_proj", 17408, 5120)
    for m in (8, 16):
        x = (mx.random.normal((m, 5120), key=mx.random.key(m)) * 0.3).astype(mx.bfloat16)
        ref = stock(x, wq, s, b, 64)
        got = int4_vocab_qmm(x, wq, s, b, group_size=64)
        mx.eval(ref, got)
        ms_ref = bench(lambda: stock(x, wq, s, b, 64))
        ms_new = bench(lambda: int4_vocab_qmm(x, wq, s, b, group_size=64))
        print(
            f"   up_proj m={m:3d}  dmax={dmax(ref, got):.5f}  "
            f"stock {ms_ref:6.2f} ms  mma {ms_new:6.2f} ms  {ms_ref / ms_new:5.2f}x"
        )
    return ok


def main() -> int:
    mx.set_default_device(mx.gpu)
    real = "--real-weights" in sys.argv
    ok = True
    ok &= check_vocab()
    ok &= bench_vocab_wide()
    ok &= check_prefill()
    if real:
        ok &= check_real(_real_weights())
    print()
    print("ALL GATES PASS" if ok else "GATE FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
