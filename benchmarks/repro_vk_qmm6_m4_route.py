"""Repro: the 6-bit split-K hexpack route is a LOSS at m == 4 on a 27B trunk.

Why this benchmark exists, and why it is not the obvious one: timing one
weight tensor repeatedly in a loop leaves that tensor hot, which flatters
whichever kernel has the better launch geometry. A real decode round touches
every layer's weights exactly once, so the whole model streams from DRAM.
This builds the full Qwen3.8-27B linear stack at a chosen bit width and
sweeps it once per iteration, which reproduces that traffic. Weight VALUES
are random (timing does not depend on them), so no checkpoint is needed.

Every (m, kernel) cell is re-measured once per round with the rounds
interleaved, so run order and thermal drift cannot ride along with the cell
index. An earlier fixed-order version of this same benchmark disagreed with
itself by 45% on one cell for exactly that reason.

Measured on Apple M3 Max (applegpu_g15s), MLX 0.32.1, macOS 26.3, as the
range over three independent runs:

    bits=6  m=4   stock  63.3-64.7 ms   hexpack  77.5-82.8 ms   +20 to +30%  LOSS
    bits=6  m=5   stock  82.8-87.3 ms   hexpack  67.3-67.7 ms   -19 to -22%  win
    bits=6  m=6   stock 131.0-134.5 ms  hexpack  74.0-86.2 ms   -35 to -45%  win
    bits=4  m=4   stock  48.6-54.2 ms   hexpack  40.4-45.0 ms   -17%         win

so the regression is specific to 6 bits at m == 4, which is why
mtplx.nax_verify.vk_qmm6_m4_enabled gates that route alone and leaves the
4-bit m4 route and the 6-bit m5/m6 routes untouched.

Run:
    PYTHONPATH=. python benchmarks/repro_vk_qmm6_m4_route.py 6
    PYTHONPATH=. python benchmarks/repro_vk_qmm6_m4_route.py 4
"""

import statistics
import sys
import time

import mlx.core as mx

from mtplx.verify_kernels import (
    vk_eligible_ksplit,
    vk_qmm_m4_ksplit,
    vk_qmm_m6_ksplit,
)

GS, DT = 64, mx.float16
H, INTER, VOCAB = 5120, 17408, 248320
N_LAYERS, FULL_EVERY = 64, 4

# Qwen3.8-27B: 48 gated-delta-net layers, 16 full-attention layers, one
# lm_head. Shapes read from the checkpoint's config.json.
GDN = [("lin.in_proj_qkv", H, 10240), ("lin.in_proj_z", H, 6144),
       ("lin.out_proj", 6144, H), ("mlp.gate_proj", H, INTER),
       ("mlp.up_proj", H, INTER), ("mlp.down_proj", INTER, H)]
FULL = [("attn.q_proj", H, 12288), ("attn.k_proj", H, 1024),
        ("attn.v_proj", H, 1024), ("attn.o_proj", 6144, H),
        ("mlp.gate_proj", H, INTER), ("mlp.up_proj", H, INTER),
        ("mlp.down_proj", INTER, H)]


def qweights(K, N, bits):
    return (mx.random.randint(0, 2**31 - 1, (N, K * bits // 32)).astype(mx.uint32),
            mx.random.normal((N, K // GS), dtype=DT) * 0.01,
            mx.random.normal((N, K // GS), dtype=DT) * 0.01)


def wbytes(K, N, bits):
    return N * K * bits / 8 + 2 * N * (K // GS) * 2


def roofline():
    """Sustained read rate on a plain streaming reduce, for context."""
    x = mx.random.normal((2 * 1024**3 // 2,), dtype=DT)
    mx.eval(x)
    mx.eval(mx.sum(x))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(6):
        mx.eval(mx.sum(x))
    mx.synchronize()
    gbs = (x.size * 2) / ((time.perf_counter() - t0) / 6) / 1e9
    del x
    mx.clear_cache()
    return gbs


def build(bits):
    layers, total = [], 0.0
    for i in range(N_LAYERS):
        spec = FULL if (i + 1) % FULL_EVERY == 0 else GDN
        layer = [(K, N, qweights(K, N, bits)) for _, K, N in spec]
        total += sum(wbytes(K, N, bits) for _, K, N in spec)
        layers.append(layer)
        mx.eval([a for _, _, t in layer for a in t])
    lm = (H, VOCAB, qweights(H, VOCAB, bits))
    mx.eval(lm[2])
    return layers, lm, total + wbytes(H, VOCAB, bits)


def make_forward(layers, lm, bits, m, use_vk):
    xs = {K: mx.random.normal((m, K), dtype=DT) for K in (H, 6144, INTER)}
    mx.eval(list(xs.values()))

    def call(K, N, wq, sc, bi):
        if use_vk and vk_eligible_ksplit(m, K, N, bits, GS, DT):
            fn = vk_qmm_m4_ksplit if m == 4 else vk_qmm_m6_ksplit
            return fn(xs[K], wq, sc, bi, bits=bits, group_size=GS)
        return mx.quantized_matmul(xs[K], wq, sc, bi, transpose=True,
                                   group_size=GS, bits=bits)

    def forward():
        outs = [call(K, N, *t) for layer in layers for K, N, t in layer]
        outs.append(call(lm[0], lm[1], *lm[2]))
        mx.eval(outs)

    return forward


def main() -> int:
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ms = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "4,5,6").split(",")]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 9

    gbs = roofline()
    layers, lm, total = build(bits)
    floor_ms = total / gbs / 1e9 * 1e3
    print(f"device={mx.device_info()['architecture']}  roofline={gbs:.0f} GB/s")
    print(f"bits={bits}  weights resident={total / 2**30:.2f} GiB  "
          f"bandwidth floor={floor_ms:.1f} ms/forward  rounds={rounds}")

    cells = [(m, vk) for m in ms for vk in (False, True)]
    fns = {c: make_forward(layers, lm, bits, c[0], c[1]) for c in cells}
    for f in fns.values():
        f()
    mx.synchronize()

    samples: dict = {c: [] for c in cells}
    for _ in range(rounds):
        for c in cells:               # interleaved: no cell owns a run phase
            t0 = time.perf_counter()
            fns[c]()
            mx.synchronize()
            samples[c].append((time.perf_counter() - t0) * 1e3)

    print(f"\n{'m':>4}{'stock ms':>11}{'hexpack ms':>13}{'delta':>9}"
          f"{'stock %roof':>13}{'hexpack %roof':>15}")
    bad = 0
    for m in ms:
        s = statistics.median(samples[(m, False)])
        v = statistics.median(samples[(m, True)])
        if m == 4 and bits == 6 and v > s:
            bad += 1
        print(f"{m:>4}{s:>11.2f}{v:>13.2f}{(v / s - 1) * 100:>+8.1f}%"
              f"{100 * floor_ms / s:>12.0f}%{100 * floor_ms / v:>14.0f}%")
    if bits == 6:
        print("\nm=4 hexpack slower than stock: "
              f"{'YES (the regression this gate exists for)' if bad else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
