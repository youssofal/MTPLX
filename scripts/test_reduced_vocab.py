#!/usr/bin/env python3
"""PASS/FAIL harness for the reduced-vocabulary draft head.

    python scripts/test_reduced_vocab.py --ids <ids.npy>

 1. row slice   : reduced logits on S == full-head logits on S
 2. masking     : logits outside S are the sentinel and never win
 3. argmax      : reduced argmax == full argmax whenever it is in S
 4. samplers    : MTPLX draft samplers give finite q, support inside S, sum 1
 5. mx.compile  : the head traces and runs inside a compiled graph
 6. timing      : ms/call, full vs reduced
 7. optional    : wrap a real artifact draft head (--artifact <model dir>)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def skip(name: str, detail: str = "") -> None:
    print(f"SKIP  {name}" + (f"  -- {detail}" if detail else ""))


def load_module():
    try:
        import mtplx.reduced_vocab_draft as mod

        print(f"module: {mod.__file__} (installed)")
        return mod
    except Exception:
        pass
    path = HERE / "reduced_vocab_draft.py"
    spec = importlib.util.spec_from_file_location("reduced_vocab_draft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reduced_vocab_draft"] = mod
    spec.loader.exec_module(mod)
    print(f"module: {path} (local file)")
    return mod


def build_random_quantized_head(vocab: int, hidden: int, bits: int, group_size: int):
    import mlx.core as mx
    import mlx.nn as nn

    head = nn.QuantizedLinear(
        hidden, 64, bias=False, group_size=group_size, bits=bits, mode="affine"
    )
    weights, scales, biases = [], [], []
    step = 16384
    for start in range(0, vocab, step):
        rows = min(step, vocab - start)
        dense = (mx.random.normal((rows, hidden)) * 0.05).astype(mx.float16)
        wq, sc, bi = mx.quantize(dense, group_size=group_size, bits=bits, mode="affine")
        mx.eval(wq, sc, bi)
        weights.append(wq)
        scales.append(sc)
        biases.append(bi)
        del dense
    head.weight = mx.concatenate(weights, axis=0)
    head.scales = mx.concatenate(scales, axis=0)
    head.biases = mx.concatenate(biases, axis=0)
    mx.eval(head.weight, head.scales, head.biases)
    return head


def timeit(fn, x, iters: int) -> float:
    import mlx.core as mx

    for _ in range(10):
        mx.eval(fn(x))
    started = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(x))
    return 1000.0 * (time.perf_counter() - started) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default=str(HERE / "ids/ids_32k.npy"))
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--mode", default="full", choices=["full", "compact"])
    parser.add_argument("--neg", default="finite")
    parser.add_argument("--artifact", default=None, help="model dir for the live smoke test")
    args = parser.parse_args()

    import mlx.core as mx

    print(f"mlx {getattr(mx, '__version__', '?')}  python {sys.version.split()[0]}")
    mod = load_module()

    ids = mod.load_subset_ids(args.ids, vocab_size=args.vocab)
    print(
        f"subset: {args.ids}  |S|={ids.size} "
        f"({100.0 * ids.size / args.vocab:.1f}% of {args.vocab})"
    )

    print("\n-- building the reference head --")
    started = time.perf_counter()
    base = build_random_quantized_head(args.vocab, args.hidden, args.bits, args.group_size)
    print(
        f"   base head {tuple(base.weight.shape)} uint32 + scales "
        f"{tuple(base.scales.shape)}  built in {time.perf_counter() - started:.1f}s"
    )

    started = time.perf_counter()
    head = mod.build_reduced_head(base, ids, mode=args.mode, neg=args.neg, keep_base=True)
    print(f"   reduced head built in {time.perf_counter() - started:.1f}s")
    check(
        "1a. sliced weight/scales/biases row counts",
        int(head.head.weight.shape[0]) == ids.size
        and int(head.head.scales.shape[0]) == ids.size
        and int(head.head.biases.shape[0]) == ids.size,
        f"{tuple(head.head.weight.shape)} / {tuple(head.head.scales.shape)}",
    )
    check(
        "1b. sliced packed row bit-identical to the full head",
        bool(mx.all(head.head.weight == base.weight[mx.array(ids.astype(np.int32))]).item()),
    )

    x = (mx.random.normal((1, 1, args.hidden)) * 0.5).astype(mx.float16)
    mx.eval(x)
    full = base(x)
    red = head(x)
    mx.eval(full, red)

    if args.mode == "full":
        check("2a. output shape is full-vocab", tuple(red.shape) == (1, 1, args.vocab),
              str(tuple(red.shape)))
        idx = mx.array(ids.astype(np.int32))
        on_s_full = mx.take(full.reshape(-1), idx)
        on_s_red = mx.take(red.reshape(-1), idx)
        mx.eval(on_s_full, on_s_red)
        exact = bool(mx.all(on_s_full == on_s_red).item())
        max_abs = float(mx.max(mx.abs(on_s_full.astype(mx.float32) - on_s_red.astype(mx.float32))).item())
        check("2b. logits on S identical to the full head", exact,
              f"max|diff| = {max_abs:g}" + ("" if exact else "  (NOT bit-exact)"))

        mask = np.ones(args.vocab, dtype=bool)
        mask[ids] = False
        off = np.asarray(red.reshape(-1), dtype=np.float32)[mask]
        on = np.asarray(on_s_red, dtype=np.float32)
        check("2c. off-subset logits are the mask sentinel",
              bool(np.all(off == off[0])), f"sentinel = {off[0]:g}")
        check("2d. sentinel is far below every in-subset logit",
              bool(off[0] < on.min() - 1e3),
              f"min on-S logit = {on.min():.3f}")
        check("2e. no NaN anywhere in the reduced row",
              not bool(np.isnan(np.asarray(red.reshape(-1), dtype=np.float32)).any()))
    else:
        check("2a. output shape is compact", tuple(red.shape) == (1, 1, ids.size),
              str(tuple(red.shape)))

    print("\n-- argmax agreement over 64 random hidden states --")
    id_set = set(int(i) for i in ids)
    agree = total_in_s = 0
    always_in_s = True
    for _ in range(64):
        xi = (mx.random.normal((1, 1, args.hidden)) * 0.5).astype(mx.float16)
        fa = int(mx.argmax(base(xi), axis=-1).item())
        ra_raw = int(mx.argmax(head(xi), axis=-1).item())
        ra = ra_raw if args.mode == "full" else int(ids[ra_raw])
        always_in_s &= ra in id_set
        if fa in id_set:
            total_in_s += 1
            agree += int(fa == ra)
    check("3a. reduced argmax always inside S", always_in_s)
    check(
        "3b. reduced argmax == full argmax whenever full argmax in S",
        total_in_s == 0 or agree == total_in_s,
        f"{agree}/{total_in_s} cases (full argmax landed in S {total_in_s}/64 times)",
    )

    print("\n-- MTPLX draft samplers on the reduced row --")
    row = head(x)[:, -1, :][0]
    mx.eval(row)
    try:
        from mtplx.fast_sampling import sparse_distribution_from_mlx_logits
        from mtplx.sampling import SamplerConfig
    except ImportError as exc:
        skip("4A. host draft sampler", f"mtplx not importable ({exc})")
        sparse_distribution_from_mlx_logits = None
    if sparse_distribution_from_mlx_logits is not None:
        try:
            cfg = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
            dist = sparse_distribution_from_mlx_logits(row, cfg)
            probs = np.asarray(dist.probs, dtype=np.float64)
            support = np.asarray(dist.token_ids, dtype=np.int64)
            if args.mode == "compact":
                support = ids[support]
            check("4a. sparse_distribution_from_mlx_logits returns finite mass",
                  bool(np.all(np.isfinite(probs))) and abs(probs.sum() - 1.0) < 1e-9,
                  f"support={support.size}, sum={probs.sum():.12f}")
            check("4b. every support id is inside S",
                  bool(np.all(np.isin(support, ids))))
            check("4c. SparseDistribution.vocab_size is the full vocabulary",
                  int(dist.vocab_size) == args.vocab or args.mode == "compact",
                  f"vocab_size={dist.vocab_size}")
            stable = True
            for temperature in (0.6, 1.0, 2.0):
                for top_p in (0.95, 1.0):
                    d = sparse_distribution_from_mlx_logits(
                        row, SamplerConfig(temperature=temperature, top_p=top_p, top_k=20)
                    )
                    pr = np.asarray(d.probs, dtype=np.float64)
                    stable &= bool(np.all(np.isfinite(pr))) and abs(pr.sum() - 1.0) < 1e-9
            check("4d. sampler stable across T in {0.6,1,2} x top_p in {0.95,1.0}", stable)
        except Exception as exc:
            check("4A. host draft sampler", False, f"{type(exc).__name__}: {exc}")

    if args.mode == "full":
        try:
            from mtplx.generation import _device_draft_q_arrays
        except ImportError as exc:
            skip("4B. device draft core q arrays", f"mtplx.generation not importable ({exc})")
            _device_draft_q_arrays = None
        if _device_draft_q_arrays is not None:
            try:
                top_idx, q_norm = _device_draft_q_arrays(
                    row, temperature=1.0, top_k=20, top_p=0.95
                )
                mx.eval(top_idx, q_norm)
                q = np.asarray(q_norm, dtype=np.float64)
                tid = np.asarray(top_idx, dtype=np.int64)
                check("4e. device-core q arrays finite and normalized",
                      bool(np.all(np.isfinite(q))) and abs(q.sum() - 1.0) < 1e-6,
                      f"sum={q.sum():.9f}")
                check("4f. device-core support inside S", bool(np.all(np.isin(tid, ids))))
            except Exception as exc:
                check("4B. device draft core q arrays", False, f"{type(exc).__name__}: {exc}")

    print("\n-- mx.compile --")
    try:
        def chain(v):
            logits = head(v)
            return mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)

        compiled = mx.compile(chain)
        out_c = compiled(x)
        out_e = chain(x)
        mx.eval(out_c, out_e)
        check("5. compiled graph matches eager", int(out_c.item()) == int(out_e.item()),
              f"token={int(out_c.item())}")
    except Exception as exc:
        check("5. mx.compile", False, f"{type(exc).__name__}: {exc}")

    print(f"\n-- timing ({args.iters} iters, one mx.eval per call) --")
    t_full = timeit(base, x, args.iters)
    t_red = timeit(head, x, args.iters)
    t_slice = timeit(head.head, x, args.iters)
    head_bytes = args.vocab * (args.hidden * args.bits // 8) + args.vocab * (
        args.hidden // args.group_size
    ) * 4
    print(f"   full head          : {t_full:8.3f} ms/call")
    print(f"   reduced (sliced)   : {t_slice:8.3f} ms/call   (matmul only, |S| logits)")
    print(f"   reduced + scatter  : {t_red:8.3f} ms/call   (full-vocab output)")
    print(f"   scatter overhead   : {t_red - t_slice:8.3f} ms/call")
    print(f"   speedup            : {t_full / max(t_red, 1e-9):8.2f}x  "
          f"(saved {t_full - t_red:.3f} ms/draft step)")
    print(f"   full-head weights  : {head_bytes / 1e6:.1f} MB "
          f"-> implied BW {head_bytes / (t_full * 1e-3) / 1e9:.1f} GB/s")
    check("6. reduced head is faster than the full head", t_red < t_full,
          f"{t_full:.3f} -> {t_red:.3f} ms")

    if args.artifact:
        print("\n-- live artifact smoke (loads the model) --")
        try:
            from mtplx.draft_lm_head import _install_draft_lm_head
            from mtplx.runtime import load

            rt = load(Path(args.artifact), mtp=True)
            report = _install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")
            text = getattr(rt.model, "language_model", rt.model)
            installed = getattr(text, "_mtplx_draft_lm_head", None)
            print(f"   report keys: {sorted(report)}")
            wrapped = mod.build_reduced_head(installed, ids, mode="full", neg=args.neg)
            hidden = (mx.random.normal((1, 1, args.hidden)) * 0.02).astype(
                installed.weight.dtype if hasattr(installed, "weight") else mx.float16
            )
            out = wrapped(hidden.astype(mx.float16))
            mx.eval(out)
            check("7. wrapper works on the real artifact draft head",
                  tuple(out.shape) == (1, 1, args.vocab), str(tuple(out.shape)))
        except Exception as exc:
            check("7. live artifact smoke", False, f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failing checks)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("RESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
