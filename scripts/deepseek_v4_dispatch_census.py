"""Real Metal dispatch census for one DeepSeek-V4 decode step.

Where ``deepseek_v4_op_census.py`` counts graph *primitives*, this counts the
kernels Metal is actually asked to run, per name, plus the command buffers they
were encoded into and the host-side waits around them.  It uses the
instrumented MLX build in ``mlx-profiler``, which writes one JSONL row per
dispatch (``record: op``), per committed command buffer (``record: cb``, with the
host encode interval *and* the GPU execution interval — the overlap signal) and
per wait bucket (``record: wait``).

**Per-step isolation.**  The census file covers the whole process, so a single
run cannot say what one decode step cost.  This driver therefore runs the same
prefill twice, once followed by ``--decode 1`` and once by ``--decode 9``, and
reports the difference divided by 8.  That cancels load, prefill, compile
tracing and warm-up exactly, and it averages over eight steps rather than
trusting one.

**What it is not.**  The instrumented build is MLX 0.32.1.dev, not the 0.31.2 the
runtime serves on, so kernel *selection* is 0.32's.  Counts and structure carry
over; absolute timings do not, and on a shared box the ``cb`` intervals are
contended by whatever else holds the GPU.  Treat the op/cb counts as the
measurement and the intervals as an indication.

Usage::

    # the whole before/after table (spawns its own child runs)
    PYTHONPATH=/path/to/mlx-profiler/python python scripts/deepseek_v4_dispatch_census.py \\
        --report --baseline /path/to/head/deepseek_v4.py

    # one child run, for driving by hand
    MLX_DISPATCH_CENSUS=/abs/census.jsonl PYTHONPATH=... \\
        python scripts/deepseek_v4_dispatch_census.py --run --decode 9
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_CENSUS = os.path.join(_HERE, "deepseek_v4_op_census.py")


def _op_census_module():
    spec = importlib.util.spec_from_file_location("dsv4_op_census", _OP_CENSUS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv4_op_census"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# child: run the workload
# ---------------------------------------------------------------------------
def run_child(model_path, layers, prompt, decode):
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    C = _op_census_module()
    D = C._load_module(model_path)
    args, model = C._seeded_model(D, layers)
    # The serving dtype matters here: at fp32 the ``dense`` arm's upcast of the
    # score block is a no-op, so the fp32 arms understate what ``fused`` removes.
    if os.environ.get("DSV4_CENSUS_DTYPE", "").strip().lower() in ("bf16", "bfloat16"):
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

    mx.random.seed(1234)
    ids = mx.random.randint(0, args.vocab_size, (1, prompt + decode))
    cache = model.make_cache()
    logits = model(ids[:, :prompt], cache=cache)
    mx.eval(logits)
    C._eval_cache(cache)
    mx.synchronize()

    for t in range(decode):
        logits = model(ids[:, prompt + t : prompt + t + 1], cache=cache)
        mx.eval(logits)
        C._eval_cache(cache)
    mx.synchronize()
    print(f"child ok: mlx={mx.__version__} file={mx.__file__} decode={decode}")
    return 0


# ---------------------------------------------------------------------------
# parent: spawn, parse, diff
# ---------------------------------------------------------------------------
def _parse(path):
    ops, cbs, waits = Counter(), [], Counter()
    wait_ns, summary = Counter(), None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rec = row.get("record")
            if rec == "op":
                ops[row.get("kernel_name", "?")] += 1
            elif rec == "cb":
                cbs.append(row)
            elif rec == "wait":
                waits[row.get("bucket", "?")] += 1
                wait_ns[row.get("bucket", "?")] += int(row.get("wait_ns", 0))
            elif rec == "summary" and row.get("final"):
                summary = row
    if summary is None:
        raise SystemExit(f"{path}: no final summary row — trace truncated")
    if summary.get("dropped_rows", 0) or not summary.get("complete", False):
        raise SystemExit(
            f"{path}: INVALID trace (dropped_rows="
            f"{summary.get('dropped_rows')} complete={summary.get('complete')})"
        )
    return ops, cbs, waits, wait_ns, summary


def _run_arm(label, model_path, env_extra, layers, prompt, decode, workdir):
    out = os.path.join(workdir, f"{label}-d{decode}.jsonl")
    env = dict(os.environ)
    env["MLX_DISPATCH_CENSUS"] = out
    env.update(env_extra)
    cmd = [
        sys.executable, os.path.abspath(__file__), "--run",
        "--layers", str(layers), "--prompt", str(prompt), "--decode", str(decode),
    ]
    if model_path:
        cmd += ["--model", model_path]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"{label} d={decode} failed:\n{res.stdout}\n{res.stderr}")
    return _parse(out)


def _per_step(label, model_path, env_extra, layers, prompt, workdir, lo=1, hi=9):
    """(ops, cbs, waits, wait_ns) attributable to one decode step."""
    a = _run_arm(label, model_path, env_extra, layers, prompt, lo, workdir)
    b = _run_arm(label, model_path, env_extra, layers, prompt, hi, workdir)
    n = hi - lo
    ops = Counter({k: (b[0][k] - a[0][k]) / n for k in set(a[0]) | set(b[0])})
    cbs = (len(b[1]) - len(a[1])) / n
    waits = Counter({k: (b[2][k] - a[2][k]) / n for k in set(a[2]) | set(b[2])})
    wait_ns = Counter({k: (b[3][k] - a[3][k]) / n for k in set(a[3]) | set(b[3])})
    # host-encode vs GPU-exec, over the command buffers of the decode tail only
    tail = b[1][len(a[1]):]
    enc = sum(c["encode_end_ns"] - c["encode_start_ns"] for c in tail) / n
    gpu = sum(c["gpu_end_ns"] - c["gpu_start_ns"] for c in tail) / n
    return ops, cbs, waits, wait_ns, enc, gpu


def _fmt(ops):
    return sum(ops.values())


def report(args):
    workdir = args.workdir or tempfile.mkdtemp(prefix="dsv4-census-")
    os.makedirs(workdir, exist_ok=True)
    dt = {"DSV4_CENSUS_DTYPE": args.dtype}
    arms = []
    if args.baseline:
        arms.append(("BEFORE (branch point)", args.baseline, dict(dt)))
    for label, env in (
        ("dense + no hc compile", {"MTPLX_DSV4_ATTN": "dense",
                                   "MTPLX_DSV4_HC_COMPILE": "0"}),
        ("fused + no hc compile", {"MTPLX_DSV4_ATTN": "fused",
                                   "MTPLX_DSV4_HC_COMPILE": "0"}),
        ("dense + hc compile", {"MTPLX_DSV4_ATTN": "dense",
                                "MTPLX_DSV4_HC_COMPILE": "1"}),
        ("AFTER: fused + hc compile", {"MTPLX_DSV4_ATTN": "fused",
                                       "MTPLX_DSV4_HC_COMPILE": "1"}),
        ("sdpa + hc compile", {"MTPLX_DSV4_ATTN": "sdpa",
                               "MTPLX_DSV4_HC_COMPILE": "1"}),
    ):
        env.update(dt)
        arms.append((label, None, env))
    results = {}
    print(f"# Metal dispatch census, per decode step "
          f"(layers={args.layers} prompt={args.prompt} dtype={args.dtype}, "
          f"8-step difference)")
    print(f"# traces in {workdir}")
    print()
    print(f"{'arm':<28} {'kernels/step':>13} {'cmd bufs':>9} "
          f"{'encode us':>10} {'gpu us':>9}")
    for label, path, env in arms:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        ops, cbs, waits, wait_ns, enc, gpu = _per_step(
            slug, path, env, args.layers, args.prompt, workdir,
        )
        results[label] = (ops, cbs, waits, wait_ns)
        print(f"{label:<28} {_fmt(ops):>13.1f} {cbs:>9.1f} "
              f"{enc / 1000:>10.1f} {gpu / 1000:>9.1f}")

    print()
    print("TOP KERNELS PER DECODE STEP")
    keys = ["BEFORE (branch point)"] if args.baseline else []
    keys += ["dense + no hc compile", "AFTER: fused + hc compile"]
    keys = [k for k in keys if k in results]
    names = set()
    for k in keys:
        names |= {n for n, v in results[k][0].items() if v >= 1}
    head = "  {:<44}".format("kernel") + "".join(f"{k.split()[0][:14]:>15}" for k in keys)
    print(head)
    for name in sorted(names, key=lambda n: -results[keys[-1]][0].get(n, 0)
                       - results[keys[0]][0].get(n, 0)):
        row = "".join(f"{results[k][0].get(name, 0):>15.1f}" for k in keys)
        print(f"  {name[:44]:<44}{row}")

    print()
    print("WAIT BUCKETS PER DECODE STEP (count, total us)")
    for k in keys:
        _, _, waits, wait_ns = results[k]
        parts = ", ".join(
            f"{b}={waits[b]:.1f}/{wait_ns[b] / 1000:.1f}us"
            for b in sorted(waits, key=lambda b: -wait_ns[b])[:6]
        )
        print(f"  {k:<28} {parts}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="child mode")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--model", default=None, help="deepseek_v4.py to load")
    ap.add_argument("--baseline", default=None,
                    help="a second deepseek_v4.py to census as the BEFORE arm")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", type=int, default=200)
    ap.add_argument("--decode", type=int, default=9)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()
    if args.run:
        return run_child(args.model, args.layers, args.prompt, args.decode)
    if not os.environ.get("PYTHONPATH", "").find("mlx-profiler") >= 0:
        print("warning: PYTHONPATH does not mention mlx-profiler; the census "
              "env var is a no-op on a stock MLX build", file=sys.stderr)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
