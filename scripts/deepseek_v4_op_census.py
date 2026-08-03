"""Primitive-graph census for one DeepSeek-V4 decode step.

**Read ``deepseek_v4_dispatch_census.py`` first.**  That one counts the kernels
Metal is actually asked to run, per name, with host-encode and GPU-execution
intervals, and it is the measurement of record.  This script counts *graph
primitives* instead — a strictly coarser proxy, since MLX services some of them
by rewriting strides and fuses others — but it needs no instrumented MLX build,
runs on CPU, and gives a clean **per-component** breakdown (one ``pre`` call, one
attention call per layer type) that a whole-step dispatch trace cannot separate.
Use it to find *where* the primitives are; use the dispatch census to say what
the change is worth.

**Why either exists.**  The measured decode cycle on this box is ``84.8 ms``
fixed plus ``8.9 ms/K``, with the target GPU forward occupying 71-81% of every
cycle.  That leaves the *structure* of the dispatch stream — how many kernels the
host has to build and encode per token — as the remaining lever, not bytes.

**How.**  ``mx.export_to_dot`` serialises the unevaluated graph behind a set of
output arrays, one ``label ="Primitive"`` per node.  Evaluating the inputs first
(parameters, cache tensors) truncates the graph exactly at the step boundary, so
the census is "primitives this decode step adds", not "primitives since load".
The same trick censuses a single component (evaluate its inputs, call it, count).

Two numbers are reported per census:

``nodes``
    Every primitive in the graph.  This is what the Python side has to build and
    what the scheduler walks.
``kernels``
    ``nodes`` minus the primitives Metal services by rewriting strides rather
    than launching anything (``Broadcast``, ``Reshape``, ``Transpose``,
    ``Squeeze``, ``ExpandDims``, ``StopGradient``, ``Depends``).  ``Slice`` and
    ``Copy`` are *not* in that set: they launch on Metal whenever the result is
    not contiguous, which is the common case here.

``kernels`` is therefore a slight over-count and ``nodes`` a large one; the pair
brackets the true dispatch count, and the ratio before/after a change is the
quantity that carries over to the GPU window.

The model is the shrunk seeded config from tests/test_deepseek_v4_decode.py with
DeepSeek-V4-Flash's *structural* constants restored (43 layers, the shipped
compress-ratio pattern, ``hc_mult=4``, ``hc_sinkhorn_iters=20``): primitive count
depends on the structure, not on the widths, so a 32-wide model has the same
graph shape as the 4096-wide one and costs nothing to run on CPU.

Usage::

    python scripts/deepseek_v4_op_census.py                 # default arms
    python scripts/deepseek_v4_op_census.py --prompt 600    # deeper context
    MTPLX_DSV4_ATTN=dense python scripts/deepseek_v4_op_census.py
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import re
import sys
import time
from collections import Counter

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")

#: Primitives Metal services by rewriting strides — no kernel launch.
_VIEW_PRIMS = {
    "Broadcast",
    "Reshape",
    "Transpose",
    "Squeeze",
    "ExpandDims",
    "StopGradient",
    "Depends",
}

_LABEL = re.compile(r'label ="([^"]+)"')


def _load_module(path=None):
    path = path or _MODEL
    spec = importlib.util.spec_from_file_location("dsv4_census", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv4_census"] = mod
    spec.loader.exec_module(mod)
    return mod


def census(*outputs):
    """``(nodes, kernels, Counter)`` for the graph behind ``outputs``."""
    outs = [o for o in outputs if isinstance(o, mx.array)]
    buf = io.StringIO()
    mx.export_to_dot(buf, *outs)
    names = _LABEL.findall(buf.getvalue())
    counts = Counter(names)
    kernels = sum(n for p, n in counts.items() if p not in _VIEW_PRIMS)
    return len(names), kernels, counts


def _seeded_model(D, layers, seed=0):
    """43-layer shrunk model with DeepSeek-V4-Flash's structural constants."""
    ratios = list(D._DEFAULT_COMPRESS_RATIOS)[:layers]
    mx.random.seed(seed)
    args = D.ModelArgs(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=layers,
        num_hash_layers=3,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=512,
        compress_ratios=ratios,
        compress_rope_theta=160000.0,
        sliding_window=16,
        rope_scaling={
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=0.0,
    )
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, args.n_routed_experts, value.shape).astype(
                mx.int32
            )
        elif value.ndim == 1:
            centre = 1.0 if leaf == "scale" or name.endswith("norm.weight") else 0.0
            new = mx.random.normal(value.shape) * 0.1 + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return args, model


def _eval_cache(cache):
    mx.eval([a for c in cache for a in c.state if isinstance(a, mx.array)])


def _cache_outputs(cache):
    return [a for c in cache for a in c.state if isinstance(a, mx.array)]


def _decode_step_census(D, args, model, prompt):
    """One ``s == 1`` decode step against a warm cache."""
    mx.random.seed(1234)
    ids = mx.random.randint(0, args.vocab_size, (1, prompt + 1))
    cache = model.make_cache()
    logits = model(ids[:, :prompt], cache=cache)
    mx.eval(logits)
    _eval_cache(cache)
    logits = model(ids[:, prompt : prompt + 1], cache=cache)
    return census(logits, *_cache_outputs(cache))


def _component_censuses(D, args, model, prompt):
    """Per-component graphs at decode shape (``b = s = 1``)."""
    out = {}
    hc = args.hc_mult
    dim = args.hidden_size
    h = mx.random.normal((1, 1, hc, dim))
    mx.eval(h)
    layer = model.model.layers[0]

    y, post, comb = layer.attn_hc.pre(h)
    out["HyperConnection.pre"] = census(y, post, comb)
    mx.eval(y, post, comb)
    out["HyperConnection.post"] = census(layer.attn_hc.post(y, h, post, comb))
    out["HeadHC (once per token)"] = census(model.model.hc_head(h))

    # Attention, one call per layer type, against the warm cache a real prefill
    # of ``prompt`` tokens leaves behind.
    mx.random.seed(1234)
    ids = mx.random.randint(0, args.vocab_size, (1, prompt))
    cache = model.make_cache()
    mx.eval(model(ids, cache=cache))
    _eval_cache(cache)
    x = mx.random.normal((1, 1, dim))
    mx.eval(x)
    seen = set()
    for idx, ratio in enumerate(args.compress_ratios[: len(model.model.layers)]):
        if ratio in seen:
            continue
        seen.add(ratio)
        attn = model.model.layers[idx].attn
        out[f"Attention ratio={ratio}"] = census(
            attn(x, cache=cache[idx]), *_cache_outputs([cache[idx]])
        )
    return out


def _hc_wall_clock(model, reps=200):
    """Host-side cost of one HC pre call, in microseconds (median of ``reps``).

    Tensors are 4x4; on CPU essentially all of this is graph-build + dispatch
    overhead, which is the same quantity that binds the GPU decode cycle.
    """
    layer = model.model.layers[0]
    hc = layer.attn_hc.hc
    dim = layer.attn_hc.dim
    h = mx.random.normal((1, 1, hc, dim))
    mx.eval(h)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y, post, comb = layer.attn_hc.pre(h)
        mx.eval(y, post, comb)
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", type=int, default=200)
    ap.add_argument("--no-timing", action="store_true")
    args_cli = ap.parse_args()

    # CPU: MLX's fp32 matmul is bit-exact there, and the graph is the same graph.
    mx.set_default_device(mx.cpu)
    D = _load_module()
    args, model = _seeded_model(D, args_cli.layers)

    print(f"# DeepSeek-V4 primitive census  (mlx {mx.__version__}, CPU trace)")
    print(
        f"#   layers={args_cli.layers}  prompt={args_cli.prompt}  "
        f"hc_mult={args.hc_mult}  sinkhorn_iters={args.hc_sinkhorn_iters}"
    )
    print(f"#   attn={D._attn_mode_from_env()}  "
          f"hc_compile={D._HC_COMPILE} (max_rows={D._HC_COMPILE_MAX_ROWS})  "
          f"o_lora={D._o_lora_mode_from_env()}")
    print()

    nodes, kernels, counts = _decode_step_census(D, args, model, args_cli.prompt)
    print(f"ONE DECODE STEP (s=1):  nodes={nodes}  kernels={kernels}")
    print("  top primitives: " + ", ".join(
        f"{p}={n}" for p, n in counts.most_common(12)
    ))
    print()

    print("PER-COMPONENT (decode shape, b=s=1):")
    for name, (n, k, c) in _component_censuses(D, args, model, args_cli.prompt).items():
        print(f"  {name:<28} nodes={n:<6} kernels={k:<6} "
              f"top={', '.join(f'{p}:{v}' for p, v in c.most_common(4))}")
    print()

    if not args_cli.no_timing:
        print(f"HC pre host cost (CPU, median of 200): "
              f"{_hc_wall_clock(model):.1f} us/call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
