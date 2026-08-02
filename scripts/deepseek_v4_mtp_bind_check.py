#!/usr/bin/env python3
"""Bind the REAL MTP draft head out of a merged model dir, without the trunk.

The merged directory (``scripts/deepseek_v4_build_mtp_model.py``) is ~96 GiB, so
loading it whole needs a guarded GPU window.  Nothing about *binding the draft
head* does: ``mtp.0.*`` lives in one 6.5 GiB shard, and the only other tensors
its forward touches are the shared embedding and lm_head (reference
``Transformer.__init__`` L792-793).  This script therefore builds exactly those
three pieces at full config dimensions, quantizes them with the per-path entries
the merged ``config.json`` declares, binds strictly, and runs one token through.

What it proves that the synthetic loader tests cannot: the shipped tensor names,
shapes, dtypes and quantization parameters actually fit the module tree, and the
real weights produce finite draft logits.

    python scripts/deepseek_v4_mtp_bind_check.py \
        --model ~/models/DeepSeek-V4-Flash-2bit-DQ-mtp
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "dsv4_bind", _HERE.parent / "mtplx" / "models" / "deepseek_v4.py"
)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_bind"] = D
_spec.loader.exec_module(D)


class _EmbedOnly(nn.Module):
    """Stands in for ``DeepseekV4Model`` so the embedding keeps its real path
    (``model.embed_tokens.weight``) without constructing 43 trunk layers."""

    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)


class MTPOnly(nn.Module):
    """The draft head plus the two tensors it shares with the trunk."""

    def __init__(self, args):
        super().__init__()
        self.model = _EmbedOnly(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.mtp = [D.DeepseekV4MTP(args, args.num_hidden_layers)]

    def __call__(self, h, input_ids, cache=None):
        return self.mtp[0](h, input_ids, self.model.embed_tokens, self.lm_head,
                           cache=cache)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=4)
    args_cli = ap.parse_args()

    root = Path(args_cli.model).expanduser()
    cfg = json.loads((root / "config.json").read_text())
    wmap = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    wanted = {k for k in wmap
              if k.startswith("mtp.") or k.startswith("model.embed_tokens.")
              or k.startswith("lm_head.")}
    shards = sorted({wmap[k] for k in wanted})
    mtp_shards = sorted({wmap[k] for k in wmap if k.startswith("mtp.")})
    print(f"model dir     : {root}")
    print(f"mtp tensors   : {sum(k.startswith('mtp.') for k in wmap)} in {mtp_shards}")
    print(f"shards to read: {len(shards)} ({', '.join(shards)})")

    weights = {}
    for shard in shards:
        for k, v in mx.load(str(root / shard)).items():
            if k in wanted:
                weights[k] = v
    print(f"loaded        : {len(weights)} tensors")

    args = D.ModelArgs.from_dict(cfg)
    model = MTPOnly(args)

    quant = cfg["quantization"]

    def class_predicate(p, m):
        # exactly mlx-lm's rule (mlx_lm/utils.py load_model._quantize)
        if p in quant:
            return quant[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                mode=quant.get("mode", "affine"), class_predicate=class_predicate)

    tree = {k for k, _ in tree_flatten(model.parameters())}
    missing = sorted(tree - set(weights))
    extra = sorted(set(weights) - tree)
    print(f"module tree   : {len(tree)} params")
    print(f"missing       : {len(missing)} {missing[:5]}")
    print(f"extra         : {len(extra)} {extra[:5]}")
    if missing or extra:
        return 1

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    print(f"peak memory after bind: {mx.get_peak_memory() / 2**30:.2f} GiB")

    blk = model.mtp[0]
    print(f"draft block   : layer_id={blk.attn.layer_id} "
          f"compress_ratio={blk.attn.compress_ratio} "
          f"window={blk.attn.window_size} hash_gate={blk.ffn.gate.hash}")
    print(f"                e_proj={type(blk.e_proj).__name__}"
          f"(bits={getattr(blk.e_proj, 'bits', None)},"
          f"gs={getattr(blk.e_proj, 'group_size', None)}) "
          f"switch_mlp={type(blk.ffn.switch_mlp.gate_proj).__name__}")

    # one forward: a real trunk hidden state is not available without the trunk,
    # so drive the block from the embedding of the same tokens (the tensor has
    # the right shape, scale and dtype), which is what the shape/finiteness
    # check needs.  Numerical parity is the shrunk-config oracle's job.
    s = args_cli.seq
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8][:s]])
    e = model.model.embed_tokens(ids).astype(mx.bfloat16)
    h = mx.broadcast_to(e[:, :, None, :], (1, s, args.hc_mult, args.hidden_size))
    logits = model(h, ids)
    mx.eval(logits)
    finite = bool(mx.all(mx.isfinite(logits)).item())
    print(f"draft logits  : {logits.shape} dtype={logits.dtype} finite={finite}")
    print(f"                min={float(mx.min(logits).item()):.4f} "
          f"max={float(mx.max(logits).item()):.4f} "
          f"argmax={[int(t) for t in mx.argmax(logits[0], axis=-1)]}")

    # streaming step through the block's own cache: proves make_mtp_cache's
    # shape contract against the real attention module.
    cache = D.DeepseekV4Cache(window_size=blk.attn.window_size,
                              compress_ratio=blk.attn.compress_ratio,
                              head_dim=blk.attn.head_dim)
    step = model(h[:, :1], ids[:, :1], cache=cache)
    mx.eval(step)
    print(f"cached step   : {step.shape} offset={cache.offset} "
          f"finite={bool(mx.all(mx.isfinite(step)).item())}")
    print(f"peak memory   : {mx.get_peak_memory() / 2**30:.2f} GiB")
    if not finite:
        return 1
    print("\nBIND OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
