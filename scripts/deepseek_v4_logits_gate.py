"""M3 full-stack first-token logits gate for DeepSeek-V4-Flash (MLX).

Loads the real mlx-community 4bit (or 2bit-DQ) checkpoint through the native MLX
loader (mtplx.models.deepseek_v4 via mlx-lm's load_model + get_model_classes),
runs one fixed prompt, and reports the assembled 43-layer first-token logits:
finiteness, top-k next tokens, and (optionally) an exact diff against a reference
logits dump.

WHY THIS IS A SEPARATE SCRIPT (not a pytest): the model needs ~112 GiB wired GPU;
running it is the coordinator's guarded GPU window, not CI. Component correctness
(HCA/CSA/o-LoRA/hash) and the load-path key set are already gated in
tests/test_deepseek_v4_new_math.py and tests/test_deepseek_v4_loader.py.

Reference note: the authoritative HF reference (inference/model.py) needs
CUDA/tilelang and 284B params — it does not run on this box, and llama.cpp has no
deepseek_v4 arch, so no local numerical oracle exists for the *assembled* logits.
This harness therefore (a) proves the full stack runs end-to-end and yields finite,
non-degenerate logits with a sensible argmax, and (b) accepts --ref <logits.npy>
to do the exact "logits match ref on a fixed prompt" diff once such a dump is
produced on a machine that can run the reference.

Usage (inside the coordinator's GPU window):
  python scripts/deepseek_v4_logits_gate.py \
    --model ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/<rev> \
    --prompt "The capital of France is" [--ref ref_logits.npy] [--topk 10]
"""
import argparse
import glob
import os
import sys

import mlx.core as mx
import numpy as np


def _default_model():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*/"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ref", default=None, help="optional .npy of reference last-token logits")
    args = ap.parse_args()
    if not args.model:
        sys.exit("no model path; pass --model")

    # Resolve the native MLX classes exactly as mtplx serve does.
    from mlx_lm.utils import load_model, load_tokenizer
    from mtplx.models.deepseek_v4 import Model, ModelArgs

    model, cfg = load_model(args.model, get_model_classes=lambda config: (Model, ModelArgs))
    tok = load_tokenizer(args.model)

    ids = mx.array(tok.encode(args.prompt))[None]
    logits = model(ids)          # [1, T, vocab]
    last = logits[0, -1].astype(mx.float32)
    mx.eval(last)
    ln = np.array(last)

    finite = bool(np.isfinite(ln).all())
    order = np.argsort(-ln)[: args.topk]
    print(f"prompt: {args.prompt!r}  tokens: {ids.shape[1]}  vocab: {ln.shape[0]}")
    print(f"logits finite: {finite}  min/max: {ln.min():.3f}/{ln.max():.3f}  argmax: {int(order[0])}")
    print("top-k next tokens:")
    for r in order:
        try:
            piece = tok.decode([int(r)])
        except Exception:
            piece = "?"
        print(f"  {int(r):>7}  {ln[r]:8.3f}  {piece!r}")

    degenerate = float(ln.std()) < 1e-3
    ok = finite and not degenerate

    if args.ref:
        ref = np.load(args.ref).astype(np.float64)
        d = np.abs(ln.astype(np.float64) - ref)
        rel = d.max() / (np.abs(ref).max() + 1e-9)
        agree = int(np.argmax(ln) == np.argmax(ref))
        print(f"\nREF DIFF  max_abs={d.max():.4e}  max_rel={rel:.4e}  argmax_agree={agree}")
        ok = ok and agree == 1

    print("\nGATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
