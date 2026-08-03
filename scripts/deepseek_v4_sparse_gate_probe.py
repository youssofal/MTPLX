"""Root-cause probe for a single-position streamed-vs-one-shot disagreement.

``scripts/deepseek_v4_smoke_generate.py --gate-a`` on the 3318-token sparse run
reported 127/128, diverging only at generated index 92.  Two explanations need
different owners:

  H1  the streaming state machine is wrong in the sparse regime (a backend bug
      in mtplx/models/deepseek_v4.py), or
  H2  both paths compute the same function and the argmax flipped on a near-tie:
      the two paths reduce over different shapes (one-shot pools every window in
      one call, streaming pools them in prefill chunks and then one token at a
      time), which is exact in fp32 on CPU -- what the unit tests gate -- but not
      bit-exact at real dims on Metal in bf16.

An isolated flip that does not propagate already argues for H2, but "argues for"
is not evidence.  This measures the three quantities that separate them, at the
one query position that disagreed (absolute position 3409, the query whose
argmax is generated index 92), off a single load:

  1. **Output margin.**  Both paths' full logit rows at that position: top-1,
     top-2, and the gap.  Under H2 the gap is at the noise floor and the two
     contenders are the same two tokens in both paths.  Under H1 the rows differ
     structurally and the gap is ordinary.

  2. **Indexer selection.**  Per ratio-4 layer, the set of compressed rows each
     path selected for that query.  Under H2 the sets are equal, or differ by a
     row or two whose score sits on the top-k cut.  Under H1 they differ widely,
     or the row counts themselves disagree.

  3. **Selection headroom.**  Per ratio-4 layer, the score gap across the
     ``index_topk`` boundary (rank 512 vs 513) and the max score difference
     between the paths.  A gap smaller than the inter-path score noise is a
     boundary that float noise can flip -- the mechanism H2 names, measured
     rather than asserted.

Runs in the guarded MLX window (bench/laguna/run_guarded.py); the checkpoint is
~90 GiB.  The one-shot forward over ~3.4K tokens is the memory high-water mark
(~102 GiB observed), so nothing else may be resident.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


# Filled per phase by the Indexer probes: {phase: {layer_index: array}}
SCORES: dict[str, dict[int, np.ndarray]] = {}
SELECTED: dict[str, dict[int, np.ndarray]] = {}
_CAPTURE = {"on": False, "phase": ""}


def _default_model() -> str | None:
    hits = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/"
                "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/*/"
            )
        )
    )
    return hits[0] if hits else None


def _gib(n: int) -> float:
    return n / (1024**3)


def _install_probes(model, module):
    """Wrap Indexer.scores/__call__ to stash the last query's row per layer."""
    layer_of = {}
    for i, layer in enumerate(model.layers):
        indexer = getattr(layer.attn, "indexer", None)
        if indexer is not None:
            # The loaded model must be built from the module being patched, or
            # the probes would silently never fire and every set would compare
            # equal by vacuity.
            assert isinstance(indexer, module.Indexer), (
                "loaded Indexer is not the class being patched; the load path "
                "resolved a different module object"
            )
            layer_of[id(indexer)] = i
    assert layer_of, "no ratio-4 indexer found on the loaded model"

    original_scores = module.Indexer.scores
    original_call = module.Indexer.__call__

    def scores_probe(self, x, qr, positions, rows):
        out = original_scores(self, x, qr, positions, rows)
        if _CAPTURE["on"]:
            row = out[0, -1].astype(mx.float32)
            mx.eval(row)
            SCORES.setdefault(_CAPTURE["phase"], {})[layer_of[id(self)]] = np.array(row)
        return out

    def call_probe(self, x, qr, positions, rows):
        out = original_call(self, x, qr, positions, rows)
        if _CAPTURE["on"]:
            row = out[0, -1]
            mx.eval(row)
            SELECTED.setdefault(_CAPTURE["phase"], {})[layer_of[id(self)]] = np.array(row)
        return out

    module.Indexer.scores = scores_probe
    module.Indexer.__call__ = call_probe
    return layer_of


def _eval_cache(cache, *extra) -> None:
    live = [a for layer in cache for a in layer.state if a is not None]
    mx.eval(*extra, *live)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--index", type=int, default=92,
                    help="generated index that disagreed")
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_path = Path(os.path.expanduser(args.model)).resolve()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_lm.utils import load_config

    from mtplx.models import deepseek_v4 as dsv4
    from mtplx.runtime import _load_base_model

    config = load_config(model_path)
    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    print(f"[probe] loaded in {time.perf_counter() - t0:.1f}s  "
          f"active={_gib(int(mx.get_active_memory())):.2f} GiB")
    sys.stdout.flush()

    layer_of = _install_probes(model, dsv4)
    ratio4_layers = sorted(layer_of.values())
    print(f"[probe] ratio-4 layers with an indexer: {len(ratio4_layers)} "
          f"-> {ratio4_layers}")

    receipt = json.loads(Path(args.run_json).read_text())
    prompt_ids = tokenizer.encode(receipt["prompt"])
    generated = receipt["generated_token_ids"]
    assert len(prompt_ids) == receipt["prompt_tokens"], "tokenizer drift"
    k = args.index
    n_prompt = len(prompt_ids)
    # generated[0] comes from the prefill's last query (position n_prompt-1);
    # generated[j>0] from the decode step whose input is generated[j-1] at
    # position n_prompt + j - 1.  So generated[k] is the query at n_prompt+k-1.
    query_position = n_prompt + k - 1
    print(f"[probe] generated index {k} is the query at absolute position "
          f"{query_position}; streamed emitted {generated[k]}")
    sys.stdout.flush()

    # ---- phase 1: streaming (chunked prefill + teacher-forced decode) -------
    cache = model.make_cache()
    ids = mx.array(prompt_ids)[None]
    t0 = time.perf_counter()
    for start in range(0, n_prompt, args.prefill_chunk):
        logits = model(ids[:, start : start + args.prefill_chunk], cache=cache)
        if start + args.prefill_chunk < n_prompt:
            _eval_cache(cache, logits)
            del logits
    replay = [int(mx.argmax(logits[:, -1], axis=-1).item())]
    del logits
    mismatches = []
    for j in range(1, k + 1):
        if j == k:
            _CAPTURE.update(on=True, phase="streaming")
        token = mx.array([[generated[j - 1]]])
        logits = model(token, cache=cache)
        if j == k:
            streamed_row = np.array(logits[0, -1].astype(mx.float32))
            _CAPTURE["on"] = False
        got = int(mx.argmax(logits[:, -1], axis=-1).item())
        replay.append(got)
        if got != generated[j]:
            mismatches.append((j, generated[j], got))
        del logits
    streaming_seconds = time.perf_counter() - t0
    n_comp_stream = [cache[i].n_compressed for i in ratio4_layers]
    print(f"[probe] streaming replay {streaming_seconds:.1f}s  "
          f"offset={cache[0].offset}  n_comp(ratio-4)={sorted(set(n_comp_stream))}")
    print(f"[probe] replay reproduced the receipt for indices 0..{k - 1}: "
          f"{not mismatches}"
          + (f"  MISMATCHES {mismatches[:5]}" if mismatches else ""))
    sys.stdout.flush()
    del cache
    mx.clear_cache()

    # ---- phase 2: one-shot (cache=None) over the same prefix ---------------
    sequence = list(prompt_ids) + list(generated[:k])
    assert len(sequence) == query_position + 1, "prefix length mismatch"
    _CAPTURE.update(on=True, phase="one_shot")
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    logits = model(mx.array(sequence)[None])
    one_shot_row = np.array(logits[0, -1].astype(mx.float32))
    mx.eval(logits)
    one_shot_seconds = time.perf_counter() - t0
    _CAPTURE["on"] = False
    peak = int(mx.get_peak_memory())
    del logits
    print(f"[probe] one-shot over {len(sequence)} tokens {one_shot_seconds:.1f}s  "
          f"peak={_gib(peak):.2f} GiB")
    sys.stdout.flush()

    # ---- 1. output margin --------------------------------------------------
    def top2(row):
        order = np.argsort(-row)
        return int(order[0]), float(row[order[0]]), int(order[1]), float(row[order[1]])

    s_t1, s_v1, s_t2, s_v2 = top2(streamed_row)
    o_t1, o_v1, o_t2, o_v2 = top2(one_shot_row)
    contenders = sorted({s_t1, s_t2, o_t1, o_t2})
    print("\n=== 1. OUTPUT MARGIN at the disagreeing position ===")
    print(f"streaming : top1={s_t1} ({s_v1:.5f})  top2={s_t2} ({s_v2:.5f})  "
          f"margin={s_v1 - s_v2:.6f}")
    print(f"one-shot  : top1={o_t1} ({o_v1:.5f})  top2={o_t2} ({o_v2:.5f})  "
          f"margin={o_v1 - o_v2:.6f}")
    print(f"same two contenders in both paths: "
          f"{ {s_t1, s_t2} == {o_t1, o_t2} }  ids={contenders}")
    print(f"logit row: max|diff|={np.max(np.abs(streamed_row - one_shot_row)):.6f}  "
          f"row std={float(np.std(one_shot_row)):.4f}  "
          f"mean|diff|={np.mean(np.abs(streamed_row - one_shot_row)):.6f}")
    for t in contenders:
        print(f"   token {t}: streaming {streamed_row[t]:.5f}  "
              f"one-shot {one_shot_row[t]:.5f}  diff {streamed_row[t] - one_shot_row[t]:+.6f}")

    # ---- 2/3. indexer selection + headroom ---------------------------------
    print("\n=== 2/3. INDEXER SELECTION at that query, per ratio-4 layer ===")
    print(f"{'layer':>5} {'n_comp':>7} {'|sel|s':>7} {'|sel|1':>7} {'symdiff':>8} "
          f"{'topk_gap':>10} {'score_maxdiff':>14}")
    rows = []
    for layer in ratio4_layers:
        sel_s = SELECTED.get("streaming", {}).get(layer)
        sel_o = SELECTED.get("one_shot", {}).get(layer)
        sc_s = SCORES.get("streaming", {}).get(layer)
        sc_o = SCORES.get("one_shot", {}).get(layer)
        if sel_s is None or sel_o is None:
            print(f"{layer:>5}   (indexer inactive in at least one path)")
            continue
        a = set(np.flatnonzero(sel_s).tolist())
        b = set(np.flatnonzero(sel_o).tolist())
        symdiff = len(a ^ b)
        ordered = np.sort(sc_o)[::-1]
        topk = int(min(len(ordered), model.layers[layer].attn.indexer.index_topk))
        gap = float(ordered[topk - 1] - ordered[topk]) if len(ordered) > topk else float("nan")
        maxdiff = float(np.max(np.abs(sc_s - sc_o)))
        rows.append({
            "layer": layer, "n_comp": int(sel_s.shape[-1]),
            "selected_streaming": len(a), "selected_one_shot": len(b),
            "symmetric_difference": symdiff,
            "topk_boundary_gap": gap, "score_max_abs_diff": maxdiff,
            "gap_smaller_than_noise": bool(gap < maxdiff),
        })
        print(f"{layer:>5} {sel_s.shape[-1]:>7} {len(a):>7} {len(b):>7} {symdiff:>8} "
              f"{gap:>10.6f} {maxdiff:>14.6f}"
              + ("   <- gap < noise" if gap < maxdiff else ""))

    flipped = [r for r in rows if r["symmetric_difference"]]
    fragile = [r for r in rows if r["gap_smaller_than_noise"]]
    print(f"\nlayers whose selected row set differs : {len(flipped)}/{len(rows)}")
    print(f"layers whose top-k boundary gap is below the inter-path score noise: "
          f"{len(fragile)}/{len(rows)}")

    result = {
        "harness": "scripts/deepseek_v4_sparse_gate_probe.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "model_path": str(model_path),
        "source_run": str(args.run_json),
        "generated_index": k,
        "query_position": query_position,
        "replay_reproduced_receipt": not mismatches,
        "replay_mismatches": mismatches,
        "one_shot_peak_gib": _gib(peak),
        "output_margin": {
            "streaming": {"top1": s_t1, "top1_logit": s_v1, "top2": s_t2,
                          "top2_logit": s_v2, "margin": s_v1 - s_v2},
            "one_shot": {"top1": o_t1, "top1_logit": o_v1, "top2": o_t2,
                         "top2_logit": o_v2, "margin": o_v1 - o_v2},
            "same_contenders": {s_t1, s_t2} == {o_t1, o_t2},
            "logit_row_max_abs_diff": float(np.max(np.abs(streamed_row - one_shot_row))),
            "logit_row_std": float(np.std(one_shot_row)),
        },
        "indexer_layers": rows,
        "layers_with_differing_selection": len(flipped),
        "layers_with_gap_below_noise": len(fragile),
    }
    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(f"\nreceipt: {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
