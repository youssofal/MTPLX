"""How far each group of weights moves the model's output when rounded to fewer
bits: the measurement a pack recipe is built from.

One load of the full-precision checkpoint does everything the full-precision
model is needed for:

1. the reference scores: every model-written position of the corpus, in the
   format a server's scoring returns (the top-K map keyed by the pack
   tokenizer's strings, and the log-probability of the token that follows), so
   any served pack's scores are compared with them as with another server's;
2. the full next-token distributions at a sample of those positions;
3. one group at a time (a band of layers' routed experts, the shared experts,
   attention, the routers, the embeddings, the output head) rounded to ``b``
   bits as a pack stores it (group size 64), the sample read again, the group
   put back. The exact full-vocabulary KL and the top-1 agreement against the
   recorded distributions are one row per group, width and rounding, beside the
   group's size and how much of it every token reads.

    python -m mtplx.benchmarks.quality.sensitivity SOURCE --corpus FILE --out FILE \\
        --reference-scores FILE --tokenizer PACK

A model process: run it alone, under the machine's load guard.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from mtplx.benchmarks.quality.scoring import SCORE_SCHEMA, tokens_digest, top_map
from mtplx.quant_rounding import ROUNDINGS, clip_for_rounding

GROUP_SIZE = 64
_LAYER = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class Group:
    """Modules chosen the way a forge recipe's ``module_overrides`` chooses them:
    by path suffix, optionally only in some decoder layers."""

    name: str
    suffixes: tuple[str, ...]
    layers: tuple[int, ...] | None = None
    read_fraction: float = 1.0  # share of the group every token reads (routed experts: top-k of all)

    @property
    def routed(self) -> bool:
        return self.suffixes[0].startswith("switch_mlp.")

    def member(self, path: str) -> bool:
        if not path.endswith(self.suffixes):
            return False
        return self.layers is None or _layer(path) in self.layers

    def overrides(self, bits: int) -> list[dict[str, Any]]:
        """This group at ``bits`` as forge ``module_overrides`` entries."""

        entry: dict[str, Any] = {"bits": bits, "group_size": GROUP_SIZE}
        if self.layers is not None:
            entry["layers"] = list(self.layers)
        return [{"suffix": suffix, **entry} for suffix in self.suffixes]


def _layer(path: str) -> int | None:
    match = _LAYER.search(path)
    return int(match.group(1)) if match else None


EXPERTS = ("switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj")


def default_groups(layers: int, experts: int, active: int, band: int) -> list[Group]:
    """The groups a recipe chooses bits for, for a mixture-of-experts decoder."""

    groups = [
        Group(f"routed experts, layers {start}-{min(start + band, layers) - 1}", EXPERTS,
              tuple(range(start, min(start + band, layers))), active / experts if experts else 1.0)
        for start in range(0, layers, band)
    ]
    groups += [
        Group("shared experts", ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj")),
        Group("linear attention", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                                   "linear_attn.in_proj_b", "linear_attn.out_proj")),
        Group("full attention", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")),
        Group("routers", ("mlp.gate", "mlp.shared_expert_gate")),
        Group("embeddings", ("embed_tokens",), read_fraction=0.0),
        Group("output head", ("lm_head",)),
    ]
    return groups


def _quantizable(model: Any) -> dict[str, Any]:
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchLinear

    found = {}
    for path, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, SwitchLinear)) and module.weight.shape[-1] % GROUP_SIZE == 0:
            found[path] = module
    return found


def _rounded(weight: Any, bits: int, rounding: str) -> Any:
    """The weight as a pack built with ``rounding`` would hold it."""

    import mlx.core as mx

    if rounding == "clip":
        weight = clip_for_rounding(weight, bits=bits, group_size=GROUP_SIZE)
    packed, scales, biases = mx.quantize(weight, group_size=GROUP_SIZE, bits=bits)
    return mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=bits).astype(weight.dtype)


def _scored_rows(entry: dict[str, Any]) -> list[int]:
    """Positions whose distribution predicts a model-written token."""

    return [t - 1 for start, end in entry["spans"] for t in range(max(start, 1), end)]


def select_positions(
    corpus: list[dict[str, Any]], positions: int, context: int | None = None
) -> list[tuple[str, list[int], list[int]]]:
    """``(id, token_ids, rows)`` per entry: each entry gives its share of the
    positions, spread evenly over all it wrote (thinking, code, tool calls),
    so no one kind of text dominates the measurement. With ``context``, only
    rows inside each entry's first ``context`` tokens: every pass reads that
    many tokens at most, which bounds the measurement's machine time."""

    scored = [(entry, [r for r in _scored_rows(entry) if context is None or r < context]) for entry in corpus]
    available = sum(len(rows) for _, rows in scored)
    chosen = []
    for entry, rows in scored:
        if not rows:
            continue
        share = min(len(rows), max(1, round(positions * len(rows) / available)))
        chosen.append((entry["id"], entry["token_ids"], [rows[(i * len(rows)) // share] for i in range(share)]))
    return chosen


class Reader:
    """Next-token log-probabilities at chosen positions, a chunk at a time."""

    def __init__(self, model: Any, chunk: int):
        language = getattr(model, "language_model", model)
        self.model = model
        self.body = language.model
        self.head = language.lm_head if hasattr(language, "lm_head") else language.model.embed_tokens.as_linear
        self.chunk = chunk

    def __call__(self, token_ids: list[int], rows: list[int]) -> Iterator[Any]:
        import mlx.core as mx

        cache = self.model.make_cache()
        wanted = sorted(rows)
        cursor = 0
        end_needed = wanted[-1] + 1
        for start in range(0, end_needed, self.chunk):
            end = min(start + self.chunk, end_needed)
            hidden = self.body(mx.array(token_ids[start:end])[None], cache=cache)
            mx.eval(hidden)  # the whole chunk, so the cache it leaves is final before the next one
            take = []
            while cursor < len(wanted) and wanted[cursor] < end:
                take.append(wanted[cursor] - start)
                cursor += 1
            if take:
                logits = self.head(hidden[0, mx.array(take)]).astype(mx.float32)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                mx.eval(logprobs)
                yield logprobs
            del hidden
            mx.clear_cache()


def measure(
    source: Path,
    corpus: list[dict[str, Any]],
    *,
    positions: int,
    context: int | None,
    bits: list[int],
    expert_bits: list[int],
    band: int,
    chunk: int,
    emit: Callable[[dict[str, Any]], None],
    write_score: Callable[[dict[str, Any]], None],
    text_of: Callable[[int], str],
    top_k: int = 64,
    rounding: str = "clip",
    witness: tuple[str, ...] = (),
    extra: tuple[tuple[str, int], ...] = (),
) -> None:
    import mlx.core as mx
    from mlx_lm.utils import load_model

    # The plan first, from the config alone: a mistyped group fails before the load.
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    groups = default_groups(int(text["num_hidden_layers"]), int(text.get("num_experts") or 0),
                            int(text.get("num_experts_per_tok") or 0), band)
    unknown = (set(witness) | {name for name, _ in extra}) - {group.name for group in groups}
    if unknown:
        raise ValueError(f"groups {sorted(unknown)} are not among {[g.name for g in groups]}")
    # Every group at ``rounding``; the witness groups also at the other
    # rounding, so the table shows what the choice of rounding is worth; the
    # extra widths measure a few groups where the recipe would otherwise lean
    # on the rounding law alone (8 bits, for example).
    other = next(r for r in ROUNDINGS if r != rounding)
    plan = [(group, b, rounding) for group in groups for b in bits]
    plan += [(group, b, rounding) for group in groups if group.routed for b in expert_bits]
    plan += [(group, b, other) for group in groups if group.name in witness for b in bits]
    for name, b in extra:
        step = next((group, b, rounding) for group in groups if group.name == name)
        if step not in plan:
            plan.append(step)

    if mx.default_device() == mx.gpu and mx.metal.is_available():
        info = mx.device_info()
        mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
    mx.set_cache_limit(4 * 1024**3)
    started = time.monotonic()
    model, _ = load_model(source)
    reader = Reader(model, chunk)
    modules = _quantizable(model)

    chosen = select_positions(corpus, positions, context)
    loaded_s = time.monotonic() - started
    reference, scored = _reference_pass(reader, corpus, chosen, top_k=top_k, text_of=text_of, write_score=write_score)
    emit({"group": "reference", "scored_positions": scored, "sampled_positions": sum(len(r) for _, _, r in chosen),
          "entries": len(corpus), "load_s": round(loaded_s, 1),
          "reference_s": round(time.monotonic() - started - loaded_s, 1)})

    # The same weights read again: what the stored reference itself costs
    # (float16 storage, any run-to-run difference). Group rows near it say
    # nothing; the recipe is built from the ones well above it.
    t0 = time.monotonic()
    emit({"group": "floor (no weight changed)", **_divergence(reader, chosen, reference),
          "seconds": round(time.monotonic() - t0, 1)})

    for group, b, how in plan:
        members = {path: module for path, module in modules.items() if group.member(path)}
        if not members:
            # A group this architecture does not have (tied output head, no
            # shared experts): said, so a missing row is never a silent gap.
            print(f"{group.name}: no module matched, not measured", flush=True)
            continue
        t0 = time.monotonic()
        originals = {path: module.weight for path, module in members.items()}
        for path, module in members.items():
            module.weight = _rounded(originals[path], b, how)
            mx.eval(module.weight)  # one module's rounding temporaries at a time
        divergence = _divergence(reader, chosen, reference)
        for path, module in members.items():
            module.weight = originals[path]
        del originals
        mx.clear_cache()
        params = sum(math.prod(module.weight.shape) for module in members.values())
        emit({
            "group": group.name,
            "suffixes": list(group.suffixes),
            "layers": list(group.layers) if group.layers is not None else None,
            "read_fraction": group.read_fraction,
            "bits": b,
            "rounding": how,
            "modules": len(members),
            "params": params,
            "params_read_per_token": round(params * group.read_fraction),
            **divergence,
            "seconds": round(time.monotonic() - t0, 1),
        })


def _reference_pass(
    reader: Reader,
    corpus: list[dict[str, Any]],
    chosen: list[tuple[str, list[int], list[int]]],
    *,
    top_k: int,
    text_of: Callable[[int], str],
    write_score: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], int]:
    """One read of the whole corpus: each entry's reference scores, written as
    a server's scoring rows, and the full distributions at the sampled rows,
    returned for the group passes. Also returns how many positions were scored."""

    import mlx.core as mx

    sampled = {entry_id: set(rows) for entry_id, _, rows in chosen}
    reference: dict[str, Any] = {}
    scored = 0
    for entry in corpus:
        rows = sorted(_scored_rows(entry))
        if not rows:
            continue
        token_ids, wanted = entry["token_ids"], sampled.get(entry["id"], set())
        positions: list[list[Any]] = []
        kept = []
        for logprobs in reader(token_ids, rows):
            block = rows[len(positions) : len(positions) + logprobs.shape[0]]
            top = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, :top_k]
            values = mx.take_along_axis(logprobs, top, axis=-1)
            order = mx.argsort(-values, axis=-1)
            top, values = mx.take_along_axis(top, order, axis=-1), mx.take_along_axis(values, order, axis=-1)
            following = mx.array([token_ids[r + 1] for r in block])
            actual = mx.take_along_axis(logprobs, following[:, None], axis=-1)[:, 0]
            picks = [i for i, r in enumerate(block) if r in wanted]
            if picks:
                kept.append(logprobs[mx.array(picks)].astype(mx.float16))
            mx.eval(top, values, actual, *kept[-1:])
            for r, ids, lps, lp in zip(block, top.tolist(), values.tolist(), actual.tolist()):
                positions.append([r + 1, round(lp, 5), top_map(list(zip(ids, lps)), (token_ids[r + 1], lp), text_of)])
        write_score({"schema": SCORE_SCHEMA, "id": entry["id"], "tokens_digest": tokens_digest(token_ids),
                     "top_k": top_k, "positions": positions, "scorer": "full precision, in process"})
        scored += len(positions)
        if kept:
            joined = mx.concatenate(kept)
            if not bool(mx.all(mx.isfinite(joined)).item()):
                raise RuntimeError(f"{entry['id']}: the reference distributions are not finite")
            reference[entry["id"]] = joined
    return reference, scored


def _divergence(reader: Reader, chosen: list[tuple[str, list[int], list[int]]], reference: dict[str, Any]) -> dict[str, Any]:
    """Exact KL from the reference and top-1 agreement over the chosen rows."""

    import mlx.core as mx

    kls: list[float] = []
    agree = 0
    for entry_id, token_ids, rows in chosen:
        offset = 0
        for q in reader(token_ids, rows):
            p = reference[entry_id][offset : offset + q.shape[0]].astype(mx.float32)
            offset += q.shape[0]
            kl = mx.sum(mx.exp(p) * (p - q), axis=-1)
            same = mx.argmax(p, axis=-1) == mx.argmax(q, axis=-1)
            mx.eval(kl, same)
            kls.extend(kl.tolist())
            agree += int(same.sum().item())
    ordered = sorted(kls)
    return {
        "kl_mean": sum(kls) / len(kls),
        "kl_p99": ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)],
        "top1_agreement": agree / len(kls),
        "positions": len(kls),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mtplx.benchmarks.quality.sensitivity", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="full-precision checkpoint folder")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True, help="rows are appended here (JSON lines)")
    parser.add_argument("--positions", type=int, default=6000, help="scored positions to read")
    parser.add_argument("--context", type=int, default=4096, help="read only each entry's first N tokens (0: all)")
    parser.add_argument("--bits", default="4", help="widths every group is rounded to")
    parser.add_argument("--expert-bits", default="", help="further widths for the routed-expert bands")
    parser.add_argument("--band", type=int, default=4, help="layers per routed-expert group")
    parser.add_argument("--chunk", type=int, default=1024, help="tokens per forward")
    parser.add_argument("--rounding", choices=ROUNDINGS, default="clip", help="the rounding every row is measured with")
    parser.add_argument("--witness", default="", help="';'-separated groups also measured with the other rounding")
    parser.add_argument("--extra", default="",
                        help="';'-separated GROUP=BITS: a further width for one group (checks the rounding law)")
    parser.add_argument("--reference-scores", required=True, help="the corpus's reference scores (.jsonl.gz)")
    parser.add_argument("--tokenizer", required=True, help="folder of the tokenizer the served packs use")
    parser.add_argument("--top-k", type=int, default=64, help="top-K kept per position in the reference scores")
    args = parser.parse_args(argv)

    corpus = [json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    out, scores = Path(args.out), Path(args.reference_scores)
    if scores.exists():
        raise SystemExit(f"{scores} exists: the reference is written once, by one load")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # The served packs key their maps by tokenizer.decode([id]); so do these.
    text_of = functools.lru_cache(maxsize=None)(lambda token_id: tokenizer.decode([token_id]))

    def emit(row: dict[str, Any]) -> None:
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    with gzip.open(scores, "wt", encoding="utf-8") as handle:

        def write_score(row: dict[str, Any]) -> None:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        measure(
            Path(args.source),
            corpus,
            positions=args.positions,
            context=args.context or None,
            bits=[int(b) for b in args.bits.split(",") if b],
            expert_bits=[int(b) for b in args.expert_bits.split(",") if b],
            band=args.band,
            chunk=args.chunk,
            emit=emit,
            write_score=write_score,
            text_of=text_of,
            top_k=args.top_k,
            rounding=args.rounding,
            witness=tuple(name.strip() for name in args.witness.split(";") if name.strip()),
            extra=tuple((spec.rpartition("=")[0].strip(), int(spec.rpartition("=")[2]))
                        for spec in args.extra.split(";") if spec.strip()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
