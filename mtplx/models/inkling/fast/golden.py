"""Dump real stage inputs and outputs of chosen layer visits for offline kernel tests.

    python -m mtplx.models.inkling.fast.golden PACK OUT_DIR [--contexts 41,4142] [--widths 1,4,8]
        [--visits 2,5,6,20,24,41,46]

For each context and panel width it prefills the reference model (512-row chunks), runs
one reference panel and writes, per chosen visit, ``<OUT_DIR>/c<context>_w<width>_v<visit>
.safetensors`` holding the committed state the visit reads (keys/values with ``newest`` and
``count``, the four convolution tails, ``pos``) and every stage's inputs and outputs, with a
``.json`` beside it naming the layer, pass, kind and shapes. Load with ``mx.load``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx


def _flatten(prefix, value, out):
    if value is None:
        return
    if isinstance(value, mx.array):
        out[prefix] = value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _flatten(f"{prefix}.{i}", v, out)
    elif isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif hasattr(value, "__dataclass_fields__"):
        for k in value.__dataclass_fields__:
            _flatten(f"{prefix}.{k}", getattr(value, k), out)
    elif hasattr(value, "stack") and hasattr(value, "queries"):  # step.Message
        _flatten(f"{prefix}.stack", value.stack, out)
        _flatten(f"{prefix}.queries", value.queries, out)
        _flatten(f"{prefix}.read", value.read, out)


class Recorder:
    """``record`` hook for ``decode_panel``: keeps the stage tensors of chosen visits."""

    def __init__(self, visits):
        self.visits = set(visits)
        self.tensors = {v: {} for v in visits}

    def __call__(self, visit, stage, inputs, outputs):
        if visit in self.visits:
            _flatten(f"{stage}.in", inputs, self.tensors[visit])
            _flatten(f"{stage}.out", outputs, self.tensors[visit])


def visit_kind(model, visit):
    index = model.visit_order[visit]
    layer = model.layers[index]
    before = model.visit_order[:visit].count(index)
    return {"layer": index, "pass": before + 1, "sliding": layer.sliding, "boundary": layer.boundary,
            "fp8": getattr(layer.experts.gate_up, "mode", "") == "mxfp8"}


def state_tensors(state, visit):
    out = {"pos": state.pos}
    if state.local[visit]:
        keys, values, newest, count = state.window_view(visit)
        out.update({"state.newest": newest, "state.count": count})
    else:
        keys, values, count = state.global_view(visit)
        keys, values = keys[:, :, : max(1, state.length)], values[:, :, : max(1, state.length)]
        out["state.count"] = count
    out.update({"state.keys": keys, "state.values": values})
    for slot in range(4):
        out[f"state.tail.{slot}"] = state.tail(visit, slot)
    return out


def main(argv):
    from mtplx.runtime import load

    from .state import DecodeState
    from .step import decode_panel

    pack, out_dir = Path(argv[0]), Path(argv[1])
    args = dict(zip(argv[2::2], argv[3::2]))
    contexts = [int(c) for c in args.get("--contexts", "41,4142").split(",")]
    widths = [int(w) for w in args.get("--widths", "1,4,8").split(",")]
    visits = [int(v) for v in args.get("--visits", "2,5,6,20,24,41,46").split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = load(pack, mtp=False)
    model = rt.model
    trunk = rt.trunk_entries
    text = "".join(p.read_text() for p in sorted(Path(__file__).resolve().parents[1].glob("*.py")))
    ids = rt.tokenizer.encode(text)
    while len(ids) < max(contexts) + max(widths):
        ids = ids + ids
    written = []
    for context in contexts:
        cache = rt.make_cache()[:trunk]
        for start in range(0, context, 512):
            rt.forward_ar(mx.array([ids[start:min(context, start + 512)]]), cache=cache, emit_logits=False)
            mx.eval([c.state for c in cache])
        state = DecodeState(model, cache)
        state.prepare()
        mx.eval(state.arrays())
        for width in widths:
            recorder = Recorder(visits)
            block = mx.array(ids[context:context + width], mx.int32)
            logits, captures, deltas = decode_panel(model, block, state, record=recorder)
            mx.eval(logits, captures, [d.arrays() for d in deltas])
            for visit in visits:
                tensors = {**state_tensors(state, visit), **recorder.tensors[visit]}
                d = deltas[visit]
                tensors.update({"delta.k_rows": d.keys, "delta.v_rows": d.values,
                                **{f"delta.taps.{i}": t for i, t in enumerate(d.taps)}})
                mx.eval(list(tensors.values()))
                name = f"c{context}_w{width}_v{visit:02d}"
                mx.save_safetensors(str(out_dir / f"{name}.safetensors"), tensors)
                meta = {"context": context, "width": width, "visit": visit, "tokens": ids[context:context + width],
                        **visit_kind(model, visit),
                        "tensors": {k: [list(v.shape), str(v.dtype)] for k, v in sorted(tensors.items())}}
                (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
                written.append(name)
            del recorder
    print(json.dumps({"written": len(written), "dir": str(out_dir)}))


if __name__ == "__main__":
    main(sys.argv[1:])
