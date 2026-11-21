"""Stage A of the fast decode path (fast/attn_input.py, fast/read.py) against RC6.

Device tests: they need a Metal GPU, the native pack (``INKLING_TEST_PACK``, default
``~/.mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Quality``) and the stage goldens lane L0 dumps
(``INKLING_TEST_GOLDENS``: RC6-exact inputs and outputs of chosen visits). Each is skipped when
one is absent. Real weights are read lazily for the few layers a test touches.

* every visit kind (window and global, first pass and loop passes): every output bit equals
  RC6's own outputs at 1, 4 and 8 rows, stage A and the MLP read;
* each row of a 4-, 7- and 8-row panel carries the bits of a one-row call on that row's own
  state (row stability);
* the depth read's fixed-order option (``MTPLX_INKLING_DEPTH_READ=fixed``, off by default) is
  row-stable and stays within a bf16 step or two of the exact path.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

PACK = Path(os.environ.get("INKLING_TEST_PACK",
                           str(Path.home() / ".mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Quality")))
GOLD = Path(os.environ.get("INKLING_TEST_GOLDENS", str(Path.home() / ".mtplx/inkling-goldens")))

needs = pytest.mark.skipif(
    not (mx.metal.is_available() and PACK.exists() and GOLD.exists()),
    reason="needs a Metal GPU, the native Inkling pack and the stage goldens",
)

NAMES = ("q", "rel", "sink", "gate_logits", "k_rows", "v_rows", "k_conv_in", "v_conv_in")
GOLDEN_NAME = {"k_conv_in": "A.out.k_taps", "v_conv_in": "A.out.v_taps"}


@lru_cache(maxsize=1)
def _tensors():
    index = json.loads((PACK / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(index.values())):
        out.update(mx.load(str(PACK / shard)))
    return out


@lru_cache(maxsize=16)
def _layer(index):
    from mtplx.models.inkling.model import Layer
    from mtplx.models.inkling.precision import source_modes

    cfg = json.loads((PACK / "config.json").read_text())["text_config"]
    t = _tensors()
    stem = f"model.llm.layers.{index}."
    params = {k[len(stem):]: t[k] for k in t if k.startswith(stem)}
    aq = t[f"model.llm.attn_res.attn_queries.{index - 1}"] if index else None
    mq = t[f"model.llm.attn_res.mlp_queries.{index}"]
    return Layer(params, cfg, index, aq, mq, source_modes(PACK, cfg["num_hidden_layers"])[index])


def _golden(context, width, visit):
    stem = GOLD / f"c{context}_w{width}_v{visit:02d}"
    if not stem.with_suffix(".safetensors").exists():
        pytest.skip(f"golden {stem.name} absent")
    return json.loads(stem.with_suffix(".json").read_text()), mx.load(str(stem.with_suffix(".safetensors")))


def _closed(g, prefix="A"):
    out, i = [], 0
    while f"{prefix}.in.closed.{i}" in g:
        out.append(g[f"{prefix}.in.closed.{i}"])
        i += 1
    return out


def _bits(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _stage(g, meta, rows=None):
    from mtplx.models.inkling.fast.attn_input import attention_input

    lay = _layer(meta["layer"])
    return attention_input(lay, g["A.in.x"], _closed(g), g["state.tail.0"], g["state.tail.1"], g["pos"],
                           g.get("A.in.message.stack"), g.get("A.in.message.queries"))


VISITS = [2, 5, 6, 20, 24, 41, 46]


@needs
@pytest.mark.parametrize("visit", VISITS)
@pytest.mark.parametrize("context", [41, 4142])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_every_visit_matches_rc6_bits(visit, context, width):
    meta, g = _golden(context, width, visit)
    got = _stage(g, meta)
    for name in NAMES:
        assert _bits(getattr(got, name), g[GOLDEN_NAME.get(name, "A.out." + name)]), name


@needs
@pytest.mark.parametrize("visit", [2, 5, 20, 24, 41])
@pytest.mark.parametrize("width", [4, 7, 8])
def test_panel_rows_equal_one_row_calls(visit, width):
    from mtplx.models.inkling.fast.attn_input import attention_input

    meta, g = _golden(4142, 8, visit)
    lay = _layer(meta["layer"])
    g = {k: (v[:, :width] if k in ("A.in.x", "A.in.message.stack") or k.startswith("A.in.closed.") else v)
         for k, v in g.items()}
    x, closed = g["A.in.x"], _closed(g)
    stack, mq = g.get("A.in.message.stack"), g.get("A.in.message.queries")
    panel = _stage(g, meta)
    kt, vt = g["state.tail.0"], g["state.tail.1"]
    for r in range(width):
        one = attention_input(lay, x[:, r:r + 1], [c[:, r:r + 1] for c in closed], kt, vt, g["pos"] + r,
                              None if stack is None else stack[:, r:r + 1], mq)
        for name in NAMES:
            a = getattr(panel, name)
            row = a[:, :, r:r + 1] if name in ("k_rows", "v_rows") else a[:, r:r + 1]
            assert _bits(row, getattr(one, name)), (name, r)
        kt = mx.concatenate([kt, one.k_conv_in], axis=1)[:, -3:]
        vt = mx.concatenate([vt, one.v_conv_in], axis=1)[:, -3:]


@needs
@pytest.mark.parametrize("visit", VISITS)
@pytest.mark.parametrize("width", [1, 4, 8])
def test_mlp_source_matches_rc6_bits(visit, width):
    from mtplx.models.inkling.fast.read import message_rows, mlp_source

    meta, g = _golden(4142, width, visit)
    lay = _layer(meta["layer"])
    stack = g.get("A.in.message.stack")
    rows = None if stack is None else message_rows(stack, g["A.in.message.queries"])
    src = mlp_source(lay, g["D.in.partial"], _closed(g, "D"), rows=rows)
    assert _bits(src, g["D.out.source"])


@needs
@pytest.mark.parametrize("visit", [5, 20, 41])
def test_fixed_order_read_is_row_stable_and_close(visit, monkeypatch):
    from mtplx.models.inkling.fast.read import attend_rows

    meta, g = _golden(4142, 8, visit)
    exact = _stage(g, meta)
    monkeypatch.setenv("MTPLX_INKLING_DEPTH_READ", "fixed")
    fixed = _stage(g, meta)
    for name in ("q", "k_rows", "gate_logits"):
        a, b = getattr(fixed, name).astype(mx.float32), getattr(exact, name).astype(mx.float32)
        assert float(mx.abs(a - b).max().item()) <= 0.25, name
    stack = g.get("A.in.message.stack")
    if stack is not None:
        q = g["A.in.message.queries"]
        panel = attend_rows(q, stack)
        for r in range(stack.shape[1]):
            assert _bits(panel[:, r:r + 1], attend_rows(q, stack[:, r:r + 1])), r
