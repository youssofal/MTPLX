"""The prompt chunk's head step as one launch (fast.prompt_glue.heads) against model.Layer.attention's own
statements: the query, relative and key clips, the value norm and, on global visits, the position scale,
bit for bit at chunk widths and offsets."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the head kernel needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.fast import prompt_glue as G  # noqa: E402

CFG = {"qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "use_value_norm_no_gamma": True,
       "log_scaling_alpha": 0.1, "log_scaling_n_floor": 8192}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def statements(layer, q_lin, r_lin, k_conv, v_conv, offset):
    cfg = layer.cfg
    rows = q_lin.shape[1]
    q = M.clip_heads(q_lin.reshape(1, rows, 24, 128), cfg["qk_l2_clip_threshold"], layer.q_scale)
    rel = M.clip_heads(r_lin.reshape(1, rows, 24, -1), cfg["rel_l2_clip_threshold"], layer.r_scale)
    k = M.clip_heads(k_conv.reshape(1, rows, -1, 128), cfg["qk_l2_clip_threshold"])
    v = M.norm(v_conv.reshape(1, rows, -1, 128), None, layer.eps).astype(mx.bfloat16)
    if not layer.sliding:
        pos = mx.arange(rows) + offset + 1
        tau = 1 + cfg["log_scaling_alpha"] * mx.log(mx.maximum(pos / cfg["log_scaling_n_floor"], 1))
        q = (q.astype(mx.float32) * tau[:, None, None]).astype(q.dtype)
        rel = (rel.astype(mx.float32) * tau[:, None, None]).astype(rel.dtype)
    return q, rel, k, v


@pytest.mark.parametrize("rows", [33, 700, 4096])
@pytest.mark.parametrize("sliding,offset", [(True, 0), (False, 0), (False, 12345), (False, 70000)])
def test_the_head_step_has_the_layers_bits(monkeypatch, rows, sliding, offset):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_KERNELS", raising=False)
    ks = mx.random.split(mx.random.key(rows + offset + sliding), 6)
    layer = SimpleNamespace(cfg=CFG, eps=1e-6, sliding=sliding,
                            q_scale=(1 + 0.3 * mx.random.normal((24, 128), key=ks[0])).astype(mx.bfloat16),
                            r_scale=(1 + 0.3 * mx.random.normal((24, 16), key=ks[1])).astype(mx.bfloat16))
    # magnitudes either side of the clip limits, so both branches of every minimum are taken
    q_lin = (mx.random.normal((1, rows, 3072), key=ks[2]) * 2.5).astype(mx.bfloat16)
    r_lin = (mx.random.normal((1, rows, 384), key=ks[3]) * 2.5).astype(mx.bfloat16)
    k_conv = (mx.random.normal((1, rows, 1024), key=ks[4]) * 2.5).astype(mx.bfloat16)
    v_conv = (mx.random.normal((1, rows, 1024), key=ks[5]) * 2.5).astype(mx.bfloat16)
    want = statements(layer, q_lin, r_lin, k_conv, v_conv, offset)
    got = G.heads(layer, q_lin, r_lin, k_conv, v_conv, offset)
    assert got is not None
    mx.eval(want, got)
    for name, a, b in zip(("q", "rel", "k", "v"), got, want):
        assert same(a, b), name
