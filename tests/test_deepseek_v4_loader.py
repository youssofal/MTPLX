"""M3 loader gates for deepseek_v4.

Two layers of evidence:
  * Always-on (synthetic): instantiate the FULL 43-layer structure at tiny
    per-unit dims and assert the module tree matches the V4 spec (per-layer
    compressor/indexer/hash presence, HC blocks, grouped o-LoRA, head).
  * Cache-gated (real weights): when the mlx-community 4bit checkpoint is present
    in the HF cache, assert the quantised param-key set exactly equals the
    checkpoint's, and run the new-math components on real dequantised tensors.

The assembled 43-layer first-token logits gate runs in a GPU window (the model
needs ~112 GiB wired); see scripts/deepseek_v4_logits_gate.py.
"""

import glob
import importlib.util
import json
import os
import sys

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # CPU-pinned by design, but the pin must stay test-scoped: a module-level
    # set_default_device leaks into every later-collected module (pytest
    # imports all test modules before running any) and flips the engine's
    # Metal bit-exactness suites onto CPU fallbacks process-wide.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_loader_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_loader_undertest"] = D
_spec.loader.exec_module(D)


def _tiny_full_args(**over):
    base = dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=43,
        num_hash_layers=3,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        moe_intermediate_size=8,
        n_routed_experts=256,
        num_experts_per_tok=6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        sliding_window=8,
    )
    base.update(over)
    return D.ModelArgs(**base)


def test_module_tree_matches_v4_spec():
    args = _tiny_full_args()
    model = D.Model(args)
    keys = {k for k, _ in tree_flatten(model.parameters())}

    # top-level
    for k in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.hc_head.fn",
        "model.hc_head.base",
        "model.hc_head.scale",
        "lm_head.weight",
    ):
        assert k in keys, k

    cr = args.compress_ratios
    for i in range(args.num_hidden_layers):
        p = f"model.layers.{i}"
        # every layer: attention low-ranks, HC blocks, MoE gate + experts
        for suf in (
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.attn_sink",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn_hc.fn",
            "attn_hc.base",
            "attn_hc.scale",
            "ffn_hc.fn",
            "ffn_hc.base",
            "ffn_hc.scale",
            "ffn.gate.weight",
            "ffn.switch_mlp.gate_proj.weight",
            "ffn.shared_experts.gate_proj.weight",
        ):
            assert f"{p}.{suf}" in keys, f"{p}.{suf}"
        # hash layers carry tid2eid; score layers carry the noaux bias
        if i < args.num_hash_layers:
            assert f"{p}.ffn.gate.tid2eid" in keys
            assert f"{p}.ffn.gate.e_score_correction_bias" not in keys
        else:
            assert f"{p}.ffn.gate.e_score_correction_bias" in keys
            assert f"{p}.ffn.gate.tid2eid" not in keys
        # compressor present iff compress_ratio != 0; indexer iff ratio == 4
        has_comp = f"{p}.attn.compressor.wkv.weight" in keys
        has_index = f"{p}.attn.indexer.wq_b.weight" in keys
        assert has_comp == (cr[i] != 0), (i, cr[i], has_comp)
        assert has_index == (cr[i] == 4), (i, cr[i], has_index)


def test_sanitize_flattens_0731_grouped_wo_a_storage_without_reordering():
    """Collapse only the checkpoint's explicit o-LoRA group/rank row axes."""
    args = _tiny_full_args(
        num_hidden_layers=1,
        num_hash_layers=1,
        compress_ratios=[0],
        num_nextn_predict_layers=0,
    )
    model = D.Model(args)
    prefix = "model.layers.0.attn.wo_a"
    grouped = {
        f"{prefix}.weight": mx.arange(2 * 4 * 6).reshape(2, 4, 6),
        f"{prefix}.scales": mx.arange(2 * 4 * 2).reshape(2, 4, 2),
        f"{prefix}.biases": mx.arange(2 * 4 * 2).reshape(2, 4, 2) + 100,
    }

    sanitized = model.sanitize(grouped)

    for suffix, value in grouped.items():
        assert sanitized[suffix].shape == (8, value.shape[-1])
        assert bool(mx.array_equal(sanitized[suffix], value.reshape(8, -1)))

    flat = mx.arange(8 * 6).reshape(8, 6)
    assert model.sanitize({f"{prefix}.weight": flat})[f"{prefix}.weight"] is flat


def test_sanitize_rejects_malformed_0731_grouped_wo_a_storage():
    args = _tiny_full_args(
        num_hidden_layers=1,
        num_hash_layers=1,
        compress_ratios=[0],
        num_nextn_predict_layers=0,
    )
    model = D.Model(args)

    with pytest.raises(ValueError, match="invalid grouped 0731 o-LoRA storage"):
        model.sanitize({"model.layers.0.attn.wo_a.weight": mx.zeros((1, 8, 6))})


def _find_snapshot():
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/*/"
        )
    )
    for h in hits:
        if os.path.exists(
            os.path.join(h, "model.safetensors.index.json")
        ) and os.path.exists(os.path.join(h, "config.json")):
            return h
    return None


_SNAP = _find_snapshot()
_needs_ckpt = pytest.mark.skipif(
    _SNAP is None, reason="mlx-community 4bit checkpoint not in HF cache"
)


def _args_from_config(cfg):
    return D.ModelArgs(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_hash_layers=cfg["num_hash_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        qk_rope_head_dim=cfg["qk_rope_head_dim"],
        q_lora_rank=cfg["q_lora_rank"],
        o_lora_rank=cfg["o_lora_rank"],
        o_groups=cfg["o_groups"],
        moe_intermediate_size=cfg["moe_intermediate_size"],
        n_routed_experts=cfg["n_routed_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        compress_ratios=cfg["compress_ratios"],
        compress_rope_theta=cfg["compress_rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_scaling=cfg.get("rope_scaling"),
    )


@_needs_ckpt
def test_real_checkpoint_key_set_is_exact():
    cfg = json.load(open(os.path.join(_SNAP, "config.json")))
    ckpt = set(
        json.load(open(os.path.join(_SNAP, "model.safetensors.index.json")))[
            "weight_map"
        ]
    )
    # tiny per-unit dims, real structural counts -> identical key names
    args = _tiny_full_args(
        num_hidden_layers=cfg["num_hidden_layers"],
        num_hash_layers=cfg["num_hash_layers"],
        n_routed_experts=cfg["n_routed_experts"],
        o_groups=cfg["o_groups"],
        compress_ratios=cfg["compress_ratios"],
        vocab_size=64,
    )
    model = D.Model(args)
    quantizable = {n for n, m in model.named_modules() if hasattr(m, "to_quantized")}
    expected = set()
    for path, _ in tree_flatten(model.parameters()):
        if path.endswith(".weight"):
            stem = path[: -len(".weight")]
            if stem in quantizable and f"{stem}.scales" in ckpt:
                expected |= {f"{stem}.weight", f"{stem}.scales"}
                if f"{stem}.biases" in ckpt:
                    expected.add(f"{stem}.biases")
                continue
        expected.add(path)
    assert expected == ckpt, {
        "missing_from_model": sorted(ckpt - expected)[:10],
        "extra_in_model": sorted(expected - ckpt)[:10],
    }


@_needs_ckpt
def test_real_weight_components_match_oracle():
    import numpy as np

    cfg = json.load(open(os.path.join(_SNAP, "config.json")))
    wmap = json.load(open(os.path.join(_SNAP, "model.safetensors.index.json")))[
        "weight_map"
    ]
    qcfg = cfg["quantization"]
    args = _args_from_config(cfg)
    shards = {}

    def raw(k):
        fn = wmap[k]
        if fn not in shards:
            shards[fn] = mx.load(os.path.join(_SNAP, fn))
        return shards[fn][k]

    def qp(path):
        q = qcfg.get(path)
        if isinstance(q, dict):
            return q["group_size"], q["bits"], q.get("mode", "affine")
        return qcfg["group_size"], qcfg["bits"], qcfg.get("mode", "affine")

    def dense(stem):
        w = raw(f"{stem}.weight")
        if f"{stem}.scales" in wmap:
            gs, bits, mode = qp(stem)
            b = raw(f"{stem}.biases") if f"{stem}.biases" in wmap else None
            w = mx.dequantize(
                w, raw(f"{stem}.scales"), b, group_size=gs, bits=bits, mode=mode
            )
        return w

    def npf(a):
        return np.array(a.astype(mx.float32))

    H, hc, eps = args.hidden_size, args.hc_mult, args.hc_eps
    rng = np.random.default_rng(0)

    # HC on real attn_hc (layer 0)
    hyper = D.HyperConnection(H, hc, eps)
    hyper._iters = args.hc_sinkhorn_iters
    hyper.fn = raw("model.layers.0.attn_hc.fn")
    hyper.base = raw("model.layers.0.attn_hc.base")
    hyper.scale = raw("model.layers.0.attn_hc.scale")
    h = mx.array(rng.standard_normal((1, 3, hc, H)).astype(np.float32))
    y, post, comb = hyper.pre(h)
    mx.eval(y, comb)
    assert bool(mx.all(mx.isfinite(y)).item())
    # comb approximately doubly-stochastic
    assert abs(float(npf(comb).sum(-2).mean()) - 1.0) < 5e-2

    # o-LoRA on real wo_a (layer 3, quantised affine) -> matches grouped einsum
    attn = D.DeepseekV4Attention(args, 3)
    attn.wo_a.weight = dense("model.layers.3.attn.wo_a")
    attn.wo_b.weight = dense("model.layers.3.attn.wo_b")
    o = mx.array(
        rng.standard_normal((1, 2, args.num_attention_heads * args.head_dim)).astype(
            np.float32
        )
    )
    xo = attn._o_lora(o)
    mx.eval(xo)
    g, r = args.o_groups, args.o_lora_rank
    per = args.num_attention_heads * args.head_dim // g
    o3 = np.einsum(
        "bsgp,grp->bsgr",
        npf(o).reshape(1, 2, g, per),
        npf(attn.wo_a.weight).reshape(g, r, per),
    ).reshape(1, 2, g * r)
    ref = o3 @ npf(attn.wo_b.weight).T
    assert np.allclose(npf(xo), ref, rtol=2e-4, atol=2e-5)

    # gate score (real layer 3): valid top-k, weights sum to route_scale
    gate = D.MoEGate(args, 3)
    gate.weight = raw("model.layers.3.ffn.gate.weight")
    gate.e_score_correction_bias = raw(
        "model.layers.3.ffn.gate.e_score_correction_bias"
    )
    idx, w = gate(mx.array(rng.standard_normal((5, H)).astype(np.float32)), None)
    mx.eval(idx, w)
    idxn = np.array(idx)
    assert idxn.shape == (5, args.num_experts_per_tok)
    assert idxn.min() >= 0 and idxn.max() < args.n_routed_experts
    assert np.allclose(npf(w).sum(-1), args.routed_scaling_factor, rtol=1e-4)
