"""Edge0 skeleton -> qwen4_exp shell conversion (mtplx/edge0/shell.py).

A tiny synthetic skeleton is built here from model-convention ground truth by
a test-side FORWARD transform written from the llama.cpp ``qwen4exp`` layout
(tiled GDN value heads, ``ssm_a = -exp(A_log)``, split indexer projection,
``[C, K]`` conv kernels, ``1 + w`` norms as F32, Q8_0 projections), so each
inverse in ``shell.py`` is checked against an independent implementation.
Real-folder tests run only with ``MTPLX_EDGE0_MODEL`` set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mtplx.edge0 import shell
from mtplx.edge0.gguf_lite import GGUFFile
from tests import edge0_synth as syn

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

REAL = os.environ.get("MTPLX_EDGE0_MODEL")
real_only = pytest.mark.skipif(not REAL, reason="MTPLX_EDGE0_MODEL not set")

LM = "language_model.model."
F32, F16, Q8, BF16 = 0, 1, 8, 30

# tiny geometry: 3 GDN layers + 1 QSA layer, PLE on layer 1
G = dict(H=64, L=4, NH=2, NKV=1, HD=32, KH=2, VH=6, DK=32, DV=32, CK=4, HC=2, R=32, IH=2, IKV=1, IHD=32,
         E=4, FF=32, V=96, PD=64)
SIZES = [101, 103, 107, 109]
MULTS = [23703573157769, 20109073645365, 8052911324071]


# -- test-side encoders -----------------------------------------------------------


def bf16_bytes(a: np.ndarray) -> bytes:
    a = np.asarray(a, np.float32)
    assert not np.any(a.view(np.uint32) & 0xFFFF), "ground truth must be bf16 numbers"
    return (a.view(np.uint32) >> 16).astype("<u2").tobytes()


def bf16_round(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def tile_heads(a: np.ndarray, k: int, r: int, width: int, axis: int = 0) -> np.ndarray:
    """llama.cpp's grouped -> tiled value-head order (convert_hf_to_gguf ``_reorder_v_heads``), along ``axis``."""
    a = np.moveaxis(a, axis, 0)
    shp = a.shape
    a = a.reshape((k, r, width) + shp[1:]).swapaxes(0, 1).reshape(shp)
    return np.moveaxis(a, 0, axis)


class Q8T:
    """A Q8_0 tensor in model layout: int8 codes [R, C] and float16 d [R, C/32]."""

    def __init__(self, rng, rows, cols):
        self.q = rng.integers(-127, 128, (rows, cols), dtype=np.int8)
        self.q[0, :4] = (-128, 127, 0, -1)  # extreme codes
        self.d = (rng.uniform(1e-4, 2e-2, (rows, cols // 32)) * rng.choice([-1, 1], (rows, cols // 32))).astype(np.float16)

    def values(self):
        r, c = self.q.shape
        return (self.q.reshape(r, -1, 32).astype(np.float32) * self.d.astype(np.float32)[..., None]).reshape(r, c)

    def raw(self, q=None, d=None):
        return syn.pack_q8_0(self.q if q is None else q, (self.d if d is None else d).reshape(-1))


def build(tmp_path, *, dv=G["DV"], extra=None, drop=None, retype=None, seed=0):
    """Writes a synthetic skeleton; returns (path, ground truth {target: values or Q8T}, metadata)."""
    rng = np.random.default_rng(seed)
    g = dict(G, DV=dv)
    H, HCH = g["H"], g["H"] * g["HC"]
    kd, vd = g["KH"] * g["DK"], g["VH"] * dv
    rr = g["VH"] // g["KH"]
    types = ["linear_attention"] * 3 + ["full_attention"]
    truth: dict = {}
    tensors = []

    def bf(shape, scale=0.5):
        return bf16_round(rng.standard_normal(shape).astype(np.float32) * scale)

    def put(name, shape, gtype, data):
        tensors.append((name, tuple(shape), gtype, data))

    def lin_q8(target, src, rows, cols):
        t = Q8T(rng, rows, cols)
        truth[target] = t
        put(src, (rows, cols), Q8, t.raw())

    def lin_bf(target, src, shape):
        v = bf(shape)
        truth[target] = v
        put(src, shape, BF16, bf16_bytes(v))

    def norm1(target, src, n):
        w = bf((n,), 0.1)  # the HF zero-centred gain
        v = w.astype(np.float32) + np.float32(1)
        truth[target] = v
        put(src, (n,), F32, v.astype("<f4").tobytes())

    lin_q8(LM + "embed_tokens.weight", "token_embd.weight", g["V"], H)
    lin_q8(shell.LM_HEAD, "output.weight", g["V"], H)
    norm1(LM + "hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight", HCH)
    lin_bf(LM + "hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight", (g["R"], HCH))
    lin_bf(LM + "hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight", (HCH, g["R"]))
    for i, kind in enumerate(types):
        b, p = f"blk.{i}.", f"{LM}layers.{i}."
        for hc, dst in (("attn", "attn_hyper_connection."), ("ffn", "mlp_hyper_connection.")):
            norm1(p + dst + "hc_norm.weight", b + f"hc_{hc}_norm.weight", HCH)
            lin_bf(p + dst + "input_mix_weight_down.weight", b + f"hc_{hc}_down.weight", (g["R"], HCH))
            lin_bf(p + dst + "input_mix_weight_up.weight", b + f"hc_{hc}_up.weight", (HCH, g["R"]))
            lin_bf(p + dst + "block_inject_weight.weight", b + f"hc_{hc}_inject.weight", (g["HC"], HCH))
        lin_bf(p + "mlp.gate.weight", b + "ffn_gate_inp.weight", (g["E"], H))
        lin_bf(p + "mlp.shared_expert_gate.weight", b + "ffn_gate_inp_shexp.weight", (1, H))
        lin_q8(p + "mlp.shared_expert.gate_proj.weight", b + "ffn_gate_shexp.weight", g["FF"], H)
        lin_q8(p + "mlp.shared_expert.up_proj.weight", b + "ffn_up_shexp.weight", g["FF"], H)
        lin_q8(p + "mlp.shared_expert.down_proj.weight", b + "ffn_down_shexp.weight", H, g["FF"])
        if kind == "linear_attention":
            a = p + "linear_attn."
            # in_proj_qkv: value rows tiled; codes and scales move with their rows
            t = Q8T(rng, 2 * kd + vd, H)
            truth[a + "in_proj_qkv.weight"] = t
            q, d = t.q.copy(), t.d.copy()
            q[2 * kd:] = tile_heads(t.q[2 * kd:], g["KH"], rr, dv)
            d[2 * kd:] = tile_heads(t.d[2 * kd:], g["KH"], rr, dv)
            put(b + "attn_qkv.weight", t.q.shape, Q8, t.raw(q, d))
            t = Q8T(rng, vd, H)
            truth[a + "in_proj_z.weight"] = t
            put(b + "attn_gate.weight", t.q.shape, Q8,
                t.raw(tile_heads(t.q, g["KH"], rr, dv), tile_heads(t.d, g["KH"], rr, dv)))
            # out_proj: columns tiled (whole 32-column blocks when dv % 32 == 0)
            t = Q8T(rng, H, vd)
            if dv % 32 == 0:
                truth[a + "out_proj.weight"] = t
                put(b + "ssm_out.weight", t.q.shape, Q8, t.raw(tile_heads(t.q, g["KH"], rr, dv, axis=1),
                                                              tile_heads(t.d, g["KH"], rr, dv // 32, axis=1)))
            else:  # the file's Q8_0 blocks span tiled heads: t is the stored tensor, the truth its untiled decode
                put(b + "ssm_out.weight", t.q.shape, Q8, t.raw())
                truth[a + "out_proj.weight"] = _untile_cols(t.values(), g["KH"], rr, dv)
            for nm, src in (("in_proj_b", "ssm_beta"), ("in_proj_a", "ssm_alpha")):
                v = bf((g["VH"], H))
                truth[a + nm + ".weight"] = v
                put(b + src + ".weight", v.shape, BF16, bf16_bytes(tile_heads(v, g["KH"], rr, 1)))
            conv = bf((2 * kd + vd, g["CK"]))
            truth[a + "conv1d.weight"] = conv.reshape(-1, g["CK"], 1)
            ct = conv.copy()
            ct[2 * kd:] = tile_heads(conv[2 * kd:], g["KH"], rr, dv)
            put(b + "ssm_conv1d.weight", ct.shape, F32, ct.astype("<f4").tobytes())
            dt = bf((g["VH"],))
            truth[a + "dt_bias"] = dt
            put(b + "ssm_dt.bias", dt.shape, F32, tile_heads(dt, g["KH"], rr, 1).astype("<f4").tobytes())
            alog = bf16_round(rng.uniform(-3, 2.5, g["VH"]).astype(np.float32))
            truth[a + "A_log"] = alog
            ssm_a = (-np.exp(alog.astype(np.float64))).astype(np.float32)
            put(b + "ssm_a", ssm_a.shape, F32, tile_heads(ssm_a, g["KH"], rr, 1).astype("<f4").tobytes())
            nw = bf((dv,), 0.1) + np.float32(1)
            truth[a + "norm.weight"] = nw
            put(b + "ssm_norm.weight", nw.shape, F32, nw.astype("<f4").tobytes())
        else:
            s = p + "self_attn."
            lin_q8(s + "q_proj.weight", b + "attn_q.weight", 2 * g["NH"] * g["HD"], H)
            lin_q8(s + "k_proj.weight", b + "attn_k.weight", g["NKV"] * g["HD"], H)
            lin_q8(s + "v_proj.weight", b + "attn_v.weight", g["NKV"] * g["HD"], H)
            lin_q8(s + "o_proj.weight", b + "attn_output.weight", H, g["NH"] * g["HD"])
            norm1(s + "q_norm.weight", b + "attn_q_norm.weight", g["HD"])
            norm1(s + "k_norm.weight", b + "attn_k_norm.weight", g["HD"])
            qk = bf(((g["IH"] + g["IKV"]) * g["IHD"], H))
            truth[s + "indexer.index_qk_proj.weight"] = qk
            put(b + "indexer.q_proj.weight", (g["IH"] * g["IHD"], H), BF16, bf16_bytes(qk[: g["IH"] * g["IHD"]]))
            put(b + "indexer.k_proj.weight", (g["IKV"] * g["IHD"], H), BF16, bf16_bytes(qk[g["IH"] * g["IHD"]:]))
            norm1(s + "indexer.q_layernorm.weight", b + "indexer.q_norm.weight", g["IHD"])
            norm1(s + "indexer.k_layernorm.weight", b + "indexer.k_norm.weight", g["IHD"])
        if i == 1:
            q = p + "ple."
            lin_bf(q + "key_proj.weight", b + "ple_key.weight", (HCH, g["PD"]))
            lin_bf(q + "value_proj.weight", b + "ple_value.weight", (H, g["PD"]))
            for nm in ("norm_key", "norm_query", "norm_conv"):
                norm1(q + nm + ".weight", b + f"ple_{nm}.weight", HCH)
            conv = bf((HCH, 4), 0.2).astype(np.float16)
            truth[q + "conv_weight"] = conv.astype(np.float32).reshape(HCH, 4, 1)
            put(b + "ple_conv1d.weight", conv.shape, F16, conv.astype("<f2").tobytes())
            offsets = list(np.cumsum([0] + SIZES[:-1]))
            truth[q + "ple_embedding.layer_multipliers"] = np.array(MULTS, np.int64)
            truth[q + "ple_embedding.ngram_heads_vocab_sizes"] = np.array(SIZES, np.int64)
            truth[q + "ple_embedding.ngram_heads_offsets"] = np.array(offsets, np.int64)
    meta = {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": g["L"],
        "qwen4exp.context_length": 4096,
        "qwen4exp.embedding_length": H,
        "qwen4exp.attention.head_count": g["NH"],
        "qwen4exp.attention.head_count_kv": g["NKV"],
        "qwen4exp.rope.dimension_sections": [2, 1, 1, 0],
        "qwen4exp.rope.freq_base": 10000000.0,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.expert_count": g["E"],
        "qwen4exp.expert_used_count": 2,
        "qwen4exp.attention.key_length": g["HD"],
        "qwen4exp.attention.value_length": g["HD"],
        "qwen4exp.expert_feed_forward_length": g["FF"],
        "qwen4exp.expert_shared_feed_forward_length": g["FF"],
        "qwen4exp.ssm.conv_kernel": g["CK"],
        "qwen4exp.ssm.state_size": g["DK"],
        "qwen4exp.ssm.group_count": g["KH"],
        "qwen4exp.ssm.time_step_rank": g["VH"],
        "qwen4exp.ssm.inner_size": vd,
        "qwen4exp.attention.recurrent_layers": [t == "linear_attention" for t in types],
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.rope.dimension_count": g["HD"] // 4,
        "qwen4exp.hyper_connection.count": g["HC"],
        "qwen4exp.hyper_connection.low_rank": g["R"],
        "qwen4exp.attention.indexer.head_count": g["IH"],
        "qwen4exp.attention.indexer.key_length": g["IHD"],
        "qwen4exp.attention.indexer.top_k": 64,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4],
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 2,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 7,
        "qwen4exp.ple.image_token_id": 8,
        "qwen4exp.embedding_length_per_layer_input": g["PD"] // 4,
        "qwen4exp.ple.layer_multipliers": MULTS,
        "qwen4exp.ple.head_offsets": [int(x) for x in np.cumsum([0] + SIZES[:-1])],
        "qwen4exp.ple.head_vocab_sizes": SIZES,
        "tokenizer.ggml.bos_token_id": 5,
    }
    if drop:
        tensors = [t for t in tensors if t[0] != drop]
    if retype:
        name, gtype, data = retype
        tensors = [(n, s, gtype, data) if n == name else (n, s, t, dd) for n, s, t, dd in tensors]
    if extra:
        tensors.append(extra)
    path = tmp_path / "model-skeleton.gguf"
    syn.write_gguf(path, meta, tensors)
    return path, truth, meta


def _untile_cols(a, k, r, width):
    """Independent inverse of the column tiling: tiled column block j * k + g holds grouped block g * r + j."""
    out = np.empty_like(a)
    for gi in range(k):
        for j in range(r):
            out[:, (gi * r + j) * width:(gi * r + j + 1) * width] = a[:, (j * k + gi) * width:(j * k + gi + 1) * width]
    return out


def as_values(arr) -> np.ndarray:
    return np.array(arr.astype(mx.float32)) if arr.dtype != mx.int64 else np.array(arr)


def dequant(out, base) -> np.ndarray:
    return np.array(mx.dequantize(out[base + ".weight"], out[base + ".scales"], out[base + ".biases"],
                                  group_size=32, bits=8, dtype=mx.float32))


@pytest.fixture
def synth(tmp_path):
    path, truth, meta = build(tmp_path)
    return GGUFFile(path), truth


# -- transforms and their inverses ------------------------------------------------


@pytest.mark.parametrize("k,v", [(1, 4), (2, 6), (16, 48), (4, 4), (3, 12)])
def test_v_head_order_inverts_llama_tiling(k, v):
    grouped = np.arange(v)
    tiled = tile_heads(grouped, k, v // k, 1)
    assert np.array_equal(tiled[shell._v_head_order(k, v)], grouped)
    if k > 1 and v // k > 1:
        assert not np.array_equal(tiled, grouped)  # the tiling is a real permutation


def test_q8_affine_repack_dequantizes_exactly():
    rng = np.random.default_rng(3)
    q = rng.integers(-128, 128, (8, 128), dtype=np.int8)
    d = np.concatenate([rng.uniform(-3, 3, 24), [6e-8, -6e-8, 65504 / 128, 1e-5, 0, 2.0**-14, -1.0, 0.1]])
    d = d.astype(np.float16).reshape(8, 4)
    words, s, b = shell.q8_affine_parts(q, d)
    got = np.array(mx.dequantize(mx.array(words), mx.array(s), mx.array(b), group_size=32, bits=8, dtype=mx.float32))
    want = (q.reshape(8, 4, 32).astype(np.float32) * d.astype(np.float32)[..., None]).reshape(8, 128)
    assert np.array_equal(got, want)
    x = mx.array(rng.standard_normal((3, 128)).astype(np.float32))
    y = mx.quantized_matmul(x, mx.array(words), mx.array(s), mx.array(b), transpose=True, group_size=32, bits=8)
    np.testing.assert_allclose(np.array(y), np.array(x) @ want.T, rtol=2e-5, atol=1e-4)


def test_a_log_recovers_the_bf16_source_and_keeps_float_otherwise():
    alog = bf16_round(np.linspace(-6, 3, 50).astype(np.float32))
    a = (-np.exp(alog.astype(np.float64))).astype(np.float32)
    got, snapped = shell._a_log(a)
    assert snapped and np.array_equal(got, alog)
    off = a * np.float32(1.001)  # not the image of any bf16 A_log
    got, snapped = shell._a_log(off)
    assert not snapped
    np.testing.assert_allclose(-np.exp(got.astype(np.float64)), off, rtol=1e-6)
    with pytest.raises(shell.ShellError):
        shell._a_log(np.array([0.5], np.float32))


def test_conversion_inverts_every_transform(synth):
    sk, truth = synth
    out = dict(shell.convert_shell(sk))
    q8_targets = [t for t, v in truth.items() if isinstance(v, Q8T)]
    for t, v in truth.items():
        if isinstance(v, Q8T):
            base = t[: -len(".weight")]
            assert out[base + ".scales"].dtype == mx.float32
            assert np.array_equal(dequant(out, base), v.values()), t
        else:
            assert tuple(out[t].shape) == v.shape, t
            assert np.array_equal(as_values(out[t]), v), t
    expected = {t for t, v in truth.items() if not isinstance(v, Q8T)}
    expected |= {t[: -len(".weight")] + s for t in q8_targets for s in (".weight", ".scales", ".biases")}
    assert set(out) == expected


def test_permuted_sources_differ_from_truth(synth):
    """Negative control: the stored (un-inverted) tensors are NOT the model's values."""
    sk, truth = synth
    a = LM + "layers.0.linear_attn."
    pairs = [
        ("blk.0.attn_qkv.weight", truth[a + "in_proj_qkv.weight"].values()),
        ("blk.0.attn_gate.weight", truth[a + "in_proj_z.weight"].values()),
        ("blk.0.ssm_out.weight", truth[a + "out_proj.weight"].values()),
        ("blk.0.ssm_alpha.weight", truth[a + "in_proj_a.weight"]),
        ("blk.0.ssm_beta.weight", truth[a + "in_proj_b.weight"]),
        ("blk.0.ssm_dt.bias", truth[a + "dt_bias"]),
        ("blk.0.ssm_conv1d.weight", truth[a + "conv1d.weight"].reshape(-1, 4)),
    ]
    for name, want in pairs:
        assert not np.array_equal(sk.array(name), want), name
    assert not np.array_equal(np.log(-sk.array("blk.0.ssm_a")), truth[a + "A_log"])


def test_dtypes_are_lossless(synth):
    sk, truth = synth
    out = dict(shell.convert_shell(sk))
    p = LM + "layers.0."
    assert out[p + "mlp.gate.weight"].dtype == mx.bfloat16  # BF16 stays
    assert out[p + "attn_hyper_connection.hc_norm.weight"].dtype == mx.float32  # 1 + w is not a bf16 number
    assert out[p + "linear_attn.conv1d.weight"].dtype == mx.bfloat16  # F32 holding bf16 numbers narrows
    assert out[LM + "layers.1.ple.conv_weight"].dtype == mx.bfloat16  # F16 holding bf16 numbers narrows
    assert out[p + "linear_attn.A_log"].dtype == mx.bfloat16
    wide = dict(shell.convert_shell(sk, narrow_floats=False, targets=[p + "linear_attn.conv1d.weight"]))
    assert wide[p + "linear_attn.conv1d.weight"].dtype == mx.float32
    flt = dict(shell.convert_shell(sk, q8="float32", targets=[LM + "embed_tokens.weight"]))
    assert set(flt) == {LM + "embed_tokens.weight"}
    assert np.array_equal(np.array(flt[LM + "embed_tokens.weight"]), truth[LM + "embed_tokens.weight"].values())


def test_out_proj_falls_back_to_float_when_heads_split_q8_blocks(tmp_path):
    path, truth, _ = build(tmp_path, dv=16)
    sk = GGUFFile(path)
    t = LM + "layers.0.linear_attn.out_proj.weight"
    rep: dict = {}
    out = dict(shell.convert_shell(sk, targets=[t, t[:-7]], report=rep))
    assert set(out) == {t} and out[t].dtype == mx.float32
    assert np.array_equal(np.array(out[t]), truth[t])
    assert "q8_fallback" in rep[t]
    q = shell.shell_quantization(sk)
    assert q[t[: -len(".weight")]] is False
    assert q[LM + "layers.0.linear_attn.in_proj_qkv"] == {"group_size": 32, "bits": 8, "mode": "affine"}


# -- fail closed -------------------------------------------------------------------


def test_every_tensor_is_accounted_for(synth):
    sk, _ = synth
    acc = shell.shell_accounting(sk)
    assert set(acc) == set(sk.tensors)
    assert acc["blk.3.indexer.k_proj.weight"] == acc["blk.3.indexer.q_proj.weight"]
    shell.check_plan_covers_model(sk)


def test_extra_tensor_raises(tmp_path):
    path, _, _ = build(tmp_path, extra=("blk.0.mystery.weight", (4,), F32, b"\0" * 16))
    with pytest.raises(shell.ShellError, match="not accounted"):
        shell.shell_plan(GGUFFile(path))


def test_missing_tensor_raises(tmp_path):
    path, _, _ = build(tmp_path, drop="blk.2.ssm_dt.bias")
    with pytest.raises(shell.ShellError, match="missing"):
        list(shell.convert_shell(GGUFFile(path)))


def test_wrong_shape_raises(tmp_path):
    path, _, _ = build(tmp_path, drop="blk.0.ffn_gate_inp.weight",
                       extra=("blk.0.ffn_gate_inp.weight", (3, 64), F32, b"\0" * 3 * 64 * 4))
    with pytest.raises(shell.ShellError, match="shape"):
        shell.shell_plan(GGUFFile(path))


def test_unsupported_type_raises(tmp_path):
    path, _, _ = build(tmp_path, retype=("blk.3.attn_q.weight", 2, b"\0" * (128 * 64 // 32 * 18)))
    with pytest.raises(shell.ShellError, match="not supported"):
        shell.shell_plan(GGUFFile(path))


# -- config and the real model class ----------------------------------------------------


def test_config_from_metadata(synth):
    sk, _ = synth
    tc = shell.shell_text_config(sk)
    assert tc["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert tc["linear_num_value_heads"] == 6 and tc["linear_num_key_heads"] == 2
    assert tc["linear_value_head_dim"] == 32 and tc["indexer_kv_heads"] == 1
    assert tc["ple_layer_ids"] == [2] and tc["ple_embed_dim"] == 64
    assert tc["rms_norm_eps"] == 1e-6 and tc["rope_theta"] == 10000000
    assert tc["rope_parameters"]["mrope_section"] == [2, 1, 1] and tc["partial_rotary_factor"] == 0.25
    assert tc["ngram_vocab_size_base"] == 100 and tc["eos_token_id"] == 7 and tc["bos_token_id"] == 5
    assert tc["ngram_sidecar"] is True and tc["tie_word_embeddings"] is False


def _load_like_mlx_lm(model, cfg, weights):
    """mlx_lm.utils.load_model's sanitize -> quantize(class_predicate) -> load_weights, without files."""
    weights = model.sanitize(weights)
    q = cfg["quantization"]

    def predicate(p, m):
        if p in q:
            return q[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q["mode"], class_predicate=predicate)
    return weights


def test_tiny_model_loads_the_shell(synth):
    from mlx.utils import tree_flatten

    from mtplx.models.qwen4_exp import Model, ModelArgs

    sk, _ = synth
    cfg = shell.shell_config(sk)
    model = Model(ModelArgs.from_dict(cfg))
    before = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    experts = {k for k in before if ".mlp.switch_mlp." in k}
    assert {k: v for k, v in before.items() if k not in experts} == shell.expected_shell_parameters(cfg["text_config"])
    weights = _load_like_mlx_lm(model, cfg, dict(shell.convert_shell(sk)))
    params = dict(tree_flatten(model.parameters()))
    assert set(params) - set(weights) == experts
    for k, v in weights.items():
        assert params[k].shape == v.shape, k
    model.load_weights(list(weights.items()) + [(k, params[k]) for k in experts], strict=True)
    params = dict(tree_flatten(model.parameters()))
    for k, v in weights.items():
        assert params[k].dtype == v.dtype, k
        assert np.array_equal(np.array(params[k].astype(mx.float32) if params[k].dtype == mx.bfloat16 else params[k]),
                              np.array(v.astype(mx.float32) if v.dtype == mx.bfloat16 else v)), k
    assert isinstance(model.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.language_model.lm_head, nn.QuantizedLinear)
    assert not isinstance(model.layers[0].mlp.gate, nn.QuantizedLinear)


def _gdn_engine_reference(sk, b, X, g):
    """GDN on the tensors AS STORED, with the engine's head pairing ``h % h_k``
    (edge0-strata include/strata/kernels/gdn.hpp), in float64."""
    W = lambda n: sk.array(b + n).astype(np.float64)  # noqa: E731
    kh, vh, dk, dv = g["KH"], g["VH"], g["DK"], g["DV"]
    kd = kh * dk
    st = np.zeros((vh, dk, dv))
    hist = np.zeros((g["CK"] - 1, 2 * kd + vh * dv))
    outs = []
    for x in X:
        inp = np.vstack([hist, W("attn_qkv.weight") @ x])
        hist = inp[1:]
        cv = (inp.T * W("ssm_conv1d.weight")).sum(1)
        cv = cv / (1 + np.exp(-cv))
        q, k, v = cv[:kd].reshape(kh, dk), cv[kd:2 * kd].reshape(kh, dk), cv[2 * kd:].reshape(vh, dv)
        q = q / np.sqrt((q * q).sum(1, keepdims=True) + 1e-6) / np.sqrt(dk)
        k = k / np.sqrt((k * k).sum(1, keepdims=True) + 1e-6)
        gate = np.log1p(np.exp(W("ssm_alpha.weight") @ x + W("ssm_dt.bias"))) * W("ssm_a")
        beta = 1 / (1 + np.exp(-(W("ssm_beta.weight") @ x)))
        o = np.zeros((vh, dv))
        for h in range(vh):
            st[h] *= np.exp(gate[h])
            d = (v[h] - k[h % kh] @ st[h]) * beta[h]
            st[h] += np.outer(k[h % kh], d)
            o[h] = q[h % kh] @ st[h]
        z = (W("attn_gate.weight") @ x).reshape(vh, dv)
        y = o / np.sqrt((o * o).mean(1, keepdims=True) + 1e-6) * W("ssm_norm.weight") / (1 + np.exp(-z))
        outs.append(W("ssm_out.weight") @ y.reshape(-1))
    return np.array(outs)


def test_converted_gdn_computes_like_the_engine_on_the_stored_layout(synth):
    from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs

    sk, _ = synth
    cfg = shell.shell_config(sk)
    prefix = LM + "layers.0.linear_attn."
    gdn = GatedDeltaNet(TextArgs.from_dict(cfg["text_config"]))
    w = {k[len(prefix):]: v for k, v in shell.convert_shell(sk) if k.startswith(prefix)}
    nn.quantize(gdn, class_predicate=lambda p, m: cfg["quantization"].get(prefix + p, False))
    gdn.load_weights(list(w.items()), strict=True)
    X = np.random.default_rng(5).standard_normal((4, G["H"])).astype(np.float32)
    y = np.array(gdn(mx.array(X)[None]).astype(mx.float32))[0]
    ref = _gdn_engine_reference(sk, "blk.0.", X.astype(np.float64), G)
    assert np.linalg.norm(y - ref) / np.linalg.norm(ref) < 2e-3


# -- the real folder ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    from mtplx.edge0.pack import Edge0Pack

    return Edge0Pack(REAL)


@real_only
def test_real_accounting_and_plan(real):
    acc = shell.shell_accounting(real)
    assert len(acc) == len(real.skeleton.tensors) == 1079
    plan = shell.shell_plan(real)
    assert len(plan) == 1070
    shell.check_plan_covers_model(real)


@real_only
def test_real_config_matches_the_base_model(real):
    tc = shell.shell_text_config(real)
    # values from Qwen/Qwen3.8-Flash-Next config.json (text_config)
    base = {
        "hidden_size": 2560, "num_hidden_layers": 48, "num_attention_heads": 24, "num_key_value_heads": 2,
        "head_dim": 256, "rms_norm_eps": 1e-06, "vocab_size": 248320, "max_position_embeddings": 262144,
        "full_attention_interval": 4, "linear_num_value_heads": 48, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 512,
        "num_experts_per_tok": 10, "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
        "hc_count": 4, "hc_lowrank": 320, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
        "indexer_budget": 2048, "indexer_compress_ratio": 4, "ple_layer_ids": [2], "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 20000000,
        "make_ngram_vocab_size_divisible_by": 128, "partial_rotary_factor": 0.25, "eos_token_id": 248044,
        "bos_token_id": 248044, "output_gate_type": "sigmoid", "tie_word_embeddings": False,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
    }
    assert {k: tc[k] for k in base} == base
    assert tc["layer_types"] == ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)]


@real_only
def test_real_model_tree_matches_the_plan(real):
    from mlx.utils import tree_flatten

    from mtplx.models.qwen4_exp import Model, ModelArgs

    cfg = shell.shell_config(real)
    model = Model(ModelArgs.from_dict(cfg))  # lazy: the routed experts are never evaluated
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters()) if ".mlp.switch_mlp." not in k}
    assert params == {e.target: e.shape for e in shell.shell_plan(real)}


@real_only
def test_real_spot_conversions(real):
    sk = real.skeleton
    p = LM + "layers.0."
    targets = [p + "mlp.shared_expert.down_proj", p + "linear_attn.dt_bias", p + "linear_attn.A_log",
               p + "linear_attn.in_proj_a.weight"]
    rep: dict = {}
    out = dict(shell.convert_shell(real, targets=targets, report=rep))
    assert np.array_equal(dequant(out, p + "mlp.shared_expert.down_proj"), sk.array("blk.0.ffn_down_shexp.weight"))
    tiled = sk.array("blk.0.ssm_dt.bias")
    assert np.array_equal(as_values(out[p + "linear_attn.dt_bias"]), tiled.reshape(3, 16).T.reshape(-1))
    assert rep[p + "linear_attn.A_log"]["a_log_snapped_to_bf16"] is True
    a = -np.exp(as_values(out[p + "linear_attn.A_log"]).astype(np.float64))
    np.testing.assert_allclose(a, sk.array("blk.0.ssm_a").reshape(3, 16).T.reshape(-1), rtol=1e-6)
    assert np.array_equal(as_values(out[p + "linear_attn.in_proj_a.weight"]),
                          sk.array("blk.0.ssm_alpha.weight").reshape(3, 16, -1).swapaxes(0, 1).reshape(48, -1))
