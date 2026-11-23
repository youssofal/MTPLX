"""CPU unit tests for the opt-in draft-only quantized vocabulary head (MTPLX_INKLING_DRAFT_HEAD_BITS)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mtplx.models.inkling.draft import (
    DRAFT_HEAD_BITS_ENV,
    Drafter,
    draft_head_bits,
    draft_head_group_size,
    quantize_draft_head,
)
from mtplx.models.inkling.fast import draft as fast_draft
from mtplx.models.inkling.head import vocabulary_logits
from mtplx.models.inkling.precision import AffineWeight


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _raw_bits(array: mx.array) -> np.ndarray:
    mx.eval(array)
    if array.dtype in (mx.bfloat16, mx.float16):
        return np.asarray(array.view(mx.uint16)).copy()
    if array.dtype == mx.float32:
        return np.asarray(array.view(mx.uint32)).copy()
    return np.asarray(array).copy()


def _param_nbytes(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for _name, value in tree_flatten(module.parameters()):
        if isinstance(value, mx.array) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _build_synthetic_drafter_pack(
    tmp_path: Path,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    hidden: int = 128,
    vocab: int = 96,
    padded_vocab: int = 128,
    head_dim: int = 32,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    mlp_dim: int = 192,
    rank: int = 16,
    block_size: int = 4,
    num_layers: int = 2,
):
    key = mx.random.key(42)

    def rand(shape, scale=0.05):
        nonlocal key
        key, sub = mx.random.split(key)
        return (mx.random.normal(shape, key=sub) * scale).astype(dtype)

    mtp_config = {
        "num_hidden_layers": num_layers,
        "local_layer_ids": [0],
        "sliding_window_size": 8,
        "head_dim": head_dim,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "block_size": block_size,
        "markov_rank": rank,
    }
    tensors = {
        "fuse_weight": mx.zeros((num_layers + 1, 2), dtype=dtype),
        "mask_embed": rand((1, 1, hidden)),
        "norm.weight": mx.ones((hidden,), dtype=dtype),
        "markov_head.dflare_norm.weight": mx.ones((hidden,), dtype=dtype),
        "markov_head.hidden_out.weight": rand((hidden, rank)),
        "markov_head.hidden_out.bias": rand((hidden,)),
        "markov_head.rnn_proj.weight": rand((3 * rank, 2 * rank + 2 * hidden)),
        "markov_head.rnn_proj.bias": rand((3 * rank,)),
        "markov_head.markov_w1.weight": rand((vocab, rank)),
    }
    for i in range(num_layers):
        stem = f"layers.{i}."
        tensors[stem + "dflare_norm.weight"] = mx.ones((hidden,), dtype=dtype)
        tensors[stem + "input_layernorm.weight"] = mx.ones((hidden,), dtype=dtype)
        tensors[stem + "post_attention_layernorm.weight"] = mx.ones((hidden,), dtype=dtype)
        tensors[stem + "self_attn.q_norm.weight"] = mx.ones((head_dim,), dtype=dtype)
        tensors[stem + "self_attn.k_norm.weight"] = mx.ones((head_dim,), dtype=dtype)
        tensors[stem + "self_attn.k_norm_ctx.weight"] = mx.ones((head_dim,), dtype=dtype)
        tensors[stem + "self_attn.q_proj.weight"] = rand((num_heads * head_dim, hidden))
        tensors[stem + "self_attn.k_proj.weight"] = rand((num_kv_heads * head_dim, hidden))
        tensors[stem + "self_attn.v_proj.weight"] = rand((num_kv_heads * head_dim, hidden))
        tensors[stem + "self_attn.k_proj_ctx.weight"] = rand((num_kv_heads * head_dim, hidden))
        tensors[stem + "self_attn.v_proj_ctx.weight"] = rand((num_kv_heads * head_dim, hidden))
        tensors[stem + "self_attn.o_proj.weight"] = rand((hidden, num_heads * head_dim))
        tensors[stem + "mlp.gate_proj.weight"] = rand((mlp_dim, hidden))
        tensors[stem + "mlp.up_proj.weight"] = rand((mlp_dim, hidden))
        tensors[stem + "mlp.down_proj.weight"] = rand((hidden, mlp_dim))

    mx.eval(tensors)
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), tensors)

    embed_table = rand((padded_vocab, hidden), scale=0.2)
    unembed_weight = rand((padded_vocab, hidden), scale=0.2)
    mx.eval(embed_table, unembed_weight)

    class _ToyTarget:
        def __init__(self):
            self.config = {
                "unpadded_vocab_size": vocab,
                "final_logit_softcapping": 30.0,
            }
            self.unembed = unembed_weight

        def embed_tokens(self, ids):
            return embed_table[ids]

        def head(self, h):
            cap = self.config["final_logit_softcapping"]
            logits = vocabulary_logits(h.astype(embed_table.dtype), self.unembed, vocab)
            return mx.tanh(logits / cap) * cap if cap else logits

    captures = [rand((1, 4, hidden), scale=0.2), rand((1, 4, hidden), scale=0.2)]
    mx.eval(captures)
    return mtp_config, _ToyTarget(), captures


def test_draft_head_bits_env_parsing_and_default_off(monkeypatch):
    monkeypatch.delenv(DRAFT_HEAD_BITS_ENV, raising=False)
    assert draft_head_bits() is None

    for off in ("", "0", "off", "OFF", "false", "no", "none"):
        monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, off)
        assert draft_head_bits() is None

    for bits in (4, 6, 8):
        monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, str(bits))
        assert draft_head_bits() == bits

    for bad in ("3", "5", "16", "invalid"):
        monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, bad)
        with pytest.raises(ValueError, match=DRAFT_HEAD_BITS_ENV):
            draft_head_bits()


def test_draft_head_group_size_selection():
    assert draft_head_group_size(2304) == 64
    assert draft_head_group_size(128) == 64
    assert draft_head_group_size(96) == 32
    assert draft_head_group_size(40) is None
    assert draft_head_group_size(0) is None


def test_default_unset_leaves_draft_head_shared_with_zero_extra_ram(monkeypatch, tmp_path):
    monkeypatch.delenv(DRAFT_HEAD_BITS_ENV, raising=False)
    cfg, target, captures = _build_synthetic_drafter_pack(tmp_path)
    target_bits_before = _raw_bits(target.unembed)

    drafter = Drafter(tmp_path, cfg, target)
    assert drafter.draft_unembed is None
    assert drafter.draft_head(target) is target.unembed

    cache = drafter.make_cache()
    drafter.observe(captures, cache)
    proposed = drafter.propose(target, anchor=5, cache=cache, count=3)
    mx.eval(proposed)
    assert proposed.shape == (3,)
    assert np.array_equal(_raw_bits(target.unembed), target_bits_before)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_quantized_draft_head_is_compact_leaves_target_bit_identical_and_runs_on_cpu(
    monkeypatch, tmp_path, dtype, bits
):
    cfg, target, captures = _build_synthetic_drafter_pack(tmp_path, dtype=dtype)
    target_bits_before = _raw_bits(target.unembed)
    probe_h = mx.random.normal((1, 2, 128), key=mx.random.key(99)).astype(dtype)
    target_logits_before = _raw_bits(target.head(probe_h))

    monkeypatch.delenv(DRAFT_HEAD_BITS_ENV, raising=False)
    baseline_drafter = Drafter(tmp_path, cfg, target)
    baseline_bytes = _param_nbytes(baseline_drafter)

    monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, str(bits))
    quant_drafter = Drafter(tmp_path, cfg, target)

    draft_head = quant_drafter.draft_unembed
    assert isinstance(draft_head, AffineWeight)
    assert draft_head.bits == bits
    assert draft_head.group_size == 64
    assert quant_drafter.draft_head(target) is draft_head

    # Compact packed size vs dense BF16/FP16 target matrix.
    packed_bytes = (
        draft_head.codes.nbytes + draft_head.scales.nbytes + draft_head.biases.nbytes
    )
    assert packed_bytes < target.unembed.nbytes
    assert _param_nbytes(quant_drafter) == baseline_bytes + packed_bytes

    # Draft head product closely approximates the dense target unembed on CPU.
    ref_base = (probe_h.astype(mx.float32) @ target.unembed.astype(mx.float32).T)[..., :96]
    got_base = draft_head.product(probe_h).astype(mx.float32)[..., :96]
    mx.eval(ref_base, got_base)
    ref_np, got_np = np.asarray(ref_base).reshape(-1), np.asarray(got_base).reshape(-1)
    cos_sim = float(np.dot(ref_np, got_np) / (np.linalg.norm(ref_np) * np.linalg.norm(got_np)))
    assert cos_sim > (0.995 if bits >= 6 else 0.96)

    # Stock Drafter.propose runs cleanly on CPU with the quantized draft head.
    cache = quant_drafter.make_cache()
    quant_drafter.observe(captures, cache)
    proposed = quant_drafter.propose(target, anchor=5, cache=cache, count=3)
    mx.eval(proposed)
    assert proposed.shape == (3,)
    assert all(0 <= int(t) < 96 for t in proposed.tolist())

    # fast_draft.propose (both bf16 and f32 precision routes) runs cleanly on CPU.
    for precision in ("bf16", "f32"):
        ids, law = fast_draft.propose(
            quant_drafter,
            target,
            anchor=5,
            cache=cache,
            count=3,
            position=10,
            seed=1234,
            temperature=0.0,
            precision=precision,
        )
        mx.eval(ids, law.rows, law.stats)
        assert ids.shape == (3,)
        assert law.rows.shape == (3, 96)

    # Target lm_head weights and target logits stay 100% bit-identical.
    assert np.array_equal(_raw_bits(target.unembed), target_bits_before)
    assert np.array_equal(_raw_bits(target.head(probe_h)), target_logits_before)


def test_already_quantized_head_is_never_requantized(monkeypatch):
    monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, "6")

    dense = mx.random.normal((64, 128), key=mx.random.key(7)).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    affine = AffineWeight(codes, scales, biases, bits=4, group_size=64)
    assert quantize_draft_head(affine) is None

    qlinear = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)
    assert quantize_draft_head(qlinear) is None

    # Float32 matrix is also skipped (only BF16/FP16 dense heads qualify).
    assert quantize_draft_head(dense.astype(mx.float32)) is None

    # Non-multiple of 32 input width is safely skipped instead of raising.
    odd = mx.random.normal((64, 40), key=mx.random.key(8)).astype(mx.bfloat16)
    assert quantize_draft_head(odd) is None

    # Width divisible by 32 (not 64) falls back cleanly to group_size=32.
    w96 = mx.random.normal((64, 96), key=mx.random.key(9)).astype(mx.bfloat16)
    head96 = quantize_draft_head(w96)
    assert isinstance(head96, AffineWeight)
    assert head96.bits == 6 and head96.group_size == 32

    # Dense nn.Linear with 16-bit weights also quantizes without mutating the module.
    lin = nn.Linear(128, 64, bias=False)
    lin.weight = dense
    lin_before = _raw_bits(lin.weight)
    lin_head = quantize_draft_head(lin)
    assert isinstance(lin_head, AffineWeight) and lin_head.bits == 6
    assert np.array_equal(_raw_bits(lin.weight), lin_before)


def test_draft_table_uses_quantized_draft_head_and_keeps_target_fold(monkeypatch, tmp_path):
    import json
    from mtplx.models.inkling.fast import draft_table

    cfg, target, _captures = _build_synthetic_drafter_pack(tmp_path, dtype=mx.bfloat16)
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps({"ranking": list(range(32))}))

    monkeypatch.setenv(DRAFT_HEAD_BITS_ENV, "6")
    monkeypatch.setenv("MTPLX_INKLING_DRAFT_TABLE", f"{ranking_path}:16")
    drafter = Drafter(tmp_path, cfg, target)
    table = draft_table.bind(drafter, target)
    assert table is not None
    assert isinstance(table.head, AffineWeight)
    assert table.head.bits == 6 and table.head.group_size == 64
    assert table.head.codes.shape[0] == 16
    assert table.folded.shape == (16, cfg["markov_rank"] + 1)

