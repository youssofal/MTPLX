"""Streaming-decode parity for the DeepSeek-V4 MLX backend.

The one-shot prefill forward is the oracle here: it is itself pinned against a
reference golden by tests/test_deepseek_v4_parity.py (full-stack logits max_rel
1.8e-6, argmax 160/160).  This file gates the *incremental* path against it —
prompt prefill of P tokens followed by token-by-token decode of the remaining
T-P must reproduce the one-shot logits at every position.

What the cases have to cross, because that is where a KV-cache state machine
actually breaks:
  * ``P % compress_ratio != 0`` for both live ratios, so a partial compressor
    window is carried out of prefill into decode;
  * compress-window boundaries *inside* the decode loop — many on the ratio-4
    (overlap) layers, and two full 128-token windows on the ratio-128 layer,
    which is only reachable by decoding a few hundred tokens;
  * ``T`` well past ``window_size``, so per-position KV eviction engages and the
    evicted rows are exactly the ones the prefill mask would have zeroed;
  * chunked prefill (several multi-token calls against a live cache), which is
    what the serve path does and which exercises the cached-mask branch.

Self-contained: shrunk seeded config, no downloads, no torch.  Runs on the CPU
device so MLX fp32 matmul is bit-exact (the GPU fast path carries ~7.5e-4
relative, unrelated to correctness) — same convention as the parity test.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

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
_spec = importlib.util.spec_from_file_location("dsv4_decode_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_decode_undertest"] = D
_spec.loader.exec_module(D)

# Shrunk config.  Every layer type the backend has is present: ratio-0 sliding
# window (also a hash-routed layer), ratio-4 overlap compressor + indexer,
# ratio-128 non-overlap compressor, and a second ratio-4 layer on the
# score-routed side of num_hash_layers.
VOCAB = 64
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]
WINDOW = 16


def _args(**over):
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=DIM,
        num_hidden_layers=len(RATIOS),
        num_hash_layers=1,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        index_n_heads=N_HEADS,
        index_head_dim=HEAD_DIM,
        index_topk=512,
        compress_ratios=list(RATIOS),
        compress_rope_theta=160000.0,
        sliding_window=WINDOW,
        rope_scaling={
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=0.0,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _seeded_model(seed=0, **over):
    """Build the model and fill every parameter with seeded pseudo-random values.

    Shapes come from the module tree itself, so this cannot drift from the model.
    """
    mx.random.seed(seed)
    args = _args(**over)
    model = D.Model(args)
    filled = []
    for name, value in tree_flatten(model.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, args.n_routed_experts, value.shape).astype(
                mx.int32
            )
        elif value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            # RMSNorm weights and the HC scales sit around 1; biases/sinks around 0.
            centre = 1.0 if leaf in ("scale",) or name.endswith("norm.weight") else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    model.update(tree_unflatten(filled))
    mx.eval(model.parameters())
    return args, model


def _tokens(seq_len, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, seq_len))


def _compare(ref, got, label):
    """Per-step max relative error + exact argmax, against the one-shot oracle.

    ``ref``/``got`` are ``[b, T, vocab]``.
    """
    worst_rel = 0.0
    for row in range(ref.shape[0]):
        for t in range(ref.shape[1]):
            scale = float(np.max(np.abs(ref[row, t]))) + 1e-12
            rel = float(np.max(np.abs(got[row, t] - ref[row, t]))) / scale
            worst_rel = max(worst_rel, rel)
            assert rel <= 5e-5, (
                f"{label}: row {row} step {t} logits diverge max_rel={rel:.3e}"
            )
            assert int(got[row, t].argmax()) == int(ref[row, t].argmax()), (
                f"{label}: row {row} step {t} argmax {int(got[row, t].argmax())} != "
                f"{int(ref[row, t].argmax())} (max_rel={rel:.3e})"
            )
    return worst_rel


def _prefill_then_decode(model, ids, prompt_len, prompt_chunks=1):
    """Run the incremental path: (chunked) prompt prefill, then one token at a time."""
    total = ids.shape[1]
    cache = model.make_cache()
    pieces = []
    bounds = [
        round(prompt_len * (i + 1) / prompt_chunks) for i in range(prompt_chunks)
    ]
    start = 0
    for end in bounds:
        if end == start:
            continue
        pieces.append(np.array(model(ids[:, start:end], cache=cache)))
        start = end
    for t in range(prompt_len, total):
        pieces.append(np.array(model(ids[:, t : t + 1], cache=cache)))
    assert [c.offset for c in cache] == [total] * len(cache)
    return np.concatenate(pieces, axis=1)


def _run_case(prompt_len, total, *, prompt_chunks=1, batch=1, seed=0, label=""):
    args, model = _seeded_model(seed=seed)
    ids = _tokens(total, batch=batch)
    ref = np.array(model(ids).astype(mx.float32))
    # A degenerate model (constant argmax) would make the gate vacuous.
    assert len(set(ref[0].argmax(-1).tolist())) > 1, "oracle logits are degenerate"
    got = _prefill_then_decode(model, ids, prompt_len, prompt_chunks=prompt_chunks)
    return _compare(ref, got, label or f"P={prompt_len} T={total}")


def test_make_cache_shape():
    """One cache per layer, carrying that layer's own ratio and window."""
    args, model = _seeded_model()
    cache = model.make_cache()
    assert [c.compress_ratio for c in cache] == RATIOS
    assert all(c.window_size == WINDOW for c in cache)
    assert all(c.offset == 0 and c.empty() for c in cache)
    model(_tokens(5)[:, :5], cache=cache)
    assert [c.offset for c in cache] == [5] * len(cache)
    # ratio-4 layers have pooled one window (tokens 0-3) and hold one partial row;
    # ratio-128 has emitted nothing; ratio-0 keeps no compressor state at all.
    assert [c.n_compressed for c in cache] == [0, 1, 0, 1]
    assert cache[0].comp.cur_kv is None
    assert cache[1].comp.cur_kv.shape[1] == 1
    # Window bookkeeping: the buffer holds the attendable window_size rows plus
    # rollback_capacity older ones (retained only so trim can rewind -- they are
    # never returned to attention), and window_start is where the buffer starts.
    for c in cache:
        assert c.window.shape[1] == 5 and c.window_start == 0
    model(_tokens(40)[:, 5:40], cache=cache)
    for c in cache:
        held = min(40, WINDOW + c.rollback_capacity)
        assert c.offset == 40
        assert c.window.shape[1] == held
        assert c.window_start == 40 - held
    assert [c.n_compressed for c in cache] == [0, 10, 0, 10]  # 40 // 4


def test_decode_matches_prefill_partial_window_and_eviction():
    """P % 4 != 0, and T runs well past window_size so KV eviction engages."""
    assert 13 % 4 != 0 and 40 > WINDOW
    worst = _run_case(13, 40, label="partial-window+eviction")
    assert worst <= 5e-5


def test_decode_crosses_both_compress_window_boundaries():
    """Two ratio-128 windows complete *inside* the decode loop (at 255 and 383),
    along with ~65 ratio-4 windows; the prompt ends mid-window on both ratios."""
    prompt, total = 137, 400
    assert prompt % 4 != 0 and prompt % 128 != 0
    completed_in_decode = [b for b in (127, 255, 383) if b >= prompt]
    assert len(completed_in_decode) >= 2
    worst = _run_case(prompt, total, label="cross-128-boundaries")
    assert worst <= 5e-5


def test_chunked_prompt_prefill_then_decode():
    """Serve feeds prompts in chunks: several multi-token calls against a live
    cache must land in the same state as one contiguous prefill."""
    worst = _run_case(137, 200, prompt_chunks=3, label="chunked-prefill")
    assert worst <= 5e-5


def test_decode_from_single_token_prompt():
    """Degenerate prompt: everything but token 0 goes through the s==1 path."""
    worst = _run_case(1, 40, label="single-token-prompt")
    assert worst <= 5e-5


def test_decode_from_window_aligned_prompt():
    """The opposite seam: the prompt ends exactly on a ratio-4 *and* a ratio-128
    window boundary (and the chunk edge at 64 lands on one too), so decode starts
    from an empty compressor frontier rather than a partial window."""
    prompt = 128
    assert prompt % 4 == 0 and prompt % 128 == 0
    worst = _run_case(prompt, 200, prompt_chunks=2, label="aligned-prompt")
    assert worst <= 5e-5


def test_decode_matches_prefill_batched():
    """b > 1: the cache carries a batch axis, and rows must not leak into each
    other through the window, the compressed rows, or the compressor frontier."""
    worst = _run_case(13, 60, batch=3, label="batched")
    assert worst <= 5e-5
