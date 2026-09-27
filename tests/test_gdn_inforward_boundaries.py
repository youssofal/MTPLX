"""In-forward GDN boundaries for mlx-lm's Qwen3.5/3.6 (A3B), on the invariant lane.

On a tiny quantized Qwen3.5-MoE (the A3B layout: three GDN layers per full
attention layer, routed plus shared experts) run through the real prefill
loops, these tests pin:

* installed only on the batch-invariant prefill lane; without it (or with
  ``MTPLX_GDN_BOUNDARY_INFORWARD=0``) the loops run the ladder as before;
* with the hooks the loops run the plain chunk grid, so fewer forwards
  (also in the ``MTPLX_PREFILL_CHUNK_TRACE`` records);
* every banked boundary sits at the ladder's position and equals the
  ladder's bit for bit (recurrent states and hidden row), and the prefill
  result (logits, hidden, cache) equals the ladder's bit for bit.

The bit-identity tests need Metal: the lane's claim is about Metal kernels.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm.models.qwen3_5_moe")

from mlx_lm.models import base, qwen3_5, qwen3_5_moe, qwen3_next

from mtplx import batch_invariant_prefill as bip
from mtplx import gdn_inforward_boundaries as gib
from mtplx import generation
from mtplx.attention_context import attention_phase
from mtplx.cache_state import CacheSnapshot

SWITCH = "MTPLX_GDN_BOUNDARY_INFORWARD"
needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _tiny_model(experts: int = 32, top_k: int = 4) -> qwen3_5_moe.Model:
    text = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "vocab_size": 256,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        # 32 experts, top 4: with 8 experts MLX picks another expert kernel
        # below 33 tokens even on the lane, which A3B (256 experts) never does.
        "num_experts": experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "intermediate_size": 64,
    }
    args = qwen3_5_moe.ModelArgs(model_type="qwen3_5_moe", text_config=text)
    mx.random.seed(0)
    model = qwen3_5_moe.Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


class _Runtime:
    """The runtime surface the prefill loops use, over the tiny model."""

    def __init__(self, model):
        self.model = model
        self.mtp_enabled = True
        self.model_path = Path("tiny-a3b-inforward")
        self.diagnostic_counters: dict[str, int] = {}
        self.forwards: list[int] = []

    def make_cache(self):
        return self.model.make_cache()

    def make_mtp_cache(self):
        return []

    @property
    def embed_tokens(self):
        return self.model.language_model.model.embed_tokens

    def forward_ar(
        self,
        tokens,
        *,
        cache,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
        input_embeddings=None,
    ):
        self.forwards.append(int(tokens.shape[1]))
        text = self.model.language_model
        hidden = text.model(tokens, cache, input_embeddings=input_embeddings)
        logits = text.lm_head(hidden) if emit_logits else None
        return (logits, hidden) if return_hidden else logits

    def update_mtp_cache(self, hidden_states, token_ids, **_kwargs):
        return hidden_states.sum()


@pytest.fixture()
def lane(monkeypatch):
    """Scope the invariant lane's process-wide hooks to one test."""

    monkeypatch.setattr(qwen3_next, "scaled_dot_product_attention", qwen3_next.scaled_dot_product_attention)
    monkeypatch.setattr(base, "scaled_dot_product_attention", base.scaled_dot_product_attention)
    monkeypatch.setitem(bip._STOCK_SDPA, "sdpa", None)
    monkeypatch.setitem(bip._STATE, "installed", False)
    monkeypatch.setitem(bip._STATE, "refusal", None)
    # Small numbers so a 160-token prompt has a real ladder: chunks of 96,
    # rungs of 8 on an 8-token grid, the nearest boundary 3 tokens before the
    # end, and a restore floor low enough to keep the cold tail grid.
    for name, value in (
        ("MTPLX_SUSTAINED_PREFILL", "1"),
        ("MTPLX_PREFILL_CHUNK_SIZE", "96"),
        ("MTPLX_GDN_BOUNDARY_TAIL_INTERVAL", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "3"),
        ("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "16"),
        ("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0"),
        ("MTPLX_PREFILL_CHUNK_TRACE", "1"),
    ):
        monkeypatch.setenv(name, value)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    yield
    mx.set_default_device(previous)


def _install(model) -> int:
    bip.install_batch_invariant_prefill(model)
    return gib.install_gdn_inforward_boundaries(model)


def _prompt(tokens: int, seed: int = 7) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(token) for token in rng.integers(0, 256, size=tokens)]


def _np(value):
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    return np.array(value)


def _leaves(value) -> list:
    if value is None:
        return []
    if isinstance(value, mx.array):
        return [_np(value)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


def _assert_bit_equal(left: list, right: list) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a, b)


def _record_leaves(record) -> list:
    _position, snapshot, hidden = record
    assert isinstance(snapshot, CacheSnapshot)
    return _leaves(list(snapshot.states)) + _leaves(hidden)


def _cache_leaves(cache) -> list:
    return [leaf for entry in cache for leaf in _leaves(entry.state)]


def _cold(model, prompt, *, inforward, monkeypatch):
    monkeypatch.setenv(SWITCH, "1" if inforward else "0")
    rt = _Runtime(model)
    sink: list = []
    out = generation._prefill_committed_mtp_history_streaming(
        rt, list(prompt), gdn_boundary_sink=sink
    )
    mx.eval(out[1], out[2])
    return rt, sink, out[0], out[1], out[2], generation.prefill_chunk_records()


def _warm(model, prompt, cached, *, inforward, monkeypatch):
    """Prefill ``prompt[:cached]``, then the rest through the restored-suffix
    loop, the way a warm agent turn runs after a bank restore."""

    rt = _Runtime(model)
    cache = rt.make_cache()
    mx.eval(rt.forward_ar(mx.array([prompt[:cached]]), cache=cache))
    rt.forwards.clear()
    monkeypatch.setenv(SWITCH, "1" if inforward else "0")
    restored = SimpleNamespace(
        cache=cache, mtp_history_cache=[], hidden=None, entry=SimpleNamespace(prefix_len=cached)
    )
    sink: list = []
    logits, hidden, _forward_s, _history_s = generation._prefill_restored_prompt_suffix(
        rt,
        restored,
        list(prompt[cached:]),
        base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm",
        mtp_history_policy="committed",
        cached_tokens=cached,
        gdn_boundary_sink=sink,
    )
    mx.eval(logits, hidden)
    return rt, sink, cache, logits, hidden, generation.prefill_chunk_records()


# ---------------------------------------------------------------------------
# Installation and the switch
# ---------------------------------------------------------------------------


def test_install_swaps_the_gdn_layers_and_publishes_the_hooks(lane):
    model = _tiny_model()
    before = sorted(key for key, _ in tree_flatten(model.parameters()))
    assert _install(model) == 3
    assert sorted(key for key, _ in tree_flatten(model.parameters())) == before
    layers = model.language_model.model.layers
    assert [isinstance(layer.linear_attn, gib._BoundaryCaptureMixin) for layer in layers[:3]] == [True] * 3
    assert all(isinstance(layer.linear_attn, qwen3_5.GatedDeltaNet) for layer in layers[:3])
    assert generation._resolve_inforward_boundary_hooks(_Runtime(model)) is not None


def test_a_model_without_gdn_layers_gets_no_hooks():
    model = nn.Sequential(nn.Linear(8, 8))
    assert gib.install_gdn_inforward_boundaries(model) == 0
    assert not hasattr(model, "boundary_capture_scope")


def test_without_the_invariant_lane_there_are_no_hooks(lane, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    assert generation._resolve_inforward_boundary_hooks(_Runtime(_tiny_model())) is None


def test_the_runtime_installs_the_hooks_only_with_the_lane():
    source = Path(generation.__file__).with_name("runtime.py").read_text()
    lane_block = source.split("if batch_invariant_prefill_enabled():", 1)[1].split("logger.info", 1)[0]
    assert "_install_batch_invariant_lane(model)" in lane_block
    helper = source.split("def _install_batch_invariant_lane(", 1)[1].split("\ndef ", 1)[0]
    refused, installed = helper.split("return\n", 1)
    assert "install_gdn_inforward_boundaries(model)" not in refused
    assert "install_gdn_inforward_boundaries(model)" in installed
    assert source.count("install_gdn_inforward_boundaries(") == 1


def test_the_runtime_installs_lane_and_hooks_on_the_a3b_layout(lane):
    from mtplx import runtime

    model = _tiny_model()
    runtime._install_batch_invariant_lane(model)
    assert bip.batch_invariant_prefill_installed() is True
    assert isinstance(model.language_model.model.layers[0].linear_attn, gib._BoundaryCaptureMixin)
    assert generation._resolve_inforward_boundary_hooks(_Runtime(model)) is not None
    status = bip.batch_invariant_prefill_status()
    assert status["installed"] is True
    assert status["gdn_inforward_layers"] == 3
    assert status["switch_glus"] == 4


def test_a_refused_model_gets_neither_lane_nor_hooks(lane, monkeypatch):
    from mtplx import runtime

    # 8 experts, top 2: the SwitchGLU padding is not row invariant there.
    model = _tiny_model(experts=8, top_k=2)
    before = [type(module) for _name, module in model.named_modules()]
    runtime._install_batch_invariant_lane(model)
    assert [type(module) for _name, module in model.named_modules()] == before
    assert bip.batch_invariant_prefill_installed() is False
    assert not hasattr(model, "boundary_capture_scope")
    assert bip.batch_invariant_prefill_status() == {
        "installed": False,
        "reason": "switch_glu_experts:8",
    }
    # The chunk layout stays the ladder's, as without the switch.
    prompt = _prompt(231, seed=11)
    rt, *_rest = _warm(model, prompt, 60, inforward=True, monkeypatch=monkeypatch)
    plan = generation._prefill_spans_with_tail_grid(170, tail_interval=8, chunk_size=96)
    assert rt.forwards == [end - start for start, end in plan] + [1]
    assert "prefill_inforward_boundary_captures" not in rt.diagnostic_counters


def test_zero_restores_the_ladder(lane, monkeypatch):
    model = _tiny_model()
    _install(model)
    monkeypatch.setenv(SWITCH, "0")
    assert generation._resolve_inforward_boundary_hooks(_Runtime(model)) is None


# ---------------------------------------------------------------------------
# The capture itself
# ---------------------------------------------------------------------------


def test_nothing_armed_is_the_stock_call(lane):
    model = _tiny_model()
    stock_cache = model.make_cache()
    tokens = mx.array([_prompt(40)])
    stock = model(tokens, cache=stock_cache)
    _install(model)
    cache = model.make_cache()
    ours = model(tokens, cache=cache)
    assert np.array_equal(_np(stock), _np(ours))
    assert gib.take_boundary_captures(cache) == {}


def test_captures_are_taken_once_and_only_when_complete(lane):
    model = _tiny_model()
    _install(model)
    cache = model.make_cache()
    with gib.boundary_capture_scope([0, 12, 25, 40, -3]):
        mx.eval(model(mx.array([_prompt(30)]), cache=cache))
    captures = gib.take_boundary_captures(cache)
    assert sorted(captures) == [12, 25]
    states = captures[12]
    assert [state is None for state in states] == [False, False, False, True]
    assert all(len(state) == 2 for state in states[:3])
    assert gib.take_boundary_captures(cache) == {}
    # One GDN layer without its record: that offset is no boundary.
    with gib.boundary_capture_scope([12]):
        mx.eval(model(mx.array([_prompt(30)]), cache=model.make_cache()))
    partial = model.make_cache()
    setattr(partial[0], gib._CAPTURE_ATTR, {12: ("conv", "state")})
    assert gib.take_boundary_captures(partial) == {}


@needs_metal
def test_a_captured_state_is_the_state_of_a_forward_ending_there(lane):
    model = _tiny_model()
    _install(model)
    prompt = _prompt(60)
    edge = 37
    with attention_phase("prefill"):
        cache = model.make_cache()
        with gib.boundary_capture_scope([edge]):
            mx.eval(model(mx.array([prompt]), cache=cache))
        captured = gib.take_boundary_captures(cache)[edge]
        ended = model.make_cache()
        mx.eval(model(mx.array([prompt[:edge]]), cache=ended))
    for states, entry in zip(captured[:3], ended[:3]):
        _assert_bit_equal(_leaves(states), _leaves(entry.state))


# ---------------------------------------------------------------------------
# The loops: in-forward against the ladder
# ---------------------------------------------------------------------------


@needs_metal
def test_cold_prefill_matches_the_ladder_bit_for_bit_with_fewer_forwards(lane, monkeypatch):
    model = _tiny_model()
    _install(model)
    prompt = _prompt(161)
    ladder = _cold(model, prompt, inforward=False, monkeypatch=monkeypatch)
    ours = _cold(model, prompt, inforward=True, monkeypatch=monkeypatch)
    ladder_rt, ladder_sink, ladder_cache, ladder_logits, ladder_hidden, ladder_trace = ladder
    rt, sink, cache, logits, hidden, trace = ours

    assert rt.forwards == [96, 64, 1]
    assert len(ladder_rt.forwards) > len(rt.forwards)
    assert len(trace) == 2 and len(ladder_trace) == len(ladder_rt.forwards) - 1
    assert sum(record.get("inforward_boundaries", 0.0) for record in trace) == len(sink) - 2
    assert rt.diagnostic_counters["prefill_inforward_boundary_captures"] >= 2

    assert [int(record[0]) for record in sink] == [int(record[0]) for record in ladder_sink]
    for mine, theirs in zip(sink, ladder_sink):
        _assert_bit_equal(_record_leaves(mine), _record_leaves(theirs))
    _assert_bit_equal(_leaves([logits, hidden]), _leaves([ladder_logits, ladder_hidden]))
    _assert_bit_equal(_cache_leaves(cache), _cache_leaves(ladder_cache))


@needs_metal
def test_warm_suffix_matches_the_ladder_bit_for_bit_with_fewer_forwards(lane, monkeypatch):
    model = _tiny_model()
    _install(model)
    prompt = _prompt(231, seed=11)
    ladder = _warm(model, prompt, 60, inforward=False, monkeypatch=monkeypatch)
    ours = _warm(model, prompt, 60, inforward=True, monkeypatch=monkeypatch)
    ladder_rt, ladder_sink, ladder_cache, ladder_logits, ladder_hidden, ladder_trace = ladder
    rt, sink, cache, logits, hidden, trace = ours

    assert rt.forwards == [96, 74, 1]
    assert len(ladder_rt.forwards) > len(rt.forwards)
    assert len(ladder_trace) > len(trace) == 2
    assert len(sink) >= 3
    assert [int(record[0]) for record in sink] == [int(record[0]) for record in ladder_sink]
    for mine, theirs in zip(sink, ladder_sink):
        _assert_bit_equal(_record_leaves(mine), _record_leaves(theirs))
    _assert_bit_equal(_leaves([logits, hidden]), _leaves([ladder_logits, ladder_hidden]))
    _assert_bit_equal(_cache_leaves(cache), _cache_leaves(ladder_cache))


@needs_metal
def test_without_the_lane_the_loop_runs_the_ladder(lane, monkeypatch):
    model = _tiny_model()
    prompt = _prompt(231, seed=11)
    rt, sink, *_rest = _warm(model, prompt, 60, inforward=True, monkeypatch=monkeypatch)
    plan = generation._prefill_spans_with_tail_grid(170, tail_interval=8, chunk_size=96)
    assert rt.forwards == [end - start for start, end in plan] + [1]
    assert "prefill_inforward_boundary_captures" not in rt.diagnostic_counters
    assert sink
