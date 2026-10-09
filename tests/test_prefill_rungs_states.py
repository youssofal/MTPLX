"""Prefill rungs that carry the GDN states (MTPLX_PREFILL_ASYNC_RUNGS_STATES).

On the tiny quantized Qwen3.5-MoE with a draft head, run through the real
cold streaming prefill loop:

* with rungs (with and without the states) every result bit equals the run
  without rungs: logits, hidden, every trunk cache leaf, every banked
  boundary and the draft cache;
* with the states on, rungs name the recurrent states of the finished layers
  (counter) and draft-head history appends never fire a rung;
* ``recurrent_state_arrays`` names conv tail, delta state and captures of
  recurrent entries only.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm.models.qwen3_5_moe")

from mlx_lm.models import base, qwen3_5, qwen3_next
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx import generation, prefill_rungs
from mtplx.cache_state import CacheSnapshot
from tests.a3b_tiny_synth import assert_bit_equal, prompt, tiny_model_with_draft_head

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


class _Runtime:
    mtp_enabled = True

    def __init__(self, model, tmp_path):
        self.model = model
        self.model_path = tmp_path
        self.diagnostic_counters: dict[str, int] = {}

    def make_cache(self):
        return self.model.make_cache()

    def make_mtp_cache(self):
        return self.model.make_mtp_cache()

    @property
    def embed_tokens(self):
        return self.model.language_model.model.embed_tokens

    def forward_ar(self, tokens, *, cache, return_hidden=False, hidden_variant=None,
                   emit_logits=True, logits_keep=None, input_embeddings=None):
        return self.model(tokens, cache=cache, return_hidden=return_hidden,
                          hidden_variant=hidden_variant, emit_logits=emit_logits,
                          logits_keep=logits_keep, input_embeddings=input_embeddings)

    def update_mtp_cache(self, hidden_states, token_ids, *, mtp_cache, **kwargs):
        return self.model.mtp_update_cache(hidden_states, token_ids, mtp_cache=mtp_cache, **kwargs)


@pytest.fixture()
def lane(monkeypatch):
    """Scope the rung wrapper to one test."""

    monkeypatch.setattr(qwen3_next, "scaled_dot_product_attention", qwen3_next.scaled_dot_product_attention)
    monkeypatch.setattr(base, "scaled_dot_product_attention", base.scaled_dot_product_attention)
    monkeypatch.setattr(qwen3_5.DecoderLayer, "__call__", qwen3_5.DecoderLayer.__call__)
    monkeypatch.setattr(qwen3_5, "_mtplx_prefill_rungs_installed", False, raising=False)
    monkeypatch.setattr(prefill_rungs, "_STATE", {"idx": 0, "pending": []})
    monkeypatch.setattr(prefill_rungs, "COUNTERS", dict.fromkeys(prefill_rungs.COUNTERS, 0))
    monkeypatch.setattr("atexit.register", lambda *_a, **_k: None)
    for name, value in (
        ("MTPLX_SUSTAINED_PREFILL", "1"),
        ("MTPLX_PREFILL_CHUNK_SIZE", "96"),
        ("MTPLX_GDN_BOUNDARY_TAIL_INTERVAL", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "3"),
        ("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "16"),
        ("MTPLX_PREFILL_ASYNC_RUNGS_MIN_SEQ", "32"),
        ("MTPLX_MTP_HISTORY_CACHE_ONLY", "1"),
    ):
        monkeypatch.setenv(name, value)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return tiny_model_with_draft_head(tmp_path_factory.mktemp("rungs"))


def _leaves(value) -> list:
    if value is None:
        return []
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, CacheSnapshot):
        return _leaves(list(value.states))
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


def _cold(model, tokens, tmp_path):
    rt = _Runtime(model, tmp_path)
    sink: list = []
    out = generation._prefill_committed_mtp_history_streaming(rt, list(tokens), gdn_boundary_sink=sink)
    cache, logits, hidden, mtp_cache = out[:4]
    mx.eval(logits, hidden)
    return (
        [logits, hidden]
        + [leaf for entry in cache for leaf in _leaves(entry.state)]
        + _leaves([(record[1], record[2]) for record in sink])
        + list(mtp_cache[0].state),
        [int(record[0]) for record in sink],
    )


@needs_metal
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("carry", ["0", "1"])
def test_rungs_leave_every_bit_unchanged(lane, model, monkeypatch, tmp_path, stride, carry):
    stock_call = qwen3_5.DecoderLayer.__call__
    tokens = prompt(263, seed=13)
    reference, edges = _cold(model, tokens, tmp_path)

    monkeypatch.setenv("MTPLX_PREFILL_ASYNC_RUNGS", str(stride))
    monkeypatch.setenv("MTPLX_PREFILL_ASYNC_RUNGS_STATES", carry)
    assert prefill_rungs.install_qwen3_5_prefill_rungs() is True
    assert qwen3_5.DecoderLayer.__call__ is not stock_call
    ours, our_edges = _cold(model, tokens, tmp_path)

    assert prefill_rungs.COUNTERS["rungs_dispatched"] > 0
    if carry == "1":
        assert prefill_rungs.COUNTERS["state_arrays_dispatched"] > 0
    else:
        assert prefill_rungs.COUNTERS["state_arrays_dispatched"] == 0
    assert our_edges == edges
    assert_bit_equal(ours, reference)


def test_draft_history_appends_fire_no_rung_with_the_states(lane, model, monkeypatch):
    monkeypatch.setenv("MTPLX_PREFILL_ASYNC_RUNGS", "1")
    monkeypatch.setenv("MTPLX_PREFILL_ASYNC_RUNGS_STATES", "1")
    prefill_rungs.install_qwen3_5_prefill_rungs()
    layer = model.language_model.mtp.layers[0]
    x = mx.zeros((1, 64, 128), dtype=mx.bfloat16)
    with prefill_rungs.draft_history_scope():
        layer(x, mask="causal", cache=KVCache())
    assert prefill_rungs.COUNTERS["wide_layer_calls"] == 0
    layer(x, mask="causal", cache=KVCache())
    assert prefill_rungs.COUNTERS["wide_layer_calls"] == 1


def test_recurrent_state_arrays_name_states_and_captures_only():
    recurrent = ArraysCache(size=2)
    recurrent[0] = mx.zeros((1, 3, 8))
    recurrent[1] = mx.ones((1, 2, 4, 4))
    setattr(recurrent, prefill_rungs.BOUNDARY_CAPTURE_ATTR, {5: (mx.zeros((1, 3, 8)), mx.ones((1, 2, 4, 4)))})
    empty = ArraysCache(size=2)
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)))
    arrays = prefill_rungs.recurrent_state_arrays([recurrent, empty, kv])
    assert len(arrays) == 4
    assert arrays[0] is recurrent[0] and arrays[1] is recurrent[1]


def test_states_are_off_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_PREFILL_ASYNC_RUNGS_STATES", raising=False)
    assert prefill_rungs.rungs_carry_states() is False
