from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")

from mtplx.attention_split import configure_split_full_attention


def _tiny_gemma4():
    args = gemma4_text.ModelArgs.from_dict(
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "global_head_dim": 32,
            "num_global_key_value_heads": 1,
            "attention_k_eq_v": True,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
            "vocab_size": 100,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1e6,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
            },
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
        }
    )
    return gemma4_text.Model(args)


def test_gemma4_attention_is_not_hooked_and_still_runs(monkeypatch):
    # A plain Gemma 4 load (runtime.load(..., mtp=False)) takes the generic
    # path. The generic hook has no shared_kv/offset parameters, so hooking
    # Gemma 4 attention crashed the first forward.
    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    with mx.stream(mx.cpu):
        model = _tiny_gemma4()
        tokens = mx.array([[1, 2, 3]])
        want = model(tokens, cache=model.make_cache())
        mx.eval(want)

        stats = configure_split_full_attention(model)
        got = model(tokens, cache=model.make_cache())
        mx.eval(got)

    assert stats["installed"] == 0
    assert stats["layers"] == 0
    assert mx.array_equal(got, want).item()
