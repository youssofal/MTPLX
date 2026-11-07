from types import SimpleNamespace
from unittest.mock import patch

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm.models.minicpmv4_7")
from mtplx.models.minicpmv4_7 import Model, UpstreamModel
from mtplx.runtime import _model_classes_for_config


def config(layers=40):
    return SimpleNamespace(
        model_type="minicpmv4_7",
        mrope_mode="canvas",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_hidden_layers=layers,
            hidden_size=2048,
            num_experts=256,
            num_experts_per_tok=8,
        ),
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
    )


class Language:
    def __init__(self):
        self.model = SimpleNamespace(fa_idx=0)
        self.calls = []
        self.head_calls = 0

    def __call__(self, ids, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs.get("inputs_embeds", mx.ones((1, ids.shape[1], 4)))
        return SimpleNamespace(hidden_states=[hidden])

    def lm_head(self, hidden):
        self.head_calls += 1
        return hidden * 2

    def make_cache(self):
        return [SimpleNamespace(offset=0)]


def model():
    result = Model.__new__(Model)
    nn.Module.__init__(result)
    result.language_model = Language()
    return result


def test_runtime_dispatch_and_signature():
    assert _model_classes_for_config({"model_type": "minicpmv4_7"})[0] is Model
    with patch.object(UpstreamModel, "__init__", return_value=None) as init:
        Model(config())
        init.assert_called_once()
        with pytest.raises(ValueError, match="Unqualified MiniCPM architecture"):
            Model(config(layers=39))


def test_logits_contract_skips_head_and_selects_final_rows():
    target = model()
    ids = mx.array([[1, 2, 3]])
    logits, hidden = target(ids, return_hidden=True, logits_keep=1)
    assert logits.shape == (1, 1, 4) and hidden.shape == (1, 3, 4)
    assert mx.array_equal(logits, hidden[:, -1:] * 2).item()
    assert target.language_model.calls[-1]["skip_logits"] is True
    assert target(ids, emit_logits=False) is None
    assert target.language_model.head_calls == 1


def test_chunked_vision_prefill_uses_matching_ids_and_canvas_positions():
    target = model()
    ids = mx.array([[1, 2, 3, 4]])
    embeds = mx.arange(16).reshape(1, 4, 4).astype(mx.float32)
    positions = mx.arange(12).reshape(3, 1, 4)
    features = SimpleNamespace(
        inputs_embeds=embeds, position_ids=positions, rope_deltas=mx.array([[2]])
    )
    target.get_input_embeddings = lambda *args, **kwargs: features
    assert target.prepare_vision({"input_ids": ids}) == [1, 2, 3, 4]
    cache = target.make_cache()
    target(ids[:, :2], cache=cache)
    assert mx.array_equal(
        target.language_model.calls[-1]["inputs_embeds"], embeds[:, :2]
    ).item()
    cache[0].offset = 2
    target(ids[:, 2:], cache=cache)
    assert mx.array_equal(
        target.language_model.calls[-1]["position_ids"], positions[:, :, 2:]
    ).item()
    with pytest.raises(ValueError, match="Vision prefill IDs"):
        target(mx.array([[9, 9]]), cache=cache)
    cache[0].offset = 4
    target(mx.array([[5]]), cache=cache)
    assert "inputs_embeds" not in target.language_model.calls[-1]
    target.clear_vision()
    target.make_cache()
    assert target.language_model._position_ids is None
    assert target.language_model._rope_deltas is None


def test_prepared_vision_rejects_multiple_prompt_batches():
    with pytest.raises(ValueError, match="one prompt per request"):
        model().prepare_vision({"input_ids": mx.array([[1], [2]])})
