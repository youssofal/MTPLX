"""Clipped rounding: never worse than plain rounding, exactly what the stock
quantizer then stores, and plain rounding itself when only the whole range is
allowed."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.commands.forge_mixed_convert import build_predicate
from mtplx.quant_rounding import SHRINKS, clip_for_rounding, clipping_predicate


def _group_errors(original: mx.array, stored: mx.array, bits: int) -> mx.array:
    """Squared error per group of 64 of ``stored`` as the stock quantizer keeps it."""

    packed, scales, biases = mx.quantize(stored, group_size=64, bits=bits)
    rounded = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
    diff = rounded.astype(mx.float32) - original.astype(mx.float32)
    return mx.sum(mx.square(diff.reshape(-1, 64)), axis=-1)


def _weights_with_outliers() -> mx.array:
    mx.random.seed(7)
    weight = mx.random.normal((128, 512))
    spikes = mx.where(mx.random.uniform(shape=weight.shape) < 1 / 64, 6.0, 1.0)
    return (weight * spikes).astype(mx.bfloat16)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_clipping_never_rounds_a_group_worse_and_helps_outlier_groups(bits):
    weight = _weights_with_outliers()
    plain = _group_errors(weight, weight, bits)
    clipped = _group_errors(weight, clip_for_rounding(weight, bits=bits, group_size=64), bits)
    assert bool(mx.all(clipped <= plain * (1 + 1e-6) + 1e-12).item())
    if bits == 4:
        # On these heavy-tailed weights clipping saves about 5% of the squared error.
        assert float(clipped.sum().item()) < 0.98 * float(plain.sum().item())


def test_the_stored_rounding_is_the_best_on_the_grid():
    weight = _weights_with_outliers()
    groups = weight.reshape(-1, 64)
    lowest = groups.astype(mx.float32).min(axis=-1, keepdims=True)
    highest = groups.astype(mx.float32).max(axis=-1, keepdims=True)
    middle, half = (lowest + highest) / 2, (highest - lowest) / 2
    per_shrink = [
        _group_errors(
            weight,
            mx.minimum(mx.maximum(groups, (middle - s * half).astype(weight.dtype)),
                       (middle + s * half).astype(weight.dtype)).reshape(weight.shape),
            4,
        )
        for s in SHRINKS
    ]
    best = mx.min(mx.stack(per_shrink), axis=0)
    stored = _group_errors(weight, clip_for_rounding(weight, bits=4, group_size=64), 4)
    assert mx.allclose(stored, best, rtol=1e-6, atol=1e-12).item()


def test_the_whole_range_alone_is_plain_rounding():
    weight = _weights_with_outliers()
    assert mx.array_equal(clip_for_rounding(weight, bits=4, group_size=64, shrinks=(1.0,)), weight).item()


def test_expert_banks_keep_their_shape():
    weight = mx.random.normal((4, 32, 128)).astype(mx.bfloat16)
    assert clip_for_rounding(weight, bits=4, group_size=64).shape == weight.shape


def test_the_predicate_clips_for_the_bits_each_module_gets():
    recipe = {"body_bits": 8, "module_overrides": [
        {"suffix": "mlp.down_proj", "bits": 4, "group_size": 64},
        {"suffix": "mlp.gate", "quantize": False},
    ]}
    body = {"bits": 8, "group_size": 64, "mode": "affine"}
    predicate = clipping_predicate(build_predicate(recipe), body=body)
    modules = {name: nn.Linear(128, 64, bias=False) for name in ("mlp.down_proj", "mlp.up_proj", "mlp.gate")}
    original = {name: m.weight for name, m in modules.items()}
    decisions = {name: predicate(f"model.layers.0.{name}", m) for name, m in modules.items()}

    assert decisions["mlp.down_proj"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert mx.array_equal(modules["mlp.down_proj"].weight,
                          clip_for_rounding(original["mlp.down_proj"], bits=4, group_size=64)).item()
    assert decisions["mlp.up_proj"] is True  # the body's 8 bits
    assert mx.array_equal(modules["mlp.up_proj"].weight,
                          clip_for_rounding(original["mlp.up_proj"], bits=8, group_size=64)).item()
    assert decisions["mlp.gate"] is False and modules["mlp.gate"].weight is original["mlp.gate"]


def test_each_quantized_module_is_evaluated_as_soon_as_it_is_built(monkeypatch):
    """A converter's clipped rounding reads one module's source at a time: the
    quantizer's result is evaluated module by module, quantized modules only."""

    import mlx.core as mx
    import mlx.nn as nn

    from mtplx.quant_rounding import evaluate_each_module

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.kept = nn.Linear(64, 8)
            self.rounded = nn.Linear(64, 8)

    def quantize(model, config):
        model.rounded = model.rounded.to_quantized(group_size=64, bits=4)
        return model, {**config, "quantization": {"bits": 4}}

    evaluated = []
    real_eval = mx.eval
    monkeypatch.setattr(mx, "eval", lambda *trees: evaluated.append(trees) or real_eval(*trees))
    model, config = evaluate_each_module(quantize)(Two(), {})
    assert config == {"quantization": {"bits": 4}}
    assert len(evaluated) == 1 and set(evaluated[0][0]) == {"weight", "scales", "biases", "bias"}
