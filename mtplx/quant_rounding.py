"""Affine rounding with each group's range chosen, not taken whole.

Affine quantization spreads a group's [min, max] over its 2^bits levels, so a
single outlier coarsens the step for every other weight of the group. Pulling
both ends toward the middle clips the outliers a little and refines the step
for the rest. For each group this keeps the pull, from a fixed grid, whose
rounded weights are closest to the originals in squared error. The whole range
is on the grid, so no group rounds worse than plain affine rounding.

The result is the weight with each group clamped to its chosen range. A clamped
group's min and max are the range's ends, so the stock quantizer then produces
exactly the chosen rounding: a pack built this way keeps the standard layout,
and every kernel reads it unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

# The share of each group's half-range kept; 1.0 is plain affine rounding.
SHRINKS = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.7)
ROUNDINGS = ("plain", "clip")


def clipping_predicate(decide: Callable[[str, Any], Any], *, body: dict[str, Any]) -> Callable[[str, Any], Any]:
    """A quantize predicate that also prepares each chosen module's weight for
    the rounding it is about to get.

    ``mlx_lm``'s ``quantize_model`` asks the predicate about each module once,
    just before quantizing it, so this is the one place a converter hands over
    the weight. ``decide`` returns what it always did (False, True for the
    ``body`` parameters, or a parameter dict); affine modules it quantizes get
    their groups clamped by ``clip_for_rounding`` for exactly those parameters.

    The clamp and the search for it stay lazy here: a converter evaluates each
    quantized module as soon as it is built (``evaluate_each_module``), so one
    module's source is resident at a time. Evaluated here, every source would
    be resident before the first shard is written (a 35B-A3B trunk: 65 GB,
    which the memory watchdog ended twice on 16 November).
    """

    def predicate(path: str, module: Any) -> Any:
        params = decide(path, module)
        if params:
            spec = {**body, **params} if isinstance(params, dict) else body
            if spec.get("mode", "affine") == "affine":
                module.weight = clip_for_rounding(
                    module.weight, bits=int(spec["bits"]), group_size=int(spec["group_size"])
                )
        return params

    return predicate


def clip_for_rounding(weight: Any, *, bits: int, group_size: int, shrinks: tuple[float, ...] = SHRINKS) -> Any:
    """``weight`` with each group of ``group_size`` along its last axis clamped
    to the range whose affine rounding to ``bits`` errs least.

    Lazy: nothing is read until the result is evaluated. Each candidate range
    depends on the best one before it, so an evaluation holds one candidate's
    temporaries at a time beside the weight itself.
    """

    import mlx.core as mx

    groups = weight.reshape(-1, group_size)
    low, high = _best_range(groups, bits=bits, group_size=group_size, shrinks=shrinks)
    return mx.minimum(mx.maximum(groups, low), high).reshape(weight.shape)


def _best_range(groups: Any, *, bits: int, group_size: int, shrinks: tuple[float, ...]) -> tuple[Any, Any]:
    import mlx.core as mx

    exact = groups.astype(mx.float32)
    lowest = exact.min(axis=-1, keepdims=True)
    highest = exact.max(axis=-1, keepdims=True)
    middle, half = (lowest + highest) / 2, (highest - lowest) / 2
    best = None
    for shrink in shrinks:
        # The ends in the weight's own dtype: the values a converter will see.
        low = (middle - shrink * half).astype(groups.dtype)
        high = (middle + shrink * half).astype(groups.dtype)
        clamped = mx.minimum(mx.maximum(groups, low), high)
        packed, scales, biases = mx.quantize(clamped, group_size=group_size, bits=bits)
        rounded = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        error = mx.sum(mx.square(rounded.astype(mx.float32) - exact), axis=-1, keepdims=True)
        if best is None:
            best = (error, low, high)
        else:
            better = error < best[0]
            best = tuple(mx.where(better, new, old) for new, old in zip((error, low, high), best))
    return best[1], best[2]


def evaluate_each_module(quantize: Callable[..., Any]) -> Callable[..., Any]:
    """``quantize`` (``mlx_lm``'s ``quantize_model``), then every quantized
    module evaluated in turn, so each module's range search, clamp and rounding
    run while only its own source is resident, and its source is freed before
    the next one is read. The peak is the quantized model plus one module."""

    import mlx.core as mx

    def quantize_then_evaluate(model: Any, *args: Any, **kwargs: Any) -> Any:
        model, config = quantize(model, *args, **kwargs)
        for _, module in model.named_modules():
            if getattr(module, "scales", None) is not None:
                mx.eval(module.parameters())
        return model, config

    return quantize_then_evaluate
