"""A single working buffer for the model's native full-vocabulary categorical draw."""

from bisect import bisect_right

import mlx.core as mx
import numpy as np

from mtplx.sampling import non_finite_logits_error


def is_native_sampler(config):
    return (
        config.temperature == 1.0
        and config.top_p == 1.0
        and not config.top_k
        and not config.presence_penalty
        and not config.frequency_penalty
    )


def draw_native(logits, rng):
    row = np.asarray(logits.astype(mx.float32)).reshape(-1)
    mass = row.astype(np.float64)
    mass -= mass.max()
    np.exp(mass, out=mass)
    total = mass.sum()
    if not np.isfinite(total) or total <= 0:
        raise non_finite_logits_error(row, "softmax")

    # Keep the reference's softmax, unfiltered support and categorical
    # normalizations in the same order, including their FP64 rounding.
    mass /= total
    mass /= mass.sum()
    mass /= mass.sum()
    np.cumsum(mass, out=mass)
    total = mass[-1]
    return bisect_right(mass, rng.random(), key=lambda value: value / total)
