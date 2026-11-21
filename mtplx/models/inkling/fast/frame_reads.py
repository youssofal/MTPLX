"""Two reads of the step's frame on lane L1's read kernel instead of MLX's own launches.

* The loop pass's message stack, ``rms_norm(stack(prior).astype(float32))`` (``step.decode_hidden``):
  MLX copies the prior states into one array, casts it and norms it (four launches a pass). The read
  kernel's mode 3 writes exactly those keys (lane L1: "mx.fast.rms_norm of each source without a
  weight, float32, (W, S, D) as RC6 stacks them"), in one launch.
* The final depth read and the final norm before the head, ``norm(read(closed, x, final_query),
  final_norm).astype(dtype)``: MLX's stack, norm, two matrix-vector products and a softmax, then a
  norm and a cast. The read kernel's mode 0 is that read followed by a gained norm and the bf16 cast
  (the visit's attention input reads take it at every layer), in one launch.

Class A: each form is decided per shape on the first call on this Mac, against the MLX expression
it replaces, bit for bit (one evaluation of both, before the engine records the step); a shape that
differs, a float16 stream (the FP16 siblings: the kernel rounds to bf16) or a CPU run keeps MLX's
launches, counted ``fallback.frame_reads.<why>``. ``MTPLX_INKLING_FRAME_READS=0`` turns both off.
"""

from __future__ import annotations

import os

import mlx.core as mx

from .routes import count

#: (what, shape key) -> True when the read kernel gives the MLX expression's bits on this Mac.
_VERDICT: dict = {}


def enabled():
    return os.environ.get("MTPLX_INKLING_FRAME_READS", "1").strip().lower() not in ("0", "off", "false", "no")


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(view), mx.contiguous(b).view(view)).item())


def _usable(blocks):
    """Why the read kernel cannot take these sources, or None."""
    if not enabled():
        return "off"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if any(b.dtype != mx.bfloat16 for b in blocks):
        return "stream_dtype"
    if len(blocks) > 5 or blocks[0].shape[-1] % 128 or blocks[0].shape[-1] > 4096:
        return "shape"
    return None


def _decide(key, kernel, stock):
    """``kernel()`` where proven for ``key`` on this Mac, else ``stock()`` (counted)."""
    verdict = _VERDICT.get(key)
    if verdict is None:
        try:
            got, want = kernel(), stock()
            mx.eval(got, want)
            verdict = _same(got, want)
        except (RuntimeError, ValueError):
            verdict = False
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "frame_" + "_".join(map(str, key)))
    if not verdict:
        count(f"fallback.frame_reads.{key[0]}_differs")
        return stock()
    return kernel()


def pass_stack(prior, eps, stand_in):
    """``rms_norm(stack(prior, -2).astype(float32), None, eps)`` for a loop pass's prior states
    ``prior`` (each ``(1, W, D)``): ``(1, W, P, D)`` float32. ``stand_in``: any float32 array
    (fills the read kernel's query slot; mode 3 reads no query)."""

    def stock():
        return mx.fast.rms_norm(mx.stack(prior, axis=-2).astype(mx.float32), None, eps)

    why = _usable(prior)
    if why is not None:
        if why != "off":
            count(f"fallback.frame_reads.stack_{why}")
        return stock()
    from .read import depth_keys

    rows, width = prior[-1].shape[1], prior[-1].shape[-1]
    queries = stand_in.reshape(-1)[:width].reshape(1, width)
    return _decide(("stack", rows, len(prior), width),
                   lambda: depth_keys(prior[-1], list(prior[:-1]), queries, eps), stock)


def final_hidden(model, panel, closed, x, stock):
    """The final read and norm before the head, ``(1, W, D)`` in ``x``'s dtype: the read kernel's
    mode 0 where proven, else ``stock()`` (the reference's two lines)."""
    why = _usable([*closed, x]) if model.config.get("attn_res_normalize_values", True) else "values"
    if why is not None:
        if why != "off":
            count(f"fallback.frame_reads.final_{why}")
        return stock()
    from .read import read

    rows, width = x.shape[1], x.shape[-1]

    def kernel():
        (out,) = read(x.reshape(rows, width), [c.reshape(rows, width) for c in closed], model.final_query,
                      model.eps, mode=0, gains=[model.final_norm])
        return out.reshape(1, rows, width)

    return _decide(("final", rows, len(closed), width, str(model.final_norm.dtype)), kernel, stock)


__all__ = ["enabled", "final_hidden", "pass_stack"]
