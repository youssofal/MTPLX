"""The relative-position profile of a visit, ``rel @ rel_basis``, as our launch (lane F1).

RC6 multiplies each visit's relative queries ``rel`` ``(1, W, 24, 16)`` bf16 by the layer's basis
``(16, 1024)`` with one library product on the matrix units: 48 launches a one-token step, each a
single remainder step of the library's tile loop (K = 16, padded to its 32), accumulated in float32
and stored as bf16. This launch issues the same two matrix-unit multiplies per 16 rows and 32
columns (the installed MLX's own fragment layout and multiply-accumulate, ``steel/gemm/nax.h``,
read from the installed package when the launch is built, as ``nax_reads.py`` does), with the same
zero padding and the same float32-to-bf16 store, so its outputs carry the library's bits; one
simdgroup per (16-row block, 32-column tile).

Class A, decided per shape on this Mac on the first call against the library product, bit for bit;
a float16 stream (the FP16 siblings), another K or a shape that differs keeps the library's launch,
counted ``fallback.nax_profile.<why>``. ``MTPLX_INKLING_NAX_PROFILE=0`` turns it off.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from .nax_reads import library
from .routes import count

#: (rows, K, columns, basis dtype) -> True when the launch gives the library's bits on this Mac.
_VERDICT: dict = {}
SIMDGROUPS = 8

_SOURCE = """
    using Frag = mlx::steel::BaseNAXFrag;
    typedef Frag::dtype_frag_t<float> acc_t;
    typedef Frag::dtype_frag_t<bfloat16_t> in_t;
    constexpr uint RB = (M + 15) / 16;       // 16-row blocks of rel
    constexpr uint TILES = N / 32;           // 32-column tiles of the profile
    const uint item = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
    if (item >= RB * TILES) return;
    const uint rb = item / TILES, tile = item % TILES;
    acc_t lo = acc_t(0.0f), hi = acc_t(0.0f);
    // K padded to the library's 32: two multiplies of 16 (the second over zeros when K = 16).
    for (uint kk = 0; kk < 2; ++kk) {
        in_t a, b0, b1;
        for (short i = 0; i < 8; ++i) {
            const short2 at = Frag::get_coord(i);
            const uint m = rb * 16 + uint(at.y), k = kk * 16 + uint(at.x);
            a[i] = (m < M && k < K) ? rel[m * K + k] : bfloat16_t(0.0f);
            const uint kr = kk * 16 + uint(at.y), col = tile * 32 + uint(at.x);
            b0[i] = kr < K ? static_cast<bfloat16_t>(basis[kr * N + col]) : bfloat16_t(0.0f);
            b1[i] = kr < K ? static_cast<bfloat16_t>(basis[kr * N + col + 16]) : bfloat16_t(0.0f);
        }
        Frag::mma(lo, hi, a, metal::bool_constant<false>{}, b0, b1, metal::bool_constant<false>{});
    }
    for (short i = 0; i < 8; ++i) {
        const short2 at = Frag::get_coord(i);
        const uint m = rb * 16 + uint(at.y);
        if (m < M) {
            const uint o = m * N + tile * 32 + uint(at.x);
            out[o] = static_cast<bfloat16_t>(lo[i]);
            out[o + 16] = static_cast<bfloat16_t>(hi[i]);
        }
    }
"""


def enabled():
    return os.environ.get("MTPLX_INKLING_NAX_PROFILE", "1").strip().lower() not in ("0", "off", "false", "no")


@lru_cache(maxsize=1)
def _kernel():
    text, _why = library()
    return mx.fast.metal_kernel(name="mtplx_inkling_nax_profile", input_names=["rel", "basis"],
                                output_names=["out"], source=_SOURCE,
                                header=text + "\nusing namespace mlx::steel;\n")


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(mx.uint16), mx.contiguous(b).view(mx.uint16)).item())


def _launch(rel, basis):
    k, n = basis.shape
    rows = rel.size // k
    items = -(-rows // 16) * (n // 32)
    return _kernel()(
        inputs=[rel, basis],
        template=[("M", rows), ("K", k), ("N", n), ("NSG", SIMDGROUPS)],
        grid=(-(-items // SIMDGROUPS) * SIMDGROUPS * 32, 1, 1),
        threadgroup=(SIMDGROUPS * 32, 1, 1),
        output_shapes=[(*rel.shape[:-1], n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def profile(rel, basis, stock):
    """``rel @ basis.astype(rel.dtype)`` (bf16) in our launch where proven, else ``stock()``."""
    if not enabled():
        return stock()
    why = None
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        why = "device"
    elif rel.dtype != mx.bfloat16 or basis.dtype not in (mx.bfloat16, mx.float32):
        why = "dtype"
    elif basis.ndim != 2 or rel.shape[-1] != basis.shape[0] or basis.shape[0] > 32 or basis.shape[1] % 32:
        why = "shape"
    elif library()[0] is None:
        why = library()[1]
    if why is not None:
        count(f"fallback.nax_profile.{why}")
        return stock()
    key = (rel.size // basis.shape[0], basis.shape[0], basis.shape[1], str(basis.dtype))
    verdict = _VERDICT.get(key)
    if verdict is None:
        try:
            got, want = _launch(rel, basis), stock()
            mx.eval(got, want)
            verdict = _same(got, want)
        except (RuntimeError, ValueError):
            verdict = False
        _VERDICT[key] = verdict
        from .attn_input import count_route

        count_route(("proved:" if verdict else "refused:") + "nax_profile_" + "_".join(map(str, key)))
    if not verdict:
        count("fallback.nax_profile.differs")
        return stock()
    return _launch(rel, basis)


__all__ = ["enabled", "profile"]
