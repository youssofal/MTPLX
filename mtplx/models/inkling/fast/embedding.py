"""The token embedding of a step in one launch: each id's table row gathered (and decoded, for a
table stored as affine codes), then normed with the embedding's gain, as ``model.Embedding`` does it
in two to five launches (a gather of the row or of its codes, scales and biases, the decode, the norm).

``embed_rows(ids, table, gain, eps)`` returns ``(..., COLS)`` in the table's 16-bit type, every value
equal to ``model.Embedding(table, gain, eps)(ids)``. One threadgroup owns one row; its threads own
four consecutive columns each, so the arithmetic is the reference's, value for value:

* a stored table's value is its code times its group's scale plus its group's bias, rounded once to
  the 16-bit type (``FORM`` 0: the product and the sum in float; ``FORM`` 1: in the 16-bit type, each
  operation rounded on its own); the first use on a Mac keeps the form whose rows equal the library's
  decode, as ``qproduct.affine_rows`` does;
* the norm's mean square: each thread squares and adds its four values from zero in column order,
  the simdgroup sums its threads, and the first simdgroup sums the simdgroups' totals (the slots past
  the last simdgroup are zero); the inverse is the precise reciprocal square root of the mean plus
  the epsilon;
* each output is the row's value times that inverse, rounded to the 16-bit type, times the gain,
  rounded again (a product of two 16-bit values is exact in float, so one rounding of the float
  product is the 16-bit product).

The first use per table form, width, type and Mac compares sample rows (the first, the last and
rows spread between) against the model's own embedding and refuses the launch on any difference
(counted); the caller then keeps the model's launches.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from ..precision import AffineWeight

#: The table types this launch serves: the stream's 16-bit types.
STREAM_TYPES = (mx.bfloat16, mx.float16)
#: Ways of writing a stored value's decode (``qproduct.ROW_FORMS``); a first use keeps the first that
#: matches the library's decode.
FORMS = (0, 1)

_EMBED = """
    // One threadgroup a row; thread t owns columns 4t .. 4t + 3 (threads past COLS / 4 own none).
    const uint r = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr uint GROUPS = THREADS / 32;
    threadgroup float group_sq[32];
    threadgroup float row_scale[1];
    const size_t id = size_t(uint(ids[r]));
    const uint c0 = t * 4;
    const bool owns = c0 + 4 <= uint(COLS);
    T value[4];
    if (owns) {
        if (STORED) {
            constexpr uint PER = 32 / BITS;               // codes a word
            constexpr uint MASK = (1u << BITS) - 1u;
            const uint word = codes[id * (COLS / PER) + c0 / PER];
            const T scale = scales[id * (COLS / G) + c0 / G];
            const T bias = biases[id * (COLS / G) + c0 / G];
            for (uint j = 0; j < 4; ++j) {
                const uint code = (word >> (BITS * ((c0 + j) % PER))) & MASK;
                if (FORM == 0) {
                    value[j] = static_cast<T>(static_cast<float>(scale) * static_cast<float>(code) +
                                              static_cast<float>(bias));
                } else {
                    const T scaled = scale * static_cast<T>(code);
                    value[j] = scaled + bias;
                }
            }
        } else {
            for (uint j = 0; j < 4; ++j) value[j] = table[id * COLS + c0 + j];
        }
    }
    float sq = 0.0f;
    if (owns)
        for (uint j = 0; j < 4; ++j) {
            const float v = static_cast<float>(value[j]);
            sq += v * v;
        }
    sq = simd_sum(sq);
    if (sg == 0) group_sq[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) group_sq[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(group_sq[lane]);
        if (lane == 0)
            row_scale[0] = metal::precise::rsqrt(total / float(COLS) + as_type<float>(uint(EPS)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owns) {
        const float inv = row_scale[0];
        for (uint j = 0; j < 4; ++j) {
            const T normed = static_cast<T>(static_cast<float>(value[j]) * inv);
            out[size_t(r) * COLS + c0 + j] =
                static_cast<T>(static_cast<float>(gain[c0 + j]) * static_cast<float>(normed));
        }
    }
"""


@lru_cache(maxsize=2)
def _kernel(stored):
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_embed" + ("_stored" if stored else ""),
        input_names=["ids", "codes", "scales", "biases", "table", "gain"],
        output_names=["out"],
        source=_EMBED,
    )


def reason(table, gain):
    """Why this launch does not serve the table, or None."""
    if isinstance(table, AffineWeight):
        bits, group = int(table.bits), int(table.group_size)
        cols = table.codes.shape[-1] * 32 // bits
        if bits not in (4, 8):
            return f"bits_{bits}"
        if group not in (32, 64, 128) or cols % group:
            return f"group_{group}"
        dtype = table.scales.dtype
        if dtype not in STREAM_TYPES or table.biases.dtype != dtype:
            return "scales_not_16bit"
    else:
        cols, dtype = table.shape[-1], table.dtype
        if dtype not in STREAM_TYPES:
            return "table_dtype"
    if cols % 4 or cols > 4096:
        return "width"
    if gain is None or gain.dtype != dtype or gain.shape[-1] != cols:
        return "gain"
    return None


def _launch(ids, table, gain, eps, form):
    import struct

    stored = isinstance(table, AffineWeight)
    flat = ids.reshape(-1).astype(mx.int32)
    if stored:
        bits, group = int(table.bits), int(table.group_size)
        cols, dtype = table.codes.shape[-1] * 32 // bits, table.scales.dtype
        inputs = [flat, table.codes, table.scales, table.biases, table.scales, gain]
    else:
        bits, group, (cols, dtype) = 16, 64, (table.shape[-1], table.dtype)
        inputs = [flat, flat, gain, gain, table, gain]
    threads = -(-(cols // 4) // 32) * 32
    return _kernel(stored)(
        inputs=inputs,
        template=[("T", dtype), ("STORED", stored), ("COLS", cols), ("BITS", bits), ("G", group), ("FORM", form),
                  ("THREADS", threads), ("EPS", struct.unpack("<I", struct.pack("<f", eps))[0])],
        grid=(flat.size * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(flat.size, cols)],
        output_dtypes=[dtype],
    )[0].reshape(*ids.shape, cols)


_verdicts: dict = {}


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _reference(ids, table, gain, eps):
    if isinstance(table, AffineWeight):
        rows = mx.dequantize(table.codes[ids], table.scales[ids], table.biases[ids], bits=table.bits,
                             group_size=table.group_size)
    else:
        rows = table[ids]
    return mx.fast.rms_norm(rows, gain, eps)


def embed_rows(ids, table, gain, eps, *, prove=True):
    """``model.Embedding(table, gain, eps)(ids)`` in one launch. Raises ``RuntimeError`` (counted)
    for a table this launch does not serve or when this Mac's first use disagrees."""
    from .attn_input import count_route

    why = reason(table, gain)
    if why is not None:
        count_route(f"refused:embed_{why}")
        raise RuntimeError(f"the embedding launch does not serve this table: {why}")
    stored = isinstance(table, AffineWeight)
    vocab = (table.codes if stored else table).shape[0]
    key = (stored, vocab, (table.codes if stored else table).shape[-1], getattr(table, "bits", 16),
           getattr(table, "group_size", 0), str(gain.dtype), float(eps))
    form = _verdicts.get(key)
    if form is None and prove:
        sample = mx.array(sorted({0, vocab - 1, *range(1, vocab, max(1, vocab // 509))}), mx.int32)
        want = _reference(sample, table, gain, eps)
        form = -1
        for candidate in (FORMS if stored else (0,)):
            got = _launch(sample, table, gain, eps, candidate)
            mx.eval(got, want)
            if _same(got, want):
                form = candidate
                break
        _verdicts[key] = form
        kind = "stored" if stored else "16bit"
        count_route(("proved:" if form >= 0 else "refused:") + f"embed_{vocab}_{kind}_{gain.dtype}")
    if form is None or form < 0:
        raise RuntimeError("the embedding launch did not match the model's embedding on this Mac")
    return _launch(ids, table, gain, eps, form)


__all__ = ["FORMS", "STREAM_TYPES", "embed_rows", "reason"]
