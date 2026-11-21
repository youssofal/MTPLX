"""One lane's share of the reference one-row affine product, as a Metal header (``AFFINE_LANE``).

Both stored-code launches include it: the stage A projection (``attn_input``) and the row-stable
product family (``qproduct``). The reference is MLX 0.32.2's one-row ``quantized_matmul``: a block
of ``32 * SPAN`` columns is split over the 32 lanes of a simdgroup, ``SPAN`` consecutive inputs a
lane, and the lanes' block terms are summed across the simdgroup at the end of the row. A row's bits
depend on the order of its floating-point operations, so these functions perform the reference's
operations in the reference's order. The code is this project's own, written from that arithmetic:

- 4 bits: a 32-bit code word holds eight codes, lowest first, and each 16-bit half of it is one
  unit of four. A code is multiplied where it sits in its half, so code ``j`` of a half carries a
  factor ``16**j``, and the input it meets is scaled by ``16**-j`` first: the product is exactly
  input x code (powers of two). A half's four products form one sum, in order, which is added to
  the lane's running dot product. The inputs are totalled four at a time in their own type (bf16
  for bf16 inputs), and each group of four is added to a float total.
- 8 bits: a word holds four byte codes; each product and each input joins its running value alone.
- A block adds ``scale * dot + total * bias`` to its output row (the code group's scale and bias).

Expressions whose rounding depends on how a multiply joins an add (a half's four products, an 8-bit
product joining the dot, the scale and bias step) are each written as one expression in that order.
First-use proofs compare every launch built on this header against MLX's one-row product on the
real operands of each shape, and a disagreement refuses the route (counted).
"""

AFFINE_LANE = """
namespace affine_lane {

// 16^-j: the scale of the input that meets code j of a 16-bit half (exact powers of two).
constant float kInvPlace[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};

// The four inputs that meet one 16-bit half of 4-bit codes: written scaled, so each meets its code
// where it sits; returned added together in their own type (bf16 for bf16 inputs).
template <typename P>
inline auto meet_half(P v, thread float* out) {
    out[0] = static_cast<float>(v[0]);
    out[1] = static_cast<float>(v[1]) * kInvPlace[1];
    out[2] = static_cast<float>(v[2]) * kInvPlace[2];
    out[3] = static_cast<float>(v[3]) * kInvPlace[3];
    return v[0] + v[1] + v[2] + v[3];
}

// The lane's SPAN inputs as the products take them (into out) and their float total (returned):
// at 4 bits a half's worth at a time, at 8 bits one input at a time. P: a device or thread pointer
// to bf16 or float inputs.
template <int SPAN, int BITS, typename P>
inline float take(P v, thread float* out) {
    float total = 0.0f;
    if (BITS == 4) {
        for (int h = 0; h < SPAN; h += 4) total = total + meet_half(v + h, out + h);
    } else {
        for (int j = 0; j < SPAN; ++j) {
            out[j] = static_cast<float>(v[j]);
            total = total + v[j];
        }
    }
    return total;
}

// The scaled inputs alone (the caller has their total) and the total alone.
template <int SPAN, int BITS, typename P>
inline void scaled(P v, thread float* out) {
    (void)take<SPAN, BITS>(v, out);
}

template <int SPAN, int BITS, typename P>
inline float total(P v) {
    float unused[SPAN];
    return take<SPAN, BITS>(v, unused);
}

// Code J of a 16-bit half left where it sits (so it carries 16^J), as the products take it.
template <int J>
inline float in_place(uint quad) {
    return static_cast<float>(int(quad & (0xFu << (4 * J))));
}

// One output row's SPAN codes under the lane's inputs, as the products take them: 4-bit codes a
// 16-bit half at a time (the low half of a word first), 8-bit codes as their bytes.
template <int SPAN, int BITS>
inline void unpack(const device uint32_t* words, thread float* out) {
    if (BITS == 4) {
        for (int h = 0; h < SPAN / 4; ++h) {
            const uint quad = (words[h >> 1] >> (16 * (h & 1))) & 0xFFFFu;
            out[4 * h] = in_place<0>(quad);
            out[4 * h + 1] = in_place<1>(quad);
            out[4 * h + 2] = in_place<2>(quad);
            out[4 * h + 3] = in_place<3>(quad);
        }
    } else {
        for (int j = 0; j < SPAN; ++j) out[j] = static_cast<float>(uchar(words[j >> 2] >> (8 * (j & 3))));
    }
}

// The block's term for one output row from the scaled inputs, their total and the row's codes: at 4
// bits each half's four products as one sum, in order, then into the dot; at 8 bits each product
// straight into the dot; then the group's scale and bias.
template <int SPAN, int BITS>
inline float term(const thread float* in, const thread float* code, float scale, float bias, float tot) {
    float dot = 0.0f;
    if (BITS == 4) {
        for (int h = 0; h < SPAN; h += 4)
            dot = dot + (in[h] * code[h] + in[h + 1] * code[h + 1] + in[h + 2] * code[h + 2] +
                         in[h + 3] * code[h + 3]);
    } else {
        for (int j = 0; j < SPAN; ++j) dot = dot + in[j] * code[j];
    }
    return scale * dot + tot * bias;
}

}  // namespace affine_lane
"""
