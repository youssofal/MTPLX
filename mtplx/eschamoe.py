"""Native MLX decode for the EschaLabs ``eschamoe`` 2/3-bit MoE weight format
(``EschaLabs/Qwen3.6-35B-A3B-Escha-W2``).

Reverse-engineered from the vendor CUDA kernel ``escham_reconstruct`` and validated BIT-EXACT
against reference goldens (0 mismatches over 2,097,152 K=2 + 1,048,576 K=3 weights). Pure MLX —
no custom Metal kernel is required for correctness (a fused kernel is a later perf step).

Format (per projection, stacked over E experts, key prefix ``model.language_model.…``):
    escha_code  int16 [E, in_p/16, out_p/16, 16*K]   K=2 gate_up, K=3 down_proj
    escha_rin   fp16  [E, in_p]     per-input-channel scale (trained s_in folded in)
    escha_rout  fp16  [E, out_p]    per-output-channel scale (trained s_out folded in)

Decode: the weight is tiled into 16x16 blocks. Each output weight = a "codebook" value
``dec(window)`` for a 16-bit ``window`` gathered from the tile's bits per a fixed table
(``eschamoe_gather.npz``). ``dec`` is a magic-multiply bit-twiddle (cb_id=1 / "cbA"):

    r = (window * 3417055213) mod 2**32
    dec = f16((r & 0x8fff) ^ 0x3b60) + f16(((r >> 16) & 0x8fff) ^ 0x3b60)

Expert forward wraps the decoded weight in a QuaRot-style Hadamard rotation:
    y = T128(x * rin) @ W ; y = T128(y) * rout      (T128 = normalized 128-block Walsh-Hadamard)
"""
from __future__ import annotations

import os
import numpy as np
import mlx.core as mx

_MAGIC = np.uint32(3417055213)
_MASK = np.uint16(0x8fff)
_XOR = np.uint16(0x3b60)
_GATHER_PATH = os.path.join(os.path.dirname(__file__), "eschamoe_gather.npz")

# Compute dtype for the Hadamard rotation + QMV matvec of the expert forward.
# bf16 is the model's native dtype and the default: decoding 2-bit experts only to
# compute them in fp32 is a wasteful cast storm (~30% of decode dispatches) with no
# quality payoff on this checkpoint. ESCHA_FP32=1 restores the fp32 round-trip as a
# numerical-debug escape hatch (see bench/validate_qmv.py).
COMPUTE_DTYPE = mx.float32 if os.environ.get("ESCHA_FP32") else mx.bfloat16

_DEC: mx.array | None = None
_GATHER: dict | None = None
_POW16 = mx.array((np.uint32(1) << np.arange(16, dtype=np.uint32)))
_HAD128: mx.array | None = None


def _build_dec_table() -> mx.array:
    """The 65536-entry fp16 codebook (cb_id=1). Deterministic; matches the CUDA kernel exactly."""
    w = np.arange(65536, dtype=np.uint32)
    r = (w * _MAGIC) & np.uint32(0xFFFFFFFF)
    lo = ((r & 0xFFFF).astype(np.uint16) & _MASK) ^ _XOR
    hi = (((r >> 16) & 0xFFFF).astype(np.uint16) & _MASK) ^ _XOR
    val = (lo.view(np.float16).astype(np.float32) + hi.view(np.float16).astype(np.float32)).astype(np.float16)
    return mx.array(val)


def _dec() -> mx.array:
    global _DEC
    if _DEC is None:
        _DEC = _build_dec_table()
    return _DEC


def _gather(K: int):
    """(word_of, bit_of) int32 [256,16] gather tables for K in {2,3}."""
    global _GATHER
    if _GATHER is None:
        z = np.load(_GATHER_PATH)
        _GATHER = {
            2: (mx.array(z["word_of_K2"].astype(np.int32)), mx.array(z["bit_of_K2"].astype(np.uint32))),
            3: (mx.array(z["word_of_K3"].astype(np.int32)), mx.array(z["bit_of_K3"].astype(np.uint32))),
        }
    if K not in _GATHER:
        raise ValueError(f"eschamoe: unsupported K={K} (only 2,3)")
    return _GATHER[K]


def decode_expert_weights(code: mx.array, K: int) -> mx.array:
    """Decode packed eschamoe codes to bare fp16 weights.

    code: int16 ``[..., nI, nJ, 16*K]``  ->  W: fp16 ``[..., nI*16, nJ*16]``.
    """
    word_of, bit_of = _gather(K)                                  # [256,16]
    lead = tuple(code.shape[:-3])
    nI, nJ, nw = code.shape[-3], code.shape[-2], code.shape[-1]
    if nw != 16 * K:
        raise ValueError(f"eschamoe: code last dim {nw} != 16*K ({16*K})")
    T = int(np.prod(lead)) * nI * nJ if lead else nI * nJ
    u16 = (code.astype(mx.int32) & 0xFFFF).reshape(T, nw).astype(mx.uint32)   # [T, nw]
    words = u16[:, word_of.reshape(-1)].reshape(T, 256, 16)                   # gather -> [T,256,16]
    bits = mx.bitwise_and(mx.right_shift(words, bit_of), mx.array(np.uint32(1)))
    win = (bits * _POW16).sum(axis=-1).astype(mx.int32)                       # [T,256] window value
    Wf = _dec()[win].reshape(*lead, nI, nJ, 16, 16)                           # [...,nI,nJ,16,16]
    L = len(lead)
    perm = (*range(L), L + 0, L + 2, L + 1, L + 3)                            # (nI,nJ,16,16)->(nI,16,nJ,16)
    return mx.transpose(Wf, perm).reshape(*lead, nI * 16, nJ * 16)


_KERNELS: dict = {}


def _decode_kernel(IN: int, OUT: int, K: int):
    """One-pass Metal decompression kernel: packed eschamoe codes -> bare fp16 weights.
    Each thread reconstructs one output weight (gather 16 tile bits -> window -> codebook)."""
    key = (IN, OUT, K)
    if key in _KERNELS:
        return _KERNELS[key]
    NW = 16 * K
    src = f"""
        uint gid = thread_position_in_grid.x;
        uint c = gid % {OUT}u;
        uint t = gid / {OUT}u;
        uint r = t % {IN}u;
        uint e = t / {IN}u;
        uint dr = r & 15u, dc = c & 15u;
        uint ti = r >> 4, tj = c >> 4;
        uint slot = (dr << 4) + dc;
        uint tilebase = ((e * {IN >> 4}u + ti) * {OUT >> 4}u + tj) * {NW}u;
        ushort window = 0;
        for (uint k = 0; k < 16u; ++k) {{
            uint gi = (slot << 4) + k;
            int wo = word_of[gi];
            int bo = bit_of[gi];
            ushort word = (ushort) code[tilebase + wo];
            window |= (ushort)((((uint)(word >> (ushort)bo)) & 1u) << k);
        }}
        out[gid] = dec[window];
    """
    kern = mx.fast.metal_kernel(
        name=f"eschadec_{IN}_{OUT}_{K}",
        input_names=["code", "word_of", "bit_of", "dec"],
        output_names=["out"],
        source=src,
    )
    _KERNELS[key] = kern
    return kern


def decode_expert_weights_fast(code: mx.array, K: int) -> mx.array:
    """Metal-kernel decode: code int16 [U, IN/16, OUT/16, 16*K] -> W fp16 [U, IN, OUT]."""
    word_of, bit_of = _gather(K)
    U, ni, nj, nw = code.shape
    IN, OUT = ni * 16, nj * 16
    total = U * IN * OUT                          # IN*OUT divisible by 256 for both configs
    kern = _decode_kernel(IN, OUT, K)
    (out,) = kern(
        inputs=[code, word_of.reshape(-1), bit_of.reshape(-1), _dec()],
        output_shapes=[(U, IN, OUT)],
        output_dtypes=[mx.float16],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


_FUSED: dict = {}


def _fused_matmul_kernel(IN: int, OUT: int, K: int):
    """Fused decode+GEMM: y = xh @ decode(code), tiled on 16x16 eschamoe tiles. Each threadgroup
    decodes one 16x16 weight tile into threadgroup memory and matmuls it — dense W is NEVER formed."""
    key = (IN, OUT, K)
    if key in _FUSED:
        return _FUSED[key]
    NW = 16 * K
    src = f"""
        uint n = thread_position_in_threadgroup.x;      // out col in tile (0..15)
        uint m = thread_position_in_threadgroup.y;      // out row in tile (0..15)
        uint tj = threadgroup_position_in_grid.x;       // out tile col
        uint rt = threadgroup_position_in_grid.y;       // row tile
        uint out_row = rt * 16u + m;
        uint out_col = tj * 16u + n;
        threadgroup half Wt[16][16];
        threadgroup half Xt[16][16];
        float acc = 0.0f;
        uint M = (uint) n_rows;
        for (uint ti = 0; ti < {IN // 16}u; ++ti) {{
            uint slot = m * 16u + n;
            uint tilebase = (ti * {OUT // 16}u + tj) * {NW}u;
            ushort window = 0;
            for (uint k = 0; k < 16u; ++k) {{
                uint gi = slot * 16u + k;
                int wo = word_of[gi]; int bo = bit_of[gi];
                ushort word = (ushort) code[tilebase + wo];
                window |= (ushort)((((uint)(word >> (ushort)bo)) & 1u) << k);
            }}
            Wt[m][n] = dec[window];
            uint xr = rt * 16u + m;
            Xt[m][n] = (xr < M) ? xh[xr * {IN}u + ti * 16u + n] : (half)0.0h;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 16u; ++k) acc += (float)Xt[m][k] * (float)Wt[k][n];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (out_row < M) out[out_row * {OUT}u + out_col] = (half) acc;
    """
    kern = mx.fast.metal_kernel(
        name=f"eschafused_{IN}_{OUT}_{K}",
        input_names=["xh", "code", "word_of", "bit_of", "dec", "n_rows"],
        output_names=["out"],
        source=src,
    )
    _FUSED[key] = kern
    return kern


def fused_decode_matmul(xh: mx.array, code: mx.array, K: int, OUT: int) -> mx.array:
    """xh [M, IN] fp16 ; code [IN/16, OUT/16, 16K] int16 (one expert) -> y [M, OUT] fp16.
    Decode is fused into the matmul; dense W is never materialized."""
    word_of, bit_of = _gather(K)
    M, IN = xh.shape
    kern = _fused_matmul_kernel(IN, OUT, K)
    rt = (M + 15) // 16
    (out,) = kern(
        inputs=[xh, code, word_of.reshape(-1), bit_of.reshape(-1), _dec(), mx.array(M, dtype=mx.int32)],
        output_shapes=[(M, OUT)],
        output_dtypes=[mx.float16],
        grid=(OUT, rt * 16, 1),
        threadgroup=(16, 16, 1),
    )
    return out


_WARP: dict = {}


def _warp(K: int):
    """(perm_lane, perm_m, lane_a, lane_b, lane_p) int32 for the per-lane warp-assembly decode."""
    global _WARP
    if not _WARP:
        z = np.load(_GATHER_PATH)
        for k in (2, 3):
            _WARP[k] = tuple(mx.array(z[f"{nm}_K{k}"].astype(np.int32))
                             for nm in ("perm_lane", "perm_m", "lane_a", "lane_b", "lane_p"))
    return _WARP[K]


def _fused_moe_kernel(IN: int, OUT: int, K: int):
    """Grouped fused decode+GEMM (warp-assembly decode): each threadgroup assembles v ONCE per lane
    (32 lanes), then every weight is (v >> K*m) & 0xffff + inline codebook. No DEC table, no dense W."""
    key = ("moe", IN, OUT, K)
    if key in _FUSED:
        return _FUSED[key]
    NW = 16 * K
    src = f"""
        uint n = thread_position_in_threadgroup.x;
        uint m = thread_position_in_threadgroup.y;
        uint tid = m * 16u + n;
        uint tj = threadgroup_position_in_grid.x;
        uint rt = threadgroup_position_in_grid.y;
        uint e = (uint) tile_expert[rt];
        uint out_col = tj * 16u + n;
        uint xr = rt * 16u + m;
        uint M = (uint) n_rows;
        threadgroup ulong vsh[32];
        threadgroup half Wt[16][16];
        threadgroup half Xt[16][16];
        float acc = 0.0f;
        for (uint ti = 0; ti < {IN // 16}u; ++ti) {{
            uint tilebase = ((e * {IN // 16}u + ti) * {OUT // 16}u + tj) * {NW}u;
            if (tid < 32u) {{
                uint L = tid;
                uint wa = (uint)(lane_a[L] >> 1);
                uint wb = (uint)(lane_b[L] >> 1);
                uint rd8 = (uint)(ushort)code[tilebase + wa] | ((uint)(ushort)code[tilebase + wa + 1u] << 16);
                uint rd9 = (uint)(ushort)code[tilebase + wb] | ((uint)(ushort)code[tilebase + wb + 1u] << 16);
                ulong v;
                if ({K}u == 2u) {{
                    v = (lane_p[L] != 0) ? (ulong)((rd8 >> 16) | ((rd9 & 0xffffu) << 16)) : (ulong)rd8;
                }} else {{
                    ulong rd11 = ((ulong)rd9 << 32) | (ulong)rd8;
                    v = rd11 >> (uint)lane_p[L];
                }}
                vsh[L] = v;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint slot = m * 16u + n;
            ushort window = (ushort)((vsh[(uint)perm_lane[slot]] >> ({K}u * (uint)perm_m[slot])) & 0xffffUL);
            uint rr = (uint)window * 3417055213u;
            ushort lo = ((ushort)(rr) & (ushort)0x8fff) ^ (ushort)0x3b60;
            ushort hi = ((ushort)(rr >> 16) & (ushort)0x8fff) ^ (ushort)0x3b60;
            Wt[m][n] = as_type<half>(lo) + as_type<half>(hi);
            Xt[m][n] = (xr < M) ? xh[xr * {IN}u + ti * 16u + n] : (half)0.0h;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 16u; ++k) acc += (float)Xt[m][k] * (float)Wt[k][n];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (xr < M) out[xr * {OUT}u + out_col] = (half) acc;
    """
    kern = mx.fast.metal_kernel(
        name=f"eschamoe_wa_{IN}_{OUT}_{K}",
        input_names=["xh", "code", "tile_expert", "perm_lane", "perm_m", "lane_a", "lane_b", "lane_p", "n_rows"],
        output_names=["out"],
        source=src,
    )
    _FUSED[key] = kern
    return kern


def _fused_moe_gemv_kernel(IN: int, OUT: int, K: int):
    """No-pad variant: one 16-row tile per slot, all rows read slot's single row directly from
    xh [S, IN] (no padding array, no scatter/gather). Warp-assembly decode. For decode/spec-verify."""
    key = ("gemv", IN, OUT, K)
    if key in _FUSED:
        return _FUSED[key]
    src = f"""
        uint n = thread_position_in_threadgroup.x;
        uint m = thread_position_in_threadgroup.y;
        uint tid = m * 16u + n;
        uint tj = threadgroup_position_in_grid.x;
        uint s = threadgroup_position_in_grid.y;             // slot (one tile per slot)
        uint e = (uint) eids[s];
        uint out_col = tj * 16u + n;
        threadgroup ulong vsh[32];
        threadgroup half Wt[16][16];
        threadgroup half Xt[16];
        float acc = 0.0f;
        for (uint ti = 0; ti < {IN // 16}u; ++ti) {{
            uint tilebase = ((e * {IN // 16}u + ti) * {OUT // 16}u + tj) * {16 * K}u;
            if (tid < 32u) {{
                uint L = tid;
                uint wa = (uint)(lane_a[L] >> 1); uint wb = (uint)(lane_b[L] >> 1);
                uint rd8 = (uint)(ushort)code[tilebase + wa] | ((uint)(ushort)code[tilebase + wa + 1u] << 16);
                uint rd9 = (uint)(ushort)code[tilebase + wb] | ((uint)(ushort)code[tilebase + wb + 1u] << 16);
                ulong v;
                if ({K}u == 2u) {{ v = (lane_p[L] != 0) ? (ulong)((rd8 >> 16) | ((rd9 & 0xffffu) << 16)) : (ulong)rd8; }}
                else {{ ulong rd11 = ((ulong)rd9 << 32) | (ulong)rd8; v = rd11 >> (uint)lane_p[L]; }}
                vsh[L] = v;
            }}
            if (tid < 16u) Xt[tid] = xh[s * {IN}u + ti * 16u + tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint slot = m * 16u + n;
            ushort window = (ushort)((vsh[(uint)perm_lane[slot]] >> ({K}u * (uint)perm_m[slot])) & 0xffffUL);
            uint rr = (uint)window * 3417055213u;
            ushort lo = ((ushort)(rr) & (ushort)0x8fff) ^ (ushort)0x3b60;
            ushort hi = ((ushort)(rr >> 16) & (ushort)0x8fff) ^ (ushort)0x3b60;
            Wt[m][n] = as_type<half>(lo) + as_type<half>(hi);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 16u; ++k) acc += (float)Xt[k] * (float)Wt[k][n];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (m == 0u) out[s * {OUT}u + out_col] = (half) acc;
    """
    kern = mx.fast.metal_kernel(
        name=f"eschagemv_{IN}_{OUT}_{K}",
        input_names=["xh", "code", "eids", "perm_lane", "perm_m", "lane_a", "lane_b", "lane_p"],
        output_names=["out"], source=src)
    _FUSED[key] = kern
    return kern


_CB = mx.array([3417055213, 0, 0x8FFF8FFF, 0x3B603B60], dtype=mx.uint32)
# Metal element type for the QMV output buffer (must match COMPUTE_DTYPE; Metal does not
# implicitly narrow float->bfloat, so the output write is cast explicitly).
_MTL_ODT = "bfloat" if COMPUTE_DTYPE == mx.bfloat16 else "float"


def _escha_qmv_kernel(K: int, TK: int, TN: int):
    """Ported from dusterbloom/higgs ESCHA_QMV_KERNEL, courtesy of dusterbloom
    (https://dusterbloom.github.io/): one simdgroup per output element, rows on grid.y (no M
    padding), lanes stride the contraction + decode 2 codes each + simd_sum reduce."""
    key = ("qmv", K, TK, TN, _MTL_ODT)
    if key in _FUSED:
        return _FUSED[key]
    src = f"""
        threadgroup float x_sh[{TK * 16}];
        uint row = threadgroup_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint sg = tid >> 5; uint lane = tid & 31u;
        for (uint i = tid; i < {TK * 16}u; i += 128u) x_sh[i] = xh[row * {TK * 16}u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint o = threadgroup_position_in_grid.x * 4u + sg;
        if (o >= {TN * 16}u) return;
        uint tn = o >> 4; uint c = o & 15u; uint cb2 = (c >> 3) & 1u; uint c7 = c & 7u;
        const device short* base = code + ulong(eids[row]) * {TK * TN * 16 * K}ul;
        uint q = lane & 3u; uint rh = (lane >> 2) & 1u;
        uint t = 4u * (4u * c7 + q) + 2u * cb2 + rh; uint r0 = 8u * rh + 2u * q;
        uint b0 = 2u * t * {K}u + {K + 256 * K}u - 16u; uint b2 = b0 + {K + 16}u;
        uint i0 = (b0 / 32u) % {8 * K}u; uint i1w = (b2 - 1u) / 32u; uint s1 = (i1w + 1u) * 32u - b2; uint i1 = i1w % {8 * K}u;
        float acc = 0.0f;
        for (uint tk = lane >> 3; tk < {TK}u; tk += 4u) {{
            const device short* tile = base + (tk * {TN}u + tn) * {16 * K}u;
            uint w0 = uint(ushort(tile[2u * i0])) | (uint(ushort(tile[2u * i0 + 1u])) << 16);
            uint wb = uint(ushort(tile[2u * i1])) | (uint(ushort(tile[2u * i1 + 1u])) << 16);
            ulong pair = (ulong(w0) << 32) | ulong(wb);
            uint w1 = uint(pair >> s1);
            uint x0 = ((w1 >> {K}u) & 0xFFFFu) * cb[0] + cb[1]; x0 = (x0 & cb[2]) ^ cb[3];
            uint x1 = (w1 & 0xFFFFu) * cb[0] + cb[1]; x1 = (x1 & cb[2]) ^ cb[3];
            half2 h0 = as_type<half2>(x0); half2 h1 = as_type<half2>(x1);
            float v0 = float(half(float(h0.x) + float(h0.y)));
            float v1 = float(half(float(h1.x) + float(h1.y)));
            acc = fma(x_sh[tk * 16u + r0], v0, acc);
            acc = fma(x_sh[tk * 16u + r0 + 1u], v1, acc);
        }}
        acc = simd_sum(acc);
        if (lane == 0u) dst[row * {TN * 16}u + o] = ({_MTL_ODT})acc;
    """
    kern = mx.fast.metal_kernel(name=f"escha_qmv_{K}_{TK}_{TN}_{_MTL_ODT}",
                                input_names=["xh", "code", "eids", "cb"], output_names=["dst"], source=src)
    _FUSED[key] = kern
    return kern


def escha_qmv(xh: mx.array, eids: mx.array, code: mx.array, K: int, OUT: int) -> mx.array:
    """Fast decode+matvec (rows<=32). xh [S, IN] COMPUTE_DTYPE, eids [S] u32,
    code [E, IN/16, OUT/16, 16K] -> dst [S, OUT] COMPUTE_DTYPE.  One simdgroup per output
    element; no M padding.  x_sh + acc stay float in-kernel (accuracy); only the buffer dtype
    follows COMPUTE_DTYPE, so ESCHA_BF16 drops the f32 cast on xh without changing the math."""
    S, IN = xh.shape
    TK, TN = IN // 16, OUT // 16
    kern = _escha_qmv_kernel(K, TK, TN)
    (dst,) = kern(inputs=[xh.astype(COMPUTE_DTYPE), code, eids.astype(mx.uint32), _CB],
                  output_shapes=[(S, OUT)], output_dtypes=[COMPUTE_DTYPE],
                  grid=(((OUT + 3) // 4) * 128, S, 1), threadgroup=(128, 1, 1))
    return dst


def fused_moe_gemv(xh: mx.array, eids: mx.array, code: mx.array, K: int, OUT: int) -> mx.array:
    """xh [S, IN] fp16, eids [S] -> y [S, OUT] fp16.  No padding; one tile per slot."""
    perm_lane, perm_m, lane_a, lane_b, lane_p = _warp(K)
    S, IN = xh.shape
    kern = _fused_moe_gemv_kernel(IN, OUT, K)
    (out,) = kern(
        inputs=[xh, code, eids.astype(mx.int32), perm_lane, perm_m, lane_a, lane_b, lane_p],
        output_shapes=[(S, OUT)], output_dtypes=[mx.float16],
        grid=(OUT, S * 16, 1), threadgroup=(16, 16, 1))
    return out


def fused_moe_matmul(xh: mx.array, tile_expert: mx.array, code: mx.array, K: int, OUT: int) -> mx.array:
    """xh [Mpad, IN] fp16 (rows grouped by expert, 16-aligned per expert), tile_expert [Mpad/16] int32,
    code [E, IN/16, OUT/16, 16K] -> y [Mpad, OUT] fp16.  One launch, warp-assembly decode, no dense W."""
    perm_lane, perm_m, lane_a, lane_b, lane_p = _warp(K)
    Mpad, IN = xh.shape
    kern = _fused_moe_kernel(IN, OUT, K)
    (out,) = kern(
        inputs=[xh, code, tile_expert.astype(mx.int32), perm_lane, perm_m, lane_a, lane_b, lane_p,
                mx.array(Mpad, dtype=mx.int32)],
        output_shapes=[(Mpad, OUT)],
        output_dtypes=[mx.float16],
        grid=(OUT, Mpad, 1),
        threadgroup=(16, 16, 1),
    )
    return out


def _hadamard128() -> mx.array:
    global _HAD128
    if _HAD128 is None:
        h = np.array([[1.0]], np.float32)
        while h.shape[0] < 128:
            h = np.block([[h, h], [h, -h]])
        _HAD128 = mx.array((h / np.sqrt(128.0)).astype(np.float32))
    return _HAD128


def t128(x: mx.array, pre=None, post=None) -> mx.array:
    """y = post * T128((x * pre))  — normalized 128-block Walsh-Hadamard over the last dim,
    via MLX's native fused hadamard_transform (O(n log n), single kernel).  Runs in
    COMPUTE_DTYPE; when callers store pre/post already in COMPUTE_DTYPE the .astype calls
    are no-ops (the per-call scale cast is hoisted to load time)."""
    x = x.astype(COMPUTE_DTYPE)
    if pre is not None:
        x = x * pre.astype(COMPUTE_DTYPE)
    lead = tuple(x.shape[:-1]); IC = x.shape[-1]
    x = mx.hadamard_transform(x.reshape(*lead, IC // 128, 128), scale=128.0 ** -0.5).reshape(*lead, IC)
    if post is not None:
        x = x * post.astype(COMPUTE_DTYPE)
    return x
