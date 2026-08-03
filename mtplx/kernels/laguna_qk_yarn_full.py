"""Fused Q/K RMSNorm + partial-YaRN RoPE for the Laguna S-2.1 decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel
``laguna_full_qk_norm_yarn_bf16_128_v4`` (``lagunaFullQKNormYaRNKernel`` /
``lagunaFullQKNormYaRN`` in Sources/MLXFastModel/LagunaRuntimeModel.swift),
re-expressed as a Python ``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1,
which is a different model from the challenge's XS2.1:

    full-attention heads   XS2.1 48   ->  S2.1 48   (unchanged)
    kv heads               8          ->  8         (unchanged)
    head_dim               128        ->  128       (unchanged)
    rotary dims (0.5)      64         ->  64        (32 rotary pairs)
    YaRN mscale            1.3465735912322998  ->  1.4852030263919618  (**changed**)

The mscale IS the single load-bearing S-2.1 customization.  mlx-lm's
``YarnRoPE`` computes ``mscale = yarn_get_mscale(factor, 1) /
yarn_get_mscale(factor, 0) = 0.1 * log(factor) + 1``; XS2.1 used ``factor 32``
(``1.3465735912322998``) and S-2.1 uses ``factor 128``
(``0.1*log(128)+1 == 1.4852030263919618``), which is exactly the
``rope_parameters.full_attention.attention_factor`` the pinned oQ4e config
carries (see ``models/laguna_config.py``).  Everything else — 48+8 heads at
head_dim 128, partial-rotary 0.5 (64 rotary dims == 32 pairs), theta 500000,
original_max_position 8192 — is identical to the challenge shape.

## The single dispatch it replaces

On a full-attention decode layer the stock chain (``models/laguna.py``
``Attention.__call__`` + mlx-lm ``YarnRoPE``) is: ``q_norm`` (RMSNorm), a
transpose, ``k_norm`` (RMSNorm), a transpose, then for q AND k a partial-YaRN
RoPE that is itself a copy + a sliced scalar multiply (mscale) + ``mx.fast.rope``
— ~six dispatches for arithmetic on 48x128 and 8x128 elements.  This kernel does
the whole chain for q AND k in ONE dispatch, one 32-lane simdgroup per head
(56 = 48 + 8 heads), writing directly into the ``[1, heads, 1, 128]`` head-major
layout SDPA reads.

## Bit-exactness, link for link with the stock chain

* **RMSNorm** mirrors ``rms_single_row`` (rms_norm.metal) at a 128-wide axis:
  32 lanes x 4 sequential FP32 squares, one ``simd_sum`` (which returns the total
  to *every* lane, so each derives the same ``precise::rsqrt`` locally — the
  barrier-elided form the challenge uses, no threadgroup slot).  The output is
  ``w[i] * bfloat(float(x[i]) * inv)`` — the same double rounding
  ``mx.fast.rms_norm`` writes.

* **The rotary angles** (cos/sin) are supplied as a precomputed ``angles``
  table rather than re-derived: :func:`build_full_yarn_angles` runs the layer's
  own ``mx.fast.rope`` over a ``[ones, zeros]`` seed at the current offset, so
  the table holds the EXACT ``cos``/``sin`` bits ``mx.fast.rope`` itself uses at
  those positions (a length-64 vector: 32 cos then 32 sin).  This is the
  challenge's ``_slidingRoPEAngleSeed`` trick, minus one rounding: the challenge
  seeds ``1/mscale`` through the *mscale-applying* rope so the atlas cancels back
  to cos/sin, whereas seeding ``1`` through a bare ``mx.fast.rope`` yields pure
  cos/sin directly and applies mscale only once, in this kernel, exactly where
  ``YarnRoPE`` applies it.

* **The mscale** is applied to the normed q/k value in bf16
  (``bfloat(mscale)`` then a bf16 product), matching ``YarnRoPE.__call__``'s
  ``self.mscale * x`` on a bf16 array (the scalar promotes to the array dtype);
  the CPU check measures this against the true stock rope.

Only the rotary region [0, 64) is scaled and rotated (pairs ``(p, p+32)``); the
tail [64, 128) is normed output copied through with NO mscale — exactly what a
partial-rotary ``YarnRoPE`` (``dims == 64``) produces.

Callers gate on :func:`is_qk_yarn_full_eligible` first; the public helper falls
back to the stock ``q_norm``/``k_norm`` -> transpose -> ``YarnRoPE`` chain on any
shape it does not cover, so it can be switched on without owning a correctness
branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mlx.core as mx


# S-2.1 full-attention decode shape, baked.
_N_Q_HEADS = 48
_N_KV_HEADS = 8
_HEAD_DIM = 128
_ROT_DIMS = 64  # partial_rotary_factor 0.5 * 128
_ROT_PAIRS = _ROT_DIMS // 2  # 32
_SIMD = 32
# yarn_get_mscale(128, 1) / yarn_get_mscale(128, 0) == 0.1*log(128) + 1.
_YARN_MSCALE = 1.4852030263919618


@dataclass(frozen=True)
class YarnFullSpec:
    """Shape + rotary geometry for the full-attention YaRN qk-norm+rope kernel."""

    n_q_heads: int = _N_Q_HEADS
    n_kv_heads: int = _N_KV_HEADS
    head_dim: int = _HEAD_DIM
    rot_dims: int = _ROT_DIMS
    mscale: float = _YARN_MSCALE
    eps: float = 1e-6

    @property
    def total_heads(self) -> int:
        return self.n_q_heads + self.n_kv_heads

    @property
    def rot_pairs(self) -> int:
        return self.rot_dims // 2


def build_full_yarn_angles(
    freqs: mx.array, offset: int | mx.array, spec: YarnFullSpec = YarnFullSpec()
) -> mx.array:
    """Exact cos/sin table for the partial-YaRN rotation at ``offset``.

    Runs ``mx.fast.rope`` over a ``[ones(32), zeros(32)]`` seed at ``dims ==
    rot_dims`` with the layer's own ``freqs``.  Because plain rope rotates pair
    ``p`` as ``(x_p cos - x_{p+32} sin, x_p sin + x_{p+32} cos)``, a row of ones
    followed by zeros comes back as exactly ``[cos_0..cos_31, sin_0..sin_31]`` —
    the same floats ``mx.fast.rope`` uses internally, so the kernel's rotation
    is bitwise the rope's.  No mscale here: the kernel applies mscale to the
    values, so the angles stay pure cos/sin.

    ``freqs`` is the ``YarnRoPE._freqs`` float32 buffer (length rot_dims//2).
    """

    half = spec.rot_pairs
    seed = mx.concatenate(
        [mx.ones((half,), dtype=mx.float32), mx.zeros((half,), dtype=mx.float32)]
    ).reshape(1, 1, 1, spec.rot_dims)
    angles = mx.fast.rope(
        seed,
        spec.rot_dims,
        traditional=False,
        base=None,
        scale=1.0,
        offset=offset,
        freqs=freqs,
    )
    return angles.reshape(1, 1, 1, spec.rot_dims)


def is_qk_yarn_full_eligible(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: YarnFullSpec,
) -> bool:
    """Whether the fused kernel covers this exact full-attention decode shape."""

    if not mx.metal.is_available():
        return False
    try:
        if mx.default_device() != mx.gpu:
            return False
    except Exception:
        return False
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16:
        return False
    if q_weight.dtype != mx.bfloat16 or k_weight.dtype != mx.bfloat16:
        return False
    if angles.dtype != mx.float32:
        return False
    if spec.head_dim != _HEAD_DIM or spec.rot_dims != _ROT_DIMS:
        return False
    # Decode only: one active row, [1, 1, heads*head_dim].
    if queries.ndim != 3 or keys.ndim != 3:
        return False
    if int(queries.shape[0]) != 1 or int(queries.shape[1]) != 1:
        return False
    if int(keys.shape[0]) != 1 or int(keys.shape[1]) != 1:
        return False
    if int(queries.shape[-1]) != spec.n_q_heads * spec.head_dim:
        return False
    if int(keys.shape[-1]) != spec.n_kv_heads * spec.head_dim:
        return False
    if int(q_weight.size) != spec.head_dim or int(k_weight.size) != spec.head_dim:
        return False
    if int(angles.size) != spec.rot_dims:
        return False
    return True


@lru_cache(maxsize=None)
def _qk_yarn_full_kernel(
    n_q_heads: int, n_kv_heads: int, rot_dims: int, mscale: float, eps: float
):
    header = f"""
        using namespace metal;
        constant constexpr uint HEAD_DIM = {_HEAD_DIM};
        constant constexpr uint ROT_DIMS = {rot_dims};
        constant constexpr uint ROT_PAIRS = {rot_dims // 2};
        constant constexpr uint QUERY_HEADS = {n_q_heads};
        constant constexpr float YARN_MSCALE = {mscale!r}f;
        constant constexpr float RMS_EPS = {eps!r}f;
    """

    # One 32-lane simdgroup per head; lane `l` owns the contiguous block
    # [4l, 4l+4).  simd_sum returns the whole RMS statistic to every lane, so no
    # threadgroup slot or barrier is needed (the challenge's "barrier-elision").
    source = """
        uint head = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        const device T* input;
        const device T* weight;
        if (head < QUERY_HEADS) {
            input = raw_queries + head * HEAD_DIM;
            weight = query_weight;
        } else {
            input = raw_keys + (head - QUERY_HEADS) * HEAD_DIM;
            weight = key_weight;
        }

        uint base = lane * 4;
        thread T normalized[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float value = float(input[base + i]);
            sum += value * value;
        }
        sum = simd_sum(sum);
        float inverse_rms = metal::precise::rsqrt(sum / float(HEAD_DIM) + RMS_EPS);

        for (uint i = 0; i < 4; ++i) {
            normalized[i] =
                weight[base + i] *
                static_cast<T>(float(input[base + i]) * inverse_rms);
        }

        // Element `p + ROT_PAIRS` is the rotary partner of pair `p`.  With
        // base == lane*4 and ROT_PAIRS == 32, the partner lives 8 lanes away.
        thread float paired[4];
        for (uint i = 0; i < 4; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 8);
        }

        device T* output =
            head < QUERY_HEADS
            ? queries + head * HEAD_DIM
            : keys + (head - QUERY_HEADS) * HEAD_DIM;

        // Lanes 0..7 own elements [0, 32): every rotary pair (p, p+32).  They
        // apply mscale (bf16), read the pure cos/sin from `angles`, and write
        // BOTH halves of each pair.
        if (lane < ROT_PAIRS / 4u) {
            T rounded_mscale = static_cast<T>(YARN_MSCALE);
            for (uint i = 0; i < 4; ++i) {
                uint pair = base + i;
                float first = float(static_cast<T>(normalized[i] * rounded_mscale));
                float second =
                    float(static_cast<T>(static_cast<T>(paired[i]) * rounded_mscale));
                float cosine = angles[pair];
                float sine = angles[pair + ROT_PAIRS];
                output[pair] = static_cast<T>(first * cosine - second * sine);
                output[pair + ROT_PAIRS] =
                    static_cast<T>(first * sine + second * cosine);
            }
        } else if (lane >= HEAD_DIM / 8u) {
            // Lanes 16..31 own the non-rotary tail [64, 128): normed, no mscale.
            for (uint i = 0; i < 4; ++i) {
                output[base + i] = normalized[i];
            }
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_laguna_qk_yarn_full_hq{n_q_heads}_hkv{n_kv_heads}"
            f"_r{rot_dims}_v1"
        ),
        input_names=["raw_queries", "raw_keys", "query_weight", "key_weight", "angles"],
        output_names=["queries", "keys"],
        header=header,
        source=source,
    )


def _stock_qk_yarn_full(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    freqs: mx.array,
    offset: int | mx.array,
    spec: YarnFullSpec,
) -> tuple[mx.array, mx.array]:
    """The exact stock chain: q_norm/k_norm -> transpose -> YarnRoPE, for q and k.

    Reproduces ``models/laguna.py`` ``Attention.__call__`` at T == 1 with
    mlx-lm ``YarnRoPE``: RMSNorm over head_dim, transpose to head-major, scale
    the first ``rot_dims`` by ``mscale``, then ``mx.fast.rope(dims=rot_dims)``.
    """

    n_q, n_kv, hd, rd = (
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.rot_dims,
    )

    def one(x, weight, n_heads):
        normed = mx.fast.rms_norm(
            x.reshape(1, 1, n_heads, hd), weight, spec.eps
        ).transpose(0, 2, 1, 3)  # [1, n_heads, 1, hd]
        rot = normed[..., :rd] * spec.mscale  # YarnRoPE: self.mscale * x[..., :dims]
        scaled = mx.concatenate([rot, normed[..., rd:]], axis=-1)
        return mx.fast.rope(
            scaled,
            rd,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )

    return one(queries, q_weight, n_q), one(keys, k_weight, n_kv)


def fused_qk_yarn_full_reference(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: YarnFullSpec,
) -> tuple[mx.array, mx.array]:
    """Pure-mx reference implementing the exact math the metal kernel computes.

    No ``metal_kernel`` — only primitive ``mx`` ops — so it runs on CPU and pins
    the algorithm: RMSNorm (== ``mx.fast.rms_norm``), bf16 mscale on the rotary
    region, and the pure cos/sin from ``angles`` (== ``mx.fast.rope``).  This is
    the value the kernel targets.
    """

    n_q, n_kv, hd, rd = (
        spec.n_q_heads,
        spec.n_kv_heads,
        spec.head_dim,
        spec.rot_dims,
    )
    half = rd // 2
    cos = angles.reshape(rd)[:half].reshape(1, 1, 1, half)
    sin = angles.reshape(rd)[half:].reshape(1, 1, 1, half)
    mscale_bf = mx.array(spec.mscale, dtype=mx.bfloat16)

    def one(x, weight, n_heads):
        normed = mx.fast.rms_norm(
            x.reshape(1, 1, n_heads, hd), weight, spec.eps
        ).transpose(0, 2, 1, 3)  # [1, n_heads, 1, hd], bf16
        # Rotary region, pairs (p, p+half): bf16 mscale, then rotate in fp32.
        v1 = (normed[..., :half] * mscale_bf).astype(mx.float32)
        v2 = (normed[..., half:rd] * mscale_bf).astype(mx.float32)
        rot_lo = (v1 * cos - v2 * sin).astype(mx.bfloat16)
        rot_hi = (v1 * sin + v2 * cos).astype(mx.bfloat16)
        tail = normed[..., rd:]  # no mscale, straight through
        return mx.concatenate([rot_lo, rot_hi, tail], axis=-1)

    return one(queries, q_weight, n_q), one(keys, k_weight, n_kv)


def fused_qk_yarn_full(
    queries: mx.array,
    keys: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    spec: YarnFullSpec,
    *,
    freqs: Optional[mx.array] = None,
    offset: int | mx.array = 0,
) -> tuple[mx.array, mx.array]:
    """Fused q/k RMSNorm + partial-YaRN RoPE for one full-attention decode row.

    Returns ``(queries, keys)`` shaped ``[1, n_q_heads, 1, head_dim]`` and
    ``[1, n_kv_heads, 1, head_dim]`` — the head-major layout SDPA consumes.
    ``angles`` is the cos/sin table from :func:`build_full_yarn_angles`.

    Falls back to the stock ``YarnRoPE`` chain on any shape the kernel does not
    cover (CPU, wrong shape, ...).  The fallback needs the layer's ``freqs`` and
    ``offset`` to redo the rope; pass them so a miss stays correct.
    """

    if not is_qk_yarn_full_eligible(queries, keys, q_weight, k_weight, angles, spec):
        if freqs is None:
            raise ValueError(
                "fused_qk_yarn_full fell back to stock but no `freqs` was given; "
                "pass freqs=rope._freqs and offset so the fallback can rope."
            )
        q_out, k_out = _stock_qk_yarn_full(
            queries, keys, q_weight, k_weight, freqs, offset, spec
        )
    else:
        kernel = _qk_yarn_full_kernel(
            spec.n_q_heads,
            spec.n_kv_heads,
            spec.rot_dims,
            float(spec.mscale),
            float(spec.eps),
        )
        q_out, k_out = kernel(
            inputs=[queries, keys, q_weight, k_weight, angles],
            template=[("T", queries.dtype)],
            grid=(_SIMD * spec.total_heads, 1, 1),
            threadgroup=(_SIMD, 1, 1),
            output_shapes=[
                (1, spec.n_q_heads, 1, spec.head_dim),
                (1, spec.n_kv_heads, 1, spec.head_dim),
            ],
            output_dtypes=[queries.dtype, queries.dtype],
        )

    # Fake-speedup guard: a wrong-shaped output silently does a fraction of the
    # work and FAKES a win.  Assert the exact SDPA-facing contract.
    assert tuple(q_out.shape) == (1, spec.n_q_heads, 1, spec.head_dim), q_out.shape
    assert tuple(k_out.shape) == (1, spec.n_kv_heads, 1, spec.head_dim), k_out.shape
    return q_out, k_out
