"""Exact EXL3-Trellis operators for the pinned MiaAI DeepSeek V4 artifact.

This module is deliberately format-specific.  The target archive stores each
16x16 weight tile as 48 signed int16 words: a three-bit Trellis stream in the
tensor-core order used by ExLlamaV3 revision
``787d1582267117d6ee83c90014f03b525b14754f`` with the MCG codebook.  It is not
an MLX affine-quantized matrix and must never be passed through ``mx.dequantize``
or requantized during installation.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn


EXL3_BITS = 3
EXL3_TILE = 16
EXL3_PACKED_WORDS = 48
EXL3_HADAMARD = 128
EXL3_MCG_MULTIPLIER = 0xCBAC1FED
EXL3_M6_STAGE_VECTOR_BYTES = 16
EXL3_M6_STAGE_VECTORS_PER_K_TILE = 96

_EXPERT_KEY = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w1|w2|w3)\.rank0\."
    r"(?P<field>trellis|suh|svh|mcg)$"
)
_PROJECTION_NAMES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_DSPARK_EXPERT_KEY = re.compile(
    r"^mtp\.(?P<stage>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w1|w2|w3)\.(?P<field>weight|scale)$"
)


def _tensor_core_permutation() -> tuple[int, ...]:
    """Return ExLlamaV3's encoded-index to row-major tile permutation."""

    permutation: list[int] = []
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        col0 = thread // 4
        col1 = col0 + 8
        permutation.extend(
            (
                row0 * 16 + col0,
                row1 * 16 + col0,
                row2 * 16 + col0,
                row3 * 16 + col0,
                row0 * 16 + col1,
                row1 * 16 + col1,
                row2 * 16 + col1,
                row3 * 16 + col1,
            )
        )
    return tuple(permutation)


EXL3_TENSOR_CORE_PERMUTATION = _tensor_core_permutation()
EXL3_TENSOR_CORE_INVERSE = tuple(
    sorted(range(256), key=EXL3_TENSOR_CORE_PERMUTATION.__getitem__)
)
EXL3_M6_QUAD_DESCRIPTOR_SHA256 = (
    "158d8b220411e42a910b29a47d8af0f045b4eb1feec745cfa39b9997db72efa2"
)


class _MCGQuadDescriptorPlan(NamedTuple):
    descriptors: tuple[int, ...]
    sha256: str


@lru_cache(maxsize=1)
def _mcg_quad_descriptor_plan() -> _MCGQuadDescriptorPlan:
    """Build the fixed four-row window descriptors for one MCG/K3 tile."""

    descriptors: list[int] = []
    for quad_row in range(EXL3_TILE // 4):
        row0 = quad_row * 4
        for local_n in range(EXL3_TILE):
            tensor_cores = tuple(
                EXL3_TENSOR_CORE_INVERSE[
                    (row0 + offset) * EXL3_TILE + local_n
                ]
                for offset in range(4)
            )
            if tensor_cores != (
                tensor_cores[0],
                tensor_cores[0] + 1,
                tensor_cores[0] + 8,
                tensor_cores[0] + 9,
            ):
                raise ValueError("Mia quad MCG tensor-core ownership changed")

            pairs = []
            for tensor_core in (tensor_cores[0], tensor_cores[2]):
                bit_start = tensor_core * EXL3_BITS + 755
                bit_end = bit_start + 16 + EXL3_BITS
                raw_index0 = bit_start // 32
                raw_index2 = (bit_end - 1) // 32
                pairs.append(
                    (
                        raw_index0 % 24,
                        raw_index2 % 24,
                        (raw_index2 + 1) * 32 - bit_end,
                    )
                )
            pair0, pair1 = pairs
            if quad_row in (0, 2):
                if pair1[0] != pair0[1]:
                    raise ValueError("Mia quad three-load word ownership changed")
                index0, index1, index2 = pair0[0], pair0[1], pair1[1]
                high0_is_word1 = low1_is_word1 = 0
            else:
                index0, index1, index2 = pair0[0], pair1[1], 0
                if (
                    pair0[0] not in (index0, index1)
                    or pair0[1] not in (index0, index1)
                    or pair1[0] not in (index0, index1)
                    or pair1[1] != index1
                ):
                    raise ValueError("Mia quad two-load word ownership changed")
                high0_is_word1 = int(pair0[1] == index1)
                low1_is_word1 = int(pair1[0] == index1)
            descriptor = (
                index0
                | index1 << 5
                | index2 << 10
                | pair0[2] << 15
                | pair1[2] << 20
                | high0_is_word1 << 25
                | low1_is_word1 << 26
            )
            descriptors.append(descriptor)

    packed = struct.pack("<64I", *descriptors)
    digest = hashlib.sha256(packed).hexdigest()
    if len(descriptors) != 64 or digest != EXL3_M6_QUAD_DESCRIPTOR_SHA256:
        raise ValueError("Mia quad MCG descriptor construction changed")
    return _MCGQuadDescriptorPlan(tuple(descriptors), digest)


def decode_mcg_trellis_tile(packed: Any):
    """Decode one authentic ``[48]`` MCG/K3 tile to row-major float16.

    This is the installation-time numeric oracle, transcribed from the pinned
    ExLlamaV3 ``unpack_trellis_kernel`` / ``decode_3inst<1>`` pair.  It is kept
    off the execution path; the Metal operator consumes the packed words
    directly.
    """

    import numpy as np

    source = np.asarray(packed)
    if source.shape != (EXL3_PACKED_WORDS,):
        raise ValueError(
            f"EXL3 K3 tile must have shape ({EXL3_PACKED_WORDS},), "
            f"got {source.shape}"
        )
    if source.dtype not in (np.dtype(np.int16), np.dtype(np.uint16)):
        raise TypeError(f"EXL3 packed tile must be int16/uint16, got {source.dtype}")

    words = np.ascontiguousarray(source).view(np.uint16).view(np.uint32)
    decoded_tc = np.empty(256, dtype=np.float16)
    word_count = EXL3_BITS * 256 // 32
    for offset in range(256):
        bit0 = offset * EXL3_BITS + EXL3_BITS - 16 + 256 * EXL3_BITS
        bit1 = bit0 + 16
        index0 = bit0 // 32
        index1 = (bit1 - 1) // 32
        shift = (index1 + 1) * 32 - bit1
        low = int(words[index0 % word_count])
        high = int(words[index1 % word_count])
        state = (((low << 32) | high) >> shift) & 0xFFFF

        product = (state * EXL3_MCG_MULTIPLIER) & 0xFFFFFFFF
        # PTX lop3(a, b, c, 0x6a) is c XOR (a AND b).
        half_pair_bits = 0x3B603B60 ^ (product & 0x8FFF8FFF)
        pair = np.array(
            [half_pair_bits & 0xFFFF, half_pair_bits >> 16], dtype=np.uint16
        ).view(np.float16)
        decoded_tc[offset] = np.float16(pair[0] + pair[1])

    row_major = np.empty(256, dtype=np.float16)
    row_major[list(EXL3_TENSOR_CORE_PERMUTATION)] = decoded_tc
    return row_major.reshape(EXL3_TILE, EXL3_TILE)


@lru_cache(maxsize=None)
def _mcg_qmv_kernel(
    size_k: int,
    size_n: int,
    experts: int = 1,
    topk: int = 0,
    routed_input: bool = False,
    block_n: int = 128,
):
    if size_k % EXL3_HADAMARD or size_n % EXL3_HADAMARD:
        raise ValueError("EXL3 projection dimensions must be divisible by H128")
    if block_n not in (128, 256) or size_n % block_n:
        raise ValueError("EXL3 QMV output dimensions must tile the fixed block N")
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint KBLOCKS = {size_k // 128};
        constant constexpr uint EXPERTS = {experts};
        constant constexpr uint TOPK = {topk};
        constant constexpr uint HAD = 128;
        constant constexpr uint TILE_WORDS = 48;
        constant constexpr uint BLOCK_TILES = 8;
        constant constexpr uint BLOCK_TILES_N = {block_n // 16};
        constant constexpr float HAD_SCALE = 0.088388347648f;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline float hadamard_h128(
            float value,
            uint lane,
            threadgroup float* exchange
        ) {{
            for (uint stride = 1u; stride < 32u; stride <<= 1u) {{
                float peer = simd_shuffle_xor(value, ushort(stride));
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            for (uint stride = 32u; stride < HAD; stride <<= 1u) {{
                exchange[lane] = value;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float peer = exchange[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            return value;
        }}

        inline half decode_mcg(
            threadgroup const ushort* packed,
            uint tensor_core_offset
        ) {{
            threadgroup const uint* words =
                reinterpret_cast<threadgroup const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint half_pair_bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(half_pair_bits);
            return pair.x + pair.y;
        }}
    """
    grouped_setup = (
        """
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        size_t x_row = task;
        """
        if routed_input
        else """
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        size_t x_row = row;
        """
    )
    if not topk:
        grouped_setup = """
        uint row = threadgroup_position_in_grid.z;
        uint expert = 0u;
        size_t x_row = row;
        """
    expert_trellis_offset = (
        "(size_t)expert * (SIZE_K / 16u) * NTILES_N * TILE_WORDS + "
        if topk
        else ""
    )
    expert_suh_offset = "(size_t)expert * SIZE_K + " if topk else ""
    expert_svh_offset = "(size_t)expert * SIZE_N + " if topk else ""
    load_hadamard = """
            half scaled = half(
                x[x_row * SIZE_K + k]
                * suh[__EXPERT_SUH_OFFSET__k]
            );
            float transformed = hadamard_h128(float(scaled), lane, had_values);
            x_had[lane] = half(transformed * HAD_SCALE);
        """
    source_n128 = """
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        __GROUPED_SETUP__
        uint n = n_block * HAD + lane;

        threadgroup float had_values[HAD];
        threadgroup half x_had[HAD];
        threadgroup ushort packed_tiles[
            BLOCK_TILES * BLOCK_TILES_N * TILE_WORDS
        ];

        float accumulator = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {
            uint k = k_block * HAD + lane;
            __LOAD_HADAMARD__

            for (
                uint packed_index = lane;
                packed_index < BLOCK_TILES * BLOCK_TILES_N * TILE_WORDS;
                packed_index += HAD
            ) {
                uint tile_k = packed_index / (BLOCK_TILES_N * TILE_WORDS);
                uint remainder = packed_index % (BLOCK_TILES_N * TILE_WORDS);
                uint tile_n = remainder / TILE_WORDS;
                uint word = remainder % TILE_WORDS;
                size_t source_index =
                    __EXPERT_TRELLIS_OFFSET__
                    ((size_t)(k_block * BLOCK_TILES + tile_k) * NTILES_N
                     + n_block * BLOCK_TILES_N + tile_n) * TILE_WORDS + word;
                packed_tiles[packed_index] =
                    reinterpret_cast<const device ushort*>(trellis)[source_index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_n = lane / 16u;
            uint local_n = lane & 15u;
            for (uint local_k = 0; local_k < HAD; ++local_k) {
                uint tile_k = local_k / 16u;
                uint local_row = local_k & 15u;
                uint row_major = local_row * 16u + local_n;
                uint tensor_core = uint(TC_INV[row_major]);
                threadgroup const ushort* tile =
                    packed_tiles
                    + (tile_k * BLOCK_TILES + tile_n) * TILE_WORDS;
                half weight = decode_mcg(tile, tensor_core);
                accumulator += float(x_had[local_k]) * float(weight);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ExLlamaV3's half-output GEMV rounds before its output H128 epilogue.
        float output_had = hadamard_h128(
            float(half(accumulator)), lane, had_values
        );
        half rotated = half(output_had * HAD_SCALE);
        y[(size_t)threadgroup_position_in_grid.z * SIZE_N + n] =
            half(rotated * svh[__EXPERT_SVH_OFFSET__n]);
    """
    source_n256 = """
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        __GROUPED_SETUP__
        uint n0 = n_block * 256u + lane;
        uint n1 = n0 + HAD;

        threadgroup float had_values[HAD];
        threadgroup half x_had[HAD];
        threadgroup ushort packed_tiles[
            BLOCK_TILES * BLOCK_TILES_N * TILE_WORDS
        ];

        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {
            uint k = k_block * HAD + lane;
            __LOAD_HADAMARD__

            for (
                uint packed_index = lane;
                packed_index < BLOCK_TILES * BLOCK_TILES_N * TILE_WORDS;
                packed_index += HAD
            ) {
                uint tile_k = packed_index / (BLOCK_TILES_N * TILE_WORDS);
                uint remainder = packed_index % (BLOCK_TILES_N * TILE_WORDS);
                uint tile_n = remainder / TILE_WORDS;
                uint word = remainder % TILE_WORDS;
                size_t source_index =
                    __EXPERT_TRELLIS_OFFSET__
                    ((size_t)(k_block * BLOCK_TILES + tile_k) * NTILES_N
                     + n_block * BLOCK_TILES_N + tile_n) * TILE_WORDS + word;
                packed_tiles[packed_index] =
                    reinterpret_cast<const device ushort*>(trellis)[source_index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_n0 = lane / 16u;
            uint tile_n1 = tile_n0 + BLOCK_TILES;
            uint local_n = lane & 15u;
            for (uint local_k = 0; local_k < HAD; ++local_k) {
                uint tile_k = local_k / 16u;
                uint local_row = local_k & 15u;
                uint row_major = local_row * 16u + local_n;
                uint tensor_core = uint(TC_INV[row_major]);
                threadgroup const ushort* tile0 =
                    packed_tiles
                    + (tile_k * BLOCK_TILES_N + tile_n0) * TILE_WORDS;
                threadgroup const ushort* tile1 =
                    packed_tiles
                    + (tile_k * BLOCK_TILES_N + tile_n1) * TILE_WORDS;
                half weight0 = decode_mcg(tile0, tensor_core);
                half weight1 = decode_mcg(tile1, tensor_core);
                float value = float(x_had[local_k]);
                accumulator0 += value * float(weight0);
                accumulator1 += value * float(weight1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Preserve ExLlamaV3's half-output rounding independently per N128 panel.
        float output_had0 = hadamard_h128(
            float(half(accumulator0)), lane, had_values
        );
        half rotated0 = half(output_had0 * HAD_SCALE);
        y[(size_t)threadgroup_position_in_grid.z * SIZE_N + n0] =
            half(rotated0 * svh[__EXPERT_SVH_OFFSET__n0]);

        float output_had1 = hadamard_h128(
            float(half(accumulator1)), lane, had_values
        );
        half rotated1 = half(output_had1 * HAD_SCALE);
        y[(size_t)threadgroup_position_in_grid.z * SIZE_N + n1] =
            half(rotated1 * svh[__EXPERT_SVH_OFFSET__n1]);
    """
    source = source_n256 if block_n == 256 else source_n128
    source = (
        source.replace("__GROUPED_SETUP__", grouped_setup)
        .replace("__LOAD_HADAMARD__", load_hadamard)
        .replace("__EXPERT_SUH_OFFSET__", expert_suh_offset)
        .replace("__EXPERT_TRELLIS_OFFSET__", expert_trellis_offset)
        .replace("__EXPERT_SVH_OFFSET__", expert_svh_offset)
    )
    input_names = ["x", "trellis", "suh", "svh"]
    if topk:
        input_names.append("expert_ids")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_mcg_qmv_k{size_k}_n{size_n}"
            f"_e{experts}_t{topk}_r{int(routed_input)}_bn{block_n}_v5"
        ),
        input_names=input_names,
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _m6_quad_qmv_kernel(
    size_k: int,
    size_n: int,
    routed_input: bool,
):
    """Build the fixed-M6 BN256 QMV with four-row MCG window reuse."""

    if (size_k, size_n, routed_input) not in (
        (4096, 2048, False),
        (2048, 4096, True),
    ):
        raise ValueError("Mia quad QMV requires an exact gate/up or down bank")
    descriptor_plan = _mcg_quad_descriptor_plan()
    descriptors = ",".join(str(value) for value in descriptor_plan.descriptors)
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint KBLOCKS = {size_k // 128};
        constant constexpr uint EXPERTS = 216;
        constant constexpr uint TOPK = 6;
        constant constexpr uint HAD = 128;
        constant constexpr uint TILE_WORDS = 48;
        constant constexpr uint TILE_VECTORS = 6;
        constant constexpr uint BLOCK_TILES = 8;
        constant constexpr uint BLOCK_TILES_N = 16;
        constant constexpr uint STAGE_VECTORS_PER_K_TILE = 96;
        constant constexpr float HAD_SCALE = 0.088388347648f;
        constant uint QUAD_DESCRIPTORS[64] = {{ {descriptors} }};

        inline float hadamard_h128(
            float value,
            uint lane,
            threadgroup float* exchange
        ) {{
            for (uint stride = 1u; stride < 32u; stride <<= 1u) {{
                float peer = simd_shuffle_xor(value, ushort(stride));
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            for (uint stride = 32u; stride < HAD; stride <<= 1u) {{
                exchange[lane] = value;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float peer = exchange[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            return value;
        }}

        inline uint merge_mcg_window(uint low, uint high, uint shift) {{
            if (shift == 0u) {{
                return high;
            }}
            return (high >> shift) | (low << (32u - shift));
        }}

        inline half4 decode_mcg_states(
            uint state0,
            uint state1,
            uint state2,
            uint state3
        ) {{
            uint product0 = state0 * 0xCBAC1FEDu;
            uint product1 = state1 * 0xCBAC1FEDu;
            uint product2 = state2 * 0xCBAC1FEDu;
            uint product3 = state3 * 0xCBAC1FEDu;
            uint half_pair_bits0 =
                0x3B603B60u ^ (product0 & 0x8FFF8FFFu);
            uint half_pair_bits1 =
                0x3B603B60u ^ (product1 & 0x8FFF8FFFu);
            uint half_pair_bits2 =
                0x3B603B60u ^ (product2 & 0x8FFF8FFFu);
            uint half_pair_bits3 =
                0x3B603B60u ^ (product3 & 0x8FFF8FFFu);
            half2 pair0 = as_type<half2>(half_pair_bits0);
            half2 pair1 = as_type<half2>(half_pair_bits1);
            half2 pair2 = as_type<half2>(half_pair_bits2);
            half2 pair3 = as_type<half2>(half_pair_bits3);
            return half4(
                pair0.x + pair0.y,
                pair1.x + pair1.y,
                pair2.x + pair2.y,
                pair3.x + pair3.y
            );
        }}

        inline half4 decode_mcg_quad3(
            threadgroup const ushort* packed,
            uint descriptor
        ) {{
            threadgroup const uint* words =
                reinterpret_cast<threadgroup const uint*>(packed);
            uint index0 = uint(descriptor) & 0x1fu;
            uint index1 = (uint(descriptor) >> 5u) & 0x1fu;
            uint index2 = (uint(descriptor) >> 10u) & 0x1fu;
            uint shift0 = (uint(descriptor) >> 15u) & 0x1fu;
            uint shift1 = (uint(descriptor) >> 20u) & 0x1fu;
            uint word0 = words[index0];
            uint word1 = words[index1];
            uint word2 = words[index2];
            uint window0 = merge_mcg_window(word0, word1, shift0);
            uint window1 = merge_mcg_window(word1, word2, shift1);
            uint state0 = (window0 >> 3u) & 0xffffu;
            uint state1 = window0 & 0xffffu;
            uint state2 = (window1 >> 3u) & 0xffffu;
            uint state3 = window1 & 0xffffu;
            return decode_mcg_states(state0, state1, state2, state3);
        }}

        inline half4 decode_mcg_quad2(
            threadgroup const ushort* packed,
            uint descriptor
        ) {{
            threadgroup const uint* words =
                reinterpret_cast<threadgroup const uint*>(packed);
            uint index0 = uint(descriptor) & 0x1fu;
            uint index1 = (uint(descriptor) >> 5u) & 0x1fu;
            uint shift0 = (uint(descriptor) >> 15u) & 0x1fu;
            uint shift1 = (uint(descriptor) >> 20u) & 0x1fu;
            uint word0 = words[index0];
            uint word1 = words[index1];
            bool high0_is_word1 = (uint(descriptor) & (1u << 25u)) != 0u;
            bool low1_is_word1 = (uint(descriptor) & (1u << 26u)) != 0u;
            uint high0 = select(word0, word1, high0_is_word1);
            uint low1 = select(word0, word1, low1_is_word1);
            uint window0 = merge_mcg_window(word0, high0, shift0);
            uint window1 = merge_mcg_window(low1, word1, shift1);
            uint state0 = (window0 >> 3u) & 0xffffu;
            uint state1 = window0 & 0xffffu;
            uint state2 = (window1 >> 3u) & 0xffffu;
            uint state3 = window1 & 0xffffu;
            return decode_mcg_states(state0, state1, state2, state3);
        }}
    """
    quad_blocks = []
    for quad_row, decoder in enumerate(
        ("decode_mcg_quad3", "decode_mcg_quad2") * 2
    ):
        quad_blocks.append(
            f"""
            uint local_k{quad_row} = tile_k * 16u + {quad_row * 4}u;
            uint descriptor{quad_row} =
                QUAD_DESCRIPTORS[{quad_row * 16}u + local_n];
            half4 weights{quad_row}_0 = {decoder}(tile0, descriptor{quad_row});
            half4 weights{quad_row}_1 = {decoder}(tile1, descriptor{quad_row});
            float value{quad_row}_0 = float(x_had[local_k{quad_row}]);
            accumulator0 += value{quad_row}_0 * float(weights{quad_row}_0.x);
            accumulator1 += value{quad_row}_0 * float(weights{quad_row}_1.x);
            float value{quad_row}_1 = float(x_had[local_k{quad_row} + 1u]);
            accumulator0 += value{quad_row}_1 * float(weights{quad_row}_0.y);
            accumulator1 += value{quad_row}_1 * float(weights{quad_row}_1.y);
            float value{quad_row}_2 = float(x_had[local_k{quad_row} + 2u]);
            accumulator0 += value{quad_row}_2 * float(weights{quad_row}_0.z);
            accumulator1 += value{quad_row}_2 * float(weights{quad_row}_1.z);
            float value{quad_row}_3 = float(x_had[local_k{quad_row} + 3u]);
            accumulator0 += value{quad_row}_3 * float(weights{quad_row}_0.w);
            accumulator1 += value{quad_row}_3 * float(weights{quad_row}_1.w);
            """
        )
    quad_decode = "".join(quad_blocks)
    x_row = "task" if routed_input else "row"
    source = f"""
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        size_t x_row = {x_row};
        uint n0 = n_block * 256u + lane;
        uint n1 = n0 + HAD;

        threadgroup float had_values[HAD];
        threadgroup half x_had[HAD];
        threadgroup uint4 packed_tile_vectors[
            BLOCK_TILES * STAGE_VECTORS_PER_K_TILE
        ];
        threadgroup ushort* packed_tiles =
            reinterpret_cast<threadgroup ushort*>(packed_tile_vectors);
        device const uint4* trellis_vectors =
            reinterpret_cast<const device uint4*>(trellis);
        size_t expert_base =
            (size_t)expert * (SIZE_K / 16u) * NTILES_N * TILE_VECTORS;
        size_t n_block_offset =
            (size_t)n_block * STAGE_VECTORS_PER_K_TILE;

        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {{
            uint k = k_block * HAD + lane;
            half scaled = half(
                x[x_row * SIZE_K + k]
                * suh[(size_t)expert * SIZE_K + k]
            );
            float transformed = hadamard_h128(
                float(scaled), lane, had_values
            );
            x_had[lane] = half(transformed * HAD_SCALE);

            size_t k_base = expert_base
                + (size_t)k_block * BLOCK_TILES * NTILES_N * TILE_VECTORS;
            size_t n_base = k_base + n_block_offset;
            if (lane < STAGE_VECTORS_PER_K_TILE) {{
                for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                    packed_tile_vectors[
                        tile_k * STAGE_VECTORS_PER_K_TILE + lane
                    ] = trellis_vectors[
                        n_base + (size_t)tile_k * NTILES_N * TILE_VECTORS + lane
                    ];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_n0 = lane / 16u;
            uint tile_n1 = tile_n0 + BLOCK_TILES;
            uint local_n = lane & 15u;
            for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                threadgroup const ushort* tile0 =
                    packed_tiles
                    + (tile_k * BLOCK_TILES_N + tile_n0) * TILE_WORDS;
                threadgroup const ushort* tile1 =
                    packed_tiles
                    + (tile_k * BLOCK_TILES_N + tile_n1) * TILE_WORDS;
                {quad_decode}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float output_had0 = hadamard_h128(
            float(half(accumulator0)), lane, had_values
        );
        half rotated0 = half(output_had0 * HAD_SCALE);
        y[(size_t)task * SIZE_N + n0] = half(
            rotated0 * svh[(size_t)expert * SIZE_N + n0]
        );
        float output_had1 = hadamard_h128(
            float(half(accumulator1)), lane, had_values
        );
        half rotated1 = half(output_had1 * HAD_SCALE);
        y[(size_t)task * SIZE_N + n1] = half(
            rotated1 * svh[(size_t)expert * SIZE_N + n1]
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_m6_quad_mcg_qmv_k{size_k}_n{size_n}"
            f"_e216_t6_r{int(routed_input)}_bn256_u4stage_v2"
        ),
        input_names=["x", "trellis", "suh", "svh", "expert_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


def _m6_quad_inner_parts(size_k: int, size_n: int):
    """Return the sealed MCG header and per-panel FMA used by M6 inner stages."""

    if (size_k, size_n) not in ((4096, 2048), (2048, 4096)):
        raise ValueError("Mia M6 inner stages require an exact FC1 or FC2 bank")
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint KBLOCKS = {size_k // 128};
        constant constexpr uint TOPK = 6;
        constant constexpr uint HAD = 128;
        constant constexpr uint TILE_WORDS = 48;
        constant constexpr uint TILE_VECTORS = 6;
        constant constexpr uint BLOCK_TILES = 8;
        constant constexpr uint BLOCK_TILES_N = 16;
        constant constexpr uint STAGE_VECTORS_PER_K_TILE = 96;
        constant constexpr float HAD_SCALE = 0.088388347648f;
        inline float hadamard_h128(
            float value,
            uint lane,
            threadgroup float* exchange
        ) {{
            for (uint stride = 1u; stride < 32u; stride <<= 1u) {{
                float peer = simd_shuffle_xor(value, ushort(stride));
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            for (uint stride = 32u; stride < HAD; stride <<= 1u) {{
                exchange[lane] = value;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float peer = exchange[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                value = (lane & stride) ? (peer - value) : (value + peer);
            }}
            return value;
        }}

        inline uint merge_mcg_window(uint low, uint high, uint shift) {{
            if (shift == 0u) {{
                return high;
            }}
            return (high >> shift) | (low << (32u - shift));
        }}

        inline half4 decode_mcg_states(
            uint state0,
            uint state1,
            uint state2,
            uint state3
        ) {{
            uint product0 = state0 * 0xCBAC1FEDu;
            uint product1 = state1 * 0xCBAC1FEDu;
            uint product2 = state2 * 0xCBAC1FEDu;
            uint product3 = state3 * 0xCBAC1FEDu;
            uint half_pair_bits0 =
                0x3B603B60u ^ (product0 & 0x8FFF8FFFu);
            uint half_pair_bits1 =
                0x3B603B60u ^ (product1 & 0x8FFF8FFFu);
            uint half_pair_bits2 =
                0x3B603B60u ^ (product2 & 0x8FFF8FFFu);
            uint half_pair_bits3 =
                0x3B603B60u ^ (product3 & 0x8FFF8FFFu);
            half2 pair0 = as_type<half2>(half_pair_bits0);
            half2 pair1 = as_type<half2>(half_pair_bits1);
            half2 pair2 = as_type<half2>(half_pair_bits2);
            half2 pair3 = as_type<half2>(half_pair_bits3);
            return half4(
                pair0.x + pair0.y,
                pair1.x + pair1.y,
                pair2.x + pair2.y,
                pair3.x + pair3.y
            );
        }}

        struct QuadWeights {{
            half4 q0;
            half4 q1;
            half4 q2;
            half4 q3;
        }};

        struct QuadWeightPair {{
            QuadWeights first;
            QuadWeights second;
        }};

        inline half4 decode_mcg_windows(
            uint low0,
            uint high0,
            uint shift0,
            uint low1,
            uint high1,
            uint shift1
        ) {{
            uint window0 = merge_mcg_window(low0, high0, shift0);
            uint window1 = merge_mcg_window(low1, high1, shift1);
            return decode_mcg_states(
                (window0 >> 3u) & 0xffffu,
                window0 & 0xffffu,
                (window1 >> 3u) & 0xffffu,
                window1 & 0xffffu
            );
        }}

        inline QuadWeights decode_mcg_column_words(
            threadgroup const uint* words,
            uint start,
            bool high
        ) {{
            uint next1 = (start + 1u) % 24u;
            uint next2 = (start + 2u) % 24u;
            uint next3 = (start + 3u) % 24u;
            uint a = words[start];
            uint b = words[next1];
            uint c = words[next2];
            uint d = words[next3];
            QuadWeights result;
            result.q0 = decode_mcg_windows(
                a, b, select(26u, 14u, high),
                b, select(b, c, high), select(2u, 22u, high)
            );
            result.q1 = decode_mcg_windows(
                c, select(c, d, high), select(10u, 30u, high),
                select(c, d, high), d, select(18u, 6u, high)
            );
            result.q2 = decode_mcg_windows(
                select(a, b, high), b, select(20u, 8u, high),
                b, c, select(28u, 16u, high)
            );
            result.q3 = decode_mcg_windows(
                c, select(c, d, high), select(4u, 24u, high),
                d, d, select(12u, 0u, high)
            );
            return result;
        }}

        inline QuadWeightPair decode_mcg_column_pair(
            threadgroup const ushort* packed0,
            threadgroup const ushort* packed1,
            uint local_n
        ) {{
            uint u = local_n & 7u;
            bool high = local_n >= 8u;
            uint start = select(23u, 3u * u - 1u, u != 0u);
            QuadWeightPair result;
            result.first = decode_mcg_column_words(
                reinterpret_cast<threadgroup const uint*>(packed0),
                start,
                high
            );
            result.second = decode_mcg_column_words(
                reinterpret_cast<threadgroup const uint*>(packed1),
                start,
                high
            );
            return result;
        }}
    """

    def fma_source(prefix: str, accumulator0: str, accumulator1: str) -> str:
        blocks = [
            f"""
            QuadWeightPair {prefix}_weights = decode_mcg_column_pair(
                {prefix}_tile0,
                {prefix}_tile1,
                {prefix}_local_n
            );
            """
        ]
        for quad_row in range(4):
            blocks.append(
                f"""
                uint {prefix}_local_k{quad_row} =
                    tile_k * 16u + {quad_row * 4}u;
                float {prefix}_value{quad_row}_0 = float(
                    x_had[{prefix}_local_k{quad_row}]
                );
                {accumulator0} += {prefix}_value{quad_row}_0
                    * float({prefix}_weights.first.q{quad_row}.x);
                {accumulator1} += {prefix}_value{quad_row}_0
                    * float({prefix}_weights.second.q{quad_row}.x);
                float {prefix}_value{quad_row}_1 = float(
                    x_had[{prefix}_local_k{quad_row} + 1u]
                );
                {accumulator0} += {prefix}_value{quad_row}_1
                    * float({prefix}_weights.first.q{quad_row}.y);
                {accumulator1} += {prefix}_value{quad_row}_1
                    * float({prefix}_weights.second.q{quad_row}.y);
                float {prefix}_value{quad_row}_2 = float(
                    x_had[{prefix}_local_k{quad_row} + 2u]
                );
                {accumulator0} += {prefix}_value{quad_row}_2
                    * float({prefix}_weights.first.q{quad_row}.z);
                {accumulator1} += {prefix}_value{quad_row}_2
                    * float({prefix}_weights.second.q{quad_row}.z);
                float {prefix}_value{quad_row}_3 = float(
                    x_had[{prefix}_local_k{quad_row} + 3u]
                );
                {accumulator0} += {prefix}_value{quad_row}_3
                    * float({prefix}_weights.first.q{quad_row}.w);
                {accumulator1} += {prefix}_value{quad_row}_3
                    * float({prefix}_weights.second.q{quad_row}.w);
                """
            )
        return "".join(blocks)

    return header, fma_source


@lru_cache(maxsize=1)
def _m6_dual_fc1_input_kernel():
    """Build gate/up EXL3 input rotations once for each physical M6 route."""

    header = r"""
        using namespace metal;
        constant constexpr uint HIDDEN = 4096u;
        constant constexpr uint TOPK = 6u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;

        inline float4 hadamard_h128_quad(float4 value, uint lane) {
            float s0 = value.x + value.y;
            float d0 = value.x - value.y;
            float s1 = value.z + value.w;
            float d1 = value.z - value.w;
            float h0 = s0 + s1;
            float h1 = d0 + d1;
            float h2 = s0 - s1;
            float h3 = d0 - d1;
            for (uint step = 0u; step < 5u; ++step) {
                uint stride = 1u << step;
                float p0 = simd_shuffle_xor(h0, ushort(stride));
                float p1 = simd_shuffle_xor(h1, ushort(stride));
                float p2 = simd_shuffle_xor(h2, ushort(stride));
                float p3 = simd_shuffle_xor(h3, ushort(stride));
                if (lane & stride) {
                    h0 = p0 - h0;
                    h1 = p1 - h1;
                    h2 = p2 - h2;
                    h3 = p3 - h3;
                } else {
                    h0 = h0 + p0;
                    h1 = h1 + p1;
                    h2 = h2 + p2;
                    h3 = h3 + p3;
                }
            }
            return float4(h0, h1, h2, h3);
        }
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint k_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        uint k0 = k_block * HAD + lane * 4u;

        half x0 = half(x[(size_t)row * HIDDEN + k0]);
        half x1 = half(x[(size_t)row * HIDDEN + k0 + 1u]);
        half x2 = half(x[(size_t)row * HIDDEN + k0 + 2u]);
        half x3 = half(x[(size_t)row * HIDDEN + k0 + 3u]);
        half gate_scaled0 = half(
            x0 * gate_suh[(size_t)expert * HIDDEN + k0]
        );
        half gate_scaled1 = half(
            x1 * gate_suh[(size_t)expert * HIDDEN + k0 + 1u]
        );
        half gate_scaled2 = half(
            x2 * gate_suh[(size_t)expert * HIDDEN + k0 + 2u]
        );
        half gate_scaled3 = half(
            x3 * gate_suh[(size_t)expert * HIDDEN + k0 + 3u]
        );
        float4 gate_transformed = hadamard_h128_quad(
            float4(
                float(gate_scaled0),
                float(gate_scaled1),
                float(gate_scaled2),
                float(gate_scaled3)
            ),
            lane
        );
        gate_h[(size_t)task * HIDDEN + k0] = half(
            gate_transformed.x * HAD_SCALE
        );
        gate_h[(size_t)task * HIDDEN + k0 + 1u] = half(
            gate_transformed.y * HAD_SCALE
        );
        gate_h[(size_t)task * HIDDEN + k0 + 2u] = half(
            gate_transformed.z * HAD_SCALE
        );
        gate_h[(size_t)task * HIDDEN + k0 + 3u] = half(
            gate_transformed.w * HAD_SCALE
        );

        half up_scaled0 = half(
            x0 * up_suh[(size_t)expert * HIDDEN + k0]
        );
        half up_scaled1 = half(
            x1 * up_suh[(size_t)expert * HIDDEN + k0 + 1u]
        );
        half up_scaled2 = half(
            x2 * up_suh[(size_t)expert * HIDDEN + k0 + 2u]
        );
        half up_scaled3 = half(
            x3 * up_suh[(size_t)expert * HIDDEN + k0 + 3u]
        );
        float4 up_transformed = hadamard_h128_quad(
            float4(
                float(up_scaled0),
                float(up_scaled1),
                float(up_scaled2),
                float(up_scaled3)
            ),
            lane
        );
        up_h[(size_t)task * HIDDEN + k0] = half(
            up_transformed.x * HAD_SCALE
        );
        up_h[(size_t)task * HIDDEN + k0 + 1u] = half(
            up_transformed.y * HAD_SCALE
        );
        up_h[(size_t)task * HIDDEN + k0 + 2u] = half(
            up_transformed.z * HAD_SCALE
        );
        up_h[(size_t)task * HIDDEN + k0 + 3u] = half(
            up_transformed.w * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_exl3_m6_dual_fc1_input_h4096_v2",
        input_names=["x", "gate_suh", "up_suh", "expert_ids"],
        output_names=["gate_h", "up_h"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=1)
def _m6_dual_fc1_inner_kernel():
    """Build exact-M6 dual FC1 over pre-rotated FP16 route rows."""

    header, build_fma = _m6_quad_inner_parts(4096, 2048)
    gate_fma = build_fma("gate", "gate_accumulator0", "gate_accumulator1")
    up_fma = build_fma("up", "up_accumulator0", "up_accumulator1")
    source = f"""
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        uint n0 = n_block * 256u + lane;
        uint n1 = n0 + HAD;

        threadgroup half x_had[HAD];
        threadgroup uint4 packed_tile_vectors[
            BLOCK_TILES * STAGE_VECTORS_PER_K_TILE
        ];
        threadgroup ushort* packed_tiles =
            reinterpret_cast<threadgroup ushort*>(packed_tile_vectors);
        device const uint4* gate_trellis_vectors =
            reinterpret_cast<const device uint4*>(gate_trellis);
        device const uint4* up_trellis_vectors =
            reinterpret_cast<const device uint4*>(up_trellis);
        size_t expert_base =
            (size_t)expert * (SIZE_K / 16u) * NTILES_N * TILE_VECTORS;
        size_t n_block_offset =
            (size_t)n_block * STAGE_VECTORS_PER_K_TILE;

        float gate_accumulator0 = 0.0f;
        float gate_accumulator1 = 0.0f;
        float up_accumulator0 = 0.0f;
        float up_accumulator1 = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {{
            uint k = k_block * HAD + lane;
            x_had[lane] = gate_h[(size_t)task * SIZE_K + k];

            size_t k_base = expert_base
                + (size_t)k_block * BLOCK_TILES * NTILES_N * TILE_VECTORS;
            size_t n_base = k_base + n_block_offset;
            if (lane < STAGE_VECTORS_PER_K_TILE) {{
                for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                    packed_tile_vectors[
                        tile_k * STAGE_VECTORS_PER_K_TILE + lane
                    ] = gate_trellis_vectors[
                        n_base + (size_t)tile_k * NTILES_N * TILE_VECTORS + lane
                    ];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint gate_tile_n0 = lane / 16u;
            uint gate_tile_n1 = gate_tile_n0 + BLOCK_TILES;
            uint gate_local_n = lane & 15u;
            for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                threadgroup const ushort* gate_tile0 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + gate_tile_n0) * TILE_WORDS;
                threadgroup const ushort* gate_tile1 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + gate_tile_n1) * TILE_WORDS;
                {gate_fma}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            x_had[lane] = up_h[(size_t)task * SIZE_K + k];
            if (lane < STAGE_VECTORS_PER_K_TILE) {{
                for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                    packed_tile_vectors[
                        tile_k * STAGE_VECTORS_PER_K_TILE + lane
                    ] = up_trellis_vectors[
                        n_base + (size_t)tile_k * NTILES_N * TILE_VECTORS + lane
                    ];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint up_tile_n0 = lane / 16u;
            uint up_tile_n1 = up_tile_n0 + BLOCK_TILES;
            uint up_local_n = lane & 15u;
            for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                threadgroup const ushort* up_tile0 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + up_tile_n0) * TILE_WORDS;
                threadgroup const ushort* up_tile1 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + up_tile_n1) * TILE_WORDS;
                {up_fma}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        gate_inner[(size_t)task * SIZE_N + n0] = half(gate_accumulator0);
        gate_inner[(size_t)task * SIZE_N + n1] = half(gate_accumulator1);
        up_inner[(size_t)task * SIZE_N + n0] = half(up_accumulator0);
        up_inner[(size_t)task * SIZE_N + n1] = half(up_accumulator1);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_exl3_m6_dual_fc1_inner_h4096_i2048_v3",
        input_names=[
            "gate_h",
            "up_h",
            "gate_trellis",
            "up_trellis",
            "expert_ids",
        ],
        output_names=["gate_inner", "up_inner"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=1)
def _m6_clamp10_activation_down_kernel():
    """Build exact output rotations, clamp-10 SwiGLU, and down input rotation."""

    header = r"""
        using namespace metal;
        constant constexpr uint INTERMEDIATE = 2048u;
        constant constexpr uint TOPK = 6u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;

        inline float4 hadamard_h128_quad(float4 value, uint lane) {
            float s0 = value.x + value.y;
            float d0 = value.x - value.y;
            float s1 = value.z + value.w;
            float d1 = value.z - value.w;
            float h0 = s0 + s1;
            float h1 = d0 + d1;
            float h2 = s0 - s1;
            float h3 = d0 - d1;
            for (uint step = 0u; step < 5u; ++step) {
                uint stride = 1u << step;
                float p0 = simd_shuffle_xor(h0, ushort(stride));
                float p1 = simd_shuffle_xor(h1, ushort(stride));
                float p2 = simd_shuffle_xor(h2, ushort(stride));
                float p3 = simd_shuffle_xor(h3, ushort(stride));
                if (lane & stride) {
                    h0 = p0 - h0;
                    h1 = p1 - h1;
                    h2 = p2 - h2;
                    h3 = p3 - h3;
                } else {
                    h0 = h0 + p0;
                    h1 = h1 + p1;
                    h2 = h2 + p2;
                    h3 = h3 + p3;
                }
            }
            return float4(h0, h1, h2, h3);
        }

        inline half sigmoid_mlx_exact(half value) {
            auto y = 1 / (1 + metal::exp(metal::abs(value)));
            return (value < half(0.0f)) ? y : 1 - y;
        }
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        uint column0 = block * HAD + lane * 4u;

        float4 gate_had = hadamard_h128_quad(
            float4(
                float(gate_inner[(size_t)task * INTERMEDIATE + column0]),
                float(gate_inner[(size_t)task * INTERMEDIATE + column0 + 1u]),
                float(gate_inner[(size_t)task * INTERMEDIATE + column0 + 2u]),
                float(gate_inner[(size_t)task * INTERMEDIATE + column0 + 3u])
            ),
            lane
        );
        half gate_rotated0 = half(gate_had.x * HAD_SCALE);
        half gate_rotated1 = half(gate_had.y * HAD_SCALE);
        half gate_rotated2 = half(gate_had.z * HAD_SCALE);
        half gate_rotated3 = half(gate_had.w * HAD_SCALE);
        half gate0 = half(
            gate_rotated0
            * gate_svh[(size_t)expert * INTERMEDIATE + column0]
        );
        half gate1 = half(
            gate_rotated1
            * gate_svh[(size_t)expert * INTERMEDIATE + column0 + 1u]
        );
        half gate2 = half(
            gate_rotated2
            * gate_svh[(size_t)expert * INTERMEDIATE + column0 + 2u]
        );
        half gate3 = half(
            gate_rotated3
            * gate_svh[(size_t)expert * INTERMEDIATE + column0 + 3u]
        );

        float4 up_had = hadamard_h128_quad(
            float4(
                float(up_inner[(size_t)task * INTERMEDIATE + column0]),
                float(up_inner[(size_t)task * INTERMEDIATE + column0 + 1u]),
                float(up_inner[(size_t)task * INTERMEDIATE + column0 + 2u]),
                float(up_inner[(size_t)task * INTERMEDIATE + column0 + 3u])
            ),
            lane
        );
        half up_rotated0 = half(up_had.x * HAD_SCALE);
        half up_rotated1 = half(up_had.y * HAD_SCALE);
        half up_rotated2 = half(up_had.z * HAD_SCALE);
        half up_rotated3 = half(up_had.w * HAD_SCALE);
        half up0 = half(
            up_rotated0
            * up_svh[(size_t)expert * INTERMEDIATE + column0]
        );
        half up1 = half(
            up_rotated1
            * up_svh[(size_t)expert * INTERMEDIATE + column0 + 1u]
        );
        half up2 = half(
            up_rotated2
            * up_svh[(size_t)expert * INTERMEDIATE + column0 + 2u]
        );
        half up3 = half(
            up_rotated3
            * up_svh[(size_t)expert * INTERMEDIATE + column0 + 3u]
        );

        gate0 = min(gate0, half(10.0f));
        gate1 = min(gate1, half(10.0f));
        gate2 = min(gate2, half(10.0f));
        gate3 = min(gate3, half(10.0f));
        up0 = min(max(up0, half(-10.0f)), half(10.0f));
        up1 = min(max(up1, half(-10.0f)), half(10.0f));
        up2 = min(max(up2, half(-10.0f)), half(10.0f));
        up3 = min(max(up3, half(-10.0f)), half(10.0f));
        half silu0 = half(gate0 * sigmoid_mlx_exact(gate0));
        half silu1 = half(gate1 * sigmoid_mlx_exact(gate1));
        half silu2 = half(gate2 * sigmoid_mlx_exact(gate2));
        half silu3 = half(gate3 * sigmoid_mlx_exact(gate3));
        half activated0 = half(silu0 * up0);
        half activated1 = half(silu1 * up1);
        half activated2 = half(silu2 * up2);
        half activated3 = half(silu3 * up3);
        half down_scaled0 = half(
            activated0
            * down_suh[(size_t)expert * INTERMEDIATE + column0]
        );
        half down_scaled1 = half(
            activated1
            * down_suh[(size_t)expert * INTERMEDIATE + column0 + 1u]
        );
        half down_scaled2 = half(
            activated2
            * down_suh[(size_t)expert * INTERMEDIATE + column0 + 2u]
        );
        half down_scaled3 = half(
            activated3
            * down_suh[(size_t)expert * INTERMEDIATE + column0 + 3u]
        );
        float4 down_had = hadamard_h128_quad(
            float4(
                float(down_scaled0),
                float(down_scaled1),
                float(down_scaled2),
                float(down_scaled3)
            ),
            lane
        );
        down_h[(size_t)task * INTERMEDIATE + column0] = half(
            down_had.x * HAD_SCALE
        );
        down_h[(size_t)task * INTERMEDIATE + column0 + 1u] = half(
            down_had.y * HAD_SCALE
        );
        down_h[(size_t)task * INTERMEDIATE + column0 + 2u] = half(
            down_had.z * HAD_SCALE
        );
        down_h[(size_t)task * INTERMEDIATE + column0 + 3u] = half(
            down_had.w * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_exl3_m6_clamp10_activation_down_i2048_v2",
        input_names=[
            "gate_inner",
            "up_inner",
            "gate_svh",
            "up_svh",
            "down_suh",
            "expert_ids",
        ],
        output_names=["down_h"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=1)
def _m6_down_inner_kernel():
    """Build exact-M6 FC2 from an already rotated FP16 intermediate."""

    header, build_fma = _m6_quad_inner_parts(2048, 4096)
    down_fma = build_fma("down", "accumulator0", "accumulator1")
    source = f"""
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        uint n0 = n_block * 256u + lane;
        uint n1 = n0 + HAD;

        threadgroup half x_had[HAD];
        threadgroup uint4 packed_tile_vectors[
            BLOCK_TILES * STAGE_VECTORS_PER_K_TILE
        ];
        threadgroup ushort* packed_tiles =
            reinterpret_cast<threadgroup ushort*>(packed_tile_vectors);
        device const uint4* trellis_vectors =
            reinterpret_cast<const device uint4*>(down_trellis);
        size_t expert_base =
            (size_t)expert * (SIZE_K / 16u) * NTILES_N * TILE_VECTORS;
        size_t n_block_offset =
            (size_t)n_block * STAGE_VECTORS_PER_K_TILE;

        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {{
            uint k = k_block * HAD + lane;
            x_had[lane] = down_h[(size_t)task * SIZE_K + k];
            size_t k_base = expert_base
                + (size_t)k_block * BLOCK_TILES * NTILES_N * TILE_VECTORS;
            size_t n_base = k_base + n_block_offset;
            if (lane < STAGE_VECTORS_PER_K_TILE) {{
                for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                    packed_tile_vectors[
                        tile_k * STAGE_VECTORS_PER_K_TILE + lane
                    ] = trellis_vectors[
                        n_base + (size_t)tile_k * NTILES_N * TILE_VECTORS + lane
                    ];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint down_tile_n0 = lane / 16u;
            uint down_tile_n1 = down_tile_n0 + BLOCK_TILES;
            uint down_local_n = lane & 15u;
            for (uint tile_k = 0; tile_k < BLOCK_TILES; ++tile_k) {{
                threadgroup const ushort* down_tile0 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + down_tile_n0) * TILE_WORDS;
                threadgroup const ushort* down_tile1 = packed_tiles
                    + (tile_k * BLOCK_TILES_N + down_tile_n1) * TILE_WORDS;
                {down_fma}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        down_inner[(size_t)task * SIZE_N + n0] = half(accumulator0);
        down_inner[(size_t)task * SIZE_N + n1] = half(accumulator1);
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_exl3_m6_down_inner_i2048_h4096_v2",
        input_names=["down_h", "down_trellis", "expert_ids"],
        output_names=["down_inner"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=1)
def _m6_direct_final_tail_kernel():
    """Build exact FC2 output rotation and serial BF16 top-6/shared tail."""

    header = r"""
        using namespace metal;
        constant constexpr uint HIDDEN = 4096u;
        constant constexpr uint TOPK = 6u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;

        inline float4 hadamard_h128_quad(float4 value, uint lane) {
            float s0 = value.x + value.y;
            float d0 = value.x - value.y;
            float s1 = value.z + value.w;
            float d1 = value.z - value.w;
            float h0 = s0 + s1;
            float h1 = d0 + d1;
            float h2 = s0 - s1;
            float h3 = d0 - d1;
            for (uint step = 0u; step < 5u; ++step) {
                uint stride = 1u << step;
                float p0 = simd_shuffle_xor(h0, ushort(stride));
                float p1 = simd_shuffle_xor(h1, ushort(stride));
                float p2 = simd_shuffle_xor(h2, ushort(stride));
                float p3 = simd_shuffle_xor(h3, ushort(stride));
                if (lane & stride) {
                    h0 = p0 - h0;
                    h1 = p1 - h1;
                    h2 = p2 - h2;
                    h3 = p3 - h3;
                } else {
                    h0 = h0 + p0;
                    h1 = h1 + p1;
                    h2 = h2 + p2;
                    h3 = h3 + p3;
                }
            }
            return float4(h0, h1, h2, h3);
        }
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint column0 = block * HAD + lane * 4u;
        T mixed0 = T(0.0f);
        T mixed1 = T(0.0f);
        T mixed2 = T(0.0f);
        T mixed3 = T(0.0f);
        for (uint route = 0u; route < TOPK; ++route) {
            uint task = row * TOPK + route;
            uint expert = uint(expert_ids[task]);
            float4 output_had = hadamard_h128_quad(
                float4(
                    float(down_inner[(size_t)task * HIDDEN + column0]),
                    float(down_inner[(size_t)task * HIDDEN + column0 + 1u]),
                    float(down_inner[(size_t)task * HIDDEN + column0 + 2u]),
                    float(down_inner[(size_t)task * HIDDEN + column0 + 3u])
                ),
                lane
            );
            half rotated0 = half(output_had.x * HAD_SCALE);
            half rotated1 = half(output_had.y * HAD_SCALE);
            half rotated2 = half(output_had.z * HAD_SCALE);
            half rotated3 = half(output_had.w * HAD_SCALE);
            half projected_half0 = half(
                rotated0 * down_svh[(size_t)expert * HIDDEN + column0]
            );
            half projected_half1 = half(
                rotated1 * down_svh[(size_t)expert * HIDDEN + column0 + 1u]
            );
            half projected_half2 = half(
                rotated2 * down_svh[(size_t)expert * HIDDEN + column0 + 2u]
            );
            half projected_half3 = half(
                rotated3 * down_svh[(size_t)expert * HIDDEN + column0 + 3u]
            );
            T projected0 = T(projected_half0);
            T projected1 = T(projected_half1);
            T projected2 = T(projected_half2);
            T projected3 = T(projected_half3);
            T weight = T(route_weights[task]);
            T product0 = T(projected0 * weight);
            T product1 = T(projected1 * weight);
            T product2 = T(projected2 * weight);
            T product3 = T(projected3 * weight);
            mixed0 = T(product0 + mixed0);
            mixed1 = T(product1 + mixed1);
            mixed2 = T(product2 + mixed2);
            mixed3 = T(product3 + mixed3);
        }
        output[(size_t)row * HIDDEN + column0] = T(
            mixed0 + shared[(size_t)row * HIDDEN + column0]
        );
        output[(size_t)row * HIDDEN + column0 + 1u] = T(
            mixed1 + shared[(size_t)row * HIDDEN + column0 + 1u]
        );
        output[(size_t)row * HIDDEN + column0 + 2u] = T(
            mixed2 + shared[(size_t)row * HIDDEN + column0 + 2u]
        );
        output[(size_t)row * HIDDEN + column0 + 3u] = T(
            mixed3 + shared[(size_t)row * HIDDEN + column0 + 3u]
        );
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_exl3_m6_direct_final_tail_h4096_t6_v2",
        input_names=[
            "down_inner",
            "down_svh",
            "expert_ids",
            "route_weights",
            "shared",
        ],
        output_names=["output"],
        header=header,
        source=source,
    )


def exl3_mcg_qmv(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
) -> mx.array:
    """Run one pinned-format MCG/K3 projection with fused sign/H128 stages."""

    if x.ndim != 2 or x.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV requires a two-dimensional float16 input")
    if trellis.ndim != 3 or trellis.dtype != mx.int16:
        raise ValueError("EXL3 MCG QMV requires an int16 [K/16,N/16,48] trellis")
    size_k = int(trellis.shape[0]) * 16
    size_n = int(trellis.shape[1]) * 16
    if int(trellis.shape[2]) != EXL3_PACKED_WORDS:
        raise ValueError("EXL3 MCG QMV requires exactly 48 packed words per tile")
    if int(x.shape[1]) != size_k:
        raise ValueError("EXL3 MCG QMV input width does not match its trellis")
    if tuple(suh.shape) != (size_k,) or suh.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV suh does not match its input width")
    if tuple(svh.shape) != (size_n,) or svh.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV svh does not match its output width")
    rows = int(x.shape[0])
    kernel = _mcg_qmv_kernel(size_k, size_n)
    (output,) = kernel(
        inputs=[
            mx.contiguous(x),
            mx.contiguous(trellis),
            mx.contiguous(suh),
            mx.contiguous(svh),
        ],
        grid=(128, size_n // 128, rows),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, size_n)],
        output_dtypes=[mx.float16],
    )
    return output


def exl3_mcg_grouped_qmv(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    """Project router-selected rows through one packed K216-style expert bank."""

    if trellis.ndim != 4 or trellis.dtype != mx.int16:
        raise ValueError(
            "grouped EXL3 QMV requires int16 [E,K/16,N/16,48] trellis"
        )
    experts = int(trellis.shape[0])
    size_k = int(trellis.shape[1]) * 16
    size_n = int(trellis.shape[2]) * 16
    if int(trellis.shape[3]) != EXL3_PACKED_WORDS:
        raise ValueError("grouped EXL3 QMV requires 48 packed words per tile")
    if tuple(suh.shape) != (experts, size_k) or suh.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV suh bank has the wrong geometry")
    if tuple(svh.shape) != (experts, size_n) or svh.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV svh bank has the wrong geometry")
    if expert_ids.ndim != 2 or expert_ids.dtype not in (mx.int32, mx.uint32):
        raise ValueError("grouped EXL3 QMV expert IDs must be a 2-D int32 array")
    rows, topk = (int(value) for value in expert_ids.shape)
    routed_input = x.ndim == 3
    if routed_input:
        if tuple(x.shape[:2]) != (rows, topk) or int(x.shape[2]) != size_k:
            raise ValueError("routed EXL3 QMV input does not match router geometry")
        x_rows = mx.contiguous(x.reshape(rows * topk, size_k))
    else:
        if x.ndim != 2 or tuple(x.shape) != (rows, size_k):
            raise ValueError("grouped EXL3 QMV input does not match router rows")
        x_rows = mx.contiguous(x)
    if x_rows.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV requires float16 activations")

    tasks = rows * topk
    kernel = _mcg_qmv_kernel(size_k, size_n, experts, topk, routed_input)
    (output,) = kernel(
        inputs=[
            x_rows,
            mx.contiguous(trellis),
            mx.contiguous(suh),
            mx.contiguous(svh),
            mx.contiguous(expert_ids.reshape(tasks)),
        ],
        grid=(128, size_n // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    return output.reshape(rows, topk, size_n)


@lru_cache(maxsize=None)
def _route_hadamard_kernel(
    size_k: int,
    experts: int,
    topk: int,
    routed_input: bool,
):
    source_row = "task" if routed_input else "task / TOPK"
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint TOPK = {topk};
        constant constexpr uint HAD = 128;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = f"""
        uint lane = thread_position_in_threadgroup.x;
        uint k_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        size_t source_row = {source_row};
        uint k = k_block * HAD + lane;
        threadgroup float values[HAD];
        half scaled = half(
            x[source_row * SIZE_K + k]
            * suh[(size_t)expert * SIZE_K + k]
        );
        values[lane] = float(scaled);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {{
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        y[(size_t)task * SIZE_K + k] = half(values[lane] * HAD_SCALE);
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_route_h128_k{size_k}_e{experts}"
            f"_t{topk}_r{int(routed_input)}_v1"
        ),
        input_names=["x", "suh", "expert_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


def _route_hadamard(
    x: mx.array,
    suh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    rows, topk = (int(value) for value in expert_ids.shape)
    tasks = rows * topk
    size_k = int(suh.shape[1])
    routed_input = x.ndim == 3
    kernel = _route_hadamard_kernel(
        size_k,
        int(suh.shape[0]),
        topk,
        routed_input,
    )
    (output,) = kernel(
        inputs=[
            mx.contiguous(x.astype(mx.float16)),
            mx.contiguous(suh),
            mx.contiguous(expert_ids.reshape(tasks)),
        ],
        grid=(128, size_k // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_k)],
        output_dtypes=[mx.float16],
    )
    return output


@lru_cache(maxsize=None)
def _mma_route_pack_kernel(experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {experts};
        constant constexpr uint BM = 8;
    """
    source = """
        uint lane = thread_position_in_threadgroup.x;
        threadgroup atomic_uint total;
        if (lane == 0u) atomic_store_explicit(&total, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < EXPERTS) {
            uint count = uint(row_count[lane]);
            uint blocks = (count + BM - 1u) / BM;
            uint destination = atomic_fetch_add_explicit(
                &total, blocks, memory_order_relaxed
            );
            uint start = uint(row_start[lane]);
            for (uint block = 0u; block < blocks; ++block) {
                uint offset = block * BM;
                block_expert[destination + block] = lane;
                block_row[destination + block] = start + offset;
                block_size[destination + block] = min(BM, count - offset);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) packed_count[0] = atomic_load_explicit(
            &total, memory_order_relaxed
        );
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_mma_route_pack_e{experts}_v1",
        input_names=["row_start", "row_count"],
        output_names=["block_expert", "block_row", "block_size", "packed_count"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_route_pack_kernel(experts: int, topk: int, block_m: int):
    """Source-owned histogram/prefix/route pack with no generic sort."""
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {experts}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint BM = {block_m}u;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        threadgroup atomic_uint counts[EXPERTS];
        threadgroup atomic_uint cursors[EXPERTS];
        threadgroup uint offsets[EXPERTS + 1u];
        threadgroup uint total_blocks;

        for (uint expert = tid; expert < EXPERTS; expert += 256u) {
            atomic_store_explicit(&counts[expert], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint task = tid; task < uint(n_tasks); task += 256u) {
            uint expert = uint(expert_ids[task]);
            atomic_fetch_add_explicit(
                &counts[expert], 1u, memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint offset = 0u;
            uint block = 0u;
            for (uint expert = 0u; expert < EXPERTS; ++expert) {
                offsets[expert] = offset;
                uint count = atomic_load_explicit(
                    &counts[expert], memory_order_relaxed
                );
                atomic_store_explicit(
                    &cursors[expert], offset, memory_order_relaxed
                );
                for (uint first = 0u; first < count; first += BM) {
                    block_expert[block] = expert;
                    block_row[block] = offset + first;
                    block_size[block] = min(BM, count - first);
                    block += 1u;
                }
                offset += count;
            }
            offsets[EXPERTS] = offset;
            total_blocks = block;
            packed_count[0] = block;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint task = tid; task < uint(n_tasks); task += 256u) {
            uint expert = uint(expert_ids[task]);
            uint position = atomic_fetch_add_explicit(
                &cursors[expert], 1u, memory_order_relaxed
            );
            packed_tasks[position] = task;
            inverse[task] = position;
            sorted_ids[position] = expert;
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_route_e{experts}_t{topk}"
            f"_bm{block_m}_v1"
        ),
        input_names=["expert_ids", "n_tasks"],
        output_names=[
            "packed_tasks",
            "inverse",
            "sorted_ids",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        header=header,
        source=source,
    )


def _pack_trellis_routes(
    expert_ids: mx.array,
    *,
    experts: int,
    topk: int,
    block_m: int,
    kernel,
):
    tasks = int(expert_ids.size)
    route_blocks = _trellis_route_block_capacity(tasks, experts, block_m)
    return kernel(
        inputs=[
            mx.contiguous(expert_ids.reshape(tasks).astype(mx.uint32)),
            tasks,
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tasks,)] * 3 + [(route_blocks,)] * 3 + [(1,)],
        output_dtypes=[mx.uint32] * 7,
    )


def _trellis_route_block_capacity(tasks: int, experts: int, block_m: int) -> int:
    """Exact maximum populated blocks for compact, expert-grouped routes.

    Give one route to each active expert first: each creates one block.  Once
    all experts are active, every additional block requires another ``block_m``
    routes assigned to some expert.  This shape-only bound is proven before
    Metal execution and therefore needs neither ``packed_count`` readback nor
    a task-count launch padded with inactive threadgroups.
    """
    active_experts = min(int(tasks), int(experts))
    extra_blocks = max(int(tasks) - int(experts), 0) // int(block_m)
    return max(active_experts + extra_blocks, 1)


@lru_cache(maxsize=None)
def _packed_route_hadamard_kernel(size_k: int, experts: int, topk: int):
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint k_block = threadgroup_position_in_grid.y;
        uint sorted_task = threadgroup_position_in_grid.z;
        uint original_task = uint(packed_tasks[sorted_task]);
        uint source_row = original_task / TOPK;
        uint expert = uint(sorted_ids[sorted_task]);
        uint k = k_block * HAD + lane;
        threadgroup float values[HAD];
        half scaled = half(
            float(x[size_t(source_row) * SIZE_K + k])
            * float(suh[size_t(expert) * SIZE_K + k])
        );
        values[lane] = float(scaled);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        y[size_t(sorted_task) * SIZE_K + k] = half(
            values[lane] * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_packed_h128_k{size_k}_e{experts}"
            f"_t{topk}_v1"
        ),
        input_names=["x", "suh", "packed_tasks", "sorted_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _mcg_trellis_mma_kernel(
    size_k: int,
    size_n: int,
    experts: int,
    block_m: int,
):
    if block_m not in (8, 64):
        raise ValueError("Mia Trellis block_m must be 8 or 64")
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    simdgroups = block_m // 8 * 2
    threads = simdgroups * 32
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k}u;
        constant constexpr uint SIZE_N = {size_n}u;
        constant constexpr uint NTILES_N = {size_n // 16}u;
        constant constexpr uint BM = {block_m}u;
        constant constexpr uint BN = 32u;
        constant constexpr uint BK = 32u;
        constant constexpr uint THREADS = {threads}u;
        constant constexpr uint TILE_WORDS = 48u;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline half decode_mcg_device(
            device const ushort* packed,
            uint tensor_core_offset
        ) {{
            device const uint* words = reinterpret_cast<device const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(bits);
            return pair.x + pair.y;
        }}
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint packed_block = threadgroup_position_in_grid.z;
        if (packed_block >= packed_count[0]) return;
        uint n0 = threadgroup_position_in_grid.y * BN;
        uint expert = block_expert[packed_block];
        uint first_row = block_row[packed_block];
        uint active_rows = block_size[packed_block];
        uint sg_m = sg / 2u;
        uint sg_n = (sg & 1u) * 16u;

        threadgroup half A_tile[BM * BK];
        threadgroup half B_tile[BK * BN];
        threadgroup half C_tile[BM * BN];
        simdgroup_matrix<half, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0u; k0 < SIZE_K; k0 += BK) {
            for (uint index = tid; index < BM * BK; index += THREADS) {
                uint row = index / BK;
                uint local_k = index - row * BK;
                A_tile[index] = row < active_rows
                    ? x[size_t(first_row + row) * SIZE_K + k0 + local_k]
                    : half(0.0h);
            }
            for (uint index = tid; index < BK * BN; index += THREADS) {
                uint local_k = index / BN;
                uint local_n = index - local_k * BN;
                uint k = k0 + local_k;
                uint n = n0 + local_n;
                uint tile_k = k / 16u;
                uint tile_n = n / 16u;
                uint row_major = (k & 15u) * 16u + (n & 15u);
                uint tensor_core = uint(TC_INV[row_major]);
                size_t tile_index =
                    ((size_t)expert * (SIZE_K / 16u) * NTILES_N
                     + size_t(tile_k) * NTILES_N + tile_n) * TILE_WORDS;
                B_tile[index] = decode_mcg_device(
                    reinterpret_cast<const device ushort*>(trellis) + tile_index,
                    tensor_core
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < BK; ks += 8u) {
                simdgroup_load(a, A_tile + sg_m * 8u * BK + ks, BK);
                simdgroup_load(b_left, B_tile + ks * BN + sg_n, BN);
                simdgroup_load(b_right, B_tile + ks * BN + sg_n + 8u, BN);
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<half, 8, 8> out_left, out_right;
        out_left.thread_elements()[0] = half(c_left.thread_elements()[0]);
        out_left.thread_elements()[1] = half(c_left.thread_elements()[1]);
        out_right.thread_elements()[0] = half(c_right.thread_elements()[0]);
        out_right.thread_elements()[1] = half(c_right.thread_elements()[1]);
        simdgroup_store(out_left, C_tile + sg_m * 8u * BN + sg_n, BN);
        simdgroup_store(out_right, C_tile + sg_m * 8u * BN + sg_n + 8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < active_rows * BN; index += THREADS) {
            uint row = index / BN;
            uint local_n = index - row * BN;
            y[size_t(first_row + row) * SIZE_N + n0 + local_n] = C_tile[index];
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_mma_k{size_k}_n{size_n}"
            f"_e{experts}_bm{block_m}_v1"
        ),
        input_names=[
            "x",
            "trellis",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_activation_down_hadamard_kernel(
    intermediate_size: int,
    experts: int,
    limit: float,
):
    limit_literal = format(float(limit), ".9g")
    if "." not in limit_literal and "e" not in limit_literal.lower():
        limit_literal += ".0"
    header = f"""
        using namespace metal;
        constant constexpr uint INTERMEDIATE = {intermediate_size}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
        constant constexpr float LIMIT = {limit_literal}f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(sorted_ids[task]);
        uint column = block * HAD + lane;
        threadgroup float gate_values[HAD];
        threadgroup float up_values[HAD];

        gate_values[lane] = float(gate_inner[size_t(task) * INTERMEDIATE + column]);
        up_values[lane] = float(up_inner[size_t(task) * INTERMEDIATE + column]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float gate_own = gate_values[lane];
            float gate_peer = gate_values[lane ^ stride];
            float up_own = up_values[lane];
            float up_peer = up_values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gate_values[lane] = (lane & stride)
                ? gate_peer - gate_own
                : gate_own + gate_peer;
            up_values[lane] = (lane & stride)
                ? up_peer - up_own
                : up_own + up_peer;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        half gate_half = half(
            gate_values[lane] * HAD_SCALE
            * float(gate_svh[size_t(expert) * INTERMEDIATE + column])
        );
        half up_half = half(
            up_values[lane] * HAD_SCALE
            * float(up_svh[size_t(expert) * INTERMEDIATE + column])
        );
        float gate = float(gate_half);
        float up = float(up_half);
        if (LIMIT > 0.0f) {
            gate = min(gate, LIMIT);
            up = clamp(up, -LIMIT, LIMIT);
        }
        half activated = half((gate / (1.0f + exp(-gate))) * up);
        gate_values[lane] = float(half(
            float(activated)
            * float(down_suh[size_t(expert) * INTERMEDIATE + column])
        ));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = gate_values[lane];
            float peer = gate_values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gate_values[lane] = (lane & stride) ? peer - own : own + peer;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        down_h[size_t(task) * INTERMEDIATE + column] = half(
            gate_values[lane] * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_swiglu_down_h_i{intermediate_size}"
            f"_e{experts}_l{int(round(limit * 1000.0))}_v1"
        ),
        input_names=[
            "gate_inner",
            "up_inner",
            "gate_svh",
            "up_svh",
            "down_suh",
            "sorted_ids",
        ],
        output_names=["down_h"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_final_reduce_kernel(
    hidden_size: int,
    experts: int,
    topk: int,
):
    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden_size}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint column = block * HAD + lane;
        threadgroup float values[HAD];
        float routed_sum = 0.0f;
        for (uint route = 0u; route < TOPK; ++route) {
            uint original_task = row * TOPK + route;
            uint sorted_task = uint(inverse[original_task]);
            uint expert = uint(expert_ids[original_task]);
            values[lane] = float(
                down_inner[size_t(sorted_task) * HIDDEN + column]
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1u; stride < HAD; stride <<= 1u) {
                float own = values[lane];
                float peer = values[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                values[lane] = (lane & stride) ? peer - own : own + peer;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            half projected = half(
                values[lane] * HAD_SCALE
                * float(down_svh[size_t(expert) * HIDDEN + column])
            );
            routed_sum += float(projected)
                * float(route_weights[original_task]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        output[size_t(row) * HIDDEN + column] = T(
            routed_sum + float(shared[size_t(row) * HIDDEN + column])
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_final_reduce_h{hidden_size}"
            f"_e{experts}_t{topk}_v1"
        ),
        input_names=[
            "down_inner",
            "down_svh",
            "inverse",
            "expert_ids",
            "route_weights",
            "shared",
        ],
        output_names=["output"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _mcg_grouped_mma_kernel(size_k: int, size_n: int, experts: int):
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint BM = 8;
        constant constexpr uint BN = 32;
        constant constexpr uint BK = 32;
        constant constexpr uint TILE_WORDS = 48;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline half decode_mcg_device(
            device const ushort* packed,
            uint tensor_core_offset
        ) {{
            device const uint* words = reinterpret_cast<device const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(bits);
            return pair.x + pair.y;
        }}
    """
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint sg = tid / 32u;
        uint packed_block = threadgroup_position_in_grid.z;
        if (packed_block >= packed_count[0]) return;
        uint n0 = threadgroup_position_in_grid.y * BN;
        uint expert = block_expert[packed_block];
        uint first_row = block_row[packed_block];
        uint active_rows = block_size[packed_block];

        threadgroup half A_tile[BM * BK];
        threadgroup half B_tile[BK * BN];
        threadgroup half C_tile[BM * BN];
        simdgroup_matrix<half, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);
        uint sg_n = sg * 16u;

        for (uint k0 = 0u; k0 < SIZE_K; k0 += BK) {
            for (uint index = tid; index < BM * BK; index += 64u) {
                uint row = index / BK;
                uint local_k = index % BK;
                A_tile[index] = row < active_rows
                    ? x[(size_t)(first_row + row) * SIZE_K + k0 + local_k]
                    : half(0.0h);
            }
            for (uint index = tid; index < BK * BN; index += 64u) {
                uint local_k = index / BN;
                uint local_n = index % BN;
                uint k = k0 + local_k;
                uint n = n0 + local_n;
                uint tile_k = k / 16u;
                uint tile_n = n / 16u;
                uint row_major = (k & 15u) * 16u + (n & 15u);
                uint tensor_core = uint(TC_INV[row_major]);
                size_t tile_index =
                    ((size_t)expert * (SIZE_K / 16u) * NTILES_N
                     + (size_t)tile_k * NTILES_N + tile_n) * TILE_WORDS;
                B_tile[index] = decode_mcg_device(
                    reinterpret_cast<const device ushort*>(trellis) + tile_index,
                    tensor_core
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < BK; ks += 8u) {
                simdgroup_load(a, A_tile + ks, BK);
                simdgroup_load(b_left, B_tile + ks * BN + sg_n, BN);
                simdgroup_load(b_right, B_tile + ks * BN + sg_n + 8u, BN);
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<half, 8, 8> out_left, out_right;
        out_left.thread_elements()[0] = half(c_left.thread_elements()[0]);
        out_left.thread_elements()[1] = half(c_left.thread_elements()[1]);
        out_right.thread_elements()[0] = half(c_right.thread_elements()[0]);
        out_right.thread_elements()[1] = half(c_right.thread_elements()[1]);
        simdgroup_store(out_left, C_tile + sg_n, BN);
        simdgroup_store(out_right, C_tile + sg_n + 8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < active_rows * BN; index += 64u) {
            uint row = index / BN;
            uint local_n = index % BN;
            y[(size_t)(first_row + row) * SIZE_N + n0 + local_n] = C_tile[index];
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_mcg_mma_k{size_k}_n{size_n}_e{experts}_v1",
        input_names=[
            "x",
            "trellis",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _route_output_hadamard_kernel(size_n: int, experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint HAD = 128;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = """
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        uint n = n_block * HAD + lane;
        threadgroup float values[HAD];
        values[lane] = float(x[(size_t)task * SIZE_N + n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        half rotated = half(values[lane] * HAD_SCALE);
        y[(size_t)task * SIZE_N + n] = half(
            rotated * svh[(size_t)expert * SIZE_N + n]
        );
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_route_output_h128_n{size_n}_e{experts}_v1",
        input_names=["x", "svh", "expert_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


def _mma_route_arena(expert_ids: mx.array, experts: int):
    flat_ids = expert_ids.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat_ids)
    inverse = mx.argsort(order)
    sorted_ids = mx.contiguous(flat_ids[order])
    expert_range = mx.arange(experts, dtype=mx.uint32)
    starts = mx.searchsorted(sorted_ids, expert_range, side="left").astype(mx.int32)
    ends = mx.searchsorted(sorted_ids, expert_range, side="right").astype(mx.int32)
    tasks = int(flat_ids.shape[0])
    blocks = _mma_route_pack_kernel(experts)(
        inputs=[mx.contiguous(starts), mx.contiguous(ends - starts)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tasks,), (tasks,), (tasks,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )
    return order, inverse, sorted_ids, blocks


def _mma_project_sorted(
    transformed: mx.array,
    trellis: mx.array,
    svh: mx.array,
    sorted_ids: mx.array,
    route_blocks,
) -> mx.array:
    tasks, size_k = (int(value) for value in transformed.shape)
    experts = int(trellis.shape[0])
    size_n = int(trellis.shape[2]) * 16
    block_expert, block_row, block_size, packed_count = route_blocks
    (inner,) = _mcg_grouped_mma_kernel(size_k, size_n, experts)(
        inputs=[
            transformed,
            mx.contiguous(trellis),
            block_expert,
            block_row,
            block_size,
            packed_count,
        ],
        grid=(64, size_n // 32, tasks),
        threadgroup=(64, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    (output,) = _route_output_hadamard_kernel(size_n, experts)(
        inputs=[inner, mx.contiguous(svh), sorted_ids],
        grid=(128, size_n // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    return output


def exl3_mcg_grouped_mma(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    """Pinned EXL3 M-tiled path using Metal simdgroup matrix accumulation."""

    rows, topk = (int(value) for value in expert_ids.shape)
    experts = int(trellis.shape[0])
    order, inverse, sorted_ids, route_blocks = _mma_route_arena(
        expert_ids, experts
    )
    transformed = _route_hadamard(x, suh, expert_ids)[order]
    sorted_output = _mma_project_sorted(
        transformed,
        trellis,
        svh,
        sorted_ids,
        route_blocks,
    )
    return sorted_output[inverse].reshape(rows, topk, int(svh.shape[1]))


class EXL3LinearBank(nn.Module):
    """One construction-qualified bank of MCG/K3 expert projections."""

    def __init__(
        self,
        experts: int,
        input_dims: int,
        output_dims: int,
        topk: int,
        *,
        routed_input: bool,
    ) -> None:
        super().__init__()
        self._qmv_output_tile = 256
        if input_dims % 128 or output_dims % self._qmv_output_tile:
            raise ValueError(
                "EXL3 expert input must tile H128 and output must tile QMV BN256"
            )
        self.experts = int(experts)
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.topk = int(topk)
        self.routed_input = bool(routed_input)
        self.trellis = mx.zeros(
            (self.experts, self.input_dims // 16, self.output_dims // 16, 48),
            dtype=mx.int16,
        )
        self.suh = mx.zeros((self.experts, self.input_dims), dtype=mx.float16)
        self.svh = mx.zeros((self.experts, self.output_dims), dtype=mx.float16)
        self._kernel = _mcg_qmv_kernel(
            self.input_dims,
            self.output_dims,
            self.experts,
            self.topk,
            self.routed_input,
            self._qmv_output_tile,
        )
        self._had_kernel = _route_hadamard_kernel(
            self.input_dims,
            self.experts,
            self.topk,
            self.routed_input,
        )
        self._mma_kernel = _mcg_grouped_mma_kernel(
            self.input_dims,
            self.output_dims,
            self.experts,
        )
        self._output_had_kernel = _route_output_hadamard_kernel(
            self.output_dims,
            self.experts,
        )

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        rows = int(expert_ids.shape[0])
        tasks = rows * self.topk
        x_rows = (
            mx.contiguous(x.reshape(tasks, self.input_dims))
            if self.routed_input
            else mx.contiguous(x)
        )
        (output,) = self._kernel(
            inputs=[
                x_rows,
                self.trellis,
                self.suh,
                self.svh,
                mx.contiguous(expert_ids.reshape(tasks)),
            ],
            grid=(128, self.output_dims // self._qmv_output_tile, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        return output.reshape(rows, self.topk, self.output_dims)

    def transform_routes(self, x: mx.array, flat_ids: mx.array) -> mx.array:
        tasks = int(flat_ids.shape[0])
        (output,) = self._had_kernel(
            inputs=[mx.contiguous(x), self.suh, mx.contiguous(flat_ids)],
            grid=(128, self.input_dims // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.input_dims)],
            output_dtypes=[mx.float16],
        )
        return output

    def mma_sorted(self, x: mx.array, sorted_ids: mx.array, route_blocks) -> mx.array:
        tasks = int(x.shape[0])
        block_expert, block_row, block_size, packed_count = route_blocks
        (inner,) = self._mma_kernel(
            inputs=[
                mx.contiguous(x),
                self.trellis,
                block_expert,
                block_row,
                block_size,
                packed_count,
            ],
            grid=(64, self.output_dims // 32, tasks),
            threadgroup=(64, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        (output,) = self._output_had_kernel(
            inputs=[inner, self.svh, sorted_ids],
            grid=(128, self.output_dims // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        return output

class _InstalledTrellisPlan(NamedTuple):
    block_m: int
    route_pack: Any
    hidden_to_intermediate: Any
    intermediate_to_hidden: Any


class _InstalledM6QuadQMVPlan(NamedTuple):
    geometry: tuple[int, int, int, int, float, int, int]
    descriptor_sha256: str
    stage_vector_bytes: int
    stage_vectors_per_k_tile: int
    hidden_to_intermediate: Any
    intermediate_to_hidden: Any
    dual_fc1_input: Any
    dual_fc1_inner: Any
    activation_down: Any
    down_inner: Any
    direct_final_tail: Any


class EXL3SwitchGLU(nn.Module):
    """DeepSeek routed SwiGLU over the exact Mia K216 EXL3 expert banks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        experts: int,
        topk: int,
        *,
        limit: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.experts = int(experts)
        self.topk = int(topk)
        self.limit = float(limit or 0.0)
        self._route_pack = _mma_route_pack_kernel(self.experts)
        self.gate_proj = EXL3LinearBank(
            experts,
            hidden_size,
            intermediate_size,
            topk,
            routed_input=False,
        )
        self.up_proj = EXL3LinearBank(
            experts,
            hidden_size,
            intermediate_size,
            topk,
            routed_input=False,
        )
        self.down_proj = EXL3LinearBank(
            experts,
            intermediate_size,
            hidden_size,
            topk,
            routed_input=True,
        )
        self._trellis_installed = False
        self._trellis_plans = ()
        self._trellis_input_hadamard = None
        self._trellis_activation_down = None
        self._trellis_final_reduce = None
        self._m6_quad_qmv_plan = None

    def install_trellis_runtime(self, *, max_tokens: int) -> None:
        """Install the pinned decode/prefill plans before request execution."""
        if int(max_tokens) < 1:
            raise ValueError("EXL3 Trellis max_tokens must be positive")
        self._trellis_max_tokens = int(max_tokens)
        plans = []
        for block_m in (8, 64):
            plans.append(
                _InstalledTrellisPlan(
                    block_m=block_m,
                    route_pack=_trellis_route_pack_kernel(
                        self.experts, self.topk, block_m
                    ),
                    hidden_to_intermediate=_mcg_trellis_mma_kernel(
                        self.hidden_size,
                        self.gate_proj.output_dims,
                        self.experts,
                        block_m,
                    ),
                    intermediate_to_hidden=_mcg_trellis_mma_kernel(
                        self.down_proj.input_dims,
                        self.hidden_size,
                        self.experts,
                        block_m,
                    ),
                )
            )
        self._trellis_plans = tuple(plans)
        self._trellis_input_hadamard = _packed_route_hadamard_kernel(
            self.hidden_size, self.experts, self.topk
        )
        self._trellis_activation_down = _trellis_activation_down_hadamard_kernel(
            self.gate_proj.output_dims, self.experts, self.limit
        )
        self._trellis_final_reduce = _trellis_final_reduce_kernel(
            self.hidden_size, self.experts, self.topk
        )
        self._trellis_installed = True

    def install_m6_quad_qmv_runtime(self) -> None:
        """Bind the exact Mia M6 four-row decoder before request execution."""

        geometry = (
            self.hidden_size,
            self.gate_proj.output_dims,
            self.experts,
            self.topk,
            self.limit,
            256,
            self.topk * 6,
        )
        if geometry != (4096, 2048, 216, 6, 10.0, 256, 36):
            raise ValueError(f"Mia quad QMV geometry changed: {geometry!r}")
        tile_bytes = EXL3_PACKED_WORDS * 2
        stage_vectors_per_k_tile = (
            EXL3_TILE * tile_bytes // EXL3_M6_STAGE_VECTOR_BYTES
        )
        if (
            tile_bytes % EXL3_M6_STAGE_VECTOR_BYTES != 0
            or stage_vectors_per_k_tile != EXL3_M6_STAGE_VECTORS_PER_K_TILE
        ):
            raise ValueError("Mia quad QMV uint4 staging alignment changed")
        bank_contract = (
            (
                self.gate_proj,
                (216, 256, 128, 48),
                (216, 4096),
                (216, 2048),
            ),
            (
                self.up_proj,
                (216, 256, 128, 48),
                (216, 4096),
                (216, 2048),
            ),
            (
                self.down_proj,
                (216, 128, 256, 48),
                (216, 2048),
                (216, 4096),
            ),
        )
        for bank, trellis_shape, suh_shape, svh_shape in bank_contract:
            observed = (
                tuple(bank.trellis.shape),
                tuple(bank.suh.shape),
                tuple(bank.svh.shape),
                bank.trellis.dtype,
                bank.suh.dtype,
                bank.svh.dtype,
            )
            required = (
                trellis_shape,
                suh_shape,
                svh_shape,
                mx.int16,
                mx.float16,
                mx.float16,
            )
            if observed != required:
                raise ValueError(
                    f"Mia quad QMV bank storage changed: {observed!r}"
                )
        descriptors = _mcg_quad_descriptor_plan()
        self._m6_quad_qmv_plan = _InstalledM6QuadQMVPlan(
            geometry=geometry,
            descriptor_sha256=descriptors.sha256,
            stage_vector_bytes=EXL3_M6_STAGE_VECTOR_BYTES,
            stage_vectors_per_k_tile=EXL3_M6_STAGE_VECTORS_PER_K_TILE,
            hidden_to_intermediate=_m6_quad_qmv_kernel(4096, 2048, False),
            intermediate_to_hidden=_m6_quad_qmv_kernel(2048, 4096, True),
            dual_fc1_input=_m6_dual_fc1_input_kernel(),
            dual_fc1_inner=_m6_dual_fc1_inner_kernel(),
            activation_down=_m6_clamp10_activation_down_kernel(),
            down_inner=_m6_down_inner_kernel(),
            direct_final_tail=_m6_direct_final_tail_kernel(),
        )
    def _trellis_mma(
        self,
        bank: EXL3LinearBank,
        transformed: mx.array,
        route_blocks,
        *,
        block_m: int,
        kernel,
    ) -> mx.array:
        tasks = int(transformed.shape[0])
        block_expert, block_row, block_size, packed_count = route_blocks
        route_blocks_capacity = int(block_expert.shape[0])
        threads = int(block_m) * 8
        return kernel(
            inputs=[
                mx.contiguous(transformed),
                bank.trellis,
                block_expert,
                block_row,
                block_size,
                packed_count,
            ],
            grid=(threads, bank.output_dims // 32, route_blocks_capacity),
            threadgroup=(threads, 1, 1),
            output_shapes=[(tasks, bank.output_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def fused(
        self,
        x: mx.array,
        expert_ids: mx.array,
        route_weights: mx.array,
        shared: mx.array,
    ) -> mx.array:
        """Run the installed W4A16 Trellis MoE and final weighted reduction."""
        original_dtype = x.dtype
        rows = int(expert_ids.shape[0])
        tasks = rows * self.topk
        plan = self._trellis_plans[0 if rows <= 127 else 1]
        block_m = plan.block_m
        (
            packed_tasks,
            inverse,
            sorted_ids,
            block_expert,
            block_row,
            block_size,
            packed_count,
        ) = _pack_trellis_routes(
            expert_ids,
            experts=self.experts,
            topk=self.topk,
            block_m=block_m,
            kernel=plan.route_pack,
        )
        route_blocks = (block_expert, block_row, block_size, packed_count)
        x_half = mx.contiguous(x.astype(mx.float16))
        transform = self._trellis_input_hadamard
        gate_h = transform(
            inputs=[x_half, self.gate_proj.suh, packed_tasks, sorted_ids],
            grid=(128, self.hidden_size // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.hidden_size)],
            output_dtypes=[mx.float16],
        )[0]
        up_h = transform(
            inputs=[x_half, self.up_proj.suh, packed_tasks, sorted_ids],
            grid=(128, self.hidden_size // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.hidden_size)],
            output_dtypes=[mx.float16],
        )[0]
        gate_inner = self._trellis_mma(
            self.gate_proj,
            gate_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.hidden_to_intermediate,
        )
        up_inner = self._trellis_mma(
            self.up_proj,
            up_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.hidden_to_intermediate,
        )
        intermediate = self.gate_proj.output_dims
        down_h = self._trellis_activation_down(
            inputs=[
                gate_inner,
                up_inner,
                self.gate_proj.svh,
                self.up_proj.svh,
                self.down_proj.suh,
                sorted_ids,
            ],
            grid=(128, intermediate // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, intermediate)],
            output_dtypes=[mx.float16],
        )[0]
        down_inner = self._trellis_mma(
            self.down_proj,
            down_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.intermediate_to_hidden,
        )
        return self._trellis_final_reduce(
            inputs=[
                down_inner,
                self.down_proj.svh,
                inverse,
                mx.contiguous(expert_ids.reshape(tasks).astype(mx.uint32)),
                mx.contiguous(route_weights.reshape(tasks)),
                mx.contiguous(shared),
            ],
            template=[("T", original_dtype)],
            grid=(128, self.hidden_size // 128, rows),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.hidden_size)],
            output_dtypes=[original_dtype],
        )[0]

    def direct_qmv(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        """Run the authentic direct-QMV MoE arithmetic unconditionally."""

        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        gate = self.gate_proj(x_half, expert_ids)
        up = self.up_proj(x_half, expert_ids)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        activated = (nn.silu(gate) * up).astype(mx.float16)
        return self.down_proj(activated, expert_ids).astype(original_dtype)

    @staticmethod
    def _m6_quad_project(
        bank: EXL3LinearBank,
        x_rows: mx.array,
        flat_ids: mx.array,
        kernel,
    ) -> mx.array:
        tasks = 36
        return kernel(
            inputs=[
                mx.contiguous(x_rows),
                bank.trellis,
                bank.suh,
                bank.svh,
                flat_ids,
            ],
            grid=(128, bank.output_dims // 256, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, bank.output_dims)],
            output_dtypes=[mx.float16],
        )[0].reshape(6, 6, bank.output_dims)

    def direct_qmv_m6_quad(
        self,
        x: mx.array,
        expert_ids: mx.array,
    ) -> mx.array:
        """Run the construction-bound exact-M6 four-row decoder."""

        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        flat_ids = mx.contiguous(expert_ids.reshape(36).astype(mx.uint32))
        plan = self._m6_quad_qmv_plan
        gate = self._m6_quad_project(
            self.gate_proj,
            x_half,
            flat_ids,
            plan.hidden_to_intermediate,
        )
        up = self._m6_quad_project(
            self.up_proj,
            x_half,
            flat_ids,
            plan.hidden_to_intermediate,
        )
        gate = mx.minimum(gate, self.limit)
        up = mx.clip(up, -self.limit, self.limit)
        activated = (nn.silu(gate) * up).astype(mx.float16)
        return self._m6_quad_project(
            self.down_proj,
            activated.reshape(36, self.down_proj.input_dims),
            flat_ids,
            plan.intermediate_to_hidden,
        ).astype(original_dtype)

    def direct_m6_clamp10(
        self,
        x: mx.array,
        expert_ids: mx.array,
        route_weights: mx.array,
        shared: mx.array,
    ) -> mx.array:
        """Run the construction-bound five-stage exact-M6 clamp-10 route."""

        plan = self._m6_quad_qmv_plan
        ids = mx.contiguous(expert_ids)
        gate_h, up_h = plan.dual_fc1_input(
            inputs=[
                mx.contiguous(x),
                self.gate_proj.suh,
                self.up_proj.suh,
                ids,
            ],
            grid=(32, 32, 36),
            threadgroup=(32, 1, 1),
            output_shapes=[(36, 4096), (36, 4096)],
            output_dtypes=[mx.float16, mx.float16],
        )
        gate_inner, up_inner = plan.dual_fc1_inner(
            inputs=[
                gate_h,
                up_h,
                self.gate_proj.trellis,
                self.up_proj.trellis,
                ids,
            ],
            grid=(128, 8, 36),
            threadgroup=(128, 1, 1),
            output_shapes=[(36, 2048), (36, 2048)],
            output_dtypes=[mx.float16, mx.float16],
        )
        down_h = plan.activation_down(
            inputs=[
                gate_inner,
                up_inner,
                self.gate_proj.svh,
                self.up_proj.svh,
                self.down_proj.suh,
                ids,
            ],
            grid=(32, 16, 36),
            threadgroup=(32, 1, 1),
            output_shapes=[(36, 2048)],
            output_dtypes=[mx.float16],
        )[0]
        down_inner = plan.down_inner(
            inputs=[down_h, self.down_proj.trellis, ids],
            grid=(128, 16, 36),
            threadgroup=(128, 1, 1),
            output_shapes=[(36, 4096)],
            output_dtypes=[mx.float16],
        )[0]
        return plan.direct_final_tail(
            inputs=[
                down_inner,
                self.down_proj.svh,
                ids,
                mx.contiguous(route_weights),
                mx.contiguous(shared),
            ],
            template=[("T", x.dtype)],
            grid=(32, 32, 6),
            threadgroup=(32, 1, 1),
            output_shapes=[(6, 4096)],
            output_dtypes=[x.dtype],
        )[0]

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        rows = int(expert_ids.shape[0])
        # Preserve the explicit stock/oracle compatibility route.  Production
        # binds ``direct_qmv`` and ``fused`` separately at construction.
        if rows <= 6:
            return self.direct_qmv(x, expert_ids)

        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        tasks = int(expert_ids.size)
        flat_ids = expert_ids.reshape(tasks).astype(mx.uint32)
        order = mx.argsort(flat_ids)
        inverse = mx.argsort(order)
        sorted_ids = mx.contiguous(flat_ids[order])
        expert_range = mx.arange(self.experts, dtype=mx.uint32)
        starts = mx.searchsorted(sorted_ids, expert_range, side="left").astype(
            mx.int32
        )
        ends = mx.searchsorted(sorted_ids, expert_range, side="right").astype(
            mx.int32
        )
        route_blocks = self._route_pack(
            inputs=[mx.contiguous(starts), mx.contiguous(ends - starts)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(tasks,), (tasks,), (tasks,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
        )
        gate_h = self.gate_proj.transform_routes(x_half, flat_ids)[order]
        up_h = self.up_proj.transform_routes(x_half, flat_ids)[order]
        gate = self.gate_proj.mma_sorted(gate_h, sorted_ids, route_blocks)
        up = self.up_proj.mma_sorted(up_h, sorted_ids, route_blocks)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        activated = (nn.silu(gate) * up).astype(mx.float16)
        down_h = self.down_proj.transform_routes(activated, sorted_ids)
        down = self.down_proj.mma_sorted(down_h, sorted_ids, route_blocks)
        return down[inverse].reshape(
            int(expert_ids.shape[0]), self.topk, self.hidden_size
        ).astype(
            original_dtype
        )


def install_mia_m6_quad_qmv_routes(model) -> None:
    """Install every exact-M6 quad plan, then rebind the verified owner."""

    switches = []
    for layer_id, layer in enumerate(model.layers):
        switch = layer.ffn.switch_mlp
        direct_qmv = getattr(layer.ffn, "_mia_exl3_direct_qmv", None)
        if (
            getattr(direct_qmv, "__self__", None) is not switch
            or getattr(direct_qmv, "__func__", None)
            is not type(switch).direct_qmv
        ):
            raise ValueError(
                f"Mia target layer {layer_id} generic direct QMV owner changed"
            )
        switches.append((layer.ffn, switch))
    for _ffn, switch in switches:
        switch.install_m6_quad_qmv_runtime()
    for ffn, switch in switches:
        ffn._mia_exl3_direct_qmv = switch.direct_qmv_m6_quad
        ffn._mia_exl3_m6_fused = switch.direct_m6_clamp10


def _map_mia_target_name(name: str) -> str:
    top_level = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "hc_head_fn": "model.hc_head.fn",
        "hc_head_base": "model.hc_head.base",
        "hc_head_scale": "model.hc_head.scale",
    }
    if name in top_level:
        return top_level[name]
    if not name.startswith("layers."):
        return name
    mapped = "model." + name
    replacements = (
        (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_proj"),
        (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
        (".ffn.shared_experts.w3", ".ffn.shared_experts.up_proj"),
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
    )
    for source, target in replacements:
        mapped = mapped.replace(source, target)
    return mapped


def _expand_mia_fp8_block_scales(
    scales: mx.array,
    output_dims: int,
    input_dims: int,
) -> mx.array:
    expected = ((output_dims + 127) // 128, (input_dims + 127) // 128)
    if tuple(scales.shape) != expected or scales.dtype != mx.uint8:
        raise ValueError(
            f"Mia FP8 block scales {tuple(scales.shape)}/{scales.dtype} "
            f"do not match {expected}/uint8"
        )
    expanded = mx.repeat(mx.repeat(scales, 128, axis=0), 4, axis=1)
    return mx.contiguous(expanded[:output_dims, : input_dims // 32])


def sanitize_mia_exl3_target_weights(
    weights: dict[str, mx.array],
    *,
    layers: int,
    experts: int,
) -> dict[str, mx.array]:
    """Map the exact Mia target storage onto the installed MLX module tree.

    FP8 weights remain byte-identical and are merely viewed as the uint32 words
    MLX's native ``mxfp8`` operator expects.  Their 128x128 E8M0 scale grid is
    repeated into the equivalent per-row, group-32 grid.  EXL3 experts are
    stacked expert-major without decoding or requantizing their payload.
    ``mtp.*`` belongs to the separately loaded K64 draft and is excluded here.
    """

    source = dict(weights)
    mapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    expert_fields: dict[tuple[int, str, str], dict[int, mx.array]] = {}

    for name, value in source.items():
        if name.startswith("mtp."):
            consumed.add(name)
            continue
        match = _EXPERT_KEY.match(name)
        if match is None:
            continue
        consumed.add(name)
        if match.group("field") == "mcg":
            continue
        key = (
            int(match.group("layer")),
            match.group("projection"),
            match.group("field"),
        )
        expert_fields.setdefault(key, {})[int(match.group("expert"))] = value

    for name, value in source.items():
        if name in consumed:
            continue
        if name.endswith(".scale"):
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name in source and source[weight_name].dtype == mx.uint8:
                continue
        if name.endswith(".weight"):
            scale_name = name.removesuffix(".weight") + ".scale"
            scales = source.get(scale_name)
            if scales is not None and value.dtype == mx.uint8:
                if value.ndim != 2 or int(value.shape[1]) % 128:
                    raise ValueError(f"unsupported Mia FP8 weight geometry: {name}")
                output_dims, input_dims = (int(dim) for dim in value.shape)
                target = _map_mia_target_name(name)
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                mapped[target.removesuffix(".weight") + ".scales"] = (
                    _expand_mia_fp8_block_scales(
                        scales,
                        output_dims,
                        input_dims,
                    )
                )
                consumed.update((name, scale_name))
                continue
        target = _map_mia_target_name(name)
        if target.endswith(".ffn.gate.tid2eid") and value.dtype == mx.int64:
            value = value.astype(mx.int32)
        mapped[target] = value
        consumed.add(name)

    expected_ids = set(range(experts))
    for layer in range(layers):
        for projection, target_projection in _PROJECTION_NAMES.items():
            for field in ("trellis", "suh", "svh"):
                values = expert_fields.get((layer, projection, field), {})
                if set(values) != expected_ids:
                    raise ValueError(
                        f"Mia EXL3 layer {layer} {projection}.{field} has "
                        f"{len(values)} experts, expected {experts}"
                    )
                target = (
                    f"model.layers.{layer}.ffn.switch_mlp."
                    f"{target_projection}.{field}"
                )
                mapped[target] = mx.stack(
                    [values[expert] for expert in range(experts)], axis=0
                )

    return mapped


def _open_file_identity(stream) -> tuple[int, int, int, int, int]:
    observed = os.fstat(stream.fileno())
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("Mia shard must be a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _load_verified_safetensors(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_canonical_sha256: str | None = None,
) -> dict[str, mx.array]:
    """Hash and load one stable descriptor with an optional semantic header pin."""

    path = Path(path)
    if len(expected_sha256) != 64:
        raise ValueError(f"pinned Mia shard checksum is invalid: {path.name}")
    if (
        expected_canonical_sha256 is not None
        and len(expected_canonical_sha256) != 64
    ):
        raise ValueError(
            f"pinned Mia shard canonical checksum is invalid: {path.name}"
        )
    with path.open("rb", buffering=0) as stream:
        identity = _open_file_identity(stream)
        if identity[2] != int(expected_bytes):
            raise ValueError(f"pinned Mia shard size changed: {path.name}")

        digest = hashlib.sha256()
        canonical_digest = None
        if expected_canonical_sha256 is not None:
            from mtplx.deepseek_v4_mia_engine import (
                _SAFETENSORS_CANONICAL_PREFIX,
                _canonical_safetensors_header,
            )

            encoded_header_length = stream.read(8)
            if len(encoded_header_length) != 8:
                raise ValueError(f"invalid Mia safetensors header: {path.name}")
            header_length = struct.unpack("<Q", encoded_header_length)[0]
            if header_length == 0 or header_length > identity[2] - 8:
                raise ValueError(f"invalid Mia safetensors header: {path.name}")
            encoded_header = stream.read(header_length)
            if len(encoded_header) != header_length:
                raise ValueError(f"truncated Mia safetensors header: {path.name}")
            try:
                canonical_header = _canonical_safetensors_header(encoded_header)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Mia safetensors JSON header: {path.name}"
                ) from exc
            digest.update(encoded_header_length)
            digest.update(encoded_header)
            canonical_digest = hashlib.sha256()
            canonical_digest.update(_SAFETENSORS_CANONICAL_PREFIX)
            canonical_digest.update(encoded_header_length)
            canonical_digest.update(struct.pack("<Q", len(canonical_header)))
            canonical_digest.update(canonical_header)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if canonical_digest is not None:
                canonical_digest.update(chunk)
        if _open_file_identity(stream) != identity:
            raise ValueError(
                f"pinned Mia shard changed while validating: {path.name}"
            )
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"pinned Mia shard checksum changed: {path.name} "
                f"observed={observed_sha256}, expected={expected_sha256}"
            )
        if canonical_digest is not None and (
            canonical_digest.hexdigest() != expected_canonical_sha256
        ):
            observed_canonical_sha256 = canonical_digest.hexdigest()
            raise ValueError(
                f"pinned Mia shard canonical checksum changed: {path.name} "
                f"observed={observed_canonical_sha256}, "
                f"expected={expected_canonical_sha256}"
            )

        stream.seek(0)
        weights = mx.load(stream, format="safetensors")
        if _open_file_identity(stream) != identity:
            raise ValueError(
                f"pinned Mia shard changed while loading: {path.name}"
            )
    if not isinstance(weights, dict):
        raise ValueError(f"invalid Mia safetensors shard: {path}")
    return dict(weights)


def load_indexed_safetensors(
    root: Path | str,
    *,
    weight_map: dict[str, str] | None = None,
    shard_pins: tuple[Any, ...] | None = None,
) -> dict[str, mx.array]:
    """Load exactly the tensors named by one local safetensors index."""

    root = Path(root)
    if weight_map is None:
        weight_map = _indexed_weight_map(root)
    if not weight_map:
        raise ValueError(f"invalid safetensors index: {root}")
    pins_by_name = (
        {str(pin.name): pin for pin in shard_pins}
        if shard_pins is not None
        else None
    )
    filenames = set(weight_map.values())
    if pins_by_name is not None and set(pins_by_name) != filenames:
        raise ValueError("Mia shard pins do not match the safetensors index")
    expected = set(weight_map)
    weights: dict[str, mx.array] = {}
    for filename in sorted(filenames):
        shard = root / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        if pins_by_name is None:
            weights.update(mx.load(str(shard)))
        else:
            pin = pins_by_name[filename]
            weights.update(
                _load_verified_safetensors(
                    shard,
                    expected_bytes=int(pin.bytes),
                    expected_sha256=str(pin.sha256),
                    expected_canonical_sha256=str(pin.canonical_sha256),
                )
            )
    observed = set(weights)
    if observed != expected:
        raise ValueError(
            f"safetensors index mismatch in {root}: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return weights


def _indexed_weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    return {str(name): str(filename) for name, filename in weight_map.items()}


def _install_quantized_modules(
    model: nn.Module,
    expected: dict[str, str],
    *,
    prefix: str,
) -> dict[str, str]:
    selected: set[str] = set()

    def predicate(path: str, module: nn.Module):
        if not path.startswith(prefix) or not hasattr(module, "to_quantized"):
            return False
        mode = expected.get(path)
        if mode == "mxfp4":
            selected.add(path)
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        if mode == "mxfp8":
            selected.add(path)
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return False

    nn.quantize(model, class_predicate=predicate)
    if selected != set(expected):
        raise ValueError(
            f"Mia quantized module ownership mismatch under {prefix!r}: "
            f"missing={sorted(set(expected) - selected)!r}, "
            f"extra={sorted(selected - set(expected))!r}"
        )
    installed = dict(model.named_modules())
    for path, mode in expected.items():
        module = installed.get(path)
        bits = 4 if mode == "mxfp4" else 8
        if (
            module is None
            or int(getattr(module, "group_size", 0)) != 32
            or int(getattr(module, "bits", 0)) != bits
            or str(getattr(module, "mode", "")) != mode
            or getattr(getattr(module, "weight", None), "dtype", None)
            != mx.uint32
            or getattr(getattr(module, "scales", None), "dtype", None)
            != mx.uint8
            or getattr(module, "biases", None) is not None
        ):
            raise ValueError(
                f"Mia module {path!r} did not install group-32 {mode} ownership"
            )
    return dict(expected)


def _target_quantized_modules_from_index(
    weight_map: dict[str, str],
) -> dict[str, str]:
    expected: dict[str, str] = {}
    for scale_name in weight_map:
        if scale_name.startswith("mtp.") or not scale_name.endswith(".scale"):
            continue
        weight_name = scale_name.removesuffix(".scale") + ".weight"
        if weight_name not in weight_map:
            continue
        target_weight = _map_mia_target_name(weight_name)
        expected[target_weight.removesuffix(".weight")] = "mxfp8"
    return expected


def _map_mia_target_carried_shard(
    source: dict[str, mx.array],
    *,
    fp8_geometries: dict[str, tuple[int, int]],
) -> dict[str, mx.array]:
    """Map one bounded non-expert shard without retaining the other shards.

    The TP1 package splits two FP8 weight/scale pairs across adjacent carried
    shards.  Their installed module geometry is fixed before streaming starts,
    so it supplies both cross-shard ownership and the exact scale expansion
    dimensions without retaining either source shard.
    """

    mapped: dict[str, mx.array] = {}
    for name, value in source.items():
        if name.startswith("mtp."):
            continue
        if _EXPERT_KEY.match(name) is not None:
            raise ValueError("an EXL3 expert tensor reached a carried Mia shard")
        target = _map_mia_target_name(name)
        if name.endswith(".scale"):
            module_name = target.removesuffix(".scale")
            geometry = fp8_geometries.get(module_name)
            if geometry is not None:
                output_dims, input_dims = geometry
                mapped[module_name + ".scales"] = _expand_mia_fp8_block_scales(
                    value,
                    output_dims,
                    input_dims,
                )
                continue
        if name.endswith(".weight"):
            module_name = target.removesuffix(".weight")
            geometry = fp8_geometries.get(module_name)
            if geometry is not None:
                if value.dtype != mx.uint8 or value.ndim != 2:
                    raise ValueError(f"unsupported Mia FP8 weight geometry: {name}")
                output_dims, input_dims = geometry
                if tuple(value.shape) != (output_dims, input_dims):
                    raise ValueError(
                        f"unsupported Mia FP8 weight geometry: {name} owns "
                        f"{tuple(value.shape)}, expected {(output_dims, input_dims)}"
                    )
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                continue
        if target.endswith(".ffn.gate.tid2eid") and value.dtype == mx.int64:
            value = value.astype(mx.int32)
        if ".attn.wo_a." in target and value.ndim == 3:
            value = value.reshape(
                int(value.shape[0]) * int(value.shape[1]),
                int(value.shape[2]),
            )
        mapped[target] = value
    return mapped


def _map_mia_target_expert_shard(
    source: dict[str, mx.array],
    *,
    layer: int,
    experts: int,
) -> dict[str, mx.array]:
    """Stack one layer-local EXL3 shard and discard its unused MCG mirrors."""

    fields: dict[tuple[str, str], dict[int, mx.array]] = {}
    observed: set[str] = set()
    for name, value in source.items():
        match = _EXPERT_KEY.match(name)
        if match is None or int(match.group("layer")) != int(layer):
            raise ValueError(
                f"EXL3 shard for layer {layer} owns unexpected tensor {name!r}"
            )
        observed.add(name)
        if match.group("field") == "mcg":
            continue
        key = (match.group("projection"), match.group("field"))
        fields.setdefault(key, {})[int(match.group("expert"))] = value

    expected_ids = set(range(int(experts)))
    mapped: dict[str, mx.array] = {}
    for projection, target_projection in _PROJECTION_NAMES.items():
        for field in ("trellis", "suh", "svh"):
            values = fields.get((projection, field), {})
            if set(values) != expected_ids:
                raise ValueError(
                    f"Mia EXL3 layer {layer} {projection}.{field} has "
                    f"{len(values)} experts, expected {experts}"
                )
            target = (
                f"model.layers.{int(layer)}.ffn.switch_mlp."
                f"{target_projection}.{field}"
            )
            mapped[target] = mx.stack(
                [values[expert] for expert in range(int(experts))],
                axis=0,
            )
    if observed != set(source):
        raise ValueError(f"Mia EXL3 layer {layer} shard was not fully consumed")
    return mapped


def _install_mia_weight_batch(
    model: nn.Module,
    mapped: dict[str, mx.array],
    installed_names: set[str],
) -> None:
    overlap = installed_names.intersection(mapped)
    if overlap:
        raise ValueError(f"Mia target parameters were loaded twice: {sorted(overlap)!r}")
    if mapped:
        mx.eval(*mapped.values())
        model.load_weights(list(mapped.items()), strict=False)
        installed_names.update(mapped)


def load_mia_exl3_target_streaming(
    model: nn.Module,
    root: Path | str,
    *,
    layers: int,
    experts: int,
    weight_map: dict[str, str] | None = None,
    shard_pins: tuple[Any, ...] | None = None,
) -> dict[str, str]:
    """Install the 106 GB target with one source shard live at a time.

    The rank-sliced package owns five carried shards and one complete EXL3 shard
    per target layer.  Keeping that boundary avoids the former all-shards source
    dictionary and bounds conversion scratch to one carried shard or one 2 GB
    expert layer while preserving the exact destination tensors.
    """

    from mlx.utils import tree_flatten

    root = Path(root)
    if weight_map is None:
        weight_map = _indexed_weight_map(root)
    pins_by_name = (
        {str(pin.name): pin for pin in shard_pins}
        if shard_pins is not None
        else None
    )
    if pins_by_name is not None and set(pins_by_name) != set(weight_map.values()):
        raise ValueError("Mia target shard pins do not match its weight index")

    def load_shard(filename: str) -> dict[str, mx.array]:
        shard = root / filename
        if pins_by_name is None:
            return dict(mx.load(str(shard)))
        pin = pins_by_name[filename]
        return _load_verified_safetensors(
            shard,
            expected_bytes=int(pin.bytes),
            expected_sha256=str(pin.sha256),
            expected_canonical_sha256=str(pin.canonical_sha256),
        )

    quantized = _target_quantized_modules_from_index(weight_map)
    model_quantized = {
        path: mode for path, mode in quantized.items() if path.startswith("model.")
    }
    head_quantized = {
        path: mode for path, mode in quantized.items() if path.startswith("lm_head")
    }
    receipt: dict[str, str] = {}
    receipt.update(
        _install_quantized_modules(model, model_quantized, prefix="model.")
    )
    receipt.update(
        _install_quantized_modules(model, head_quantized, prefix="lm_head")
    )
    installed_modules = dict(model.named_modules())
    fp8_geometries: dict[str, tuple[int, int]] = {}
    for path in quantized:
        module = installed_modules[path]
        output_dims = int(module.scales.shape[0])
        input_dims = int(module.scales.shape[1]) * 32
        if tuple(module.weight.shape) != (output_dims, input_dims // 4):
            raise ValueError(f"Mia module {path!r} has invalid MXFP8 geometry")
        fp8_geometries[path] = (output_dims, input_dims)

    files: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        files.setdefault(filename, set()).add(name)
    installed_names: set[str] = set()

    carried_files = {
        name for name in files if name.startswith("carried-")
    }
    for filename in sorted(carried_files):
        source = load_shard(filename)
        if set(source) != files[filename]:
            raise ValueError(f"Mia carried shard index mismatch: {filename}")
        mapped = _map_mia_target_carried_shard(
            source,
            fp8_geometries=fp8_geometries,
        )
        _install_mia_weight_batch(model, mapped, installed_names)
        del mapped, source
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    expert_files = {
        name for name in files if name.startswith("exl3-layer-")
    }
    if len(expert_files) != int(layers):
        raise ValueError(
            f"Mia target owns {len(expert_files)} EXL3 layer shards, "
            f"expected {layers}"
        )
    for layer in range(int(layers)):
        filename = f"exl3-layer-{layer:03d}-tp1-rank0.safetensors"
        if filename not in expert_files:
            raise ValueError(f"Mia target is missing {filename}")
        source = load_shard(filename)
        if set(source) != files[filename]:
            raise ValueError(f"Mia EXL3 shard index mismatch: {filename}")
        mapped = _map_mia_target_expert_shard(
            source,
            layer=layer,
            experts=experts,
        )
        _install_mia_weight_batch(model, mapped, installed_names)
        del mapped, source
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    unexpected_files = set(files) - carried_files - expert_files
    if unexpected_files:
        raise ValueError(
            f"Mia target index owns unexpected shards: {sorted(unexpected_files)!r}"
        )
    installed_parameters = {
        name for name, _value in tree_flatten(model.parameters())
    }
    if installed_names != installed_parameters:
        raise ValueError(
            "Mia streaming target parameter mismatch: "
            f"missing={len(installed_parameters - installed_names)}, "
            f"extra={len(installed_names - installed_parameters)}"
        )
    model._mia_target_load_receipt = {
        "mode": "bounded_one_shard",
        "artifact_identity": (
            "raw_canonical_sha256_same_fd"
            if pins_by_name is not None
            else "unverified_path"
        ),
        "source_shards": len(files),
        "carried_shards": len(carried_files),
        "exl3_layer_shards": len(expert_files),
        "installed_parameters": len(installed_names),
    }
    return receipt


def _map_mia_dspark_name(name: str) -> str:
    mapped = name
    replacements = (
        (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_proj"),
        (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
        (".ffn.shared_experts.w3", ".ffn.shared_experts.up_proj"),
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
        (".hc_head_fn", ".hc_head.fn"),
        (".hc_head_base", ".hc_head.base"),
        (".hc_head_scale", ".hc_head.scale"),
    )
    for source, target in replacements:
        mapped = mapped.replace(source, target)
    return mapped


def sanitize_mia_dspark_weights(
    weights: dict[str, mx.array],
    *,
    stages: int,
    experts: int,
) -> dict[str, mx.array]:
    """Map the exact Mia K64 draft onto the installed three-stage owner.

    Routed weights remain byte-identical OCP FP4 and dense projections remain
    byte-identical E4M3.  Only their array views and scale-grid ownership change
    to the native MLX ``mxfp4``/``mxfp8`` module contracts.
    """

    source = dict(weights)
    mapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    expert_fields: dict[tuple[int, str, str], dict[int, mx.array]] = {}

    for name, value in source.items():
        match = _DSPARK_EXPERT_KEY.match(name)
        if match is None:
            continue
        consumed.add(name)
        key = (
            int(match.group("stage")),
            match.group("projection"),
            match.group("field"),
        )
        expert_fields.setdefault(key, {})[int(match.group("expert"))] = value

    for name, value in source.items():
        if name in consumed:
            continue
        if name.endswith(".scale"):
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name in source and source[weight_name].dtype == mx.uint8:
                continue
        if name.endswith(".weight"):
            scale_name = name.removesuffix(".weight") + ".scale"
            scales = source.get(scale_name)
            if scales is not None and value.dtype == mx.uint8:
                if value.ndim != 2 or int(value.shape[1]) % 128:
                    raise ValueError(f"unsupported Mia DSpark FP8 geometry: {name}")
                output_dims, input_dims = (int(dim) for dim in value.shape)
                target = _map_mia_dspark_name(name)
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                mapped[target.removesuffix(".weight") + ".scales"] = (
                    _expand_mia_fp8_block_scales(scales, output_dims, input_dims)
                )
                consumed.update((name, scale_name))
                continue
        mapped[_map_mia_dspark_name(name)] = value
        consumed.add(name)

    expected_ids = set(range(experts))
    for stage in range(stages):
        for projection, target_projection in _PROJECTION_NAMES.items():
            weights_by_expert = expert_fields.get((stage, projection, "weight"), {})
            scales_by_expert = expert_fields.get((stage, projection, "scale"), {})
            if set(weights_by_expert) != expected_ids or set(scales_by_expert) != expected_ids:
                raise ValueError(
                    f"Mia DSpark stage {stage} {projection} has incomplete K{experts} storage"
                )
            stem = f"mtp.{stage}.ffn.switch_mlp.{target_projection}"
            mapped[f"{stem}.weight"] = mx.stack(
                [
                    mx.contiguous(weights_by_expert[expert]).view(mx.uint32)
                    for expert in range(experts)
                ],
                axis=0,
            )
            mapped[f"{stem}.scales"] = mx.stack(
                [scales_by_expert[expert] for expert in range(experts)], axis=0
            )

    if consumed != set(source):
        raise ValueError(
            "unmapped Mia DSpark tensors: " + ", ".join(sorted(set(source) - consumed))
        )
    return mapped


def _quantize_loaded_modules(
    model: nn.Module,
    weights: dict[str, mx.array],
    *,
    prefix: str,
) -> dict[str, str]:
    expected = {
        name.removesuffix(".scales"): (
            "mxfp4" if ".ffn.switch_mlp." in name else "mxfp8"
        )
        for name in weights
        if name.startswith(prefix) and name.endswith(".scales")
    }
    return _install_quantized_modules(model, expected, prefix=prefix)


def _default_mia_dspark_root(target_root: Path) -> Path:
    configured = json.loads((target_root / "config.json").read_text()).get(
        "dspark_draft_model"
    )
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else target_root / candidate
    suffix = "-tp1"
    if target_root.name.endswith(suffix):
        return target_root.with_name(
            target_root.name.removesuffix(suffix) + "-dspark-k64"
        )
    return target_root / "dspark-k64"


def load_mia_exl3_dspark_model(
    target_root: Path | str,
    *,
    draft_root: Path | str | None = None,
    artifact_validation=None,
    lazy: bool = False,
    context_capacity_tokens: int = 384_000,
    max_batch_tokens: int = 8_224,
):
    """Construct the exact split Mia K216 target plus K64 DSpark owner."""

    from mlx.utils import tree_flatten

    from mtplx.models.deepseek_v4 import Model, ModelArgs
    from mtplx.models.deepseek_v4_dspark import build_deepseek_v4_dspark

    target_root = Path(target_root).resolve()
    resolved_draft = (
        Path(draft_root).resolve()
        if draft_root is not None
        else _default_mia_dspark_root(target_root).resolve()
    )
    from mtplx.deepseek_v4_mia_engine import validate_pinned_mia_artifacts

    if artifact_validation is None:
        artifact_validation = validate_pinned_mia_artifacts(
            target_root,
            resolved_draft,
        )
    elif (
        artifact_validation.target_root != target_root
        or artifact_validation.draft_root != resolved_draft
    ):
        raise ValueError(
            "pinned Mia artifact validation does not own the requested roots"
        )
    target_config = dict(artifact_validation.target_config)
    # Qualify the source metadata before clearing only the separately-owned
    # draft signature for target construction.
    ModelArgs.from_dict(target_config)
    target_only = dict(target_config)
    target_only.update(
        {
            "dspark_block_size": None,
            "dspark_markov_rank": None,
            "dspark_noise_token_id": None,
            "dspark_target_layer_ids": None,
            "num_nextn_predict_layers": 0,
        }
    )
    model = Model(ModelArgs.from_dict(target_only))
    quantized_modules = load_mia_exl3_target_streaming(
        model,
        target_root,
        layers=int(model.args.num_hidden_layers),
        experts=int(model.args.n_routed_experts),
        weight_map=artifact_validation.target_weight_map,
        shard_pins=artifact_validation.target_shards,
    )
    model._mia_target_load_receipt["small_file_sha256"] = dict(
        artifact_validation.target_small_file_sha256
    )
    model.eval()
    for layer in model.layers:
        layer.ffn.install_mia_exl3_runtime(max_tokens=8224)
    install_mia_m6_quad_qmv_routes(model)

    draft_config = dict(artifact_validation.draft_config)
    draft_experts = int(draft_config.get("n_routed_experts", 0))
    draft_config["hybrid_tr3_tail"] = None
    draft_args = ModelArgs.from_dict(draft_config)
    if draft_experts != 64:
        raise ValueError(f"Mia DSpark draft must own K64, got K{draft_experts}")
    owner = build_deepseek_v4_dspark(draft_args)
    model.install_dspark_owner(owner)

    draft_source = load_indexed_safetensors(
        resolved_draft,
        weight_map=artifact_validation.draft_weight_map,
        shard_pins=artifact_validation.draft_shards,
    )
    draft_weights = sanitize_mia_dspark_weights(
        draft_source,
        stages=3,
        experts=64,
    )
    del draft_source
    model._mia_draft_load_receipt = {
        "mode": "single_shard",
        "artifact_identity": "raw_canonical_sha256_same_fd",
        "source_shards": len(artifact_validation.draft_shards),
        "source_tensors": len(artifact_validation.draft_weight_map),
        "small_file_sha256": dict(
            artifact_validation.draft_small_file_sha256
        ),
    }
    quantized_modules.update(
        _quantize_loaded_modules(model, draft_weights, prefix="mtp.")
    )
    installed = {
        name: value
        for name, value in tree_flatten(model.parameters())
        if name.startswith("mtp.")
    }
    if set(installed) != set(draft_weights):
        raise ValueError(
            "Mia DSpark installed parameter mismatch: "
            f"missing={len(set(installed) - set(draft_weights))}, "
            f"extra={len(set(draft_weights) - set(installed))}"
        )
    for name, value in installed.items():
        if value.shape != draft_weights[name].shape:
            raise ValueError(
                f"Mia DSpark shape mismatch for {name}: "
                f"installed={value.shape}, source={draft_weights[name].shape}"
            )
    model.load_weights(list(draft_weights.items()), strict=False)
    model._mia_quantized_modules = dict(quantized_modules)

    # Replace the layer-local mHC chain with the pinned carried state machine
    # for both the 43-layer target and the three-stage K64 draft.
    model.install_mia_mhc_runtime(max_tokens=8224)

    # Bind the source-derived B12X WO owner directly against the native MXFP8
    # tensors after both target and draft weights exist.  Each attention owns a
    # distinct prebound plan; generation never enters the generic o-LoRA route.
    from mtplx.models.deepseek_v4 import (
        install_mia_qkv_prologue_routes,
        install_mia_tp1_wo_projection_routes,
        install_mia_stacked_projections,
    )

    wo_projection = install_mia_tp1_wo_projection_routes(
        model,
        max_prefill_rows=max_batch_tokens,
    )
    if (
        wo_projection["route"] != "mia_tp1_b12x_wo_mxfp8"
        or wo_projection["target_attention"] != 43
        or wo_projection["draft_attention"] != 3
        or wo_projection["plan_count"] != 46
        or wo_projection["unique_plan_count"] != 46
        or wo_projection["plan_type"] != "MiaTP1WOMXFP8Plan"
        or wo_projection["max_prefill_rows"] != int(max_batch_tokens)
    ):
        raise RuntimeError(
            f"Mia TP1 WO projection route is incomplete: {wo_projection}"
        )

    stacked_projections = install_mia_stacked_projections(model)
    if stacked_projections != {
        "target_attention": 43,
        "draft_attention": 3,
        "shared_expert": 46,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }:
        raise RuntimeError(
            f"Mia stacked projection installation is incomplete: "
            f"{stacked_projections}"
        )

    qkv_prologue = install_mia_qkv_prologue_routes(model)
    if (
        qkv_prologue["route"] != "mia_fused_qkv_stock432"
        or qkv_prologue["target_attention"] != 43
        or qkv_prologue["draft_attention"] != 3
        or qkv_prologue["plan_count"] != 46
        or qkv_prologue["unique_plan_count"] != 46
        or qkv_prologue["plan_type"] != "MiaBoundQKVPrologue"
        or qkv_prologue["prefill_cutoff"] != 1024
        or qkv_prologue["proposal_rows"] != 5
        or qkv_prologue["context_rows"] != 128
    ):
        raise RuntimeError(
            f"Mia fused Q/KV prologue installation is incomplete: {qkv_prologue}"
        )

    from mtplx.deepseek_v4_mia_engine import build_mia_engine_plan

    engine_plan = build_mia_engine_plan(
        model,
        target_root=target_root,
        draft_root=resolved_draft,
        context_capacity_tokens=context_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
    )
    model.install_mia_engine_plan(engine_plan)
    if not lazy:
        mx.eval(model.parameters())
        model._mia_prewarm_receipt = engine_plan.prewarm(model)
    return model
