"""Stored per-matrix affine precision for the consumer models. Activations keep their trained format."""

import mlx.core as mx
import mlx.nn as nn


class AffineWeight(nn.Module):
    def __init__(self, codes, scales, biases, bits, group_size):
        super().__init__()
        self.codes, self.scales, self.biases = codes, scales, biases
        self.bits, self.group_size = bits, group_size

    def product(self, x):
        return mx.quantized_matmul(
            x,
            self.codes,
            self.scales,
            self.biases,
            bits=self.bits,
            group_size=self.group_size,
        )

    def rows(self, start, end):
        return mx.dequantize(
            self.codes[start:end],
            self.scales[start:end],
            self.biases[start:end],
            bits=self.bits,
            group_size=self.group_size,
        )


class LogicalShape:
    """The shape and dtype of a packed matrix, for code that sizes buffers or casts from them.

    It holds no data: indexing it or multiplying by it fails, so a caller that needs the values
    has to read the packed weight it describes."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(int(n) for n in shape), dtype

    @property
    def ndim(self):
        return len(self.shape)


def packed_logical_shape(weight):
    """(rows, columns) and dtype of an ``AffineWeight`` as its dequantized matrix."""
    rows, words = weight.codes.shape[-2:]
    return LogicalShape((rows, words * 32 // weight.bits), weight.scales.dtype)


def decoded_experts(codes, scales, mode, outer=None):
    fp4 = mode == "nvfp4"
    weight = mx.dequantize(
        codes.view(mx.uint32),
        scales,
        bits=4 if fp4 else 8,
        group_size=16 if fp4 else 32,
        mode=mode,
        dtype=mx.float32,
    )
    return (weight * outer[..., None, None] if outer is not None else weight).astype(
        mx.bfloat16
    )


def source_modes(path, count):
    import json

    quant = json.loads((path / "hf_quant_config.json").read_text())
    modes = {
        name: group["compute_dtype"]
        for group in quant["model"]["parameter_groups"]
        if group["compute_dtype"] in ("nvfp4", "mxfp8")
        for name in group["tensors"]
    }
    result = []
    for layer in range(count):
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        gate, down = (modes[prefix + name] for name in ("w13_weight", "w2_weight"))
        if gate != down:
            raise ValueError(
                f"Inconsistent trained activation precision in layer {layer}"
            )
        result.append(gate)
    return result
