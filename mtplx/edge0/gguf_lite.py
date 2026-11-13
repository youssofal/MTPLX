"""Lazy, NumPy-only GGUF v3 reader.

Parses the header (metadata and tensor table) and nothing else; tensor bytes
are exposed as read-only ``np.memmap`` views, so opening a multi-GB skeleton
costs one small read. Layout follows the format used by the Edge0 reference
engine (``edge0-strata``: ``include/strata/artifact/gguf_reader.hpp``,
``tools/gguf_reader.py``). GGUF stores dims innermost first; ``GGUFTensor.shape``
is the reversed, NumPy row-major shape.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ggml_quants import decode_bf16, decode_q8_0

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_I8 = 24
GGML_BF16 = 30

# value type id -> struct code (scalars); 8 is string, 9 is array
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING = 8
_ARRAY = 9

# type id -> (values per block, bytes per block)
_BLOCKS = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_Q8_0: (32, 34),
    GGML_I8: (1, 1),
    GGML_BF16: (1, 2),
    2: (32, 18),    # Q4_0
    40: (64, 36),   # NVFP4
}


class GGUFError(ValueError):
    """Raised for a malformed or unsupported GGUF file."""


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]  # NumPy row-major (GGUF dims reversed)
    ggml_type: int
    offset: int  # absolute file offset of the tensor data
    nbytes: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class _Cursor:
    def __init__(self, fh, path: Path, size: int):
        self.fh = fh
        self.path = path
        self.size = size

    def read(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError(f"{self.path}: truncated GGUF header (wanted {n} bytes at {self.fh.tell() - len(data)})")
        return data

    def unpack(self, code: str):
        fmt = "<" + code
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("Q")
        if n > self.size:
            raise GGUFError(f"{self.path}: implausible string length {n}")
        return self.read(n).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype = self.unpack("I")
            n = self.unpack("Q")
            if n > self.size:
                raise GGUFError(f"{self.path}: implausible array length {n}")
            return [self.value(etype) for _ in range(n)]
        raise GGUFError(f"{self.path}: unknown metadata value type {vtype}")


def tensor_nbytes(ggml_type: int, n_elements: int) -> int:
    if ggml_type not in _BLOCKS:
        raise GGUFError(f"unsupported ggml type {ggml_type} for size computation")
    values, nbytes = _BLOCKS[ggml_type]
    if n_elements % values:
        raise GGUFError(f"{n_elements} elements is not a multiple of the type-{ggml_type} block size {values}")
    return n_elements // values * nbytes


class GGUFFile:
    """A GGUF v3 file opened lazily: header only, tensor data via memmap."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, GGUFTensor] = {}
        file_size = self.path.stat().st_size
        with open(self.path, "rb") as fh:
            cur = _Cursor(fh, self.path, file_size)
            if cur.read(4) != GGUF_MAGIC:
                raise GGUFError(f"{self.path}: not a GGUF file (bad magic)")
            self.version = cur.unpack("I")
            if self.version != 3:
                raise GGUFError(f"{self.path}: GGUF version {self.version} is not supported (need 3)")
            n_tensors = cur.unpack("Q")
            n_kv = cur.unpack("Q")
            for _ in range(n_kv):
                key = cur.string()
                self.metadata[key] = cur.value(cur.unpack("I"))
            infos = []
            for _ in range(n_tensors):
                name = cur.string()
                n_dims = cur.unpack("I")
                dims = tuple(cur.unpack("Q") for _ in range(n_dims))
                ggml_type = cur.unpack("I")
                rel = cur.unpack("Q")
                infos.append((name, dims, ggml_type, rel))
            header_end = fh.tell()
        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        self.data_offset = -(-header_end // self.alignment) * self.alignment
        for name, dims, ggml_type, rel in infos:
            shape = tuple(reversed(dims))
            n = 1
            for d in shape:
                n *= d
            nbytes = tensor_nbytes(ggml_type, n)
            offset = self.data_offset + rel
            if offset + nbytes > file_size:
                raise GGUFError(
                    f"{self.path}: tensor {name!r} needs bytes {offset}..{offset + nbytes} "
                    f"but the file has {file_size}"
                )
            self.tensors[name] = GGUFTensor(name, shape, ggml_type, offset, nbytes)

    def tensor(self, name: str) -> GGUFTensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None

    def raw(self, name: str) -> np.memmap:
        """The tensor's stored bytes as a flat read-only uint8 memmap."""
        t = self.tensor(name)
        return np.memmap(self.path, dtype=np.uint8, mode="r", offset=t.offset, shape=(t.nbytes,))

    def array(self, name: str) -> np.ndarray:
        """Decode an F32, F16, BF16 or Q8_0 tensor to float32 with its row-major shape."""
        t = self.tensor(name)
        raw = self.raw(name)
        if t.ggml_type == GGML_F32:
            out = np.asarray(raw).view("<f4").astype(np.float32)
        elif t.ggml_type == GGML_F16:
            out = np.asarray(raw).view("<f2").astype(np.float32)
        elif t.ggml_type == GGML_BF16:
            out = decode_bf16(raw)
        elif t.ggml_type == GGML_Q8_0:
            out = decode_q8_0(raw)
        else:
            raise GGUFError(f"tensor {name!r} has ggml type {t.ggml_type}; array() decodes only F32, F16, BF16 and Q8_0")
        return out.reshape(t.shape)
