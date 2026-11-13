"""Reader for an Edge0 latent-MoE model folder (NVFP4 build).

Format source: ``edge0-strata`` ``src/kernels/cpu/expert_layout.cpp`` (the
``native_experts.txt`` parser), ``src/kernels/cpu/native_expert.cpp`` (blob
layout and tail scales) and ``src/core/latent_dense.cpp`` (D/U files).

Folder contents used here: ``model.json``, ``native_experts.txt``,
``experts.bin`` (headerless, per layer ``n_experts`` blobs of
``[gate | up | down | 16-byte tail]``), ``labels.bin`` (int32 [layers, experts]),
``latent_dense_q4.bin`` / ``latent_dense_q8.bin`` (per layer D then U),
``model-skeleton.gguf`` and ``ple-fp8.gguf``. Everything is memory-mapped; the
constructor only stats files and parses small text headers.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ggml_quants import (
    NVFP4_BLOCK_BYTES,
    NVFP4_BLOCK_VALUES,
    Q4_0_BLOCK_BYTES,
    Q4_0_BLOCK_VALUES,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_VALUES,
    decode_nvfp4,
    decode_q4_0,
    decode_q8_0,
)
from .gguf_lite import GGUFFile

GGML_TYPE_NVFP4 = 40
TAIL_BYTES = 16  # f32 s_gate, s_up, s_down, 0

_DENSE_FILES = {"q4": "latent_dense_q4.bin", "q8": "latent_dense_q8.bin"}
_DENSE_BLOCKS = {"q4": (Q4_0_BLOCK_VALUES, Q4_0_BLOCK_BYTES), "q8": (Q8_0_BLOCK_VALUES, Q8_0_BLOCK_BYTES)}
_DENSE_DECODERS = {"q4": decode_q4_0, "q8": decode_q8_0}
_REQUIRED = ("model.json", "native_experts.txt", "experts.bin", "labels.bin", "model-skeleton.gguf")


class Edge0PackError(ValueError):
    """Raised when a model folder is missing a file or a file has the wrong size or content."""


@dataclass
class LatentExpert:
    """One expert's latent-space weights, decoded but not yet multiplied by the tail scales."""

    gate: np.ndarray  # [I, L]
    up: np.ndarray  # [I, L]
    down: np.ndarray  # [L, I]
    s_gate: float
    s_up: float
    s_down: float

    def scaled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gate, up, down) multiplied by their projection's tail scale."""
        return self.gate * np.float32(self.s_gate), self.up * np.float32(self.s_up), self.down * np.float32(self.s_down)


def _nvfp4_bytes(rows: int, cols: int) -> int:
    return rows * (cols // NVFP4_BLOCK_VALUES) * NVFP4_BLOCK_BYTES


class Edge0Pack:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise Edge0PackError(f"{self.path}: not a directory")
        for name in _REQUIRED:
            if not (self.path / name).exists():
                raise Edge0PackError(f"{self.path / name}: required file is missing")
        try:
            meta = json.loads((self.path / "model.json").read_text())
        except json.JSONDecodeError as exc:
            raise Edge0PackError(f"{self.path / 'model.json'}: invalid JSON ({exc})") from exc
        try:
            self.model_name: str = str(meta["model_name"])
            self.n_layers: int = int(meta["layers"])
            self.n_experts: int = int(meta["experts"])
            self.latent_dim: int = int(meta["latent_dim"])
        except KeyError as exc:
            raise Edge0PackError(f"{self.path / 'model.json'}: missing key {exc}") from exc
        self.build: str = str(meta.get("build", ""))
        fmt = str((meta.get("env") or {}).get("STRATA_LATENT_DENSE_FMT", "8"))
        if fmt not in ("4", "8"):
            raise Edge0PackError(f"{self.path / 'model.json'}: STRATA_LATENT_DENSE_FMT must be 4 or 8, got {fmt!r}")
        self.default_dense_format: str = "q" + fmt

        self._parse_native_experts()
        self.hidden: int = self._resolve_hidden(meta)
        self._check_experts_file()
        self._check_labels()
        self.n_groups: int = self._resolve_groups(meta)
        self._skeleton: GGUFFile | None = None
        self._ple: GGUFFile | None = None

    # -- construction helpers -------------------------------------------------

    def _parse_native_experts(self) -> None:
        path = self.path / "native_experts.txt"
        expert_ff = None
        rows: dict[int, tuple[int, int, int, int]] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if (m := re.search(r"native experts v(\d+)", line)) and int(m.group(1)) != 6:
                    raise Edge0PackError(f"{path}: native experts layout v{m.group(1)} is not supported (need v6)")
                if m := re.search(r"expert_ff (\d+)", line):
                    expert_ff = int(m.group(1))
                if m := re.search(r"expert_embd (\d+)", line):
                    if int(m.group(1)) != self.latent_dim:
                        raise Edge0PackError(
                            f"{path}: expert_embd {m.group(1)} disagrees with model.json latent_dim {self.latent_dim}"
                        )
                continue
            parts = line.split()
            try:
                layer, gt, dt, off, blob = (int(p) for p in parts[:5])
            except ValueError:
                raise Edge0PackError(f"{path}: malformed line {line!r}") from None
            if len(parts) < 5 or not 0 <= layer < self.n_layers or layer in rows:
                raise Edge0PackError(f"{path}: bad or duplicate layer in line {line!r}")
            rows[layer] = (gt, dt, off, blob)
        if sorted(rows) != list(range(self.n_layers)):
            raise Edge0PackError(f"{path}: expected rows for layers 0..{self.n_layers - 1}, got {len(rows)} rows")
        types = {(gt, dt) for gt, dt, _, _ in rows.values()}
        if types != {(GGML_TYPE_NVFP4, GGML_TYPE_NVFP4)}:
            raise NotImplementedError(
                f"{path}: expert types {sorted(types)} (build {self.build!r}); only all-NVFP4 (ggml type 40) "
                "experts are supported, Q3_K and mixed-type packs are not implemented yet"
            )
        self.expert_format = "NVFP4"
        if expert_ff is None:
            raise Edge0PackError(f"{path}: header has no 'expert_ff' value")
        self.expert_ff: int = expert_ff
        if self.latent_dim % NVFP4_BLOCK_VALUES or expert_ff % NVFP4_BLOCK_VALUES:
            raise Edge0PackError(f"{path}: latent_dim {self.latent_dim} and expert_ff {expert_ff} must be multiples of 64")
        self._proj_bytes = _nvfp4_bytes(expert_ff, self.latent_dim)
        self.blob_bytes: int = 3 * self._proj_bytes + TAIL_BYTES
        at = 0
        self._layer_offsets: list[int] = []
        for layer in range(self.n_layers):
            _, _, off, blob = rows[layer]
            if blob != self.blob_bytes:
                raise Edge0PackError(f"{path}: layer {layer} blob is {blob} bytes, geometry needs {self.blob_bytes}")
            if off != at:
                raise Edge0PackError(f"{path}: layer {layer} offset {off} is not contiguous (expected {at})")
            self._layer_offsets.append(off)
            at += blob * self.n_experts
        self._experts_total = at

    def _size(self, name: str) -> int:
        return os.stat(self.path / name).st_size  # follows symlinks

    def _expect_size(self, name: str, expected: int, why: str) -> None:
        actual = self._size(name)
        if actual != expected:
            raise Edge0PackError(f"{self.path / name}: expected {expected} bytes ({why}), found {actual}")

    def _resolve_hidden(self, meta: dict) -> int:
        if "hidden" in meta:
            return int(meta["hidden"])
        skel = GGUFFile(self.path / "model-skeleton.gguf")
        arch = skel.metadata.get("general.architecture")
        value = skel.metadata.get(f"{arch}.embedding_length")
        if not isinstance(value, int):
            raise Edge0PackError(f"{self.path / 'model-skeleton.gguf'}: no '{arch}.embedding_length' metadata for the hidden size")
        return value

    def _check_experts_file(self) -> None:
        self._expect_size("experts.bin", self._experts_total, f"{self.n_layers} layers x {self.n_experts} experts x {self.blob_bytes} B")

    def _check_labels(self) -> None:
        self._expect_size("labels.bin", self.n_layers * self.n_experts * 4, f"int32 [{self.n_layers}, {self.n_experts}]")
        labels = np.fromfile(self.path / "labels.bin", dtype="<i4").reshape(self.n_layers, self.n_experts)
        if labels.min() < 0:
            raise Edge0PackError(f"{self.path / 'labels.bin'}: negative group label {int(labels.min())}")
        self._labels = labels.astype(np.int32)

    def _dense_layer_bytes(self, fmt: str, n_groups: int) -> tuple[int, int]:
        values, nbytes = _DENSE_BLOCKS[fmt]
        d = n_groups * self.latent_dim * (self.hidden // values) * nbytes
        u = n_groups * self.hidden * (self.latent_dim // values) * nbytes
        return d, u

    def _resolve_groups(self, meta: dict) -> int:
        fmt = self.default_dense_format
        name = _DENSE_FILES[fmt]
        if not (self.path / name).exists():
            raise Edge0PackError(f"{self.path / name}: required file is missing (the model's default dense format is {fmt})")
        values, _ = _DENSE_BLOCKS[fmt]
        if self.hidden % values or self.latent_dim % values:
            raise Edge0PackError(f"{self.path}: hidden {self.hidden} and latent_dim {self.latent_dim} must be multiples of {values}")
        d1, u1 = self._dense_layer_bytes(fmt, 1)
        per_group = self.n_layers * (d1 + u1)
        size = self._size(name)
        if "n_groups" in meta:
            groups = int(meta["n_groups"])
        else:
            groups, rem = divmod(size, per_group)
            if rem or groups < 1:
                raise Edge0PackError(
                    f"{self.path / name}: expected a multiple of {per_group} bytes ({per_group} per group for "
                    f"{self.n_layers} layers, hidden {self.hidden}, latent {self.latent_dim}), found {size}"
                )
        if int(self._labels.max()) >= groups:
            raise Edge0PackError(f"{self.path / 'labels.bin'}: group label {int(self._labels.max())} out of range for {groups} groups")
        for other, fname in _DENSE_FILES.items():
            if (self.path / fname).exists():
                d, u = self._dense_layer_bytes(other, groups)
                self._expect_size(fname, self.n_layers * (d + u), f"{self.n_layers} layers x (D {d} + U {u}), {groups} groups")
        return groups

    # -- accessors ----------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """int32 [n_layers, n_experts]: the group of each expert."""
        return self._labels

    def _check_ids(self, layer: int, expert: int | None = None) -> None:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        if expert is not None and not 0 <= expert < self.n_experts:
            raise IndexError(f"expert {expert} out of range 0..{self.n_experts - 1}")

    def expert_blob(self, layer: int, expert: int) -> np.memmap:
        """Raw bytes of one expert blob (read-only uint8 memmap)."""
        self._check_ids(layer, expert)
        off = self._layer_offsets[layer] + expert * self.blob_bytes
        return np.memmap(self.path / "experts.bin", dtype=np.uint8, mode="r", offset=off, shape=(self.blob_bytes,))

    def expert(self, layer: int, expert: int) -> LatentExpert:
        blob = np.asarray(self.expert_blob(layer, expert))
        p, i, lat = self._proj_bytes, self.expert_ff, self.latent_dim
        s_gate, s_up, s_down, _ = struct.unpack("<4f", blob[3 * p :].tobytes())
        return LatentExpert(
            gate=decode_nvfp4(blob[:p]).reshape(i, lat),
            up=decode_nvfp4(blob[p : 2 * p]).reshape(i, lat),
            down=decode_nvfp4(blob[2 * p : 3 * p]).reshape(lat, i),
            s_gate=s_gate,
            s_up=s_up,
            s_down=s_down,
        )

    def _dense(self, layer: int, fmt: str | None, which: int) -> np.ndarray:
        self._check_ids(layer)
        fmt = fmt or self.default_dense_format
        if fmt not in _DENSE_FILES:
            raise ValueError(f"dense format must be 'q4' or 'q8', got {fmt!r}")
        path = self.path / _DENSE_FILES[fmt]
        if not path.exists():
            raise Edge0PackError(f"{path}: file is missing")
        d_bytes, u_bytes = self._dense_layer_bytes(fmt, self.n_groups)
        start = layer * (d_bytes + u_bytes) + (0 if which == 0 else d_bytes)
        n = d_bytes if which == 0 else u_bytes
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=start, shape=(n,))
        values = _DENSE_DECODERS[fmt](raw)
        if which == 0:
            return values.reshape(self.n_groups, self.latent_dim, self.hidden)
        return values.reshape(self.n_groups, self.hidden, self.latent_dim)

    def dense_d(self, layer: int, fmt: str | None = None) -> np.ndarray:
        """float32 [G, L, H]: group g, latent unit i is row g * L + i of D."""
        return self._dense(layer, fmt, 0)

    def dense_u(self, layer: int, fmt: str | None = None) -> np.ndarray:
        """float32 [G, H, L]: group g, hidden unit h is row g * H + h of U."""
        return self._dense(layer, fmt, 1)

    @property
    def skeleton(self) -> GGUFFile:
        if self._skeleton is None:
            self._skeleton = GGUFFile(self.path / "model-skeleton.gguf")
        return self._skeleton

    @property
    def ple(self) -> GGUFFile:
        if self._ple is None:
            if not (self.path / "ple-fp8.gguf").exists():
                raise Edge0PackError(f"{self.path / 'ple-fp8.gguf'}: file is missing")
            self._ple = GGUFFile(self.path / "ple-fp8.gguf")
        return self._ple
