"""Readers for the on-disk format of the Edge0 latent sparse MoE models.

NumPy-only (no MLX, no network): ``ggml_quants`` decodes the ggml block
formats the packs use (NVFP4, Q4_0, Q8_0, BF16, FP8 E4M3), ``gguf_lite`` reads
GGUF v3 headers lazily, and ``pack`` opens a model folder (experts.bin,
labels.bin, latent dense D/U, skeleton) with strict size validation. Import the
submodules explicitly; nothing is imported eagerly here.
"""
