*(v0.1-era research snapshot — see docs/releases/ for the current state.)*

# Native MTP On MLX

MTPLX explores the built-in MTP heads in Qwen3-Next models on Apple Silicon.

The core idea is straightforward: use the model's own MTP head to propose tokens, then use exact speculative sampling to accept or reject them against the target distribution. This is different from greedy prefix-match systems and different from external-draft-model systems.

The v0.1 release is honest about its current boundary: the cold path is strong, while sustained no-fan long-context throughput still needs kernel and dispatch work.
