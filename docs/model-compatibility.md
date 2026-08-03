# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Loads and runs, labeled unverified (regenerate provenance to clear the label) |
| AR-only | An exact architecture-specific AR loader is installed, but the checkpoint has no MTP head | Run only with target-only AR selected |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer; experimental contract-gated backends exist for several of these families (DeepSeek V3/V4, GLM, MiMo, Nemotron-H, Step3.5, Hy-V3) |
| No MTP | No MTP head detected | Exit with a clear message |

The AR-only tier is narrow by design. It currently recognizes the exact
mixed-precision geometry and storage map of `mlx-community/Laguna-S-2.1-oQ4e` at
revision `8e3f5cad513746264940c1c4195de48d7ea345a5`. Local cache admission also
requires the pinned source marker, all 13 shards at their reviewed sizes, the
index, tokenizer, generation config, special tokens map, and Poolside chat
template. Other Laguna variants — including the earlier uniform-4bit build —
remain blocked until they have their own construction-time validation and
runtime evidence.
