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

The external AR route is reserved for
`philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly` at immutable
revision `ac33e4f3ca3546e6cec104558d42161e15814e33`. Admission requires its
DeepSeek V4 target-only configuration, all 44 exact weight shards, required
sidecars, and the closed safetensors index; cached content is hash-checked and
a same-size corrupt file is repaired through an atomic re-download. MTPLX then
executes the separately installed `mlx-serve` binary. The required zero
`num_nextn_predict_layers` / zero `dspark_block_size` contract means an MTP or
DSpark artifact is not silently routed here. The external runtime's memory
preflight remains enabled. Its streaming and throughput are unapproved.

## Embedded MTP heads and third-party loaders (#306)

An MTPLX-branded pack stores its MTP head as a standalone `mtp.safetensors`
sidecar. Do not brand or redistribute an artifact that keeps `mtp.*` tensors
embedded in the trunk shards with absolute norm gains: mlx-lm's qwen3.5-family
loader keys its +1.0 delta-norm restoration on the bare presence of those
keys, so it shifts every trunk RMSNorm of an already-absolute checkpoint a
second time. The model still loads and generates — with acceptance collapsed
to a few percent — so it benchmarks as "MTPLX models are slow" instead of
failing. MTPLX's own loader refuses such a trunk at load with the cause named
(the q-norm mean lands near 2.79 against a healthy 1.74–1.83 band). Rebuild
the pack through `mtplx forge`, which extracts the head into the sidecar and
decides the norm convention once per tensor set.
