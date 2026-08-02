# Architectures

## Supported today

- Qwen3-Next-MTP with an MTPLX runtime contract
- DeepSeek V3 MTP — shipped experimental native backend (registry `experimental-native-contract-gated`); loads verified-contract artifacts, per-model QA still gates promotion
- DeepSeek-V4-Flash (`model_type: deepseek_v4`) — experimental native AR backend, new this cycle (registry `experimental-native-ar-only`); optional single-block MTP engages when the checkpoint carries `mtp.0.*` weights

## Recognized but not yet runnable

- Llama-MTP
- generic MTP layouts — pending tier (registry `recognized-backend-pending`), not a hard reject

The registry should tell users why a model is rejected and which release track is expected to support it.
