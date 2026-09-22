<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# The fastest way to run Qwen 3.8 on a Mac.

[![PyPI](https://img.shields.io/pypi/v/mtplx?label=PyPI)](https://pypi.org/project/mtplx/)
[![CI](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml/badge.svg)](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

</div>

MTPLX is a native Mac app and a command line that runs local language models on Apple Silicon with the model's own multi-token prediction (MTP) heads. It runs Qwen 3.8 Flash Next, the 125B mixture of experts, and Qwen 3.8 27B, plus Qwen 3.6, Qwen 3.5 and Gemma 4. The model drafts several tokens ahead of itself, one batched forward pass verifies the draft, and tokens are committed through exact rejection sampling with residual correction. The output distribution is the model's own at any temperature, and decode runs at around twice the speed of plain decoding: measured 1.6x on a 16 GB M4 Mac mini and 2.24x on an M5 Max.

## Measured speeds

Every number below was measured on a MacBook Pro M5 Max with 128 GB, fans verified at maximum, sampled at the model's own settings. Conditions and sources for every row, with the raw logs where they exist, are at [mtplx.com/benchmarks](https://mtplx.com/benchmarks/).

| Model | Speed | Run |
|---|---|---|
| Qwen 3.8 Flash Next, Optimized Speed | 125.8 tok/s | one OpenCode request: 1,301 tokens generated, 18,539-token prompt with 18,364 tokens served from cache, MTP depth 3, MTPLX 2.11.3, 16 September 2026 |
| Qwen 3.8 Flash Next | 79.3 tok/s | 9k-token code prompt, 1,500 tokens generated, thinking off, two alternating boots each, MTPLX 2.11.3 (62.5 on 2.11.2) |
| Qwen 3.8 Flash Next | 61.8 tok/s | 109k-token OpenCode turn, mean of two runs, MTPLX 2.11.3 |
| Qwen 3.8 Flash Next | 50.3 tok/s | 200k-token OpenCode turn, warm, mean of two runs, MTPLX 2.11.3 |
| Qwen 3.8 27B, Optimized Speed | 87.6 tok/s | rewriting a file it just wrote, stock settings, MTPLX 2.10.0 |
| Qwen 3.8 27B, Bare Speed | 65.2 tok/s | fresh coding task at official Qwen 3.8 sampling, generation to the model's own stop, MTPLX 2.7.0 |
| Qwen 3.6 27B, Optimized Speed | 81.74 tok/s | the 27B record on a fresh generation: 192-token bench, thinking off, temperature 0.6, twin runs, 2.69x over 30.37 plain decode, 2 July 2026, [raw logs](https://mtplx.com/benchmarks/receipts/2026-07-02-record/) |
| Qwen 3.5 4B, Optimized Speed | 227.8 tok/s | depth 3, 1.71x over 133.6 plain decode, MTPLX 2.2.0 |

The speed comes with the output unchanged. On every release MTPLX draws a thousand four-token samples from the fast path and a thousand from the plain path at temperature 1, top-p 0.95, top-k 20, and compares the two distributions by token id. On 2.11.3 they match within the plain path's own noise on both Flash Next and the 27B Quality pack. The acceptance rule is the Leviathan and Chen rejection sampling theorem with residual correction, so `temperature=0.6, top_p=0.95` behaves exactly like normal decoding, just faster. There is no second draft model eating your RAM, and no greedy shortcut that quietly changes what the model would have said.

How MTPLX compares with mlx-serve, oMLX, LM Studio, Ollama, llama.cpp and mlx-lm, with a version, a machine and a date on every number: [mtplx.com/compare](https://mtplx.com/compare/).

## Get it

**The Mac app** is the easiest way in. Download the DMG at [mtplx.com](https://mtplx.com/download), drag it to Applications, and the app takes care of everything else: it checks your hardware, recommends a model that actually fits your memory, downloads it, sets up its own Python engine (no Homebrew needed), installs fan control, puts `mtplx` on your PATH, and then measures your machine to pick the fastest decoding depth.

**Recommended for coding:** Qwen 3.8 27B Optimized Speed is a 4-bit dynamic
quant with great coding speeds and good quality. Its two siblings sit right
under it in the app and CLI: Bare Speed (quickest burst chat speeds, lower
quality and slower on long coding tasks) and Optimized Quality (8-bit dynamic
quant, good coding speeds and perfect quality). On a Mac with 96 GB or more,
Qwen 3.8 Flash Next Optimized Speed is the pick. Qwen 3.6 Optimized Speed V2
remains available directly below them.

**The CLI** on its own:

```bash
brew install youssofal/mtplx/mtplx
mtplx start
```

or `python3 -m pip install mtplx` if you prefer pip. All releases are listed at [mtplx.com/releases](https://mtplx.com/releases/).

Requirements: Apple Silicon (M1 or newer), macOS 14+. 16 GB of memory runs the
4B and 9B models comfortably. Qwen 3.8 Optimized Speed is recommended on Macs
with 32 GB or more; on M1 and M2 the app and CLI pick its FP16 build (same
weights, native precision for those chips) automatically. Both check your Mac
before recommending anything.

## Qwen 3.8 Flash Next on a Mac

Qwen 3.8 Flash Next is Qwen's 125B-A6B preview of the Qwen4 architecture: a hybrid GatedDeltaNet mixture of experts with Qwen Sparse Attention and a 51B-parameter n-gram table. MTPLX 2.10.0 was the first Apple Silicon backend for the family, and it runs the model's own MTP head as an exact speculative decoder. Two packs, both for Macs with 96 GB of unified memory or more:

- `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`: dynamic 4-bit with the sparse-attention projections at 8-bit. The recommended build. 115.1 GB download including the 32 GB n-gram table, about 83 GB resident.
- `Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed`: flat 4-bit, the quickest build. 106.3 GB download, about 74 GB resident.

The n-gram table streams from SSD by default, so the weights stay resident and the table does not have to. Context window 262,144 tokens; 261,120-token prompts decode on 2.11.3. Image input works. In the app, pick "Qwen 3.8 Flash-Next Optimized Speed"; from the terminal:

```bash
mtplx serve --model Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed
```

Then point OpenCode, Pi, Hermes, Claude Code, Cline, Cursor or anything that speaks the OpenAI or Anthropic API at `http://127.0.0.1:8000`. The guide with every measured number and its conditions: [mtplx.com/models/qwen3.8-flash-next](https://mtplx.com/models/qwen3.8-flash-next/).

## Qwen 3.8 27B on a Mac

Qwen 3.8 27B is the coding flagship for Macs with 32 GB or more. MTPLX shipped it on 15 August 2026, the day after Qwen released it. Three packs, each with an FP16 sibling for M1 and M2 that the app and CLI pick automatically:

- `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`: 4-bit dynamic, 20.4 GB, 23.6 GB peak. The default for coding. Against the bf16 model on a mixed corpus of code, prose and JSON it agrees on 96.0 percent of top-1 tokens with a KL divergence of 0.012.
- `Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed`: flat 4-bit, 16.0 GB, 17.0 GB peak. Quickest chat speeds.
- `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality`: 8-bit dynamic, 29.4 GB, 32.7 GB peak, for 36 GB or more. 99.3 percent top-1 agreement with bf16, KL 0.0005.

The guide: [mtplx.com/models/qwen3.8-27b](https://mtplx.com/models/qwen3.8-27b/).

## Models and recommended settings

Every model here is an official MTPLX pack on Hugging Face under [Youssofal](https://huggingface.co/Youssofal). "Fits" is the memory the pack actually peaks at while serving, next to the smallest Mac the app and CLI will offer it on. The preset column is what MTPLX resolves by itself, so this table is what you get by doing nothing.

| Model (`Youssofal/...`) | Fits | What it is for | Preset |
|---|---|---|---|
| `Qwen3.5-4B-MTPLX-Optimized-Speed` | 8 GB and up, peaks at 2.9 GiB | 4-bit. The fastest fit for smaller Macs. | Sustained, depth 3 |
| `Qwen3.5-4B-MTPLX-Optimized-Quality` | 8 GB and up, peaks at 4.8 GiB | 8-bit. The highest-fidelity 4B. | Sustained, depth 3 |
| `Qwen3.5-9B-MTPLX-Optimized-Speed` | 16 GB and up, peaks at 10.0 GiB | 6-bit. The strong small-Mac speed pick. | Turbo. Tuning this one on a 16 GB M4 Mac mini lands on depth 1 |
| `Qwen3.8-27B-MTPLX-Bare-Speed` | 32 GB and up, peaks at 20.0 GiB | Quickest burst chat speeds. Lower quality and slower on long coding tasks. | Turbo, depth 3 |
| `Qwen3.8-27B-MTPLX-Optimized-Speed` | 32 GB and up, peaks at 25.0 GiB | 4-bit dynamic quant. Great coding speeds and good quality. The recommended coding model. | Turbo, depth 3 |
| `Qwen3.8-27B-MTPLX-Optimized-Quality` | 36 GB and up, peaks at 33.0 GiB | 8-bit dynamic quant. Good coding speeds and perfect quality. | Turbo, depth 3 |
| `Qwen3.8-27B-MTPLX-Bare-Speed-FP16` | 32 GB and up, peaks at 20.0 GiB | The Bare Speed pack for M1 and M2: same weights, every 16-bit tensor cast to fp16. | Turbo, depth 3 |
| `Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` | 32 GB and up, peaks at 25.0 GiB | The recommended coding model for M1 and M2. | Turbo, depth 3 |
| `Qwen3.8-27B-MTPLX-Optimized-Quality-FP16` | 36 GB and up, peaks at 33.0 GiB | The Optimized Quality pack for M1 and M2. | Turbo, depth 3 |
| `Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` | 96 GB and up, peaks at 87 GiB resident | The 125B MoE, dynamic 4-bit with 8-bit attention. Its 32 GB n-gram table streams from SSD instead of taking RAM. 125.8 tok/s on an OpenCode request on an M5 Max. | Turbo, depth 3. This family accepts up to depth 5 |
| `Gemma4-MTPLX-Optimized-Speed` | 32 GB and up, peaks at 18.0 GiB | High quality, moderate speeds. Runs as an assistant pair, so the tuned control is the draft block size rather than depth. | Sustained |
| **What the author runs** | M5 Max, 128 GB | Flash-Next Optimized Speed, for everything | Turbo, depth 3 |

Depth 3 is the launch default. `mtplx tune --retune` measures autoregressive decoding against each depth on your own Mac and saves a shallower one when a shallower one wins, which is why the 9B row above is depth 1 on a Mac mini. M1 and M2 Macs are offered the FP16 builds and every other Mac the bf16 parents, so you never pick the precision by hand. The "fits" numbers are peak serving memory, not download size, and they are the same numbers MTPLX checks your Mac against before it offers you anything.

Qwen 3.6 is still published and still supported: 27B in speed and quality builds, and the 35B MoE in speed and balance builds. The 3.8 packs above replaced it as the default recommendation, and the app and CLI still list the 3.6 packs below them.

## The app

<img src="docs/assets/readme/app-dashboard.jpg" alt="MTPLX dashboard with live decode gauge" width="100%" />

The dashboard shows what your model is doing while it does it: live tokens per second, acceptance rate by draft depth, the verify waterfall, cache state, and system pressure. When you start a chat, code an agent against the local server, or run a benchmark, the numbers are right there.

<img src="docs/assets/readme/app-chat.jpg" alt="Chat streaming with live speed badge" width="100%" />

Chat is native, streams with thinking cards, takes file attachments, and can search the web. One click launches OpenCode, Pi, Hermes, Open WebUI, or anything else that speaks the OpenAI or Anthropic API against your local server. There is also a built-in AIME benchmark runner with fully disclosed, coaching-free prompts, so you can score a model yourself instead of trusting a chart.

### Model libraries

You can keep every model in one folder or configure several ordered folders in
Settings under Model libraries. The primary folder receives downloads and
Forge output. Additional folders are discovery roots, so external drives and
existing collections stay in place. MTPLX remembers unavailable volumes and
uses the first complete copy it finds in folder order.

The CLI uses the same primary plus additional-root policy:

```toml
# ~/.mtplx/config.toml
model_dir = "/Volumes/Models/MTPLX"
model_dirs = ["/Volumes/Model Archive", "/Users/me/Models"]
```

`model_dir` is the writable primary root. `model_dirs` are ordered discovery
roots. You can also repeat `--model-search-dir` on supported commands or set
`MTPLX_MODEL_DIRS` to a colon-separated path list on macOS and Linux. Pulls and
Forge builds always write to the primary root. CLI updates and removals in an
additional root require selecting that root explicitly with `--cache-dir`.

## Auto-tune

The right draft depth depends on your specific Mac: chip, memory bandwidth, thermals. During onboarding (and any time after), MTPLX runs the real model on your machine at each depth, with fans pinned for clean timing, and keeps autoregressive decoding as the baseline. If an MTP depth beats it, that depth is saved. If nothing beats the baseline, nothing is saved and the app says so. From the terminal it is one command:

```bash
mtplx tune --model <model-or-path> --retune
```

On a 16 GB M4 Mac mini, tuning the 9B model lands on depth 1: 14.4 tok/s baseline becomes 23.0 tok/s.

## Forge: make your own MTP models

<img src="docs/assets/readme/app-forge.jpg" alt="Forge verifying a freshly built MTP model" width="100%" />

Forge takes a Hugging Face repo and turns it into an MTPLX-ready MTP model: convert to MLX, train the MTP adapter, verify that the result is actually faster and still exact, and publish back to the Hub if you want to share it. The honest part matters: Forge measures before and after on your hardware and shows you the verdict ("Depth 1 is fastest: 227.1 to 296.1, 1.30x") rather than assuming the adapter helped. Available in the app and as `mtplx forge` subcommands.

MTPLX does not support attaching a separately supplied MTP sidecar to an arbitrary MLX trunk. Matching architecture fields, tensor shapes, or provenance labels cannot prove that the head was trained against those exact trunk weights. Use a complete model that already includes its matching MTP weights, or use Forge to build and verify an artifact from its original source checkpoint.

The official catalog lives on Hugging Face under [Youssofal](https://huggingface.co/Youssofal): Qwen 3.8 Flash Next (Optimized Speed, Bare Speed), Qwen 3.8 27B (Bare Speed, Optimized Speed, Optimized Quality, each with an FP16 build for M1 and M2), Qwen 3.6 (27B, 35B MoE) in speed and quality builds (the 35B MoE adds a balance build), Qwen 3.5 (4B, 9B), plus Gemma 4. The app and the CLI recommend from these based on your hardware.

## The server

`mtplx start` (or the app's play button) serves an OpenAI-compatible API on `127.0.0.1:8000`: `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/models`, and the optional `/v1/embeddings` and `/v1/rerank` (see below), plus an Anthropic-compatible `/v1/messages` with streaming, tool calls in both styles, `/health`, and `/metrics`. The stateless, text-only Responses route provides Codex Responses compatibility with hosted tools disabled: it supports client-executed function/custom tools and namespace-grouped functions. Codex 0.146 sends `web_search` by default, and MTPLX intentionally returns a precise `400` for that request unless the hosted tool is disabled or removed. Codex may send `reasoning.effort: "xhigh"`; MTPLX resolves it against the loaded model. Qwen 3.8 preserves `xhigh`, Step 3.5 clamps it to `high`, and Qwen 3.6 exposes no reasoning-effort tier. Request observability records the requested and effective values plus whether the request was downgraded. Claude Code, Cline, Continue, Open WebUI, curl, the openai and anthropic Python clients: if it speaks the API, it works; the Responses hosted-tool limitation above still applies. The app and CLI share one server, so `mtplx start` attaches to the app's running model instead of loading a second copy.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Sessions survive: a warm-prefix session bank keeps multi-turn chats fast, and a default-on SSD session cache restores sessions near-instantly across restarts (disable with `--ssd-session-cache off`). A restored turn decodes as if it had never been paused: on 2.11.3 a seeded warm turn and the same turn recomputed cold were measured to part only on a literal coin toss, and a 96,760-token conversation restored in 8 ms in the app.

### Embeddings and reranking

The same daemon can serve retrieval models, so a RAG or agent-memory setup does not need a second inference server beside MTPLX. Point it at any MLX embedding or reranker model — Hugging Face id or local path, optionally with a `REF=served-id` alias:

```bash
mtplx serve \
  --embedding-model mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --reranker-model vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
```

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-Embedding-8B-4bit-DWQ","input":["hello","world"]}'

curl http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"where is the cache?","documents":["the cache lives in ~/.mtplx","unrelated text"]}'
```

Both flags repeat, so several models can be served at once and picked per request via `"model"`. Listing the same reference as both an embedder and a reranker loads **one** copy of the weights and serves both roles from it. Retrieval models load on first request and are capped by `--retrieval-max-resident` (default 2), which unloads the least recently used one beyond the cap — an unused endpoint costs nothing. `/v1/models` stays chat-only by default so chat clients that enumerate models never offer an embedder as a conversation target; list retrieval models with `?capability=embedding` or `?capability=rerank` (every entry carries its `capability`), and a chat completion that requests a retrieval id gets a clear 400 rather than a silent answer from the chat model.

These models do not go through the MTP path, and that is deliberate: multi-token prediction makes *next-token* decoding cheaper, which means nothing for a model that returns a vector instead of a token stream. Configure them in the app under Settings → Retrieval endpoints, or persist them in `~/.mtplx/config.toml` as `embedding_models` and `reranker_models`. With nothing configured the endpoints answer 404 and chat behaves exactly as before. One safety gate: checkpoints that bundle their own Python inference code (the jina embedding/reranker MLX releases do) are refused with a 403 until you opt in with `--retrieval-trust-remote-code` (or `retrieval_trust_remote_code = true` in the config file) — a model download never gains code execution just by being pointed at.

Sampler controls cover `temperature`, `top_p`, `top_k`, and the OpenAI penalty pair `presence_penalty` / `frequency_penalty` — per request, as server defaults (`--default-presence-penalty` / `--default-frequency-penalty` on `start`/`serve`/`quickstart`), or live via `mtplx settings set` and the app's Presence Penalty dial. Penalties default to 0, which is an exact no-op that preserves MTP exactness. Qwen's guidance: leave them at 0 for coding and agent work; ~0.5–1.5 presence penalty helps creative writing or when a model loops on itself.

Concurrent scheduler modes, ownership guarantees, and backend-specific
implementations are documented in [Concurrency modes](docs/concurrency.md).

## CLI quick reference

```bash
mtplx start                # interactive: pick model, mode, surface, then chat
mtplx serve --port 8000    # API server only
mtplx stop                 # stop the running server cleanly
mtplx pull <hf-repo>       # download a model safely
mtplx models               # what is cached, sizes, validation
mtplx inspect <model>      # compatibility report before anything runs
mtplx tune --retune        # measure AR vs D1/D2/D3 on your Mac
mtplx forge --help         # build, verify, and publish MTP models (probe/build/publish/verify subcommands)
mtplx bench aime --quick   # run the AIME benchmark from the terminal
mtplx doctor               # install and integration health
mtplx max --install        # fan control (one sudo prompt, crash-safe)
mtplx settings get/set     # read or change live server settings
```

Every command takes `--help`, and most inspection/diagnostic commands take `--json`. The CLI works without MLX installed for everything that does not need a model, so `doctor` and `inspect` run on any machine.

## Modes

| Mode | What it does | When |
|---|---|---|
| **Turbo** | NAX verify kernels + compiled verify; the default for the quantized 27B and 9B flagship models | Picked automatically for those models |
| **Sustained** | Default for all other models. Long-context MTP path with chunked prefill and request-sized KV | Everyday use, big files, 16K-200K prompts |
| **Sustained Max** | Sustained with fans pinned at 100% | Long work where you want maximum cooling |
| **Burst** | Legacy short-context benchmark lane, loud | Short prompts and benchmarks only |

Fan-backed modes restore your fans to automatic if MTPLX dies for any reason, including `kill -9` and closing the terminal. A detached watchdog handles it; this is verified on hardware, not assumed.

## The Splash engine

MTPLX can also serve its API over [Inco's Splash](https://inco.ai/blog/splash/),
an Apple-silicon engine that rebuilds itself around each model it supports —
Metal kernels compiled for that model's exact shapes and a DFlash 2 draft
trained for it:

```bash
brew install incoai/tap/splash
mtplx serve --engine splash --model incoai/Qwen3.8-27B-Splash
```

Same endpoints, same dashboard, same app: only the kernels change. Splash is
faster on the two packages it supports, and narrower everywhere else — it
loads only its own packages, and its KV cache is **fixed at 8-bit** because its
kernels read 8-bit KV directly. For 4-bit or unquantized KV, stay on the MLX
engine and use `--kv-quant`. `--engine splash` imports no MLX at all.

The dashboard works unchanged — live decode gauge, per-request prefill and
decode speeds, min/max/p95 — with every finished request's numbers taken from
Splash's own counters. What Splash cannot do is reported rather than faked:
MTP depth reads absent, and the app disables the KV control with the reason
attached. See [docs/splash-engine.md](docs/splash-engine.md).

## Compatibility, honestly

`mtplx inspect` classifies models before anything runs: verified, family-compatible but unverified, architecture-compatible but unverified, AR-only, incompatible architecture, or no MTP heads at all. Unverified models load with an explicit unverified label. There are no silent fallbacks: if MTPLX cannot run a model correctly, it tells you instead of running it badly.

[Laguna-S-2.1 oQ4e](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e) is supported through its exact MLX architecture in target-only AR mode:

```bash
mtplx start cli \
  --model mlx-community/Laguna-S-2.1-oQ4e \
  --download \
  --no-mtp
```

MTPLX pins that model to revision
`8e3f5cad513746264940c1c4195de48d7ea345a5` and verifies the 13-shard layout,
tokenizer, generation config, special tokens map, and Poolside chat template
before admitting it. The checkpoint has no native MTP head, so an MTP launch is
rejected before weights load instead of falling back during execution. The
weights occupy 59.72 GiB, a 64.13 GB snapshot on disk. The launch preflight
requires about 85 GiB of unified memory (weights, runtime headroom, and a
16 GiB system reserve) — in practice a 96 GB Mac; 128 GB is
comfortable. MTPLX defaults Laguna to a 32,768-token context
and response cap, and checks larger explicit server contexts against the active
Metal memory cap.

## What MTPLX is not

- Not an external-drafter system. The drafter is the target model's own MTP heads.
- Not a greedy-argmax trick. Acceptance is exact rejection sampling, correct at any temperature.
- Not a CUDA project. MTPLX is MLX-native and Apple Silicon first. For Linux, use vLLM.

## History

MTPLX was the first runtime on Apple Silicon to run a model's own MTP heads
with mathematically exact speculative sampling: 27 April 2026, before
llama.cpp had MTP at all, and months before it reached the hybrid GDN family.
The 27B record followed on 2 July 2026 (81.74 tok/s on Qwen 3.6 27B, raw logs
published), the first Apple Silicon backend for Qwen 3.8 Flash Next on
29 August 2026, and 125.8 tok/s on a Flash Next OpenCode request on
16 September 2026. The dated record, with a public source for every claim, is
in [HISTORY.md](HISTORY.md) and at [mtplx.com/history](https://mtplx.com/history/).

## License and credit

Apache-2.0: use it, modify it, ship it commercially. Keep the license and the [NOTICE](NOTICE) file if you redistribute.

**Attribution is required.** If you ship a product, app, or service that includes or is built on MTPLX, it has to say so inside the product itself, somewhere a user can see it (About screen, credits, settings, shipped docs, or a CLI startup banner):

> Powered by MTPLX
> https://github.com/youssofal/MTPLX

A mention in your repo or on your website does not cover it. The full terms are in [NOTICE](NOTICE), which Apache-2.0 section 4(d) carries with every copy.

MTPLX builds on [MLX](https://github.com/ml-explore/mlx) and the Qwen and Gemma model families; the speculative sampling math follows Leviathan and Chen (2023). Fan control via [ThermalForge](https://github.com/ProducerGuy/ThermalForge). Model weights remain governed by their upstream licenses.

Built by [Youssof Altoukhi](https://github.com/youssofal). Bug reports and benchmark replications welcome via [Issues](https://github.com/youssofal/MTPLX/issues).
