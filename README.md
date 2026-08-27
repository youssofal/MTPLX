<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# MTPLX-RAMP

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

</div>

MTPLX-RAMP is a fork of [MTPLX](https://github.com/youssofal/MTPLX), the
Apple-Silicon inference server built by
[Youssof Altoukhi](https://github.com/youssofal) that runs a model's own
multi-token-prediction heads as an exact speculative decoder. Everything MTPLX
does, this fork does, unchanged. It adds one feature, off by default: **RAMP**
("Retrieval-Augmented Multi-token Prediction"), a fixed long-block policy plus a
mismatch-tolerant fuzzy re-anchor for MTPLX's built-in context-copy
(prompt-lookup) drafter. RAMP targets one specific bottleneck: when a coding
agent has to re-emit a file that is already in its prompt with an edit applied,
the stock drafter proposes short blocks and stops firing at every point where
the edit diverges from the original. On that workload shape RAMP measured
**+53.9 % median decode throughput at ~800 tokens of context** and
**+45.9 % at 128K**. On genuinely open-ended work — reviews, explanations,
new prose — it measured **between −2.0 % and +1.1 %**, i.e. no benefit, which
is why it ships off by default and stays scoped to edit-shaped turns.

Everything below the RAMP sections describes stock MTPLX behaviour, which this
fork inherits as-is.

## What RAMP is

MTPLX already ships a context-copy drafter: when the tail of the generated
stream matches an n-gram that occurs in the prompt, the prompt's continuation is
proposed verbatim as a block and verified in one forward pass, so the MTP head
is skipped for that cycle. Two properties of that drafter limit it on edit
work:

- **Block length is chosen from a confidence ladder** (`8, 12, 16, 24, 32`
  tokens, keyed on how far the suffix match extends backwards).
- **The lookup key is an exact `ng_min`-gram** (6 tokens by default): only the
  trailing 6 tokens of the generated stream. A divergence inside that trailing
  window — a renamed identifier, a small diff right at the cursor — makes the
  key miss; earlier divergence further back just shortens the backward-match
  extension without causing a miss.

RAMP changes those two things, plus one supporting change to stay consistent
with them (widening the block cap so a fixed 48-token block is not silently
re-clamped by the stock 24-token cap — see the trap noted below), all inside
`mtplx/context_copy.py` and nowhere else:

1. **Fixed long block.** `block_for_ext()` returns a fixed length (default 48)
   instead of the ladder value. The reason is measured, not assumed: the tiled
   attention kernel this engine hits at longer blocks reads the KV cache at a
   cost roughly equivalent to 8-10 single-row passes regardless of block
   length, not per-row, so a longer block commits far more tokens per pass even
   though a larger fraction of each block is rejected (marginal cost does rise
   for very large blocks, it just rises slower than block length). Backward-match
   extension turned out to be a poor predictor of how long a block should be —
   better retrieval on the ladder gained +7.0 % offline, while stock retrieval
   with a fixed 48-token block gained +42.7 %.
2. **Fuzzy re-anchor fallback.** When the exact index misses, `NgramIndex.find()`
   falls through to `_RampFuzzyAnchor`, which anchors on a shorter n-gram
   (default 3 tokens) and ranks candidate positions by backward similarity
   against the prompt over a 24-token window. A context that diverges in one
   token can still anchor.

Both live behind `MTPLX_RAMP_ENABLED`. With it unset, `_RampFuzzyAnchor` is
never even constructed and every path reduces to the upstream code.

The feature needed no engine patch. `generate_mtpk` imports its drafter symbols
function-locally on every call, so replacing the proposer is a matter of
rebinding module attributes — a seam confirmed by running a fully patched server,
not by reading. One trap found by execution: `block_for_ext(ext, k_cap)` clamps
against `context_copy_block_k()`, whose stock cap of 24 silently re-clamps a
longer block, so both must move together.

**Where it helps:** file-edit, refactor, and apply-diff turns — output that
largely already exists verbatim in the context.

**Where it does not:** open-ended chat, code review, explanation, reasoning, and
new-code generation. On two real open-ended tasks measured at ~24K context,
RAMP's exact n-gram index produced **zero hits in 1 544 probes**; the dark
fraction (probes that found nothing at all) was **0.923** on a code review and
**0.856** on a multi-file explanation. Everything RAMP proposed there came from
the fuzzy fallback, at 2.5 %–4.5 % block acceptance. The copy path supplies only
**1.3 %–3.1 % of emitted tokens** on that work in *either* arm, so even a perfect
proposer could not matter.

## Measured results

Full evidence trail, including the rounds that were wrong and got corrected:
[`docs/ramp/`](docs/ramp/).

### Throughput

| context | task shape | stock | RAMP (block=48) | delta | source |
|---|---|---|---|---|---|
| ~800 tok | mechanical edit (3 cases, 5 reps, interleaved A/B) | 138.2 t/s | 212.6 t/s | **+53.9 %** | `POC-FINDINGS.md` §5 |
| ~800 tok | same case as the 128K run | 115.93 t/s | 175.44 t/s | **+51.3 %** | decision 008 |
| ~24K | mechanical control | 102.70 t/s | 116.30 t/s | **+13.2 %** | decision 009, round 2 |
| ~24K | code review (open-ended) | 31.95 t/s | 31.31 t/s | **−2.0 %** | decision 009, round 2 |
| ~24K | multi-file explanation (open-ended) | 31.23 t/s | 31.59 t/s | **+1.1 %** | decision 009, round 2 |
| 128K | mechanical edit | 57.40 t/s | 83.72 t/s | **+45.9 %** | decision 008, round 1 |

The 128K figure is a pooled median over 6 requests per arm (+44.2 % cold on an
n=1 pair, +46.2 % warm on n=5 per arm, warm relative spreads 3.4 % and 1.7 %).
The mechanism behind it: RAMP lands 6.6 % more accepted tokens per response
(618 vs 580) while spending **37 % fewer verify passes** (19 rounds vs 30). Its
per-block acceptance is *lower* (0.678 vs 0.890) — that is the trade, not a
defect.

Decision 009's earlier round disagreed in sign on the open-ended cells
(−6.2 % and −4.5 % against round 2's −2.0 % and +1.1 %). The supported claim is
therefore **no benefit**, not reliable harm. What both rounds agree on is the
contrast: the same build, same machine, same round, is worth +11 %–+13 % on the
mechanical cell and nothing on the open-ended ones.

One observation is recorded without an explanation: RAMP's advantage on its own
best task is **not monotonic in context length** (+51.3 % at ~800 tokens,
+13.2 % at ~24K, +45.9 % at 128K). No model in this project predicts that.

### Output identity

At ~800 tokens and at 128K, on mechanical edit tasks, temperature-0 output was
byte-identical between arms — the same sha256 across all 21 requests of the 128K
A/B, and across every block length tried in both sweeps (16, 24, 32, 48, 64, 96).

**That property does not generalise.** On the two open-ended cells the arms
produce *different* temperature-0 output, deterministically and reproducibly
across independent rounds (word-level similarity 0.29 and 0.39). Chasing it down
found that engine session history moves the output too, within the stock arm
alone. So the honest statement is not "RAMP corrupts output" but: on this engine,
temperature-0 output on open-ended prompts is not stable against the kinds of
execution-path perturbation actually tested here (RAMP vs. stock, and session-
history changes within the stock arm alone), and byte-identity is only a
meaningful gate on
copy-shaped tasks. The unverified hypothesis is that verify-batch width changes
reduction shapes and flips argmax on near-ties. No unverified speculative token
reaches output either way — the target model still verifies every token.

### What was proposed and killed

RAMP was proposed with four mechanisms. Three were measured and rejected in the
POC ablation (offline replay of real captured traces, aggregated over three
cases):

| killed mechanism | measured | why it deserved it |
|---|---|---|
| index the model's own generated output | **−0.7 %** | the mechanism the name "retrieval-augmented" pointed at; combined with the rest it is actively harmful (+3.0 % with it vs +31.7 % without) |
| multi-candidate consensus ranking | **+0.0 %** | identical to baseline on every counter — with an exact 6-gram key the candidate set is degenerate, there is nothing to rank |
| adaptive block length from candidate agreement | **−52.4 %** | achieved the *best* per-block acceptance measured (0.968) and the worst throughput; optimising acceptance rate while shrinking blocks trades the metric that pays for the metric that flatters |

The shipped feature is the narrow survivor. The evidence directory also keeps an
invalidated block sweep, an invalidated 128K round, an excluded open-ended round,
and three consecutive long-context modelling rounds that each found a real error
in the previous one — including a proposed production fix ruled `INVALID_CARD`
because it would have applied to zero shipped configurations. See
[`docs/ramp/README.md`](docs/ramp/README.md) for the reading order.

### Known hazards

- **The EMA-suspend guard is live on open-ended work.** The engine suspends
  drafting when per-block acceptance EMA falls below 0.35, once at least 4
  context-copy blocks have been seen; a suspension applies backoff and resets
  the EMA state. Longer blocks lower that ratio. At 128K on the 21 mechanical
  requests it never fired; on open-ended work the stock ladder suspends 4–12
  times per 4 requests and RAMP 16–20. That guard is what limits the damage on
  the workloads RAMP does not suit. Recorded in
  [`docs/ramp/006-ramp-ema-guard-hazard-and-block-length.md`](docs/ramp/006-ramp-ema-guard-hazard-and-block-length.md);
  not fixed.
- **Block length 48 is pinned on A/B evidence, not on a clean sweep.** Blocks
  16, 24, 32, 64, and 96 were tried at various points, but the clean 32/48/64/96
  sweep needed to compare them on equal footing was aborted under
  memory-pressure contamination and never re-run cleanly. The one offline data
  point that did survive suggests fixed 64 may beat fixed 48 (+57.5% vs
  +42.7%), but that was never validated at long context. 48 is the best
  configuration *cleanly measured end-to-end*, not a proven optimum.
- **Untested:** any model other than Qwen3.8-27B (MTPLX Optimized Quality) and
  any machine other than the M5 Max used for every measurement here; 256K
  context; the append-only agent-harness shape (the real target workload,
  unmeasured across two decision records); tool-calling loops; concurrent
  requests; streaming; a clean decision-quality comparison across block
  lengths other than 48; and `fuzzy=False`.
- **No power or bandwidth counter data.** `powermetrics` needs root and was never
  captured.

## Install

RAMP only exists in this fork. The upstream Homebrew tap, the PyPI `mtplx`
package, and the mtplx.com DMG all install **stock MTPLX without RAMP**. The
only way to get RAMP is to install from this repository's source.

```bash
git clone https://github.com/johninthewinter/MTPLX-RAMP.git
cd MTPLX-RAMP
python3 -m pip install -e ".[dev,server]"
mtplx doctor
```

Requirements are the same as stock MTPLX: Apple Silicon (M1 or newer),
macOS 14+, Python 3.11+, and enough memory and disk for the model you pick.
`mtplx doctor` and `mtplx inspect` run without MLX; generation and serving need
MLX and a verified model.

Then start the server with RAMP enabled:

```bash
MTPLX_RAMP_ENABLED=1 MTPLX_RAMP_BLOCK=48 MTPLX_RAMP_FUZZY=1 mtplx start
```

`mtplx serve --port 8000` works the same way if you want the API server without
the interactive picker. Starting without the environment variables gives you
stock behaviour.

### RAMP settings

| Env var | Default | Meaning |
|---|---|---|
| `MTPLX_RAMP_ENABLED` | off | Master switch. Off = byte-identical to stock; the fuzzy index is not constructed. |
| `MTPLX_RAMP_BLOCK` | `48` (when enabled) | Fixed proposal length in tokens. `0` keeps the stock confidence ladder. |
| `MTPLX_RAMP_FUZZY` | on (when enabled) | Mismatch-tolerant fallback. Only validated as a net win **combined with** a long block — the short-block combination was only measured in an invalidated, contaminated sweep, so treat it as unverified rather than confirmed-bad. Do not enable with a short or zero block. |
| `MTPLX_RAMP_ANCHOR_LEN` | `3` | Fuzzy anchor length in tokens (clamped to at most `ng_min - 1`). |
| `MTPLX_RAMP_MAX_FUZZY_CANDIDATES` | `8` | Ranked candidate cap. The search itself scans up to 8x this many stored positions before ranking down to the cap. |
| `MTPLX_RAMP_SIMILARITY_SPAN` | `24` | Backward-similarity window for ranking fuzzy candidates. |

`tests_ramp/verify_ramp_equivalence.py` is a standalone check (no pytest) that
(a) RAMP-off reduces byte-for-byte to stock and (b) RAMP-on reproduces the
measured win direction on the three real committed traces in
`tests_ramp/fixtures/`.

---

# MTPLX

The rest of this document describes stock MTPLX, which this fork inherits
unmodified.

MTPLX is a native Mac app and a command line for running local language models with multi-token prediction. Modern models like Qwen 3.5/3.6/3.8 ship with built-in MTP heads. Almost no runtime uses them. MTPLX does: the model drafts several tokens ahead of itself, verifies each drafted block in a single batched forward pass, and commits tokens through exact rejection sampling with residual correction. Same model, same output distribution, measured 1.6x faster on a 16 GB M4 Mac mini and 2.24x on an M5 Max.

There is no second draft model eating your RAM, and no greedy shortcut that quietly changes what the model would have said at real sampling settings. The acceptance math is the Leviathan and Chen rejection sampling theorem with residual correction, so `temperature=0.6, top_p=0.95` behaves exactly like normal decoding, just faster.

## Get it

Every install path in this section installs **stock MTPLX, without RAMP**. For
RAMP, use the source install above.

**The Mac app** is the easiest way in. Download the DMG at [mtplx.com](https://mtplx.com/download), drag it to Applications, and the app takes care of everything else: it checks your hardware, recommends a model that actually fits your memory, downloads it, sets up its own Python engine (no Homebrew needed), installs fan control, puts `mtplx` on your PATH, and then measures your machine to pick the fastest decoding depth.

**Recommended for coding:** Qwen 3.8 27B Optimized Speed is a 4-bit dynamic
quant with great coding speeds and good quality. Its two siblings sit right
under it in the app and CLI: Bare Speed (quickest burst chat speeds, lower
quality and slower on long coding tasks) and Optimized Quality (8-bit dynamic
quant, good coding speeds and perfect quality). Qwen 3.6 Optimized Speed V2
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

## The app

<img src="docs/assets/readme/app-dashboard.jpg" alt="MTPLX dashboard with live decode gauge" width="100%" />

The dashboard shows what your model is doing while it does it: live tokens per second, acceptance rate by draft depth, the verify waterfall, cache state, and system pressure. When you start a chat, code an agent against the local server, or run a benchmark, the numbers are right there.

<img src="docs/assets/readme/app-chat.jpg" alt="Chat streaming with live speed badge" width="100%" />

Chat is native, streams with thinking cards, takes file attachments, and can search the web. One click launches OpenCode, Pi, Hermes, Open WebUI, or anything else that speaks the OpenAI or Anthropic API against your local server. There is also a built-in AIME benchmark runner with fully disclosed, coaching-free prompts, so you can score a model yourself instead of trusting a chart.

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

The official catalog lives on Hugging Face under [Youssofal](https://huggingface.co/Youssofal): Qwen 3.8 27B (Bare Speed, Optimized Speed, Optimized Quality, each with an FP16 build for M1 and M2), Qwen 3.6 (27B, 35B MoE) in speed and quality builds (the 35B MoE adds a balance build), Qwen 3.5 (4B, 9B), plus Gemma 4. The app and the CLI recommend from these based on your hardware.

## The server

`mtplx start` (or the app's play button) serves an OpenAI-compatible API on `127.0.0.1:8000`: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, the optional `/v1/embeddings` and `/v1/rerank` (see below), plus an Anthropic-compatible `/v1/messages` with streaming, tool calls in both styles, `/health`, and `/metrics`. Claude Code, Cline, Continue, Open WebUI, curl, the openai and anthropic Python clients: if it speaks the API, it works. The app and CLI share one server, so `mtplx start` attaches to the app's running model instead of loading a second copy.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Sessions survive: a warm-prefix session bank keeps multi-turn chats fast, and a default-on SSD session cache restores sessions near-instantly across restarts (disable with `--ssd-session-cache off`).

### Embeddings and reranking

The same daemon can serve retrieval models, so a RAG or agent-memory setup does not need a second inference server beside MTPLX. Point it at an MLX embedding model, or a reranker with a Qwen-style yes/no tokenizer (plus specifically-supported Jina checkpoints) — Hugging Face id or local path, optionally with a `REF=served-id` alias:

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
with mathematically exact speculative sampling — 27 April 2026, before
llama.cpp had MTP at all, and months before it reached the hybrid GDN family.
The dated record, with a public receipt for every claim, is in
[HISTORY.md](HISTORY.md) and at [mtplx.com/history](https://mtplx.com/history/).

## Credit and license

**This repository is a fork of [MTPLX](https://github.com/youssofal/MTPLX), built
by [Youssof Altoukhi](https://github.com/youssofal).** All of the inference
engine, the app, Forge, the server, the tuner, and the MTP speculative decoding
math are his work. This fork's only addition is RAMP, contained in
`mtplx/context_copy.py` and marked there with an Apache-2.0 section 4(b)
modification notice. Bug reports about stock MTPLX behaviour belong upstream, at
[youssofal/MTPLX/issues](https://github.com/youssofal/MTPLX/issues).

Licensed under Apache-2.0. Keep the [LICENSE](LICENSE) and the [NOTICE](NOTICE)
file if you redistribute.

**Attribution is required, in-product.** This is MTPLX's own NOTICE
requirement, not something Apache-2.0 mandates on its own — §4(d) only
requires that redistributions carry the NOTICE file, not that its contents be
shown in-product. The in-product obligation comes from what this project's
NOTICE itself says:

> This NOTICE file is part of the Apache License 2.0 terms for MTPLX (see
> section 4(d) of the LICENSE). Any product, application, service, or
> distribution that includes, embeds, or is built on MTPLX, in whole or in
> part, modified or unmodified, must display the following attribution within
> the product itself, in a place a user of that product can see (for example an
> About screen, a credits or acknowledgements screen, a settings or help page,
> documentation shipped with the product, or the startup banner of a command
> line tool):
>
> ```
>   Powered by MTPLX
>   https://github.com/youssofal/mtplx
> ```
>
> Attribution in a source repository, a README, or a marketing page alone does
> not satisfy this requirement. The words "Powered by MTPLX" must appear
> in-product. The link is required wherever the display medium supports it.
>
> Public benchmarks, articles, and research that use or build on MTPLX should
> credit "MTPLX by Youssof Altoukhi" with the same link.

This README does not satisfy 4(d) on its own. Any distribution of this fork must
surface "Powered by MTPLX" and its link inside the product — the CLI startup
banner and the app's About screen are the two surfaces that need it. That is an
outstanding engineering task in this fork, tracked separately from this document.

MTPLX builds on [MLX](https://github.com/ml-explore/mlx) and the Qwen and Gemma model families; the speculative sampling math follows Leviathan and Chen (2023). Fan control via [ThermalForge](https://github.com/ProducerGuy/ThermalForge). Model weights remain governed by their upstream licenses.
