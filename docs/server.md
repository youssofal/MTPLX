# Server

The server target is OpenAI-compatible local serving, with Anthropic
Messages compatibility available for coding harness smoke tests.

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

See [Concurrency modes](concurrency.md) for scheduler selection, ownership
rules, and model/backend-specific implementations.

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses` (Codex Responses compatibility with hosted tools
  disabled; stateless text + client-executed function/custom/namespace tools)
- `POST /v1/completions`
- `POST /v1/messages`
- `GET /admin/sessions`
- `POST /admin/cache/clear`

## n-gram table pre-read (`--ngram-prewarm`, on by default)

Models with a streamed n-gram sidecar (the Qwen 3.8 Flash-Next family) keep a
29.8 GiB PLE table on SSD and gather rows from it through a memmap. Those rows
are hash-scattered, so a cold table means demand faults during decode -
serial, and flat at ~1.4 GiB/s however many threads touch them - while reading
the file sequentially runs at ~12 GiB/s. Cold sidecar rows measured 56 tok/s
against 68.8 tok/s warm, and the first prefill chunk stops being bimodal
(1.9 s vs 4.4 s on the same build, tracking nothing but table residency).

So the server pre-reads the table at model load, and logs what it decided:

```
[mtplx] n-gram table pre-read plan: mode=auto source=default table=29.8 GiB free=44.0 GiB reserved=8.0 GiB margin=6.0 GiB budget=29.8 GiB order=prefix
[mtplx] n-gram table pre-read 29.8 GiB in 2.5 s (12.1 GiB/s)
```

### How much

`--ngram-prewarm auto|all|off|<GiB>` (env `MTPLX_NGRAM_PREWARM`; the flag wins).

| value | meaning |
| --- | --- |
| `auto` (default) | `min(table, free − KV reservation − 6 GiB margin)` |
| `all` | the whole table, whatever the machine has left |
| `<GiB>` | a fixed budget, e.g. `--ngram-prewarm 12` or `12GiB` |
| `off` (or `--no-ngram-prewarm`) | serve at the as-found page-cache rate |

`auto` exists because the whole table usually does *not* fit. On a 128 GB Mac
MTPLX wires ~85 GB of weights; a 32 GB table plus the KV cache does not go in
what is left. (A 4-bit pack that wires ~68 GB can afford the full read, which
is why other servers get away with an unconditional pre-read.)

- **free** is `vm_stat` free + inactive + purgeable, read at the moment of the
  pre-read - i.e. after the weights are mapped, so it is real headroom rather
  than a boot-time guess. Darwin counts purgeable pages inside active/inactive
  too, so the number is slightly optimistic, and `speculative` is deliberately
  excluded because it is largely the page cache the pre-read competes with.
- **KV reservation** is the same arithmetic the server's `MemoryPlan` uses -
  `(dense KV bytes/token × quant factor + QSA aux bytes/token) × context
  window`, straight out of `config.json`. The plan itself is built after the
  model load and the pre-read happens inside it, so the number is computed
  from the same inputs and published before the load. Unknown inputs publish
  zero and say so in `/health`.
- **margin** is a flat 6 GiB: roughly what macOS keeps for the window server
  and the compositor plus one MLX allocator cache round. Being wrong here is a
  swap storm rather than a slow first token, so it is a constant, not a ratio.

If the headroom goes negative the pre-read is skipped with
`skipped_reason: "no_headroom"` - it never evicts the weights it just loaded.

### Which rows

A budget smaller than the table has to choose. `--ngram-prewarm-order PATH`
(default `<model>/ngram-hotness.npy` when present) supplies row ids in
descending gather frequency; the pre-read warms those, coalesced into
page-aligned runs, instead of the first N bytes of the file.

At a given budget both orders warm the *same number of pages* - a 16 KiB page
holds ~163 of the 100-byte rows and the rows are hash-scattered, so a hot row
costs a whole page either way. What changes is which pages. Build the file
with:

```bash
python tests/ngram_row_hotness.py --model <model dir> --prompt-tokens 65536
```

It hashes a corpus through the model's own n-gram hash (lifted out of the
shipped source, not copied) and prints its coverage. On HumanEval, 50,000 rows
- 0.016% of the table's 320,001,536 rows, ~0.8 GiB of pages - cover 76% of
that corpus's gathers. The cost is random reads: measured ~4.6 GiB/s against
~12 GiB/s sequential, so hotness ordering pays when the budget is a fraction
of the table and loses when it is not (a budget that covers the table always
takes the sequential path).

### The other pool

The pre-read competes with MLX's own reclaimable buffer cache, which is a
separate, existing knob: `--mlx-cache-limit` (env `MTPLX_MLX_CACHE_LIMIT`,
`off`/`none`/`unlimited` to disable). It defaults to
`max(1 GiB, min(8 GiB, memory_budget / 8))` when `--memory-budget` is set, and
otherwise to a RAM tier - 8 GiB at 100 GB and above. `/health` reports it as
`mlx_cache_limit` with the applied `limit_bytes`. That pool is reclaimable, so
it is not lost memory, but it is 8 GiB the page cache does not get; lowering it
is the first lever if `auto` keeps sizing the budget too small.

### What `/health` reports

```json
"ngram_prewarm": {
  "enabled": true, "mode": "auto", "order": "prefix",
  "table_bytes": 32000154008, "budget_bytes": 32000154008,
  "warmed_bytes": 32000154008, "seconds": 2.5, "gib_per_s": 12.1,
  "free_bytes": 47244640256, "reserved_bytes": 8589934592,
  "margin_bytes": 6442450944, "source": "default", "skipped_reason": null
}
```

**Caveat.** This warms the page cache; it does not pin it. Under memory
pressure macOS can evict those pages again, and the pre-read has no way to
notice - `/health` will still report the load-time numbers. The hot-row LRU
(`MTPLX_NGRAM_HOT_MB`, default 1024) holds the popular rows in RAM and is
unaffected. A periodic re-warm, or growing that LRU to cover the hot set the
hotness file already identifies, is the follow-up - deliberately not part of
this change.

## Sharing on your network (other devices, Parallels/VM guests)

The default bind is `127.0.0.1`: only this Mac can connect. To reach MTPLX
from other devices — or from a Windows VM in Parallels/VMware/UTM on the same
Mac, which arrives over the virtual network rather than loopback — bind all
interfaces. Non-localhost binds require an API key; if the key file doesn't
exist yet it is created with a fresh key and printed once:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key-file ~/.mtplx/api-key
```

Startup prints a `Network OpenAI API Base URL` (your Mac's LAN address, e.g.
`http://192.168.1.20:8000/v1`). On the other machine, point any
OpenAI-compatible client at that base URL with the printed key as the API
key (sent as a Bearer token). Parallels shared networking reaches the Mac's
LAN address directly; macOS may ask once to allow incoming connections —
click Allow. To pass the key inline instead of a file:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_API_KEY"
```

For Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

For Dockerized Open WebUI, the container must use the host gateway URL, not the host's loopback URL:

```bash
mtplx openwebui docker-command
```

That helper disables Open WebUI's Ollama probe and background task generations
so MTPLX only serves visible chat turns by default.

For Anthropic Messages-compatible clients, point the client base URL at the
bare server root — no `/v1` suffix:

```text
http://127.0.0.1:8000
```

The Anthropic SDK appends `/v1/messages` itself; a `/v1` base would request
`/v1/v1/messages`, which is not a registered route.

## Android Studio

Android Studio's external model provider should use the OpenAI-compatible URL
schema and the MTPLX `/v1` base URL:

```text
URL: http://127.0.0.1:8008/v1
URL schema: OpenAI-compatible
API key: leave blank for localhost unless MTPLX was started with --api-key
```

Refresh the model list after the server starts. MTPLX supports the OpenAI chat,
streaming, and tool-call request shape used by local coding clients; Gemini-only
proprietary behavior is outside that compatibility contract. To verify a local
setup, run:

```bash
mtplx doctor android-studio --port 8008
```

Since 2.5.3 the stats footer only appears on MTPLX-owned surfaces (the app
and the built-in browser chat); API clients such as Open WebUI, Claude Code,
and OpenCode never receive it, so no flag is needed for them.
`--no-stats-footer` still turns it off everywhere, and
`MTPLX_STATS_FOOTER_SCOPE=all` restores the pre-2.5.3 behavior. Metrics
remain available at `/metrics`.
