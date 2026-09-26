# Server

The server target is OpenAI-compatible local serving, with Anthropic
Messages compatibility available for coding harness smoke tests.

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

See [Concurrency modes](concurrency.md) for scheduler selection, ownership
rules, and model/backend-specific implementations.

On M3, M4 and M5, the app and CLI recommend Qwen 3.5 4B Optimized Speed
below 16 GB, Ternary Bonsai 2 27B from 16 GB, Qwen 3.8 27B Optimized Speed
from 32 GB, and Qwen 3.8 Flash-Next Optimized Speed from 256 GB, with
Flash-Next Optimized Quality second. Flash-Next Bare Speed is offered from
96 GB and Flash-Next Optimized Speed from 128 GB. M1 and M2 keep the FP16
policy: the 9B below 32 GB when it fits, then the 27B trio. An 8 GB M1 or M2
Mac has no fitting curated FP16 model. An explicit model selection always
wins.

Ternary Bonsai 2 27B and Flash-Next Optimized Quality need MTPLX 2.12.0 or
later. The app shows the Recommended badge when the Mac has at least 1.5
times a model's peak memory. Before a download, the app and `mtplx pull` ask
for the bytes still to download plus 5 GiB of free disk. Neither rule changes
engine memory limits, context windows or admission checks.

The catalog's memory figure for Flash-Next Optimized Quality is the memory
planner's estimate at a 128K context. Serving derives memory use from the
actual weights and context, and the catalog neither caps the context nor
changes the planner.

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

## Warm session cache (RAM bank) limits

Every conversation's KV state is kept warm in RAM after a turn so the next
message in that conversation restores instead of re-reading the prompt. Nothing
in the daemon drops warm state on a short clock: the only idle timer is one
hour, and the ten-minute number that appears in the code decides which
session is evicted *first* when the cache is over budget, not whether anything
is evicted at all. `/health` reports the live values under `session_bank`
(`idle_ttl_s`, shown as `null` when the sweep is off, `active_pin_ttl_s`,
the byte budgets and, after a miss, `last_miss_reason`).

These environment variables are read by the daemon at start (`mtplx start`,
`mtplx serve`, and the app's daemon, which inherits the login environment):

| Variable | Default | Meaning |
|---|---|---|
| `MTPLX_SESSION_BANK_IDLE_TTL_S` | `3600` | Seconds a warm entry and its session may sit untouched before the idle sweep drops them. `0` disables the sweep: entries live until the byte budgets, real memory pressure or a restart take them (the SSD tier keeps its copy either way). |
| `MTPLX_SESSION_BANK_ACTIVE_PIN_TTL_S` | `600` | Sessions that touched the cache within this many seconds are evicted last when the cache is over budget. `0` turns the preference off. |
| `MTPLX_SESSION_BANK_MAX_BYTES` | `auto` | Total warm-cache budget. `auto` is half of the RAM left after the model weights, floored at 1 GiB and capped at 48 GiB. Sizes such as `24G` are accepted. |
| `MTPLX_SESSION_BANK_PER_SESSION_BYTES` | `auto` | Budget for one conversation's warm state: two thirds of the total budget, held under what the machine can restore (half of the engine budget left after the weights and 3 GiB of transients, since a restore holds the snapshot next to its banked copy). 64 GB Mac with the 27B: 13.2 GiB; 128 GB with the 27B: 32 GiB; 128 GB with Flash-Next: 10.5 GiB. Sizes such as `12G` are accepted. |
| `MTPLX_SESSION_BANK_MAX_ENTRIES` | `24` (`48` on Macs with 96 GB or more) | Maximum number of warm entries across all conversations. |

A long conversation that starts over after a pause is a cache *miss*, not an
expiry: open `/health` right after the slow turn and read
`session_bank.last_miss_reason` and `last_prefix_diagnostic`.

**A conversation over its per-session budget.** Its snapshot is not copied.
The cache keeps a reference to the live KV instead, so the next turn can
continue without a prefill. That reference holds real memory, and `/health`
reports it: `session_bank.lease_entries` and `lease_nbytes`, and
`held_nbytes` on each entry (`nbytes` stays the snapshot size, which is 0
for a reference). It counts against `MTPLX_SESSION_BANK_MAX_BYTES`, a
conversation keeps one at most, and memory pressure can release it; the
conversation then restores from the SSD tier or prefills. Setting
`MTPLX_SESSION_BANK_PER_SESSION_BYTES` *lower* to save memory does not save
any: it only moves more conversations onto this path, where every turn after
the limit depends on that single reference.

**What the memory guards compare.** The guards that run before a long prompt
and in the background compare MLX's own account (`active + cache`) with the
Metal memory limit. They also read the process's real footprint from macOS
(`phys_footprint`, the number the system's own memory-pressure logic uses)
and add the part of it that MLX's account does not explain, beyond what a
daemon normally holds outside Metal. `/health` and `/v1/mtplx/snapshot` show
both under `mem`: `phys_footprint_bytes` and `host_overhang_bytes`.

| Variable | Default | Meaning |
|---|---|---|
| `MTPLX_HOST_MEMORY_ALLOWANCE_BYTES` | `auto` | Process memory outside MLX's account that the guards treat as normal. `auto` is the larger of 8 GiB and what the machine leaves after the system reserve and the Metal limit (16 GiB on a 128 GB Mac with default limits). `0` makes every byte above MLX's account count, which is stricter than the memory plan and reads a full session on a 48 GB Mac as critical. Sizes such as `12G` are accepted. |

## SSD session cache (cold tier) limits

Committed sessions are also written to `~/.mtplx/session-bank/` (or
`--ssd-session-cache-dir`) so they restore after a restart. The store is
content-addressed: `entries/` holds one `payload.json` per snapshot naming
the `blobs/` files that make it up, and `manifest.sqlite` names the entries
a restore may use. A blob shared by several snapshots of one conversation is
stored once.

| Setting | Default | Meaning |
|---|---|---|
| `--ssd-session-cache {on,write-only,off}` | `on` | Whether sessions are written to and restored from SSD. |
| `--ssd-session-cache-max-size` | `100GB` (`32GB` on 64 GB Macs unless the disk has 150 GiB free, `24GB` on 32 GB, `16GB` on 16 GB) | Cap on the **whole** store directory as it sits on disk, orphaned files included. The effective cap is `min(this, free_disk / 4)`, and writes stop below 10 GiB free. When a write would take the store over the cap, garbage is reclaimed first and only then are the least recently used entries evicted. |
| `MTPLX_SSD_WRITE_BUDGET_PER_HOUR` | `128G` | Rolling one-hour byte budget for SSD writes (SSD wear); writes beyond it are skipped for the hour. |
| `MTPLX_SSD_WRITER_BACKLOG_BYTES` | `4G` | Bytes of encoded snapshots the writer may hold in RAM while waiting to write; larger single entries stream to disk instead. |

**Orphaned files.** A crash between a blob write and its manifest row, an
interrupted write, or a blob whose last entry was evicted while a sibling
still shared it can leave files the manifest no longer reaches. The daemon
reconciles the store against its manifest every time it opens the cache:
a background pass that yields to requests, deletes what nothing names, and
prints one `mtplx_ssd_session_cache_reconcile` line to the daemon log with
what it reclaimed and what the store holds. The same pass runs whenever a
write finds the store over its cap. `/health` reports the store under
`ssd_session_cache` (`entries`, `managed_disk_bytes`, `orphan_disk_bytes`,
`orphan_cleanup_runs`, `startup_reconcile`).

To inspect or clean a store by hand, with or without a daemon running:

```bash
mtplx gc            # report: live entries, bytes on disk, what is orphaned
mtplx gc --apply    # delete the orphaned files (refuses while a daemon runs)
mtplx gc --apply --force   # ...even beside a running daemon
mtplx gc --dir /path/to/session-bank --json
```

`mtplx gc` needs no MLX and no model: it reads `manifest.sqlite` and the
directory tree. `--apply` refuses to run while a daemon answers on the
default ports, because an out-of-process pass cannot see that daemon's
writes in flight; `--force` overrides that when you know the daemon is idle.
`--dir` defaults to the `ssd_session_cache_dir` in your saved config, then
`~/.mtplx/session-bank`.

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

Leave Open WebUI's Controls (temperature, top P, top K and the rest) on
Default. A value left on Default is not sent, so MTPLX applies the sampler and
reasoning settings it uses for the loaded model, the ones the dashboard shows.
A value changed in Open WebUI is sent with every request; see
[Who controls connected-app settings](api.md#who-controls-connected-app-settings)
for when MTPLX follows it (#513).

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

## Serving several models: `mtplx supervise`

`mtplx serve` loads exactly one model for the life of the process. `mtplx
supervise` puts one front door in front of several models instead, loading
and unloading engine child processes on demand. `mtplx serve` itself is
unchanged: supervise is a separate command, not a flag on serve.

```bash
mtplx supervise --models 9b,27b --preload 9b --default 9b \
  --host 0.0.0.0 --port 8000 --api-key-file ~/.mtplx/api-key
```

Each `--models` entry resolves the same way `serve --model` does: a local
path, a `~/.mtplx/models/<Org>--<Name>` directory, or an installed catalog
id. `--models all` picks up every installed pack. A name that isn't
installed is a fatal error that lists what is.

A request's `model` field routes it: a match on a loaded or installed model
routes there (loading it just-in-time if it isn't running yet, holding the
request until it's ready or `--load-timeout-s` runs out). An unmatched or
missing `model` falls back to `--default` and adds the response header
`x-mtplx-routed-model: <default>`. Pass `--strict-model` to return a 404
`model_not_found` error there instead. A model that is draining (mid-unload
or mid-restart) answers 503 `{"error": {"code": "engine_draining"}}` with
`Retry-After: 2` rather than racing a fresh JIT load against the drain.

`--memory-budget <GiB>` caps how much engine memory the supervisor will
admit at once; the default is this machine's usable budget. Without
`--evict-to-fit`, a model that would exceed the budget is refused with a 507
carrying `{"error": {"code": "insufficient_memory", "needed_bytes",
"available_bytes", "would_free": [...]}}`. With `--evict-to-fit`, the
supervisor unloads idle models in least-recently-used order (never one with
an in-flight request) until the new one fits.

A model with no in-flight requests for `--idle-ttl-s` (default 1800, `0`
disables) is drained and unloaded; `--default` is exempt unless you also
pass `--unload-default`.

If an engine child crashes, the supervisor restarts it with backoff, up to 3
crashes in 120 seconds, after which it stays `failed` until an admin
restart. A crash classified as out-of-memory never auto-restarts: retrying
into the same OOM only spins.

Admin routes always require the API key; inference routes follow the same
rule `serve` uses: no key on a localhost bind, a key required otherwise
(`--insecure-lan --yes` lifts that for inference only, and prints a warning
at start). Every route (admin and inference) also normalizes its path
before matching: `/HEALTH` and `//v1//models` match `/health` and
`/v1/models` the same as the canonical spelling, instead of falling through
to a literal (and likely 404) request against an engine.

| Route | Method | Auth |
| --- | --- | --- |
| `/mtplx/admin/models` | GET | key always |
| `/mtplx/admin/load` | POST | key always |
| `/mtplx/admin/unload` | POST | key always |
| `/mtplx/admin/restart` | POST | key always |
| `/mtplx/admin/health` | GET | key always |
| `/v1/*` (chat, embeddings, ...) | as usual | localhost free, else key (or `--insecure-lan`) |

`/health` (unlike the routes above) always answers, even with no key or the
wrong one: native-app daemon-ownership checks need `ok`/`model`/`startup`
without a credential. What the admin key gates is how much it discloses.
Without a valid key the `supervisor` block is redacted to
`{"engines": [{"model", "state"}]}`; with a valid key it's the full
`{"engines": [{"model", "state", "port", "pid", "pins", "last_used",
"failure_reason"}], "budget": {...}}`, the same shape `/mtplx/admin/health`
always returns. `/v1/models` lists each model's registered `aliases`
(engine-reported ids that route to it, once it has come up READY at least
once) alongside `id`/`state`/`loaded`.

**Rate limits.** Both apps throttle per client host (keyed off the ASGI
connection's remote address, not the API key, so it also works against
unauthenticated callers): `/mtplx/admin/*` allows 30 requests/minute per
host, and failed auth (a 401 on any route, admin or inference) counts
against a shared 10/minute-per-host budget; once that's exhausted, that
host gets 429 `{"error": {"code": "rate_limited"}}` with `Retry-After` on
every route, not just the one it failed auth on, until the window rolls
off. This is in-process and per-supervisor-process state, not shared across
restarts or machines.

**Request body cap.** The auth check runs before the body is ever read;
once a request is authorized, its body is capped at 32 MiB while being
read (not buffered first and checked after) and a request over that limit
gets 413 `{"error": {"code": "request_too_large"}}` without ever holding
the oversized body in memory.

**Exit code.** `kill -TERM` (or Ctrl-C) on the supervisor drains and stops
every engine child through uvicorn's own signal handling, but the process
itself still exits by the raw signal once that finishes, not through a
normal Python return: `kill -TERM <pid>` yields exit code 143 (128 + 15),
the same as an unhandled SIGTERM, even though the shutdown itself was
clean. Treat 143 after a `supervise` run as the expected exit for a
requested stop, not a crash.
