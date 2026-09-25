# API

MTPLX targets OpenAI-compatible local serving first.

## `GET /health`

Reports model load state, profile, exactness baseline, MLX/runtime information, fan mode, and warmup status.
The payload includes `generation_mode`, `load_mtp`, `mtp_enabled`, `depth`, `api_key_required`, `rate_limit_per_minute`, `stream_interval`, `warmup`, and `reasoning_parser` so client harnesses can confirm the active serving policy.

## `GET /metrics`

Returns a JSON snapshot of runtime KPIs: `latest` (most recent turn), `recent` (last 32 turns), and `tool_parse_counters`.

## `GET /v1/models`

Lists cached and active models.

## `POST /v1/chat/completions`

OpenAI-compatible chat completions. Streaming uses server-sent events.
Use `--stream-interval N` to batch committed-token SSE chunks when a client prefers less frequent events.
Requests may set `generation_mode` to `"mtp"` or `"ar"`. `"ar"` uses target-only AR generation and reports `mtp_depth: 0`; it does not unload MTP weights, so the server can switch back to MTP on a later request.
When tools are active, Qwen XML tool calls are translated into OpenAI
`delta.tool_calls` chunks as the function name and arguments stream. Unknown or
malformed tool-shaped output falls back to assistant content rather than hanging
or returning a server 500.
`logprobs: true` with `top_logprobs: K` returns `choices[0].logprobs.content` for
the first generated token only; it requires `max_tokens: 1` and no streaming
(see First-token logprobs below).

## `POST /v1/completions`

Legacy OpenAI completions.
Two logprobs modes:

- Prompt scoring: `echo: true`, `max_tokens: 0`, `logprobs: K` returns the top-K
  distribution for every prompt position.
- First-token logprobs: `max_tokens: 1`, `logprobs: K` (echo off) returns the
  top-K distribution of the first generated token.

### First-token logprobs

Both completion routes can return the next-token distribution after the prompt
through the normal generation path. The values are the raw model
log-probabilities of the row that produced the token, before temperature,
penalties, grammar masks and steering. The top-K list always contains the sampled
token with its true value. Limits: `max_tokens` must be 1, `stream` and a
non-empty `stop` are refused, and `K` is capped by `MTPLX_PROMPT_LOGPROBS_MAX`
(default 128). Requests the MTP batch lane cannot serve are refused with 400
rather than answered without logprobs.

## `POST /v1/responses`

Codex Responses compatibility with hosted tools disabled, through the same
chat inference path. The adapter is stateless and text-only, supports
structured text input, instructions, streaming lifecycle events, JSON-schema
text formats, client-executed function/custom tools, and namespace-grouped
function tools.
Namespace tools are flattened only for local chat rendering; returned
`function_call` items restore the official original `name` plus `namespace`.
OpenAI Python 2.52's explicit function `tool_choice` selector has only `type`
and `name`, so it cannot unambiguously qualify a nested namespace function;
use `auto` or `required` when namespace tools are present. MTPLX rejects an
ambiguous nested-function selector instead of silently selecting the wrong
tool.

MTPLX does not execute hosted Responses tools. Requests containing
`web_search`, `tool_search`, MCP, code-interpreter, or similar hosted tool types
return `400` naming the unavailable type. `previous_response_id`, background
jobs, and server-side Response storage are also outside this stateless route;
send the full conversation and matching tool calls/outputs in `input`.
Codex 0.146 sends `web_search` in its default Responses request, so that default
is intentionally rejected unless hosted tools are disabled or removed from the
client request.

Running Codex against MTPLX (measured with Codex 0.144.1 on 2026-09-06): Codex
also sends a hosted `image_generation` tool by default, and with Codex apps
installed its tools array carries the connector schemas (about 615 KB, which
MTPLX counts as roughly 186k prompt tokens against the context window). Three
settings make the first request work:

```toml
# ~/.codex/config.toml, or a dedicated CODEX_HOME so installed apps stay out
model_provider = "mtplx"
web_search = "disabled"

[features]
image_generation = false

[model_providers.mtplx]
name = "MTPLX"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
```

`codex exec -m <served model id> ...` then completes shell and patch turns
through `/v1/responses`; `--disable image_generation` on the command line is
the same as the `[features]` entry.

Codex `reasoning.effort: "xhigh"` is accepted as request vocabulary and resolved
against the loaded model. Qwen 3.8 preserves `xhigh`, Step 3.5 clamps it to
`high`, and Qwen 3.6 has no effective reasoning-effort tier. Request
observability records the requested and effective values plus whether the
request was downgraded. The Responses payload echoes the client's requested
reasoning configuration.

## Reasoning effort

Thinking depth is a request-level setting with a server-level default.

- Per request: `reasoning_effort` in the chat completions body (`"reasoning_effort": "high"`), or `reasoning.effort` on the Responses API. Accepted values are `low`, `medium`, `high` and `xhigh`. OpenAI's `minimal` and `none` (and `off`, `disable`, `disabled`) are accepted and map to `low`; they do not switch thinking off.
- Server default: `mtplx start --reasoning-effort high` (or `mtplx serve`) applies to every request that carries no value. `auto` (the default) uses the loaded model family's own default.
- App: the effort picker in the chat settings lists the levels the loaded model supports.

The levels that take effect depend on the family: Qwen 3.8 honours all four including `xhigh`, Step 3.5 clamps `xhigh` to `high`, and Qwen 3.6 has no effort tiers (the value is accepted and ignored). To turn thinking off entirely use `--reasoning off` on the server or `chat_template_kwargs: {"enable_thinking": false}` on the request. The request log records the requested and effective values and whether the request was downgraded.

## `POST /v1/messages`

Anthropic Messages baseline. Requests are translated into the same internal chat path as `/v1/chat/completions` and returned as Anthropic-shaped message payloads.

Supported now:

- `system` as text or text content blocks
- `messages[].content` as text or text/tool-result content blocks
- `max_tokens`, `temperature`, `top_p`, and `top_k`
- `tools`, `tool_choice`, `stop_sequences`, and `thinking`
- `stream=false`
- `stream=true` server-sent events with `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`

Streaming note: Qwen reasoning maps to Anthropic thinking blocks — a `content_block_start` with content-block type `thinking`, then `thinking_delta` events, with answer text resuming in a separate text block.

Examples:

- [Anthropic Python client](../examples/anthropic-python-client.py)
- [Anthropic Messages curl](../examples/curl-messages.sh)
- [OpenAI Python client](../examples/openai-python-client.py)
- [OpenAI chat completions curl](../examples/curl-chat-completions.sh)

## Server Flags

```bash
mtplx serve --port 8000
mtplx serve --host 0.0.0.0 --api-key "$MTPLX_API_KEY"
mtplx serve --rate-limit 120
mtplx serve --stream-interval 4
mtplx serve --warmup-tokens 16
mtplx serve --reasoning-parser qwen3
mtplx serve --no-mtp
```

Non-localhost binds require `--api-key`. Requests may authenticate with either:

```text
Authorization: Bearer <key>
X-API-Key: <key>
```

`--warmup-tokens` runs a small startup generation after model load and reports the result in `/health`. `--strict-warmup` makes warmup failure fatal.

`--adaptive-policy expected_value` lets the engine stop a draft cycle early
when another draft step is not expected to pay for its verify work;
`--adaptive-policy none` (the default) drafts to `--depth` every cycle. The
app's Pi and Hermes launches name the policy for the 27B family and leave it
off for Flash-Next, where a fixed depth 3 measured 7 to 8 percent faster. The
policy can be flipped live through `POST /v1/mtplx/settings` (the app's
Adaptive depth switch), and the flip is part of the session bank's cache
identity: every banked session misses once after it and the next turn
re-prefills from scratch (42 s for a 19k-token prompt on the 27B), so change
it between sessions rather than in the middle of a long one.

### Who controls connected-app settings

The macOS app's inference panel includes **Use MTPLX settings for connected apps**.
It is on by default for app-launched daemons:

- **On:** MTPLX's live reasoning and sampling settings govern its configured Pi,
  OpenCode, Hermes and Open WebUI connections. Pi mirrors the effective reasoning
  level in its footer, at startup, before a new turn and while idle.
- **Off:** those clients' explicit request settings take precedence. Server values
  remain defaults for fields the client omits. Pi restores the reasoning choice
  it had before following the app during that session.

This switch does not override ordinary API requests. Their explicit parameters
continue to be honored, including `temperature: 0` and `enable_thinking: false`.
Native MTPLX chat also retains its per-chat controls. Explicit response limits
remain client-owned in either mode.

`GET /v1/mtplx/settings` reports `managed_client_controls`. An authenticated settings
update can set it to `app` or `client` without a restart. The macOS app persists
this selection; CLI operators can set `MTPLX_MANAGED_CLIENT_CONTROLS=app` or
`client` for a server launch. The CLI default, `auto`, preserves the prior
contract: managed-client sampling stays server-owned and their reasoning controls
are honored. Request records include the policy and effective ownership fields.
