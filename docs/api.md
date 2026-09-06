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

## `POST /v1/completions`

Legacy OpenAI completions.

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
`web_search`, `image_generation`, `tool_search`, MCP, code-interpreter, or similar
hosted tool types return `400` naming the unavailable type. The adapter does not
silently remove them: dropping a requested capability would change the request's
meaning, and a warning in metadata is not a reliable acknowledgement from the
client. Required or explicitly selected unsupported tools must not be converted
into an ordinary text answer.

`previous_response_id`, background jobs, and server-side Response storage are
also outside this stateless route; send the full conversation and matching tool
calls/outputs in `input`.

### Codex first-request setup

After configuring Codex's model/provider to use the MTPLX Responses endpoint,
start the CLI with hosted tools explicitly disabled:

```bash
codex -c 'web_search="disabled"' --disable image_generation
```

The flags were checked with **Codex CLI 0.144.1** against a loopback request-capture
endpoint on September 6, 2026. This is request-construction evidence, not a claim
of end-to-end model validation. See the [validation receipt](validation/codex-0.144.1-local-tool-profile.json)
and its linked reproducible workflow. A fresh-home default request advertised
`web_search`; the explicit settings left only `function` and `namespace` tool
types. The image-generation feature was enabled by default but was not advertised
by that particular custom-provider/model configuration. Other configurations can
advertise it, so disable it explicitly rather than depending on that omission.
The existing Codex 0.146 fixtures also carry default `web_search`.

Equivalent persistent settings can be merged into the configuration used for
this local client. Keep `web_search` at the top level and merge, rather than
duplicate, an existing `[features]` table. Do not replace provider or credential
settings:

```toml
web_search = "disabled"

[features]
image_generation = false
```

For a deliberately minimal local profile without Codex Apps, add `--disable apps`
to the CLI invocation, or set `apps = false` in the same `[features]` table. This
is separate from disabling hosted tools. Installed app or MCP schemas, skills,
and conversation history consume context; remove unnecessary client integrations
rather than bypassing the server's context limit. The fresh-home capture did not
measure an account with installed connectors, so it does not establish a specific
schema-size reduction.

A separate `CODEX_HOME` can isolate the local provider configuration, but it also
isolates existing settings and authentication. Configure that home deliberately;
do not delete or overwrite the normal home. These settings do not disable shell
network access and are not a network sandbox policy.

The verified scope is CLI 0.144.1. Recheck `codex --version`, `codex features list`,
and first-request tool types for other versions, desktop app-server, or extensions.
An unsupported hosted type should still produce an explicit `400`, not an
implicit downgrade.

References: [Codex configuration](https://developers.openai.com/codex/config-reference)
and [Responses function tool choice](https://developers.openai.com/api/docs/guides/function-calling).

Codex `reasoning.effort: "xhigh"` is accepted as request vocabulary and resolved
against the loaded model. Qwen 3.8 preserves `xhigh`, Step 3.5 clamps it to
`high`, and Qwen 3.6 has no effective reasoning-effort tier. Request
observability records the requested and effective values plus whether the
request was downgraded. The Responses payload echoes the client's requested
reasoning configuration.

## `POST /v1/messages`

Anthropic Messages baseline. Requests are translated into the same internal chat path as `/v1/chat/completions` and returned as Anthropic-shaped message payloads.

Supported now:

- `system` as text or text content blocks
- `messages[].content` as text or text/tool-result content blocks
- `max_tokens`, `temperature`, `top_p`, and `top_k`
- `tools`, `tool_choice`, `stop_sequences`, and `thinking`
- `stream=false`
- `stream=true` server-sent events with `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, and `message_delta`, and `message_stop`

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
