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
