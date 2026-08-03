# Troubleshooting

Start with:

```bash
mtplx doctor --json
```

## `mlx` Missing

`doctor` should report missing MLX as an actionable runtime dependency issue, not as a traceback. Help, inspect, and init should still work.

## Model Refuses To Run

Run:

```bash
mtplx inspect model --model /path/or/repo --json
```

The model must be `verified` tier for the default path (`mtplx inspect` prints the tier). Architecture-compatible unverified models load with an explicit unverified label and cannot be used for release claims.

## Slow Long Responses

This is a known caveat. Use the benchmark output and profile name when filing an issue. Do not compare `--max` diagnostic runs against no-fan product claims.

## Model Repeats Itself / Loops

If an agent or chat session degenerates into repeating the same phrase or tool call, raise the presence penalty. It is available per request (`presence_penalty` in the OpenAI payload), as a server default (`--default-presence-penalty 1.0` on `start`/`serve`), live via `mtplx settings set`, or with the Presence Penalty dial in the app and dashboard. Values around 0.5–1.5 break repetition; 0 (the default) is an exact no-op. Qwen recommends keeping penalties at 0 for coding and tool-calling work, so prefer fixing the prompt or context before reaching for the dial in agent flows.

## Server Binding

Binding to `0.0.0.0` should require an API key. Prefer localhost for local clients:

```bash
mtplx serve --host 127.0.0.1 --port 8000
```

For a non-localhost bind:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_AUTH"
```

Clients should send `Authorization: Bearer <key>` or `X-API-Key: <key>`.

## `--max` Does Not Change Fans

Run:

```bash
mtplx max --status --json
```

If no supported thermal tool is detected, install ThermalForge or TG Pro and ensure the CLI is on `PATH`. MTPLX will not enable hidden spin-loop or clock-anchor fallbacks.

## Fans Stay On Briefly After a Response (Smart Mode)

That is intentional, not a leak. After the response finishes streaming, MTPLX may run a short background "postcommit" pass that re-processes the conversation into the session cache so your *next* message starts from a warm prefix instead of a cold prefill. That pass uses the GPU at full tilt for a few seconds, so Smart fan mode deliberately holds the fan lease through it and only restores the Apple automatic curve once the GPU work is actually done (plus a ~2 s debounce so back-to-back agent tool calls don't flap the fans down and up).

In Smart mode the ramp is issued the moment your request arrives — before prompt processing starts — and the server verifies the fan daemon accepted the target RPM (retrying once if it didn't). You can watch this in `GET /health`: `smart_fan_target_verified`, `smart_fan_actual_ramp_verified`, `smart_fan_ramp_latency_s`, and `smart_fan_last_error` tell you exactly what the fan controller last did. If `smart_fan_last_error` mentions sudo, run `mtplx max --grant-sudo` once.
