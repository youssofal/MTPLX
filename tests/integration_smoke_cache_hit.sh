#!/usr/bin/env bash
# Integration smoke test: prove the tools-plumbing fix delivers cached>0
# on the second turn of a session that uses tool definitions.
#
# Background: pre-fix the postcommit pathways encoded the synthetic
# history WITHOUT `tools=`, while the next request encoded its prompt
# WITH `tools=tool_specs`. The Qwen3.6 chat template injects tool
# definitions into the system message, so the stored snapshot's tokens
# diverged from the next prompt at the system boundary - SessionBank
# lookups failed with `cache_miss_reason=prefix_divergence_at_token`.
#
# This script:
#   1. Starts a unique session_id (one per run).
#   2. Issues turn 1 with a `tools=[...]` array and a simple user message.
#   3. Sleeps to let the async postcommit settle.
#   4. Replays the actual assistant message from turn 1, then issues turn 2
#      with the same tools plus a new user message.
#   5. Parses the SSE stream for the final chunk's `mtplx_stats` and
#      reports PASS if `cached_tokens > 0`, FAIL otherwise.
#
# Usage:
#   ./tests/integration_smoke_cache_hit.sh                # default port 8088
#   MTPLX_HOST=http://127.0.0.1:8088 ./tests/integration_smoke_cache_hit.sh
#
# IMPORTANT: hits MTPLX directly, NOT the dashboard at 9099 (the
# dashboard proxy may strip `x-mtplx-session-id`).

set -euo pipefail

HOST="${MTPLX_HOST:-http://127.0.0.1:8088}"
SESSION_ID="smoke-cache-hit-$(date +%s)-$$"
SLEEP_S="${SMOKE_SLEEP_S:-12}"

echo "[smoke] host=${HOST}"
echo "[smoke] session_id=${SESSION_ID}"
echo "[smoke] inter-turn sleep=${SLEEP_S}s"
echo

TOOLS_JSON='[
  {
    "type": "function",
    "function": {
      "name": "grep",
      "description": "Search files for a pattern",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {"type": "string"},
          "path": {"type": "string"}
        },
        "required": ["pattern"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read file contents",
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  }
]'

TURN1_PAYLOAD=$(cat <<EOF
{
  "model": "auto",
  "stream": true,
  "max_tokens": 128,
  "enable_thinking": false,
  "tools": ${TOOLS_JSON},
  "messages": [
    {"role": "system", "content": "You are a helpful coding agent."},
    {"role": "user", "content": "Find any references to 'session_bank' in the repo."}
  ]
}
EOF
)

TMPDIR_SMOKE=$(mktemp -d)
trap 'rm -rf "${TMPDIR_SMOKE}"' EXIT

echo "[smoke] turn 1 -> ${HOST}/v1/chat/completions"
curl -sS -N \
  -H "Content-Type: application/json" \
  -H "x-mtplx-session-id: ${SESSION_ID}" \
  -X POST "${HOST}/v1/chat/completions" \
  -d "${TURN1_PAYLOAD}" \
  > "${TMPDIR_SMOKE}/turn1.sse"

echo "[smoke] turn 1 SSE bytes: $(wc -c < "${TMPDIR_SMOKE}/turn1.sse")"

# Pull the session_postcommit_snapshot from turn 1 (best-effort).
PC1=$(grep -o '"session_postcommit_snapshot":[^}]*}' "${TMPDIR_SMOKE}/turn1.sse" | tail -1 || true)
echo "[smoke] turn 1 postcommit: ${PC1:-<not present>}"

ASSISTANT_MSG_JSON=$("${PYTHON:-python3}" - "${TMPDIR_SMOKE}/turn1.sse" <<'PY'
import json
import sys

content_parts = []
for raw in open(sys.argv[1], encoding="utf-8"):
    if not raw.startswith("data: "):
        continue
    data = raw[len("data: "):].strip()
    if not data or data == "[DONE]":
        continue
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        continue
    choices = payload.get("choices") or []
    if not choices:
        continue
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content:
        content_parts.append(str(content))

print(
    json.dumps(
        {"role": "assistant", "content": "".join(content_parts)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
PY
)

TURN2_PAYLOAD=$(cat <<EOF
{
  "model": "auto",
  "stream": true,
  "max_tokens": 128,
  "enable_thinking": false,
  "tools": ${TOOLS_JSON},
  "messages": [
    {"role": "system", "content": "You are a helpful coding agent."},
    {"role": "user", "content": "Find any references to 'session_bank' in the repo."},
    ${ASSISTANT_MSG_JSON},
    {"role": "user", "content": "Now read the first matching file."}
  ]
}
EOF
)

echo "[smoke] sleeping ${SLEEP_S}s for async postcommit to settle..."
sleep "${SLEEP_S}"

echo "[smoke] turn 2 -> ${HOST}/v1/chat/completions"
curl -sS -N \
  -H "Content-Type: application/json" \
  -H "x-mtplx-session-id: ${SESSION_ID}" \
  -X POST "${HOST}/v1/chat/completions" \
  -d "${TURN2_PAYLOAD}" \
  > "${TMPDIR_SMOKE}/turn2.sse"

echo "[smoke] turn 2 SSE bytes: $(wc -c < "${TMPDIR_SMOKE}/turn2.sse")"

# The stats are emitted on the final chunk as `mtplx_stats`. Pull the
# last occurrence of cached_tokens / cache_miss_reason from the stream.
CACHED=$(grep -o '"cached_tokens":[ ]*[0-9]\+' "${TMPDIR_SMOKE}/turn2.sse" | tail -1 | grep -o '[0-9]\+' || true)
MISS=$(grep -o '"cache_miss_reason":[ ]*"[^"]*"' "${TMPDIR_SMOKE}/turn2.sse" | tail -1 || true)
RESTORE=$(grep -o '"session_restore_mode":[ ]*"[^"]*"' "${TMPDIR_SMOKE}/turn2.sse" | tail -1 || true)
HIT=$(grep -o '"session_cache_hit":[ ]*\(true\|false\)' "${TMPDIR_SMOKE}/turn2.sse" | tail -1 || true)

echo
echo "[smoke] turn 2 cached_tokens   = ${CACHED:-<missing>}"
echo "[smoke] turn 2 cache_miss_reason = ${MISS:-<missing>}"
echo "[smoke] turn 2 session_restore_mode = ${RESTORE:-<missing>}"
echo "[smoke] turn 2 session_cache_hit    = ${HIT:-<missing>}"
echo
echo "[smoke] (raw turn 2 SSE saved at ${TMPDIR_SMOKE}/turn2.sse - inspect if PASS/FAIL is unclear)"
# Persist the artifacts to a stable path for the user to inspect after
# the script exits if needed. (Pre-trap.)
DEST="/tmp/mtplx-smoke-cache-hit-${SESSION_ID}"
mkdir -p "${DEST}"
cp "${TMPDIR_SMOKE}/turn1.sse" "${DEST}/turn1.sse"
cp "${TMPDIR_SMOKE}/turn2.sse" "${DEST}/turn2.sse"
echo "[smoke] artifacts copied to ${DEST}/"

if [[ -n "${CACHED}" && "${CACHED}" -gt 0 ]]; then
  echo "[smoke] PASS - cached_tokens=${CACHED} on turn 2 (session_id=${SESSION_ID})"
  exit 0
else
  echo "[smoke] FAIL - turn 2 was cold (cached_tokens=${CACHED:-0}, miss_reason=${MISS})"
  exit 1
fi
