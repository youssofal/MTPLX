#!/bin/zsh
# Discarded full K3 control primer -> C0 -> MoE-tail candidate -> C1 in one
# attested GPU window. Invoke only through bench/laguna/run_guarded.py.
set -euo pipefail

usage() {
  /bin/cat <<'EOF'
Do not execute this inner child directly. Run the postflight wrapper instead:

  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
    /private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/8e9b6abf-6a38-4e6e-ade0-6b0f191bb256/scratchpad/moe-tail/scripts/deepseek_v4_moe_tail_guarded_bracket.py

The wrapper invokes `/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py`
with `/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist`,
`--lock-timeout-seconds 3600 --child-timeout-seconds 3600`, and this script as
its only child. `run_guarded` owns Qwen teardown/restoration. Its exact plist
restore is:

  launchctl bootstrap gui/501 \
    /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist

Only use that manual restore after run_guarded exits, the canonical GPU lock is
free, and :8080 remains down. Never bootstrap Qwen while another owner holds the
lock. After every window, require both the exact model identity and a real chat:

  curl -sf --max-time 10 http://127.0.0.1:8080/v1/models | \
    /usr/bin/python3 -c 'import json,sys; p=json.load(sys.stdin); assert [m["id"] for m in p["data"]] == ["mtplx-qwen36-27b-optimized-quality"]'

  curl -sf --max-time 60 http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"mtplx-qwen36-27b-optimized-quality","messages":[{"role":"user","content":"Say READY"}],"max_tokens":8,"temperature":0}' | \
    /usr/bin/python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]; assert c["finish_reason"] == "stop"; assert c["message"]["content"].strip() == "READY"'

Required receipt: content == "READY" and finish_reason == "stop". A successful
/v1/models response alone is not a serving restoration proof.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

VENV=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
WORKTREE=/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/8e9b6abf-6a38-4e6e-ade0-6b0f191bb256/scratchpad/moe-tail
BENCH=/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp
PROMPT="$BENCH/smoke-2bitdq-20260731-prompt2.txt"
VALIDATOR="$WORKTREE/scripts/deepseek_v4_validate_moe_tail_k3_bracket.py"
PROMPT_SHA256=ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33
CONFIG_SHA256=c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f
INDEX_SHA256=c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8
TAG="${1:-moe-tail-k3-$(date -u +%Y%m%dT%H%M%SZ)}"

# The outer wrapper owns the mandatory read-only postflight after run_guarded
# restores Quality and releases the lock. Refuse a direct guard invocation so
# no execution path can silently skip that receipt.
[[ "${MTPLX_DSV4_MOE_TAIL_POSTFLIGHT_WRAPPER:-}" == "1" ]] || {
  print -u2 "[moe-tail-arms] invoke deepseek_v4_moe_tail_guarded_bracket.py, not this inner child"
  exit 1
}

# Consume run_guarded's one-shot pipe before any MLX import. The issued private
# receipt is reusable by the one benchmark and receipt-only validator while it
# remains bound to this process ancestry and the still-held canonical lock.
GUARD_PIPE_FD=${MTPLX_GUARD_ATTEST_FD:-}
GUARD_ISSUED=$("$VENV" -u "$WORKTREE/scripts/deepseek_v4_guard_window.py" issue)
GUARD_RECEIPT=${GUARD_ISSUED%%$'\t'*}
GUARD_DIGEST=${GUARD_ISSUED#*$'\t'}
[[ -n "$GUARD_PIPE_FD" && "$GUARD_RECEIPT" != "$GUARD_ISSUED" && ${#GUARD_DIGEST} == 64 ]] || {
  print -u2 "[moe-tail-arms] malformed guard-window metadata"
  exit 1
}
exec {GUARD_PIPE_FD}<&-
unset MTPLX_GUARD_ATTEST_FD MTPLX_GUARD_ATTEST_NONCE GUARD_ISSUED
GUARD_DIR=${GUARD_RECEIPT:h}
cleanup_guard_receipt() {
  /bin/rm -f -- "$GUARD_RECEIPT"
  /bin/rmdir -- "$GUARD_DIR" 2>/dev/null || true
}
trap cleanup_guard_receipt EXIT

[[ -x "$VENV" && -f "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
  && -f "$VALIDATOR" && -f "$PROMPT" && -d "$MODEL" ]] || {
  print -u2 "[moe-tail-arms] interpreter, scripts, prompt, or model missing"
  exit 1
}
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || {
  print -u2 "[moe-tail-arms] worktree is dirty; refusing an unrepeatable bracket"
  exit 1
}
actual_prompt_sha=$(shasum -a 256 "$PROMPT" | awk '{print $1}')
actual_config_sha=$(shasum -a 256 "$MODEL/config.json" | awk '{print $1}')
actual_index_sha=$(shasum -a 256 "$MODEL/model.safetensors.index.json" | awk '{print $1}')
[[ "$actual_prompt_sha" == "$PROMPT_SHA256" \
  && "$actual_config_sha" == "$CONFIG_SHA256" \
  && "$actual_index_sha" == "$INDEX_SHA256" ]] || {
  print -u2 "[moe-tail-arms] canonical prompt/config/index identity mismatch"
  exit 1
}
# Remove every inherited experiment selector, including future MTPLX knobs.
# Re-export only the fixed Stage-4 arm below; the wired-memory knob is untouched.
for entry in ${(f)"$(env)"}; do
  name=${entry%%=*}
  if [[ "$name" == MTPLX_* ]]; then
    unset "$name"
  fi
done
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKTREE/scripts:$WORKTREE"
export HF_HUB_OFFLINE=1
export MTPLX_COMPILED_VERIFY=off
export MTPLX_DSV4_ATTN=fused
export MTPLX_DSV4_FP32_ACTIVATIONS=0
export MTPLX_DSV4_HC_COMPILE=1
export MTPLX_DSV4_MOE_TAIL=1
export MTPLX_DSV4_O_LORA=cached
export MTPLX_DSV4_SINKHORN_KERNEL=1
export MTPLX_DSV4_GUARD_WINDOW_PATH="$GUARD_RECEIPT"
export MTPLX_DSV4_GUARD_WINDOW_SHA256="$GUARD_DIGEST"

# Exactly one MLX process and one model load. It captures the construction-time
# candidate callables, binds stock for the discarded_control_primer and C0,
# binds the candidate only for B's K3 sub-arm (B's AR remains stock), and
# restores stock before C1. Generation-local caches/counters are reset between
# every sub-arm while compiled/model state stays married to this one process.
set +e
"$VENV" -u "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
  --moe-tail-bracket \
  --model "$MODEL" --prompt-file "$PROMPT" --max-tokens 256 --depths 3 \
  --verify-strategy capture_commit --verify-core stock \
  --mtp-history-policy committed --warmup-tokens 0 \
  --out "$BENCH/$TAG"
benchmark_rc=$?
set -e

VALIDATION="$BENCH/$TAG-validation.json"
if "$VENV" -u "$VALIDATOR" \
  --primer "$BENCH/$TAG-primer.json" \
  --before "$BENCH/$TAG-before.json" \
  --candidate "$BENCH/$TAG-candidate.json" \
  --after "$BENCH/$TAG-after.json" \
  --benchmark-exit-code "$benchmark_rc" \
  --peak-ceiling-gib 108 --require-live-guard --out "$VALIDATION"; then
  print "[moe-tail-arms] PASS: $VALIDATION"
else
  validation_rc=$?
  print -u2 "[moe-tail-arms] non-promotable (exit=$validation_rc); receipts preserved at $BENCH/$TAG-{primer,before,candidate,after,validation}.json"
  exit "$validation_rc"
fi
