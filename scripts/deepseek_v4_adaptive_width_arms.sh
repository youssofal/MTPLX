#!/bin/zsh
# Canonical adaptive-width performance bracket. Invoke only via its wrapper.
set -euo pipefail

[[ "${MTPLX_DSV4_ADAPTIVE_WIDTH_POSTFLIGHT_WRAPPER:-}" == 1 ]] || {
  print -u2 'invoke deepseek_v4_adaptive_width_guarded.py, not this child'
  exit 1
}
WORKTREE=${0:A:h:h}
# The guard wrapper supplies these deployment-specific locations.  Keeping
# them out of the source makes the exact artifact identity—not one developer's
# filesystem—the reproducibility contract.
VENV=${MTPLX_DSV4_PYTHON:-python3}
BENCH=${MTPLX_DSV4_BENCH_DIR:?set MTPLX_DSV4_BENCH_DIR}
MODEL=${MTPLX_DSV4_MODEL_PATH:?set MTPLX_DSV4_MODEL_PATH}
PROMPT=${MTPLX_DSV4_PROMPT_FILE:?set MTPLX_DSV4_PROMPT_FILE}
PROMPT_SHA256=ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33

(( $# <= 1 )) || {
  print -u2 'invalid bracket tag: expected zero or one argument'
  exit 1
}
TAG=${1:-adaptive-width-policy-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -z "$TAG" || "$TAG" == '.' || "$TAG" == '..' ]] ||
   [[ ! "$TAG" =~ '^[A-Za-z0-9][A-Za-z0-9._-]*$' ]]; then
  print -u2 'invalid bracket tag: expected a safe basename'
  exit 1
fi

GUARD_PIPE_FD=${MTPLX_GUARD_ATTEST_FD:-}
GUARD_ISSUED=$("$VENV" -u "$WORKTREE/scripts/deepseek_v4_guard_window.py" issue)
GUARD_RECEIPT=${GUARD_ISSUED%%$'\t'*}
GUARD_DIGEST=${GUARD_ISSUED#*$'\t'}
[[ -n "$GUARD_PIPE_FD" && "$GUARD_RECEIPT" != "$GUARD_ISSUED" && ${#GUARD_DIGEST} == 64 ]] || exit 1
exec {GUARD_PIPE_FD}<&-
unset MTPLX_GUARD_ATTEST_FD MTPLX_GUARD_ATTEST_NONCE GUARD_ISSUED
trap '/bin/rm -f -- "$GUARD_RECEIPT"; /bin/rmdir -- "${GUARD_RECEIPT:h}" 2>/dev/null || true' EXIT

[[ -x "$VENV" && -f "$PROMPT" && -d "$MODEL" ]] || exit 1
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || {
  print -u2 'worktree is dirty; refusing an unrepeatable bracket'
  exit 1
}
[[ "$(shasum -a 256 "$PROMPT" | awk '{print $1}')" == "$PROMPT_SHA256" ]] || {
  print -u2 'canonical prompt SHA256 mismatch'
  exit 1
}

# Drop inherited selectors. Both context-copy selectors remain absent, which is
# the canonical default used by every bracket cell.
for entry in ${(f)"$(env)"}; do
  name=${entry%%=*}
  [[ "$name" == MTPLX_* ]] && unset "$name"
done
unset MTPLX_CONTEXT_COPY MTPLX_CONTEXT_COPY_TARGET_PREFIX
export PYTHONNOUSERSITE=1 PYTHONPATH="$WORKTREE/scripts:$WORKTREE" HF_HUB_OFFLINE=1
export MTPLX_COMPILED_VERIFY=off MTPLX_DSV4_ATTN=fused MTPLX_DSV4_FP32_ACTIVATIONS=0
export MTPLX_DSV4_HC_COMPILE=1 MTPLX_DSV4_MOE_TAIL=1 MTPLX_DSV4_O_LORA=gather_qmm
export MTPLX_DSV4_SINKHORN_KERNEL=1 MTPLX_DSV4_GUARD_WINDOW_PATH="$GUARD_RECEIPT"
export MTPLX_DSV4_GUARD_WINDOW_SHA256="$GUARD_DIGEST"

exec "$VENV" -u "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
  --adaptive-width-bracket --model "$MODEL" --prompt-file "$PROMPT" \
  --max-tokens 256 --depths 3 --verify-strategy capture_commit --verify-core stock \
  --mtp-history-policy committed --warmup-tokens 0 --out "$BENCH/$TAG"
