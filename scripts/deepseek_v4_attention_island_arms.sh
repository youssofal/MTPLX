#!/bin/zsh
# Canonical one-load attention-island bracket. Guard-wrapper only.
set -euo pipefail

[[ "${MTPLX_DSV4_ATTENTION_ISLAND_POSTFLIGHT_WRAPPER:-}" == 1 ]] || {
  print -u2 'invoke deepseek_v4_attention_island_guarded.py, not this child'
  exit 1
}

WORKTREE=${0:A:h:h}
# The guard wrapper supplies these deployment-specific locations.  Keeping
# them out of the source makes the exact artifact identity—not one developer's
# filesystem—the reproducibility contract (mirrors the adaptive-width pair).
VENV=${MTPLX_DSV4_PYTHON:-python3}
BENCH=${MTPLX_DSV4_BENCH_DIR:?set MTPLX_DSV4_BENCH_DIR}
MODEL=${MTPLX_DSV4_MODEL_PATH:?set MTPLX_DSV4_MODEL_PATH}
PROMPT=${MTPLX_DSV4_PROMPT_FILE:?set MTPLX_DSV4_PROMPT_FILE}
PROMPT_SHA256=ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33
CONFIG_SHA256=c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f
INDEX_SHA256=c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8
# 0 disables the exact wired-limit gate (one operator's sysctl tuning, not a
# portable invariant); set it to pin a deployment's canonical value.
EXPECTED_WIRED_LIMIT_MB=${MTPLX_DSV4_EXPECTED_WIRED_LIMIT_MB:-114688}

(( $# == 3 )) || {
  print -u2 'expected tag, authorized commit, and wrapper-observed commit'
  exit 1
}
TAG=$1
EXPECTED_SOURCE_COMMIT=$2
WRAPPER_OBSERVED_SOURCE_COMMIT=$3
[[ -n "$TAG" && "$TAG" != '.' && "$TAG" != '..' \
  && "$TAG" =~ '^[A-Za-z0-9][A-Za-z0-9._-]*$' ]] || {
  print -u2 'invalid attention-island tag: expected a safe basename'
  exit 1
}
[[ "$EXPECTED_SOURCE_COMMIT" =~ '^[0-9a-f]{40}$' \
  && "$WRAPPER_OBSERVED_SOURCE_COMMIT" =~ '^[0-9a-f]{40}$' ]] || {
  print -u2 'attention-island commits must be exact lowercase 40-hex SHAs'
  exit 1
}
CHILD_OBSERVED_SOURCE_COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)
[[ "$EXPECTED_SOURCE_COMMIT" == "$WRAPPER_OBSERVED_SOURCE_COMMIT" \
  && "$EXPECTED_SOURCE_COMMIT" == "$CHILD_OBSERVED_SOURCE_COMMIT" ]] || {
  print -u2 'attention-island source commit changed after wrapper authorization'
  exit 1
}
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || {
  print -u2 'attention-island worktree is dirty after wrapper authorization'
  exit 1
}

GUARD_PIPE_FD=${MTPLX_GUARD_ATTEST_FD:-}
GUARD_ISSUED=$("$VENV" -u "$WORKTREE/scripts/deepseek_v4_guard_window.py" issue)
GUARD_RECEIPT=${GUARD_ISSUED%%$'\t'*}
GUARD_DIGEST=${GUARD_ISSUED#*$'\t'}
[[ -n "$GUARD_PIPE_FD" && "$GUARD_RECEIPT" != "$GUARD_ISSUED" \
  && ${#GUARD_DIGEST} == 64 ]] || {
  print -u2 'malformed attention-island guard-window metadata'
  exit 1
}
exec {GUARD_PIPE_FD}<&-
unset MTPLX_GUARD_ATTEST_FD MTPLX_GUARD_ATTEST_NONCE GUARD_ISSUED
GUARD_DIR=${GUARD_RECEIPT:h}
CHILD_STATUS="$BENCH/$TAG-child-status.json"
CHILD_STATUS_TMP="$BENCH/.$TAG-child-status.$$.tmp"
cleanup_guard_receipt() {
  /bin/rm -f -- "$CHILD_STATUS_TMP"
  /bin/rm -f -- "$GUARD_RECEIPT"
  /bin/rmdir -- "$GUARD_DIR" 2>/dev/null || true
}
trap cleanup_guard_receipt EXIT

[[ -x "$VENV" && -f "$PROMPT" && -d "$MODEL" \
  && -f "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" ]] || {
  print -u2 'attention-island interpreter, prompt, model, or benchmark is missing'
  exit 1
}
actual_prompt_sha=$(shasum -a 256 "$PROMPT" | awk '{print $1}')
actual_config_sha=$(shasum -a 256 "$MODEL/config.json" | awk '{print $1}')
actual_index_sha=$(shasum -a 256 "$MODEL/model.safetensors.index.json" | awk '{print $1}')
[[ "$actual_prompt_sha" == "$PROMPT_SHA256" \
  && "$actual_config_sha" == "$CONFIG_SHA256" \
  && "$actual_index_sha" == "$INDEX_SHA256" ]] || {
  print -u2 'attention-island canonical prompt/config/index identity mismatch'
  exit 1
}

# Source identity is already pinned by the exact-commit + clean-worktree
# checks above (rev-parse HEAD equality and empty porcelain): a frozen SHA
# manifest of individual files duplicated that guarantee and broke on every
# legitimate commit to those files (PR #223 review edit).

if (( EXPECTED_WIRED_LIMIT_MB > 0 )); then
  observed_wired_limit=$(/usr/sbin/sysctl -n iogpu.wired_limit_mb)
  [[ "$observed_wired_limit" == "$EXPECTED_WIRED_LIMIT_MB" ]] || {
    print -u2 "wired limit changed: expected $EXPECTED_WIRED_LIMIT_MB, got $observed_wired_limit"
    exit 1
  }
fi

for entry in ${(f)"$(env)"}; do
  name=${entry%%=*}
  [[ "$name" == MTPLX_* ]] && unset "$name"
done
unset MTPLX_CONTEXT_COPY MTPLX_CONTEXT_COPY_TARGET_PREFIX
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKTREE/scripts:$WORKTREE"
export HF_HUB_OFFLINE=1
export MTPLX_COMPILED_VERIFY=off
export MTPLX_DSV4_ATTN=fused
export MTPLX_DSV4_FP32_ACTIVATIONS=0
export MTPLX_DSV4_HC_COMPILE=1
export MTPLX_DSV4_MOE_TAIL=1
export MTPLX_DSV4_O_LORA=gather_qmm
export MTPLX_DSV4_SINKHORN_KERNEL=1
export MTPLX_DSV4_ATTN_PROJ_WIDE_M3=1
export MTPLX_DSV4_ATTENTION_ISLAND=1
export MTPLX_DSV4_GUARD_WINDOW_PATH="$GUARD_RECEIPT"
export MTPLX_DSV4_GUARD_WINDOW_SHA256="$GUARD_DIGEST"

set +e
"$VENV" -u "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
  --attention-island-bracket --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
  --model "$MODEL" --prompt-file "$PROMPT" --max-tokens 256 --depths 3 \
  --verify-strategy capture_commit --verify-core stock \
  --mtp-history-policy committed --warmup-tokens 0 --out "$BENCH/$TAG"
BENCHMARK_EXIT=$?
set -e
print -r -- "{\"schema_version\":1,\"kind\":\"attention_island_child_status\",\"tag\":\"$TAG\",\"expected_source_commit\":\"$EXPECTED_SOURCE_COMMIT\",\"observed_source_commit\":\"$CHILD_OBSERVED_SOURCE_COMMIT\",\"benchmark_exit_code\":$BENCHMARK_EXIT}" > "$CHILD_STATUS_TMP"
/bin/chmod 600 "$CHILD_STATUS_TMP"
/bin/mv -f -- "$CHILD_STATUS_TMP" "$CHILD_STATUS"
exit "$BENCHMARK_EXIT"
