#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
cd "$root"

fail=0
candidate_files="$(mktemp)"

report_failure() {
  local title="$1"
  local file="$2"
  echo "::error::$title"
  sed -n '1,120p' "$file"
  fail=1
}

collect_candidate_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files --cached --others --exclude-standard -z >"$candidate_files"
  else
    find . \( \
      -path './.git' -o \
      -path './.venv' -o \
      -path './venv' -o \
      -path './.tox' -o \
      -path './.nox' -o \
      -path './.pytest_cache' -o \
      -path './.mypy_cache' -o \
      -path './__pycache__' -o \
      -path './build' -o \
      -path './dist' -o \
      -path './*.egg-info' \
    \) -prune -o -type f -print0 >"$candidate_files"
  fi
}

for_each_candidate() {
  local callback="$1"
  while IFS= read -r -d '' path; do
    [[ -f "$path" ]] || continue
    "$callback" "$path"
  done <"$candidate_files"
}

print_public_path() {
  local path="$1"
  if [[ "$path" == ./* ]]; then
    printf '%s\n' "$path"
  else
    printf './%s\n' "$path"
  fi
}

collect_candidate_files

large_files="$(mktemp)"
record_large_file() {
  local path="$1"
  if find "$path" -prune -type f -size +50M -print | grep -q .; then
    print_public_path "$path"
  fi
}
for_each_candidate record_large_file >"$large_files"
if [[ -s "$large_files" ]]; then
  report_failure "Files larger than 50 MB are not allowed in the source repo" "$large_files"
fi

model_artifacts="$(mktemp)"
record_model_artifact() {
  local path="$1"
  case "$path" in
    # The Mac app intentionally bundles small adapter weights as
    # product resources; stray checkpoints everywhere else stay
    # forbidden (and the 50 MB cap above still applies here).
    apps/MTPLXApp/Sources/MTPLXAppCore/Resources/*|./apps/MTPLXApp/Sources/MTPLXAppCore/Resources/*)
      return 0
      ;;
    # The deepseek_v4 parity suite gates the whole forward against a 2.3 MB
    # deterministic golden (toy dims, float32 arrays, no pickled objects) —
    # a test fixture, not a checkpoint. Exempted by exact path; everything
    # else under tests/ stays forbidden.
    tests/fixtures/deepseek_v4_parity_golden.npz|./tests/fixtures/deepseek_v4_parity_golden.npz)
      return 0
      ;;
  esac
  case "$path" in
    *.safetensors|*.gguf|*.mlx|*.bin|*.npz|*.npy)
      print_public_path "$path"
      ;;
  esac
}
for_each_candidate record_model_artifact >"$model_artifacts"
if [[ -s "$model_artifacts" ]]; then
  report_failure "Model artifacts do not belong in Git" "$model_artifacts"
fi

workspace_residue="$(mktemp)"
record_workspace_residue() {
  local path="$1"
  case "$path" in
    models|models/*|./models|./models/*|\
    outputs|outputs/*|./outputs|./outputs/*|\
    REFERENCES:TOOLS|REFERENCES:TOOLS/*|./REFERENCES:TOOLS|./REFERENCES:TOOLS/*|\
    "DEEP RESEARCH HANDOFF"|"DEEP RESEARCH HANDOFF"/*|./"DEEP RESEARCH HANDOFF"|./"DEEP RESEARCH HANDOFF"/*|\
    "IDE RESEARCH"|"IDE RESEARCH"/*|./"IDE RESEARCH"|./"IDE RESEARCH"/*|\
    "ProAgent Details"|"ProAgent Details"/*|./"ProAgent Details"|./"ProAgent Details"/*|\
    *.webui_secret_key|*/.webui_secret_key)
      print_public_path "$path"
      ;;
  esac
}
for_each_candidate record_workspace_residue >"$workspace_residue"
if [[ -s "$workspace_residue" ]]; then
  report_failure "Private workspace residue found" "$workspace_residue"
fi

bad_names="$(mktemp)"
record_bad_name() {
  local path="$1"
  if [[ "$path" == *" "* || "$path" == *":"* ]]; then
    print_public_path "$path"
  fi
}
for_each_candidate record_bad_name >"$bad_names"
if [[ -s "$bad_names" ]]; then
  report_failure "Tracked/public paths must not contain spaces or colons" "$bad_names"
fi

secret_matches="$(mktemp)"
rg --hidden --line-number \
  --glob '!.git/**' \
  --glob '!.venv/**' \
  --glob '!venv/**' \
  --glob '!.tox/**' \
  --glob '!.nox/**' \
  --glob '!.pytest_cache/**' \
  --glob '!.mypy_cache/**' \
  --glob '!__pycache__/**' \
  --glob '!dist/**' \
  --glob '!build/**' \
  --glob '!*.egg-info/**' \
  -e '\bhf_[A-Za-z0-9]{30,}' \
  -e '\bgh[pousr]_[A-Za-z0-9]{30,}' \
  -e '\bgithub_pat_[A-Za-z0-9_]{30,}' \
  -e '\bsk-[A-Za-z0-9_-]{30,}' \
  -e '\bxox[baprs]-[A-Za-z0-9-]{20,}' \
  -e '\bAKIA[0-9A-Z]{16}\b' \
  -e 'BEGIN [A-Z ]*PRIVATE KEY' \
  -e '(?i)(api[_-]?key|client[_-]?secret|password|auth[_-]?token)[" ]*[:=][ ]*["'\''][A-Za-z0-9+/=_-]{20,}["'\'']' \
  . >"$secret_matches" || true
# The patterns above match credential-shaped strings (token prefixes
# at credential length, key blocks, long literals assigned to
# secret-named variables) rather than keyword mentions. The keyword
# grep this replaced drowned the 1.0.0 tree in false positives:
# max_tokens, hf_model, and local placeholder keys like "mtplx-local"
# are vocabulary here, not leaks.

filtered_secrets="$(mktemp)"
awk '
  /scripts\/hygiene_scan\.sh/ { next }
  /\.gitignore:/ { next }
  /PREFIX_DIVERGENCE_AT_TOKEN/ { next }
  /MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS/ { next }
  /hf_path/ { next }
  /mtplx\/artifacts\.py:.*(_hf_|hf_hub_|huggingface_hub|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /mtplx\/hf_loader\.py:.*(hf_|_hf_|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /mtplx\/cli\.py:.*(hf_loader|DEFAULT_HF_MODEL_ID)/ { next }
  /mtplx\/commands\/public\.py:.*(hf_loader|hf_cache_report|huggingface)/ { next }
  /examples\/openwebui\.md:.*(WEBUI_SECRET_KEY|OPENAI_API_KEYS)/ { next }
  /tests\/test_artifacts\.py:.*(_hf_|test_hf|hf_)/ { next }
  /tests\/test_hf_loader\.py:.*(hf_|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)/ { next }
  /tests\/test_no_mlx_imports\.py:.*test_run_reports_uncached_hf_model/ { next }
  { print }
' "$secret_matches" >"$filtered_secrets"

if [[ -s "$filtered_secrets" ]]; then
  report_failure "Potential secret patterns found" "$filtered_secrets"
fi

exit "$fail"
