#!/bin/sh
# Isolated candidate only. This file never manages the production service.
set -eu
umask 077

SERVICE_ROOT=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/services/deepseek-v4-0731
WORKTREE=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-0731-oQ2e-mtp
PYTHON=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv/bin/python
PYTHON_TARGET=/Users/davidtai/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
ENTRY="$SERVICE_ROOT/candidate_entry.py"
ENCODING="$SERVICE_ROOT/encoding"
REVIEWED_REF=refs/tags/mtplx-dsv4-0731-reviewed
ARTIFACT_VALIDATOR_COMMIT=bbf02944aab3e17be754ba3c88d6aad3c488d10d
ARTIFACT_VALIDATOR_PATH=scripts/deepseek_v4_0731_artifact_check.py
ARTIFACT_VALIDATOR_BLOB_SHA256=672e3bafa8381c5264960d065730d9894b12f832eeb358922e0dd703042ac67b
PORT=8081

die() { printf '%s\n' "deepseek-v4-0731 candidate: $1" >&2; exit 64; }
sha256() { /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }

fixture=${MTPLX_DSV4_0731_TEST_FIXTURE:-}
if [ -n "${MTPLX_DSV4_0731_EXECUTABLE:-}" ] && [ "$fixture" != 1 ]; then
  die "command environment override rejected"
fi
if [ "$fixture" = 1 ]; then
  [ "${1:-}" = --print-command ] || die "fixture mode only permits --print-command"
  printf '%s\n' "$PYTHON $ENTRY (fixed 127.0.0.1:$PORT)"
  exit 0
fi
[ "$#" -eq 0 ] || die "arguments are not accepted"

[ -L "$PYTHON" ] && [ "$(/usr/bin/readlink "$PYTHON")" = "$PYTHON_TARGET" ] || die "trusted python link changed"
[ -x "$PYTHON_TARGET" ] && [ ! -L "$PYTHON_TARGET" ] || die "trusted python target is missing or unsafe"
[ "$(sha256 "$PYTHON_TARGET")" = 96793b100c947cdc81a38e8fb8c9c1889abccda9840ce1bef58d372bf3f2c263 ] || die "trusted python hash changed"
[ -f "$ENTRY" ] && [ ! -L "$ENTRY" ] || die "candidate entrypoint is missing or unsafe"
[ "$(sha256 "$ENTRY")" = 35b268195eba1af59028f96dd5e6b474d76dcc42844e610743c48a55771d2268 ] || die "candidate entrypoint hash changed"
[ -d "$MODEL" ] && [ ! -L "$MODEL" ] || die "pinned model path is missing or unsafe"
[ "$(sha256 "$MODEL/config.json")" = 6d0297a4329d55dccf3cd48fd168efea8044996245195d518a9e8aaa14906d3e ] || die "model configuration hash changed"
[ "$(sha256 "$MODEL/model.safetensors.index.json")" = 9edcd0db7e6b8f0b8e02978d73c30083b2aa64c2e3a8fd77d3b776a4efb4bc91 ] || die "model index hash changed"
[ "$(sha256 "$MODEL/tokenizer_config.json")" = 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547 ] || die "model tokenizer configuration hash changed"
[ "$(sha256 "$ENCODING/SHA256SUMS")" = 6758dfda8a39afdd00d907606c42c1a268289c463351b9628ac07f4f916d7d0a ] || die "official encoding manifest hash changed"
(cd "$ENCODING" && /usr/bin/shasum -a 256 -c SHA256SUMS >/dev/null) || die "official encoding/vector asset hash changed"

reviewed_commit=$(/usr/bin/git -C "$WORKTREE" rev-parse --verify "${REVIEWED_REF}^{commit}") || die "reviewed commit ref is missing"
current_commit=$(/usr/bin/git -C "$WORKTREE" rev-parse --verify HEAD) || die "worktree HEAD is missing"
[ "$current_commit" = "$reviewed_commit" ] || die "worktree is not the exact reviewed commit"
[ -z "$(/usr/bin/git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all)" ] || die "reviewed worktree is not clean"

# Execute the exact reviewed validator blob. It pins tokenizer.json plus every
# one of the 20 model shards and rejects unknown, missing, or changing files.
validator_blob_sha=$(
  /usr/bin/git -C "$WORKTREE" cat-file blob "$ARTIFACT_VALIDATOR_COMMIT:$ARTIFACT_VALIDATOR_PATH" |
    /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
) || die "reviewed artifact validator is unavailable"
[ "$validator_blob_sha" = "$ARTIFACT_VALIDATOR_BLOB_SHA256" ] || die "reviewed artifact validator hash changed"
/usr/bin/git -C "$WORKTREE" cat-file blob "$ARTIFACT_VALIDATOR_COMMIT:$ARTIFACT_VALIDATOR_PATH" |
  "$PYTHON" - "$MODEL" >/dev/null || die "reviewed artifact validation failed"

exec /usr/bin/env -i \
  HOME=/Users/davidtai \
  LC_ALL=C \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$WORKTREE" \
  VIRTUAL_ENV=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service/.venv \
  "$PYTHON" "$ENTRY"
