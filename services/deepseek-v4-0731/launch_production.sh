#!/bin/zsh
set -euo pipefail

ROOT=/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-prod
SERVICE_ROOT=$ROOT/services/deepseek-v4-0731
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed
PYTHON=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
ENTRY=$SERVICE_ROOT/production_entry.py
LOCK=/tmp/mtplx-gpu-exclusive.lock

die() { print -u2 -- "[deepseek-v4-0731] preflight failed: $*"; exit 1; }
sha256() { /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }

[[ -x "$PYTHON" ]] || die "MLX 0.32 interpreter is unavailable"
[[ -f "$ENTRY" ]] || die "production entrypoint is unavailable"
[[ -d "$MODEL" && ! -L "$MODEL" ]] || die "pinned model is unavailable"
[[ "$(sha256 "$MODEL/config.json")" == 44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f ]] || die "model config changed"
[[ "$(sha256 "$MODEL/model.safetensors.index.json")" == f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861 ]] || die "model index changed"
[[ "$(sha256 "$MODEL/tokenizer_config.json")" == 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547 ]] || die "tokenizer config changed"

"$PYTHON" - "$MODEL" <<'PY' || die "mlx==0.32.0 or model invariant differs"
import importlib.metadata
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
if importlib.metadata.version("mlx") != "0.32.0":
    raise SystemExit("MLX must be exactly 0.32.0")
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
expected = {
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"unexpected {key}")
if int(config.get("max_position_embeddings", 0)) < 262144:
    raise SystemExit("model context is below 262144")
PY

export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT"
export MTPLX_MEMORY_LIMIT_BYTES=111669149696
export MTPLX_WIRED_LIMIT_BYTES=111669149696

exec "$PYTHON" - "$LOCK" "$PYTHON" "$ENTRY" <<'PY'
import fcntl
import os
import sys

lock_path, python, entry = sys.argv[1:]
lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError as error:
    raise SystemExit("GPU lock is already held") from error
os.set_inheritable(lock_fd, True)
os.environ["MTPLX_GPU_LOCK_FD"] = str(lock_fd)
os.execv(python, [python, entry])
PY
