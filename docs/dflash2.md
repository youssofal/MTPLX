# Qwen3.8 DFlash2

MTPLX supports Qwen3.8 DFlash2 as the explicit `dflash2` backend. The bundle
keeps the Qwen target and the DFlash2 draft in separate directories; it is not
a native `mtp.safetensors` sidecar.

## Install and launch

Install the pinned optional dependency:

```sh
.venv/bin/python -m pip install 'mtplx[dflash2]'
# The extra provides dflash==0.1.0.
```

A bundle contains `mtplx_dflash2.json`, `target/`, and `dflash2/`. A normal
launch auto-detects the bundle; `--backend-id dflash2` makes the selection
explicit:

```sh
mtplx serve --model /models/qwen38-dflash2 --backend-id dflash2
mtplx quickstart --model /models/qwen38-dflash2 --backend-id dflash2
mtplx ask --model /models/qwen38-dflash2 --backend-id dflash2 "Explain speculative decoding."
```

DFlash2 defaults are sampler `temperature=1.0`, `top_p=0.95`, `top_k=20`, and
block/depth `5`. `--depth` or `--draft-block-size` overrides the block size.
`--generation-mode ar` or `--no-mtp` deliberately routes to the bundle's
`target/` model with MTP disabled; it never loads the DFlash2 draft or silently
falls back to native MTP.

The portable Homebrew binary path is:

```sh
$(brew --prefix mtplx)/bin/mtplx serve --model /models/qwen38-dflash2
```

For another installation, set `MTPLX_BREW_VENV` to a virtual-environment
directory or Python executable. It contains no credentials:

```sh
MTPLX_BREW_VENV=/path/to/venv mtplx serve --model /models/qwen38-dflash2
```

## Manifest contract

The manifest pins both revisions, records draft precision, and records SHA-256
checksums. Paths are relative to the bundle root:

```json
{
  "schemaVersion": 1,
  "backend": "dflash2",
  "target": {
    "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
    "base_model": "Qwen/Qwen3.8-27B",
    "revision": "<target artifact revision>",
    "precision": "unquantized"
  },
  "draft": {
    "repo": "z-lab/Qwen3.8-27B-DFlash2",
    "revision": "<dflash revision>",
    "precision": "4bit"
  },
  "layout": {"target": "target", "draft": "dflash2"},
  "checksums": {
    "target_config": {
      "path": "target/config.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "draft_config": {
      "path": "dflash2/config.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "draft_weights": [
      {
        "path": "dflash2/model.safetensors",
        "sha256": "<64 lowercase hex characters>"
      }
    ]
  },
  "algorithm": {
    "repo": "z-lab/dflash",
    "revision": "<algorithm revision>",
    "version": "0.1.0"
  }
}
```

The canonical resolver fails closed on an invalid manifest. Do not rename
DFlash2 tensors to `mtp.*` or modify source checkpoints.

## Comparable benchmarks

Use one target, prompt set, sampler, and token budget for every run. The
official single-prompt parity command is:

```sh
TARGET=/models/Qwen3.8-27B
DRAFT=/models/Qwen3.8-27B-DFlash2
PROMPT='Explain speculative decoding in one paragraph.'
TOKENS=128
TEMP=1.0
TOP_P=0.95
TOP_K=20

# Official DFlash MLX single-prompt parity.
dflash generate mlx --model "$TARGET" --draft "$DRAFT" \
  --block-size 5 --max-new-tokens "$TOKENS" \
  --temperature "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" "$PROMPT"
```

For same-harness performance over a prompt file, run DFlash and native MTPLX
with the same target, prompt file, sampler, and token budget:

```sh
BUNDLE=/models/qwen38-dflash2
PROMPTS=/path/to/prompts.jsonl
OUTPUT=/tmp/mtplx-dflash-baseline.jsonl

mtplx dflash-mlx-baseline --model "$TARGET" --draft-model "$DRAFT" \
  --prompts "$PROMPTS" --temperature "$TEMP" --top-p "$TOP_P" \
  --top-k "$TOP_K" --max-tokens "$TOKENS" --block-size 5 \
  --output "$OUTPUT"

mtplx mtp-depth-sweep --model "$TARGET" --prompts "$PROMPTS" \
  --depths 1,2,3,4,5 --compare-ar --temperature "$TEMP" \
  --top-p "$TOP_P" --top-k "$TOP_K" --max-tokens "$TOKENS"
```

The sweep's `--compare-ar` is the native MTPLX target-only AR comparison.
Record target and draft revisions, acceptance length, generated-token
throughput, and peak memory. Compare unquantized, 8-bit, and 4-bit drafts only
after deterministic committed tokens match target-only AR. MTPLX does not use
llama.cpp for this integration.

Known-risk upstream parity references: [z-lab/dflash#159](https://github.com/z-lab/dflash/issues/159)
and [z-lab/dflash#160](https://github.com/z-lab/dflash/issues/160).
