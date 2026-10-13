# MiniCPM-V 4.7 35B-A3B

This adapter supports the qualified `minicpmv4_7` / `qwen3_5_moe_text` geometry: 40 layers, width 2048, 256 experts with top-8 routing, a 27-layer width-1152 vision tower, and canvas rotary positions. Other geometries raise `ValueError` before constructing weights. Install MLX-VLM with MiniCPM-V 4.7 support; imports remain lazy for other models and no-MLX CLI use.

The adapter returns MTPLX logits/hidden states, skips unused vocabulary projections, and exposes prepared single-prompt vision prefill with cache-aligned embeddings and canvas positions. `clear_vision()` closes that request's state. MTPLX's main server does not yet dispatch MiniCPM image requests through this preparation API; use the SDK below or the published release launcher.

Quantized embeddings require the [MLX-VLM dtype fix](https://github.com/Blaizzy/mlx-vlm/pull/2511). The [published release](https://huggingface.co/Shiftedx/MiniCPM-V-4.7-35B-A3B-Attention8-BF16Recurrence-MLX-MTPLX/tree/6080763a2fa239127b30480e3866be66b3cf80cb) includes a bootstrap with exact runtime dependencies and both patches. The original base checkpoint declares MTP but supplies no MTP tensors; load this quant with `mtp=False`. Its separate publisher DSpark draft uses MLX-VLM's verifier.

```python
from pathlib import Path
from PIL import Image
import mlx.core as mx
from mlx_vlm.utils import load_processor
from mtplx.runtime import load
from mtplx.generation import generate_ar
from mtplx.sampling import SamplerConfig

path = Path("MiniCPM-V-4.7-Attention8")
mx.set_cache_limit(2 * 1024**3)
with mx.stream(mx.cpu):
    runtime = load(path, mtp=False)
processor = load_processor(path)
prompt = processor.apply_chat_template(
    [{"role": "user", "content": "<image>\nRead the text. Return text only."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = processor(text=prompt, images=[Image.open("ocr.png").convert("RGB")])
ids = runtime.model.prepare_vision(inputs)
try:
    result = generate_ar(runtime, ids, max_tokens=128,
                         sampler=SamplerConfig(temperature=0), seed=42)
    print(result.text)
finally:
    runtime.model.clear_vision()
```

Run native AR with `MTPLX_SUSTAINED_PREFILL=1 MTPLX_PREFILL_CHUNK_SIZE=512 MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS=0`. CPU materialization and the allocator cap are deployment choices; this PR leaves global defaults intact.

## Evidence

- Model weights: HF revision `5ec49aa0f69e263e41d29ca0c64c4535c83d91ba`; direct BF16 conversion, 22.17 GiB Attention8/BF16-recurrence mixed affine quant.
- Hardware/date: Apple M4 Max, 64 GiB, 2026-10-13. MLX 0.32.2, MLX-LM 0.31.3, Transformers 5.14.1, MTPLX 2.12.2, MLX-VLM `9790231827a890eade1a626643a6bc94a1317c72` plus the dtype fix. Fan mode: not recorded; these numbers describe the deployment, not a controlled patch speedup.
- Local contract: temperature 0, seed 42, thinking off, unquantized KV caches, 512-token sustained prefill, 2 GiB allocator cache cap. Native MTPLX AR: 73.66 tok/s median. Separate DSpark block-8 verifier: 191.80 tok/s warm generation and 110.16 tok/s end-to-end medians, three workloads × three repeats with a 128-token cap.
- Local gates: 2,096 stored arrays match source-derived conversion; loaded/offloaded log probabilities match on eight text cases; BF16-parent mean KL 0.000834 nats across 11 cases. API 9/9, image AR/DSpark 10/10, video SDK AR/DSpark 2/2; hard code/logic 8/10.
- [Post-publication Shiftedx Bench](https://huggingface.co/datasets/Shiftedx/shiftedx-bench/tree/9a43027a84dc4a4a283220f58314b5768245ec58): quality 5/10, context 10/15, tools 6/6, agentic 2/2, vision 4/4. The 65K/131K requests complete but fail strict retrieval. These results do not establish BF16 full-generation parity or the advertised 262K window.

The synthetic regressions cover architecture dispatch/rejection, logits and hidden-state contracts, head suppression, two-chunk canvas prefill, mismatched prompt rejection, decode transition, request-state reset, single-prompt admission, and position-argument forwarding. The original attention wrapper raises `TypeError` on the position fixture. A full-weight smoke test loads this in-tree adapter through `mtplx.runtime.load` and returns `42` for arithmetic and `BLUE 739` for OCR.
