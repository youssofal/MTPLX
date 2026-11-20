# MTPLX Benchmarks

Benchmark assets are split by role:

- `prompts/` - prompt suites for code, warm-code, JSON/tool, prose, reasoning, and long-context tests.
- `validators/` - output validators for JSON, tool calls, code, cache equivalence, and stochastic distribution checks.
- `runners/` - executable benchmark harnesses.

Generated summaries and large raw outputs should go under ignored output folders; there is no tracked `reports/` directory.

## Quality instrument

`quality/` answers whether a pack still gives the model's answers and at what length: tasks passed, tokens written to finish each task (thinking and answer apart), draft acceptance by step, and divergence from a reference model over fixed continuations. Its tasks live in `prompts/quality/` (one TOML file each). Everything runs against an OpenAI-compatible server:

```
python -m mtplx.benchmarks.quality run    http://127.0.0.1:18391 out/ref   --seeds 1,2
python -m mtplx.benchmarks.quality corpus out/ref --tokenizer <model folder> --out corpus.jsonl --seeds 1
python -m mtplx.benchmarks.quality score  http://127.0.0.1:18391 out/ref   --corpus corpus.jsonl
python -m mtplx.benchmarks.quality report --arm ref=out/ref --arm pack=out/pack --base ref --reference ref
```

`quality.sensitivity` measures how far each group of weights moves the output when rounded to fewer bits (one load of the full-precision model), and `quality.recipe` turns those rows into a forge recipe under a size budget and a budget of bytes read per token.
