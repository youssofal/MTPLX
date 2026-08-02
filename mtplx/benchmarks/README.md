# MTPLX Benchmarks

Benchmark assets are split by role:

- `prompts/` - prompt suites for code, warm-code, JSON/tool, prose, reasoning, and long-context tests.
- `validators/` - output validators for JSON, tool calls, code, cache equivalence, and stochastic distribution checks.
- `runners/` - executable benchmark harnesses.

Generated summaries and large raw outputs should go under ignored output folders; there is no tracked `reports/` directory.
