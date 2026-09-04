# Representative sanitized Codex Responses request contracts

Reduced and sanitized from a local `codex-cli 0.146.0` capture made on
2026-08-01 against a loopback Responses provider. No request was sent
externally.

- `codex_0.146.0_responses_default.sanitized.json` preserves the default
  `reasoning: {"summary":"auto"}` shape.
- `codex_0.146.0_responses_xhigh.sanitized.json` preserves the configured
  `reasoning: {"effort":"xhigh","summary":"auto"}` shape.

Sanitization replaces volatile IDs, paths, timestamps, and long prompt text.
The fixtures retain representative top-level field types, structured `input`
message/content examples, mixed top-level function/namespace/web-search tools,
nested namespace function shapes and JSON Schemas, `client_metadata`, and
streaming flags. They are reduced contracts, not complete structured-input
copies: the source capture's three developer content parts are represented by
three parts, while its three parts in the first user message are reduced to two
(the `recommended_plugins` placeholder is omitted). The source capture carried
23 tools (9 function, 13 namespace, 1 web_search); the fixtures reduce that
list to one representative tool of each type plus two nested namespace
functions.

These sanitized fixtures are committed as contract tests. The 878 KB raw
capture is intentionally excluded.
