# DeepSeek-V4-Flash-0731 service

This directory owns both the reviewed isolated candidate and the pinned local
production profile. The candidate remains separate under
`com.tea.deepseek-v4-0731.candidate` on `127.0.0.1:8081`.

## Production profile

`production_entry.py`, `launch_production.sh`, and
`com.tea.deepseek-v4.plist` define the local production service on
`127.0.0.1:8080`. The launcher acquires the exclusive GPU lock before model
construction and pins the exact model hashes, DeepSeek-V4 topology, MLX 0.32.0,
official 0731 encoder, 262,144-token context, greedy sampling, K3, and the
construction-bound optimized DSpark route. Cline should use model ID
`mtplx-deepseek-v4-flash-0731-2.4bit-k3`.

The following warm production probes were measured under that lock on an Apple
M5 Max with 128 GB unified memory. MLX active memory stayed flat at 86.73 GiB
across the probes; the process-wide 139.71 GiB MLX peak includes model loading
and first compilation and is not concurrent resident memory.

| probe | prompt tokens | output tokens | prefill tok/s | decode tok/s | TTFT s | accepted / drafted | active GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact-text smoke | 17 | 20 | 20.85 | 44.10 | 0.92 | 14 / 15 | 86.73 |
| coding response | 40 | 109 | 44.59 | 38.63 | 1.00 | 75 / 101 | 86.73 |
| native tool call | 295 | 55 | 154.79 | 43.16 | 2.01 | 39 / 45 | 86.73 |
| streamed tool call | 295 | 55 | 166.22 | 43.48 | 1.88 | 39 / 45 | 86.73 |

The exact-text smoke deterministically repeated the requested marker twice, so
it is a liveness/performance probe rather than an instruction-following pass.
The coding response was coherent, and both nonstream and stream probes emitted
a valid OpenAI tool call with incrementally valid streamed arguments. A replay
of the prior 540 KB Cline transcript remained compute-bound for more than 1,323
seconds and exceeded the 900-second client timeout. System free-memory pressure
oscillated between 14% and 44% with zero throttled pages instead of growing
monotonically. That establishes bounded chunk cleanup, but not acceptable
long-context prefill latency; the replay is not listed as a successful TPS row.

The historical K0-K3 diagnostic and the promoted physical-M3 K2 bracket remain
in `docs/perf/receipts/deepseek-v4-0731-dspark.md`. Those runs used different
target geometry and must not be compared directly with this native-M4 K3
production profile.

## Candidate profile

The `encoding/` directory vendors the exact official Python encoder and four
input/output vectors from
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
`candidate_entry.py` verifies all nine assets and runs every official vector at
construction. It then installs the encoder directly at MTPLX's prompt-ID call
site and installs the official DSML parser at the nonstream and streaming
response call sites. The stream translator retains ordinary preamble text while
continuing to scan later chunks and holding any suffix that could grow into a
split DSML marker; raw markup is never released. The complete turn still passes
through the official parser. Plain no-tool turns also use that parser and report
its engagement. No tokenizer-template, stock prompt, or stock completion-parser
fallback remains in the enabled 0731 lane. Per-request observability reports
`backend_chat_encoding=deepseek-v4-flash-0731-official`.

`launch_candidate.sh` accepts no production arguments. Its only test seam is
`MTPLX_DSV4_0731_TEST_FIXTURE=1 ... --print-command`, which cannot start the
service. A real launch requires:

- the exact commit referenced by `refs/tags/mtplx-dsv4-0731-reviewed`;
- a completely clean worktree;
- the pinned interpreter, model config/index, manifest, encoder, and official
  vector hashes;
- the exact reviewed artifact validator at commit `bbf02944`, which hashes the
  0731 tokenizer and all 20 Safetensors shards and checks their index closure,
  topology, and quantization assignment;
- the separately pinned official `tokenizer_config.json`; and
- the fixed, absolute entrypoint and minimal `env -i` environment.

`promote_cutover.py` remains an explicit operator action. Before it can stop a
service it requires `--promote`, a detached SSH signature over a strict-schema
candidate receipt, a passing 8081 preflight/smoke, a separately hashed
production plist, the nonblocking GPU lock, and exact current launchd
label/PID/listener/plist identity. Candidate model IDs are taken only from the
signed receipt for cutover verification; the prior model IDs are used only to
verify rollback. The same lock remains held through restoration and the real
`/v1/models` plus exact `content.strip() == "READY"`/`finish_reason=stop` smoke.
After the new plist is bootstrapped, its hash, launchd PID, and ownership of the
8080 listener are reattested under the lock before any HTTP identity or readiness
probe is allowed. The prior live identity is pinned to `com.tea.qwen` serving
`mtplx-qwen36-27b-optimized-quality`; the target is pinned to
`com.tea.deepseek-v4-0731.production`. Promotion parses `Label` and
`ProgramArguments` from one descriptor read, then writes those exact bytes to a
content-addressed, owner-only (`0400` file in a `0700` directory), same-filesystem
snapshot and fsyncs it before any service action. Both promotion and rollback
bootstrap that durable snapshot path rather than the mutable source path, and
post-bootstrap attestation requires launchd's loaded path, current snapshot
bytes and metadata, program, and arguments to still match. A snapshot remains
on disk for as long as launchd references it, including after all Python context
managers exit. The unreferenced side is removed only after a successful
`launchctl print` proves that its label is absent or loaded from another path;
an ambiguous probe preserves the snapshot and fails closed. Successful backend
readiness is the cutover commit point: prior-snapshot cleanup happens afterward,
and a cleanup failure reports a warning without stopping the verified production
service or entering rollback. Backend PID attestation is bound specifically to
the listener on `127.0.0.1:8080`; a same-port gateway on another interface is
ignored, while wildcard listeners and multiple loopback owners are rejected.

Receipts have an exact allowlist and recursively reject local paths, request
content, tool schemas, secrets, argv/env, and captured process output.

The reviewed, dedicated signer list lives at
`~/.config/mtplx/deepseek-v4-0731-allowed-signers` (owner-only mode `0600`),
with its SHA-256 pinned in `promote_cutover.py`. Sign a receipt without
printing key material using `/usr/bin/ssh-keygen -Y sign -f
~/.config/mtplx/deepseek-v4-0731-signing -n mtplx-deepseek-v4-0731
candidate-receipt.json`; this dedicated key is not a reused login key.
Promotion rejects a missing, changed, wrongly owned, or group/world-writable
signer list; it does not discover trust at runtime.
