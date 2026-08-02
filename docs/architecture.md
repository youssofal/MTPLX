# Architecture

MTPLX is organized around a small public CLI, profile selection, a model compatibility registry, and a backend that owns architecture-specific MTP details.

```mermaid
flowchart TB
    cli["CLI surface"]
    profiles["Profiles"]
    spec["Speculative sampling"]
    registry["Compatibility registry"]
    backend["Qwen3NextMTPBackend"]
    session["SessionBank"]
    server["OpenAI server"]

    cli --> profiles
    profiles --> spec
    spec --> registry
    registry --> backend
    server --> session
    server --> backend
```

The speculative sampler should remain backend-agnostic. Backends provide proposal and verification mechanics for a specific model family. The backend node above stands for the whole family set — Qwen3-Next (the promoted default), DeepSeek MTP, DeepSeek-V4, GLM, Hy-V3, MiMo, Nemotron-H, Step3.5, and Gemma4, plus the Laguna AR-only route.
