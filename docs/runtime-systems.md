# Read-only runtime systems

`GET /v1/mtplx/systems` uses the same authentication and rate-limit middleware as other `/v1` endpoints. The server now publishes a concrete `serving` provider from its existing CPU-side state whenever this endpoint is read. No separate polling thread or second runtime controller is introduced.

The provider exposes loaded/released availability, AR/MTP mode, foreground/dashboard request counts, completed/cancelled counters, and whether MTP is enabled. Its phase changes between unavailable, unknown, idle and busy; unknown counters are null, not invented zeroes. These aggregate scalars are sampled sequentially, not promised to be a transactional multi-counter snapshot.

Only fixed enums, booleans and bounded integers are published. Model paths, API keys, request/client IDs, prompt/tool content and exception messages are not copied. A failed refresh replaces prior healthy data with an unavailable status and fixed reason rather than leaving stale success visible. Publication is serialized per provider and the registry returns detached JSON.

The provider never loads or runs a model, acquires the model lock, resizes caches, changes configuration or touches the Metal allocator. Existing inference/runtime owners remain authoritative. The generic registry does not import this adapter; the server wires it as an optional refresh callback. Other components can publish independent bounded statuses through the same registry.

The Systems dashboard in #365 can render this actual provider without a placeholder. Production-app tests use the real `create_app` and HTTP route with an injected runtime test state, including authentication and live counter changes. That proves the wiring and contract, not a physical-model or packaged-desktop smoke.
