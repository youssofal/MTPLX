# Embedded Hermes Agent Selection

Date: 2026-08-01
Status: Approved

## Objective

Turn the existing Hermes surface in MTPLX into a working embedded Hermes
client. A user selects a Hermes profile, then selects or creates a session in
that profile, and uses the agent directly inside the MTPLX GUI. The embedded
session uses the model currently served by MTPLX without changing the selected
profile's persistent provider configuration or disrupting Telegram, the root
Hermes gateway, Hermes Desktop, or other Hermes clients.

Sessions created or resumed from MTPLX remain ordinary Hermes sessions in the
selected profile. They must remain visible and resumable later from Hermes
Desktop, the Hermes TUI, and MTPLX.

## Existing Integration

MTPLX already contains most of the presentation and client structure:

- `HermesIntegration` discovers the default Hermes home and named profiles.
- `HermesAgentStore` models profile selection, session lists, transcripts,
  tool events, and JSON-RPC calls.
- `HermesOverlay` contains the intended Profiles, Agents, transcript, and
  composer layout.
- `HermesGatewayClient` implements the JSON-RPC/WebSocket client.
- `HermesSidecar` owns a child process and its connection information.

The embedded surface is currently disabled because
`HermesIntegration.nativeDashboardSupported` is false and `startDashboard()`
unconditionally reports an incompatible Hermes build. That assumption is now
stale: the installed Hermes exposes `hermes serve`, a headless JSON-RPC and
WebSocket backend, and supports a profile-scoped `--isolated` process.

The current messaging diagnostic is separate from this feature. It inspects
only the root `.env`, which is not sufficient for a multiplex gateway whose
Telegram credentials live in a routed profile. Embedded agent selection must
not depend on that root-only diagnostic and must not mutate gateway routing.

## User Experience

The existing Hermes overlay becomes the embedded agent surface.

### Profile selection

The Profiles section lists the default Hermes home and every valid named
profile under `~/.hermes/profiles`. All profiles are selectable. Each row shows
one of these routing states:

- **MTPLX**: the persistent profile already targets the active MTPLX endpoint.
- **External**: the persistent profile uses another provider or endpoint; the
  embedded sidecar will use a process-local MTPLX route.
- **Unavailable**: the profile cannot be read or cannot be started safely.

Selecting a profile starts or reuses an MTPLX-owned isolated Hermes sidecar for
that profile and loads its sessions. Profile and session selection are saved in
the existing `lastHermesProfile`, `lastHermesSessionID`, and
`lastHermesSessionTitle` settings.

### Session selection

The Agents section lists saved sessions for the selected profile with title,
last activity, and state:

- **Ready**: resumable in MTPLX.
- **Running in MTPLX**: owned by the current embedded sidecar.
- **Externally active**: active in another Hermes surface and not writable from
  MTPLX at the same time.

The user may resume a ready session or create a new one. An externally active
session remains visible but read-only; MTPLX offers to create a new session
instead of attempting concurrent writes.

### Embedded conversation

The right pane displays the selected profile, session title, and MTPLX routing
state in its header. It renders the persisted transcript, live assistant
streaming, reasoning, tool progress, approval or clarification requests, and
errors. The composer submits prompts over Hermes' native JSON-RPC protocol.

The GUI does not embed Hermes Desktop and does not reimplement the agent loop.
Hermes remains responsible for sessions, tool execution, streaming events, and
persistence.

## Runtime Architecture

MTPLX starts a dedicated process for the selected profile:

```text
hermes -p <profile> serve --isolated --host 127.0.0.1 --port 0
```

The default profile uses the equivalent default-home invocation without an
invalid profile name. The process binds only to loopback and selects an
OS-assigned port.

The sidecar receives process-local overrides for:

- the active MTPLX OpenAI-compatible base URL;
- the local MTPLX API key;
- the model identifier currently served by MTPLX;
- inference provider and API mode;
- reasoning, approval, workspace, and tool settings owned by MTPLX;
- an MTPLX launch identifier and the MTPLX parent PID.

These overrides are applied only to the child process. MTPLX does not rewrite
the selected profile's `config.yaml`, `.env`, gateway routes, channel
credentials, or provider settings. The selected profile still supplies its
identity, sessions, skills, memories, rules, and other user-owned configuration.

The sidecar publishes its actual port and ephemeral authentication material
through Hermes' supported startup contract. MTPLX waits for that information,
connects `HermesGatewayClient`, and treats the agent as ready only after the
`gateway.ready` event.

## Data Flow

1. Opening the Hermes overlay refreshes the Hermes installation state and
   discovers profiles.
2. Selecting a profile builds the process-local MTPLX routing environment and
   starts `hermes serve --isolated`.
3. The WebSocket client connects and waits for `gateway.ready`.
4. MTPLX calls `session.list` and displays the selected profile's saved
   sessions.
5. `session.create` creates an ordinary session in that profile;
   `session.resume` resumes an ordinary saved session.
6. `prompt.submit` starts a turn. Hermes emits transcript, reasoning, tool,
   approval, clarification, completion, and error events.
7. Hermes persists the session in the original profile store. MTPLX stores only
   the last selected profile/session reference in its own settings.
8. Switching profiles or closing the surface disconnects the client and stops
   the MTPLX-owned sidecar without changing the profile files.

## Coexistence and Concurrency

The root Hermes gateway and all non-MTPLX Hermes processes remain independent.
Telegram continues to use its existing multiplex routing and profile
credentials. If Telegram and the embedded GUI both call the same MTPLX model
server, the server's scheduler may queue one generation behind the other, but
both request paths remain valid.

Different sessions in the same profile may be used concurrently. MTPLX must
not submit to the same concrete session while another Hermes process owns or
actively writes it. Before enabling the composer, MTPLX checks Hermes' active
session state and resumes through Hermes' ownership-aware API. An external-active
result or ownership conflict makes the session read-only. If the installed
Hermes build cannot provide a trustworthy ownership result for an existing
session, MTPLX fails closed for that session and offers a new one. It never
guesses that an ambiguous session is safe to write.

MTPLX never stops, restarts, or repairs the root Hermes gateway as part of
embedded profile/session selection.

## Lifecycle and Recovery

Each sidecar has an MTPLX-specific launch identifier and records the MTPLX
parent PID. Normal profile changes, overlay closure, app termination, and an
explicit stop action disconnect the client and terminate the owned sidecar.

On startup, MTPLX may remove only orphaned sidecars that carry a valid MTPLX
ownership marker and whose recorded parent no longer exists. It must not match
or terminate generic `hermes serve`, Hermes Desktop, TUI, gateway, or profile
processes.

Because provider routing is process-local, crash recovery does not restore
profile files. Terminating or losing the sidecar discards the temporary route
automatically. Persisted Hermes session data remains in the original profile.

## Error Handling

- Missing or incompatible Hermes keeps the existing setup/recheck state.
- A sidecar launch failure shows the concrete bounded stderr summary in the
  Hermes pane and does not silently open Terminal.
- Failure to obtain the port or ephemeral authentication material stops the
  child and reports a startup error.
- The WebSocket client uses a bounded connection timeout. A disconnect leaves
  the displayed transcript intact and offers reconnect to the same profile and
  session.
- MTPLX connection failures preserve the Hermes session and allow retry after
  the model server recovers.
- Busy or externally owned sessions are not resumed writable.
- Profile parsing errors affect only that profile and do not hide healthy
  profiles.
- Secrets are excluded from UI text and logs.

## Security Boundaries

- The sidecar binds to `127.0.0.1` only.
- Hermes' ephemeral WebSocket authentication remains enabled.
- API keys and routing overrides exist only in the child environment.
- The app never copies messaging credentials between profiles for this feature.
- The app never changes Telegram routes or gateway service configuration.
- Process cleanup requires an exact MTPLX ownership marker and dead parent PID.
- User-owned profile sections and files remain unchanged.

## Test Strategy

Implementation follows red-green-refactor TDD.

### Unit and command tests

- Discover default and named profiles and classify MTPLX/external/unavailable
  routing states.
- Build the exact default-profile and named-profile isolated serve commands.
- Build process-local routing overrides without writing profile files.
- Persist and restore the selected profile/session reference.
- Restrict orphan cleanup to exact MTPLX-owned processes.

### JSON-RPC integration tests

A fake Hermes backend exercises:

- startup information and `gateway.ready`;
- `session.list`, `session.create`, and `session.resume`;
- transcript restoration and `prompt.submit`;
- message streaming, reasoning, tool, approval, clarification, completion, and
  error events;
- reconnect behavior and profile switching;
- busy/external session handling.

### Regression tests

- The chosen profile's `config.yaml` and `.env` remain byte-identical before,
  during, and after an embedded session.
- Starting and stopping an embedded sidecar does not change root gateway state.
- A session created from MTPLX appears in the normal Hermes session list for
  the original profile.
- Parallel requests from Telegram and a different embedded session remain
  valid; the MTPLX server may serialize generation without changing routing.
- App shutdown and orphan recovery terminate only MTPLX-owned sidecars.

### Build and live acceptance

- Run the focused Swift tests and the complete MTPLXApp test suite.
- Build through `apps/MTPLXApp/script/build_and_run.sh` using the configured
  Xcode toolchain.
- With a real profile, create a session in MTPLX, stream a response, close the
  embedded client, and verify that the same session is visible and resumable in
  Hermes.
- While the embedded client is connected, send a Telegram message routed to a
  different session and verify that it still completes through MTPLX.

## Acceptance Criteria

The feature is complete when:

1. Every readable Hermes profile is selectable in MTPLX.
2. Every saved session for the selected profile is visible with an honest
   activity state.
3. New and resumed sessions work inside the MTPLX Hermes pane with native
   streaming and tool events.
4. Embedded sessions use the currently active MTPLX model regardless of the
   profile's persistent provider.
5. Original profile configuration and gateway routing remain unchanged.
6. Telegram and other Hermes clients continue to operate concurrently.
7. The same session cannot be written concurrently from MTPLX and another
   Hermes process.
8. Sessions created in MTPLX remain visible and resumable in Hermes later.
9. Sidecar shutdown, reconnect, app termination, and orphan recovery are
   bounded and ownership-safe.
10. Automated tests and the macOS app build pass with no new warnings
    attributable to this feature.

## Out of Scope

- Reconfiguring Telegram, Discord, or other messaging platforms.
- Changing Hermes multiplex profile routes.
- Replacing Hermes' session store or agent loop.
- Synchronizing simultaneous writes to the same session across surfaces.
- Editing a profile's permanent provider/model selection from MTPLX.
- Embedding the Hermes Desktop application or its web frontend.
