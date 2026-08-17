# Embedded Hermes Agent Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing MTPLX Hermes pane a native client for any local Hermes profile and its saved sessions while routing only the embedded process through the currently running MTPLX model.

**Architecture:** MTPLX launches one loopback-only `hermes serve --isolated` child for the selected profile, injects the MTPLX endpoint/model only into that child environment, and talks to it through Hermes' authenticated JSON-RPC WebSocket API. Focused runtime, transport, and ownership types keep process lifecycle and fail-closed session permissions out of the SwiftUI view; existing Hermes profile files, the root messaging gateway, and Telegram routing are never modified.

**Tech Stack:** Swift 6, SwiftUI, Combine, Foundation `Process`, `URLSessionWebSocketTask`, XCTest, Hermes 0.19.1 JSON-RPC/WebSocket gateway, macOS 14.

## Global Constraints

- Bind every embedded Hermes sidecar to `127.0.0.1` and request port `0`.
- Start named profiles as `hermes -p <profile> serve --isolated`; start the default profile without `-p`.
- Put MTPLX base URL, API key, model, provider, reasoning, approval, workspace, tool, launch ID, and parent PID overrides only in the child process environment or command arguments.
- Never call `HermesIntegration.sync(configuration:)` from the embedded path.
- Never rewrite a selected profile's `config.yaml`, `.env`, gateway routes, channel credentials, provider settings, skills, memories, or rules.
- Never stop, start, repair, or reconfigure the root Hermes gateway during profile/session selection.
- Preserve ordinary Hermes session persistence so sessions created in MTPLX remain visible in Hermes Desktop and TUI.
- Treat external or ambiguous ownership as read-only and re-check ownership immediately before every `prompt.submit`.
- Kill only a sidecar whose PID, exact launch ID, exact `--isolated` command shape, and dead recorded parent all match an MTPLX ownership record.
- Keep authentication tokens, API keys, and messaging secrets out of UI text and logs.
- Follow red-green-refactor TDD and commit each independently testable task.

---

## File Structure

- Create `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift`: embedded routing classification, launch specification, readiness parsing, ownership registry inspection, ownership records, and process-sidecar lifecycle.
- Create `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesGatewayClient.swift`: JSON-RPC request/response transport, event decoding, authenticated WebSocket connection, and bounded `gateway.ready` wait.
- Modify `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift`: conform the existing integration to the embedded-runtime interface and replace the stale `startDashboard` failure with the isolated sidecar launcher while leaving legacy Desktop/Terminal handoff intact.
- Modify `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift`: profile route state, sidecar switching, session ownership, read-only protection, native session RPCs, pending user requests, streaming, reconnect, and shutdown.
- Modify `apps/MTPLXApp/Sources/MTPLXAppHost/Views/Hermes/HermesOverlay.swift`: remove the terminal-only gate, render profile/session state, make pending requests actionable, and stop the embedded sidecar when the surface closes.
- Create `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift`: command, environment, classification, startup parsing, byte-preservation, and ownership-safe cleanup tests.
- Create `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesGatewayClientTests.swift`: real local fake-WebSocket tests for JSON-RPC, authentication URL, `gateway.ready`, disconnect, timeout, and event decoding.
- Create `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift`: fake runtime/transport tests for profile switching, list/create/resume/send, permissions, streaming, prompts, persistence references, reconnect, and shutdown.
- Create `apps/MTPLXApp/Tests/MTPLXAppCoreTests/Support/FakeHermesGateway.swift`: loopback RFC 6455 fixture shared only by gateway-client tests.

---

### Task 1: Embedded Profile Routing and Launch Specification

**Files:**
- Create: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift`
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift:7-18,209-250,252-397`
- Test: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift`

**Interfaces:**
- Consumes: `HermesProfile`, `MTPLXAppConfiguration`, `OpenCodeIntegration.modelID(for:)`, `OpenCodeIntegration.baseURLString(host:port:)`, and `HermesIntegration.launchEnvironment(configuration:)`.
- Produces: `HermesProfileRoutingState`, `HermesServeLaunchSpec`, `HermesEmbeddedRuntime`, `HermesIntegration.routingState(for:configuration:)`, and `HermesIntegration.serveLaunchSpec(profile:configuration:token:launchID:parentPID:)`.

- [ ] **Step 1: Write failing routing and command tests**

```swift
func testNamedProfileLaunchUsesIsolatedServeAndProcessLocalMTPLXRoute() throws {
    let profile = HermesProfile(name: "bernd", path: tempProfile.path, isDefault: false)
    let spec = try integration.serveLaunchSpec(
        profile: profile,
        configuration: configuration,
        token: "test-session-token",
        launchID: "0123456789abcdef",
        parentPID: 4242
    )

    XCTAssertEqual(spec.arguments, [
        "-p", "bernd", "serve", "--isolated", "--host", "127.0.0.1",
        "--port", "0", "--ssh-owner-nonce", "0123456789abcdef",
    ])
    XCTAssertEqual(spec.environment["HERMES_INFERENCE_PROVIDER"], "custom")
    XCTAssertEqual(spec.environment["CUSTOM_BASE_URL"], "http://127.0.0.1:18080/v1")
    XCTAssertEqual(spec.environment["HERMES_INFERENCE_MODEL"], "current-model")
    XCTAssertEqual(spec.environment["HERMES_DASHBOARD_SESSION_TOKEN"], "test-session-token")
    XCTAssertEqual(spec.environment["MTPLX_HERMES_PARENT_PID"], "4242")
    XCTAssertNil(spec.environment["HERMES_HOME"])
}

func testDefaultProfileLaunchOmitsProfileFlag() throws {
    let spec = try integration.serveLaunchSpec(
        profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
        configuration: configuration,
        token: "test-session-token",
        launchID: "fedcba9876543210",
        parentPID: 4242
    )
    XCTAssertEqual(Array(spec.arguments.prefix(5)), ["serve", "--isolated", "--host", "127.0.0.1", "--port"])
    XCTAssertFalse(spec.arguments.contains("-p"))
}

func testProfileRoutingClassifiesMTPLXExternalAndUnavailableIndependently() throws {
    XCTAssertEqual(integration.routingState(for: mtplxProfile, configuration: configuration), .mtplx)
    XCTAssertEqual(integration.routingState(for: externalProfile, configuration: configuration), .external)
    guard case .unavailable = integration.routingState(for: unreadableProfile, configuration: configuration) else {
        return XCTFail("Unreadable profile must remain visible as unavailable")
    }
}
```

- [ ] **Step 2: Run the new tests and confirm the missing interfaces fail**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests`

Expected: compile failures naming `HermesProfileRoutingState`, `HermesServeLaunchSpec`, `routingState`, and `serveLaunchSpec`.

- [ ] **Step 3: Implement the routing and launch types**

```swift
public enum HermesProfileRoutingState: Equatable, Sendable {
    case mtplx
    case external
    case unavailable(String)
}

public struct HermesServeLaunchSpec: Equatable, Sendable {
    public let executableURL: URL
    public let arguments: [String]
    public let environment: [String: String]
    public let token: String
    public let launchID: String
    public let parentPID: Int32
}

public protocol HermesEmbeddedRuntime: Sendable {
    func routingState(
        for profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) -> HermesProfileRoutingState
    func startEmbeddedSidecar(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> any HermesSidecarControlling
    func sessionOwnership(
        profile: HermesProfile,
        sessionID: String,
        ownedSidecarPID: Int32?
    ) -> HermesSessionOwnership
    @discardableResult func reapOrphanedEmbeddedSidecars() -> [Int32]
}
```

In `serveLaunchSpec`, validate the launch ID with `^[0-9a-f]{16}$`, start from `launchEnvironment(configuration:)`, remove inherited `HERMES_HOME`, set `CUSTOM_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `HERMES_MODEL`, `HERMES_INFERENCE_MODEL`, `HERMES_INFERENCE_PROVIDER=custom`, `HERMES_DASHBOARD_SESSION_TOKEN`, `HERMES_SESSION_PLATFORM=mtplx-app`, `MTPLX_HERMES_LAUNCH_ID`, and `MTPLX_HERMES_PARENT_PID`, then build the exact default/named argument arrays asserted above.

Classify a profile as `.mtplx` only when its readable effective `model.provider`, `model.base_url`/`CUSTOM_BASE_URL`, and model reference equal the active configuration; classify other readable profiles as `.external`; preserve discovery but return `.unavailable(redactedReason)` for an unreadable or structurally invalid profile.

- [ ] **Step 4: Re-run focused and existing Hermes environment tests**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests && swift test --filter MTPLXAppCoreTests.testHermesIntegrationSyncsMTPLXProfileAndLaunchEnvironment`

Expected: both commands pass, proving the new child-only path coexists with the legacy explicit `mtplx` profile sync.

- [ ] **Step 5: Commit the routing slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift \
  apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift
git commit -m "feat(hermes): define embedded profile routing"
```

### Task 2: Isolated Sidecar Startup, Authentication, and Ownership-Safe Cleanup

**Files:**
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift`
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift:146-195,735-744,1487-1539,1661-1735`
- Test: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift`

**Interfaces:**
- Consumes: `HermesServeLaunchSpec` from Task 1 and `SubprocessTailBuffer` from `Services/SubprocessSupport.swift`.
- Produces: `HermesSidecarControlling`, `HermesBackendReadyParser`, `HermesSidecarOwnershipRecord`, `HermesIntegration.startEmbeddedSidecar(profile:configuration:)`, and `HermesIntegration.reapOrphanedEmbeddedSidecars()`.

- [ ] **Step 1: Write failing startup and cleanup tests**

```swift
func testReadyParserAcceptsHeadlessSentinelOnly() {
    XCTAssertEqual(HermesBackendReadyParser.port(from: "HERMES_BACKEND_READY port=45123"), 45123)
    XCTAssertNil(HermesBackendReadyParser.port(from: "Hermes backend listening on 0.0.0.0:45123"))
    XCTAssertNil(HermesBackendReadyParser.port(from: "HERMES_BACKEND_READY port=0"))
}

func testSidecarUsesCallerTokenAndRemovesOwnershipRecordOnStop() async throws {
    let sidecar = try await integration.startEmbeddedSidecar(
        profile: defaultProfile,
        configuration: configuration
    )
    XCTAssertEqual(sidecar.webSocketURL.host, "127.0.0.1")
    XCTAssertEqual(sidecar.webSocketURL.path, "/api/ws")
    XCTAssertNotNil(URLComponents(url: sidecar.webSocketURL, resolvingAgainstBaseURL: false)?
        .queryItems?.first(where: { $0.name == "token" })?.value)
    XCTAssertTrue(fileManager.fileExists(atPath: sidecar.ownershipRecordURL.path))
    sidecar.stop()
    XCTAssertFalse(fileManager.fileExists(atPath: sidecar.ownershipRecordURL.path))
}

func testOrphanCleanupRequiresExactMarkerCommandAndDeadParent() {
    let killed = HermesOrphanSidecarScanner.orphanPIDs(
        records: records,
        processes: processSnapshot,
        livePIDs: [9001]
    )
    XCTAssertEqual(killed, [7101])
    XCTAssertFalse(killed.contains(7102)) // generic hermes serve
    XCTAssertFalse(killed.contains(7103)) // marker mismatch
    XCTAssertFalse(killed.contains(7104)) // parent still alive
}
```

Use a temporary executable fixture that writes `HERMES_BACKEND_READY port=<fixture port>` to stdout, records its received environment with secret values replaced by `present`, and stays alive until SIGTERM. The test must compare profile `config.yaml` and `.env` bytes before and after start/stop.

- [ ] **Step 2: Run the lifecycle tests and verify red**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests`

Expected: compile failures for the parser, sidecar protocol, ownership record, and orphan scanner.

- [ ] **Step 3: Implement bounded sidecar launch and exact ownership**

```swift
public protocol HermesSidecarControlling: AnyObject, Sendable {
    var processIdentifier: Int32 { get }
    var isRunning: Bool { get }
    var webSocketURL: URL { get }
    var ownershipRecordURL: URL { get }
    func stop()
}

public struct HermesSidecarOwnershipRecord: Codable, Equatable, Sendable {
    public let launchID: String
    public let pid: Int32
    public let parentPID: Int32
    public let profileName: String
    public let createdAt: Date
}

enum HermesBackendReadyParser {
    static func port(from line: String) -> Int? {
        guard line.hasPrefix("HERMES_BACKEND_READY port="),
              let port = Int(line.dropFirst("HERMES_BACKEND_READY port=".count)),
              (1...65535).contains(port) else { return nil }
        return port
    }
}
```

Launch `Process` with pipes attached before `run()`, retain a 4 KiB redacted stderr tail, wait at most 15 seconds for the sentinel, and terminate the child on exit-before-ready, timeout, malformed port, or ownership-record write failure. Generate a 32-byte URL-safe session token in MTPLX, pass it through `HERMES_DASHBOARD_SESSION_TOKEN`, and construct `ws://127.0.0.1:<port>/api/ws?token=<token>` without logging it.

Store records under the injected `sidecarRuntimeDirectory` (default `~/.mtplx/hermes-sidecars`). On cleanup, require the exact PID plus the exact `serve --isolated --ssh-owner-nonce <launchID>` argv sequence and a dead `parentPID`; send TERM, wait two seconds, then KILL only that verified PID. Delete stale records whose process is already dead. Do not call `hermes serve --stop`.

- [ ] **Step 4: Run lifecycle and legacy cleanup regressions**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests && swift test --filter MTPLXAppCoreTests.testHermesTerminalCleanupOnlyMatchesAppLaunchedChat`

Expected: all tests pass; generic Hermes Desktop/TUI/serve commands remain unmatched.

- [ ] **Step 5: Commit the sidecar slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift \
  apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift
git commit -m "feat(hermes): launch owned isolated sidecars"
```

### Task 3: Authenticated JSON-RPC WebSocket Client with Readiness Gate

**Files:**
- Create: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesGatewayClient.swift`
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift:90-231`
- Create: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/Support/FakeHermesGateway.swift`
- Create: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesGatewayClientTests.swift`

**Interfaces:**
- Consumes: authenticated `HermesSidecarControlling.webSocketURL` from Task 2 and the existing `JSONValue` type.
- Produces: `HermesGatewayEvent`, `HermesGatewayClientProtocol`, `URLSessionHermesGatewayClient`, and `HermesGatewayClientFactory`.

- [ ] **Step 1: Write failing local-WebSocket tests**

```swift
@MainActor
func testConnectWaitsForGatewayReadyBeforeReturning() async throws {
    let backend = try FakeHermesGateway(eventsOnConnect: [])
    let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: "fixture-token"))
    let connect = Task { try await client.connectAndWaitUntilReady(timeoutSeconds: 1) }
    try await Task.sleep(for: .milliseconds(50))
    XCTAssertFalse(connect.isCancelled)
    backend.sendEvent(type: "gateway.ready", sessionID: nil, payload: [:])
    try await connect.value
}

@MainActor
func testRPCResponseAndEventAreDecoded() async throws {
    let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
    backend.respond(to: "session.list", result: .object(["sessions": .array([])]))
    let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: "fixture-token"))
    var received: [HermesGatewayEvent] = []
    client.onEvent = { received.append($0) }
    try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
    let value = try await client.call(method: "session.list", params: ["limit": .number(200)])
    XCTAssertEqual(value.objectValue?["sessions"]?.arrayValue, [])
    backend.sendEvent(type: "message.delta", sessionID: "live-1", payload: ["text": .string("Hi")])
    await eventually { received.contains(where: { $0.type == "message.delta" }) }
}
```

Add timeout, RPC-error, disconnect-with-pending-request, malformed-frame, and token-query tests. `FakeHermesGateway` binds only to `127.0.0.1`, performs the RFC 6455 handshake using `Insecure.SHA1`, decodes masked client text frames, and emits unmasked server text frames.

- [ ] **Step 2: Run the gateway tests and verify red**

Run: `cd apps/MTPLXApp && swift test --filter HermesGatewayClientTests`

Expected: compile failure because the extracted protocol/client and fixture do not exist.

- [ ] **Step 3: Extract and implement the transport**

```swift
struct HermesGatewayEvent: Equatable, Sendable {
    let type: String
    let sessionID: String?
    let payload: [String: JSONValue]
}

@MainActor
protocol HermesGatewayClientProtocol: AnyObject {
    var onEvent: ((HermesGatewayEvent) -> Void)? { get set }
    var onDisconnect: ((String) -> Void)? { get set }
    func connectAndWaitUntilReady(timeoutSeconds: Double) async throws
    func call(method: String, params: [String: JSONValue]) async throws -> JSONValue
    func close()
}

typealias HermesGatewayClientFactory = @MainActor (URL) -> any HermesGatewayClientProtocol
```

Move the current JSON-RPC encoding, response continuation table, and event parsing into `URLSessionHermesGatewayClient`. Install the readiness continuation before resuming the socket, resolve it only on `gateway.ready`, fail it on disconnect, and race it against a 10-second timeout. Redact query strings from every surfaced connection error.

- [ ] **Step 4: Run the client tests**

Run: `cd apps/MTPLXApp && swift test --filter HermesGatewayClientTests`

Expected: all gateway tests pass without external network access.

- [ ] **Step 5: Commit the transport slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesGatewayClient.swift \
  apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/Support/FakeHermesGateway.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesGatewayClientTests.swift
git commit -m "feat(hermes): add authenticated gateway client"
```

### Task 4: Store Lifecycle, Profile Switching, and Native Session RPCs

**Files:**
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift:233-563,708-760`
- Create: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift`

**Interfaces:**
- Consumes: `HermesEmbeddedRuntime`, `HermesSidecarControlling`, and `HermesGatewayClientFactory` from Tasks 1-3.
- Produces: the injectable store initializer, honest readiness state, profile route map, and working `session.list`, `session.create`, `session.resume`, `session.interrupt`, and `prompt.submit` orchestration.

- [ ] **Step 1: Write failing lifecycle and RPC orchestration tests**

```swift
@MainActor
func testLoadSessionsWaitsForReadyAndReusesMatchingSidecar() async {
    let runtime = FakeHermesEmbeddedRuntime()
    let client = FakeHermesGatewayClient()
    let store = HermesAgentStore(
        integration: integration,
        embeddedRuntime: runtime,
        clientFactory: { _ in client }
    )
    let load = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
    XCTAssertEqual(store.connectionState, .starting)
    XCTAssertEqual(client.calls, [])
    client.finishReady()
    await load.value
    XCTAssertEqual(client.calls.first?.method, "session.list")
    XCTAssertEqual(store.connectionState, .connected)
    await store.loadSessions(profile: bernd, configuration: configuration)
    XCTAssertEqual(runtime.startCount, 1)
}

@MainActor
func testProfileSwitchClosesClientAndStopsOnlyOwnedSidecar() async {
    let firstLoad = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
    firstClient.finishReady()
    await firstLoad.value
    let secondLoad = Task { await store.loadSessions(profile: researcher, configuration: configuration) }
    secondClient.finishReady()
    await secondLoad.value
    XCTAssertTrue(firstClient.didClose)
    XCTAssertEqual(firstSidecar.stopCount, 1)
    XCTAssertEqual(secondSidecar.stopCount, 0)
}
```

Add tests that create a new native session, resume a saved session with transcript messages, submit a prompt, interrupt a turn, restore `lastHermesProfile/sessionID/title`, reconnect after disconnect without clearing the visible transcript, and call orphan cleanup once during `prepare`.

- [ ] **Step 2: Run the store tests and verify red**

Run: `cd apps/MTPLXApp && swift test --filter HermesAgentStoreTests`

Expected: initializer and readiness-related compile failures.

- [ ] **Step 3: Inject runtime/client dependencies and gate connected state**

```swift
@MainActor
init(
    integration: HermesIntegration,
    embeddedRuntime: any HermesEmbeddedRuntime,
    clientFactory: @escaping HermesGatewayClientFactory
) {
    self.integration = integration
    self.embeddedRuntime = embeddedRuntime
    self.clientFactory = clientFactory
}
```

Keep `public convenience init(integration:)` as the live path using the same `HermesIntegration` for `embeddedRuntime` and `URLSessionHermesGatewayClient.init(url:)` for the factory. Store the sidecar as `any HermesSidecarControlling`. In `ensureGateway`, tear down the previous generation, start the new isolated sidecar, call `connectAndWaitUntilReady(timeoutSeconds: 10)`, then set `gatewayReady=true` and `.connected`; never report connected before the event.

In `prepare`, call `reapOrphanedEmbeddedSidecars()`, discover all profiles, populate `[profile.id: routingState]`, and restore the remembered profile. Keep per-profile failures local. Preserve transcript/session references across a transport disconnect and expose `reconnect(configuration:)` that reopens the selected profile and resumes the selected saved session.

- [ ] **Step 4: Run store, transport, and persistence tests**

Run: `cd apps/MTPLXApp && swift test --filter HermesAgentStoreTests && swift test --filter HermesGatewayClientTests && swift test --filter MTPLXAppCoreTests.testAppConfigurationPersistsHermesResumeState`

Expected: all pass.

- [ ] **Step 5: Commit the store lifecycle slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift
git commit -m "feat(hermes): connect profiles and native sessions"
```

### Task 5: Fail-Closed Cross-Process Session Ownership

**Files:**
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift`
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift:65-88,334-462,550-563`
- Test: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift`
- Test: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift`

**Interfaces:**
- Consumes: Hermes' per-profile `runtime/active_sessions.json`, the selected profile path, sidecar PID, and session IDs returned by `session.list`/`session.resume`.
- Produces: `HermesSessionOwnership`, `HermesSessionActivityState`, `HermesSavedSession.activity`, `HermesAgentStore.activeSessionWritable`, and `HermesAgentStore.readOnlyReason`.

- [ ] **Step 1: Write failing ownership and send-gate tests**

```swift
func testOwnershipRegistryDistinguishesCurrentSidecarFromExternalProcess() throws {
    try writeRegistry(entries: [
        ["session_id": "ours", "surface": "mtplx-app", "pid": 7001],
        ["session_id": "telegram", "surface": "telegram", "pid": 7002],
    ])
    XCTAssertEqual(runtime.sessionOwnership(profile: bernd, sessionID: "ours", ownedSidecarPID: 7001), .ownedByMTPLX)
    XCTAssertEqual(runtime.sessionOwnership(profile: bernd, sessionID: "telegram", ownedSidecarPID: 7001), .external(surface: "telegram"))
    XCTAssertEqual(runtime.sessionOwnership(profile: bernd, sessionID: "idle", ownedSidecarPID: 7001), .ready)
}

@MainActor
func testExternalAndUnknownSessionsRemainReadableButCannotSubmit() async {
    _ = try? await store.resume(externalSession, profile: bernd, configuration: configuration)
    XCTAssertFalse(store.activeSessionWritable)
    await store.send("must not leave MTPLX")
    XCTAssertFalse(client.calls.contains(where: { $0.method == "prompt.submit" }))
    XCTAssertNotNil(store.readOnlyReason)
}

@MainActor
func testSendRechecksOwnershipImmediatelyBeforeSubmit() async {
    await resumeReadySession()
    runtime.nextOwnership = .external(surface: "telegram")
    await store.send("race check")
    XCTAssertFalse(client.calls.contains(where: { $0.method == "prompt.submit" }))
    XCTAssertEqual(store.activeSessionActivity, .externallyActive(surface: "telegram"))
}
```

Add corrupt registry, unreadable registry, dead-PID pruning, fresh-session writable, and different-session concurrency tests. Treat no registry file as `.ready`; treat malformed/unreadable content as `.unknown(redactedReason)`; ignore dead entries after `kill(pid, 0)`/start-time validation.

- [ ] **Step 2: Run ownership tests and verify red**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests && swift test --filter HermesAgentStoreTests`

Expected: failures for missing ownership/activity state and send guard.

- [ ] **Step 3: Implement activity mapping and fail-closed checks**

```swift
public enum HermesSessionOwnership: Equatable, Sendable {
    case ready
    case ownedByMTPLX
    case external(surface: String)
    case unknown(String)
}

public enum HermesSessionActivityState: Equatable, Sendable {
    case ready
    case runningInMTPLX
    case externallyActive(surface: String)
    case ownershipUnknown(String)
}
```

Add `public let activity: HermesSessionActivityState` to `HermesSavedSession` and give its initializer the source-compatible parameter `activity: HermesSessionActivityState = .ready`.

Parse only `entries` containing a nonempty `session_id`, positive PID, and live process identity. A matching session entry owned by the current sidecar PID is `.ownedByMTPLX`; a live different PID is `.external`; multiple conflicting entries are `.unknown`. Refresh all list-row states after `session.list`, before `session.resume`, and immediately before `prompt.submit`. A resumed external/unknown session may load its native transcript but must keep `activeSessionWritable=false`; `send` must return without appending a user message or issuing RPC. New sessions are writable unless a later pre-submit check finds a conflict.

- [ ] **Step 4: Run ownership and session regression tests**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests && swift test --filter HermesAgentStoreTests`

Expected: all pass, including simultaneous use of different session IDs.

- [ ] **Step 5: Commit the ownership slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesEmbeddedRuntime.swift \
  apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesEmbeddedRuntimeTests.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift
git commit -m "feat(hermes): guard cross-process session ownership"
```

### Task 6: Streaming Events and Actionable Hermes Requests

**Files:**
- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift:19-63,565-706`
- Test: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift`

**Interfaces:**
- Consumes: `HermesGatewayEvent` and the active runtime session ID.
- Produces: `HermesPendingRequest`, `HermesPendingRequestKind`, `HermesAgentStore.pendingRequest`, `respondToPendingRequest(value:)`, and `denyPendingApproval()`.

- [ ] **Step 1: Write failing event and request-response tests**

```swift
@MainActor
func testStreamingReasoningToolsAndCompletionUpdateTranscript() async {
    client.emit(.init(type: "message.start", sessionID: "live-1", payload: [:]))
    client.emit(.init(type: "message.delta", sessionID: "live-1", payload: ["text": .string("Hel")]))
    client.emit(.init(type: "tool.start", sessionID: "live-1", payload: ["name": .string("terminal")]))
    client.emit(.init(type: "message.complete", sessionID: "live-1", payload: [
        "text": .string("Hello"), "reasoning": .string("checked state"),
    ]))
    XCTAssertEqual(store.messages.last?.text, "Hello")
    XCTAssertTrue(store.toolTraces.contains(where: { $0.name == "Thought" }))
    XCTAssertFalse(store.isStreaming)
}

@MainActor
func testAskFirstSurfacesApprovalAndRespondsWithSelectedChoice() async {
    client.emit(.init(type: "approval.request", sessionID: "live-1", payload: [
        "command": .string("git status"),
        "choices": .array([.string("once"), .string("deny")]),
    ]))
    XCTAssertEqual(store.pendingRequest?.kind, .approval)
    await store.respondToPendingRequest(value: "once")
    XCTAssertEqual(client.calls.last?.method, "approval.respond")
    XCTAssertEqual(client.calls.last?.params["choice"], .string("once"))
}
```

Add tests for configured auto-approve, clarify answer, sudo password, secret value, request expiry matched by `request_id`, error events, disconnect during streaming, and events from non-active sessions being ignored.

- [ ] **Step 2: Run event tests and verify red**

Run: `cd apps/MTPLXApp && swift test --filter HermesAgentStoreTests`

Expected: pending-request model and response method failures.

- [ ] **Step 3: Implement request state and response RPC mapping**

```swift
public enum HermesPendingRequestKind: Equatable, Sendable {
    case approval
    case clarification
    case sudo
    case secret
}

public struct HermesPendingRequest: Identifiable, Equatable, Sendable {
    public let id: String
    public let kind: HermesPendingRequestKind
    public let prompt: String
    public let choices: [String]
}
```

Map request events and response fields exactly:

| Event | RPC | Value field |
|---|---|---|
| `approval.request` | `approval.respond` | `choice` plus `session_id`; deny uses `deny` |
| `clarify.request` | `clarify.respond` | `answer` plus `request_id` |
| `sudo.request` | `sudo.respond` | `password` plus `request_id` |
| `secret.request` | `secret.respond` | `value` plus `request_id` |

When `configuration.hermesAutoApprove` is true, answer approvals with `choice=once` and never set `all=true`; when false, keep the composer paused and show the request. Clear only the pending request whose ID matches the corresponding `*.expire` event. Never log sudo/secret values.

- [ ] **Step 4: Run the complete store test file**

Run: `cd apps/MTPLXApp && swift test --filter HermesAgentStoreTests`

Expected: all streaming, request, ownership, and lifecycle tests pass.

- [ ] **Step 5: Commit the event slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift
git commit -m "feat(hermes): handle native agent events and prompts"
```

### Task 7: Enable the Embedded Hermes GUI

**Files:**
- Modify: `apps/MTPLXApp/Sources/MTPLXAppHost/Views/Hermes/HermesOverlay.swift:5-975`

**Interfaces:**
- Consumes: `profileRoutingStates`, `HermesSavedSession.activity`, `activeSessionWritable`, `readOnlyReason`, `pendingRequest`, reconnect, create/resume/send/interrupt, and stop APIs from Tasks 4-6.
- Produces: selectable profile/session UI, routing/activity badges, embedded transcript/composer, read-only new-session offer, request cards, reconnect action, and close cleanup.

- [ ] **Step 1: Build once to capture the existing disabled-surface baseline**

Run: `cd apps/MTPLXApp && swift build --product MTPLXApp`

Expected: pass before the view edit; the source still contains three `nativeDashboardSupported` gates.

- [ ] **Step 2: Replace the terminal gates with the embedded state**

Remove the sidebar branch at line 140, composer branch at line 604, and `prepare()` early return at line 770. Render:

```swift
Text(routeLabel(for: hermes.profileRoutingStates[profile.id] ?? .external))
    .font(.system(size: 9, weight: .heavy, design: .monospaced))
    .foregroundStyle(routeColor(for: profile))

Text(activityLabel(session.activity))
    .font(.system(size: 9, weight: .bold, design: .monospaced))

TextEditor(text: $composerText)
    .disabled(!hermes.activeSessionWritable || !hermes.gatewayReady || hermes.pendingRequest != nil)
```

Show `MTPLX`, `External`, or `Unavailable` on profile rows. Show `Ready`, `Running in MTPLX`, `Externally active`, or `Ownership unknown` on session rows. Disable unavailable profile selection. Keep external profiles selectable because their child-only override is the feature.

- [ ] **Step 3: Add read-only, reconnect, and pending-request actions**

For external/unknown ownership, retain the transcript and replace the composer with the concrete reason plus a `New Agent` button calling `startNew()`. For `.failed`, retain transcript and render a `Reconnect` button calling `hermes.reconnect(configuration:)`. Render approval choices as buttons; render clarification as a text field; render sudo/secret values with `SecureField`; submit through `respondToPendingRequest(value:)`.

- [ ] **Step 4: Stop only the embedded sidecar on close and start only the model daemon**

Change `ensureDaemonReady()` to:

```swift
private func ensureDaemonReady() async -> Bool {
    guard backend.daemonState.kind != .running else { return true }
    await backend.startDaemon(target: nil)
    guard backend.daemonState.kind == .running else {
        localError = "MTPLX is not ready yet."
        return false
    }
    return true
}
```

Inject `HermesAgentStore` into `HermesOverlay` and wrap collapse:

```swift
private func collapseAndStop() {
    Task {
        await hermes.stop()
        onCollapse()
    }
}
```

Use `ChatCloseButton(action: collapseAndStop)` and `.onDisappear { Task { await hermes.stop() } }`. This preserves the root gateway and MTPLX daemon while ending only the owned embedded sidecar.

- [ ] **Step 5: Build and scan out stale gates**

Run: `cd apps/MTPLXApp && swift build --product MTPLXApp`

Run: `rg -n "nativeDashboardSupported|startDaemon\(target: \.hermes\)" apps/MTPLXApp/Sources/MTPLXAppHost/Views/Hermes/HermesOverlay.swift`

Expected: build passes; the scan returns no matches.

- [ ] **Step 6: Commit the GUI slice**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppHost/Views/Hermes/HermesOverlay.swift
git commit -m "feat(app): embed Hermes profile sessions"
```

### Task 8: Regression, Bundle Build, and Live Coexistence Acceptance

**Files:**
- Modify only if a failing regression identifies a feature-owned defect in files from Tasks 1-7.

**Interfaces:**
- Consumes: complete feature from Tasks 1-7.
- Produces: verified unit/integration suite, release app bundle, profile byte-preservation evidence, native Hermes session evidence, and Telegram coexistence evidence.

- [ ] **Step 1: Run focused Hermes tests**

Run: `cd apps/MTPLXApp && swift test --filter HermesEmbeddedRuntimeTests && swift test --filter HermesGatewayClientTests && swift test --filter HermesAgentStoreTests && swift test --filter MTPLXAppCoreTests.testHermes`

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete Swift suite**

Run: `cd apps/MTPLXApp && swift test`

Expected: zero failures and no new warnings attributable to the Hermes feature.

- [ ] **Step 3: Build the distributable app with the configured Xcode toolchain**

Run:

```bash
DEVELOPER_DIR=/Volumes/nugly/Applications/Xcode.app \
  apps/MTPLXApp/script/build_and_run.sh --no-launch
```

Expected: exit 0 and `apps/MTPLXApp/dist/MTPLXApp.app/Contents/MacOS/MTPLXApp` exists.

- [ ] **Step 4: Record non-secret live baselines**

For the selected external-provider profile, record SHA-256 checksums of `config.yaml` and `.env` without printing their contents. Record `env -u HERMES_HOME hermes gateway status`, the existing root gateway PID, and the profile/session IDs used for MTPLX and Telegram. Use different concrete session IDs.

- [ ] **Step 5: Exercise a real embedded native session**

Launch the built app deliberately, start the current MTPLX model, open Hermes, select the external-provider profile, create a new agent, send a harmless prompt, observe streaming completion/tool state, close the pane, reopen it, and resume the same saved session. Verify the profile checksums remain identical throughout.

- [ ] **Step 6: Verify Hermes-native persistence**

Use Hermes Desktop or the profile-scoped Hermes TUI/session list to confirm the MTPLX-created session ID and transcript are visible and resumable after the MTPLX sidecar stops. Do not inspect or copy secrets.

- [ ] **Step 7: Verify Telegram coexistence on a different session**

While the embedded pane is connected, send one Telegram message to an agent routed through the existing root multiplex gateway. Confirm the root gateway PID remains unchanged, the Telegram turn completes through MTPLX on its different session ID, and the embedded turn still completes. A serialized model generation is acceptable; route/config changes are not.

- [ ] **Step 8: Verify ownership and cleanup behavior live**

Open one saved session in another Hermes surface, confirm MTPLX marks it read-only and offers a new agent, then close the external owner and confirm the next refresh marks it ready. Quit MTPLX during an embedded sidecar run, relaunch, and confirm only the exact orphan ownership record is reaped while generic Hermes Desktop/TUI/root-gateway processes remain alive.

- [ ] **Step 9: Re-run checks after any acceptance fix and commit**

Run: `cd apps/MTPLXApp && swift test`

Run:

```bash
DEVELOPER_DIR=/Volumes/nugly/Applications/Xcode.app \
  apps/MTPLXApp/script/build_and_run.sh --no-launch
```

Expected: both pass after the last change.

```bash
git add apps/MTPLXApp/Sources apps/MTPLXApp/Tests
git commit -m "test(hermes): verify embedded coexistence"
```

Skip the final commit when Tasks 1-7 already contain every needed change and the working tree has no feature-owned edits.
