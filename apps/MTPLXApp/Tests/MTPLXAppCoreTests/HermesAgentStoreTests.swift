import Foundation
import XCTest
@testable import MTPLXAppCore

final class HermesAgentStoreTests: XCTestCase {
    private var root: URL!
    private var integration: HermesIntegration!
    private var configuration: MTPLXAppConfiguration!
    private var bernd: HermesProfile!
    private var researcher: HermesProfile!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("HermesAgentStoreTests-\(UUID().uuidString)", isDirectory: true)
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        let profiles = hermesHome.appendingPathComponent("profiles", isDirectory: true)
        try FileManager.default.createDirectory(at: profiles, withIntermediateDirectories: true)
        for name in ["bernd", "researcher"] {
            try FileManager.default.createDirectory(
                at: profiles.appendingPathComponent(name, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        integration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: "/usr/bin/true",
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            sidecarRuntimeDirectory: root.appendingPathComponent("sidecars", isDirectory: true)
        )
        configuration = MTPLXAppConfiguration(
            model: "current-model",
            host: "127.0.0.1",
            port: 18080,
            apiKey: "test-key"
        )
        bernd = HermesProfile(name: "bernd", path: profiles.appendingPathComponent("bernd").path, isDefault: false)
        researcher = HermesProfile(name: "researcher", path: profiles.appendingPathComponent("researcher").path, isDefault: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    @MainActor
    func testLoadSessionsWaitsForReadyAndReusesMatchingSidecar() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient()
        client.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])

        let load = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
        await waitUntil { store.connectionState == .starting }
        XCTAssertEqual(client.calls, [])
        client.finishReady()
        await load.value

        XCTAssertEqual(client.calls.first?.method, "session.list")
        XCTAssertTrue(store.gatewayReady)
        XCTAssertEqual(store.connectionState, .connected)
        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertEqual(runtime.startCount, 1)
    }

    @MainActor
    func testProfileSwitchClosesClientAndStopsOnlyCurrentOwnedSidecar() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])

        await store.loadSessions(profile: bernd, configuration: configuration)
        await store.loadSessions(profile: researcher, configuration: configuration)

        XCTAssertTrue(firstClient.didClose)
        XCTAssertEqual(runtime.sidecars[0].stopCount, 1)
        XCTAssertEqual(runtime.sidecars[1].stopCount, 0)
        XCTAssertEqual(store.selectedProfile?.name, "researcher")
    }

    @MainActor
    func testNativeSessionRPCsRestoreTranscriptAndRememberedSession() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("new-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.resultByMethod["session.resume"] = .object([
            "session_id": .string("live-7"),
            "resumed": .string("saved-7"),
            "messages": .array([
                .object(["role": .string("user"), "text": .string("Earlier question")]),
                .object(["role": .string("assistant"), "content": .string("Earlier answer")]),
            ]),
        ])
        let store = makeStore(runtime: runtime, clients: [client])

        let created = try await store.startNewAgent(profile: bernd, configuration: configuration)
        XCTAssertEqual(created.sessionID, "new-1")
        let resumed = try await store.resume(
            HermesSavedSession(id: "saved-7", title: "Saved", preview: "", startedAt: 0, messageCount: 2, source: ""),
            profile: bernd,
            configuration: configuration
        )

        XCTAssertEqual(resumed, HermesSessionReference(profileName: "bernd", sessionID: "saved-7", title: "Saved"))
        XCTAssertEqual(store.messages.map(\.text), ["Earlier question", "Earlier answer"])
        XCTAssertEqual(store.activeReference, resumed)

        let remembered = MTPLXAppConfiguration(
            model: configuration.model,
            host: configuration.host,
            port: configuration.port,
            apiKey: configuration.apiKey,
            lastHermesProfile: "bernd",
            lastHermesSessionID: "saved-7",
            lastHermesSessionTitle: "Saved"
        )
        _ = try await store.resumeLast(configuration: remembered)
        XCTAssertEqual(store.activeReference?.sessionID, "saved-7")
    }

    @MainActor
    func testSendAndInterruptUseNativeSessionRPCs() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        await store.send("Run diagnostics")
        await store.interrupt()

        XCTAssertEqual(
            client.calls.filter { $0.method == "prompt.submit" }.first?.params,
            ["session_id": .string("live-1"), "text": .string("Run diagnostics")]
        )
        XCTAssertEqual(
            client.calls.filter { $0.method == "session.interrupt" }.first?.params,
            ["session_id": .string("live-1")]
        )
    }

    @MainActor
    func testReconnectPreservesVisibleTranscriptAndResumesSelectedSession() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        let resume = JSONValue.object([
            "session_id": .string("live-9"),
            "resumed": .string("saved-9"),
            "messages": .array([
                .object(["role": .string("assistant"), "text": .string("Persisted answer")]),
            ]),
        ])
        firstClient.resultByMethod["session.resume"] = resume
        secondClient.resultByMethod["session.resume"] = resume
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        let saved = HermesSavedSession(id: "saved-9", title: "Saved", preview: "", startedAt: 0, messageCount: 1, source: "")
        _ = try await store.resume(saved, profile: bernd, configuration: configuration)
        firstClient.disconnect("transport lost")
        XCTAssertEqual(store.messages.map(\.text), ["Persisted answer"])

        try await store.reconnect(configuration: configuration)

        XCTAssertEqual(store.connectionState, .connected)
        XCTAssertEqual(store.activeReference?.sessionID, "saved-9")
        XCTAssertEqual(store.messages.map(\.text), ["Persisted answer"])
        XCTAssertEqual(secondClient.calls.filter { $0.method == "session.resume" }.count, 1)
    }

    @MainActor
    func testLateResumeCannotOverwriteStateAfterProfileSwitch() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.suspendedMethods = ["session.resume"]
        firstClient.resultByMethod["session.resume"] = .object([
            "session_id": .string("obsolete-live"),
            "resumed": .string("obsolete-saved"),
            "messages": .array([.object(["role": .string("assistant"), "text": .string("stale")])]),
        ])
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        let oldSession = HermesSavedSession(
            id: "obsolete-saved", title: "Old", preview: "", startedAt: 0, messageCount: 1, source: ""
        )

        let resume = Task { try await store.resume(oldSession, profile: bernd, configuration: configuration) }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "session.resume" }) }
        await store.loadSessions(profile: researcher, configuration: configuration)
        firstClient.finishCall(method: "session.resume")
        do {
            _ = try await resume.value
            XCTFail("Stale resume must not complete after a profile switch")
        } catch is CancellationError {
            // Expected: the newer profile owns the store state.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        XCTAssertEqual(store.selectedProfile?.name, "researcher")
        XCTAssertNil(store.activeSessionID)
        XCTAssertEqual(store.messages, [])
    }

    @MainActor
    func testPrepareReapsOnceAndPublishesProfileRouting() async {
        let runtime = FakeHermesEmbeddedRuntime()
        runtime.routingByProfile = ["default": .mtplx, "bernd": .external, "researcher": .unavailable("invalid")]
        let store = makeStore(runtime: runtime, clients: [])
        var remembered = configuration!
        remembered.lastHermesProfile = "bernd"

        await store.prepare(configuration: remembered)
        await store.prepare(configuration: remembered)

        XCTAssertEqual(runtime.reapCount, 1)
        XCTAssertEqual(store.selectedProfile?.name, "bernd")
        XCTAssertEqual(store.profileRoutingStates["default"], .mtplx)
        XCTAssertEqual(store.profileRoutingStates["bernd"], .external)
        XCTAssertEqual(store.profileRoutingStates["researcher"], .unavailable("invalid"))
    }

    @MainActor
    private func makeStore(
        runtime: FakeHermesEmbeddedRuntime,
        clients: [FakeHermesGatewayClient]
    ) -> HermesAgentStore {
        var remaining = clients
        return HermesAgentStore(
            integration: integration,
            embeddedRuntime: runtime,
            clientFactory: { _ in
                guard !remaining.isEmpty else { fatalError("Missing fake client") }
                return remaining.removeFirst()
            }
        )
    }

    @MainActor
    private func waitUntil(
        timeout: TimeInterval = 1,
        condition: @escaping @MainActor () -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline {
            await Task.yield()
        }
        XCTAssertTrue(condition())
    }
}

private final class FakeHermesEmbeddedRuntime: HermesEmbeddedRuntime, @unchecked Sendable {
    var routingByProfile: [String: HermesProfileRoutingState] = [:]
    var startCount = 0
    var reapCount = 0
    private(set) var sidecars: [FakeHermesSidecar] = []

    func routingState(for profile: HermesProfile, configuration: MTPLXAppConfiguration) -> HermesProfileRoutingState {
        routingByProfile[profile.name] ?? .external
    }

    func startEmbeddedSidecar(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> any HermesSidecarControlling {
        startCount += 1
        let sidecar = FakeHermesSidecar(index: startCount)
        sidecars.append(sidecar)
        return sidecar
    }

    func sessionOwnership(
        profile: HermesProfile,
        sessionID: String,
        ownedSidecarPID: Int32?
    ) -> HermesSessionOwnership {
        .appOwned
    }

    @discardableResult
    func reapOrphanedEmbeddedSidecars() -> [Int32] {
        reapCount += 1
        return []
    }
}

private final class FakeHermesSidecar: HermesSidecarControlling, @unchecked Sendable {
    let processIdentifier: Int32
    var isRunning = true
    let webSocketURL: URL
    let ownershipRecordURL: URL
    private(set) var stopCount = 0

    init(index: Int) {
        processIdentifier = Int32(index + 100)
        webSocketURL = URL(string: "ws://127.0.0.1:18080/api/ws?token=fake")!
        ownershipRecordURL = URL(fileURLWithPath: "/tmp/fake-hermes-sidecar-\(index)")
    }

    func stop() {
        stopCount += 1
        isRunning = false
    }
}

@MainActor
private final class FakeHermesGatewayClient: HermesGatewayClientProtocol {
    struct Call: Equatable {
        let method: String
        let params: [String: JSONValue]
    }

    var onEvent: ((HermesGatewayEvent) -> Void)?
    var onDisconnect: ((String) -> Void)?
    var resultByMethod: [String: JSONValue] = [:]
    private(set) var calls: [Call] = []
    private(set) var didClose = false
    private var ready = false
    private var readinessWaiters: [CheckedContinuation<Void, Error>] = []
    var suspendedMethods: Set<String> = []
    private var callWaiters: [String: [CheckedContinuation<JSONValue, Error>]] = [:]

    init(readyImmediately: Bool = false) {
        ready = readyImmediately
    }

    func connectAndWaitUntilReady(timeoutSeconds: Double) async throws {
        if ready { return }
        try await withCheckedThrowingContinuation { readinessWaiters.append($0) }
    }

    func call(method: String, params: [String: JSONValue]) async throws -> JSONValue {
        calls.append(Call(method: method, params: params))
        if suspendedMethods.contains(method) {
            return try await withCheckedThrowingContinuation { continuation in
                callWaiters[method, default: []].append(continuation)
            }
        }
        return resultByMethod[method] ?? .object([:])
    }

    func close() {
        didClose = true
        readinessWaiters.forEach { $0.resume(throwing: HermesGatewayClientError.disconnected) }
        readinessWaiters.removeAll()
    }

    func finishReady() {
        ready = true
        onEvent?(HermesGatewayEvent(type: "gateway.ready", sessionID: nil, payload: [:]))
        readinessWaiters.forEach { $0.resume() }
        readinessWaiters.removeAll()
    }

    func disconnect(_ message: String) {
        onDisconnect?(message)
    }

    func finishCall(method: String) {
        let value = resultByMethod[method] ?? .object([:])
        let waiters = callWaiters.removeValue(forKey: method) ?? []
        waiters.forEach { $0.resume(returning: value) }
    }
}
