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

        XCTAssertEqual(firstClient.closeCount, 1)
        XCTAssertEqual(runtime.sidecars[0].stopCount, 1)
        XCTAssertEqual(runtime.sidecars[1].stopCount, 0)
        XCTAssertEqual(store.selectedProfile?.name, "researcher")
    }

    @MainActor
    func testReleasedDisconnectCallbacksDoNotRetainOldClientsOrSidecars() async {
        let runtime = FakeHermesEmbeddedRuntime()
        var firstClient: FakeHermesGatewayClient? = FakeHermesGatewayClient(readyImmediately: true)
        var secondClient: FakeHermesGatewayClient? = FakeHermesGatewayClient(readyImmediately: true)
        let thirdClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient!.resultByMethod["session.list"] = .object(["sessions": .array([])])
        secondClient!.resultByMethod["session.list"] = .object(["sessions": .array([])])
        thirdClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        weak let releasedFirstClient = firstClient
        weak let releasedSecondClient = secondClient
        let store = makeStore(runtime: runtime, clients: [firstClient!, secondClient!, thirdClient])

        await store.loadSessions(profile: bernd, configuration: configuration)
        weak let releasedFirstSidecar = runtime.sidecars[0]
        await store.loadSessions(profile: researcher, configuration: configuration)

        firstClient!.disconnect("late first disconnect")
        XCTAssertEqual(store.selectedProfile?.name, "researcher")
        XCTAssertEqual(store.connectionState, .connected)

        weak let releasedSecondSidecar = runtime.sidecars[1]
        await store.loadSessions(profile: bernd, configuration: alternateConfiguration())
        secondClient!.disconnect("late second disconnect")
        XCTAssertEqual(store.selectedProfile?.name, "bernd")
        XCTAssertEqual(store.connectionState, .connected)

        runtime.discardStoppedSidecars()
        firstClient = nil
        secondClient = nil

        XCTAssertNil(releasedFirstClient)
        XCTAssertNil(releasedSecondClient)
        XCTAssertNil(releasedFirstSidecar)
        XCTAssertNil(releasedSecondSidecar)
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
    func testStreamingReasoningToolsAndCompletionUpdateTranscript() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "message.start", sessionID: "live-1", payload: [:]))
        client.emit(.init(type: "message.delta", sessionID: "live-1", payload: ["text": .string("Hel")]))
        client.emit(.init(type: "tool.start", sessionID: "live-1", payload: ["name": .string("terminal")]))
        client.emit(.init(type: "message.complete", sessionID: "live-1", payload: [
            "text": .string("Hello"),
            "reasoning": .string("checked state"),
        ]))

        XCTAssertEqual(store.messages.last?.text, "Hello")
        XCTAssertTrue(store.toolTraces.contains(where: { $0.name == "Thought" }))
        XCTAssertFalse(store.isStreaming)
    }

    @MainActor
    func testTerminalCompletionErrorNeverSurfacesRawPayload() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        let rawTerminalDetail = UUID().uuidString

        client.emit(.init(type: "message.start", sessionID: "live-1", payload: [:]))
        client.emit(.init(type: "message.complete", sessionID: "live-1", payload: [
            "status": .string("error"),
            "text": .string(rawTerminalDetail),
            "error": .string(rawTerminalDetail),
        ]))

        XCTAssertEqual(store.messages.last?.role, .system)
        XCTAssertEqual(store.messages.last?.text, "Hermes reported an error.")
        XCTAssertFalse(store.messages.contains(where: { $0.text == rawTerminalDetail }))
        XCTAssertFalse(store.toolTraces.contains(where: { $0.detail == rawTerminalDetail }))
        XCTAssertFalse(store.isStreaming)
    }

    @MainActor
    func testReasoningAvailableAddsThoughtTraceWithoutPersistingVerboseMetadata() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "reasoning.available", sessionID: "live-1", payload: [
            "text": .string("checked state"),
            "verbose": .bool(true),
        ]))

        XCTAssertTrue(store.toolTraces.contains(where: { $0.name == "Thought" && $0.detail == "checked state" }))
    }

    @MainActor
    func testAskFirstSurfacesApprovalAndRespondsWithSelectedChoice() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: [
            "command": .string("git status"),
            "choices": .array([.string("once"), .string("deny")]),
        ]))

        XCTAssertEqual(store.pendingRequest?.kind, .approval)
        XCTAssertEqual(store.pendingRequest?.choices, ["once", "deny"])
        await store.respondToPendingRequest(value: "once")
        XCTAssertEqual(client.calls.last?.method, "approval.respond")
        XCTAssertEqual(client.calls.last?.params, [
            "session_id": .string("live-1"),
            "choice": .string("once"),
        ])
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testManualApprovalFIFOMapsAThenBToNativeHeads() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(store.pendingRequest?.prompt, "B")
        await store.respondToPendingRequest(value: "once")
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.map { $0.params["choice"] }, [.string("deny"), .string("once")])
    }

    @MainActor
    func testUnifiedInboxKeepsApprovalAheadOfLaterClarification() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("B"), "question": .string("B?")]))
        XCTAssertEqual(store.pendingRequest?.kind, .approval)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(store.pendingRequest?.id, "B")
        await store.respondToPendingRequest(value: "answer")
        XCTAssertEqual(client.calls.suffix(2).map(\.method), ["approval.respond", "clarify.respond"])
    }

    @MainActor
    func testUnifiedInboxDefersAutoApprovalUntilEarlierClarificationResolves() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("B"), "question": .string("B?")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        XCTAssertEqual(store.pendingRequest?.id, "B")
        XCTAssertFalse(client.calls.contains { $0.method == "approval.respond" })
        await store.respondToPendingRequest(value: "answer")
        await waitUntil { client.calls.contains { $0.method == "approval.respond" } }
        XCTAssertEqual(client.calls.last?.params["choice"], .string("once"))
    }

    @MainActor
    func testUnifiedInboxAutoApprovalThenClarificationDoesNotOverwriteHead() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.contains { $0.method == "approval.respond" } }
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("B"), "question": .string("B?")]))
        XCTAssertNil(store.pendingRequest)
        client.finishCall(method: "approval.respond")
        client.suspendedMethods = []
        await waitUntil { store.pendingRequest?.id == "B" }
    }

    @MainActor
    func testUnifiedInboxPreservesMixedArrivalOrderAndQueuedExpiry() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("B"), "question": .string("B?")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("C")]))
        client.emit(.init(type: "secret.request", sessionID: "live-1", payload: ["request_id": .string("D"), "prompt": .string("D?")]))
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(store.pendingRequest?.id, "B")
        await store.respondToPendingRequest(value: "answer")
        XCTAssertEqual(store.pendingRequest?.prompt, "C")
        client.emit(.init(type: "secret.expire", sessionID: "live-1", payload: ["request_id": .string("D")]))
        await store.respondToPendingRequest(value: "deny")
        XCTAssertNil(store.pendingRequest)
        XCTAssertEqual(client.calls.suffix(3).map(\.method), ["approval.respond", "clarify.respond", "approval.respond"])
    }

    @MainActor
    func testIdenticalManualApprovalRequestsRemainDistinctFIFOEntries() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        let event = HermesGatewayEvent(type: "approval.request", sessionID: "live-1", payload: ["command": .string("same")])
        client.emit(event); client.emit(event)
        let firstID = store.pendingRequest?.id
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(store.pendingRequest?.prompt, "same")
        XCTAssertNotEqual(store.pendingRequest?.id, firstID)
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 2)
    }

    @MainActor
    func testApprovalResponseFailureKeepsHeadAndBlocksUntilFreshSidecar() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient(readyImmediately: true)
        for client in [first, second] {
            client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
            client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        }
        let store = makeStore(runtime: runtime, clients: [first, second])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        first.suspendedMethods = ["approval.respond"]
        let answer = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.failCall(method: "approval.respond")
        await answer.value
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(first.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(first.closeCount, 1)
        XCTAssertEqual(runtime.sidecars.first?.stopCount, 1)
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testTransportTerminationRetainsApprovalHeadBeforeOldResponseFailureRuns() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))

        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.terminatePendingCallsThenDisconnect("transport lost")

        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        XCTAssertEqual(store.pendingRequestInboxCount, 2)
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(client.closeCount, 1)
        XCTAssertEqual(runtime.sidecars.first?.stopCount, 1)
        XCTAssertEqual(store.messages.filter { $0.text.contains("approval recovery") }.count, 1)

        await response.value
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        XCTAssertEqual(store.pendingRequestInboxCount, 2)
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(client.closeCount, 1)
        XCTAssertEqual(runtime.sidecars.first?.stopCount, 1)
        XCTAssertEqual(store.messages.filter { $0.text.contains("approval recovery") }.count, 1)
    }

    @MainActor
    func testBlockedApprovalSurvivesPreReadyReconnectUntilFreshHandshakeSucceeds() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient()
        let third = FakeHermesGatewayClient(readyImmediately: true)
        first.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        first.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        first.suspendedMethods = ["approval.respond"]
        second.resultByMethod["session.list"] = .object(["sessions": .array([])])
        third.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [first, second, third])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.terminatePendingCallsThenDisconnect("transport lost")
        await response.value

        let reconnect = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
        await waitUntil { second.hasDisconnectHandler }
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")

        second.disconnect("reconnect lost before ready")
        await reconnect.value
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")

        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testAutoApprovalOverflowProjectsSuspendedHeadBeforeExactOnceTeardown() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }

        for index in 1...64 {
            switch index % 3 {
            case 0:
                client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("approval-\(index)")]))
            case 1:
                client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
                    "request_id": .string("clarify-\(index)"), "question": .string("Q\(index)?"),
                ]))
            default:
                client.emit(.init(type: "secret.request", sessionID: "live-1", payload: [
                    "request_id": .string("secret-\(index)"), "prompt": .string("S\(index)?"),
                ]))
            }
        }

        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        XCTAssertEqual(store.pendingRequestInboxCount, 64)
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(client.closeCount, 1)
        XCTAssertEqual(runtime.sidecars.first?.stopCount, 1)
    }

    @MainActor
    func testOrdinaryDisconnectStillClearsNonApprovalPendingRequests() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("clarify"), "question": .string("Q?")]))
        client.emit(.init(type: "sudo.request", sessionID: "live-1", payload: ["request_id": .string("sudo"), "prompt": .string("S?")]))
        client.emit(.init(type: "secret.request", sessionID: "live-1", payload: ["request_id": .string("secret"), "prompt": .string("Secret?")]))

        client.disconnect("transport lost")

        XCTAssertNil(store.pendingRequest)
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertEqual(runtime.sidecars.first?.stopCount, 1)
    }

    @MainActor
    func testFailedPreReadyProfileSwitchClearsBlockedApprovalBeforePublishingNewProfile() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient()
        first.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        first.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        first.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [first, second])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.terminatePendingCallsThenDisconnect("transport lost")
        await response.value

        let switchProfile = Task { await store.loadSessions(profile: researcher, configuration: configuration) }
        await waitUntil { second.hasDisconnectHandler }
        XCTAssertEqual(store.selectedProfile?.name, "researcher")
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
        XCTAssertEqual(store.pendingRequestInboxCount, 0)

        second.disconnect("researcher not ready")
        await switchProfile.value
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testFailedPreReadyDifferentSessionSwitchClearsBlockedApproval() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient()
        first.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        first.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        first.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [first, second])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.terminatePendingCallsThenDisconnect("transport lost")
        await response.value

        let saved = HermesSavedSession(id: "saved-2", title: "Other", preview: "", startedAt: 0, messageCount: 0, source: "")
        let switchSession = Task { try await store.resume(saved, profile: bernd, configuration: configuration) }
        await waitUntil { second.hasDisconnectHandler }
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
        XCTAssertEqual(store.pendingRequestInboxCount, 0)

        second.disconnect("session not ready")
        do {
            _ = try await switchSession.value
            XCTFail("Pre-ready session switch must fail")
        } catch {
            // Expected: no fresh client reached gateway.ready.
        }
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testConsecutiveFailedPreReadySameProfileRecoveriesRetainBlockedFIFOUntilSuccess() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient()
        let third = FakeHermesGatewayClient()
        let fourth = FakeHermesGatewayClient(readyImmediately: true)
        first.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        first.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        first.suspendedMethods = ["approval.respond"]
        fourth.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [first, second, third, fourth])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.terminatePendingCallsThenDisconnect("transport lost")
        await response.value

        for failedClient in [second, third] {
            let reconnect = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
            await waitUntil { failedClient.hasDisconnectHandler }
            failedClient.disconnect("same profile not ready")
            await reconnect.value
            XCTAssertTrue(store.approvalPipelineBlocked)
            XCTAssertEqual(store.pendingRequest?.prompt, "A")
            XCTAssertEqual(store.pendingRequestInboxCount, 2)
        }

        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
        XCTAssertEqual(store.pendingRequestInboxCount, 0)
    }

    @MainActor
    func testConsecutiveSameProfileSidecarStartFailuresRetainBlockedFIFOUntilSuccess() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let first = FakeHermesGatewayClient(readyImmediately: true)
        let second = FakeHermesGatewayClient(readyImmediately: true)
        first.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        first.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        first.suspendedMethods = ["approval.respond"]
        second.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [first, second])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        first.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        let response = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { first.calls.contains { $0.method == "approval.respond" } }
        first.terminatePendingCallsThenDisconnect("transport lost")
        await response.value

        runtime.startFailuresRemaining = 2
        await store.loadSessions(profile: bernd, configuration: configuration)
        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        XCTAssertEqual(store.pendingRequestInboxCount, 2)

        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertFalse(store.approvalPipelineBlocked)
        XCTAssertNil(store.pendingRequest)
        XCTAssertEqual(store.pendingRequestInboxCount, 0)
    }

    @MainActor
    func testAutoApprovalFailureKeepsUnknownHeadAndRetainsLaterFIFOEntries() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: ["request_id": .string("B"), "question": .string("B?")]))
        client.failCall(method: "approval.respond")
        await waitUntil { store.approvalPipelineBlocked }
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        // A lifecycle reset is required before B may ever become actionable.
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
    }

    @MainActor
    func testApprovalExpireIsIgnoredAndQueueOverflowFailsClosedWithoutRPC() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        for index in 0...64 {
            if index.isMultiple(of: 2) {
                client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A\(index)")]))
            } else {
                client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
                    "request_id": .string("clarify-\(index)"), "question": .string("Q\(index)?"),
                ]))
            }
        }
        client.emit(.init(type: "approval.expire", sessionID: "live-1", payload: [:]))
        XCTAssertTrue(store.approvalPipelineBlocked)
        XCTAssertEqual(store.pendingRequest?.prompt, "A0")
        XCTAssertFalse(client.calls.contains { $0.method == "approval.respond" })
    }

    @MainActor
    func testRetiredApprovalFingerprintUsesInjectedMonotonicClock() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let clock = TestMonotonicClock(now: 100)
        let store = makeStore(runtime: runtime, clients: [client], monotonicClock: { clock.now })
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        let event = HermesGatewayEvent(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")])
        client.emit(event)
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.emit(event)
        XCTAssertEqual(store.pendingRequest?.prompt, "A")
        await store.respondToPendingRequest(value: "deny")
        clock.now += 31
        client.emit(event)
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 3 }
    }

    @MainActor
    func testPendingApprovalPausesComposerAndDeniesWithNativeChoice() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")]))
        await store.send("must remain paused")
        await store.denyPendingApproval()

        XCTAssertFalse(client.calls.contains(where: { $0.method == "prompt.submit" }))
        XCTAssertEqual(client.calls.last?.method, "approval.respond")
        XCTAssertEqual(client.calls.last?.params["choice"], .string("deny"))
    }

    @MainActor
    func testOwnershipTakeoverBeforePendingResponsePreventsRPC() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")]))

        runtime.ownershipBySession["live-1"] = .external(surface: "telegram")
        await store.respondToPendingRequest(value: "once")

        XCTAssertFalse(client.calls.contains(where: { $0.method == "approval.respond" }))
        XCTAssertEqual(store.pendingRequest?.kind, .approval)
        XCTAssertFalse(store.activeSessionWritable)
    }

    @MainActor
    func testConcurrentApprovalResponsesIssueOnlyOneRPC() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")]))

        let first = Task { await store.respondToPendingRequest(value: "once") }
        await waitUntil { client.calls.contains(where: { $0.method == "approval.respond" }) }
        let second = Task { await store.respondToPendingRequest(value: "once") }
        await Task.yield()
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        client.finishCall(method: "approval.respond")
        await first.value
        await second.value

        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testConfiguredAutoApproveRespondsOnceWithoutPersistentScope() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")]))
        await waitUntil { client.calls.contains(where: { $0.method == "approval.respond" }) }

        let response = try XCTUnwrap(client.calls.last(where: { $0.method == "approval.respond" }))
        XCTAssertEqual(response.params["session_id"], .string("live-1"))
        XCTAssertEqual(response.params["choice"], .string("once"))
        XCTAssertNil(response.params["all"])
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testDuplicateAutoApproveEventsIssueOneOnceRPC() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        let event = HermesGatewayEvent(type: "approval.request", sessionID: "live-1", payload: [
            "command": .string("git status"),
            "choices": .array([.string("once"), .string("deny")]),
        ])

        client.emit(event)
        client.emit(event)
        await waitUntil { client.calls.contains(where: { $0.method == "approval.respond" }) }

        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(client.calls.last?.params["choice"], .string("once"))
        XCTAssertNil(client.calls.last?.params["all"])
        client.finishCall(method: "approval.respond")
        client.suspendedMethods = []

        await waitUntil { store.pendingRequest?.kind == .approval }
        XCTAssertEqual(store.pendingRequest?.prompt, "git status")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(
            client.calls.filter { $0.method == "approval.respond" }.map { $0.params["choice"] },
            [.string("once"), .string("deny")]
        )
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testQueuedAutoApprovalsDowngradeToManualFIFO() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("C")]))
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)

        client.finishCall(method: "approval.respond")
        client.suspendedMethods = []
        await waitUntil { store.pendingRequest?.prompt == "B" }
        await store.respondToPendingRequest(value: "deny")
        await waitUntil { store.pendingRequest?.prompt == "C" }
        await store.respondToPendingRequest(value: "once")

        XCTAssertEqual(
            client.calls.filter { $0.method == "approval.respond" }.map { $0.params["choice"] },
            [.string("once"), .string("deny"), .string("once")]
        )
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testLateDuplicateOfAutoApprovedRequestDowngradesToManual() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        let event = HermesGatewayEvent(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")])

        client.emit(event)
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        for _ in 0..<8 { await Task.yield() }
        client.emit(event)

        await waitUntil { store.pendingRequest?.kind == .approval }
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
        XCTAssertEqual(store.pendingRequest?.prompt, "git status")
        await store.respondToPendingRequest(value: "deny")
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 2)
    }

    @MainActor
    func testQueuedAutoApprovalTakeoverRemainsManualAndReadOnly() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        runtime.ownershipBySession["live-1"] = .external(surface: "telegram")
        client.finishCall(method: "approval.respond")

        await waitUntil { store.pendingRequest?.prompt == "B" }
        XCTAssertFalse(store.activeSessionWritable)
        await store.respondToPendingRequest(value: "deny")
        XCTAssertFalse(store.activeSessionWritable)
        XCTAssertEqual(client.calls.filter { $0.method == "approval.respond" }.count, 1)
    }

    @MainActor
    func testDisconnectClearsQueuedAutoApprovals() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        firstClient.suspendedMethods = ["approval.respond"]
        secondClient.resultByMethod["session.create"] = .object(["session_id": .string("live-2")])
        secondClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        firstClient.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { firstClient.calls.filter { $0.method == "approval.respond" }.count == 1 }
        firstClient.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))

        firstClient.disconnect("transport lost")
        firstClient.finishCall(method: "approval.respond")
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        for _ in 0..<8 { await Task.yield() }

        XCTAssertNil(store.pendingRequest)
        XCTAssertFalse(secondClient.calls.contains(where: { $0.method == "approval.respond" }))
    }

    @MainActor
    func testExpiryDuringQueuedManualResponseAdvancesFIFO() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("A")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("B")]))
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("C")]))
        client.finishCall(method: "approval.respond")
        client.suspendedMethods = []
        await waitUntil { store.pendingRequest?.prompt == "B" }

        client.suspendedMethods = ["approval.respond"]
        let response = Task { await store.respondToPendingRequest(value: "deny") }
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 2 }
        client.emit(.init(type: "approval.expire", sessionID: "live-1", payload: [:]))
        client.finishCall(method: "approval.respond")
        client.suspendedMethods = []
        await response.value

        await waitUntil { store.pendingRequest?.prompt == "C" }
        await store.respondToPendingRequest(value: "once")
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testLaterAutoApprovalCanRespondAfterEarlierLeaseCompletes() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("first")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 1 }
        client.finishCall(method: "approval.respond")
        for _ in 0..<8 { await Task.yield() }
        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("later")]))
        await waitUntil { client.calls.filter { $0.method == "approval.respond" }.count == 2 }
        client.finishCall(method: "approval.respond")
    }

    @MainActor
    func testAutoApprovalTakeoverPreventsRPC() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        runtime.ownershipBySession["live-1"] = .external(surface: "telegram")

        client.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("git status")]))
        await Task.yield()

        XCTAssertFalse(client.calls.contains(where: { $0.method == "approval.respond" }))
        XCTAssertFalse(store.activeSessionWritable)
    }

    @MainActor
    func testStaleAutoApprovalCompletionCannotReleaseNewGenerationLease() async throws {
        configuration.hermesAutoApprove = true
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        secondClient.resultByMethod["session.create"] = .object(["session_id": .string("live-2")])
        secondClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        firstClient.suspendedMethods = ["approval.respond"]
        secondClient.suspendedMethods = ["approval.respond"]
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        firstClient.emit(.init(type: "approval.request", sessionID: "live-1", payload: ["command": .string("old")]))
        await waitUntil { firstClient.calls.contains(where: { $0.method == "approval.respond" }) }

        firstClient.disconnect("transport lost")
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        secondClient.emit(.init(type: "approval.request", sessionID: "live-2", payload: ["command": .string("new")]))
        await waitUntil { secondClient.calls.contains(where: { $0.method == "approval.respond" }) }
        firstClient.finishCall(method: "approval.respond")
        for _ in 0..<8 { await Task.yield() }
        secondClient.emit(.init(type: "approval.request", sessionID: "live-2", payload: ["command": .string("new")]))

        XCTAssertEqual(secondClient.calls.filter { $0.method == "approval.respond" }.count, 1)
        secondClient.finishCall(method: "approval.respond")
    }

    @MainActor
    func testClarificationResponseAndMatchingExpiryUseNativeRequestID() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
            "request_id": .string("clarify-1"),
            "question": .string("Which target?"),
            "choices": .array([.string("A"), .string("B")]),
        ]))
        client.emit(.init(type: "clarify.expire", sessionID: "live-1", payload: ["request_id": .string("stale")]))
        XCTAssertEqual(store.pendingRequest?.id, "clarify-1")

        await store.respondToPendingRequest(value: "B")
        XCTAssertEqual(client.calls.last?.method, "clarify.respond")
        XCTAssertEqual(client.calls.last?.params, [
            "request_id": .string("clarify-1"),
            "answer": .string("B"),
        ])
        XCTAssertNil(store.pendingRequest)

        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
            "request_id": .string("clarify-2"),
            "question": .string("Again?"),
        ]))
        client.emit(.init(type: "clarify.expire", sessionID: "live-1", payload: ["request_id": .string("clarify-2")]))
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testSudoAndSecretResponsesUseNativeFieldsWithoutPersistingInput() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "sudo.request", sessionID: "live-1", payload: ["request_id": .string("sudo-1")]))
        let sudoInput = UUID().uuidString
        await store.respondToPendingRequest(value: sudoInput)
        XCTAssertEqual(client.calls.last?.method, "sudo.respond")
        XCTAssertEqual(client.calls.last?.params["request_id"], .string("sudo-1"))
        XCTAssertEqual(client.calls.last?.params["password"], .string(sudoInput))

        client.emit(.init(type: "secret.request", sessionID: "live-1", payload: [
            "request_id": .string("secret-1"),
            "prompt": .string("Credential requested"),
        ]))
        let secretInput = UUID().uuidString
        await store.respondToPendingRequest(value: secretInput)
        XCTAssertEqual(client.calls.last?.method, "secret.respond")
        XCTAssertEqual(client.calls.last?.params["request_id"], .string("secret-1"))
        XCTAssertEqual(client.calls.last?.params["value"], .string(secretInput))
        XCTAssertFalse(store.messages.contains(where: { $0.text == sudoInput || $0.text == secretInput }))
    }

    @MainActor
    func testExpiredOrSupersededRequestCannotClearNewerPromptAfterResponseReturns() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        client.suspendedMethods = ["clarify.respond"]
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
            "request_id": .string("clarify-old"), "question": .string("Old prompt?"),
        ]))
        let response = Task { await store.respondToPendingRequest(value: "old") }
        await waitUntil { client.calls.contains(where: { $0.method == "clarify.respond" }) }
        client.emit(.init(type: "clarify.expire", sessionID: "live-1", payload: ["request_id": .string("clarify-old")]))
        client.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
            "request_id": .string("clarify-new"), "question": .string("New prompt?"),
        ]))
        client.finishCall(method: "clarify.respond")
        await response.value

        XCTAssertEqual(store.pendingRequest?.id, "clarify-new")
    }

    @MainActor
    func testErrorDisconnectAndSessionSwitchClearStreamingAndPendingState() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        firstClient.emit(.init(type: "message.start", sessionID: "live-1", payload: [:]))
        firstClient.emit(.init(type: "clarify.request", sessionID: "live-1", payload: [
            "request_id": .string("clarify-1"), "question": .string("Continue?"),
        ]))
        firstClient.emit(.init(type: "error", sessionID: "live-1", payload: ["message": .string("turn failed")]))
        XCTAssertFalse(store.isStreaming)
        XCTAssertNotNil(store.pendingRequest)

        firstClient.disconnect("transport lost")
        XCTAssertFalse(store.isStreaming)
        XCTAssertNil(store.pendingRequest)

        await store.loadSessions(profile: researcher, configuration: configuration)
        XCTAssertFalse(store.isStreaming)
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testEventsFromNonActiveSessionAndRetiredClientAreIgnored() async throws {
        configuration.hermesAutoApprove = false
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("live-1")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        firstClient.emit(.init(type: "message.delta", sessionID: "other-live", payload: ["text": .string("ignored")]))
        firstClient.emit(.init(type: "clarify.request", sessionID: "other-live", payload: [
            "request_id": .string("other-request"), "question": .string("ignored"),
        ]))
        XCTAssertEqual(store.messages, [])
        XCTAssertNil(store.pendingRequest)

        await store.loadSessions(profile: researcher, configuration: configuration)
        firstClient.emit(.init(type: "message.delta", sessionID: "live-1", payload: ["text": .string("retired")]))
        XCTAssertEqual(store.messages, [])
        XCTAssertNil(store.pendingRequest)
    }

    @MainActor
    func testExternalSessionRemainsReadableButCannotSubmitOrInterrupt() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        runtime.ownershipBySession["saved-external"] = .external(surface: "telegram")
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.resume"] = .object([
            "session_id": .string("live-external"),
            "resumed": .string("saved-external"),
            "messages": .array([.object(["role": .string("assistant"), "text": .string("Existing reply")])]),
        ])
        let store = makeStore(runtime: runtime, clients: [client])
        let session = HermesSavedSession(
            id: "saved-external", title: "External", preview: "", startedAt: 0, messageCount: 1, source: ""
        )

        _ = try await store.resume(session, profile: bernd, configuration: configuration)
        await store.send("must not leave MTPLX")
        await store.interrupt()

        XCTAssertEqual(store.messages.map(\.text), ["Existing reply"])
        XCTAssertFalse(store.activeSessionWritable)
        XCTAssertEqual(store.activeSessionActivity, .externallyActive(surface: "telegram"))
        XCTAssertNotNil(store.readOnlyReason)
        XCTAssertFalse(client.calls.contains(where: { $0.method == "prompt.submit" || $0.method == "session.interrupt" }))
    }

    @MainActor
    func testSendRechecksOwnershipImmediatelyBeforeSubmit() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.resume"] = resumeResult(liveID: "live-1", savedID: "saved-1")
        let store = makeStore(runtime: runtime, clients: [client])
        _ = try await store.resume(
            HermesSavedSession(id: "saved-1", title: "Saved", preview: "", startedAt: 0, messageCount: 0, source: ""),
            profile: bernd,
            configuration: configuration
        )
        XCTAssertTrue(store.activeSessionWritable)

        runtime.ownershipBySession["saved-1"] = .external(surface: "telegram")
        await store.send("race check")

        XCTAssertFalse(store.activeSessionWritable)
        XCTAssertEqual(store.activeSessionActivity, .externallyActive(surface: "telegram"))
        XCTAssertFalse(client.calls.contains(where: { $0.method == "prompt.submit" }))
        XCTAssertEqual(store.messages, [])
    }

    @MainActor
    func testListActivityAndDifferentSessionConcurrencyDoNotBlockFreshSession() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        runtime.ownershipBySession["telegram-session"] = .external(surface: "telegram")
        let client = FakeHermesGatewayClient(readyImmediately: true)
        client.resultByMethod["session.list"] = sessionsResult(id: "telegram-session", title: "Telegram")
        client.resultByMethod["session.create"] = .object(["session_id": .string("fresh-live")])
        client.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])

        await store.loadSessions(profile: bernd, configuration: configuration)
        XCTAssertEqual(store.sessions.first?.activity, .externallyActive(surface: "telegram"))

        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)
        XCTAssertTrue(store.activeSessionWritable)
        await store.send("new session is independent")
        XCTAssertTrue(client.calls.contains(where: { $0.method == "prompt.submit" }))
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
    func testLateSameProfileLoadCannotOverwriteNewerConfigurationGeneration() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.suspendedMethods = ["session.list"]
        firstClient.resultByMethod["session.list"] = sessionsResult(id: "old", title: "Old")
        secondClient.resultByMethod["session.list"] = sessionsResult(id: "new", title: "New")
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])

        let oldLoad = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "session.list" }) }
        await store.loadSessions(profile: bernd, configuration: alternateConfiguration())
        firstClient.finishCall(method: "session.list")
        await oldLoad.value

        XCTAssertEqual(store.sessions.map(\.id), ["new"])
        XCTAssertEqual(store.connectionState, .connected)
    }

    @MainActor
    func testLateSameProfileCreateCannotOverwriteNewerConfigurationGeneration() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.suspendedMethods = ["session.create"]
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("obsolete")])
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])

        let create = Task { try await store.startNewAgent(profile: bernd, configuration: configuration) }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "session.create" }) }
        await store.loadSessions(profile: bernd, configuration: alternateConfiguration())
        firstClient.finishCall(method: "session.create")

        await assertCancellation(create)
        XCTAssertNil(store.activeSessionID)
        XCTAssertEqual(store.connectionState, .connected)
    }

    @MainActor
    func testLateSameProfileResumeCannotOverwriteNewerConfigurationGeneration() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.suspendedMethods = ["session.resume"]
        firstClient.resultByMethod["session.resume"] = resumeResult(liveID: "obsolete-live", savedID: "obsolete-saved")
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        let saved = HermesSavedSession(id: "obsolete-saved", title: "Old", preview: "", startedAt: 0, messageCount: 1, source: "")

        let resume = Task { try await store.resume(saved, profile: bernd, configuration: configuration) }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "session.resume" }) }
        await store.loadSessions(profile: bernd, configuration: alternateConfiguration())
        firstClient.finishCall(method: "session.resume")

        await assertCancellation(resume)
        XCTAssertNil(store.activeSessionID)
        XCTAssertEqual(store.messages, [])
    }

    @MainActor
    func testConcurrentReconnectKeepsOnlyNewestSameProfileGeneration() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let initial = FakeHermesGatewayClient(readyImmediately: true)
        let slowReconnect = FakeHermesGatewayClient()
        let newestReconnect = FakeHermesGatewayClient(readyImmediately: true)
        initial.resultByMethod["session.resume"] = resumeResult(liveID: "initial-live", savedID: "saved")
        newestReconnect.resultByMethod["session.resume"] = resumeResult(liveID: "new-live", savedID: "saved")
        let store = makeStore(runtime: runtime, clients: [initial, slowReconnect, newestReconnect])
        let saved = HermesSavedSession(id: "saved", title: "Saved", preview: "", startedAt: 0, messageCount: 1, source: "")
        _ = try await store.resume(saved, profile: bernd, configuration: configuration)
        initial.disconnect("transport lost")

        let olderReconnect = Task { try await store.reconnect(configuration: configuration) }
        await waitUntil { runtime.startCount == 2 }
        let newerReconnect = Task { try await store.reconnect(configuration: configuration) }
        try await newerReconnect.value
        await assertCancellation(olderReconnect)

        XCTAssertEqual(store.activeSessionID, "new-live")
        XCTAssertEqual(store.activeReference?.sessionID, "saved")
        XCTAssertEqual(slowReconnect.closeCount, 1)
        XCTAssertEqual(runtime.startCount, 2)
    }

    @MainActor
    func testGatewayReadyEventDoesNotExposeStoreBeforeConnectReturns() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient()
        client.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [client])

        let load = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
        await waitUntil { client.hasEventHandler }
        client.emitReadyEventWithoutCompletingConnect()
        await Task.yield()

        XCTAssertFalse(store.gatewayReady)
        XCTAssertEqual(store.connectionState, .starting)
        XCTAssertEqual(client.calls, [])

        client.completeReadyConnect()
        await load.value
        XCTAssertTrue(store.gatewayReady)
    }

    @MainActor
    func testDisconnectBeforeReadyStopsOwnedSidecarExactlyOnce() async {
        let runtime = FakeHermesEmbeddedRuntime()
        let client = FakeHermesGatewayClient()
        let store = makeStore(runtime: runtime, clients: [client])

        let load = Task { await store.loadSessions(profile: bernd, configuration: configuration) }
        await waitUntil { client.hasDisconnectHandler }
        client.disconnect("transport lost")
        await load.value

        XCTAssertEqual(runtime.sidecars[0].stopCount, 1)
    }

    @MainActor
    func testLatePromptFailureCannotAppendErrorToNewProfileTranscript() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("old-live")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        firstClient.suspendedMethods = ["prompt.submit"]
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        let send = Task { await store.send("old prompt") }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "prompt.submit" }) }
        await store.loadSessions(profile: researcher, configuration: configuration)
        firstClient.failCall(method: "prompt.submit")
        await send.value

        XCTAssertEqual(store.messages, [])
        XCTAssertFalse(store.isStreaming)
    }

    @MainActor
    func testLateInterruptFailureCannotAppendErrorToNewProfileTranscript() async throws {
        let runtime = FakeHermesEmbeddedRuntime()
        let firstClient = FakeHermesGatewayClient(readyImmediately: true)
        let secondClient = FakeHermesGatewayClient(readyImmediately: true)
        firstClient.resultByMethod["session.create"] = .object(["session_id": .string("old-live")])
        firstClient.resultByMethod["session.active_list"] = .object(["sessions": .array([])])
        firstClient.suspendedMethods = ["session.interrupt"]
        secondClient.resultByMethod["session.list"] = .object(["sessions": .array([])])
        let store = makeStore(runtime: runtime, clients: [firstClient, secondClient])
        _ = try await store.startNewAgent(profile: bernd, configuration: configuration)

        let interrupt = Task { await store.interrupt() }
        await waitUntil { firstClient.calls.contains(where: { $0.method == "session.interrupt" }) }
        await store.loadSessions(profile: researcher, configuration: configuration)
        firstClient.failCall(method: "session.interrupt")
        await interrupt.value

        XCTAssertEqual(store.messages, [])
        XCTAssertFalse(store.isStreaming)
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
    func testStoppedOrCancelledPrepareCannotRepublishAfterDelayedInstallStatus() async throws {
        let script = root.appendingPathComponent("delayed-hermes")
        let source = """
        #!/bin/sh
        sleep 0.2
        case \"$1\" in
          --version) echo \"Hermes 0.19.1\" ;;
          gateway) echo \"running\" ;;
          chat) echo \"--query --source\" ;;
        esac
        """
        try source.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let delayedIntegration = HermesIntegration(
            hermesHome: root.appendingPathComponent(".hermes", isDirectory: true),
            executablePath: script.path,
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            sidecarRuntimeDirectory: root.appendingPathComponent("sidecars", isDirectory: true)
        )
        let runtime = FakeHermesEmbeddedRuntime()
        let store = HermesAgentStore(
            integration: delayedIntegration,
            embeddedRuntime: runtime,
            clientFactory: { _ in FakeHermesGatewayClient(readyImmediately: true) }
        )

        let cancelledPrepare = Task { await store.prepare(configuration: self.configuration) }
        await waitUntil { store.connectionState == .checkingInstall }
        cancelledPrepare.cancel()
        await store.stop()
        await cancelledPrepare.value

        XCTAssertNil(store.installStatus)
        XCTAssertTrue(store.profiles.isEmpty)
        XCTAssertNil(store.selectedProfile)
        XCTAssertEqual(store.connectionState, .idle)

        await store.prepare(configuration: configuration)

        XCTAssertNotNil(store.installStatus)
        XCTAssertFalse(store.profiles.isEmpty)
        XCTAssertEqual(store.selectedProfile?.name, "default")
        XCTAssertEqual(store.connectionState, .idle)
    }

    @MainActor
    private func makeStore(
        runtime: FakeHermesEmbeddedRuntime,
        clients: [FakeHermesGatewayClient],
        monotonicClock: @escaping @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    ) -> HermesAgentStore {
        var remaining = clients
        return HermesAgentStore(
            integration: integration,
            embeddedRuntime: runtime,
            clientFactory: { _ in
                guard !remaining.isEmpty else { fatalError("Missing fake client") }
                return remaining.removeFirst()
            },
            monotonicClock: monotonicClock
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

    @MainActor
    private func alternateConfiguration() -> MTPLXAppConfiguration {
        var copy = configuration!
        copy.port += 1
        return copy
    }

    @MainActor
    private func sessionsResult(id: String, title: String) -> JSONValue {
        .object(["sessions": .array([.object([
            "id": .string(id),
            "title": .string(title),
            "preview": .string(""),
            "started_at": .number(0),
            "message_count": .number(0),
            "source": .string(""),
        ])])])
    }

    @MainActor
    private func resumeResult(liveID: String, savedID: String) -> JSONValue {
        .object([
            "session_id": .string(liveID),
            "resumed": .string(savedID),
            "messages": .array([]),
        ])
    }

    @MainActor
    private func assertCancellation<T>(_ task: Task<T, Error>) async {
        do {
            _ = try await task.value
            XCTFail("Expected stale task cancellation")
        } catch is CancellationError {
            // Expected: a newer generation owns store state.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}

private final class TestMonotonicClock: @unchecked Sendable {
    var now: TimeInterval

    init(now: TimeInterval) {
        self.now = now
    }
}

private final class FakeHermesEmbeddedRuntime: HermesEmbeddedRuntime, @unchecked Sendable {
    var routingByProfile: [String: HermesProfileRoutingState] = [:]
    var ownershipBySession: [String: HermesSessionOwnership] = [:]
    var startCount = 0
    var startFailuresRemaining = 0
    var reapCount = 0
    private(set) var sidecars: [FakeHermesSidecar] = []

    func routingState(for profile: HermesProfile, configuration: MTPLXAppConfiguration) -> HermesProfileRoutingState {
        routingByProfile[profile.name] ?? .external
    }

    func startEmbeddedSidecar(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> any HermesSidecarControlling {
        if startFailuresRemaining > 0 {
            startFailuresRemaining -= 1
            throw FakeHermesRuntimeError.startFailed
        }
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
        ownershipBySession[sessionID] ?? .ready
    }

    @discardableResult
    func reapOrphanedEmbeddedSidecars() -> [Int32] {
        reapCount += 1
        return []
    }

    func discardStoppedSidecars() {
        sidecars.removeAll { !$0.isRunning }
    }
}

private enum FakeHermesRuntimeError: Error {
    case startFailed
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
    private(set) var closeCount = 0
    private var ready = false
    private var readinessWaiters: [CheckedContinuation<Void, Error>] = []
    var suspendedMethods: Set<String> = []
    private var callWaiters: [String: [CheckedContinuation<JSONValue, Error>]] = [:]

    var hasEventHandler: Bool { onEvent != nil }
    var hasDisconnectHandler: Bool { onDisconnect != nil }

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
        closeCount += 1
        readinessWaiters.forEach { $0.resume(throwing: HermesGatewayClientError.disconnected) }
        readinessWaiters.removeAll()
    }

    func finishReady() {
        ready = true
        onEvent?(HermesGatewayEvent(type: "gateway.ready", sessionID: nil, payload: [:]))
        readinessWaiters.forEach { $0.resume() }
        readinessWaiters.removeAll()
    }

    func emitReadyEventWithoutCompletingConnect() {
        ready = true
        onEvent?(HermesGatewayEvent(type: "gateway.ready", sessionID: nil, payload: [:]))
    }

    func completeReadyConnect() {
        readinessWaiters.forEach { $0.resume() }
        readinessWaiters.removeAll()
    }

    func disconnect(_ message: String) {
        onDisconnect?(message)
    }

    func emit(_ event: HermesGatewayEvent) {
        onEvent?(event)
    }

    func finishCall(method: String) {
        let value = resultByMethod[method] ?? .object([:])
        let waiters = callWaiters.removeValue(forKey: method) ?? []
        waiters.forEach { $0.resume(returning: value) }
    }

    func failCall(method: String) {
        let waiters = callWaiters.removeValue(forKey: method) ?? []
        waiters.forEach { $0.resume(throwing: HermesGatewayClientError.disconnected) }
    }

    /// Matches URLSessionHermesGatewayClient's termination order: fail RPCs,
    /// then synchronously notify its owner during the same termination turn.
    func terminatePendingCallsThenDisconnect(_ message: String) {
        let waiters = callWaiters
        callWaiters.removeAll()
        for methodWaiters in waiters.values {
            methodWaiters.forEach { $0.resume(throwing: HermesGatewayClientError.disconnected) }
        }
        onDisconnect?(message)
    }
}
