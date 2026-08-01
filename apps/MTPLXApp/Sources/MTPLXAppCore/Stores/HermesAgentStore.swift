import Combine
import Foundation

public enum HermesConnectionState: Equatable, Sendable {
    case idle
    case checkingInstall
    case needsSetup(String)
    case starting
    case connected
    case failed(String)
}

public enum HermesMessageRole: String, Equatable, Sendable {
    case user
    case assistant
    case system
}

public struct HermesTranscriptMessage: Identifiable, Equatable, Sendable {
    public let id: String
    public var role: HermesMessageRole
    public var text: String
    public var isStreaming: Bool

    public init(
        id: String = UUID().uuidString,
        role: HermesMessageRole,
        text: String,
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
    }
}

public enum HermesToolStatus: String, Equatable, Sendable {
    case running
    case complete
    case approval
    case waiting
    case failed
}

public struct HermesToolTrace: Identifiable, Equatable, Sendable {
    public let id: String
    public var name: String
    public var status: HermesToolStatus
    public var detail: String

    public init(
        id: String = UUID().uuidString,
        name: String,
        status: HermesToolStatus,
        detail: String
    ) {
        self.id = id
        self.name = name
        self.status = status
        self.detail = detail
    }
}

public struct HermesSavedSession: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let preview: String
    public let startedAt: Double
    public let messageCount: Int
    public let source: String

    public init(
        id: String,
        title: String,
        preview: String,
        startedAt: Double,
        messageCount: Int,
        source: String
    ) {
        self.id = id
        self.title = title
        self.preview = preview
        self.startedAt = startedAt
        self.messageCount = messageCount
        self.source = source
    }
}

@MainActor
public final class HermesAgentStore: ObservableObject {
    @Published public private(set) var connectionState: HermesConnectionState = .idle
    @Published public private(set) var installStatus: HermesInstallStatus?
    @Published public private(set) var profiles: [HermesProfile] = []
    @Published public private(set) var profileRoutingStates: [String: HermesProfileRoutingState] = [:]
    @Published public private(set) var selectedProfile: HermesProfile?
    @Published public private(set) var sessions: [HermesSavedSession] = []
    @Published public private(set) var messages: [HermesTranscriptMessage] = []
    @Published public private(set) var toolTraces: [HermesToolTrace] = []
    @Published public private(set) var activeSessionID: String?
    @Published public private(set) var activeSessionKey: String?
    @Published public private(set) var activeSessionTitle: String?
    @Published public private(set) var isStreaming: Bool = false
    @Published public private(set) var gatewayReady: Bool = false
    @Published public private(set) var gatewayRepairInFlight: Bool = false
    @Published public private(set) var gatewayRepairMessage: String?
    @Published public private(set) var terminalAgentRunning: Bool = false

    private let integration: HermesIntegration
    private let embeddedRuntime: any HermesEmbeddedRuntime
    private let clientFactory: HermesGatewayClientFactory
    private var sidecar: (any HermesSidecarControlling)?
    private var sidecarProfileName: String?
    private var sidecarConfigurationSignature: String?
    private var client: (any HermesGatewayClientProtocol)?
    private var shuttingDown = false
    private var gatewayGeneration = 0
    private var didReapOrphanedSidecars = false

    private struct GatewayOperation {
        let generation: Int
        let profileID: String
        let clientIdentity: ObjectIdentifier
    }

    public init(
        integration: HermesIntegration,
        embeddedRuntime: any HermesEmbeddedRuntime,
        clientFactory: @escaping HermesGatewayClientFactory
    ) {
        self.integration = integration
        self.embeddedRuntime = embeddedRuntime
        self.clientFactory = clientFactory
    }

    public convenience init(integration: HermesIntegration = HermesIntegration()) {
        self.init(
            integration: integration,
            embeddedRuntime: integration,
            clientFactory: { URLSessionHermesGatewayClient(url: $0) }
        )
    }

    public var activeReference: HermesSessionReference? {
        guard let selectedProfile,
              let sessionID = activeSessionKey ?? activeSessionID
        else { return nil }
        return HermesSessionReference(
            profileName: selectedProfile.name,
            sessionID: sessionID,
            title: activeSessionTitle
        )
    }

    public func prepare(configuration: MTPLXAppConfiguration) async {
        connectionState = .checkingInstall
        gatewayRepairMessage = nil
        if !didReapOrphanedSidecars {
            _ = embeddedRuntime.reapOrphanedEmbeddedSidecars()
            didReapOrphanedSidecars = true
        }
        let status = await integration.installStatus()
        installStatus = status
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
        profiles = integration.discoverProfiles()
        profileRoutingStates = Dictionary(
            uniqueKeysWithValues: profiles.map { profile in
                (profile.id, embeddedRuntime.routingState(for: profile, configuration: configuration))
            }
        )
        if let remembered = configuration.lastHermesProfile,
           let profile = profiles.first(where: { $0.name == remembered }) {
            selectedProfile = profile
        } else if selectedProfile == nil {
            selectedProfile = profiles.first
        }
        switch status.kind {
        case .ready:
            connectionState = .idle
        case .missing, .incompatible:
            connectionState = .needsSetup(status.detail)
        }
    }

    /// Open (or re-open) the Hermes chat in a Terminal window. Used by
    /// the handoff panel's "Open in Terminal" action so the user can get
    /// back to the live agent without restarting the daemon.
    public func openTerminal(configuration: MTPLXAppConfiguration) {
        _ = integration.launchInTerminal(configuration: configuration)
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
    }

    public func repairGateway() async {
        guard gatewayRepairInFlight == false else { return }
        gatewayRepairInFlight = true
        gatewayRepairMessage = nil
        defer { gatewayRepairInFlight = false }
        do {
            let result = try await integration.repairGateway()
            if let statusSummary = result.statusSummary {
                gatewayRepairMessage = "\(result.startSummary) \(statusSummary)"
            } else {
                gatewayRepairMessage = result.startSummary
            }
            let status = await integration.installStatus()
            installStatus = status
            terminalAgentRunning = integration.hasLaunchedTerminalAgent()
            profiles = integration.discoverProfiles()
            switch status.kind {
            case .ready:
                connectionState = .idle
            case .missing, .incompatible:
                connectionState = .needsSetup(status.detail)
            }
            if result.statusHealth != .healthy {
                gatewayReady = false
            }
        } catch {
            gatewayRepairMessage = Self.message(for: error)
            connectionState = .failed(Self.message(for: error))
        }
    }

    public func loadSessions(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async {
        selectedProfile = profile
        var operation: GatewayOperation?
        do {
            operation = try await ensureGateway(profile: profile, configuration: configuration)
            guard let operation, isCurrent(operation) else { return }
            let result = try await rpc(operation, method: "session.list", params: ["limit": .number(200)])
            guard isCurrent(operation) else { return }
            sessions = Self.parseSessions(result)
            connectionState = .connected
        } catch {
            guard let operation, isCurrent(operation), !shuttingDown else { return }
            sessions = []
            connectionState = .failed(Self.message(for: error))
        }
    }

    @discardableResult
    public func startNewAgent(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        selectedProfile = profile
        let operation = try await ensureGateway(profile: profile, configuration: configuration)
        guard isCurrent(operation) else {
            throw HermesGatewayClientError.disconnected
        }
        let result = try await rpc(operation, method: "session.create", params: ["cols": .number(100)])
        guard isCurrent(operation) else {
            throw CancellationError()
        }
        guard let sessionID = result.objectValue?["session_id"]?.stringValue else {
            throw HermesGatewayClientError.malformedResponse
        }
        let sessionKey = (try? await liveSessionKey(for: sessionID, operation: operation)) ?? sessionID
        guard isCurrent(operation) else {
            throw CancellationError()
        }
        activeSessionID = sessionID
        activeSessionTitle = "New Hermes Agent"
        messages = []
        toolTraces = []
        activeSessionKey = sessionKey
        connectionState = .connected
        return HermesSessionReference(
            profileName: profile.name,
            sessionID: activeSessionKey ?? sessionID,
            title: activeSessionTitle
        )
    }

    @discardableResult
    public func resume(
        _ session: HermesSavedSession,
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        selectedProfile = profile
        let operation = try await ensureGateway(profile: profile, configuration: configuration)
        guard isCurrent(operation) else {
            throw HermesGatewayClientError.disconnected
        }
        let result = try await rpc(
            operation,
            method: "session.resume",
            params: [
                "session_id": .string(session.id),
                "cols": .number(100),
            ]
        )
        guard isCurrent(operation) else {
            throw CancellationError()
        }
        try applyResumedSession(
            result,
            savedSessionID: session.id,
            title: session.title.isEmpty ? session.preview : session.title,
            preserveVisibleTranscript: false
        )
        return HermesSessionReference(
            profileName: profile.name,
            sessionID: activeSessionKey ?? session.id,
            title: activeSessionTitle
        )
    }

    @discardableResult
    public func resumeLast(
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSessionReference {
        guard let profileName = configuration.lastHermesProfile,
              let sessionID = configuration.lastHermesSessionID,
              let profile = (
                profiles.first(where: { $0.name == profileName })
                    ?? selectedProfile.flatMap { $0.name == profileName ? $0 : nil }
              )
        else {
            throw HermesGatewayClientError.rpcError
        }
        let session = HermesSavedSession(
            id: sessionID,
            title: configuration.lastHermesSessionTitle ?? "",
            preview: "",
            startedAt: 0,
            messageCount: 0,
            source: ""
        )
        return try await resume(session, profile: profile, configuration: configuration)
    }

    public func send(_ rawText: String) async {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty,
              let sessionID = activeSessionID,
              let profile = selectedProfile,
              let operation = currentOperation(for: profile),
              !isStreaming
        else { return }
        messages.append(HermesTranscriptMessage(role: .user, text: text))
        isStreaming = true
        do {
            _ = try await rpc(
                operation,
                method: "prompt.submit",
                params: [
                    "session_id": .string(sessionID),
                    "text": .string(text),
                ]
            )
        } catch {
            guard isCurrent(operation, sessionID: sessionID) else { return }
            isStreaming = false
            messages.append(
                HermesTranscriptMessage(
                    role: .system,
                    text: Self.message(for: error)
                )
            )
        }
    }

    public func interrupt() async {
        guard let sessionID = activeSessionID,
              let profile = selectedProfile,
              let operation = currentOperation(for: profile)
        else { return }
        do {
            _ = try await rpc(operation, method: "session.interrupt", params: ["session_id": .string(sessionID)])
        } catch {
            guard isCurrent(operation, sessionID: sessionID) else { return }
            messages.append(
                HermesTranscriptMessage(role: .system, text: Self.message(for: error))
            )
        }
        guard isCurrent(operation, sessionID: sessionID) else { return }
        isStreaming = false
    }

    public func createProfile(named name: String, configuration: MTPLXAppConfiguration) async {
        do {
            let profile = try await integration.createProfile(named: name)
            profiles = integration.discoverProfiles()
            profileRoutingStates = Dictionary(
                uniqueKeysWithValues: profiles.map { profile in
                    (profile.id, embeddedRuntime.routingState(for: profile, configuration: configuration))
                }
            )
            selectedProfile = profiles.first(where: { $0.name == profile.name }) ?? profile
            await loadSessions(profile: selectedProfile ?? profile, configuration: configuration)
        } catch {
            connectionState = .failed(Self.message(for: error))
        }
    }

    public func stop() async {
        shuttingDown = true
        tearDownGateway(clearSession: true)
        activeSessionID = nil
        activeSessionKey = nil
        activeSessionTitle = nil
        sessions = []
        messages = []
        toolTraces = []
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
        connectionState = .idle
    }

    /// Re-establishes the embedded sidecar for the selected profile and
    /// reopens the selected persisted session without discarding the locally
    /// visible transcript while transport recovery is in progress.
    public func reconnect(configuration: MTPLXAppConfiguration) async throws {
        guard let profile = selectedProfile else { throw HermesGatewayClientError.disconnected }
        let savedSessionID = activeSessionKey ?? activeSessionID ?? configuration.lastHermesSessionID
        let title = activeSessionTitle ?? configuration.lastHermesSessionTitle
        let operation = try await ensureGateway(profile: profile, configuration: configuration, preserveSession: true)
        guard isCurrent(operation) else { throw CancellationError() }
        guard let savedSessionID else { return }
        let result = try await rpc(
            operation,
            method: "session.resume",
            params: ["session_id": .string(savedSessionID), "cols": .number(100)]
        )
        guard isCurrent(operation) else {
            throw CancellationError()
        }
        try applyResumedSession(
            result,
            savedSessionID: savedSessionID,
            title: title,
            preserveVisibleTranscript: true
        )
    }

    public func refreshTerminalAgentState() {
        terminalAgentRunning = integration.hasLaunchedTerminalAgent()
    }

    private func ensureGateway(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration,
        preserveSession: Bool = false
    ) async throws -> GatewayOperation {
        let signature = Self.configurationSignature(configuration)
        if sidecarProfileName == profile.name,
           sidecarConfigurationSignature == signature,
           sidecar?.isRunning == true,
           client != nil,
           gatewayReady {
            guard let operation = currentOperation(for: profile) else {
                throw HermesGatewayClientError.disconnected
            }
            return operation
        }
        let reuseSidecar = sidecarProfileName == profile.name
            && sidecarConfigurationSignature == signature
            && sidecar?.isRunning == true
        shuttingDown = true
        gatewayGeneration += 1
        let generation = gatewayGeneration
        client?.close()
        client = nil
        if !reuseSidecar {
            sidecar?.stop()
            sidecar = nil
            sidecarProfileName = nil
            sidecarConfigurationSignature = nil
            if !preserveSession {
                sessions = []
                activeSessionID = nil
                activeSessionKey = nil
                activeSessionTitle = nil
                messages = []
                toolTraces = []
                isStreaming = false
            }
        }
        gatewayReady = false
        shuttingDown = false
        connectionState = .starting
        let nextSidecar: any HermesSidecarControlling
        do {
            if let sidecar, reuseSidecar {
                nextSidecar = sidecar
            } else {
                nextSidecar = try await embeddedRuntime.startEmbeddedSidecar(
                    profile: profile,
                    configuration: configuration
                )
            }
        } catch {
            guard generation == gatewayGeneration, !shuttingDown else { throw CancellationError() }
            connectionState = .failed(Self.message(for: error))
            throw error
        }
        guard generation == gatewayGeneration, !shuttingDown else {
            if !reuseSidecar { nextSidecar.stop() }
            throw CancellationError()
        }
        let nextClient = clientFactory(nextSidecar.webSocketURL)
        nextClient.onEvent = { [weak self] event in
            guard let self, self.gatewayGeneration == generation, !self.shuttingDown else { return }
            self.handle(event)
        }
        nextClient.onDisconnect = { [weak self, weak nextClient, weak nextSidecar] message in
            guard let self,
                  let nextClient,
                  let nextSidecar,
                  self.gatewayGeneration == generation,
                  !self.shuttingDown
            else { return }
            if self.releaseGatewayIfOwned(
                generation: generation,
                client: nextClient,
                sidecar: nextSidecar
            ) {
                self.connectionState = .failed(message)
            }
        }
        sidecar = nextSidecar
        sidecarProfileName = profile.name
        sidecarConfigurationSignature = signature
        client = nextClient
        selectedProfile = profile
        do {
            try await nextClient.connectAndWaitUntilReady(timeoutSeconds: 10)
        } catch {
            guard generation == gatewayGeneration, !shuttingDown else { throw CancellationError() }
            if releaseGatewayIfOwned(generation: generation, client: nextClient, sidecar: nextSidecar) {
                connectionState = .failed(Self.message(for: error))
            }
            throw error
        }
        guard generation == gatewayGeneration, !shuttingDown else {
            _ = releaseGatewayIfOwned(generation: generation, client: nextClient, sidecar: nextSidecar)
            throw CancellationError()
        }
        gatewayReady = true
        connectionState = .connected
        return GatewayOperation(
            generation: generation,
            profileID: profile.id,
            clientIdentity: ObjectIdentifier(nextClient)
        )
    }

    private func rpc(
        _ operation: GatewayOperation,
        method: String,
        params: [String: JSONValue] = [:]
    ) async throws -> JSONValue {
        guard isCurrent(operation), let client else {
            throw HermesGatewayClientError.disconnected
        }
        return try await client.call(method: method, params: params)
    }

    private func currentOperation(for profile: HermesProfile) -> GatewayOperation? {
        guard gatewayReady,
              selectedProfile?.id == profile.id,
              let client
        else { return nil }
        return GatewayOperation(
            generation: gatewayGeneration,
            profileID: profile.id,
            clientIdentity: ObjectIdentifier(client)
        )
    }

    private func isCurrent(_ operation: GatewayOperation, sessionID: String? = nil) -> Bool {
        guard gatewayReady,
              gatewayGeneration == operation.generation,
              selectedProfile?.id == operation.profileID,
              let client,
              ObjectIdentifier(client) == operation.clientIdentity
        else { return false }
        return sessionID.map { $0 == activeSessionID } ?? true
    }

    private func releaseGatewayIfOwned(
        generation: Int,
        client expectedClient: any HermesGatewayClientProtocol,
        sidecar expectedSidecar: any HermesSidecarControlling
    ) -> Bool {
        guard gatewayGeneration == generation,
              let client,
              ObjectIdentifier(client) == ObjectIdentifier(expectedClient),
              let sidecar,
              ObjectIdentifier(sidecar) == ObjectIdentifier(expectedSidecar)
        else { return false }
        self.client = nil
        self.sidecar = nil
        sidecarProfileName = nil
        sidecarConfigurationSignature = nil
        gatewayReady = false
        isStreaming = false
        expectedClient.close()
        expectedSidecar.stop()
        return true
    }

    private func tearDownGateway(clearSession: Bool) {
        gatewayGeneration += 1
        client?.close()
        client = nil
        sidecar?.stop()
        sidecar = nil
        sidecarProfileName = nil
        sidecarConfigurationSignature = nil
        gatewayReady = false
        isStreaming = false
        if clearSession {
            activeSessionID = nil
            activeSessionKey = nil
            activeSessionTitle = nil
        }
    }

    private func applyResumedSession(
        _ result: JSONValue,
        savedSessionID: String,
        title: String?,
        preserveVisibleTranscript: Bool
    ) throws {
        guard let object = result.objectValue,
              let sessionID = object["session_id"]?.stringValue
        else { throw HermesGatewayClientError.malformedResponse }
        activeSessionID = sessionID
        activeSessionKey = object["resumed"]?.stringValue ?? savedSessionID
        activeSessionTitle = title
        if !preserveVisibleTranscript || messages.isEmpty {
            messages = Self.parseMessages(object["messages"])
        }
        toolTraces = []
        connectionState = .connected
    }

    private func liveSessionKey(for sessionID: String, operation: GatewayOperation) async throws -> String? {
        let result = try await rpc(
            operation,
            method: "session.active_list",
            params: ["current_session_id": .string(sessionID)]
        )
        guard let rows = result.objectValue?["sessions"]?.arrayValue else { return nil }
        for row in rows {
            guard let object = row.objectValue,
                  object["id"]?.stringValue == sessionID
            else { continue }
            return object["session_key"]?.stringValue
        }
        return nil
    }

    private func handle(_ event: HermesGatewayEvent) {
        if event.type == "gateway.ready" {
            return
        }
        if let activeSessionID,
           let eventSessionID = event.sessionID,
           eventSessionID != activeSessionID {
            return
        }

        switch event.type {
        case "message.start":
            if messages.last?.role != .assistant || messages.last?.isStreaming == false {
                messages.append(
                    HermesTranscriptMessage(
                        role: .assistant,
                        text: "",
                        isStreaming: true
                    )
                )
            }
            isStreaming = true
        case "message.delta":
            appendAssistantDelta(event.payload["text"]?.stringValue ?? "")
        case "message.complete":
            completeAssistantMessage(
                text: event.payload["text"]?.stringValue,
                reasoning: event.payload["reasoning"]?.stringValue
            )
        case "tool.start":
            toolTraces.append(
                HermesToolTrace(
                    name: Self.toolName(from: event.payload),
                    status: .running,
                    detail: Self.toolDetail(from: event.payload)
                )
            )
        case "tool.progress":
            updateLastTool(
                name: Self.toolName(from: event.payload),
                status: .running,
                detail: event.payload["preview"]?.stringValue ?? Self.toolDetail(from: event.payload)
            )
        case "tool.complete":
            updateLastTool(
                name: Self.toolName(from: event.payload),
                status: .complete,
                detail: Self.toolDetail(from: event.payload)
            )
        case "approval.request":
            toolTraces.append(
                HermesToolTrace(
                    name: "Approval",
                    status: .approval,
                    detail: "Auto-approved by MTPLX Hermes mode."
                )
            )
            if let sessionID = activeSessionID,
               let profile = selectedProfile,
               let operation = currentOperation(for: profile) {
                Task { [weak self] in
                    guard let self, self.isCurrent(operation, sessionID: sessionID) else { return }
                    _ = try? await self.rpc(
                        operation,
                        method: "approval.respond",
                        params: [
                            "session_id": .string(sessionID),
                            "choice": .string("allow"),
                            "all": .bool(true),
                        ]
                    )
                }
            }
        case "clarify.request", "sudo.request", "secret.request":
            toolTraces.append(
                HermesToolTrace(
                    name: event.type.replacingOccurrences(of: ".request", with: ""),
                    status: .waiting,
                    detail: Self.toolDetail(from: event.payload)
                )
            )
        case "error":
            messages.append(
                HermesTranscriptMessage(
                    role: .system,
                    text: event.payload["message"]?.stringValue ?? "Hermes reported an error."
                )
            )
            isStreaming = false
        case "session.info":
            if let key = event.payload["session_key"]?.stringValue {
                activeSessionKey = key
            }
        default:
            break
        }
    }

    private func appendAssistantDelta(_ delta: String) {
        guard !delta.isEmpty else { return }
        if messages.last?.role != .assistant || messages.last?.isStreaming == false {
            messages.append(
                HermesTranscriptMessage(role: .assistant, text: delta, isStreaming: true)
            )
        } else {
            messages[messages.count - 1].text += delta
        }
        isStreaming = true
    }

    private func completeAssistantMessage(text: String?, reasoning: String?) {
        if let reasoning, !reasoning.isEmpty {
            toolTraces.append(
                HermesToolTrace(name: "Thought", status: .complete, detail: reasoning)
            )
        }
        let finalText = text ?? ""
        if messages.last?.role == .assistant {
            messages[messages.count - 1].text = finalText.isEmpty
                ? messages[messages.count - 1].text
                : finalText
            messages[messages.count - 1].isStreaming = false
        } else if !finalText.isEmpty {
            messages.append(
                HermesTranscriptMessage(role: .assistant, text: finalText, isStreaming: false)
            )
        }
        isStreaming = false
        if let activeSessionID,
           let profile = selectedProfile,
           let operation = currentOperation(for: profile) {
            Task { [weak self] in
                guard let self else { return }
                let key = try? await self.liveSessionKey(for: activeSessionID, operation: operation)
                guard self.isCurrent(operation, sessionID: activeSessionID) else { return }
                self.activeSessionKey = key ?? self.activeSessionKey
            }
        }
    }

    private func updateLastTool(name: String, status: HermesToolStatus, detail: String) {
        if let idx = toolTraces.lastIndex(where: { $0.name == name && $0.status == .running }) {
            toolTraces[idx].status = status
            if !detail.isEmpty {
                toolTraces[idx].detail = detail
            }
        } else {
            toolTraces.append(HermesToolTrace(name: name, status: status, detail: detail))
        }
    }

    private static func parseSessions(_ value: JSONValue) -> [HermesSavedSession] {
        guard let rows = value.objectValue?["sessions"]?.arrayValue else { return [] }
        return rows.compactMap { row in
            guard let object = row.objectValue,
                  let id = object["id"]?.stringValue
            else { return nil }
            return HermesSavedSession(
                id: id,
                title: object["title"]?.stringValue ?? "",
                preview: object["preview"]?.stringValue ?? "",
                startedAt: object["started_at"]?.doubleValue ?? 0,
                messageCount: object["message_count"]?.intValue ?? 0,
                source: object["source"]?.stringValue ?? ""
            )
        }
    }

    private static func parseMessages(_ value: JSONValue?) -> [HermesTranscriptMessage] {
        guard let rows = value?.arrayValue else { return [] }
        return rows.compactMap { row in
            guard let object = row.objectValue else { return nil }
            let role = HermesMessageRole(rawValue: object["role"]?.stringValue ?? "") ?? .assistant
            let text = textFromMessageObject(object)
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return HermesTranscriptMessage(role: role, text: text)
        }
    }

    private static func textFromMessageObject(_ object: [String: JSONValue]) -> String {
        if let text = object["text"]?.stringValue {
            return text
        }
        if let content = object["content"] {
            return text(from: content)
        }
        return ""
    }

    private static func text(from value: JSONValue) -> String {
        switch value {
        case .string(let text):
            return text
        case .array(let array):
            return array.map(text(from:)).joined(separator: "\n")
        case .object(let object):
            if let text = object["text"]?.stringValue {
                return text
            }
            if let content = object["content"] {
                return text(from: content)
            }
            return ""
        default:
            return ""
        }
    }

    private static func toolName(from payload: [String: JSONValue]) -> String {
        payload["name"]?.stringValue
            ?? payload["tool"]?.stringValue
            ?? payload["command"]?.stringValue
            ?? "Tool"
    }

    private static func toolDetail(from payload: [String: JSONValue]) -> String {
        if let preview = payload["preview"]?.stringValue, !preview.isEmpty {
            return preview
        }
        if let message = payload["message"]?.stringValue, !message.isEmpty {
            return message
        }
        if let result = payload["result"]?.stringValue, !result.isEmpty {
            return result
        }
        if let args = payload["args"]?.objectValue {
            return args
                .map { "\($0.key)=\(Self.text(from: $0.value))" }
                .sorted()
                .joined(separator: " ")
        }
        return ""
    }

    private static func message(for error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription {
            return description
        }
        return String(describing: error)
    }

    private static func configurationSignature(_ configuration: MTPLXAppConfiguration) -> String {
        [
            configuration.host,
            String(configuration.port),
            configuration.model,
            configuration.apiKey ?? "",
        ].joined(separator: "|")
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { value } else { nil }
    }
}
