import Foundation

public enum MTPLXAPIClientError: Error, Equatable {
    case invalidResponse
    case httpStatus(Int, String)
}

public struct MTPLXAPIClient: Sendable {
    public var baseURL: URL
    public var apiKey: String?
    public var session: URLSession
    public var decoder: JSONDecoder
    public var encoder: JSONEncoder

    public init(
        baseURL: URL,
        apiKey: String? = nil,
        session: URLSession = .shared,
        decoder: JSONDecoder = MTPLXAPIClient.makeDefaultDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
        self.decoder = decoder
        self.encoder = encoder
    }

    public func health() async throws -> HealthPayload {
        try await get("/health")
    }

    /// Liveness truth for the daemon watchdog.
    ///
    /// `healthy` carries the decoded payload. `aliveUndecodable` means the
    /// daemon answered 2xx but the payload failed the app's Codable schema —
    /// the process is provably alive, so the watchdog must NOT count it as
    /// a miss (2026-07-06 incident: an idle-postcommit grew `/health` by one
    /// bank entry, the decode threw, two `try?`-swallowed probes counted as
    /// misses, and the app reaped a daemon that was answering 200 in 14 ms).
    /// `unreachable` is a transport failure, non-2xx, or deadline timeout —
    /// the only states that may count toward reaping.
    public enum LivenessProbeResult: Sendable {
        case healthy(HealthPayload)
        case aliveUndecodable(String)
        /// The daemon answered 401/403 — provably alive, credentials wrong.
        /// An API-key mismatch is an app configuration bug and never
        /// grounds to reap a serving process (liveness = transport truth).
        case aliveUnauthorized
        case unreachable
    }

    /// Probe `/health` against a hard deadline.
    ///
    /// A wedged daemon can hold an accepted connection open without
    /// ever answering, which surfaces as a request that neither
    /// completes nor fails — the watchdog would never count a miss
    /// (QA-114's stale-green hole). Racing the probe against a hard
    /// deadline turns "no answer in time" into a definite miss; the
    /// losing side is cancelled.
    public func livenessWithinDeadline(seconds: TimeInterval) async -> LivenessProbeResult {
        await withTaskGroup(of: LivenessProbeResult?.self) { group in
            group.addTask {
                do {
                    return .healthy(try await self.health())
                } catch let error as DecodingError {
                    // 2xx arrived and the body was read; only the schema
                    // mapping failed. The daemon is alive.
                    return .aliveUndecodable(String(describing: error))
                } catch MTPLXAPIClientError.httpStatus(401, _),
                        MTPLXAPIClientError.httpStatus(403, _) {
                    // An auth rejection is a live daemon speaking HTTP.
                    return .aliveUnauthorized
                } catch {
                    // Transport failures, timeouts, other non-2xx, non-HTTP.
                    return .unreachable
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
                return nil
            }
            let winner = await group.next() ?? nil
            group.cancelAll()
            return winner ?? .unreachable
        }
    }

    /// Back-compat shim for callers that only need the payload.
    public func healthWithinDeadline(seconds: TimeInterval) async -> HealthPayload? {
        if case .healthy(let payload) = await livenessWithinDeadline(seconds: seconds) {
            return payload
        }
        return nil
    }

    /// Client for the daemon watchdog's liveness probes.
    ///
    /// Probes must fail independently of everything else the app has
    /// in flight: the shared session's connection pool can be
    /// saturated — or wedged — by streams and long requests, leaving a
    /// probe queued with no error and no timeout. A dedicated
    /// single-connection ephemeral session with tight timeouts keeps
    /// probe latency a readout of daemon health rather than of
    /// client-side pool contention.
    public static func livenessProbe(baseURL: URL, apiKey: String?) -> MTPLXAPIClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        configuration.timeoutIntervalForResource = 10
        configuration.httpMaximumConnectionsPerHost = 1
        configuration.waitsForConnectivity = false
        return MTPLXAPIClient(
            baseURL: baseURL,
            apiKey: apiKey,
            session: URLSession(configuration: configuration)
        )
    }

    /// One-time browser sign-in ticket minted by the daemon. Opening
    /// `url` sets the browser-auth cookie without the API key ever being
    /// part of a URL, so the key stays out of browser history and logs.
    public struct BrowserAuthTicket: Decodable, Equatable, Sendable {
        public let url: URL
        public let expiresIn: Int?

        enum CodingKeys: String, CodingKey {
            case url
            case expiresIn = "expires_in"
        }
    }

    /// `POST /mtplx/browser-auth/ticket` with `{"next": next}`, Bearer
    /// authenticated. Throws on any transport failure or non-2xx answer,
    /// including the 404 an older daemon returns for the route.
    public func browserAuthTicket(next: String) async throws -> BrowserAuthTicket {
        try await post("/mtplx/browser-auth/ticket", body: ["next": next])
    }

    /// Client for the one-shot browser hand-off: an ephemeral session
    /// whose request and resource timeouts are both `timeout` seconds, so
    /// a daemon that is slow or gone cannot hold the Open Dashboard button
    /// for longer than that before the caller falls back.
    public static func browserAuthClient(
        baseURL: URL,
        apiKey: String?,
        timeout: TimeInterval = 3
    ) -> MTPLXAPIClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.waitsForConnectivity = false
        return MTPLXAPIClient(
            baseURL: baseURL,
            apiKey: apiKey,
            session: URLSession(configuration: configuration)
        )
    }

    public func capabilities() async throws -> AppCapabilities {
        try await get("/v1/mtplx/app/capabilities")
    }

    public func agentModels() async throws -> AgentModelsPayload {
        try await get("/v1/mtplx/agent/models")
    }

    public func agentProfiles() async throws -> AgentProfilesPayload {
        try await get("/v1/mtplx/agent/profiles")
    }

    public func memoryContext(
        query: String,
        scope: String = "all",
        agentID: String? = nil,
        sessionID: String? = nil,
        limit: Int = 8,
        maxChars: Int = 12000
    ) async throws -> MemoryContextPayload {
        var items = [
            URLQueryItem(name: "query", value: query),
            URLQueryItem(name: "scope", value: scope),
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "max_chars", value: String(maxChars)),
        ]
        if let agentID {
            items.append(URLQueryItem(name: "agent_id", value: agentID))
        }
        if let sessionID {
            items.append(URLQueryItem(name: "session_id", value: sessionID))
        }
        return try await get("/v1/mtplx/memory/context", queryItems: items)
    }

    public func workspaces(limit: Int = 100) async throws -> AgentWorkspacesPayload {
        try await get(
            "/v1/mtplx/workspaces",
            queryItems: [URLQueryItem(name: "limit", value: String(limit))]
        )
    }

    public func createWorkspace(
        name: String,
        rootPath: String,
        workspaceID: String? = nil,
        agentProfile: String = "mtplx-agent",
        model: String? = nil,
        instructions: String = "",
        toolPolicy: [String: String] = [:]
    ) async throws -> AgentWorkspace {
        try await post(
            "/v1/mtplx/workspaces",
            body: AgentWorkspaceCreateRequest(
                name: name,
                rootPath: rootPath,
                workspaceID: workspaceID,
                agentProfile: agentProfile,
                model: model,
                instructions: instructions,
                toolPolicy: toolPolicy
            )
        )
    }

    public func runs(workspaceID: String, limit: Int = 100) async throws -> AgentRunsPayload {
        try await get(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/runs",
            queryItems: [URLQueryItem(name: "limit", value: String(limit))]
        )
    }

    public func run(runID: String) async throws -> AgentRun {
        try await get("/v1/mtplx/runs/\(runID.urlPathComponentEscaped)")
    }

    @discardableResult
    public func resumeRun(runID: String) async throws -> AgentRun {
        try await post(
            "/v1/mtplx/runs/\(runID.urlPathComponentEscaped)/resume",
            body: EmptyBody()
        )
    }

    public func runEvents(
        runID: String,
        limit: Int = 500,
        after: Int = 0
    ) async throws -> AgentRunEventsPayload {
        try await get(
            "/v1/mtplx/runs/\(runID.urlPathComponentEscaped)/events",
            queryItems: [
                URLQueryItem(name: "limit", value: String(limit)),
                URLQueryItem(name: "after", value: String(after))
            ]
        )
    }

    public func graphs(
        workspaceID: String? = nil,
        limit: Int = 100
    ) async throws -> AgentGraphsPayload {
        var query = [URLQueryItem(name: "limit", value: String(limit))]
        if let workspaceID {
            query.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        return try await get("/v1/mtplx/graphs", queryItems: query)
    }

    public func graphRuns(
        workspaceID: String? = nil,
        graphID: String? = nil,
        limit: Int = 100
    ) async throws -> AgentGraphRunsPayload {
        var query = [URLQueryItem(name: "limit", value: String(limit))]
        if let workspaceID {
            query.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        if let graphID {
            query.append(URLQueryItem(name: "graph_id", value: graphID))
        }
        return try await get("/v1/mtplx/graph-runs", queryItems: query)
    }

    public func graphRun(runID: String) async throws -> AgentGraphRun {
        try await get("/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)")
    }

    @discardableResult
    public func startGraphRun(
        graphID: String,
        revision: Int? = nil,
        inputs: DynamicObject = DynamicObject(),
        model: String? = nil,
        runtimeProfile: String = "auto",
        runID: String? = nil
    ) async throws -> AgentGraphRun {
        try await post(
            "/v1/mtplx/graph-runs",
            body: AgentGraphRunRequest(
                graphID: graphID,
                revision: revision,
                inputs: inputs,
                model: model,
                runtimeProfile: runtimeProfile,
                runID: runID
            )
        )
    }

    @discardableResult
    public func pauseGraphRun(runID: String) async throws -> AgentGraphRun {
        try await post(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/pause",
            body: EmptyBody()
        )
    }

    @discardableResult
    public func resumeGraphRun(runID: String) async throws -> AgentGraphRun {
        try await post(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/resume",
            body: EmptyBody()
        )
    }

    @discardableResult
    public func cancelGraphRun(runID: String) async throws -> AgentGraphRun {
        try await post(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/cancel",
            body: EmptyBody()
        )
    }

    @discardableResult
    public func retryGraphRun(
        runID: String,
        nodeID: String? = nil,
        allowSideEffectRetry: Bool = false,
        forceNewSideEffect: Bool = false
    ) async throws -> AgentGraphRun {
        try await post(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/retry",
            body: AgentGraphRetryRequest(
                nodeID: nodeID,
                allowSideEffectRetry: allowSideEffectRetry,
                forceNewSideEffect: forceNewSideEffect
            )
        )
    }

    public func graphRunEvents(
        runID: String,
        limit: Int = 500,
        after: Int = 0
    ) async throws -> AgentRunEventsPayload {
        try await get(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/events",
            queryItems: [
                URLQueryItem(name: "limit", value: String(limit)),
                URLQueryItem(name: "after", value: String(after))
            ]
        )
    }

    public func graphRunApprovals(
        runID: String,
        status: String? = "pending",
        limit: Int = 100
    ) async throws -> AgentGraphApprovalsPayload {
        var query = [URLQueryItem(name: "limit", value: String(limit))]
        if let status {
            query.append(URLQueryItem(name: "status", value: status))
        }
        return try await get(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/approvals",
            queryItems: query
        )
    }

    public func resolveGraphApproval(
        runID: String,
        approvalID: String,
        decision: String,
        reason: String? = nil,
        resolvedBy: String = "desktop"
    ) async throws -> AgentGraphApprovalResponse {
        try await post(
            "/v1/mtplx/graph-runs/\(runID.urlPathComponentEscaped)/approve",
            body: AgentGraphApprovalRequest(
                approvalID: approvalID,
                decision: decision,
                resolvedBy: resolvedBy,
                reason: reason
            )
        )
    }

    public func delegations(
        workspaceID: String,
        parentRunID: String? = nil,
        limit: Int = 100
    ) async throws -> AgentDelegationsPayload {
        var query = [URLQueryItem(name: "limit", value: String(limit))]
        if let parentRunID {
            query.append(URLQueryItem(name: "parent_run_id", value: parentRunID))
        }
        return try await get(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/delegations",
            queryItems: query
        )
    }

    public func delegateAgent(
        workspaceID: String,
        role: String = "reviewer",
        prompt: String = "",
        parentRunID: String? = nil,
        model: String? = nil,
        budget: Int? = nil,
        contextWindow: Int? = nil,
        sourceDelegationID: String? = nil
    ) async throws -> AgentDelegation {
        try await post(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/delegations",
            body: AgentDelegationRequest(
                role: role,
                prompt: prompt,
                parentRunID: parentRunID,
                model: model,
                budget: budget,
                contextWindow: contextWindow,
                sourceDelegationID: sourceDelegationID
            )
        )
    }

    public func delegation(delegationID: String) async throws -> AgentDelegation {
        try await get("/v1/mtplx/delegations/\(delegationID.urlPathComponentEscaped)")
    }

    public func cancelDelegation(delegationID: String) async throws -> AgentDelegation {
        try await post(
            "/v1/mtplx/delegations/\(delegationID.urlPathComponentEscaped)/cancel",
            body: EmptyBody()
        )
    }

    public func retryDelegation(delegationID: String) async throws -> AgentDelegation {
        try await post(
            "/v1/mtplx/delegations/\(delegationID.urlPathComponentEscaped)/retry",
            body: EmptyBody()
        )
    }

    public func delegationWorktree(delegationID: String) async throws -> DynamicObject {
        try await get(
            "/v1/mtplx/delegations/\(delegationID.urlPathComponentEscaped)/worktree"
        )
    }

    public func executeWorkspaceTool(
        workspaceID: String,
        name: String,
        runID: String? = nil,
        arguments: DynamicObject = DynamicObject(),
        approvalID: String? = nil,
        executorID: String = "desktop",
        idempotencyKey: String? = nil
    ) async throws -> AgentWorkspaceToolResponse {
        try await post(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/tools/\(name.urlPathComponentEscaped)",
            body: AgentWorkspaceToolRequest(
                runID: runID,
                arguments: arguments,
                approvalID: approvalID,
                executorID: executorID,
                idempotencyKey: idempotencyKey
            )
        )
    }

    public func authorizeExternalAction(
        workspaceID: String,
        tool: String,
        runID: String? = nil,
        arguments: DynamicObject,
        approvalID: String? = nil,
        executorID: String = "desktop"
    ) async throws -> AgentWorkspaceToolResponse {
        try await post(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/external-actions/authorize",
            body: AgentExternalActionRequest(
                runID: runID,
                tool: tool,
                arguments: arguments,
                approvalID: approvalID,
                executorID: executorID
            )
        )
    }

    public func createRun(
        workspaceID: String,
        sessionID: String? = nil,
        title: String = "Agent run",
        model: String? = nil,
        runID: String? = nil
    ) async throws -> AgentRun {
        try await post(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/runs",
            body: AgentWorkspaceRunRequest(
                sessionID: sessionID,
                title: title,
                model: model,
                runID: runID
            )
        )
    }

    public func appendRunEvent(
        runID: String,
        kind: String,
        payload: DynamicObject = DynamicObject()
    ) async throws -> AgentRunEvent {
        try await post(
            "/v1/mtplx/runs/\(runID.urlPathComponentEscaped)/events",
            body: AgentEventRequest(kind: kind, payload: payload)
        )
    }

    public func updateRun(
        runID: String,
        title: String? = nil,
        status: String? = nil,
        error: String? = nil
    ) async throws -> AgentRun {
        try await patch(
            "/v1/mtplx/runs/\(runID.urlPathComponentEscaped)",
            body: AgentRunUpdateRequest(title: title, status: status, error: error)
        )
    }

    public func approvals(
        workspaceID: String,
        runID: String? = nil,
        status: String? = "pending",
        limit: Int = 100
    ) async throws -> AgentApprovalsPayload {
        var items = [URLQueryItem(name: "limit", value: String(limit))]
        if let runID {
            items.append(URLQueryItem(name: "run_id", value: runID))
        }
        if let status {
            items.append(URLQueryItem(name: "status", value: status))
        }
        return try await get(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/approvals",
            queryItems: items
        )
    }

    public func createApproval(
        workspaceID: String,
        runID: String? = nil,
        tool: String,
        action: String,
        description: String,
        target: String? = nil,
        risk: String = "medium",
        arguments: DynamicObject = DynamicObject()
    ) async throws -> AgentApproval {
        try await post(
            "/v1/mtplx/workspaces/\(workspaceID.urlPathComponentEscaped)/approvals",
            body: AgentApprovalRequest(
                runID: runID,
                tool: tool,
                action: action,
                description: description,
                target: target,
                risk: risk,
                arguments: arguments,
                expiresInSeconds: 600
            )
        )
    }

    public func resolveApproval(
        approvalID: String,
        decision: String,
        reason: String? = nil,
        resolvedBy: String = "user"
    ) async throws -> AgentApproval {
        try await post(
            "/v1/mtplx/approvals/\(approvalID.urlPathComponentEscaped)",
            body: AgentApprovalDecisionRequest(
                decision: decision,
                resolvedBy: resolvedBy,
                reason: reason
            )
        )
    }

    public func snapshot() async throws -> DashboardSnapshot {
        try await get("/v1/mtplx/snapshot")
    }

    public func sessions() async throws -> SessionsPayload {
        try await get("/admin/sessions")
    }

    public func prefillHistory() async throws -> PrefillHistoryPayload {
        try await get("/v1/mtplx/prefill_history")
    }

    public func settings() async throws -> MutableSettings {
        try await get("/v1/mtplx/settings")
    }

    public func updateSettings(_ settings: MutableSettings) async throws -> MutableSettings {
        try await post("/v1/mtplx/settings", body: settings)
    }

    public func cancel(requestId: String) async throws -> DynamicObject {
        try await post("/v1/mtplx/cancel/\(requestId.urlPathComponentEscaped)", body: EmptyBody())
    }

    public func clearSession(sessionId: String) async throws -> DynamicObject {
        try await post("/admin/sessions/\(sessionId.urlPathComponentEscaped)/clear", body: EmptyBody())
    }

    public func clearCache() async throws -> DynamicObject {
        try await post("/admin/cache/clear", body: EmptyBody())
    }

    public func setFanMode(
        _ mode: String,
        requireActualRamp: Bool = false,
        timeoutS: Double? = nil
    ) async throws -> FanModeResponse {
        try await post(
            "/v1/mtplx/thermal/fan_mode",
            body: FanModeRequest(
                mode: mode,
                requireActualRamp: requireActualRamp,
                timeoutS: timeoutS
            )
        )
    }

    public func thermalStatus() async throws -> DynamicObject {
        try await get("/v1/mtplx/thermal/status")
    }

    public func models() async throws -> ModelsResponse {
        try await get("/v1/models")
    }

    public func metricsStreamURL(snapshotIntervalMs: Int? = nil) -> URL {
        var components = URLComponents(url: makeURL("/v1/mtplx/metrics/stream"), resolvingAgainstBaseURL: false)!
        if let snapshotIntervalMs {
            components.queryItems = [
                URLQueryItem(name: "snapshot_interval_ms", value: String(snapshotIntervalMs))
            ]
        }
        return components.url!
    }

    // MARK: - AIME 2026 benchmark surface

    /// POST /v1/mtplx/benchmarks/aime/start.
    ///
    /// Sampler fields default to `nil` so the benchmark inherits the
    /// daemon's live app settings. Pass explicit values only for a
    /// deliberate sampler ablation.
    public func aimeStart(
        year: Int = 2026,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        maxTokens: Int? = nil,
        enableThinking: Bool? = nil,
        questionProcessIsolation: String? = nil,
        questionLimit: Int? = nil
    ) async throws -> BenchStartResponse {
        try await post(
            "/v1/mtplx/benchmarks/aime/start",
            body: _AIMEStartBody(
                year: year,
                temperature: temperature,
                topP: topP,
                topK: topK,
                maxTokens: maxTokens,
                enableThinking: enableThinking,
                questionProcessIsolation: questionProcessIsolation,
                questionLimit: questionLimit
            )
        )
    }

    /// GET /v1/mtplx/benchmarks/aime/active
    public func aimeActive() async throws -> BenchActiveResponse {
        try await get("/v1/mtplx/benchmarks/aime/active")
    }

    /// GET /v1/mtplx/benchmarks/aime/history?limit=N
    public func aimeHistory(limit: Int = 5) async throws -> BenchHistoryResponse {
        var components = URLComponents(
            url: makeURL("/v1/mtplx/benchmarks/aime/history"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyAuth(to: &request)
        return try await send(request)
    }

    /// GET /v1/mtplx/benchmarks/aime/{run_id}
    public func aimeSnapshot(runId: String) async throws -> BenchSnapshotResponse {
        try await get(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)"
        )
    }

    /// POST /v1/mtplx/benchmarks/aime/{run_id}/pause
    @discardableResult
    public func aimePause(runId: String) async throws -> BenchSnapshotResponse {
        try await post(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)/pause",
            body: EmptyBody()
        )
    }

    /// POST /v1/mtplx/benchmarks/aime/{run_id}/resume
    @discardableResult
    public func aimeResume(runId: String) async throws -> BenchSnapshotResponse {
        try await post(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)/resume",
            body: EmptyBody()
        )
    }

    /// POST /v1/mtplx/benchmarks/aime/{run_id}/skip
    @discardableResult
    public func aimeSkip(runId: String) async throws -> BenchSnapshotResponse {
        try await post(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)/skip",
            body: EmptyBody()
        )
    }

    /// POST /v1/mtplx/benchmarks/aime/{run_id}/cancel
    @discardableResult
    public func aimeCancel(runId: String) async throws -> BenchSnapshotResponse {
        try await post(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)/cancel",
            body: EmptyBody()
        )
    }

    /// URL for the SSE stream endpoint.
    public func aimeStreamURL(runId: String) -> URL {
        makeURL(
            "/v1/mtplx/benchmarks/aime/\(runId.urlPathComponentEscaped)/stream"
        )
    }

    /// Authorization header value (or nil) - exposed so the
    /// `BenchmarkStreamClient` can stamp the same Bearer token on its
    /// `URLRequest`.
    public var authorizationHeader: String? {
        guard let apiKey, !apiKey.isEmpty else { return nil }
        return "Bearer \(apiKey)"
    }

    /// Decoder shared by REST clients that read daemon JSON. The daemon
    /// usually emits unix seconds for dashboard timestamps, but newer
    /// benchmark endpoints emit Python ISO-8601 strings such as
    /// `2026-05-26T20:59:57.123456Z`. Accept both so feature endpoints do
    /// not need ad hoc model-specific date shims.
    public static func makeDefaultDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            if let seconds = try? container.decode(Double.self) {
                return Date(timeIntervalSince1970: seconds)
            }
            let raw = try container.decode(String.self)
            if let date = parseDaemonDate(raw) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported daemon date: \(raw)"
            )
        }
        return decoder
    }

    private static func parseDaemonDate(_ raw: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: raw) {
            return date
        }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        if let date = plain.date(from: raw) {
            return date
        }
        if let seconds = Double(raw) {
            return Date(timeIntervalSince1970: seconds)
        }
        return nil
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        var request = URLRequest(url: makeURL(path))
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyAuth(to: &request)
        return try await send(request)
    }

    private func get<T: Decodable>(
        _ path: String,
        queryItems: [URLQueryItem]
    ) async throws -> T {
        var components = URLComponents(
            url: makeURL(path),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = queryItems
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        applyAuth(to: &request)
        return try await send(request)
    }

    private func post<T: Decodable, Body: Encodable>(_ path: String, body: Body) async throws -> T {
        var request = URLRequest(url: makeURL(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        applyAuth(to: &request)
        return try await send(request)
    }

    private func patch<T: Decodable, Body: Encodable>(_ path: String, body: Body) async throws -> T {
        var request = URLRequest(url: makeURL(path))
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        applyAuth(to: &request)
        return try await send(request)
    }

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MTPLXAPIClientError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? ""
            throw MTPLXAPIClientError.httpStatus(http.statusCode, text)
        }
        return try decoder.decode(T.self, from: data)
    }

    private func makeURL(_ path: String) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        let basePath = components.percentEncodedPath
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let endpointPath = path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let joinedPath = [basePath, endpointPath]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        components.percentEncodedPath = joinedPath.isEmpty ? "/" : "/\(joinedPath)"
        components.query = nil
        components.fragment = nil
        return components.url!
    }

    private func applyAuth(to request: inout URLRequest) {
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
    }
}

private struct EmptyBody: Encodable {}

private struct _AIMEStartBody: Encodable {
    let year: Int
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let maxTokens: Int?
    let enableThinking: Bool?
    let questionProcessIsolation: String?
    let questionLimit: Int?

    enum CodingKeys: String, CodingKey {
        case year
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case maxTokens = "max_tokens"
        case enableThinking = "enable_thinking"
        case questionProcessIsolation = "question_process_isolation"
        case questionLimit = "question_limit"
    }
}

private extension String {
    var urlPathComponentEscaped: String {
        addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? self
    }
}
