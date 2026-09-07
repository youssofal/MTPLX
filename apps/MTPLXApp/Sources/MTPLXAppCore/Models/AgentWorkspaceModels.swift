import Foundation

public struct AgentModelInfo: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let path: String
    public let backend: String?
    public let displayName: String?
    public let contextWindow: Int?
    public let reasoning: Bool
    public let vision: Bool
    public let loaded: Bool

    public init(
        id: String,
        path: String,
        backend: String? = nil,
        displayName: String? = nil,
        contextWindow: Int? = nil,
        reasoning: Bool = false,
        vision: Bool = false,
        loaded: Bool = false
    ) {
        self.id = id
        self.path = path
        self.backend = backend
        self.displayName = displayName
        self.contextWindow = contextWindow
        self.reasoning = reasoning
        self.vision = vision
        self.loaded = loaded
    }

    private enum CodingKeys: String, CodingKey {
        case id, path, backend
        case displayName = "display_name"
        case contextWindow = "context_window"
        case reasoning, vision, loaded
    }
}

public struct AgentModelsPayload: Codable, Equatable, Sendable {
    public let provider: AgentProviderInfo
    public let models: [AgentModelInfo]
}

public struct AgentProviderInfo: Codable, Equatable, Sendable {
    public let id: String
    public let name: String
    public let kind: String
    public let sourceOfTruth: String

    private enum CodingKeys: String, CodingKey {
        case id, name, kind
        case sourceOfTruth = "source_of_truth"
    }
}

public struct AgentWorkspace: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public var name: String
    public var rootPath: String
    public let createdAt: Date
    public var updatedAt: Date
    public var agentProfile: String
    public var model: String?
    public var instructions: String
    public var toolPolicy: [String: String]

    private enum CodingKeys: String, CodingKey {
        case id, name
        case rootPath = "root_path"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case agentProfile = "agent_profile"
        case model, instructions
        case toolPolicy = "tool_policy"
    }
}

public struct AgentWorkspacesPayload: Codable, Equatable, Sendable {
    public let workspaces: [AgentWorkspace]
    public let status: DynamicObject
}

public struct AgentRun: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let workspaceID: String
    public let sessionID: String
    public var title: String
    public var status: String
    public let createdAt: Date
    public var updatedAt: Date
    public let model: String?
    public var eventCount: Int
    public var lastEventAt: Date?
    public var error: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case workspaceID = "workspace_id"
        case sessionID = "session_id"
        case title, status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case model
        case eventCount = "event_count"
        case lastEventAt = "last_event_at"
        case error
    }
}

public struct AgentRunsPayload: Codable, Equatable, Sendable {
    public let workspaceID: String
    public let runs: [AgentRun]

    private enum CodingKeys: String, CodingKey {
        case workspaceID = "workspace_id"
        case runs
    }
}

public struct AgentRunEvent: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let runID: String
    public let sequence: Int
    public let kind: String
    public let payload: DynamicObject
    public let createdAt: Date

    private enum CodingKeys: String, CodingKey {
        case id
        case runID = "run_id"
        case sequence, kind, payload
        case createdAt = "created_at"
    }
}

public struct AgentRunEventsPayload: Codable, Equatable, Sendable {
    public let runID: String
    public let events: [AgentRunEvent]

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case events
    }
}

public struct AgentRunSnapshot: Sendable {
    public let run: AgentRun
    public let events: [AgentRunEvent]

    public init(run: AgentRun, events: [AgentRunEvent]) {
        self.run = run
        self.events = events
    }
}

public struct AgentProfile: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let description: String
    public let permissions: [String]
    public let instructions: String?
    public let tokenBudget: Int?
    public let contextWindow: Int?
    public let model: String?
    public let builtIn: Bool?
    public let sha256: String?

    private enum CodingKeys: String, CodingKey {
        case id, name, description, permissions, instructions, model, sha256
        case tokenBudget = "token_budget"
        case contextWindow = "context_window"
        case builtIn = "built_in"
    }
}

public struct AgentProfilesPayload: Codable, Equatable, Sendable {
    public let profiles: [AgentProfile]
    public let delegationStatuses: [String]

    private enum CodingKeys: String, CodingKey {
        case profiles
        case delegationStatuses = "delegation_statuses"
    }
}

public struct AgentGraphNode: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let type: String
    public let name: String
    public let config: DynamicObject
    public let timeoutSeconds: Int?
    public let retry: DynamicObject
    public let approval: DynamicObject

    private enum CodingKeys: String, CodingKey {
        case id, type, name, config, retry, approval
        case timeoutSeconds = "timeout_seconds"
    }
}

public struct AgentGraphEdge: Codable, Equatable, Sendable {
    public let source: String
    public let target: String
    public let condition: DynamicObject?
}

public struct AgentGraphDefinition: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let projectID: String
    public let workspaceID: String
    public let name: String
    public let description: String
    public let schemaVersion: Int
    public let revision: Int
    public let inputs: DynamicObject
    public let outputs: DynamicObject
    public let nodes: [AgentGraphNode]
    public let edges: [AgentGraphEdge]
    public let limits: DynamicObject
    public let policies: [String: String]
    public let runtimeRequirements: DynamicObject
    public let retry: DynamicObject
    public let timeoutSeconds: Int
    public let approvalRequirements: DynamicObject
    public let createdAt: Date
    public let updatedAt: Date
    public let contentSHA256: String

    private enum CodingKeys: String, CodingKey {
        case id, name, description, revision, inputs, outputs, nodes, edges, limits
        case policies, retry
        case projectID = "project_id"
        case workspaceID = "workspace_id"
        case schemaVersion = "schema_version"
        case runtimeRequirements = "runtime_requirements"
        case timeoutSeconds = "timeout_seconds"
        case approvalRequirements = "approval_requirements"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case contentSHA256 = "content_sha256"
    }
}

public struct AgentGraphsPayload: Codable, Equatable, Sendable {
    public let graphs: [AgentGraphDefinition]
}

public struct AgentGraphRun: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let graphID: String
    public let graphRevision: Int
    public let graphSHA256: String
    public let workspaceID: String
    public let projectID: String
    public let workspaceRoot: String?
    public let status: String
    public let pinnedModel: String?
    public let runtimeProfile: String
    public let inputs: DynamicObject
    public let outputs: DynamicObject
    public let nodeStates: [String: DynamicObject]
    public let currentNodeID: String?
    public let pendingApprovalID: String?
    public let resourceMetrics: DynamicObject
    public let createdAt: Date
    public let updatedAt: Date
    public let stateVersion: Int
    public let pauseRequested: Bool
    public let error: String?

    private enum CodingKeys: String, CodingKey {
        case id, status, inputs, outputs, error
        case graphID = "graph_id"
        case graphRevision = "graph_revision"
        case graphSHA256 = "graph_sha256"
        case workspaceID = "workspace_id"
        case projectID = "project_id"
        case workspaceRoot = "workspace_root"
        case pinnedModel = "pinned_model"
        case runtimeProfile = "runtime_profile"
        case nodeStates = "node_states"
        case currentNodeID = "current_node_id"
        case pendingApprovalID = "pending_approval_id"
        case resourceMetrics = "resource_metrics"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case stateVersion = "state_version"
        case pauseRequested = "pause_requested"
    }
}

public struct AgentGraphRunsPayload: Codable, Equatable, Sendable {
    public let runs: [AgentGraphRun]
}

public struct AgentGraphRunRequest: Encodable, Sendable {
    public let graphID: String
    public let revision: Int?
    public let inputs: DynamicObject
    public let model: String?
    public let runtimeProfile: String
    public let runID: String?
    public let start: Bool

    public init(
        graphID: String,
        revision: Int? = nil,
        inputs: DynamicObject = DynamicObject(),
        model: String? = nil,
        runtimeProfile: String = "auto",
        runID: String? = nil,
        start: Bool = true
    ) {
        self.graphID = graphID
        self.revision = revision
        self.inputs = inputs
        self.model = model
        self.runtimeProfile = runtimeProfile
        self.runID = runID
        self.start = start
    }

    private enum CodingKeys: String, CodingKey {
        case graphID = "graph_id"
        case revision, inputs, model
        case runtimeProfile = "runtime_profile"
        case runID = "run_id"
        case start
    }
}

public struct AgentGraphRetryRequest: Encodable, Sendable {
    public let nodeID: String?
    public let allowSideEffectRetry: Bool
    public let forceNewSideEffect: Bool

    public init(
        nodeID: String? = nil,
        allowSideEffectRetry: Bool = false,
        forceNewSideEffect: Bool = false
    ) {
        self.nodeID = nodeID
        self.allowSideEffectRetry = allowSideEffectRetry
        self.forceNewSideEffect = forceNewSideEffect
    }

    private enum CodingKeys: String, CodingKey {
        case nodeID = "node_id"
        case allowSideEffectRetry = "allow_side_effect_retry"
        case forceNewSideEffect = "force_new_side_effect"
    }
}

public struct AgentGraphApprovalResponse: Codable, Equatable, Sendable {
    public let run: AgentGraphRun
    public let approval: AgentApproval
}

public struct AgentGraphApprovalsPayload: Codable, Equatable, Sendable {
    public let runID: String
    public let approvals: [AgentApproval]

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case approvals
    }
}

public struct AgentGraphApprovalRequest: Encodable, Sendable {
    public let approvalID: String
    public let decision: String
    public let resolvedBy: String
    public let reason: String?
    public let resume: Bool

    public init(
        approvalID: String,
        decision: String,
        resolvedBy: String = "desktop",
        reason: String? = nil,
        resume: Bool = true
    ) {
        self.approvalID = approvalID
        self.decision = decision
        self.resolvedBy = resolvedBy
        self.reason = reason
        self.resume = resume
    }

    private enum CodingKeys: String, CodingKey {
        case approvalID = "approval_id"
        case decision
        case resolvedBy = "resolved_by"
        case reason, resume
    }
}

public struct AgentDelegation: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let workspaceID: String
    public let parentRunID: String?
    public let childRunID: String
    public let role: String
    public let permissions: [String]
    public let prompt: String
    public let model: String?
    public let budget: Int
    public let contextWindow: Int?
    public let profileSHA256: String?
    public var status: String
    public let createdAt: Date
    public var updatedAt: Date
    public let worktreePath: String?
    public let worktreeCommit: String?
    public let sourceDelegationID: String?
    public let tokensUsed: Int?
    public let attempts: Int?
    public let evidence: DynamicObject?
    public let error: String?

    public var remainingTokenBudget: Int {
        max(0, budget - (tokensUsed ?? 0))
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case workspaceID = "workspace_id"
        case parentRunID = "parent_run_id"
        case childRunID = "child_run_id"
        case role, permissions, prompt, model, budget, status
        case contextWindow = "context_window"
        case profileSHA256 = "profile_sha256"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case worktreePath = "worktree_path"
        case worktreeCommit = "worktree_commit"
        case sourceDelegationID = "source_delegation_id"
        case tokensUsed = "tokens_used"
        case attempts
        case evidence, error
    }
}

public struct AgentDelegationsPayload: Codable, Equatable, Sendable {
    public let workspaceID: String
    public let delegations: [AgentDelegation]

    private enum CodingKeys: String, CodingKey {
        case workspaceID = "workspace_id"
        case delegations
    }
}

public struct AgentDelegationRequest: Encodable, Sendable {
    public let role: String
    public let prompt: String
    public let parentRunID: String?
    public let model: String?
    public let budget: Int?
    public let contextWindow: Int?
    public let sourceDelegationID: String?
    public let start: Bool

    public init(
        role: String = "reviewer",
        prompt: String = "",
        parentRunID: String? = nil,
        model: String? = nil,
        budget: Int? = nil,
        contextWindow: Int? = nil,
        sourceDelegationID: String? = nil,
        start: Bool = true
    ) {
        self.role = role
        self.prompt = prompt
        self.parentRunID = parentRunID
        self.model = model
        self.budget = budget
        self.contextWindow = contextWindow
        self.sourceDelegationID = sourceDelegationID
        self.start = start
    }

    private enum CodingKeys: String, CodingKey {
        case role, prompt
        case parentRunID = "parent_run_id"
        case model, budget, start
        case contextWindow = "context_window"
        case sourceDelegationID = "source_delegation_id"
    }
}

public struct AgentApproval: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let workspaceID: String
    public let runID: String?
    public let tool: String
    public let action: String
    public let description: String
    public let target: String?
    public let risk: String
    public var status: String
    public let createdAt: Date
    public let arguments: DynamicObject?
    public let argumentsSHA256: String?
    public let expiresAt: Date?
    public var resolvedAt: Date?
    public var resolvedBy: String?
    public var reason: String?
    public let consumedAt: Date?
    public let consumedBy: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case workspaceID = "workspace_id"
        case runID = "run_id"
        case tool, action, description, target, risk, status
        case createdAt = "created_at"
        case arguments
        case argumentsSHA256 = "arguments_sha256"
        case expiresAt = "expires_at"
        case resolvedAt = "resolved_at"
        case resolvedBy = "resolved_by"
        case reason
        case consumedAt = "consumed_at"
        case consumedBy = "consumed_by"
    }
}

public struct AgentApprovalsPayload: Codable, Equatable, Sendable {
    public let workspaceID: String
    public let approvals: [AgentApproval]

    private enum CodingKeys: String, CodingKey {
        case workspaceID = "workspace_id"
        case approvals
    }
}

public struct MemoryContextPayload: Codable, Equatable, Sendable {
    public let query: String
    public let context: String
    public let hits: [DynamicObject]
    public let truncated: Bool
}

public struct AgentWorkspaceCreateRequest: Encodable, Sendable {
    public let name: String
    public let rootPath: String
    public let workspaceID: String?
    public let agentProfile: String
    public let model: String?
    public let instructions: String
    public let toolPolicy: [String: String]

    public init(
        name: String,
        rootPath: String,
        workspaceID: String? = nil,
        agentProfile: String = "mtplx-agent",
        model: String? = nil,
        instructions: String = "",
        toolPolicy: [String: String] = [:]
    ) {
        self.name = name
        self.rootPath = rootPath
        self.workspaceID = workspaceID
        self.agentProfile = agentProfile
        self.model = model
        self.instructions = instructions
        self.toolPolicy = toolPolicy
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case rootPath = "root_path"
        case workspaceID = "workspace_id"
        case agentProfile = "agent_profile"
        case model, instructions
        case toolPolicy = "tool_policy"
    }
}

public struct AgentWorkspaceRunRequest: Encodable, Sendable {
    public let sessionID: String?
    public let title: String
    public let model: String?
    public let runID: String?

    public init(
        sessionID: String? = nil,
        title: String = "Agent run",
        model: String? = nil,
        runID: String? = nil
    ) {
        self.sessionID = sessionID
        self.title = title
        self.model = model
        self.runID = runID
    }

    private enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case title, model
        case runID = "run_id"
    }
}

public struct AgentApprovalRequest: Encodable, Sendable {
    public let runID: String?
    public let tool: String
    public let action: String
    public let description: String
    public let target: String?
    public let risk: String
    public let arguments: DynamicObject
    public let expiresInSeconds: Int

    public init(
        runID: String? = nil,
        tool: String,
        action: String,
        description: String,
        target: String? = nil,
        risk: String = "medium",
        arguments: DynamicObject = DynamicObject(),
        expiresInSeconds: Int = 600
    ) {
        self.runID = runID
        self.tool = tool
        self.action = action
        self.description = description
        self.target = target
        self.risk = risk
        self.arguments = arguments
        self.expiresInSeconds = expiresInSeconds
    }

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case tool, action, description, target, risk, arguments
        case expiresInSeconds = "expires_in_seconds"
    }
}

public struct AgentWorkspaceToolRequest: Encodable, Sendable {
    public let runID: String?
    public let arguments: DynamicObject
    public let approvalID: String?
    public let executorID: String
    public let idempotencyKey: String?

    public init(
        runID: String? = nil,
        arguments: DynamicObject = DynamicObject(),
        approvalID: String? = nil,
        executorID: String = "desktop",
        idempotencyKey: String? = nil
    ) {
        self.runID = runID
        self.arguments = arguments
        self.approvalID = approvalID
        self.executorID = executorID
        self.idempotencyKey = idempotencyKey
    }

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case arguments
        case approvalID = "approval_id"
        case executorID = "executor_id"
        case idempotencyKey = "idempotency_key"
    }
}

public struct AgentWorkspaceToolResponse: Codable, Equatable, Sendable {
    public let ok: Bool
    public let status: String
    public let error: String?
    public let tool: String?
    public let argumentsSHA256: String?
    public let approvalID: String?
    public let approval: AgentApproval?
    public let preview: DynamicObject?
    public let result: DynamicObject?
    public let elapsedMS: Int?

    private enum CodingKeys: String, CodingKey {
        case ok, status, error, tool, approval, preview, result
        case argumentsSHA256 = "arguments_sha256"
        case approvalID = "approval_id"
        case elapsedMS = "elapsed_ms"
    }
}

public struct AgentExternalActionRequest: Encodable, Sendable {
    public let runID: String?
    public let tool: String
    public let arguments: DynamicObject
    public let approvalID: String?
    public let executorID: String

    public init(
        runID: String? = nil,
        tool: String,
        arguments: DynamicObject,
        approvalID: String? = nil,
        executorID: String = "desktop"
    ) {
        self.runID = runID
        self.tool = tool
        self.arguments = arguments
        self.approvalID = approvalID
        self.executorID = executorID
    }

    private enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case tool, arguments
        case approvalID = "approval_id"
        case executorID = "executor_id"
    }
}

public struct AgentRunUpdateRequest: Encodable, Sendable {
    public let title: String?
    public let status: String?
    public let error: String?

    public init(title: String? = nil, status: String? = nil, error: String? = nil) {
        self.title = title
        self.status = status
        self.error = error
    }
}

public struct AgentEventRequest: Encodable, Sendable {
    public let kind: String
    public let payload: DynamicObject

    public init(kind: String, payload: DynamicObject = DynamicObject()) {
        self.kind = kind
        self.payload = payload
    }
}

public struct AgentApprovalDecisionRequest: Encodable, Sendable {
    public let decision: String
    public let resolvedBy: String
    public let reason: String?

    public init(
        decision: String,
        resolvedBy: String = "user",
        reason: String? = nil
    ) {
        self.decision = decision
        self.resolvedBy = resolvedBy
        self.reason = reason
    }

    private enum CodingKeys: String, CodingKey {
        case decision
        case resolvedBy = "resolved_by"
        case reason
    }
}
