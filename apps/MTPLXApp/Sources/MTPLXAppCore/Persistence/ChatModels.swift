import Foundation
import SwiftData

// MARK: - MessageRole
//
// OpenAI/Anthropic-compatible role taxonomy as it actually appears on
// the daemon's `/v1/chat/completions` wire surface. Stored as a raw
// String so SwiftData schema stays primitive-only; the typed enum is a
// computed property on `ChatMessage`.

public enum MessageRole: String, Codable, Hashable, Sendable, CaseIterable {
    case user
    case assistant
    case tool
    /// System messages are not persisted today (the daemon owns the
    /// system prompt). Reserved here so the schema does not need a
    /// migration if a future feature surfaces them.
    case system
}

// MARK: - ChatConversation

/// One conversation == one SessionBank session on the daemon side.
/// `id` is sent as `X-MTPLX-Session-Id` on every request in this
/// conversation, so in-conversation prefix reuse stays warm in RAM.
@Model
public final class ChatConversation {
    /// Stable identity; persisted across app restarts and used as the
    /// session id for the daemon.
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var createdAt: Date
    /// Bumped whenever a message is added; the sidebar sorts by this
    /// so the most-recent conversation floats to the top.
    public var updatedAt: Date
    /// Whether the web-search toggle is on for this conversation. Lives
    /// on the conversation (not globally) so a user can leave search on
    /// for a research thread and off for everyday chat without paging
    /// through Settings.
    public var webSearchEnabled: Bool
    /// Optional local workspace selected for this conversation. The daemon
    /// stays the model authority; this id binds the chat to the user's
    /// project, run history, and approval policy.
    public var workspaceID: String?
    /// Model override for this conversation. Nil means use the loaded MTPLX
    /// runtime reported by the backend.
    public var modelOverride: String?
    /// Agent mode controls are conversation-scoped so a planning thread can
    /// coexist with ordinary chat without changing global runtime settings.
    // These defaults live on the stored properties, not only on init.
    // SwiftData needs schema defaults to backfill conversations created by
    // older MTPLX releases during lightweight migration.
    public var planModeEnabled: Bool = false
    public var reasoningEffortRaw: String = "auto"
    public var goalText: String?
    /// User feedback kept with this conversation. Feedback stays local until
    /// the user explicitly exports or shares the chat.
    public var feedbackNotes: String?
    /// Most recent durable agent run for this conversation. Keeping the id
    /// here lets /resume rehydrate the run timeline after app or daemon
    /// restarts without relying on in-memory view state.
    public var activeRunID: String?
    /// Optional persisted daemon session replacement. Cancelling or manually
    /// compacting a conversation rotates the session so a later app restart
    /// cannot accidentally reuse stale server-side prefix state.
    public var sessionIDOverride: UUID?

    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    public var messages: [ChatMessage]

    public init(
        id: UUID = UUID(),
        title: String = "New Chat",
        createdAt: Date = Date(),
        updatedAt: Date? = nil,
        webSearchEnabled: Bool = false,
        workspaceID: String? = nil,
        modelOverride: String? = nil,
        planModeEnabled: Bool = false,
        reasoningEffortRaw: String = "auto",
        goalText: String? = nil,
        feedbackNotes: String? = nil,
        activeRunID: String? = nil,
        sessionIDOverride: UUID? = nil,
        messages: [ChatMessage] = []
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
        self.webSearchEnabled = webSearchEnabled
        self.workspaceID = workspaceID
        self.modelOverride = modelOverride
        self.planModeEnabled = planModeEnabled
        self.reasoningEffortRaw = reasoningEffortRaw
        self.goalText = goalText
        self.feedbackNotes = feedbackNotes
        self.activeRunID = activeRunID
        self.sessionIDOverride = sessionIDOverride
        self.messages = messages
    }
}

// MARK: - ChatMessage

/// One turn in the conversation. Stores enough to round-trip the
/// rendered UI exactly (visible text + reasoning + tool calls + the
/// attachments the user added). Tool calls and per-turn stats are
/// stored as JSON blobs to avoid SwiftData relationship complexity for
/// short-lived debug fields.
@Model
public final class ChatMessage {
    @Attribute(.unique) public var id: UUID
    /// Stored as raw String because SwiftData rejects enum types in
    /// some Xcode 26 builds; the typed accessor below is the API.
    public var roleRaw: String
    public var visibleContent: String
    public var reasoningContent: String?
    /// Filled when role == .tool so the daemon can match this tool
    /// result back to the assistant turn that requested it.
    public var toolCallId: String?
    /// Filled when role == .assistant and the model emitted tool calls.
    /// Encoded as `[ToolCallRecord]` JSON; decoded only when needed.
    public var toolCallsJSON: String?
    /// JSON-encoded `[ChatTurnStats]` for the assistant turn (decode
    /// TPS, accepted/drafted, verify time). Optional; assistant turns
    /// before the stats wiring stayed nil.
    public var statsJSON: String?
    /// Finish state for assistant turns. Completed turns use the server's
    /// finish reason (`stop`, `length`, etc.); interrupted local turns use
    /// app reasons such as `cancelled` or `error`.
    public var finishReason: String?
    /// Groups every assistant/tool message persisted by ONE user turn's
    /// tool loop (think → search → think → answer) so the transcript can
    /// render the whole turn as a single surface: one thinking card, one
    /// activity chip, one answer, one sources footer. Nil on messages
    /// persisted before this field existed (and on user messages) — the
    /// renderer treats those as singleton groups, so old conversations
    /// keep their historical layout. Optional => SwiftData lightweight
    /// migration, no store version bump.
    public var turnGroupID: UUID?
    /// JSON-encoded `[SourceRecord]` — the deduped web sources gathered
    /// across ALL tool rounds of this turn. Persisted only on the FINAL
    /// assistant message of a group so completed turns re-render their
    /// sources footer without re-parsing tool-trace JSON.
    public var sourcesJSON: String?
    public var createdAt: Date
    /// Denormalized conversation identity for resilient transcript fetches.
    /// SwiftData relationship queries can lag after rapid inserts/reloads;
    /// this keeps the selected chat from rendering blank while the
    /// relationship graph catches up.
    public var conversationID: UUID?
    public var conversation: ChatConversation?

    @Relationship(deleteRule: .cascade, inverse: \ChatAttachment.message)
    public var attachments: [ChatAttachment]

    @Relationship(deleteRule: .cascade, inverse: \ToolTraceRecord.message)
    public var toolTraces: [ToolTraceRecord]

    public var role: MessageRole {
        get { MessageRole(rawValue: roleRaw) ?? .user }
        set { roleRaw = newValue.rawValue }
    }

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        visibleContent: String,
        reasoningContent: String? = nil,
        toolCallId: String? = nil,
        toolCallsJSON: String? = nil,
        statsJSON: String? = nil,
        finishReason: String? = nil,
        turnGroupID: UUID? = nil,
        sourcesJSON: String? = nil,
        createdAt: Date = Date(),
        conversation: ChatConversation? = nil,
        attachments: [ChatAttachment] = [],
        toolTraces: [ToolTraceRecord] = []
    ) {
        self.id = id
        self.roleRaw = role.rawValue
        self.visibleContent = visibleContent
        self.reasoningContent = reasoningContent
        self.toolCallId = toolCallId
        self.toolCallsJSON = toolCallsJSON
        self.statsJSON = statsJSON
        self.finishReason = finishReason
        self.turnGroupID = turnGroupID
        self.sourcesJSON = sourcesJSON
        self.createdAt = createdAt
        self.conversationID = conversation?.id
        self.conversation = conversation
        self.attachments = attachments
        self.toolTraces = toolTraces
    }
}

// MARK: - ChatAttachment

@Model
public final class ChatAttachment {
    @Attribute(.unique) public var id: UUID
    public var filename: String
    public var mimeType: String
    public var sizeBytes: Int
    /// Plain-text content extracted client-side by FileExtractor. Always
    /// present (failed extractions are not persisted — the user gets a
    /// red-dot composer chip and can send without the attachment).
    public var extractedText: String
    /// Raw encoded image bytes for vision attachments (PNG/JPEG/WebP,
    /// already downscaled client-side). Nil for text attachments.
    public var imageData: Data?
    public var createdAt: Date
    public var message: ChatMessage?

    public var isImage: Bool { imageData != nil }

    public init(
        id: UUID = UUID(),
        filename: String,
        mimeType: String,
        sizeBytes: Int,
        extractedText: String,
        imageData: Data? = nil,
        createdAt: Date = Date(),
        message: ChatMessage? = nil
    ) {
        self.id = id
        self.filename = filename
        self.mimeType = mimeType
        self.sizeBytes = sizeBytes
        self.extractedText = extractedText
        self.imageData = imageData
        self.createdAt = createdAt
        self.message = message
    }
}

// MARK: - ToolTraceRecord

/// One tool call's lifecycle, persisted alongside the assistant
/// message that triggered it. `name` is the OpenAI tool name (e.g.
/// `web_search`, `fetch_url`); arguments and result are JSON strings
/// so the trace surface can re-hydrate them on render without paying a
/// schema-migration cost when tool shapes change.
@Model
public final class ToolTraceRecord {
    @Attribute(.unique) public var id: UUID
    public var name: String
    public var statusRaw: String
    public var argumentsJSON: String?
    public var resultJSON: String?
    public var activityLog: [String]
    public var startedAt: Date
    public var completedAt: Date?
    public var message: ChatMessage?

    public var status: ToolTraceStatus {
        get { ToolTraceStatus(rawValue: statusRaw) ?? .pending }
        set { statusRaw = newValue.rawValue }
    }

    public init(
        id: UUID = UUID(),
        name: String,
        status: ToolTraceStatus = .pending,
        argumentsJSON: String? = nil,
        resultJSON: String? = nil,
        activityLog: [String] = [],
        startedAt: Date = Date(),
        completedAt: Date? = nil,
        message: ChatMessage? = nil
    ) {
        self.id = id
        self.name = name
        self.statusRaw = status.rawValue
        self.argumentsJSON = argumentsJSON
        self.resultJSON = resultJSON
        self.activityLog = activityLog
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.message = message
    }
}

public enum ToolTraceStatus: String, Codable, Sendable, CaseIterable {
    case pending
    case success
    case failed
}

// MARK: - Versioned schema
//
// V1 is exactly the four models above as shipped today. Every store on a
// user's Mac was created from these definitions (earlier field additions
// were all optional, so they went through lightweight migration), and the
// versioned container opens those stores unchanged. The next change to a
// model that lightweight migration cannot absorb adds a V2 here and a
// stage to `ChatSchemaMigrationPlan` instead of failing every upgrading
// user's store open.

public enum ChatSchemaV1: VersionedSchema {
    public static let versionIdentifier = Schema.Version(1, 0, 0)

    public static var models: [any PersistentModel.Type] {
        [
            ChatConversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            ToolTraceRecord.self,
        ]
    }
}

public enum ChatSchemaMigrationPlan: SchemaMigrationPlan {
    public static var schemas: [any VersionedSchema.Type] {
        [ChatSchemaV1.self]
    }

    public static var stages: [MigrationStage] {
        []
    }
}

// MARK: - Decoded helpers

/// Decoded shape of one tool call as it appears on the assistant turn's
/// `tool_calls` array. Persisted as JSON inside `ChatMessage.toolCallsJSON`
/// so the schema stays primitive.
public struct ToolCallRecord: Codable, Hashable, Sendable {
    public var id: String
    public var name: String
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// Decoded shape of the assistant turn's per-request stats. Persisted as
/// JSON inside `ChatMessage.statsJSON` so we don't need a schema bump
/// every time a new metric is exposed in `/v1/chat/completions`'s
/// `mtplx_stats` envelope.
public struct ChatTurnStats: Codable, Hashable, Sendable {
    public var rawDecodeTokS: Double?
    public var displayDecodeTokS: Double?
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var ttftS: Double?
    public var acceptedByDepth: [Int]?
    public var draftedByDepth: [Int]?
    public var verifyCalls: Int?
    public var verifyTimeS: Double?
    /// Total wall time the turn spent in reasoning across ALL tool
    /// rounds (think → search → think → answer sums every think span).
    /// Drives the collapsed "Thought · 12.4s" chip. Optional so stats
    /// persisted before this field decode unchanged.
    public var thinkingTimeMs: Int?

    public init(
        rawDecodeTokS: Double? = nil,
        displayDecodeTokS: Double? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        ttftS: Double? = nil,
        acceptedByDepth: [Int]? = nil,
        draftedByDepth: [Int]? = nil,
        verifyCalls: Int? = nil,
        verifyTimeS: Double? = nil,
        thinkingTimeMs: Int? = nil
    ) {
        self.rawDecodeTokS = rawDecodeTokS
        self.displayDecodeTokS = displayDecodeTokS
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.ttftS = ttftS
        self.acceptedByDepth = acceptedByDepth
        self.draftedByDepth = draftedByDepth
        self.verifyCalls = verifyCalls
        self.verifyTimeS = verifyTimeS
        self.thinkingTimeMs = thinkingTimeMs
    }
}

// MARK: - SourceRecord
//
// One web source the assistant consulted during a turn (a web_search
// result it was shown, or a page it fetched). The chat renders these as
// a single compact "Sources" footer under the final answer instead of
// spraying per-tool result cards through the transcript.

public struct SourceRecord: Codable, Hashable, Sendable, Identifiable {
    public var url: String
    public var title: String
    /// Registrable host for the pill label ("anthropic.com"). Computed
    /// once at extraction so rendering never re-parses URLs.
    public var domain: String

    public var id: String { url }

    public init(url: String, title: String, domain: String) {
        self.url = url
        self.title = title
        self.domain = domain
    }

    public init?(url: String, title: String?) {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let parsed = URL(string: trimmed), parsed.host != nil else {
            return nil
        }
        self.url = trimmed
        self.title = (title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        self.domain = Self.displayDomain(for: parsed)
    }

    private static func displayDomain(for url: URL) -> String {
        guard var host = url.host?.lowercased() else { return url.absoluteString }
        if host.hasPrefix("www.") {
            host = String(host.dropFirst(4))
        }
        return host
    }

    /// Pull sources out of one completed tool call. Tolerant of missing
    /// fields — tool result shapes drift and a sources footer that
    /// silently shows fewer pills beats a decode crash.
    public static func extract(
        toolName: String,
        argumentsJSON: String?,
        resultJSON: String?
    ) -> [SourceRecord] {
        switch toolName {
        case "web_search":
            guard let json = resultJSON,
                let data = json.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let results = dict["results"] as? [[String: Any]]
            else { return [] }
            return results.compactMap { entry in
                guard let url = entry["url"] as? String else { return nil }
                return SourceRecord(url: url, title: entry["title"] as? String)
            }
        case "fetch_url":
            var url: String?
            if let json = argumentsJSON,
                let data = json.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                url = dict["url"] as? String
            }
            var title: String?
            if let json = resultJSON,
                let data = json.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                title = dict["title"] as? String
                if url == nil { url = dict["url"] as? String }
            }
            guard let url, let record = SourceRecord(url: url, title: title) else { return [] }
            return [record]
        default:
            return []
        }
    }

    /// Order-preserving dedupe by normalized URL (scheme/case/trailing
    /// slash insensitive), keeping the first-seen title unless a later
    /// duplicate has one and the kept record does not.
    public static func dedupe(_ records: [SourceRecord]) -> [SourceRecord] {
        var seenIndex: [String: Int] = [:]
        var output: [SourceRecord] = []
        for record in records {
            let key = normalizedKey(record.url)
            if let existing = seenIndex[key] {
                if output[existing].title.isEmpty, !record.title.isEmpty {
                    output[existing].title = record.title
                }
                continue
            }
            seenIndex[key] = output.count
            output.append(record)
        }
        return output
    }

    private static func normalizedKey(_ url: String) -> String {
        var key = url.lowercased()
        for prefix in ["https://", "http://"] where key.hasPrefix(prefix) {
            key = String(key.dropFirst(prefix.count))
        }
        if key.hasPrefix("www.") {
            key = String(key.dropFirst(4))
        }
        while key.hasSuffix("/") {
            key = String(key.dropLast())
        }
        return key
    }

    public static func encodeJSON(_ records: [SourceRecord]) -> String? {
        guard !records.isEmpty,
            let data = try? JSONEncoder().encode(records),
            let json = String(data: data, encoding: .utf8)
        else { return nil }
        return json
    }

    public static func decodeJSON(_ json: String?) -> [SourceRecord] {
        guard let json,
            let data = json.data(using: .utf8),
            let records = try? JSONDecoder().decode([SourceRecord].self, from: data)
        else { return [] }
        return records
    }
}
