import Foundation

// MARK: - Errors

public enum MTPLXChatClientError: Error, Equatable {
    case invalidResponse
    case httpStatus(Int, String)
    case unauthorized
    case daemonUnreachable
    case bodyEncodingFailed
}

// MARK: - Request shapes
//
// These are the wire-shape types the client serialises into the JSON
// body of `POST /v1/chat/completions`. They are intentionally separate
// from the SwiftData `ChatMessage` model so persistence schema changes
// do not couple to the daemon's request schema (and vice-versa).

public struct ChatRequestMessage: Codable, Hashable, Sendable {
    public var role: String
    public var content: String?
    public var name: String?
    /// Present on `role == "tool"` so the daemon matches the tool
    /// result back to the assistant's tool call id.
    public var toolCallId: String?
    /// Present on `role == "assistant"` when the model emitted tool
    /// calls in the previous turn. The viewmodel echoes these back so
    /// the daemon's SessionBank prefix-match stays tight.
    public var toolCalls: [ChatRequestToolCall]?
    /// data: URLs for attached images. When non-empty the message
    /// encodes as OpenAI content parts (images first, then the text)
    /// instead of a plain string.
    public var imageDataURLs: [String]?

    public init(
        role: String,
        content: String? = nil,
        name: String? = nil,
        toolCallId: String? = nil,
        toolCalls: [ChatRequestToolCall]? = nil,
        imageDataURLs: [String]? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.toolCallId = toolCallId
        self.toolCalls = toolCalls
        self.imageDataURLs = imageDataURLs
    }

    private enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCallId = "tool_call_id"
        case toolCalls = "tool_calls"
    }

    private struct ContentPart: Codable {
        struct ImageURL: Codable {
            var url: String
        }

        var type: String
        var text: String?
        var imageURL: ImageURL?

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageURL = "image_url"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        content = try? container.decodeIfPresent(String.self, forKey: .content)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        toolCalls = try container.decodeIfPresent(
            [ChatRequestToolCall].self, forKey: .toolCalls
        )
        imageDataURLs = nil
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(toolCallId, forKey: .toolCallId)
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        if let imageDataURLs, !imageDataURLs.isEmpty {
            var parts: [ContentPart] = imageDataURLs.map {
                ContentPart(type: "image_url", text: nil, imageURL: .init(url: $0))
            }
            let text = content ?? ""
            if !text.isEmpty {
                parts.append(ContentPart(type: "text", text: text, imageURL: nil))
            }
            try container.encode(parts, forKey: .content)
        } else {
            try container.encodeIfPresent(content, forKey: .content)
        }
    }
}

public struct ChatRequestToolCall: Codable, Hashable, Sendable {
    public var id: String
    public var type: String
    public var function: ChatRequestToolCallFunction

    public init(id: String, type: String = "function", function: ChatRequestToolCallFunction) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ChatRequestToolCallFunction: Codable, Hashable, Sendable {
    public var name: String
    /// Stringified JSON exactly as the daemon emitted it. Round-tripped
    /// verbatim so the daemon's prefix-cache stays aligned.
    public var arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

/// Mirrors the OpenAI tool definition body: `{type: "function",
/// function: {name, description, parameters}}`. `parameters` is left
/// as `JSONValue` so the chat tool factory can ship arbitrary schemas
/// without a Swift type per tool.
public struct ChatRequestTool: Codable, Sendable {
    public var type: String
    public var function: ChatRequestToolDefinition

    public init(type: String = "function", function: ChatRequestToolDefinition) {
        self.type = type
        self.function = function
    }
}

public struct ChatRequestToolDefinition: Codable, Sendable {
    public var name: String
    public var description: String
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// One full chat-completions request body. The viewmodel builds one of
/// these per turn (including the multi-round tool loop). Field omission
/// is meaningful: the app mutates daemon settings through MTPLX's live
/// settings endpoint, while chat requests only carry the conversation,
/// tools, and response limits needed for the current turn.
public struct ChatRequest: Encodable, Sendable {
    public var model: String?
    public var messages: [ChatRequestMessage]
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var presencePenalty: Double?
    public var stream: Bool
    public var tools: [ChatRequestTool]?
    public var toolChoice: String?
    public var enableThinking: Bool?
    public var generationMode: String?
    public var depth: Int?
    public var reasoningEffort: String?
    public var metadata: [String: String]?

    public init(
        model: String? = nil,
        messages: [ChatRequestMessage],
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        presencePenalty: Double? = nil,
        stream: Bool = true,
        tools: [ChatRequestTool]? = nil,
        toolChoice: String? = nil,
        enableThinking: Bool? = nil,
        generationMode: String? = nil,
        depth: Int? = nil,
        reasoningEffort: String? = nil,
        metadata: [String: String]? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.stream = stream
        self.tools = tools
        self.toolChoice = toolChoice
        self.enableThinking = enableThinking
        self.generationMode = generationMode
        self.depth = depth
        self.reasoningEffort = reasoningEffort
        self.metadata = metadata
    }

    private enum CodingKeys: String, CodingKey {
        case model, messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case presencePenalty = "presence_penalty"
        case stream, tools
        case toolChoice = "tool_choice"
        case enableThinking = "enable_thinking"
        case generationMode = "generation_mode"
        case depth
        case reasoningEffort = "reasoning_effort"
        case metadata
    }
}

// MARK: - Stream events

/// One observed event from the SSE stream. The client emits these as
/// they arrive; the viewmodel folds them into UI-bound state.
public enum ChatStreamEvent: Sendable {
    /// First chunk only. Carries the daemon-assigned request id used
    /// for cancellation.
    case requestId(String)
    case role(String)
    case reasoningDelta(String)
    case contentDelta(String)
    case toolCallStart(index: Int, id: String, name: String)
    case toolCallArgumentsDelta(index: Int, fragment: String)
    case progress(ChatProgressFrame)
    case finished(
        finishReason: String,
        usage: ChatUsage?,
        stats: ChatStreamStats?
    )
    /// The daemon failed the request mid-stream: a chunk with
    /// `finish_reason: "error"` and a top-level `error` object
    /// (memory guard, context overflow, tool-loop exception), followed
    /// by `[DONE]`. Carries the server's message verbatim. Emitted
    /// INSTEAD of `.finished`, so a failed reply is never mistaken for
    /// a completed one.
    case serverError(message: String)
}

public struct ChatProgressFrame: Sendable {
    public var completionTokens: Int?
    public var decodeTokS: Double?
    public var displayDecodeTokS: Double?
    public var phase: String?
    public var raw: DynamicObject

    public init(
        completionTokens: Int? = nil,
        decodeTokS: Double? = nil,
        displayDecodeTokS: Double? = nil,
        phase: String? = nil,
        raw: DynamicObject = DynamicObject()
    ) {
        self.completionTokens = completionTokens
        self.decodeTokS = decodeTokS
        self.displayDecodeTokS = displayDecodeTokS
        self.phase = phase
        self.raw = raw
    }
}

public struct ChatUsage: Sendable {
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var totalTokens: Int?
}

public struct ChatStreamStats: Sendable {
    public var rawDecodeTokS: Double?
    public var displayDecodeTokS: Double?
    public var ttftS: Double?
    public var requestElapsedS: Double?
    public var acceptedByDepth: [Int]?
    public var draftedByDepth: [Int]?
    public var verifyCalls: Int?
    public var verifyTimeS: Double?
    public var raw: DynamicObject

    public init(
        rawDecodeTokS: Double? = nil,
        displayDecodeTokS: Double? = nil,
        ttftS: Double? = nil,
        requestElapsedS: Double? = nil,
        acceptedByDepth: [Int]? = nil,
        draftedByDepth: [Int]? = nil,
        verifyCalls: Int? = nil,
        verifyTimeS: Double? = nil,
        raw: DynamicObject = DynamicObject()
    ) {
        self.rawDecodeTokS = rawDecodeTokS
        self.displayDecodeTokS = displayDecodeTokS
        self.ttftS = ttftS
        self.requestElapsedS = requestElapsedS
        self.acceptedByDepth = acceptedByDepth
        self.draftedByDepth = draftedByDepth
        self.verifyCalls = verifyCalls
        self.verifyTimeS = verifyTimeS
        self.raw = raw
    }
}

// MARK: - MTPLXChatClient
//
// Streams `POST /v1/chat/completions` against the local MTPLX daemon
// and emits typed events. Identifies as `X-MTPLX-Client: mtplx_app` and
// stamps `X-MTPLX-Session-Id` so each conversation gets its own
// SessionBank slot. Reuses the existing `MTPLXAPIClient` for base URL,
// auth, and cancel — never duplicates that wiring.

public struct MTPLXChatClient: Sendable {
    public var apiClient: MTPLXAPIClient
    public var session: URLSession
    public var encoder: JSONEncoder
    public var decoder: JSONDecoder

    public static let clientIdentifierHeader = "X-MTPLX-Client"
    public static let clientIdentifierValue = "mtplx_app"
    public static let sessionIdHeader = "X-MTPLX-Session-Id"

    public init(
        apiClient: MTPLXAPIClient,
        session: URLSession = .shared,
        encoder: JSONEncoder = JSONEncoder(),
        decoder: JSONDecoder = JSONDecoder()
    ) {
        self.apiClient = apiClient
        self.session = session
        self.encoder = encoder
        self.decoder = decoder
    }

    /// Streams the chat completion. `onEvent` is invoked in arrival
    /// order. Cancellation is caller-driven: cancel the surrounding
    /// `Task` to abort the local read loop, and call `cancel(requestId:)`
    /// to tell the daemon to stop generating server-side (the viewmodel
    /// is expected to do both).
    public func stream(
        request: ChatRequest,
        sessionId: UUID,
        onEvent: @escaping @Sendable (ChatStreamEvent) async -> Void
    ) async throws {
        var httpRequest = URLRequest(url: chatCompletionsURL())
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        httpRequest.setValue(
            Self.clientIdentifierValue,
            forHTTPHeaderField: Self.clientIdentifierHeader
        )
        httpRequest.setValue(
            sessionId.uuidString,
            forHTTPHeaderField: Self.sessionIdHeader
        )
        if let apiKey = apiClient.apiKey, !apiKey.isEmpty {
            httpRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        var streamRequest = request
        streamRequest.stream = true
        do {
            httpRequest.httpBody = try encoder.encode(streamRequest)
        } catch {
            throw MTPLXChatClientError.bodyEncodingFailed
        }

        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: httpRequest)
        } catch {
            if let urlError = error as? URLError,
                [URLError.Code.cannotConnectToHost,
                 .networkConnectionLost,
                 .cannotFindHost,
                 .notConnectedToInternet].contains(urlError.code) {
                throw MTPLXChatClientError.daemonUnreachable
            }
            throw error
        }
        guard let http = response as? HTTPURLResponse else {
            throw MTPLXChatClientError.invalidResponse
        }
        if http.statusCode == 401 {
            throw MTPLXChatClientError.unauthorized
        }
        guard (200..<300).contains(http.statusCode) else {
            // Drain a small body window for the error message; do not
            // hang on a giant non-streaming error body.
            var body = Data()
            var byteCount = 0
            for try await byte in bytes {
                body.append(byte)
                byteCount += 1
                if byteCount >= 4096 { break }
            }
            let text = String(data: body, encoding: .utf8) ?? ""
            throw MTPLXChatClientError.httpStatus(http.statusCode, text)
        }

        // Byte-buffer SSE parser. Reads bytes from the URLSession byte
        // stream and only checks the TRAILING 4 bytes for the SSE
        // block delimiter (\n\n or \r\n\r\n) — O(1) per byte instead
        // of the O(N) full-buffer rescan the first cut did. We do
        // this instead of `bytes.lines` because the latter has a
        // long-standing macOS buffering issue where it can hold
        // bytes until the stream closes, which masquerades as
        // "streaming is broken".
        var buffer = Data()
        buffer.reserveCapacity(8192)
        var emittedRequestId = false

        for try await byte in bytes {
            // Throw on cancel (rather than returning silently) so the
            // caller can distinguish a user Stop from a normal finish and
            // not persist a stale assistant turn.
            try Task.checkCancellation()
            buffer.append(byte)
            guard let blockLength = Self.trailingDelimiterBlockLength(in: buffer) else {
                continue
            }
            let block = buffer.prefix(blockLength.payloadCount)
            buffer.removeFirst(blockLength.payloadCount + blockLength.delimiterCount)
            let blockText = String(decoding: block, as: UTF8.self)
            let dataPayload = Self.dataPayload(from: blockText)
            guard let payload = dataPayload else { continue }
            if payload == "[DONE]" { return }
            guard let chunk = decodeChunk(payload) else { continue }
            if !emittedRequestId, let id = chunk.id {
                emittedRequestId = true
                await onEvent(.requestId(id))
            }
            await emitEvents(from: chunk, onEvent: onEvent)
        }
        // Server closed without a trailing delimiter; flush whatever
        // payload we have buffered so we don't drop the closing chunk
        // if an intermediary strips the terminator.
        if !buffer.isEmpty {
            let blockText = String(decoding: buffer, as: UTF8.self)
            if let payload = Self.dataPayload(from: blockText),
                payload != "[DONE]",
                let chunk = decodeChunk(payload)
            {
                if !emittedRequestId, let id = chunk.id {
                    await onEvent(.requestId(id))
                }
                await emitEvents(from: chunk, onEvent: onEvent)
            }
        }
    }

    // MARK: - SSE block parsing helpers

    /// Detects whether the buffer ends with an SSE block delimiter
    /// (`\n\n` or `\r\n\r\n`). Returns `(payloadCount, delimiterCount)`
    /// when found so the caller can slice the block off without
    /// rescanning the whole buffer.
    ///
    /// NOTE: indexes via `buffer.endIndex`, not `buffer.count - 1`.
    /// `Data.removeFirst(_:)` advances `startIndex` rather than
    /// re-basing storage, so `buffer[count - 1]` traps after the
    /// first removeFirst.
    private static func trailingDelimiterBlockLength(in buffer: Data) -> (payloadCount: Int, delimiterCount: Int)? {
        let count = buffer.count
        let end = buffer.endIndex
        if count >= 4,
            buffer[end - 4] == 0x0D, buffer[end - 3] == 0x0A,
            buffer[end - 2] == 0x0D, buffer[end - 1] == 0x0A
        {
            return (payloadCount: count - 4, delimiterCount: 4)
        }
        if count >= 2,
            buffer[end - 2] == 0x0A, buffer[end - 1] == 0x0A
        {
            return (payloadCount: count - 2, delimiterCount: 2)
        }
        return nil
    }

    /// Parses one SSE block's text into a joined `data:` payload, or
    /// nil if the block contained no `data:` lines (heartbeat / comment).
    private static func dataPayload(from block: String) -> String? {
        var pieces: [Substring] = []
        for raw in block.split(separator: "\n", omittingEmptySubsequences: false) {
            // Strip CR for CRLF terminators (the trailing-delimiter
            // check above accepts either; some intermediate lines may
            // also use \r\n).
            let line: Substring
            if raw.last == "\r" {
                line = raw.dropLast()
            } else {
                line = raw
            }
            if line.isEmpty || line.first == ":" { continue }
            if line.hasPrefix("data:") {
                var body = line.dropFirst("data:".count)
                if body.first == " " { body = body.dropFirst() }
                pieces.append(body)
            }
        }
        if pieces.isEmpty { return nil }
        return pieces.joined(separator: "\n")
    }

    /// Cancel an in-flight server-side generation. Returns silently if
    /// the daemon already cleaned it up (the cancel endpoint may 404
    /// for ids the daemon no longer knows about).
    public func cancel(requestId: String) async {
        _ = try? await apiClient.cancel(requestId: requestId)
    }

    // MARK: - Internals

    private func chatCompletionsURL() -> URL {
        // Use URLComponents the same way MTPLXAPIClient.makeURL does so
        // we don't get tripped up by a baseURL with or without a path.
        var components = URLComponents(
            url: apiClient.baseURL,
            resolvingAgainstBaseURL: false
        )!
        let basePath = components.percentEncodedPath
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let endpoint = "v1/chat/completions"
        let joined = [basePath, endpoint]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        components.percentEncodedPath = "/\(joined)"
        components.query = nil
        components.fragment = nil
        return components.url!
    }

    private func decodeChunk(_ payload: String) -> ChatCompletionChunk? {
        try? decoder.decode(ChatCompletionChunk.self, from: Data(payload.utf8))
    }

    /// The events one decoded `data:` payload folds into, in emission
    /// order, or nil when the payload is not a chat-completion chunk.
    /// Internal (not private) so the chunk-shape contract can be pinned
    /// by fixture tests without a socket.
    func streamEvents(fromDataPayload payload: String) -> [ChatStreamEvent]? {
        guard let chunk = decodeChunk(payload) else { return nil }
        return events(from: chunk)
    }

    private func emitEvents(
        from chunk: ChatCompletionChunk,
        onEvent: @escaping @Sendable (ChatStreamEvent) async -> Void
    ) async {
        for event in events(from: chunk) {
            await onEvent(event)
        }
    }

    private func events(from chunk: ChatCompletionChunk) -> [ChatStreamEvent] {
        var events: [ChatStreamEvent] = []
        let choice = chunk.choices?.first
        if let role = choice?.delta?.role {
            events.append(.role(role))
        }
        if let reasoning = choice?.delta?.reasoningContent, !reasoning.isEmpty {
            events.append(.reasoningDelta(reasoning))
        }
        if let content = choice?.delta?.content, !content.isEmpty {
            events.append(.contentDelta(content))
        }
        if let toolCalls = choice?.delta?.toolCalls, !toolCalls.isEmpty {
            for toolCall in toolCalls {
                let idx = toolCall.index ?? 0
                if let id = toolCall.id, let name = toolCall.function?.name, !name.isEmpty {
                    events.append(.toolCallStart(index: idx, id: id, name: name))
                }
                if let fragment = toolCall.function?.arguments, !fragment.isEmpty {
                    events.append(.toolCallArgumentsDelta(index: idx, fragment: fragment))
                }
            }
        }
        if let progress = chunk.mtplxProgress {
            events.append(.progress(progressFrame(from: progress)))
        }
        if let finish = choice?.finishReason {
            if finish == Self.errorFinishReason {
                // The daemon's failure frame: the top-level `error`
                // object carries the message users need to read. A
                // frame that claims failure without one still fails
                // the turn — a missing message must not turn a failed
                // reply into a completed one.
                let message = chunk.error?.message?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                events.append(
                    .serverError(
                        message: message.isEmpty
                            ? tr("The model server reported an error.")
                            : message
                    )
                )
            } else {
                events.append(
                    .finished(
                        finishReason: finish,
                        usage: usage(from: chunk.usage),
                        stats: stats(from: chunk.mtplxStats)
                    )
                )
            }
        }
        return events
    }

    /// `finish_reason` value the daemon uses on its failure frame.
    static let errorFinishReason = "error"

    private func progressFrame(from raw: DynamicObject) -> ChatProgressFrame {
        ChatProgressFrame(
            completionTokens: raw.intValue(for: "completion_tokens"),
            decodeTokS: raw.doubleValue(for: "decode_tok_s"),
            displayDecodeTokS: raw.doubleValue(for: "display_decode_tok_s"),
            phase: raw.stringValue(for: "phase"),
            raw: raw
        )
    }

    private func usage(from raw: ChatCompletionUsage?) -> ChatUsage? {
        guard let raw else { return nil }
        return ChatUsage(
            promptTokens: raw.promptTokens,
            completionTokens: raw.completionTokens,
            totalTokens: raw.totalTokens
        )
    }

    private func stats(from raw: DynamicObject?) -> ChatStreamStats? {
        guard let raw else { return nil }
        return ChatStreamStats(
            rawDecodeTokS: raw.doubleValue(for: "raw_decode_tok_s")
                ?? raw.doubleValue(for: "decode_tok_s"),
            displayDecodeTokS: raw.doubleValue(for: "display_decode_tok_s"),
            ttftS: raw.doubleValue(for: "ttft_s"),
            requestElapsedS: raw.doubleValue(for: "request_elapsed_s"),
            acceptedByDepth: raw.intArrayValue(for: "accepted_by_depth"),
            draftedByDepth: raw.intArrayValue(for: "drafted_by_depth"),
            verifyCalls: raw.intValue(for: "verify_calls"),
            verifyTimeS: raw.doubleValue(for: "verify_time_s"),
            raw: raw
        )
    }
}

// MARK: - Internal chunk decoders

private struct ChatCompletionChunk: Decodable {
    var id: String?
    var choices: [ChatCompletionChoice]?
    var usage: ChatCompletionUsage?
    var mtplxProgress: DynamicObject?
    var mtplxStats: DynamicObject?
    /// Top-level OpenAI-shape error object the daemon attaches to its
    /// `finish_reason: "error"` frame.
    var error: ChatCompletionError?

    private enum CodingKeys: String, CodingKey {
        case id, choices, usage, error
        case mtplxProgress = "mtplx_progress"
        case mtplxStats = "mtplx_stats"
    }
}

/// Only `message` is read: `type`/`code` vary in shape across servers
/// (string or number), and a field nobody consumes must never be able
/// to fail the decode and drop the whole failure frame.
private struct ChatCompletionError: Decodable {
    var message: String?
}

private struct ChatCompletionChoice: Decodable {
    var index: Int?
    var delta: ChatCompletionDelta?
    var finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

private struct ChatCompletionDelta: Decodable {
    var role: String?
    var content: String?
    var reasoningContent: String?
    var toolCalls: [ChatCompletionDeltaToolCall]?

    private enum CodingKeys: String, CodingKey {
        case role, content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }
}

private struct ChatCompletionDeltaToolCall: Decodable {
    var index: Int?
    var id: String?
    var type: String?
    var function: ChatCompletionDeltaToolCallFunction?
}

private struct ChatCompletionDeltaToolCallFunction: Decodable {
    var name: String?
    var arguments: String?
}

private struct ChatCompletionUsage: Decodable {
    var promptTokens: Int?
    var completionTokens: Int?
    var totalTokens: Int?

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

// MARK: - DynamicObject convenience

private extension DynamicObject {
    func intValue(for key: String) -> Int? {
        switch values[key] {
        case .number(let n): return Int(n)
        default: return nil
        }
    }

    func doubleValue(for key: String) -> Double? {
        switch values[key] {
        case .number(let n): return n
        default: return nil
        }
    }

    func stringValue(for key: String) -> String? {
        switch values[key] {
        case .string(let s): return s
        default: return nil
        }
    }

    func intArrayValue(for key: String) -> [Int]? {
        switch values[key] {
        case .array(let arr):
            return arr.compactMap { element -> Int? in
                if case .number(let n) = element { return Int(n) }
                return nil
            }
        default:
            return nil
        }
    }
}
