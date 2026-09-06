import Foundation
import os

// MARK: - MTPLXChatToolFactory
//
// Builds the OpenAI-style tool surface the in-app chat exposes to the
// model and routes each tool call to the right service. The schemas
// (`web_search`, `fetch_url`) match Aphanes V2 verbatim so the model's
// learned tool-use behaviour transfers directly.
//
// Differences from Aphanes (deliberate, per plan):
//   - No memory/RAG tools. MTPLX has no embedder; those tools depend
//     on one.
//   - No comparison-query expansion or embedding rerank. Results are
//     DDG+Brave merge-order only.
//   - Tool-round policy is enforced by `ChatViewModel`, not here; the
//     factory dispatches one call at a time and is stateless except for
//     the Jaccard duplicate-query guard.

// MARK: - Dispatch outcome

/// Why a tool call produced no usable result. `detail` is plain
/// English for the model's tool result and the activity strip's
/// caption (the UI adds its own localised label in front of it).
public struct ChatToolFailure: Error, Equatable, Sendable {
    public enum Kind: String, Equatable, Sendable {
        case unknownTool = "unknown_tool"
        case emptyQuery = "empty_query"
        case invalidURL = "invalid_url"
        case searchFailed = "search_failed"
        case fetchFailed = "fetch_failed"
    }

    public var kind: Kind
    public var detail: String

    public init(kind: Kind, detail: String) {
        self.kind = kind
        self.detail = detail
    }
}

/// What one tool call came back with. `resultJSON` is always the
/// content of the `role: "tool"` message the model receives; when the
/// call failed it says so and why, so the model reports the failure
/// instead of answering as if a search had found nothing. `failure` is
/// the typed reason the app records and shows — the dispatch loop used
/// to hard-code every call as a success because the factory only ever
/// returned a string.
public struct ChatToolDispatchResult: Equatable, Sendable {
    public var resultJSON: String
    public var failure: ChatToolFailure?

    public init(resultJSON: String, failure: ChatToolFailure? = nil) {
        self.resultJSON = resultJSON
        self.failure = failure
    }

    public var succeeded: Bool { failure == nil }
}

public struct MTPLXChatToolFactory: Sendable {
    public let webSearch: WebSearchService
    public let urlFetcher: URLFetcher
    public let session: ToolSessionState

    private static let log = Logger(subsystem: "com.mtplx.app", category: "ChatTools")

    public init(
        webSearch: WebSearchService = WebSearchService(),
        urlFetcher: URLFetcher = URLFetcher(),
        session: ToolSessionState = ToolSessionState()
    ) {
        self.webSearch = webSearch
        self.urlFetcher = urlFetcher
        self.session = session
    }

    // MARK: - Tool definitions

    /// OpenAI-shape tool definitions the chat client puts on the wire.
    /// Returns typed `ChatRequestTool` values so the viewmodel can drop
    /// them straight into `ChatRequest.tools`.
    public func toolDefinitions(
        webSearchEnabled: Bool = true,
        workspaceRoot: String? = nil
    ) -> [ChatRequestTool] {
        var definitions: [ChatRequestTool] = []
        if webSearchEnabled {
            definitions += [webSearchToolDefinition(), fetchURLToolDefinition()]
        }
        if workspaceRoot != nil {
            definitions += MTPLXWorkspaceToolService().toolDefinitions()
        }
        return definitions
    }

    public func requiresApproval(for name: String) -> Bool {
        requiresApproval(for: name, policy: [:])
    }

    public func requiresApproval(for name: String, policy: [String: String]) -> Bool {
        guard let key = Self.workspacePolicyKey(for: name) else { return false }
        return policy[key]?.lowercased() != "allow"
    }

    public static func workspacePolicyKey(for name: String) -> String? {
        switch name {
        case "list_files", "read_file", "inspect_repo", "git_status", "git_diff":
            return "read"
        case "search_files":
            return "search"
        case "write_file", "apply_patch":
            return "write"
        case "run_tests", "run_command":
            return "terminal"
        case "web_search", "fetch_url":
            return "browser"
        default:
            return nil
        }
    }

    /// Wakes the per-turn state machine — call once at the start of
    /// each user turn so the Jaccard guard sees a fresh slate.
    public func beginTurn() async {
        await session.reset()
    }

    // MARK: - Dispatch

    /// Routes a tool call. The result's JSON is the content of the
    /// `role: "tool"` message sent back to the model; its `failure` is
    /// set when the call could not be carried out.
    public func dispatch(name: String, argumentsJSON: String) async -> ChatToolDispatchResult {
        switch name {
        case "web_search":
            return await dispatchWebSearch(argumentsJSON: argumentsJSON)
        case "fetch_url":
            return await dispatchFetchURL(argumentsJSON: argumentsJSON)
        default:
            Self.log.warning("Unknown tool name: \(name, privacy: .public)")
            return failed(
                .unknownTool,
                detail: "\(name) is not a tool MTPLX chat provides",
                fields: ["tool": name],
                note: "This tool is not available in MTPLX chat, so the call did nothing. "
                    + "Tell the user if they asked for it, and answer from what you already have."
            )
        }
    }

    // MARK: - web_search

    private func dispatchWebSearch(argumentsJSON: String) async -> ChatToolDispatchResult {
        let query = parseQuery(from: argumentsJSON).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            return failed(
                .emptyQuery,
                detail: "web_search was called without a query",
                note: "web_search was called with an empty query, so nothing was searched. "
                    + "Call it again with a specific query, or answer from knowledge."
            )
        }

        switch await session.begin(query: query) {
        case .proceed:
            break
        case .duplicate(let previous, let warningCount):
            Self.log.info(
                "Skipping duplicate web_search query='\(query, privacy: .public)' previous='\(previous, privacy: .public)' warnings=\(warningCount, privacy: .public)"
            )
            return succeeded([
                "query": query,
                "previous_query": previous,
                "warning_count": warningCount,
                "note": "Query is too similar to a previous search this turn. Use the earlier results instead of repeating the search.",
            ])
        case .disabled:
            Self.log.info("web_search disabled for the remainder of this turn")
            return succeeded([
                "query": query,
                "note": "web_search is disabled for the rest of this turn. Answer from knowledge or the previously fetched sources.",
            ])
        }

        let searchRequest = WebSearchRequest(query: query, maxResults: 5)
        let searchResults: [WebSearchResult]
        do {
            searchResults = try await webSearch.search(searchRequest)
        } catch {
            Self.log.error("web_search failed: \(error.localizedDescription, privacy: .public)")
            return failed(
                .searchFailed,
                detail: error.localizedDescription,
                fields: ["query": query],
                note: "The web search could not be carried out (network or provider error), "
                    + "so nothing is known about what it would have found. Tell the user the "
                    + "search failed, answer from your own knowledge if you can, and do not "
                    + "claim that nothing was found. Do not retry this turn."
            )
        }

        guard !searchResults.isEmpty else {
            return succeeded([
                "query": query,
                "results": [] as [Any],
                "note": "No results. Answer the user's question from your knowledge.",
            ])
        }

        // Fetch full readable text for the top 3 URLs in parallel.
        // Limit is conservative on purpose — beyond 3 we'd inflate the
        // tool-result body past the model's useful attention budget.
        let fetchCount = min(3, searchResults.count)
        let fetched = await withTaskGroup(of: (URL, URLFetchResult?).self) { group in
            for result in searchResults.prefix(fetchCount) {
                group.addTask { @Sendable [urlFetcher] in
                    do {
                        let page = try await urlFetcher.fetch(URLFetchRequest(url: result.url))
                        return (result.url, page)
                    } catch {
                        return (result.url, nil)
                    }
                }
            }
            var results: [URL: URLFetchResult?] = [:]
            for await pair in group {
                results[pair.0] = pair.1
            }
            return results
        }

        let resultObjects: [[String: Any]] = searchResults.map { result in
            var dict: [String: Any] = [
                "title": result.title,
                "url": result.url.absoluteString,
                "snippet": result.snippet,
                "host": result.url.host ?? "",
            ]
            if let page = fetched[result.url] ?? nil {
                dict["page_content"] = page.content
            }
            return dict
        }

        return succeeded([
            "query": query,
            "results": resultObjects,
        ])
    }

    // MARK: - fetch_url

    private func dispatchFetchURL(argumentsJSON: String) async -> ChatToolDispatchResult {
        let rawURL = parseString(from: argumentsJSON, key: "url")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: rawURL), let scheme = url.scheme,
            scheme == "http" || scheme == "https"
        else {
            return failed(
                .invalidURL,
                detail: "\"\(rawURL)\" is not an http(s) URL",
                fields: ["url": rawURL],
                note: "fetch_url needs an http(s) URL, so nothing was fetched."
            )
        }

        do {
            let result = try await urlFetcher.fetch(URLFetchRequest(url: url))
            return succeeded([
                "url": result.url.absoluteString,
                "title": result.title ?? "",
                "content": result.content,
            ])
        } catch {
            Self.log.error("fetch_url failed for \(url, privacy: .public): \(error.localizedDescription, privacy: .public)")
            return failed(
                .fetchFailed,
                detail: error.localizedDescription,
                fields: ["url": url.absoluteString],
                note: "The page could not be fetched (network or server error), so its "
                    + "content is unknown. Tell the user the fetch failed; do not retry the "
                    + "same URL this turn."
            )
        }
    }

    // MARK: - Outcomes

    private func succeeded(_ dict: [String: Any]) -> ChatToolDispatchResult {
        ChatToolDispatchResult(resultJSON: jsonObject(dict))
    }

    /// A failed call still hands the model a result — one that names the
    /// failure and says what to do — and the app a typed reason.
    private func failed(
        _ kind: ChatToolFailure.Kind,
        detail: String,
        fields: [String: Any] = [:],
        note: String
    ) -> ChatToolDispatchResult {
        var payload = fields
        payload["error"] = kind.rawValue
        payload["detail"] = detail
        payload["note"] = note
        return ChatToolDispatchResult(
            resultJSON: jsonObject(payload),
            failure: ChatToolFailure(kind: kind, detail: detail)
        )
    }

    // MARK: - Schema definitions

    private func webSearchToolDefinition() -> ChatRequestTool {
        ChatRequestTool(
            function: ChatRequestToolDefinition(
                name: "web_search",
                description: "Search the web and automatically read the strongest current sources. Use this for current facts, product comparisons, recent releases, reputational questions, or any claim you need to verify.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("The search query"),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
    }

    private func fetchURLToolDefinition() -> ChatRequestTool {
        ChatRequestTool(
            function: ChatRequestToolDefinition(
                name: "fetch_url",
                description: "Fetch and extract readable text content from a URL. Use this when the user provides a URL and wants to know what is on that page.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "url": .object([
                            "type": .string("string"),
                            "description": .string("The URL to fetch"),
                        ]),
                    ]),
                    "required": .array([.string("url")]),
                ])
            )
        )
    }

    // MARK: - JSON parsing helpers

    private func parseQuery(from argumentsJSON: String) -> String {
        parseString(from: argumentsJSON, key: "query")
    }

    private func parseString(from argumentsJSON: String, key: String) -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "" }
        return (object[key] as? String) ?? ""
    }

    private func jsonObject(_ dict: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
            let text = String(data: data, encoding: .utf8)
        else {
            return "{\"error\":\"json_encode_failed\"}"
        }
        return text
    }
}

// MARK: - Per-turn duplicate-query guard
//
// Mirrors Aphanes' `SearchSessionState`. After two near-duplicate
// `web_search` queries within one turn, the third call is denied with
// an explicit JSON note so the model can recover instead of spinning.

public actor ToolSessionState {
    /// Lower-case, whitespace-stripped query strings already issued
    /// this turn. Used by the Jaccard guard.
    private var seenQueries: [String] = []
    private var warningCount: Int = 0
    private var disabled: Bool = false
    private static let jaccardThreshold: Double = 0.85
    private static let maxWarnings: Int = 2

    public init() {}

    public func reset() {
        seenQueries.removeAll()
        warningCount = 0
        disabled = false
    }

    public enum Decision: Sendable {
        case proceed
        case duplicate(previous: String, warningCount: Int)
        case disabled
    }

    public func begin(query rawQuery: String) -> Decision {
        if disabled { return .disabled }
        let normalized = rawQuery.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return .proceed }
        for previous in seenQueries {
            if jaccard(previous, normalized) >= Self.jaccardThreshold {
                warningCount += 1
                if warningCount >= Self.maxWarnings {
                    disabled = true
                }
                return .duplicate(previous: previous, warningCount: warningCount)
            }
        }
        seenQueries.append(normalized)
        return .proceed
    }

    private func jaccard(_ lhs: String, _ rhs: String) -> Double {
        let left = Set(lhs.split(separator: " ").map { String($0) })
        let right = Set(rhs.split(separator: " ").map { String($0) })
        let intersection = left.intersection(right).count
        let union = left.union(right).count
        guard union > 0 else { return 0 }
        return Double(intersection) / Double(union)
    }
}
