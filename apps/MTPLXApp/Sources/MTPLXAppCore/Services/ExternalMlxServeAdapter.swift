import Foundation

/// The two deliberately different server contracts the app can supervise.
///
/// MTPLX's `/health` contains startup ownership, fan, and model metadata.
/// The exact DeepSeek V4 target-only route is intentionally not an MTPLX
/// daemon: its native `mlx-serve` process answers only `{ "status": "ok" }`.
/// Keeping the distinction in the type system prevents a successful native
/// launch from being followed by calls to MTPLX-only admin endpoints.
public enum DaemonBackendKind: Equatable, Sendable {
    case mtplx
    case externalMlxServe
}

public struct ExternalMlxServeHealth: Codable, Equatable, Sendable {
    public let status: String

    public var ok: Bool {
        status.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() == "ok"
    }
}

public enum ExternalMlxServeLiveness: Sendable {
    case healthy(ExternalMlxServeHealth)
    case aliveUnauthorized
    case unreachable
}

/// Minimal adapter for the native target-only server. It is intentionally
/// limited to `/health`; OpenAI chat traffic continues through the existing
/// chat client, while MTPLX-specific capabilities, sessions, settings, and
/// metrics are never assumed to exist on this server.
public struct ExternalMlxServeAdapter: Sendable {
    public var baseURL: URL
    public var apiKey: String?
    public var session: URLSession

    public init(
        baseURL: URL,
        apiKey: String? = nil,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
    }

    public func health() async throws -> ExternalMlxServeHealth {
        var request = URLRequest(url: makeURL("/health"))
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MTPLXAPIClientError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw MTPLXAPIClientError.httpStatus(
                http.statusCode,
                String(data: data, encoding: .utf8) ?? ""
            )
        }
        return try JSONDecoder().decode(ExternalMlxServeHealth.self, from: data)
    }

    /// A raw `status: ok` health response is the external server's complete
    /// readiness contract. Authentication failures still prove the listener
    /// is alive and must not cause an app-owned process reap.
    public func livenessWithinDeadline(seconds: TimeInterval) async -> ExternalMlxServeLiveness {
        await withTaskGroup(of: ExternalMlxServeLiveness?.self) { group in
            group.addTask {
                do {
                    let health = try await self.health()
                    return health.ok ? .healthy(health) : .unreachable
                } catch MTPLXAPIClientError.httpStatus(401, _),
                        MTPLXAPIClientError.httpStatus(403, _) {
                    return .aliveUnauthorized
                } catch {
                    return .unreachable
                }
            }
            group.addTask {
                try? await Task.sleep(
                    nanoseconds: UInt64(max(0, seconds) * 1_000_000_000)
                )
                return nil
            }
            let winner = await group.next() ?? nil
            group.cancelAll()
            return winner ?? .unreachable
        }
    }

    public static func livenessProbe(baseURL: URL, apiKey: String?) -> ExternalMlxServeAdapter {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        configuration.timeoutIntervalForResource = 10
        configuration.httpMaximumConnectionsPerHost = 1
        configuration.waitsForConnectivity = false
        return ExternalMlxServeAdapter(
            baseURL: baseURL,
            apiKey: apiKey,
            session: URLSession(configuration: configuration)
        )
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
}
