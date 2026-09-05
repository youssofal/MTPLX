import Foundation

// MARK: - PortPreflight
//
// Classifies who owns the configured daemon port BEFORE the app spawns a
// server. Without this, a foreign occupant (some other dev server on 8000)
// surfaced as a raw "Degraded: Port is already used…" failure after a full
// launch attempt. With it, the store can fall back to the next free port
// and tell the user in one humane sentence.
//
// SYNC PAIR: mtplx/daemon_client.py classify_port_occupant implements the
// same classification for the CLI. Update both sides together.

public enum PortOccupantKind: Equatable, Sendable {
    /// Nothing is listening; safe to bind.
    case free
    /// A healthy MTPLX daemon answered `/health`. The supervisor decides
    /// separately whether it is adoptable (app-owned, same model).
    case mtplxServer(HealthPayload)
    /// A live listener rejected the probe with 401/403 — an auth-protected
    /// server (often an MTPLX daemon with a different API key). Provably
    /// alive, never adoptable with the current credentials.
    case unauthorized
    /// Something is listening but does not speak MTPLX health.
    case foreign
}

public enum PortPreflight {
    /// Classify the occupant of `baseURL`'s port with a short timeout.
    ///
    /// `baseURL` must be a CONNECT address (see MTPLXServerURLs): probing a
    /// wildcard bind address like http://0.0.0.0 times out on a free port
    /// and misclassified it as `.foreign` (issue #109's bogus "Port 8000
    /// was in use by another app").
    public static func classify(
        baseURL: URL,
        apiKey: String?,
        timeoutSeconds: TimeInterval = 2
    ) async -> PortOccupantKind {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeoutSeconds
        configuration.timeoutIntervalForResource = timeoutSeconds
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let client = MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey, session: session)
        do {
            let health = try await client.health()
            return health.ok ? .mtplxServer(health) : .foreign
        } catch let error as URLError {
            switch error.code {
            case .cannotConnectToHost, .cannotFindHost, .networkConnectionLost:
                return .free
            default:
                // Timeouts and protocol garbage both mean "occupied by
                // something we cannot use".
                return .foreign
            }
        } catch MTPLXAPIClientError.httpStatus(401, _),
                MTPLXAPIClientError.httpStatus(403, _) {
            // The daemon's own /health requires the API key; a wrong or
            // missing key must read as "auth-protected listener", not as a
            // foreign app.
            return .unauthorized
        } catch {
            // Decode failures / non-2xx statuses: a listener that is not an
            // MTPLX daemon.
            return .foreign
        }
    }

    /// Issue #409 (app path): a stopping MTPLX daemon keeps its listener
    /// while it drains, but its `/health` stops answering first, so a plain
    /// probe reads our own draining server as `.foreign` for those few
    /// seconds — every stop/start (and the first launch after an in-app
    /// update, whose predecessor is still draining) then moved the
    /// configured port to +1. Long enough to cover a normal drain, short
    /// enough that a genuinely foreign listener does not stall a launch.
    /// SYNC PAIR: mtplx/daemon_client.py wait_for_port_settle.
    public static let settleTimeoutSeconds: TimeInterval = 5
    public static let settlePollSeconds: TimeInterval = 0.25

    /// `classify`, re-probed while the answer is `.foreign` until it settles
    /// or the window closes. Any other classification is terminal and
    /// returns immediately. Separates a TRANSIENT occupant (our own draining
    /// server, a socket in the tail of TIME_WAIT, a listener mid-restart)
    /// from a STEADY foreign listener that is still there after the window.
    public static func classifySettled(
        baseURL: URL,
        apiKey: String?,
        settleTimeoutSeconds: TimeInterval = settleTimeoutSeconds,
        pollSeconds: TimeInterval = settlePollSeconds
    ) async -> PortOccupantKind {
        let deadline = Date().addingTimeInterval(max(0, settleTimeoutSeconds))
        var occupant = await classify(baseURL: baseURL, apiKey: apiKey)
        while case .foreign = occupant, Date() < deadline {
            try? await Task.sleep(nanoseconds: UInt64(max(0.01, pollSeconds) * 1_000_000_000))
            occupant = await classify(baseURL: baseURL, apiKey: apiKey)
        }
        return occupant
    }

    /// First port strictly after `port` that the daemon's own bind address
    /// can take. `bindHost` must be the CONFIGURED bind host: a wildcard
    /// daemon needs INADDR_ANY free, which a loopback-only check misses
    /// (and vice versa a busy loopback port may be irrelevant to a
    /// specific-interface bind).
    public static func nextFreePort(
        after port: Int,
        bindHost: String = "127.0.0.1",
        attempts: Int = 50
    ) -> Int? {
        guard port < 65_535 else { return nil }
        let upperBound = min(port + max(1, attempts), 65_535)
        for candidate in (port + 1)...upperBound
        where portIsBindable(candidate, bindHost: bindHost) {
            return candidate
        }
        return nil
    }

    static func portIsBindable(_ port: Int, bindHost: String = "127.0.0.1") -> Bool {
        let raw = MTPLXServerURLs.isWildcardBind(bindHost)
            ? nil
            : bindHost.trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, raw.contains(":") {
            return ipv6PortIsBindable(port, host: raw)
        }
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        var reuse: Int32 = 1
        setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            socklen_t(MemoryLayout<Int32>.size)
        )
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        if let raw {
            let parsed = inet_addr(raw.isEmpty ? "127.0.0.1" : raw)
            guard parsed != INADDR_NONE else { return false }
            address.sin_addr.s_addr = parsed
        } else {
            // Wildcard daemon bind: test the address family it will use.
            address.sin_addr.s_addr = INADDR_ANY
        }
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { rebound in
                bind(descriptor, rebound, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    private static func ipv6PortIsBindable(_ port: Int, host: String) -> Bool {
        let descriptor = socket(AF_INET6, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        var reuse: Int32 = 1
        setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            socklen_t(MemoryLayout<Int32>.size)
        )
        var address = sockaddr_in6()
        address.sin6_family = sa_family_t(AF_INET6)
        address.sin6_port = in_port_t(UInt16(port).bigEndian)
        let bare = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast())
            : host
        guard inet_pton(AF_INET6, bare, &address.sin6_addr) == 1 else { return false }
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { rebound in
                bind(descriptor, rebound, socklen_t(MemoryLayout<sockaddr_in6>.size))
            }
        }
        return result == 0
    }
}
