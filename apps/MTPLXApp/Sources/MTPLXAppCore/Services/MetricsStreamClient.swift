import Foundation

public enum MetricsConnectionState: Equatable, Sendable {
    case idle
    case connecting
    case open
    case healthyWithoutMetrics
    case reconnecting(Int)
    case failed(String)
}

public enum MetricsStreamEvent: Equatable, Sendable {
    case snapshot(DashboardSnapshot)
    case progress(DynamicObject)
    case completed(DynamicObject)
    case newMaxTPS(DynamicObject)
    case thermal(DynamicObject)
    case prefill(DynamicObject)
    case raw(name: String, payload: DynamicObject)
}

public struct SSEMessage: Equatable, Sendable {
    public var event: String
    public var data: String

    public init(event: String, data: String) {
        self.event = event
        self.data = data
    }
}

/// Incremental SSE framing: feed bytes, get complete messages. Two
/// compares and an array append per byte — replaces the per-byte
/// Data.append + delimiter rescan + whole-event Foundation re-split
/// that burned ~15% of a core at the 10 Hz snapshot rate (2026-08-17
/// field regression). NOT built on `AsyncLineSequence`: that SKIPS
/// blank lines (verified 2026-08-17), and the blank line IS the SSE
/// message boundary. CR is dropped, matching the historical
/// "\r\n" -> "\n" normalization.
public struct SSELineAccumulator: Sendable {
    private var line: [UInt8] = []
    private var event = "message"
    private var dataLines: [String] = []

    public init() {
        line.reserveCapacity(1024)
    }

    public mutating func consume(_ byte: UInt8) -> SSEMessage? {
        if byte == 0x0D { return nil }
        if byte != 0x0A {
            line.append(byte)
            return nil
        }
        if line.isEmpty {
            // Blank line: one complete SSE message.
            defer {
                event = "message"
                dataLines.removeAll(keepingCapacity: true)
            }
            guard !dataLines.isEmpty else { return nil }
            return SSEMessage(event: event, data: dataLines.joined(separator: "\n"))
        }
        let text = String(decoding: line, as: UTF8.self)
        line.removeAll(keepingCapacity: true)
        if text.hasPrefix(":") { return nil }
        if text.hasPrefix("event:") {
            event = String(text.dropFirst("event:".count))
                .trimmingCharacters(in: .whitespaces)
        } else if text.hasPrefix("data:") {
            dataLines.append(
                String(text.dropFirst("data:".count))
                    .trimmingCharacters(in: .whitespaces)
            )
        }
        return nil
    }
}

public struct SSEParser: Sendable {
    public init() {}

    public func parse(_ text: String) -> [SSEMessage] {
        text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n\n")
            .compactMap(parseBlock)
    }

    private func parseBlock(_ block: String) -> SSEMessage? {
        var event = "message"
        var dataLines: [String] = []
        for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix(":") { continue }
            if line.hasPrefix("event:") {
                event = String(line.dropFirst("event:".count)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                dataLines.append(String(line.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces))
            }
        }
        guard !dataLines.isEmpty else { return nil }
        return SSEMessage(event: event, data: dataLines.joined(separator: "\n"))
    }
}

public final class MetricsStreamClient: Sendable {
    private let apiClient: MTPLXAPIClient
    private let decoder = JSONDecoder()
    private let parser = SSEParser()

    public init(apiClient: MTPLXAPIClient) {
        self.apiClient = apiClient
    }

    public func decode(message: SSEMessage) throws -> MetricsStreamEvent {
        let data = Data(message.data.utf8)
        if message.event == "snapshot" {
            return .snapshot(try decoder.decode(DashboardSnapshot.self, from: data))
        }
        let payload = try decoder.decode(DynamicObject.self, from: data)
        switch message.event {
        case "progress":
            return .progress(payload)
        case "completed":
            return .completed(payload)
        case "new_max_tps":
            return .newMaxTPS(payload)
        case "thermal":
            return .thermal(payload)
        case "prefill":
            return .prefill(payload)
        default:
            return .raw(name: message.event, payload: payload)
        }
    }

    public func connect(
        snapshotIntervalMs: Int,
        onState: @escaping @Sendable (MetricsConnectionState) async -> Void,
        onEvent: @escaping @Sendable (MetricsStreamEvent) async -> Void
    ) async {
        await onState(.connecting)
        var attempt = 0
        while !Task.isCancelled {
            do {
                let request = makeRequest(snapshotIntervalMs: snapshotIntervalMs)
                let (bytes, response) = try await apiClient.session.bytes(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw MTPLXAPIClientError.invalidResponse
                }
                guard (200..<300).contains(http.statusCode) else {
                    if http.statusCode == 401 || http.statusCode == 403 {
                        await onState(.failed("Metrics stream rejected the app API key. Update Settings or restart MTPLX with the same key."))
                        return
                    }
                    throw MTPLXAPIClientError.httpStatus(http.statusCode, "")
                }
                attempt = 0
                await onState(.open)
                var accumulator = SSELineAccumulator()
                for try await byte in bytes {
                    guard let message = accumulator.consume(byte) else { continue }
                    if Task.isCancelled { return }
                    await onEvent(try decode(message: message))
                }
            } catch {
                if Task.isCancelled { return }
                attempt += 1
                await onState(.reconnecting(attempt))
                try? await Task.sleep(nanoseconds: UInt64(min(30, attempt * 2)) * 1_000_000_000)
            }
        }
    }

    func makeRequest(snapshotIntervalMs: Int) -> URLRequest {
        var request = URLRequest(
            url: apiClient.metricsStreamURL(snapshotIntervalMs: snapshotIntervalMs)
        )
        request.httpMethod = "GET"
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        if let auth = apiClient.authorizationHeader {
            request.setValue(auth, forHTTPHeaderField: "Authorization")
        }
        return request
    }
}

// Internal so sibling SSE clients in MTPLXAppCore (e.g.
// BenchmarkStreamClient) can share the trailing-delimiter trick. The
// implementation never traps because we range-check `count` first.
extension Data {
    var hasSSEDelimiterSuffix: Bool {
        (count >= 2 && self[count - 2] == 0x0A && self[count - 1] == 0x0A)
            || (count >= 4
                && self[count - 4] == 0x0D
                && self[count - 3] == 0x0A
                && self[count - 2] == 0x0D
                && self[count - 1] == 0x0A)
    }
}
