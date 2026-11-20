import Foundation
import OSLog
import os

public enum AIMERenderMode: String, CaseIterable, Sendable {
    case fullScroll = "full_scroll"
    case noAutoscroll = "no_autoscroll"
    case tailLatex = "tail_latex"
    case plainTail = "plain_tail"
    case hidden
}

public enum AIMESignpost: Sendable {
    case streamEvent
    case bufferFlush
    case documentAppend
    case mathParse
    case renderPublication
    case scroll
    case backendMetricsReceive
    case displayedTPSSelection
}

public enum AIMEDiagnosticValue: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else {
            self = .string(try container.decode(String.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        }
    }
}

public struct AIMEDiagnosticEvent: Codable, Equatable, Sendable {
    public var wallTime: String
    public var uptimeS: Double
    public var name: String
    public var fields: [String: AIMEDiagnosticValue]

    public init(
        wallTime: String,
        uptimeS: Double,
        name: String,
        fields: [String: AIMEDiagnosticValue]
    ) {
        self.wallTime = wallTime
        self.uptimeS = uptimeS
        self.name = name
        self.fields = fields
    }
}

public enum AIMEDiagnostics {
    public static let logger = Logger(subsystem: "com.mtplx.app", category: "AIMEPerf")
    public static let signpostLog = OSLog(subsystem: "com.mtplx.app", category: "AIMEPerf")

    /// Cached once: `ProcessInfo.environment` rebuilds the whole
    /// dictionary from `environ` on every access, and this flag is
    /// checked twice per document append at up to 62 Hz × 2 documents
    /// (~250 environment rebuilds/s on the MainActor for a value that
    /// cannot change after launch — 2026-08-17 field regression).
    public static let isEnabled: Bool =
        isEnabled(environment: ProcessInfo.processInfo.environment)

    /// Cached once, like `isEnabled`: the Benchmark tab's reasoning pane
    /// reads both on every body evaluation, once per streamed revision.
    public static let renderMode: AIMERenderMode =
        renderMode(environment: ProcessInfo.processInfo.environment)

    public static let tailBlockLimit: Int =
        tailBlockLimit(environment: ProcessInfo.processInfo.environment)

    private static let sampleGate = AIMEDiagnosticSampleGate()

    public static func isEnabled(environment: [String: String]) -> Bool {
        guard let raw = environment["MTPLX_AIME_DIAGNOSTICS"]?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return false
        }
        switch raw.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    public static func renderMode(environment: [String: String]) -> AIMERenderMode {
        guard let raw = environment["MTPLX_AIME_RENDER_MODE"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              let mode = AIMERenderMode(rawValue: raw) else {
            return .tailLatex
        }
        return mode
    }

    public static func tailBlockLimit(environment: [String: String]) -> Int {
        let fallback = 48
        guard let raw = environment["MTPLX_AIME_TAIL_BLOCKS"],
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return fallback
        }
        return min(max(value, 20), 1_000)
    }

    public static func record(
        _ name: String,
        fields: [String: AIMEDiagnosticValue] = [:],
        flushImmediately: Bool = false,
        force: Bool = false
    ) {
        guard force || isEnabled else { return }
        let event = AIMEDiagnosticEvent(
            wallTime: ISO8601DateFormatter().string(from: Date()),
            uptimeS: ProcessInfo.processInfo.systemUptime,
            name: name,
            fields: fields
        )
        logger.debug("AIMEPerf event=\(name, privacy: .public) fields=\(fields.count, privacy: .public)")
        Task.detached(priority: .utility) {
            await AIMEDiagnosticJSONLWriter.shared.write(event)
            if flushImmediately {
                await AIMEDiagnosticJSONLWriter.shared.flushPending()
            }
        }
    }

    public static func shouldRecordCadenced(
        _ name: String,
        intervalS: TimeInterval = 1,
        tokenCount: Int? = nil,
        identity: String? = nil,
        force: Bool = false
    ) -> Bool {
        guard isEnabled else { return false }
        return sampleGate.shouldRecord(
            name: name,
            intervalS: intervalS,
            tokenCount: tokenCount,
            identity: identity,
            force: force
        )
    }

    public static func signpost(_ signpost: AIMESignpost) {
        guard isEnabled else { return }
        switch signpost {
        case .streamEvent:
            os_signpost(.event, log: signpostLog, name: "StreamEvent")
        case .bufferFlush:
            os_signpost(.event, log: signpostLog, name: "BufferFlush")
        case .documentAppend:
            os_signpost(.event, log: signpostLog, name: "DocumentAppend")
        case .mathParse:
            os_signpost(.event, log: signpostLog, name: "MathParse")
        case .renderPublication:
            os_signpost(.event, log: signpostLog, name: "RenderPublication")
        case .scroll:
            os_signpost(.event, log: signpostLog, name: "Scroll")
        case .backendMetricsReceive:
            os_signpost(.event, log: signpostLog, name: "BackendMetricsReceive")
        case .displayedTPSSelection:
            os_signpost(.event, log: signpostLog, name: "DisplayedTPSSelection")
        }
    }

    public static func fields(
        _ entries: (String, AIMEDiagnosticValue?)...
    ) -> [String: AIMEDiagnosticValue] {
        var result: [String: AIMEDiagnosticValue] = [:]
        for (key, value) in entries {
            if let value {
                result[key] = value
            }
        }
        return result
    }

    public static func string(_ value: String?) -> AIMEDiagnosticValue? {
        guard let value, !value.isEmpty else { return nil }
        return .string(value)
    }

    public static func int(_ value: Int?) -> AIMEDiagnosticValue? {
        guard let value else { return nil }
        return .int(value)
    }

    public static func double(_ value: Double?) -> AIMEDiagnosticValue? {
        guard let value, value.isFinite else { return nil }
        return .double(value)
    }

    public static func bool(_ value: Bool?) -> AIMEDiagnosticValue? {
        guard let value else { return nil }
        return .bool(value)
    }

    public static func metricFields(
        from values: [String: JSONValue],
        prefix: String = ""
    ) -> [String: AIMEDiagnosticValue] {
        var fields: [String: AIMEDiagnosticValue] = [:]
        func addString(_ key: String, as outputKey: String? = nil) {
            if let value = values[key]?.stringValue, !value.isEmpty {
                fields[prefix + (outputKey ?? key)] = .string(value)
            }
        }
        func addInt(_ key: String, as outputKey: String? = nil) {
            if let value = values[key]?.intValue {
                fields[prefix + (outputKey ?? key)] = .int(value)
            }
        }
        func addDouble(_ key: String, as outputKey: String? = nil) {
            if let value = values[key]?.doubleValue, value.isFinite {
                fields[prefix + (outputKey ?? key)] = .double(value)
            }
        }
        func addBool(_ key: String, as outputKey: String? = nil) {
            if let value = values[key]?.boolValue {
                fields[prefix + (outputKey ?? key)] = .bool(value)
            }
        }

        addString("request_id")
        addString("session_id")
        addString("cache_source")
        addDouble("decode_tok_s")
        addDouble("display_decode_tok_s")
        addDouble("live_decode_tok_s")
        addDouble("sliding_decode_tok_s_first_32")
        addDouble("sliding_decode_tok_s_first_64")
        addDouble("sliding_decode_tok_s_first_128")
        addDouble("sliding_decode_tok_s_first_256")
        addDouble("sliding_decode_tok_s_last_32")
        addDouble("sliding_decode_tok_s_last_64")
        addDouble("sliding_decode_tok_s_last_128")
        addDouble("sliding_decode_tok_s_last_256")
        addDouble("prefill_tok_s")
        addDouble("cumulative_prefill_tok_s")
        addDouble("request_elapsed_s")
        addDouble("decode_elapsed_s")
        addDouble("elapsed_s")
        addInt("prompt_tokens")
        addInt("completion_tokens")
        addInt("generated_tokens")
        addInt("reasoning_tokens")
        addInt("answer_tokens")
        addInt("cached_tokens")
        addInt("dashboard_progress_published_events")
        addInt("dashboard_progress_throttled_events")
        addInt("dashboard_progress_last_completion_tokens")
        addDouble("dashboard_progress_decision_time_s")
        addDouble("dashboard_progress_registry_update_time_s")
        addDouble("dashboard_progress_rolling_update_time_s")
        addDouble("dashboard_progress_bus_publish_time_s")
        addBool("cache_hit")
        addBool("ssd_cache_hit")
        return fields
    }

    public static func inFlightFields(
        _ requests: [InFlightRequest],
        prefix: String = ""
    ) -> [String: AIMEDiagnosticValue] {
        var fields: [String: AIMEDiagnosticValue] = [
            prefix + "in_flight_count": .int(requests.count)
        ]
        let ids = requests.map(\.requestId).joined(separator: ",")
        if !ids.isEmpty {
            fields[prefix + "in_flight_ids"] = .string(ids)
        }
        let sessions = requests.compactMap(\.sessionId).joined(separator: ",")
        if !sessions.isEmpty {
            fields[prefix + "in_flight_session_ids"] = .string(sessions)
        }
        let promptTokens = requests.compactMap(\.promptTokens)
        if !promptTokens.isEmpty {
            fields[prefix + "in_flight_prompt_tokens"] = .string(promptTokens.map(String.init).joined(separator: ","))
        }
        return fields
    }
}

private final class AIMEDiagnosticSampleGate: @unchecked Sendable {
    private let lock = NSLock()
    private var lastRecordUptime: [String: TimeInterval] = [:]
    private var recordedMilestones: [String: Set<Int>] = [:]

    func shouldRecord(
        name: String,
        intervalS: TimeInterval,
        tokenCount: Int?,
        identity: String?,
        force: Bool
    ) -> Bool {
        if force { return true }
        let key = identity.map { "\(name):\($0)" } ?? name
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        defer { lock.unlock() }

        if let tokenCount, Self.isMilestone(tokenCount) {
            var milestones = recordedMilestones[key] ?? []
            if !milestones.contains(tokenCount) {
                milestones.insert(tokenCount)
                recordedMilestones[key] = milestones
                lastRecordUptime[key] = now
                return true
            }
        }

        let last = lastRecordUptime[key] ?? -.infinity
        if now - last >= intervalS {
            lastRecordUptime[key] = now
            return true
        }
        return false
    }

    private static func isMilestone(_ tokenCount: Int) -> Bool {
        switch tokenCount {
        case 1, 32, 64, 128, 256:
            return true
        default:
            return tokenCount > 0 && tokenCount.isMultiple(of: 256)
        }
    }
}

private actor AIMEDiagnosticJSONLWriter {
    static let shared = AIMEDiagnosticJSONLWriter()

    private var fileURL: URL?
    private var handle: FileHandle?
    private var buffer = Data()
    private var bufferedEvents = 0
    private var lastFlushUptime = ProcessInfo.processInfo.systemUptime
    private let encoder = JSONEncoder()
    private let maxBufferedEvents = 128
    private let maxBufferedBytes = 256 * 1024
    private let flushIntervalS: TimeInterval = 1

    func write(_ event: AIMEDiagnosticEvent) {
        do {
            var data = try encoder.encode(event)
            data.append(contentsOf: Data("\n".utf8))
            buffer.append(data)
            bufferedEvents += 1
            let now = ProcessInfo.processInfo.systemUptime
            if bufferedEvents >= maxBufferedEvents
                || buffer.count >= maxBufferedBytes
                || now - lastFlushUptime >= flushIntervalS {
                try flush()
            }
        } catch {
            AIMEDiagnostics.logger.error("AIME diagnostics write failed: \(String(describing: type(of: error)), privacy: .public)")
        }
    }

    func flushPending() {
        do {
            try flush()
        } catch {
            AIMEDiagnostics.logger.error("AIME diagnostics flush failed: \(String(describing: type(of: error)), privacy: .public)")
        }
    }

    private func flush() throws {
        guard !buffer.isEmpty else { return }
        let handle = try ensureHandle()
        try handle.seekToEnd()
        handle.write(buffer)
        buffer.removeAll(keepingCapacity: true)
        bufferedEvents = 0
        lastFlushUptime = ProcessInfo.processInfo.systemUptime
    }

    private func ensureHandle() throws -> FileHandle {
        if let handle { return handle }
        let url = try ensureFileURL()
        let opened = try FileHandle(forWritingTo: url)
        handle = opened
        return opened
    }

    private func ensureFileURL() throws -> URL {
        if let fileURL { return fileURL }

        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let directory = base
            .appendingPathComponent("MTPLX", isDirectory: true)
            .appendingPathComponent("Diagnostics", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"

        let url = directory.appendingPathComponent("aime-\(formatter.string(from: Date())).jsonl")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        fileURL = url
        AIMEDiagnostics.logger.info("AIME diagnostics JSONL created")
        return url
    }
}
