import Foundation

public enum HermesProfileRoutingState: Equatable, Sendable {
    case mtplx
    case external
    case unavailable(String)
}

public struct HermesServeLaunchSpec: Equatable, Sendable {
    public let executableURL: URL
    public let arguments: [String]
    public let environment: [String: String]
    public let token: String
    public let launchID: String
    public let parentPID: Int32

    public init(
        executableURL: URL,
        arguments: [String],
        environment: [String: String],
        token: String,
        launchID: String,
        parentPID: Int32
    ) {
        self.executableURL = executableURL
        self.arguments = arguments
        self.environment = environment
        self.token = token
        self.launchID = launchID
        self.parentPID = parentPID
    }
}

public protocol HermesSidecarControlling: AnyObject, Sendable {
    var processIdentifier: Int32 { get }
    var isRunning: Bool { get }
    var webSocketURL: URL { get }
    var ownershipRecordURL: URL { get }
    func stop()
}

public struct HermesSidecarOwnershipRecord: Codable, Equatable, Sendable {
    public let launchID: String
    public let pid: Int32
    public let parentPID: Int32
    public let profileName: String
    public let createdAt: Date
    public let executablePath: String
    public let argv0: String
    public let arguments: [String]

    public init(
        launchID: String,
        pid: Int32,
        parentPID: Int32,
        profileName: String,
        createdAt: Date,
        executablePath: String = "",
        argv0: String? = nil,
        arguments: [String] = []
    ) {
        self.launchID = launchID
        self.pid = pid
        self.parentPID = parentPID
        self.profileName = profileName
        self.createdAt = createdAt
        self.executablePath = executablePath
        self.argv0 = argv0 ?? executablePath
        self.arguments = arguments
    }

    private enum CodingKeys: String, CodingKey {
        case launchID, pid, parentPID, profileName, createdAt, executablePath, argv0, arguments
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        launchID = try container.decode(String.self, forKey: .launchID)
        pid = try container.decode(Int32.self, forKey: .pid)
        parentPID = try container.decode(Int32.self, forKey: .parentPID)
        profileName = try container.decode(String.self, forKey: .profileName)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        executablePath = try container.decodeIfPresent(String.self, forKey: .executablePath) ?? ""
        argv0 = try container.decodeIfPresent(String.self, forKey: .argv0) ?? executablePath
        arguments = try container.decodeIfPresent([String].self, forKey: .arguments) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(launchID, forKey: .launchID)
        try container.encode(pid, forKey: .pid)
        try container.encode(parentPID, forKey: .parentPID)
        try container.encode(profileName, forKey: .profileName)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(executablePath, forKey: .executablePath)
        try container.encode(argv0, forKey: .argv0)
        try container.encode(arguments, forKey: .arguments)
    }
}

public struct HermesSidecarProcessSnapshot: Equatable, Sendable {
    public let pid: Int32
    public let executablePath: String
    public let argv0: String
    public let arguments: [String]

    public init(
        pid: Int32,
        executablePath: String = "",
        argv0: String? = nil,
        arguments: [String]
    ) {
        self.pid = pid
        self.executablePath = executablePath
        self.argv0 = argv0 ?? executablePath
        self.arguments = arguments
    }
}

enum HermesBackendReadyParser {
    static func port(from line: String) -> Int? {
        let prefix = "HERMES_BACKEND_READY port="
        guard line.hasPrefix(prefix),
              let port = Int(line.dropFirst(prefix.count)),
              (1...65_535).contains(port)
        else { return nil }
        return port
    }
}

/// Pure command matching used by orphan recovery.  It intentionally does not
/// attempt to infer ownership from a generic Hermes command: a persisted MTPLX
/// record, exact PID, dead recorded parent, canonical executable, and complete
/// argument vector are all required before the caller sends a signal.
enum HermesOrphanSidecarScanner {
    static func orphanPIDs(
        records: [HermesSidecarOwnershipRecord],
        processes: [HermesSidecarProcessSnapshot],
        livePIDs: Set<Int32>
    ) -> [Int32] {
        let processesByPID = Dictionary(grouping: processes, by: \.pid)
        let recordsByPID = Dictionary(grouping: records, by: \.pid)
        return records.compactMap { record in
            guard record.pid > 1,
                  record.parentPID > 1,
                  !livePIDs.contains(record.parentPID),
                  recordsByPID[record.pid]?.count == 1,
                  let snapshots = processesByPID[record.pid],
                  snapshots.count == 1,
                  let process = snapshots.first,
                  isExactOwnedSidecar(process, record: record)
            else { return nil }
            return record.pid
        }
        .sorted()
    }

    static func isExactOwnedSidecar(
        _ process: HermesSidecarProcessSnapshot,
        record: HermesSidecarOwnershipRecord
    ) -> Bool {
        guard !record.executablePath.isEmpty,
              process.executablePath == record.executablePath,
              !record.argv0.isEmpty,
              process.argv0 == record.argv0,
              process.arguments == record.arguments
        else { return false }
        return hasCanonicalArguments(
            record.arguments,
            profileName: record.profileName,
            launchID: record.launchID
        )
    }

    static func hasCanonicalArguments(
        _ arguments: [String],
        profileName: String,
        launchID: String
    ) -> Bool {
        let profilePrefix = profileName == "default" ? [] : ["-p", profileName]
        return arguments == profilePrefix + [
            "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", launchID,
        ]
    }
}

/// Redacts secrets while bytes arrive from stderr.  It never emits a suffix
/// that could still be the beginning of a secret, so a token split across pipe
/// callbacks cannot enter the diagnostic tail as separate visible fragments.
final class HermesStreamingSecretRedactor: @unchecked Sendable {
    private let lock = NSLock()
    private let secrets: [Data]
    private var pending = Data()
    private let replacement = Data("[redacted]".utf8)

    init(secrets: [String]) {
        self.secrets = Array(Set(secrets.filter { !$0.isEmpty })).map { Data($0.utf8) }
    }

    func redact(_ chunk: Data) -> Data {
        lock.lock()
        defer { lock.unlock() }
        pending.append(chunk)
        var output = Data()
        while true {
            if let match = earliestSecretMatch(in: pending) {
                output.append(contentsOf: pending[..<match.lowerBound])
                output.append(replacement)
                pending.removeSubrange(..<match.upperBound)
                continue
            }
            let retainedCount = longestSecretPrefixSuffix(in: pending)
            let emittedCount = pending.count - retainedCount
            if emittedCount > 0 {
                output.append(contentsOf: pending.prefix(emittedCount))
                pending.removeFirst(emittedCount)
            }
            return output
        }
    }

    private func earliestSecretMatch(in data: Data) -> Range<Data.Index>? {
        secrets.reduce(nil) { current, secret in
            guard let candidate = data.range(of: secret) else { return current }
            guard let current else { return candidate }
            if candidate.lowerBound < current.lowerBound { return candidate }
            if candidate.lowerBound == current.lowerBound,
               data.distance(from: candidate.lowerBound, to: candidate.upperBound)
                    > data.distance(from: current.lowerBound, to: current.upperBound) {
                return candidate
            }
            return current
        }
    }

    private func longestSecretPrefixSuffix(in data: Data) -> Int {
        guard !data.isEmpty else { return 0 }
        var best = 0
        for secret in secrets where secret.count > 1 {
            let maximum = min(secret.count - 1, data.count)
            guard maximum > best else { continue }
            for length in stride(from: maximum, through: best + 1, by: -1) {
                if Array(data.suffix(length)) == Array(secret.prefix(length)) {
                    best = length
                    break
                }
            }
        }
        return best
    }
}

/// Thread-safe readiness state shared by pipe reader callbacks and the
/// bounded launcher worker.  Sentinel parsing is deliberately the only source
/// of a port; prose emitted by Hermes can never become a connection target.
final class HermesSidecarReadiness: @unchecked Sendable {
    private let lock = NSLock()
    private var remainder = ""
    private var readyPort: Int?
    private var malformedSentinel = false

    func consume(_ data: Data) {
        let text = String(decoding: data, as: UTF8.self)
        lock.lock()
        remainder.append(text)
        let lines = remainder.split(separator: "\n", omittingEmptySubsequences: false)
        if remainder.hasSuffix("\n") {
            remainder = ""
        } else {
            remainder = lines.last.map(String.init) ?? ""
        }
        let completeLines = remainder.isEmpty ? lines : lines.dropLast()
        for line in completeLines {
            let string = String(line).trimmingCharacters(in: .newlines)
            guard string.hasPrefix("HERMES_BACKEND_READY port=") else { continue }
            if let port = HermesBackendReadyParser.port(from: string) {
                readyPort = port
            } else {
                malformedSentinel = true
            }
        }
        lock.unlock()
    }

    func result() -> (port: Int?, malformed: Bool) {
        lock.lock()
        defer { lock.unlock() }
        return (readyPort, malformedSentinel)
    }
}

public enum HermesSessionOwnership: Equatable, Sendable {
    case ready
    case ownedByMTPLX
    case external(surface: String)
    case unknown(String)
}

public enum HermesSessionActivityState: Equatable, Sendable {
    case ready
    case runningInMTPLX
    case externallyActive(surface: String)
    case ownershipUnknown(String)

    init(ownership: HermesSessionOwnership) {
        switch ownership {
        case .ready: self = .ready
        case .ownedByMTPLX: self = .runningInMTPLX
        case .external(let surface): self = .externallyActive(surface: surface)
        case .unknown(let reason): self = .ownershipUnknown(reason)
        }
    }
}

/// Process observations used only to decide whether an active-session entry is
/// still trustworthy.  The inspector never signals a process and never writes
/// the Hermes registry.
public enum HermesSessionProcessIdentity: Equatable, Sendable {
    case live(startedAt: TimeInterval)
    case dead
    case unknown
}

/// Read-only, fail-closed parser for Hermes' per-profile active-session
/// registry.  Its process probe is injected so registry semantics can be
/// verified without probing real processes in tests.
public struct HermesActiveSessionRegistryInspector: @unchecked Sendable {
    public typealias ProcessIdentity = @Sendable (Int32) -> HermesSessionProcessIdentity

    private let processIdentity: ProcessIdentity
    private let readData: @Sendable (URL) throws -> Data

    public init(
        processIdentity: @escaping ProcessIdentity = HermesActiveSessionRegistryInspector.liveProcessIdentity,
        readData: @escaping @Sendable (URL) throws -> Data = { try Data(contentsOf: $0) }
    ) {
        self.processIdentity = processIdentity
        self.readData = readData
    }

    public func ownership(
        registryURL: URL,
        sessionID: String,
        ownedSidecarPID: Int32?
    ) -> HermesSessionOwnership {
        guard !sessionID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .unknown(Self.inspectionUnavailableReason)
        }
        let data: Data
        do {
            data = try readData(registryURL)
        } catch {
            if Self.isNotFound(error) { return .ready }
            return .unknown(Self.inspectionUnavailableReason)
        }

        let entries: [Entry]
        do {
            entries = try JSONDecoder().decode(Registry.self, from: data).entries
        } catch {
            return .unknown(Self.inspectionUnavailableReason)
        }

        let matching = entries.filter { $0.sessionID == sessionID }
        guard !matching.isEmpty else { return .ready }

        var liveEntries: [Entry] = []
        for entry in matching {
            switch processIdentity(entry.pid) {
            case .dead:
                continue
            case .unknown:
                return .unknown(Self.inspectionUnavailableReason)
            case .live(let processStart):
                // A live PID without the recorded start identity is a reused
                // PID, not an active Hermes writer for this session.
                guard abs(processStart - entry.startedAt) < 0.001 else { continue }
                liveEntries.append(entry)
            }
        }

        guard liveEntries.count <= 1 else {
            return .unknown(Self.inspectionUnavailableReason)
        }
        guard let entry = liveEntries.first else { return .ready }
        if entry.pid == ownedSidecarPID {
            return .ownedByMTPLX
        }
        return .external(surface: Self.sanitizedSurface(entry.surface))
    }

    private static let inspectionUnavailableReason = "Session activity could not be inspected."

    private static func isNotFound(_ error: Error) -> Bool {
        let nsError = error as NSError
        return (nsError.domain == NSCocoaErrorDomain
            && (nsError.code == CocoaError.Code.fileNoSuchFile.rawValue
                || nsError.code == CocoaError.Code.fileReadNoSuchFile.rawValue))
            || (nsError.domain == NSPOSIXErrorDomain && nsError.code == ENOENT)
    }

    private static func sanitizedSurface(_ surface: String) -> String {
        let trimmed = surface.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_"))
        guard !trimmed.isEmpty,
              trimmed.unicodeScalars.allSatisfy(allowed.contains),
              trimmed.count <= 40
        else { return "another Hermes surface" }
        return trimmed
    }

    private struct Registry: Decodable {
        let entries: [Entry]
    }

    private struct Entry: Decodable {
        let sessionID: String
        let surface: String
        let pid: Int32
        let startedAt: TimeInterval

        private enum CodingKeys: String, CodingKey {
            case sessionID = "session_id"
            case surface
            case pid
            case startedAt = "process_start_time"
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            sessionID = try values.decode(String.self, forKey: .sessionID)
            surface = try values.decode(String.self, forKey: .surface)
            pid = try values.decode(Int32.self, forKey: .pid)
            startedAt = try values.decode(TimeInterval.self, forKey: .startedAt)
            guard !sessionID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  pid > 1,
                  startedAt.isFinite,
                  startedAt > 0
            else { throw DecodingError.dataCorruptedError(forKey: .sessionID, in: values, debugDescription: "Invalid active session entry.") }
        }
    }

    public static func liveProcessIdentity(_ pid: Int32) -> HermesSessionProcessIdentity {
        guard pid > 1 else { return .dead }
        if kill(pid, 0) != 0 {
            return errno == ESRCH ? .dead : .unknown
        }
        var info = proc_bsdinfo()
        let byteCount = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, Int32(MemoryLayout<proc_bsdinfo>.size))
        guard byteCount == MemoryLayout<proc_bsdinfo>.size else { return .unknown }
        let startedAt = TimeInterval(info.pbi_start_tvsec) + TimeInterval(info.pbi_start_tvusec) / 1_000_000
        return startedAt > 0 ? .live(startedAt: startedAt) : .unknown
    }
}

public protocol HermesEmbeddedRuntime: Sendable {
    func routingState(
        for profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) -> HermesProfileRoutingState
    func startEmbeddedSidecar(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> any HermesSidecarControlling
    func sessionOwnership(
        profile: HermesProfile,
        sessionID: String,
        ownedSidecarPID: Int32?
    ) -> HermesSessionOwnership
    @discardableResult func reapOrphanedEmbeddedSidecars() -> [Int32]
}
