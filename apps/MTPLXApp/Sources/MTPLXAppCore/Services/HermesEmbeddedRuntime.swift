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

    public init(launchID: String, pid: Int32, parentPID: Int32, profileName: String, createdAt: Date) {
        self.launchID = launchID
        self.pid = pid
        self.parentPID = parentPID
        self.profileName = profileName
        self.createdAt = createdAt
    }
}

public struct HermesSidecarProcessSnapshot: Equatable, Sendable {
    public let pid: Int32
    public let arguments: [String]

    public init(pid: Int32, arguments: [String]) {
        self.pid = pid
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
/// record, exact PID, dead recorded parent, and the two exact argument pairs
/// are all required before the caller sends a signal.
enum HermesOrphanSidecarScanner {
    static func orphanPIDs(
        records: [HermesSidecarOwnershipRecord],
        processes: [HermesSidecarProcessSnapshot],
        livePIDs: Set<Int32>
    ) -> [Int32] {
        let processesByPID = Dictionary(uniqueKeysWithValues: processes.map { ($0.pid, $0) })
        return records.compactMap { record in
            guard record.pid > 1,
                  record.parentPID > 1,
                  !livePIDs.contains(record.parentPID),
                  let process = processesByPID[record.pid],
                  isExactOwnedSidecar(process, launchID: record.launchID)
            else { return nil }
            return record.pid
        }
        .sorted()
    }

    static func isExactOwnedSidecar(
        _ process: HermesSidecarProcessSnapshot,
        launchID: String
    ) -> Bool {
        let arguments = process.arguments
        guard let serveIndex = arguments.firstIndex(of: "serve"),
              arguments.indices.contains(serveIndex + 1),
              arguments[serveIndex + 1] == "--isolated",
              arguments.filter({ $0 == "serve" }).count == 1,
              arguments.filter({ $0 == "--isolated" }).count == 1,
              let markerIndex = arguments.firstIndex(of: "--ssh-owner-nonce"),
              arguments.indices.contains(markerIndex + 1),
              arguments[markerIndex + 1] == launchID,
              arguments.filter({ $0 == "--ssh-owner-nonce" }).count == 1
        else { return false }
        return true
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
    case appOwned
    case external
    case unavailable(String)
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
