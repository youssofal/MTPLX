import Foundation

// MARK: - DiscoveryEntry

public extension DiscoveryEntry {
    /// Multi-token, separator-insensitive match against branded name, owner, and repo ID.
    func matchesSearchQuery(_ query: String) -> Bool {
        let tokens = query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
        guard !tokens.isEmpty else { return true }
        let haystacks = [
            brandedName.lowercased(),
            owner.lowercased(),
            repo.lowercased(),
        ]
        let collapsedHaystacks = haystacks.map {
            $0.components(separatedBy: CharacterSet(charactersIn: "-_./ ")).joined()
        }
        for rawToken in tokens {
            let token = String(rawToken)
            if haystacks.contains(where: { $0.contains(token) }) {
                continue
            }
            let collapsedToken = token
                .components(separatedBy: CharacterSet(charactersIn: "-_./"))
                .joined()
            if collapsedToken.isEmpty
                || !collapsedHaystacks.contains(where: { $0.contains(collapsedToken) })
            {
                return false
            }
        }
        return true
    }
}

public struct DiscoveryEntry: Identifiable, Equatable, Sendable {
    public var id: String { repo }
    public var repo: String
    public var owner: String
    public var brandedName: String
    public var downloads: Int
    public var sizeBytes: Int64?
    public var depth: Int?
    public var multiplierVsAr: Double?
    public var license: String?
    public var lastUpdated: String?

    public init(
        repo: String,
        owner: String,
        brandedName: String,
        downloads: Int,
        sizeBytes: Int64? = nil,
        depth: Int? = nil,
        multiplierVsAr: Double? = nil,
        license: String? = nil,
        lastUpdated: String? = nil
    ) {
        self.repo = repo
        self.owner = owner
        self.brandedName = brandedName
        self.downloads = downloads
        self.sizeBytes = sizeBytes
        self.depth = depth
        self.multiplierVsAr = multiplierVsAr
        self.license = license
        self.lastUpdated = lastUpdated
    }
}

// MARK: - ForgeDiscoveryError

public enum ForgeDiscoveryError: Error, Equatable, Sendable {
    case backendNotAvailable
    case hfUnreachable
    case malformedResponse(String)
    case subprocessFailed(exitCode: Int32?, stderrTail: String)
}

// MARK: - ForgeDiscoveryService
//
// Wraps `mtplx forge discover --json [--query …] [--limit N]
// [--offset N]`. Backend queries HF's list_models endpoint sorted by
// `downloads` descending and keeps every repo whose NAME contains
// "MTPLX" (any case, any position — Forge brands `<base>-MTPLX`, and
// community naming varies placement and case freely). NO curated
// allow-list and no architecture gate — the brand name is the
// discovery signal; compatibility is still checked where it always
// was, at install/load time.
//
// On HF-unreachable conditions (DNS down, captive portal, etc.) the
// backend exits with a recognisable error string and we surface
// `.hfUnreachable` so the wall can render an explicit "no
// connection" empty state instead of a generic failure.

public struct ForgeDiscoveryService: Sendable {
    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.processEnvironment = processEnvironment
    }

    private let processEnvironment: [String: String]

    public struct Query: Equatable, Sendable {
        public var search: String?
        public var limit: Int
        public var offset: Int

        /// Default limit = the backend's per-call cap. The wall has no
        /// pagination affordance, so this IS the wall: at 30 the fresh
        /// low-download models (the 3.8 FP16 siblings on drop day, most
        /// new community forges) ranked below the fold and simply never
        /// existed for the user. 100 covers the live MTPLX result set
        /// (109 repos, 2026-08-16) minus a single-digit tail.
        public init(search: String? = nil, limit: Int = 500, offset: Int = 0) {
            self.search = search
            self.limit = limit
            self.offset = offset
        }
    }

    public func discover(_ query: Query = Query()) async throws -> [DiscoveryEntry] {
        let executable = try ForgeBuilder.resolveMtplxExecutable(env: processEnvironment)

        var args: [String] = ["forge", "discover", "--json", "--limit", String(query.limit)]
        if query.offset > 0 {
            args.append(contentsOf: ["--offset", String(query.offset)])
        }
        if let search = query.search, !search.isEmpty {
            args.append(contentsOf: ["--query", search])
        }

        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
            environment: processEnvironment
        )

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            throw ForgeDiscoveryError.backendNotAvailable
        }

        // Drain both pipes concurrently and bound the wait (#158): the
        // old sequential readToEnd() pair deadlocked once a traceback
        // filled stderr's 64KB pipe buffer while stdout was still open,
        // and a wedged HF connection hung the wall forever. stdout is
        // the JSON payload, so both sides use the lossless drain.
        let stdoutDrain = SubprocessPipeDrain(outPipe)
        let stderrDrain = SubprocessPipeDrain(errPipe)
        let timeout: TimeInterval = 120
        guard watchdog.wait(for: process, timeout: timeout) else {
            stderrDrain.join(timeout: 1)
            throw ForgeDiscoveryError.subprocessFailed(
                exitCode: -2,
                stderrTail: stderrDrain.snapshot()
                    + "\n[forge discover timed out after \(Int(timeout))s and was terminated]"
            )
        }
        stdoutDrain.join()
        stderrDrain.join()
        let outData = stdoutDrain.snapshotData()

        let stderrText = stderrDrain.snapshot()

        if process.terminationStatus == 2,
           stderrText.range(of: "invalid choice", options: .caseInsensitive) != nil,
           stderrText.range(of: "forge", options: .caseInsensitive) != nil
        {
            throw ForgeDiscoveryError.backendNotAvailable
        }

        if process.terminationStatus != 0 {
            if stderrText.range(of: "hf_unreachable", options: .caseInsensitive) != nil
                || stderrText.range(of: "name resolution failed", options: .caseInsensitive) != nil
                || stderrText.range(of: "connection refused", options: .caseInsensitive) != nil
            {
                throw ForgeDiscoveryError.hfUnreachable
            }
            throw ForgeDiscoveryError.subprocessFailed(
                exitCode: process.terminationStatus,
                stderrTail: stderrText
            )
        }

        guard let array = try? JSONSerialization.jsonObject(with: outData) as? [[String: Any]] else {
            throw ForgeDiscoveryError.malformedResponse("Expected JSON array at stdout")
        }
        return array.compactMap(Self.parseEntry)
    }

    static func parseEntry(_ json: [String: Any]) -> DiscoveryEntry? {
        guard let repo = json["repo"] as? String, !repo.isEmpty else { return nil }
        let owner = (json["owner"] as? String)
            ?? repo.split(separator: "/").first.map(String.init)
            ?? ""
        let brandedName = (json["branded_name"] as? String)
            ?? repo.split(separator: "/").last.map(String.init)
            ?? repo
        let downloads = (json["downloads"] as? Int) ?? 0
        let sizeBytes = (json["size_bytes"] as? Int64)
            ?? (json["size_bytes"] as? Int).map(Int64.init)
        let depth = json["depth"] as? Int
            ?? (json["verification"] as? [String: Any])?["depth"] as? Int
        let multiplier = json["multiplier_vs_ar"] as? Double
            ?? (json["verification"] as? [String: Any])?["multiplier_vs_ar"] as? Double
        let license = json["license"] as? String
        let lastUpdated = json["last_updated"] as? String
        return DiscoveryEntry(
            repo: repo,
            owner: owner,
            brandedName: brandedName,
            downloads: downloads,
            sizeBytes: sizeBytes,
            depth: depth,
            multiplierVsAr: multiplier,
            license: license,
            lastUpdated: lastUpdated
        )
    }
}
