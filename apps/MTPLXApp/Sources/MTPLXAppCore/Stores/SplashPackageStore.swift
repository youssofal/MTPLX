import Foundation

/// Splash's model packages, as Settings sees them.
///
/// Splash loads only its own packages and they are large (17–21 GB), so the
/// app must be able to say which are on disk and fetch one on request rather
/// than discovering the gap at daemon launch. Every operation goes through
/// `mtplx splash`, never Splash's private libexec layout: the CLI is the
/// supported seam, and it delegates the actual download and manifest
/// verification to Splash's own installer.
@MainActor
public final class SplashPackageStore: ObservableObject {

    public struct Package: Decodable, Identifiable, Equatable, Sendable {
        public let id: String
        public let installed: Bool
        public let path: String
        public let downloadBytes: Int?

        enum CodingKeys: String, CodingKey {
            case id, installed, path
            case downloadBytes = "download_bytes"
        }

        /// "17.4 GB", for a row that has not been downloaded yet.
        public var downloadSizeLabel: String? {
            guard let bytes = downloadBytes, bytes > 0 else { return nil }
            return String(format: "%.1f GB", Double(bytes) / 1e9)
        }
    }

    private struct Listing: Decodable {
        let ok: Bool
        let version: String?
        let packages: [Package]?
        let error: String?
    }

    @Published public private(set) var packages: [Package] = []
    @Published public private(set) var version: String = ""
    @Published public private(set) var refreshing: Bool = false
    /// The package currently downloading, if any.
    @Published public private(set) var installing: String?
    /// The installer's most recent output line, for a live progress caption.
    @Published public private(set) var progressLine: String = ""
    @Published public private(set) var errorMessage: String?
    /// False when Splash itself is not installed; the UI then points at brew.
    @Published public private(set) var splashAvailable: Bool = true

    private var task: Task<Void, Never>?
    private var running: Process?

    public init() {}

    public func isInstalled(_ modelID: String) -> Bool {
        packages.first { $0.id == modelID }?.installed ?? false
    }

    public func package(_ modelID: String) -> Package? {
        packages.first { $0.id == modelID }
    }

    // MARK: - listing

    public func refresh() async {
        guard !refreshing, installing == nil else { return }
        refreshing = true
        defer { refreshing = false }
        do {
            let output = try await Self.run(arguments: ["splash", "list", "--json"])
            guard let data = Self.lastJSONObject(in: output) else {
                errorMessage = "Could not read the Splash package list."
                return
            }
            let listing = try JSONDecoder().decode(Listing.self, from: data)
            if listing.ok {
                packages = listing.packages ?? []
                version = listing.version ?? ""
                splashAvailable = true
                errorMessage = nil
            } else {
                // The one expected failure is "Splash is not installed"; the
                // CLI already phrases it with the brew command to run.
                splashAvailable = false
                errorMessage = listing.error
                packages = []
            }
        } catch {
            splashAvailable = false
            errorMessage = error.localizedDescription
            packages = []
        }
    }

    // MARK: - install / update

    /// Download and verify a package, streaming progress into `progressLine`.
    ///
    /// `force` re-runs against an already-verified package, which is how an
    /// update is applied: Splash's installer re-checks every artifact against
    /// the manifest and refetches whatever no longer matches.
    public func install(_ modelID: String, force: Bool = false) {
        guard installing == nil else { return }
        installing = modelID
        progressLine = force ? "Verifying…" : "Starting download…"
        errorMessage = nil
        var arguments = ["splash", "install", "--model", modelID]
        if force { arguments.append("--force") }
        task = Task { [self] in
            do {
                try await Self.stream(
                    arguments: arguments,
                    onLine: { line in
                        Task { @MainActor in self.progressLine = line }
                    },
                    register: { process in
                        Task { @MainActor in self.running = process }
                    }
                )
            } catch {
                errorMessage = error.localizedDescription
            }
            installing = nil
            running = nil
            progressLine = ""
            await refresh()
        }
    }

    public func cancelInstall() {
        running?.terminate()
        task?.cancel()
        installing = nil
        running = nil
        progressLine = ""
    }

    // MARK: - subprocess

    /// The last top-level JSON object in mixed output.
    ///
    /// Splash's installer prints human progress on the same stream, so the
    /// machine-readable payload is the final line, not the whole buffer.
    nonisolated static func lastJSONObject(in output: String) -> Data? {
        for line in output.split(separator: "\n").reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}") else { continue }
            if let data = trimmed.data(using: .utf8) { return data }
        }
        return nil
    }

    private static func run(arguments: [String]) async throws -> String {
        let sink = OutputSink()
        try await stream(
            arguments: arguments,
            onLine: { sink.append($0) },
            register: { _ in }
        )
        return sink.text()
    }

    private nonisolated static func stream(
        arguments: [String],
        onLine: @escaping @Sendable (String) -> Void,
        register: @escaping @Sendable (Process) -> Void
    ) async throws {
        let executable = try MTPLXCommandBuilder.resolveInstalledExecutable(
            allowDevelopmentWrapper: true
        )
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            let buffer = LineBuffer(onLine: onLine)
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if chunk.isEmpty { return }
                buffer.append(chunk)
            }
            process.terminationHandler = { finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                buffer.flush()
                if finished.terminationStatus == 0 {
                    continuation.resume()
                } else {
                    continuation.resume(
                        throwing: SplashPackageError.failed(
                            status: finished.terminationStatus,
                            detail: buffer.tail()
                        )
                    )
                }
            }
            do {
                try process.run()
                register(process)
            } catch {
                pipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume(throwing: error)
            }
        }
    }
}

/// Collects subprocess output from the reader thread for the caller.
private final class OutputSink: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""

    func append(_ line: String) {
        lock.lock()
        buffer += line + "\n"
        lock.unlock()
    }

    func text() -> String {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }
}


public enum SplashPackageError: LocalizedError {
    case failed(status: Int32, detail: String)

    public var errorDescription: String? {
        switch self {
        case let .failed(status, detail):
            let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty
                ? "Splash package command failed (exit \(status))."
                : trimmed
        }
    }
}

/// Splits subprocess output into lines.
///
/// Hugging Face draws its transfer bars with carriage returns rather than
/// newlines, so a newline-only split would show nothing at all for the hour a
/// 17 GB package takes. Both terminators end a line here.
private final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()
    private var recent: [String] = []
    private let onLine: @Sendable (String) -> Void

    init(onLine: @escaping @Sendable (String) -> Void) {
        self.onLine = onLine
    }

    func append(_ chunk: Data) {
        lock.lock()
        pending.append(chunk)
        var lines: [String] = []
        while let index = pending.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
            let raw = pending.subdata(in: pending.startIndex..<index)
            pending.removeSubrange(pending.startIndex...index)
            if let text = String(data: raw, encoding: .utf8) {
                let trimmed = text.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty { lines.append(trimmed) }
            }
        }
        recent.append(contentsOf: lines)
        if recent.count > 40 { recent.removeFirst(recent.count - 40) }
        lock.unlock()
        lines.forEach(onLine)
    }

    func flush() {
        lock.lock()
        let raw = pending
        pending = Data()
        lock.unlock()
        if let text = String(data: raw, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { onLine(trimmed) }
        }
    }

    func tail() -> String {
        lock.lock()
        defer { lock.unlock() }
        return recent.suffix(6).joined(separator: "\n")
    }
}
