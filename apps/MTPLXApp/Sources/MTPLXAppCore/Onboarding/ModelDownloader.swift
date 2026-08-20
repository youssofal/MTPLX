import Foundation

// MARK: - DownloadEvent
//
// One frame of progress emitted to the consumer (orchestrator → view).
// `progress` events fire every 500 ms while the subprocess is alive
// and the destination directory has changed size; `complete` /
// `failed` / `cancelled` are terminal.

public enum DownloadEvent: Sendable {
    /// The first event — emitted as soon as the subprocess is alive.
    /// Carries the on-disk destination path so the view can show
    /// "Downloading into ~/.mtplx/models/Foo--Bar".
    case started(path: String)
    case status(message: String, bytesOnDisk: Int64?, totalBytes: Int64?, path: String?)
    /// Periodic progress sample. `smoothedBytesPerSecond` is a 3-sample
    /// EMA so the displayed MB/s isn't jittery; `etaSeconds` is `nil`
    /// when either the total is unknown or smoothing is still cold.
    case progress(
        bytesOnDisk: Int64,
        totalBytes: Int64?,
        smoothedBytesPerSecond: Double,
        etaSeconds: Double?
    )
    /// No bytes have landed for the past `seconds` while the process
    /// is still alive — likely Hugging Face rate-limiting or network
    /// glitch. The view shows an amber warning but does NOT cancel.
    case stalled(seconds: Int)
    /// Subprocess exited cleanly. Final byte count from one last
    /// directory scan.
    case complete(bytesOnDisk: Int64, path: String)
    /// Subprocess exited non-zero or the process couldn't be launched.
    /// `stderrTail` is the last ~2 KB of stderr for the inline banner.
    case failed(exitCode: Int32?, stderrTail: String)
    /// Consumer explicitly cancelled the stream. Partial bytes remain
    /// on disk for resume.
    case cancelled
}

// MARK: - ModelDownloader
//
// Shells out to `mtplx pull <repo> --progress-json` and consumes the
// structured NDJSON stream. Directory polling remains as a fallback so
// older launchers still show honest movement, including hidden HF
// staging files.

public struct ModelDownloader: Sendable {
    private static let structuredStalledThresholdSeconds = 30

    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        modelCacheRoot: URL? = nil,
        pollInterval: TimeInterval = 0.5,
        stalledThreshold: TimeInterval = 30,
        executableOverride: URL? = nil
    ) {
        self.processEnvironment = processEnvironment
        self.modelCacheRoot = modelCacheRoot
        self.pollInterval = pollInterval
        self.stalledThreshold = stalledThreshold
        self.executableOverride = executableOverride
    }

    private let processEnvironment: [String: String]
    private let modelCacheRoot: URL?
    private let pollInterval: TimeInterval
    private let stalledThreshold: TimeInterval
    private let executableOverride: URL?

    // MARK: Public surface

    /// Streams a download of `repo` and yields `DownloadEvent`s.
    /// Cancelling the consumer task tears down the subprocess and
    /// fires a `.cancelled` event. Partial bytes survive on disk so
    /// a subsequent run resumes via `huggingface_hub`'s native Range
    /// support.
    /// Build the CLI invocation for a stream. Updates go through
    /// `models --update`, which pins to the published revision and handles
    /// legacy cache layouts; a plain `pull` reuses "fresh-looking" caches
    /// and silently no-ops on exactly the packs an update targets.
    static func streamArguments(
        repo: String,
        update: Bool,
        destinationPath: String? = nil
    ) -> [String] {
        guard update else { return ["pull", repo, "--progress-json"] }
        var arguments = ["models", "--update", repo, "--progress-json"]
        if let destinationPath, !destinationPath.isEmpty {
            arguments += ["--installed-path", destinationPath]
        }
        return arguments
    }

    public func stream(
        repo: String,
        totalBytes: Int64?,
        extraEnvironment: [String: String] = [:],
        update: Bool = false,
        sizeProbePath: String? = nil
    ) -> AsyncStream<DownloadEvent> {
        AsyncStream { continuation in
            let destination = sizeProbePath.map { URL(fileURLWithPath: $0) }
                ?? self.cachedModelPath(for: repo)
            // Make the destination dir up-front so the first poll returns 0
            // rather than spuriously matching "exists". Never for updates:
            // an empty canonical dir would shadow a populated legacy-layout
            // pack and turn the delta into a full re-download.
            if !update {
                try? FileManager.default.createDirectory(
                    at: destination,
                    withIntermediateDirectories: true
                )
            }
            let executable: URL
            do {
                executable = try self.resolveMtplxExecutable { message in
                    continuation.yield(.status(
                        message: message,
                        bytesOnDisk: Self.recursiveSize(of: destination),
                        totalBytes: totalBytes,
                        path: destination.path
                    ))
                }
                continuation.yield(.status(
                    message: "MTPLX runtime ready",
                    bytesOnDisk: Self.recursiveSize(of: destination),
                    totalBytes: totalBytes,
                    path: destination.path
                ))
            } catch {
                let message = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
                continuation.yield(.failed(exitCode: nil, stderrTail: message))
                continuation.finish()
                return
            }

            let process = Process()
            process.executableURL = executable
            process.arguments = Self.streamArguments(
                repo: repo,
                update: update,
                destinationPath: sizeProbePath
            )
            // Inherit a sensible PATH so Homebrew installs and wrappers can
            // find their helpers. Apply caller-owned download knobs first,
            // then pin Python's cache location so an override cannot send
            // bytecode back into the signed app bundle.
            var env = self.processEnvironment
            env["PATH"] = MTPLXCommandBuilder.expandedPATH(environment: self.processEnvironment)
            env.merge(extraEnvironment) { _, new in new }
            process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                environment: env
            )

            let errPipe = Pipe()
            let outPipe = Pipe()
            process.standardError = errPipe
            process.standardOutput = outPipe

            // Buffer the tail of stderr for failure messages without
            // pumping it back to the UI live (the user doesn't need
            // raw Python progress on a Swift bar).
            let stderrBuffer = StderrTailBuffer(capacity: 2048)
            let progressState = DownloadProgressJSONState(
                displayBaseBytes: Self.recursiveSize(of: destination)
            )
            let stdoutLines = LineBuffer()
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if !chunk.isEmpty {
                    stderrBuffer.append(chunk)
                }
            }
            outPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                guard !chunk.isEmpty else { return }
                for line in stdoutLines.append(chunk) {
                    for event in Self.events(
                        fromProgressJSONLine: line,
                        destination: destination,
                        fallbackTotalBytes: totalBytes,
                        state: progressState
                    ) {
                        continuation.yield(event)
                    }
                }
            }

            do {
                try process.run()
            } catch {
                continuation.yield(.failed(exitCode: nil, stderrTail: error.localizedDescription))
                continuation.finish()
                return
            }
            continuation.yield(.status(
                message: "Resolving files",
                bytesOnDisk: Self.recursiveSize(of: destination),
                totalBytes: totalBytes,
                path: destination.path
            ))

            // Spawn the polling task. Lives until the process exits or
            // the consumer cancels.
            let pollTask = Task.detached(priority: .userInitiated) { [pollInterval = self.pollInterval, stalledThreshold = self.stalledThreshold] in
                let smoother = ProgressSmoother(windowSize: 3)
                var lastSize: Int64 = 0
                var lastChangeAt = Date()
                while !Task.isCancelled, process.isRunning {
                    try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
                    if progressState.sawStructuredEvents {
                        continue
                    }
                    let bytes = Self.recursiveSize(of: destination)
                    if bytes != lastSize {
                        let now = Date()
                        let delta = bytes - lastSize
                        let dt = max(pollInterval, now.timeIntervalSince(lastChangeAt))
                        let smoothed = smoother.observe(bytesPerSecond: Double(delta) / dt)
                        let eta: Double?
                        if let total = totalBytes, total > 0, smoothed > 1024 {
                            eta = max(0, (Double(total) - Double(bytes)) / smoothed)
                        } else {
                            eta = nil
                        }
                        continuation.yield(.progress(
                            bytesOnDisk: bytes,
                            totalBytes: totalBytes,
                            smoothedBytesPerSecond: smoothed,
                            etaSeconds: eta
                        ))
                        lastSize = bytes
                        lastChangeAt = now
                    } else {
                        let stalledFor = Int(Date().timeIntervalSince(lastChangeAt))
                        if Double(stalledFor) >= stalledThreshold {
                            continuation.yield(.progress(
                                bytesOnDisk: bytes,
                                totalBytes: totalBytes,
                                smoothedBytesPerSecond: 0,
                                etaSeconds: nil
                            ))
                            continuation.yield(.stalled(seconds: stalledFor))
                        }
                    }
                }
            }

            let finishGate = OneShotGate()
            let finishProcess: @Sendable (Process) -> Void = { completedProcess in
                guard finishGate.claim() else { return }
                completedProcess.terminationHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                for line in stdoutLines.flush() {
                    for event in Self.events(
                        fromProgressJSONLine: line,
                        destination: destination,
                        fallbackTotalBytes: totalBytes,
                        state: progressState
                    ) {
                        continuation.yield(event)
                    }
                }
                pollTask.cancel()
                let finalBytes = Self.recursiveSize(of: destination)
                if progressState.sawTerminalEvent {
                    continuation.finish()
                    return
                }
                if process.terminationStatus == 0 {
                    continuation.yield(.complete(
                        bytesOnDisk: finalBytes,
                        path: destination.path
                    ))
                } else if process.terminationReason == .uncaughtSignal {
                    // SIGINT / SIGTERM — usually a cancel.
                    continuation.yield(.cancelled)
                } else {
                    continuation.yield(.failed(
                        exitCode: process.terminationStatus,
                        stderrTail: stderrBuffer.snapshot()
                    ))
                }
                continuation.finish()
            }
            process.terminationHandler = finishProcess

            continuation.onTermination = { @Sendable _ in
                if process.isRunning {
                    process.interrupt()
                }
                pollTask.cancel()
            }
        }
    }

    private static func events(
        fromProgressJSONLine line: String,
        destination: URL,
        fallbackTotalBytes: Int64?,
        state: DownloadProgressJSONState
    ) -> [DownloadEvent] {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawEvent = payload["event"] as? String
        else { return [] }

        state.markStructured()
        let event = rawEvent.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let path = (payload["path"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? destination.path
        let bytes = state.displayBytes(
            downloadedBytes: int64(payload["downloaded_bytes"]),
            fallback: int64(payload["size_bytes"])
                ?? int64(payload["bytes_on_disk"])
                ?? int64(payload["bytes"])
        )
        let total = int64(payload["total_bytes"]) ?? fallbackTotalBytes

        switch event {
        case "resolving":
            return [.status(message: "Resolving files", bytesOnDisk: bytes, totalBytes: total, path: path)]
        case "start", "resume":
            return [
                .status(message: "Downloading", bytesOnDisk: bytes ?? 0, totalBytes: total, path: path),
            ]
        case "progress":
            let rate = double(payload["rate_bps"]) ?? 0
            let eta: Double?
            if let total, let bytes, total > 0, rate > 1024 {
                eta = max(0, (Double(total) - Double(bytes)) / rate)
            } else {
                eta = nil
            }
            var events: [DownloadEvent] = [
                .progress(
                    bytesOnDisk: bytes ?? state.lastBytes ?? 0,
                    totalBytes: total,
                    smoothedBytesPerSecond: rate,
                    etaSeconds: eta
                ),
            ]
            state.observe(bytes: bytes, total: total)
            let stalled = Int(double(payload["stalled_s"]) ?? 0)
            if stalled >= structuredStalledThresholdSeconds {
                events.append(.stalled(seconds: stalled))
            }
            return events
        case "verifying":
            return [.status(message: "Verifying model files", bytesOnDisk: bytes, totalBytes: total, path: path)]
        case "complete":
            state.markTerminal()
            return [.complete(bytesOnDisk: bytes ?? Self.recursiveSize(of: destination), path: path)]
        case "result":
            guard let ok = payload["ok"] as? Bool else { return [] }
            state.markTerminal()
            if ok {
                return [.complete(bytesOnDisk: bytes ?? Self.recursiveSize(of: destination), path: path)]
            }
            let message = (payload["message"] as? String)
                ?? (payload["detail"] as? String)
                ?? (payload["error"] as? String)
                ?? "Download failed."
            return [.failed(exitCode: nil, stderrTail: message)]
        case "failed", "error":
            state.markTerminal()
            let message = (payload["message"] as? String)
                ?? (payload["detail"] as? String)
                ?? (payload["error"] as? String)
                ?? "Download failed."
            return [.failed(exitCode: nil, stderrTail: message)]
        case "cancelled", "interrupted":
            state.markTerminal()
            return [.cancelled]
        default:
            return []
        }
    }

    private static func int64(_ value: Any?) -> Int64? {
        switch value {
        case let value as Int:
            return Int64(value)
        case let value as Int64:
            return value
        case let value as Double:
            return Int64(value)
        case let value as String:
            return Int64(value)
        default:
            return nil
        }
    }

    private static func double(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Int:
            return Double(value)
        case let value as Int64:
            return Double(value)
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    // MARK: - Path helpers

    /// Mirrors `mtplx/hf_loader.py:cached_model_path` exactly so the
    /// directory we poll matches the directory `mtplx pull` writes to.
    public func cachedModelPath(for repo: String) -> URL {
        let root = modelCacheRoot ?? Self.defaultCacheRoot(env: processEnvironment)
        let safeName = repo.replacingOccurrences(of: "/", with: "--")
        return root.appendingPathComponent(safeName, isDirectory: true)
    }

    public static func defaultCacheRoot(env: [String: String]) -> URL {
        if let override = env["MTPLX_MODEL_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        }
        let home = env["HOME"] ?? NSHomeDirectory()
        return URL(fileURLWithPath: home)
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
    }

    /// Recursive sum of all regular file sizes under `url`. Returns 0
    /// if the directory doesn't exist yet (first poll, before HF
    /// writes anything).
    public static func recursiveSize(of url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: []
        ) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true {
                total += Int64(values?.fileSize ?? 0)
            }
        }
        return total
    }

    // MARK: - Executable resolution

    /// Runs `mtplx models --check --json` and decodes the per-pack update
    /// states. The model-pack counterpart of the Sparkle appcast fetch:
    /// network access and freshness logic live entirely in the CLI, this
    /// just shells and decodes. Safe to run while a daemon is serving.
    public func checkModelUpdates(
        timeoutSeconds: TimeInterval = 120
    ) async throws -> [ModelUpdateInfo] {
        let executable = try resolveMtplxExecutable { _ in }
        let process = Process()
        process.executableURL = executable
        process.arguments = ["models", "--check", "--json"]
        var env = processEnvironment
        env["PATH"] = MTPLXCommandBuilder.expandedPATH(environment: processEnvironment)
        process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
            environment: env
        )
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        try process.run()
        let watchdog = Task.detached(priority: .utility) {
            try? await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
            if process.isRunning {
                process.terminate()
            }
        }
        defer { watchdog.cancel() }
        let stdout = try await Task.detached(priority: .utility) { () -> Data in
            let data = outPipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return data
        }.value
        guard process.terminationStatus == 0 else {
            let stderr = errPipe.fileHandleForReading.readDataToEndOfFile()
            let tail = String(decoding: stderr.suffix(512), as: UTF8.self)
            throw NSError(
                domain: "ModelDownloader",
                code: Int(process.terminationStatus),
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "model update check exited \(process.terminationStatus): \(tail)"
                ]
            )
        }
        return try JSONDecoder().decode(ModelUpdateCheckPayload.self, from: stdout).models
    }

    private func resolveMtplxExecutable(
        status: @escaping @Sendable (String) -> Void
    ) throws -> URL {
        if let executableOverride,
           FileManager.default.isExecutableFile(atPath: executableOverride.path)
        {
            return executableOverride
        }
        return try MTPLXRuntimeBootstrapper(environment: processEnvironment).installOrUpdate(status: status)
    }
}

// MARK: - ProgressSmoother
//
// Tiny moving-average smoother for the bytes-per-second display.
// Operates as a class so it can survive across the polling closure
// without being copied; isolated to its owning Task so concurrency
// is implicitly safe.

private final class ProgressSmoother: @unchecked Sendable {
    private let windowSize: Int
    private var samples: [Double] = []

    init(windowSize: Int) {
        self.windowSize = windowSize
    }

    func observe(bytesPerSecond: Double) -> Double {
        samples.append(max(0, bytesPerSecond))
        if samples.count > windowSize {
            samples.removeFirst(samples.count - windowSize)
        }
        return samples.reduce(0, +) / Double(samples.count)
    }
}

private final class DownloadProgressJSONState: @unchecked Sendable {
    private let displayBaseBytes: Int64
    private let lock = NSLock()
    private var structured = false
    private var terminal = false
    private var observedBytes: Int64?
    private var observedTotal: Int64?

    init(displayBaseBytes: Int64) {
        self.displayBaseBytes = displayBaseBytes
    }

    func displayBytes(downloadedBytes: Int64?, fallback: Int64?) -> Int64? {
        downloadedBytes.map { displayBaseBytes + max(0, $0) } ?? fallback
    }

    var sawStructuredEvents: Bool {
        lock.lock()
        defer { lock.unlock() }
        return structured
    }

    var sawTerminalEvent: Bool {
        lock.lock()
        defer { lock.unlock() }
        return terminal
    }

    var lastBytes: Int64? {
        lock.lock()
        defer { lock.unlock() }
        return observedBytes
    }

    func markStructured() {
        lock.lock()
        structured = true
        lock.unlock()
    }

    func markTerminal() {
        lock.lock()
        terminal = true
        lock.unlock()
    }

    func observe(bytes: Int64?, total: Int64?) {
        lock.lock()
        if let bytes { observedBytes = bytes }
        if let total { observedTotal = total }
        lock.unlock()
    }
}

private final class OneShotGate: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !claimed else { return false }
        claimed = true
        return true
    }
}

private final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        data.append(chunk)
        return drainCompleteLines()
    }

    func flush() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        guard !data.isEmpty else { return [] }
        let line = String(data: data, encoding: .utf8) ?? ""
        data.removeAll(keepingCapacity: false)
        return line.isEmpty ? [] : [line]
    }

    private func drainCompleteLines() -> [String] {
        var lines: [String] = []
        while let index = data.firstIndex(of: 0x0A) {
            let lineData = data[..<index]
            if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                lines.append(line)
            }
            data.removeSubrange(...index)
        }
        return lines
    }
}

// MARK: - StderrTailBuffer
//
// Keeps the last N bytes of stderr around for the failure banner.
// Class-bound + locked so the readability handler (background thread)
// and the exit-Task snapshot can mutate / read safely.

private final class StderrTailBuffer: @unchecked Sendable {
    private let capacity: Int
    private var buffer = Data()
    private let lock = NSLock()

    init(capacity: Int) {
        self.capacity = capacity
    }

    func append(_ chunk: Data) {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(chunk)
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return String(data: buffer, encoding: .utf8) ?? ""
    }
}
