import Foundation

// MARK: - ForgePhase
//
// Names match the canonical per-phase file the backend writes to
// `<output-dir>/<run-id>/<phase>.json`. The frontend polls each file
// and yields per-phase events as their `progress` advances.

public enum ForgePhase: String, CaseIterable, Equatable, Sendable {
    case download
    case convert
    case calibrate
    case verify
    case brand
    case publish
}

// MARK: - ForgePhaseProgress
//
// Snapshot of one phase as seen on disk. Phase-specific subfields
// live on the typed enum; everything else (start/finish timestamps,
// optional auxiliary numbers) stays as raw JSON for the table
// renderer to surface.

public struct ForgePhaseProgress: Equatable, Sendable {
    public var phase: ForgePhase
    public var progress: Double           // 0...1 (best-effort; 0 when unknown)
    public var label: String?             // sub-phase the backend is currently in
    public var finished: Bool

    public init(phase: ForgePhase, progress: Double = 0, label: String? = nil, finished: Bool = false) {
        self.phase = phase
        self.progress = max(0, min(1, progress))
        self.label = label
        self.finished = finished
    }
}

// MARK: - ForgeDownloadProgress

public struct ForgeDownloadProgress: Equatable, Sendable {
    public var bytesOnDisk: Int64
    public var totalBytes: Int64?
    public var bytesPerSecond: Double
    public var etaSeconds: Double?
    public var label: String?
    public var stalledSeconds: Double?

    public init(
        bytesOnDisk: Int64,
        totalBytes: Int64?,
        bytesPerSecond: Double,
        etaSeconds: Double?,
        label: String? = nil,
        stalledSeconds: Double? = nil
    ) {
        self.bytesOnDisk = bytesOnDisk
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
        self.etaSeconds = etaSeconds
        self.label = label
        self.stalledSeconds = stalledSeconds
    }
}

// MARK: - ForgeVerifyRow
//
// One row from `verify.json`. Mirrors what `mtplx tune` writes for
// the equivalent ar.json/d1.json/d2.json/d3.json — exposed here as a
// pre-aggregated array so the frontend doesn't have to know which
// underlying file produced it.

public struct ForgeVerifyRow: Equatable, Sendable {
    public var depth: Int                  // 0 == AR baseline; 1/2/3 == MTP depth
    public var tokS: Double
    public var multiplierVsAr: Double
    public var acceptanceByPosition: [Double]
    public var verifyTimeSeconds: Double

    public init(
        depth: Int,
        tokS: Double,
        multiplierVsAr: Double,
        acceptanceByPosition: [Double],
        verifyTimeSeconds: Double
    ) {
        self.depth = depth
        self.tokS = tokS
        self.multiplierVsAr = multiplierVsAr
        self.acceptanceByPosition = acceptanceByPosition
        self.verifyTimeSeconds = verifyTimeSeconds
    }
}

// MARK: - ForgeBuildOutcome
//
// Terminal backend outcome for builds that converted successfully
// but did not become product-valid speed models. The backend writes
// this to build_outcome.json and intentionally does not write
// forge.json / mtplx_runtime.json unless MTP beats AR.

public struct ForgeBuildOutcome: Equatable, Sendable {
    public var convertedPath: String?
    public var phase: String
    public var verdict: String
    public var failureReasons: [String]
    public var message: String
    public var diagnostic: String?
    public var architectureID: String?
    public var arTokS: Double?
    public var bestMTPDepth: Int?
    public var bestMTPTokS: Double?
    public var bestMTPMultiplierVsAR: Double?
    public var verifyRows: [ForgeVerifyRow]

    public var isSpeedWin: Bool {
        verdict == "mtp_depth_wins"
    }

    public init(
        convertedPath: String? = nil,
        phase: String = "verify",
        verdict: String,
        failureReasons: [String] = [],
        message: String,
        diagnostic: String? = nil,
        architectureID: String? = nil,
        arTokS: Double? = nil,
        bestMTPDepth: Int? = nil,
        bestMTPTokS: Double? = nil,
        bestMTPMultiplierVsAR: Double? = nil,
        verifyRows: [ForgeVerifyRow] = []
    ) {
        self.convertedPath = convertedPath
        self.phase = phase
        self.verdict = verdict
        self.failureReasons = failureReasons
        self.message = message
        self.diagnostic = diagnostic
        self.architectureID = architectureID
        self.arTokS = arTokS
        self.bestMTPDepth = bestMTPDepth
        self.bestMTPTokS = bestMTPTokS
        self.bestMTPMultiplierVsAR = bestMTPMultiplierVsAR
        self.verifyRows = verifyRows
    }

    public static func parse(_ json: [String: Any]) -> ForgeBuildOutcome? {
        guard let verdict = json["verdict"] as? String else { return nil }
        let rowsJSON = (json["verify_rows"] as? [[String: Any]])
            ?? (json["rows"] as? [[String: Any]])
            ?? []
        let rows = rowsJSON.compactMap { row -> ForgeVerifyRow? in
            guard let depth = Self.intLike(row["depth"]) else { return nil }
            return ForgeVerifyRow(
                depth: depth,
                tokS: Self.doubleLike(row["tok_s"]) ?? 0,
                multiplierVsAr: Self.doubleLike(row["multiplier_vs_ar"]) ?? 0,
                acceptanceByPosition: Self.doubleArray(row["acceptance_by_position"]),
                verifyTimeSeconds: Self.doubleLike(row["verify_time_s"]) ?? 0
            )
        }
        return ForgeBuildOutcome(
            convertedPath: json["converted_path"] as? String,
            phase: json["phase"] as? String ?? "verify",
            verdict: verdict,
            failureReasons: json["failure_reasons"] as? [String] ?? [],
            message: json["message"] as? String ?? "MTP did not accelerate this model.",
            diagnostic: json["diagnostic"] as? String,
            architectureID: json["architecture_id"] as? String,
            arTokS: Self.doubleLike(json["ar_tok_s"]),
            bestMTPDepth: Self.intLike(json["best_mtp_depth"]),
            bestMTPTokS: Self.doubleLike(json["best_mtp_tok_s"]),
            bestMTPMultiplierVsAR: Self.doubleLike(json["best_mtp_multiplier_vs_ar"]),
            verifyRows: rows
        )
    }

    private static func intLike(_ value: Any?) -> Int? {
        if let int = value as? Int { return int }
        if let int64 = value as? Int64 { return Int(int64) }
        if let double = value as? Double { return Int(double) }
        return nil
    }

    private static func doubleLike(_ value: Any?) -> Double? {
        if let double = value as? Double { return double }
        if let int = value as? Int { return Double(int) }
        if let int64 = value as? Int64 { return Double(int64) }
        return nil
    }

    private static func doubleArray(_ value: Any?) -> [Double] {
        guard let values = value as? [Any] else { return [] }
        return values.compactMap(Self.doubleLike)
    }
}

// MARK: - ForgeEvent
//
// One frame from the forge subprocess (orchestrator → view). The
// stream order is intentionally NOT total: phases overlap (download
// is still finalising when convert starts on a chunked artifact),
// and the orchestrator's job is to fan these out to per-stage UI.

public enum ForgeEvent: Sendable {
    case started(runID: String, outputDir: String, destination: String)
    case downloadProgress(ForgeDownloadProgress)
    case phaseProgress(ForgePhaseProgress)
    case verifyRowLanded(ForgeVerifyRow)
    /// Backend wrote `brand.json` containing the staged
    /// `runtime_metadata` plus the branded name. The view shows the
    /// preview card; the next event is `.completed`. The runtime
    /// metadata is shipped as raw JSON Data so the event stays
    /// Sendable; the orchestrator decodes it through
    /// `MTPLXRuntimeMetadata.parse(_:)` at the actor boundary.
    case brandStaged(brandedName: String, runtimeMetadataJSON: Data)
    /// Final success — backend wrote `forge.json`. Same Data shape
    /// as `.brandStaged` for the same reason.
    case completed(localPath: String, runtimeMetadataJSON: Data)
    /// Backend wrote build_outcome.json. Usually terminal for failed
    /// speed gates, but emitted as its own event so the UI can render
    /// a structured "converted, not accelerated" state.
    case buildOutcome(ForgeBuildOutcome)
    case speedGateFailed(ForgeBuildOutcome, stderrTail: String)
    /// Subprocess exited non-zero. `phase` identifies where it died
    /// when the backend was kind enough to write a partial phase file.
    case failed(exitCode: Int32?, phase: ForgePhase?, stderrTail: String)
    case cancelled
    /// `mtplx forge` subcommand does not exist on this MTPLX install.
    /// Frontend renders the "Forge backend not available" empty state
    /// so the rest of the wizard chrome stays usable.
    case backendNotAvailable
}

// MARK: - ForgeBuilder
//
// Wraps `mtplx forge build` with the same file-polling pattern as
// AutoTuner.swift (per-candidate files) + ModelDownloader.swift
// (subprocess lifecycle + stderr tail). Does NOT use NDJSON
// streaming because the backend's other long-running commands
// (`mtplx tune`, `mtplx pull`) don't either — keeping the contract
// uniform makes the Python agent's job smaller.
//
// On-disk contract (mirrors plan section 6):
//   <output-dir>/<run-id>/
//     download.json   { bytes_on_disk, total_bytes?, mb_per_s, eta_s?, label?, finished }
//     convert.json    { progress, label?, finished }
//     calibrate.json  { progress, label?, finished, loss?, ppl? }
//     verify.json     { rows: [ { depth, tok_s, multiplier_vs_ar, acceptance_by_position, verify_time_s } ] }
//     brand.json      { branded_name, runtime_metadata: { ... } }
//     forge.json      { local_path, runtime_metadata: { ... } }
//
// `forge.json` is read on subprocess exit (success only) and yields
// `.completed`. The intermediate files are polled at `pollInterval`.

public struct ForgeBuilder: Sendable {
    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        pollInterval: TimeInterval = 0.5
    ) {
        self.processEnvironment = processEnvironment
        self.pollInterval = pollInterval
    }

    private let processEnvironment: [String: String]
    private let pollInterval: TimeInterval

    /// Input bundle handed to `stream(_:)`. Captured here as a value
    /// type so the orchestrator can persist + replay it on resume.
    public struct BuildRequest: Equatable, Sendable {
        public var sourceRepo: String
        public var recipe: ForgeRecipe
        public var brandedName: String
        public var outputDir: String
        public var runID: String
        public var maxFans: Bool
        public var modelDirectory: String
        /// When true, frontend has explicitly acknowledged degraded
        /// MTP warning in PlanStage and the backend is asked to
        /// honour `mtpPolicy == .requantize`.
        public var allowDegradedMtp: Bool

        public init(
            sourceRepo: String,
            recipe: ForgeRecipe,
            brandedName: String,
            outputDir: String,
            runID: String,
            maxFans: Bool = true,
            allowDegradedMtp: Bool = false,
            modelDirectory: String = ModelLibrary.defaultPrimaryDirectory().path
        ) {
            self.sourceRepo = sourceRepo
            self.recipe = recipe
            self.brandedName = brandedName
            self.outputDir = outputDir
            self.runID = runID
            self.maxFans = maxFans
            self.allowDegradedMtp = allowDegradedMtp
            self.modelDirectory = ModelLibrary.canonicalURL(for: modelDirectory).path
        }
    }

    /// Returns true if `mtplx forge --help` exits cleanly — i.e. the
    /// subcommand is registered. Use this from the orchestrator at
    /// app launch to disable the wizard CTA cleanly if the user is
    /// running a pre-Forge MTPLX install.
    public func backendAvailable() async -> Bool {
        guard let executable = try? Self.resolveMtplxExecutable(env: processEnvironment) else {
            return false
        }
        return await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = executable
            process.arguments = ["forge", "--help"]
            process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                environment: processEnvironment
            )
            // Null-route the help text — the old unread Pipe deadlocked
            // the child once its output crossed the 64KB pipe buffer —
            // and bound the wait so a wedged CLI reads as "no Forge
            // backend" instead of hanging the launch-time probe (#158).
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            let watchdog = SubprocessWatchdog(process)
            do {
                try process.run()
            } catch {
                continuation.resume(returning: false)
                return
            }
            Task.detached {
                let exited = watchdog.wait(for: process, timeout: 30)
                continuation.resume(returning: exited && process.terminationStatus == 0)
            }
        }
    }

    public func stream(_ request: BuildRequest) -> AsyncStream<ForgeEvent> {
        AsyncStream<ForgeEvent>(ForgeEvent.self, bufferingPolicy: .unbounded) { continuation in
            let outputDir = URL(fileURLWithPath: request.outputDir, isDirectory: true)
            let runDir = outputDir.appendingPathComponent(request.runID, isDirectory: true)
            try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)

            let executable: URL
            do {
                executable = try Self.resolveMtplxExecutable(env: processEnvironment)
            } catch {
                continuation.yield(.backendNotAvailable)
                continuation.finish()
                return
            }

            // Encode the recipe as a single JSON arg. The Python
            // agent's argparse handler decodes it. Falling back to
            // discrete --body-bits / --body-group-size flags would
            // bloat the CLI surface; one JSON arg keeps the contract
            // matching ForgeRecipe one-for-one.
            let recipeData: Data
            do {
                recipeData = try JSONEncoder().encode(request.recipe)
            } catch {
                continuation.yield(.failed(exitCode: nil, phase: nil, stderrTail: "Failed to encode recipe: \(error.localizedDescription)"))
                continuation.finish()
                return
            }
            let recipeJSON = String(data: recipeData, encoding: .utf8) ?? "{}"

            var arguments: [String] = [
                "forge", "build",
                "--repo", request.sourceRepo,
                "--out", outputDir.path,
                "--run-id", request.runID,
                "--recipe", recipeJSON,
                "--branded-name", request.brandedName,
                "--model-root", request.modelDirectory,
            ]
            if request.maxFans {
                arguments.append("--max")
            }
            if request.allowDegradedMtp {
                arguments.append("--allow-degraded-mtp")
            }

            let process = Process()
            process.executableURL = executable
            process.arguments = arguments
            var environment = processEnvironment
            environment["MTPLX_MODEL_DIR"] = request.modelDirectory
            environment["MTPLX_FORGE_MODEL_ROOT"] = request.modelDirectory
            process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                environment: environment
            )

            let errPipe = Pipe()
            let outPipe = Pipe()
            process.standardError = errPipe
            process.standardOutput = outPipe

            let stderrBuffer = ForgeStderrTailBuffer(capacity: 4096)
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if !chunk.isEmpty {
                    stderrBuffer.append(chunk)
                }
            }
            outPipe.fileHandleForReading.readabilityHandler = { handle in
                _ = handle.availableData
            }

            do {
                try process.run()
            } catch {
                continuation.yield(.failed(exitCode: nil, phase: nil, stderrTail: error.localizedDescription))
                continuation.finish()
                return
            }
            continuation.yield(.started(
                runID: request.runID,
                outputDir: outputDir.path,
                destination: runDir.path
            ))

            let pollInterval = self.pollInterval
            let pollTask = Task.detached(priority: .userInitiated) {
                var lastDownload: ForgeDownloadProgress?
                var lastPhase: [ForgePhase: ForgePhaseProgress] = [:]
                var seenVerifyDepths: Set<Int> = []
                var brandEmitted = false
                var outcomeEmitted = false
                let yieldEvent: @Sendable (ForgeEvent) -> Void = { event in
                    continuation.yield(event)
                }
                while !Task.isCancelled, process.isRunning {
                    try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
                    Self.scanPhaseFiles(
                        runDir: runDir,
                        lastDownload: &lastDownload,
                        lastPhase: &lastPhase,
                        seenVerifyDepths: &seenVerifyDepths,
                        brandEmitted: &brandEmitted,
                        outcomeEmitted: &outcomeEmitted,
                        yield: yieldEvent
                    )
                }
                // One last scan after the process exits so any final
                // phase files written just before exit aren't missed.
                Self.scanPhaseFiles(
                    runDir: runDir,
                    lastDownload: &lastDownload,
                    lastPhase: &lastPhase,
                    seenVerifyDepths: &seenVerifyDepths,
                    brandEmitted: &brandEmitted,
                    outcomeEmitted: &outcomeEmitted,
                    yield: yieldEvent
                )
            }

            Task.detached(priority: .userInitiated) {
                process.waitUntilExit()
                errPipe.fileHandleForReading.readabilityHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                pollTask.cancel()

                if process.terminationStatus == 0 {
                    let forgePath = runDir.appendingPathComponent("forge.json")
                    if let data = try? Data(contentsOf: forgePath),
                        let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                        let localPath = root["local_path"] as? String,
                        let runtimeMeta = root["runtime_metadata"] as? [String: Any],
                        let runtimeMetaData = try? JSONSerialization.data(withJSONObject: runtimeMeta)
                    {
                        continuation.yield(.completed(localPath: localPath, runtimeMetadataJSON: runtimeMetaData))
                    } else {
                        continuation.yield(.failed(
                            exitCode: 0,
                            phase: nil,
                            stderrTail: "forge.json missing or malformed at \(forgePath.path)"
                        ))
                    }
                } else if process.terminationReason == .uncaughtSignal {
                    continuation.yield(.cancelled)
                } else {
                    // Common failure mode on a pre-Forge MTPLX: argparse
                    // exits with status 2 and a "invalid choice 'forge'"
                    // message. Translate that into the explicit
                    // backendNotAvailable signal so the UI renders the
                    // right empty state instead of a generic failure.
                    let tail = stderrBuffer.snapshot()
                    let invalidChoice = tail.range(of: "invalid choice", options: .caseInsensitive) != nil
                        && tail.range(of: "forge", options: .caseInsensitive) != nil
                    if process.terminationStatus == 2 && invalidChoice {
                        continuation.yield(.backendNotAvailable)
                    } else {
                        if let outcome = Self.readBuildOutcome(in: runDir) {
                            continuation.yield(.speedGateFailed(outcome, stderrTail: tail))
                        } else {
                            continuation.yield(.failed(
                                exitCode: process.terminationStatus,
                                phase: Self.lastWrittenPhase(in: runDir),
                                stderrTail: tail
                            ))
                        }
                    }
                }
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                // Escalating cancel (#158 sweep): SIGINT alone left a
                // SIGINT-deaf build running invisibly after the user
                // cancelled, contending with the daemon for GPU/RAM.
                SubprocessWatchdog.escalateCancel(process)
                pollTask.cancel()
            }
        }
    }

    // MARK: - Phase-file scanners

    private static func scanPhaseFiles(
        runDir: URL,
        lastDownload: inout ForgeDownloadProgress?,
        lastPhase: inout [ForgePhase: ForgePhaseProgress],
        seenVerifyDepths: inout Set<Int>,
        brandEmitted: inout Bool,
        outcomeEmitted: inout Bool,
        yield: @Sendable (ForgeEvent) -> Void
    ) {
        // download.json — bytes, total, mb_per_s, eta
        if let download = readJSON(at: runDir.appendingPathComponent("download.json")) {
            let bytes = Self.intLike(download["bytes_on_disk"]) ?? 0
            let total = Self.intLike(download["total_bytes"])
            let mbps = (download["mb_per_s"] as? Double) ?? 0
            let eta = download["eta_s"] as? Double
            let stalled = download["stalled_s"] as? Double
            let snapshot = ForgeDownloadProgress(
                bytesOnDisk: bytes,
                totalBytes: total,
                bytesPerSecond: mbps * 1_048_576,
                etaSeconds: eta,
                label: download["label"] as? String,
                stalledSeconds: stalled
            )
            if snapshot != lastDownload {
                lastDownload = snapshot
                yield(.downloadProgress(snapshot))
            }
            if let finished = download["finished"] as? Bool, finished {
                let p = ForgePhaseProgress(phase: .download, progress: 1, label: download["label"] as? String, finished: true)
                if lastPhase[.download] != p {
                    lastPhase[.download] = p
                    yield(.phaseProgress(p))
                }
            }
        }

        // convert / calibrate / publish — generic progress fields
        for phase in [ForgePhase.convert, .calibrate, .publish] {
            let path = runDir.appendingPathComponent(phase.rawValue + ".json")
            guard let json = readJSON(at: path) else { continue }
            let progress = (json["progress"] as? Double) ?? 0
            let label = json["label"] as? String
            let finished = (json["finished"] as? Bool) ?? (progress >= 1)
            let snapshot = ForgePhaseProgress(phase: phase, progress: progress, label: label, finished: finished)
            if lastPhase[phase] != snapshot {
                lastPhase[phase] = snapshot
                yield(.phaseProgress(snapshot))
            }
        }

        // verify.json — array of rows, one per (AR + each depth)
        if let verify = readJSON(at: runDir.appendingPathComponent("verify.json")),
           let rows = verify["rows"] as? [[String: Any]]
        {
            for row in rows {
                guard let depth64 = Self.intLike(row["depth"]) else { continue }
                let depth = Int(depth64)
                if seenVerifyDepths.contains(depth) { continue }
                let tokS = (row["tok_s"] as? Double) ?? 0
                let multiplier = (row["multiplier_vs_ar"] as? Double) ?? 1.0
                let accept = (row["acceptance_by_position"] as? [Double]) ?? []
                let verifyTime = (row["verify_time_s"] as? Double) ?? 0
                seenVerifyDepths.insert(depth)
                yield(.verifyRowLanded(ForgeVerifyRow(
                    depth: depth,
                    tokS: tokS,
                    multiplierVsAr: multiplier,
                    acceptanceByPosition: accept,
                    verifyTimeSeconds: verifyTime
                )))
            }
        }

        // brand.json — emit exactly once
        if !brandEmitted, let brand = readJSON(at: runDir.appendingPathComponent("brand.json")),
           let brandedName = brand["branded_name"] as? String,
           let runtimeMeta = brand["runtime_metadata"] as? [String: Any],
           let runtimeMetaData = try? JSONSerialization.data(withJSONObject: runtimeMeta)
        {
            brandEmitted = true
            yield(.brandStaged(brandedName: brandedName, runtimeMetadataJSON: runtimeMetaData))
        }

        if !outcomeEmitted, let outcome = readBuildOutcome(in: runDir) {
            outcomeEmitted = true
            yield(.buildOutcome(outcome))
        }
    }

    private static func readBuildOutcome(in runDir: URL) -> ForgeBuildOutcome? {
        guard let json = readJSON(at: runDir.appendingPathComponent("build_outcome.json")) else {
            return nil
        }
        return ForgeBuildOutcome.parse(json)
    }

    private static func readJSON(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    /// Accept both Int and Double leaves — Python `int` survives the
    /// JSON round-trip as either depending on writer mood.
    private static func intLike(_ value: Any?) -> Int64? {
        if let int = value as? Int64 { return int }
        if let int = value as? Int { return Int64(int) }
        if let double = value as? Double { return Int64(double) }
        return nil
    }

    /// Best-effort determination of which phase the backend died in,
    /// based on which phase file is newest on disk. Used for the
    /// failure banner.
    private static func lastWrittenPhase(in runDir: URL) -> ForgePhase? {
        var best: (ForgePhase, Date)?
        for phase in ForgePhase.allCases {
            let path = runDir.appendingPathComponent(phase.rawValue + ".json").path
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
               let modified = attrs[.modificationDate] as? Date
            {
                if best == nil || modified > best!.1 {
                    best = (phase, modified)
                }
            }
        }
        return best?.0
    }

    // MARK: - Executable resolution

    static func resolveMtplxExecutable(env: [String: String]) throws -> URL {
        try MTPLXCommandBuilder.resolveInstalledExecutable(environment: env)
    }
}

// MARK: - Shared StderrTailBuffer
//
// Same shape as ModelDownloader / AutoTuner. Kept fileprivate to
// avoid clashing with the others.

private final class ForgeStderrTailBuffer: @unchecked Sendable {
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
