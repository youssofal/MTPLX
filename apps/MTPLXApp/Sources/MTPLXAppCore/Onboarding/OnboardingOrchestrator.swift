import Foundation
import Combine

// MARK: - OnboardingOrchestrator
//
// `@MainActor ObservableObject` that owns the pure `state` value plus
// all live progress published by the long-running services (download
// + tune). The views read everything off of this single object.
//
// Service tasks (the download and tune subprocesses) are stored on the
// orchestrator so they can be torn down on `cancelAll()` (called by
// `OnboardingExperienceView` when it disappears, or by the user
// pressing Stop on a step).
//
// Lives in `MTPLXAppCore` so the orchestrator can be unit-tested
// without the SwiftUI host. The view layer in MTPLXAppHost imports
// this and renders.

@MainActor
public final class OnboardingOrchestrator: ObservableObject {
    @Published public private(set) var state: OnboardingFeatureState
    @Published public private(set) var runtimeSetupRows: [RuntimeSetupRow]
    @Published public private(set) var runtimeSetupOutcome: RuntimeSetupOutcome?
    @Published public private(set) var isRunningRuntimeSetup: Bool
    @Published public private(set) var downloadProgress: DownloadProgressSnapshot?
    @Published public private(set) var downloadFailure: String?
    @Published public private(set) var tuneCandidatesLanded: [TuneCandidate: TuneCandidateResult]
    @Published public private(set) var tuneResult: TuneResult?
    @Published public private(set) var tuneFailure: String?
    @Published public private(set) var tuneStatusMessage: String?
    @Published public private(set) var isProbingOther: Bool
    @Published public private(set) var isDetectingHardware: Bool
    @Published public private(set) var isDownloading: Bool
    @Published public private(set) var isTuning: Bool
    /// Optional Hugging Face mirror for the wizard's download step
    /// (issue #96: huggingface.co is blocked in mainland China). The
    /// completion handler carries a valid value into the saved app
    /// configuration.
    @Published public var hfMirrorEndpoint: String = ""

    public init(
        hardwareInspector: HardwareInspector = HardwareInspector(),
        huggingFaceProbe: HuggingFaceProbe = HuggingFaceProbe(),
        modelDownloader: ModelDownloader = ModelDownloader(),
        autoTuner: AutoTuner = AutoTuner(),
        runtimeSetup: RuntimeSetupService = RuntimeSetupService(),
        feasibility: ModelFeasibility = ModelFeasibility(),
        modelLibrary: ModelLibrary = .default,
        initialState: OnboardingFeatureState = OnboardingFeatureState()
    ) {
        self.hardwareInspector = hardwareInspector
        self.huggingFaceProbe = huggingFaceProbe
        self.modelDownloader = modelDownloader
        self.autoTuner = autoTuner
        self.runtimeSetup = runtimeSetup
        self.feasibility = feasibility
        self.modelLibrary = modelLibrary
        self.state = initialState
        self.runtimeSetupRows = []
        self.isRunningRuntimeSetup = false
        self.tuneCandidatesLanded = [:]
        self.isProbingOther = false
        self.isDetectingHardware = false
        self.isDownloading = false
        self.isTuning = false
    }

    // MARK: - Service dependencies (Sendable values)

    private let hardwareInspector: HardwareInspector
    private let huggingFaceProbe: HuggingFaceProbe
    private let modelDownloader: ModelDownloader
    private let autoTuner: AutoTuner
    private let runtimeSetup: RuntimeSetupService
    private let feasibility: ModelFeasibility
    private var modelLibrary: ModelLibrary

    // MARK: - Cancellable task handles

    private var hardwareTask: Task<Void, Never>?
    private var probeTask: Task<Void, Never>?
    private var runtimeSetupTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?
    private var tuneTask: Task<Void, Never>?

    // MARK: - Step navigation

    public func goNext() {
        if state.step == .runtimeSetup, case .local = state.pick {
            // Local model folders skip the download step — the bytes
            // are already on disk — but never the runtime setup that
            // precedes it.
            state.step = .tune
            return
        }
        state.goNext()
    }
    public func goBack() {
        if state.step == .tune, case .local = state.pick {
            state.step = .runtimeSetup
            return
        }
        state.goBack()
    }

    public func returnToModelPick() {
        cancelDownload()
        cancelTune()
        downloadProgress = nil
        downloadFailure = nil
        tuneResult = nil
        tuneFailure = nil
        tuneCandidatesLanded = [:]
        state.step = .modelPick
    }

    public func select(_ choice: ModelPickChoice) {
        state.select(choice)
        downloadProgress = nil
        downloadFailure = nil
        tuneResult = nil
        tuneFailure = nil
        tuneCandidatesLanded = [:]
    }

    // MARK: - Hardware detection

    public func detectHardware() {
        guard !isDetectingHardware, state.hardware == nil else { return }
        isDetectingHardware = true
        let inspector = hardwareInspector
        hardwareTask?.cancel()
        hardwareTask = Task { [weak self] in
            let result = await inspector.detect()
            await MainActor.run {
                guard let self else { return }
                self.state.hardware = result
                self.isDetectingHardware = false
            }
        }
    }

    // MARK: - HuggingFace probe

    public func probeOther(repo: String) {
        guard !isProbingOther else { return }
        isProbingOther = true
        let probe = huggingFaceProbe
        probeTask?.cancel()
        probeTask = Task { [weak self] in
            let result = await probe.probe(repo: repo)
            await MainActor.run {
                guard let self else { return }
                self.state.record(result)
                self.isProbingOther = false
            }
        }
    }

    public func probeLocal(path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            state.record(LocalModelProbe(
                verdict: .notFound,
                path: path,
                message: "Paste a local model folder first."
            ))
            return
        }

        let expanded = NSString(string: trimmed).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            state.record(LocalModelProbe(
                verdict: .notFound,
                path: trimmed,
                message: "That folder doesn't exist on this Mac.",
                diagnostic: expanded
            ))
            return
        }

        guard MTPLXModelOption.hasCompleteInstall(at: expanded) else {
            state.record(LocalModelProbe(
                verdict: .incomplete,
                path: trimmed,
                message: "That folder is not a complete MTPLX model yet.",
                diagnostic: "Need config/tokenizer/runtime metadata, full model weights, and an MTP sidecar."
            ))
            return
        }

        let family = MTPLXModelOption.modelFamily(for: expanded)
        state.record(LocalModelProbe(
            verdict: .ready,
            path: trimmed,
            message: "\(Self.modelFamilyLabel(family)) model ready from this folder.",
            diagnostic: expanded
        ))
    }

    // MARK: - Install detection (read-only convenience)

    /// Returns true when the model's on-disk copy looks complete
    /// enough to skip the download. Mirrors the daemon's
    /// `cached_model_is_complete` + MTPLX-required-files check
    /// (`mtplx/hf_loader.py`): every file listed in
    /// `REQUIRED_MTPLX_MODEL_FILES` must be present on disk.
    ///
    /// Uses `FileManager.fileExists` (which follows symlinks) so
    /// research setups where weight shards are symlinked from another
    /// model directory (e.g. `~/.lmstudio/models/...`) pass cleanly.
    /// A naive byte-size check would falsely fail those because
    /// `enumerator(at:)` only sums the symlink inode size (~80 B),
    /// not the resolved target — leaving 15 GB symlinked installs
    /// looking like a few KB on disk.
    /// Delegates to `MTPLXModelOption.isInstalled`, which now owns the
    /// MTPLX-completeness contract (core files + MTP sidecar + every
    /// trunk shard from the safetensors index). Keeping the public
    /// method around so onboarding callsites don't have to be
    /// rewritten and the single source of truth is the option type.
    public func isModelInstalled(_ model: MTPLXModelOption) -> Bool {
        model.isInstalled(in: modelLibrary)
    }

    public func installedLocalPath(for model: MTPLXModelOption) -> String? {
        model.installedLocalPath(in: modelLibrary)
    }

    public func configureModelLibrary(_ library: ModelLibrary) {
        guard !isDownloading else { return }
        modelLibrary = library
    }

    // MARK: - Feasibility (read-only convenience)

    public func evaluateFeasibility(for model: MTPLXModelOption) -> ModelFeasibilityVerdict {
        let hw = state.hardware
        let chipTier = hw?.tier ?? .unknown
        let ramGiB = hw?.unifiedMemoryGiB ?? 0
        let diskFreeGiB = model.isInstalled(in: modelLibrary)
            ? Double.greatestFiniteMagnitude
            : freeDiskGiB()
        return feasibility.evaluate(
            model: model,
            chipTier: chipTier,
            ramGiB: ramGiB,
            diskFreeGiB: diskFreeGiB
        )
    }

    public var tuneCandidates: [TuneCandidate] {
        state.tuneCandidates
    }

    public func freeDiskGiB() -> Double {
        let volume = modelLibrary.nearestExistingDirectory(to: modelLibrary.primaryDirectory)
        let values = try? volume.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let bytes = values?.volumeAvailableCapacityForImportantUsage ?? 0
        return Double(bytes) / 1_073_741_824.0
    }

    private static func modelFamilyLabel(_ family: String) -> String {
        switch family {
        case "qwen3_5": return "Qwen 3.5"
        case "qwen3_6": return "Qwen 3.6"
        case "qwen3_8": return "Qwen 3.8"
        case "qwen4_exp": return "Flash-Next"
        case "gemma4": return "Gemma"
        case "step": return "Step"
        case "glm": return "GLM"
        case "deepseek": return "DeepSeek"
        default: return "MTPLX"
        }
    }

    // MARK: - Runtime setup

    /// True once the engine phase has succeeded; fan-control and
    /// global-CLI warnings never block Continue.
    public var runtimeSetupComplete: Bool {
        runtimeSetupOutcome?.engineReady == true
    }

    /// The engine row's failure message when setup is blocked.
    public var runtimeSetupFailure: String? {
        guard let row = runtimeSetupRows.first(where: { $0.id == .engine }),
              row.state == .failed
        else { return nil }
        return row.detail
    }

    /// Auto-invoked when the runtime-setup step appears. A completed
    /// run is sticky for the session; use `retryRuntimeSetup()` to
    /// force a re-run after a failure.
    public func startRuntimeSetup() {
        guard !isRunningRuntimeSetup, !runtimeSetupComplete else { return }
        runtimeSetupTask?.cancel()
        runtimeSetupOutcome = nil
        isRunningRuntimeSetup = true
        let service = runtimeSetup
        runtimeSetupTask = Task { [weak self] in
            for await event in service.stream() {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handleRuntimeSetupEvent(event)
                }
            }
            await MainActor.run {
                guard let self else { return }
                // Stream ended without a `.finished` (cancellation or
                // engine failure already handled) — stop the spinner.
                self.isRunningRuntimeSetup = false
            }
        }
    }

    public func retryRuntimeSetup() {
        cancelRuntimeSetup()
        runtimeSetupRows = []
        runtimeSetupOutcome = nil
        startRuntimeSetup()
    }

    public func cancelRuntimeSetup() {
        runtimeSetupTask?.cancel()
        runtimeSetupTask = nil
        isRunningRuntimeSetup = false
    }

    private func handleRuntimeSetupEvent(_ event: RuntimeSetupEvent) {
        switch event {
        case .rows(let rows):
            runtimeSetupRows = rows
        case .finished(let outcome):
            runtimeSetupRows = outcome.rows
            runtimeSetupOutcome = outcome
            isRunningRuntimeSetup = false
        }
    }

    // MARK: - Download

    public func startDownload() {
        guard !isDownloading, let repo = state.resolvedRepoID else { return }
        let totalBytes = state.resolvedModel?.sizeBytes
        // Disk pre-flight before spawning the subprocess. Mirrors the
        // daemon's `required_download_free_bytes` heuristic (model * 2.5).
        // For `Other`, we don't know the size yet, so we skip this gate
        // — the user accepted the risk by pasting a custom repo.
        if let bytes = totalBytes, bytes > 0 {
            let freeGiB = freeDiskGiB()
            let neededGiB = Double(bytes) / 1_073_741_824.0 * ModelFeasibility.diskMultiplier
            if freeGiB < neededGiB {
                downloadFailure = String(
                    format: "Not enough free disk space. Need %.0f GB free, you have %.0f GB.",
                    neededGiB, freeGiB
                )
                return
            }
        }
        downloadFailure = nil
        downloadProgress = nil
        isDownloading = true
        let downloader = modelDownloader
        let cacheRoot = modelLibrary.primaryDirectory
        let extraEnvironment = MTPLXAppConfiguration.hfMirrorEnvironment(hfMirrorEndpoint) ?? [:]
        downloadTask?.cancel()
        downloadTask = Task.detached(priority: .userInitiated) { [weak self, downloader, repo, totalBytes, extraEnvironment, cacheRoot] in
            for await event in downloader.stream(
                repo: repo,
                totalBytes: totalBytes,
                extraEnvironment: extraEnvironment,
                cacheRoot: cacheRoot
            ) {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handleDownloadEvent(event)
                }
            }
        }
    }

    public func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        if var snapshot = downloadProgress, !snapshot.isComplete {
            snapshot.bytesPerSecond = 0
            snapshot.etaSeconds = nil
            snapshot.stalledSeconds = 0
            snapshot.statusMessage = "Paused"
            downloadProgress = snapshot
        }
    }

    /// Wizard-facing failure copy. Network-shaped failures point at the
    /// mirror field rendered directly under the banner; everything else
    /// passes through untouched.
    nonisolated static func downloadFailureMessage(stderrTail: String, mirrorActive: Bool) -> String {
        let base = stderrTail.isEmpty ? "Download failed." : stderrTail
        let lower = base.lowercased()
        let networkShaped = lower.contains("timed out")
            || lower.contains("connection")
            || lower.contains("network")
            || lower.contains("max retries")
            || lower.contains("unreachable")
        guard networkShaped, !mirrorActive else { return base }
        return base
            + "\nIf huggingface.co is blocked on your network, set a download mirror below and retry."
    }

    private func handleDownloadEvent(_ event: DownloadEvent) {
        switch event {
        case .started(let path):
            downloadFailure = nil
            downloadProgress = DownloadProgressSnapshot(
                destinationPath: path,
                bytesOnDisk: 0,
                totalBytes: state.resolvedModel?.sizeBytes,
                bytesPerSecond: 0,
                etaSeconds: nil,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: "Resolving files"
            )
        case .status(let message, let bytes, let total, let path):
            downloadFailure = nil
            var snapshot = downloadProgress ?? DownloadProgressSnapshot(
                destinationPath: path ?? "",
                bytesOnDisk: bytes ?? 0,
                totalBytes: total ?? state.resolvedModel?.sizeBytes,
                bytesPerSecond: 0,
                etaSeconds: nil,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: message
            )
            if let path { snapshot.destinationPath = path }
            if let bytes { snapshot.bytesOnDisk = bytes }
            if let total { snapshot.totalBytes = total }
            snapshot.statusMessage = message
            downloadProgress = snapshot
        case .progress(let bytes, let total, let smoothed, let eta):
            downloadFailure = nil
            downloadProgress = DownloadProgressSnapshot(
                destinationPath: downloadProgress?.destinationPath ?? "",
                bytesOnDisk: bytes,
                totalBytes: total,
                bytesPerSecond: smoothed,
                etaSeconds: eta,
                stalledSeconds: 0,
                isComplete: false,
                statusMessage: "Downloading"
            )
        case .stalled(let seconds):
            downloadFailure = nil
            if var snapshot = downloadProgress {
                snapshot.stalledSeconds = seconds
                snapshot.bytesPerSecond = 0
                snapshot.etaSeconds = nil
                snapshot.statusMessage = "Waiting on Hugging Face"
                downloadProgress = snapshot
            }
        case .complete(let bytes, let path):
            guard MTPLXModelOption.hasCompleteInstall(at: path) else {
                downloadProgress = DownloadProgressSnapshot(
                    destinationPath: path,
                    bytesOnDisk: bytes,
                    totalBytes: state.resolvedModel?.sizeBytes ?? bytes,
                    bytesPerSecond: 0,
                    etaSeconds: nil,
                    stalledSeconds: 0,
                    isComplete: false,
                    statusMessage: "Incomplete"
                )
                downloadFailure = "Download finished, but files the source repo ships are still missing from the model folder. Press Retry to resume the Hugging Face download."
                isDownloading = false
                return
            }
            downloadFailure = nil
            downloadProgress = DownloadProgressSnapshot(
                destinationPath: path,
                bytesOnDisk: bytes,
                totalBytes: state.resolvedModel?.sizeBytes ?? bytes,
                bytesPerSecond: 0,
                etaSeconds: 0,
                stalledSeconds: 0,
                isComplete: true,
                statusMessage: "Ready"
            )
            isDownloading = false
            // Auto-advance to the tune step. Views observe both
            // `state.step` and `downloadProgress.isComplete`; the
            // step bump is the canonical signal.
            state.goNext()
        case .failed(_, let stderrTail):
            downloadFailure = Self.downloadFailureMessage(
                stderrTail: stderrTail,
                mirrorActive: MTPLXAppConfiguration.hfMirrorEnvironment(hfMirrorEndpoint) != nil
            )
            isDownloading = false
        case .cancelled:
            isDownloading = false
            if var snapshot = downloadProgress, !snapshot.isComplete {
                snapshot.bytesPerSecond = 0
                snapshot.etaSeconds = nil
                snapshot.stalledSeconds = 0
                snapshot.statusMessage = "Paused"
                downloadProgress = snapshot
            }
        }
    }

    // MARK: - Tune

    public func startTune() {
        guard !isTuning else { return }
        tuneTask?.cancel()
        tuneFailure = nil
        tuneStatusMessage = nil
        tuneResult = nil
        tuneCandidatesLanded = [:]
        guard state.supportsTune else {
            skipTuneForModelDefaults()
            return
        }
        let candidates = tuneCandidates
        guard !candidates.isEmpty else {
            skipTuneForModelDefaults()
            return
        }
        let modelPath = resolvedTuneModelPath()
        guard let modelPath else {
            tuneFailure = "No model selected to tune."
            return
        }
        isTuning = true
        let tuner = autoTuner
        tuneTask = Task { [weak self] in
            for await event in tuner.stream(modelPath: modelPath, candidates: candidates) {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handleTuneEvent(event)
                }
            }
        }
    }

    public func cancelTune() {
        tuneTask?.cancel()
        tuneTask = nil
        isTuning = false
        tuneStatusMessage = nil
    }

    public func skipTuneWithSafeDefault() {
        guard state.supportsTune else {
            skipTuneForModelDefaults()
            return
        }
        guard MTPLXModelOption.supportsTune(family: state.resolvedModelFamily) else {
            skipTuneForModelDefaults()
            return
        }
        // Used by the ThermalForge-missing fallback. depth=2 is the
        // codebase's documented safe Qwen heuristic.
        tuneResult = TuneResult(
            bestCandidate: .d2,
            bestDepth: 2,
            bestTokS: 0,
            bestMultiplierVsAR: 0,
            allCandidates: []
        )
    }

    public func skipTuneForModelDefaults() {
        tuneTask?.cancel()
        tuneFailure = nil
        tuneStatusMessage = nil
        tuneCandidatesLanded = [:]
        isTuning = false
        tuneResult = TuneResult(
            bestCandidate: .ar,
            bestDepth: 0,
            bestTokS: 0,
            bestMultiplierVsAR: 0,
            allCandidates: []
        )
    }

    public func thermalHelperPresent() -> Bool {
        autoTuner.thermalHelperPresent()
    }

    private func handleTuneEvent(_ event: TuneEvent) {
        switch event {
        case .installingFanControl(let message):
            tuneStatusMessage = message
        case .started:
            tuneStatusMessage = nil
        case .candidateLanded(let result):
            tuneCandidatesLanded[result.candidate] = result
        case .completed(let result):
            tuneStatusMessage = nil
            tuneResult = result
            // Also fold the per-candidate map so the view doesn't have
            // to merge from two sources.
            for entry in result.allCandidates {
                tuneCandidatesLanded[entry.candidate] = entry
            }
            isTuning = false
        case .failed(_, let stderrTail):
            tuneStatusMessage = nil
            tuneFailure = stderrTail.isEmpty ? "Tuning failed." : stderrTail
            isTuning = false
        case .cancelled:
            tuneStatusMessage = nil
            isTuning = false
        }
    }

    private func resolvedTuneModelPath() -> String? {
        if let model = state.resolvedModel,
           let local = model.installedLocalPath(in: modelLibrary) {
            return local
        }
        // The tune subprocess can also resolve an HF id directly via
        // `mtplx pull` semantics, so we hand it the catalog hfModelID
        // when the local copy isn't present yet (shouldn't happen
        // after step 4, but kept for safety).
        return state.resolvedModel?.hfModelID ?? state.resolvedRepoID
    }

    // MARK: - Cleanup

    public func cancelAll() {
        hardwareTask?.cancel()
        probeTask?.cancel()
        runtimeSetupTask?.cancel()
        downloadTask?.cancel()
        tuneTask?.cancel()
        isDetectingHardware = false
        isProbingOther = false
        isRunningRuntimeSetup = false
        isDownloading = false
        isTuning = false
    }
}

// MARK: - DownloadProgressSnapshot
//
// Tiny value type the orchestrator owns and the view binds against.
// All fields are stable across the step so SwiftUI animations don't
// flicker when `bytesOnDisk` changes 2 Hz.

public struct DownloadProgressSnapshot: Equatable, Sendable {
    public var destinationPath: String
    public var bytesOnDisk: Int64
    public var totalBytes: Int64?
    public var bytesPerSecond: Double
    public var etaSeconds: Double?
    public var stalledSeconds: Int
    public var isComplete: Bool
    public var statusMessage: String?

    public init(
        destinationPath: String,
        bytesOnDisk: Int64,
        totalBytes: Int64?,
        bytesPerSecond: Double,
        etaSeconds: Double?,
        stalledSeconds: Int,
        isComplete: Bool,
        statusMessage: String? = nil
    ) {
        self.destinationPath = destinationPath
        self.bytesOnDisk = bytesOnDisk
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
        self.etaSeconds = etaSeconds
        self.stalledSeconds = stalledSeconds
        self.isComplete = isComplete
        self.statusMessage = statusMessage
    }

    public var fraction: Double {
        guard let total = totalBytes, total > 0 else { return 0 }
        return min(1, max(0, Double(bytesOnDisk) / Double(total)))
    }
}
