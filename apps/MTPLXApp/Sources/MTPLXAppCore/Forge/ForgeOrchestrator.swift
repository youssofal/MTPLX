import Foundation
import Combine
#if canImport(AppKit)
import AppKit
#endif

// MARK: - ForgeOrchestrator
//
// `@MainActor ObservableObject` that owns the pure `ForgeFeatureState`
// value plus every live signal the wizard surfaces (probe / download
// / convert / calibrate / verify / brand / publish progress). Views
// read everything off this single object so SwiftUI invalidation is
// scoped per @Published property, not the whole state tree.
//
// Mirrors the shape of `OnboardingOrchestrator` deliberately —
// service tasks are stored on the orchestrator so `cancelAll()` can
// TERM the subprocesses on view-disappear, and the wizard's stage
// views can be stateless ("read these published properties; never
// own Tasks").
//
// Lives in `MTPLXAppCore` so the orchestrator can be unit-tested
// without the SwiftUI host.

@MainActor
public final class ForgeOrchestrator: ObservableObject {
    // MARK: Published surface

    @Published public private(set) var state: ForgeFeatureState

    // Service-driven signals (one @Published per "section" of the
    // wizard; SwiftUI invalidates only the views that bind them).
    @Published public private(set) var isProbing: Bool
    @Published public private(set) var probeFailure: String?

    @Published public private(set) var convertPhases: [ForgePhase: ForgePhaseProgress]
    @Published public private(set) var downloadProgress: ForgeDownloadProgress?
    @Published public private(set) var verifyRows: [Int: ForgeVerifyRow]
    @Published public private(set) var brandedRuntimeMetadata: MTPLXRuntimeMetadata?
    @Published public private(set) var completedLocalPath: String?
    @Published public private(set) var buildOutcome: ForgeBuildOutcome?
    @Published public private(set) var buildRunDir: String?
    @Published public private(set) var buildFailure: String?
    @Published public private(set) var buildPhase: ForgePhase?
    @Published public private(set) var isBuilding: Bool
    @Published public private(set) var backendUnavailable: Bool

    @Published public private(set) var publishProgress: ForgePhaseProgress?
    @Published public private(set) var publishFailure: String?
    @Published public private(set) var isPublishing: Bool

    /// Orphan run dirs discovered on app launch. The Forge tab
    /// shows a Resume / Discard banner when this is non-empty.
    @Published public private(set) var orphanRuns: [ForgeOrphanRun] = []

    // MARK: Init

    public init(
        hardwareInspector: HardwareInspector = HardwareInspector(),
        huggingFaceProbe: HuggingFaceProbe = HuggingFaceProbe(),
        forgeBuilder: ForgeBuilder = ForgeBuilder(),
        hfPublisher: HFPublisher = HFPublisher(),
        hfTokenStore: HFTokenStore = HFTokenStore(),
        feasibility: ModelFeasibility = ModelFeasibility(),
        initialState: ForgeFeatureState = ForgeFeatureState()
    ) {
        self.hardwareInspector = hardwareInspector
        self.huggingFaceProbe = huggingFaceProbe
        self.forgeBuilder = forgeBuilder
        self.hfPublisher = hfPublisher
        self.hfTokenStore = hfTokenStore
        self.feasibility = feasibility
        self.state = initialState
        self.isProbing = false
        self.convertPhases = [:]
        self.verifyRows = [:]
        self.buildOutcome = nil
        self.buildRunDir = nil
        self.isBuilding = false
        self.backendUnavailable = false
        self.isPublishing = false
    }

    // MARK: Services (Sendable values, injectable for tests)

    private let hardwareInspector: HardwareInspector
    private let huggingFaceProbe: HuggingFaceProbe
    private let forgeBuilder: ForgeBuilder
    private let hfPublisher: HFPublisher
    private let hfTokenStore: HFTokenStore
    private let feasibility: ModelFeasibility

    // MARK: Task handles (cancel surface)

    private var hardwareTask: Task<Void, Never>?
    private var probeTask: Task<Void, Never>?
    private var buildTask: Task<Void, Never>?
    private var publishTask: Task<Void, Never>?

    // MARK: - Step + sub-mode navigation

    public func goNext() { state.goNext() }
    public func goBack() { state.goBack() }
    public func continueAfterVerify() {
        guard completedLocalPath != nil, !isBuilding, state.hasSpeedWinningVerification else { return }
        state.step = .brand
    }
    public func resetWizard() {
        cancelAll()
        state.resetWizard()
        convertPhases = [:]
        downloadProgress = nil
        verifyRows = [:]
        brandedRuntimeMetadata = nil
        completedLocalPath = nil
        buildOutcome = nil
        buildRunDir = nil
        buildFailure = nil
        buildPhase = nil
        probeFailure = nil
        publishProgress = nil
        publishFailure = nil
    }

    public func selectSubMode(_ mode: ForgeSubMode) {
        state.selectSubMode(mode)
    }

    // MARK: - Source step

    public func setSourceRepo(_ input: String) {
        state.setSourceRepo(input)
        probeFailure = nil
    }

    public func probeSource() {
        guard let repo = state.resolvedRepoID, !isProbing else { return }
        isProbing = true
        probeFailure = nil
        let probe = huggingFaceProbe
        probeTask?.cancel()
        probeTask = Task { [weak self] in
            let result = await probe.forgeProbe(repo: repo)
            await MainActor.run {
                guard let self else { return }
                self.state.recordProbe(result)
                if result.verdict == .probeFailed {
                    self.probeFailure = result.diagnostic ?? result.message
                }
                self.isProbing = false
            }
        }
    }

    // MARK: - Hardware (read on demand for Plan stage feasibility)

    public func detectHardwareIfNeeded() {
        guard state.hardware == nil else { return }
        let inspector = hardwareInspector
        hardwareTask?.cancel()
        hardwareTask = Task { [weak self] in
            let result = await inspector.detect()
            await MainActor.run {
                self?.state.hardware = result
            }
        }
    }

    public func evaluateFeasibility() -> ModelFeasibilityVerdict? {
        guard let probe = state.sourceProbe, probe.verdict == .forgeable else { return nil }
        let hw = state.hardware
        let chipTier = hw?.tier ?? .unknown
        let ramGiB = hw?.unifiedMemoryGiB ?? 0
        let diskFreeGiB = freeDiskGiB()
        // Use the probe's estimates if available, otherwise fall
        // back to a generous default that won't false-positive on
        // small models.
        let sizeBytes = probe.estimatedSizeBytes ?? 0
        let peakGiB = probe.estimatedPeakGiB ?? 0
        let modelOption = MTPLXModelOption(
            id: "forge-feasibility-stub",
            displayName: probe.hfRepo,
            shortName: probe.hfRepo,
            detail: "",
            hfModelID: probe.hfRepo,
            localCandidates: [],
            aliases: [],
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakGiB,
            recommendedFor: []
        )
        return feasibility.evaluate(
            model: modelOption,
            chipTier: chipTier,
            ramGiB: ramGiB,
            diskFreeGiB: diskFreeGiB
        )
    }

    public func freeDiskGiB() -> Double {
        // The model store's own volume, not the home volume (#466).
        ModelStoreVolume.freeGiB()
    }

    // MARK: - Plan step

    public func updateRecipe(_ recipe: ForgeRecipe) {
        state.updateRecipe(recipe)
    }

    public func acknowledgeDegradedMTP() {
        state.acknowledgeDegradedMTP()
    }

    // MARK: - Build (drives convert + calibrate + verify + brand)

    public func startBuild() {
        guard !isBuilding else { return }
        guard let probe = state.sourceProbe, probe.verdict == .forgeable else { return }
        guard !(state.recipe.degradesMtp && !state.hasAcknowledgedDegradedMTP) else { return }

        // Reset per-build live signals before launching.
        convertPhases = [:]
        downloadProgress = nil
        verifyRows = [:]
        brandedRuntimeMetadata = nil
        completedLocalPath = nil
        buildOutcome = nil
        buildRunDir = nil
        buildFailure = nil
        buildPhase = nil
        backendUnavailable = false
        isBuilding = true

        // Auto-advance the wizard to the convert step so the
        // Convert stage shows live progress immediately.
        if state.step == .plan {
            state.goNext()
        }

        let runID = "mtplx-forge-" + UUID().uuidString.lowercased().prefix(8)
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mtplx-forge", isDirectory: true)
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let brandedName = ForgeBrandInfo.resolvedBrandedName(
            userName: state.brand.brandedName,
            fallbackSourceRepo: probe.hfRepo
        )
        if state.brand.brandedName != brandedName {
            state.brand.brandedName = brandedName
        }

        let request = ForgeBuilder.BuildRequest(
            sourceRepo: probe.hfRepo,
            recipe: state.recipe,
            brandedName: brandedName,
            outputDir: outputDir.path,
            runID: String(runID),
            maxFans: true,
            allowDegradedMtp: state.recipe.degradesMtp
        )

        let builder = forgeBuilder
        buildTask?.cancel()
        buildTask = Task { [weak self] in
            for await event in builder.stream(request) {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handleBuildEvent(event)
                }
            }
        }
    }

    public func cancelBuild() {
        buildTask?.cancel()
        buildTask = nil
        isBuilding = false
        // Roll wizard back to the Plan step so the user can adjust
        // recipe + retry; partial files in the run dir survive for
        // an opportunistic resume in the future.
        if [ForgeStep.convert, .calibrate, .verify].contains(state.step) {
            state.step = .plan
        }
    }

    private func handleBuildEvent(_ event: ForgeEvent) {
        switch event {
        case .started(_, _, let destination):
            buildRunDir = destination

        case .downloadProgress(let progress):
            downloadProgress = progress
            buildPhase = .download

        case .phaseProgress(let progress):
            convertPhases[progress.phase] = progress
            buildPhase = progress.phase
            // Auto-advance the wizard step to keep the UI honest
            // about which phase is on-screen.
            switch progress.phase {
            case .download:
                break // handled by downloadProgress
            case .convert:
                if state.step != .calibrate && state.step != .verify {
                    state.step = .convert
                }
            case .calibrate:
                if state.step != .verify {
                    state.step = .calibrate
                }
            case .verify:
                state.step = .verify
            case .brand, .publish:
                break
            }

        case .verifyRowLanded(let row):
            verifyRows[row.depth] = row
            if state.step != .verify { state.step = .verify }

        case .buildOutcome(let outcome):
            buildOutcome = outcome
            for row in outcome.verifyRows {
                verifyRows[row.depth] = row
            }
            if !outcome.isSpeedWin {
                buildFailure = outcome.message
            }
            state.step = .verify

        case .brandStaged(let brandedName, let runtimeMetadataData):
            // Carry the staged metadata so the Brand stage can show
            // the preview card with the actual published shape.
            if let json = try? JSONSerialization.jsonObject(with: runtimeMetadataData) as? [String: Any] {
                brandedRuntimeMetadata = MTPLXRuntimeMetadata.parse(json)
            }
            // Don't overwrite a user-chosen branded name if they
            // edited it on BrandStage; the backend's name is the
            // fallback shape (`<source>-MTPLX-<role>`).
            let userBranded = state.brand.brandedName.trimmingCharacters(in: .whitespacesAndNewlines)
            if userBranded.isEmpty {
                state.brand.brandedName = brandedName
            }

        case .completed(let localPath, let runtimeMetadataData):
            completedLocalPath = localPath
            if let json = try? JSONSerialization.jsonObject(with: runtimeMetadataData) as? [String: Any] {
                brandedRuntimeMetadata = MTPLXRuntimeMetadata.parse(json)
            }
            // Surface verification numbers to the state so BrandStage
            // can render the verification footer + so the Registered
            // celebration card can summarise the multiplier.
            if let json = try? JSONSerialization.jsonObject(with: runtimeMetadataData) as? [String: Any],
               let v = Self.extractForgeVerification(from: json)
            {
                state.recordVerification(v)
            }
            isBuilding = false
            buildPhase = nil

        case .speedGateFailed(let outcome, let stderrTail):
            buildOutcome = outcome
            for row in outcome.verifyRows {
                verifyRows[row.depth] = row
            }
            completedLocalPath = nil
            brandedRuntimeMetadata = nil
            buildFailure = outcome.message.isEmpty
                ? (stderrTail.isEmpty ? tr("MTP did not accelerate this model.") : stderrTail)
                : outcome.message
            buildPhase = .verify
            isBuilding = false
            state.step = .verify

        case .failed(_, let phase, let stderrTail):
            buildFailure = stderrTail.isEmpty ? tr("Forge build failed.") : stderrTail
            buildPhase = phase
            isBuilding = false

        case .cancelled:
            isBuilding = false

        case .backendNotAvailable:
            backendUnavailable = true
            buildFailure = tr("Forge backend not available. Install or update MTPLX 1.x to use this tab.")
            isBuilding = false
        }
    }

    /// Extracts a ForgeVerification view from the staged
    /// `runtime_metadata`. The Python agent writes the verification
    /// numbers into the existing `speed_evidence` block (per the
    /// contract — it has a home in the spine, not in the new
    /// `forge_provenance`). We're tolerant of missing fields.
    private static func extractForgeVerification(from runtimeMeta: [String: Any]) -> ForgeVerification? {
        ForgeVerification.fromRuntimeMetadata(runtimeMeta)
    }

    // MARK: - Brand + Registered transitions

    public func updateBrand(_ info: ForgeBrandInfo) {
        state.updateBrand(info)
    }

    public func confirmBrandAndContinue() {
        guard state.hasSpeedWinningVerification else { return }
        state.step = .registered
    }

    // MARK: - Publish

    public func updatePublishOptions(_ options: ForgePublishOptions) {
        state.updatePublish(options)
    }

    public func openPublishStage() {
        guard state.hasSpeedWinningVerification else {
            publishFailure = tr("Forge only publishes models that proved an MTP speed win on this Mac.")
            return
        }
        guard let path = completedLocalPath, !path.isEmpty else {
            publishFailure = tr("No completed forge to publish.")
            return
        }
        publishFailure = nil
        state.step = .publishing
    }

    /// Seed the wizard from an existing local entry (used by the
    /// My Models browser's "Publish to HF" action) and jump straight
    /// into the publish form. Idempotent — repeated invocations are
    /// safe because resetWizard cancels any in-flight tasks. Existing
    /// local models must hydrate verification from mtplx_runtime.json;
    /// otherwise the publish screen would render D0 / 1.00x and reject
    /// an artifact that the My Models card already proved is accelerated.
    public func startPublishForExistingForge(brandedName: String, localPath: String) {
        resetWizard()
        state.brand.brandedName = brandedName.trimmingCharacters(in: .whitespacesAndNewlines)
        completedLocalPath = localPath
        hydrateExistingForgeMetadata(localPath: localPath)
        state.step = .publishing
        selectSubMode(.create)
        if state.hasSpeedWinningVerification {
            publishFailure = nil
        } else {
            publishFailure = tr("Forge only publishes models that proved an MTP speed win on this Mac.")
        }
    }

    private func hydrateExistingForgeMetadata(localPath: String) {
        let runtimePath = URL(fileURLWithPath: localPath)
            .appendingPathComponent("mtplx_runtime.json")
            .path
        guard let metadata = MTPLXRuntimeMetadata.read(at: runtimePath) else { return }
        brandedRuntimeMetadata = metadata
        if let verification = Self.extractForgeVerification(from: metadata.rawJSON) {
            state.recordVerification(verification)
        }
        if let provenance = metadata.forgeProvenance {
            state.sourceProbe = ForgeSourceProbe(
                verdict: .forgeable,
                hfRepo: provenance.sourceRepo,
                sourceFormat: provenance.sourceFormat,
                hasMtpWeights: true,
                message: tr("Loaded from local Forge metadata.")
            )
            state.recipe = provenance.forgeRecipe
        }
    }

    public func startPublish() {
        guard state.hasSpeedWinningVerification else {
            publishFailure = tr("Forge only publishes models that proved an MTP speed win on this Mac.")
            return
        }
        guard let path = completedLocalPath, !path.isEmpty else {
            publishFailure = tr("No completed forge to publish.")
            return
        }
        let trimmedRepo = state.publish.repoName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRepo.isEmpty else {
            publishFailure = tr("Repository name (owner/name) is required.")
            return
        }
        guard let token = hfTokenStore.load(), !token.isEmpty else {
            publishFailure = tr("Hugging Face token missing. Paste a write token to publish.")
            return
        }
        isPublishing = true
        publishProgress = nil
        publishFailure = nil
        state.step = .publishing

        let runID = "mtplx-forge-publish-" + UUID().uuidString.lowercased().prefix(8)
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mtplx-forge-publish", isDirectory: true)
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let request = HFPublisher.PublishRequest(
            localPath: path,
            repo: trimmedRepo,
            visibility: state.publish.visibility,
            licenseSpdx: state.publish.licenseSPDX,
            readmeBody: state.publish.readmeBody.isEmpty ? nil : state.publish.readmeBody,
            outputDir: outputDir.path,
            runID: String(runID)
        )

        let publisher = hfPublisher
        publishTask?.cancel()
        publishTask = Task { [weak self] in
            for await event in publisher.stream(request, token: token) {
                if Task.isCancelled { break }
                await MainActor.run {
                    self?.handlePublishEvent(event)
                }
            }
        }
    }

    private func handlePublishEvent(_ event: HFPublishEvent) {
        switch event {
        case .started:
            publishProgress = ForgePhaseProgress(phase: .publish, progress: 0, label: tr("uploading"))
        case .progress(let bytes, let total, let mbps):
            let fraction: Double
            if let total, total > 0 {
                fraction = min(1, Double(bytes) / Double(total))
            } else {
                fraction = 0
            }
            publishProgress = ForgePhaseProgress(
                phase: .publish,
                progress: fraction,
                label: mbps > 0 ? String(format: "%.1f MB/s", mbps) : "uploading"
            )
        case .repoCreated:
            // Surface in the progress label so the user knows the
            // server-side creation worked even before the first byte
            // of the upload completes.
            if var current = publishProgress {
                current.label = tr("repo created — uploading")
                publishProgress = current
            }
        case .completed(_, _):
            publishProgress = ForgePhaseProgress(phase: .publish, progress: 1, label: tr("uploaded"), finished: true)
            isPublishing = false
        case .failed(_, let stderrTail):
            publishFailure = stderrTail.isEmpty ? tr("Publish failed.") : stderrTail
            isPublishing = false
        case .cancelled:
            isPublishing = false
        case .backendNotAvailable:
            backendUnavailable = true
            publishFailure = tr("Forge backend not available — install MTPLX 1.x to publish.")
            isPublishing = false
        }
    }

    public var savedHFToken: String? {
        hfTokenStore.load()
    }

    public func saveHFToken(_ token: String) -> Bool {
        hfTokenStore.save(token)
    }

    public func deleteHFToken() {
        hfTokenStore.delete()
    }

    public func cancelPublish() {
        publishTask?.cancel()
        publishTask = nil
        isPublishing = false
        if state.step == .publishing { state.step = .registered }
    }

    // MARK: - Backend availability

    public func checkBackendAvailability() async {
        let available = await forgeBuilder.backendAvailable()
        await MainActor.run {
            self.backendUnavailable = !available
        }
    }

    // MARK: - Orphan-run detection (resilience step 20)

    /// Scans the canonical forge tmp roots for run dirs that lack a
    /// final `forge.json` — i.e. builds the user (or a crash) abandoned
    /// mid-way. Populates `orphanRuns` so the Forge tab can show a
    /// Resume / Discard banner.
    public func scanForOrphanRuns() {
        let roots = [
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("mtplx-forge", isDirectory: true),
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("mtplx-forge-publish", isDirectory: true)
        ]
        let fm = FileManager.default
        var found: [ForgeOrphanRun] = []
        for root in roots {
            guard let entries = try? fm.contentsOfDirectory(atPath: root.path) else { continue }
            for name in entries {
                let runDir = root.appendingPathComponent(name, isDirectory: true)
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: runDir.path, isDirectory: &isDir), isDir.boolValue else { continue }
                let final = runDir.appendingPathComponent("forge.json").path
                if fm.fileExists(atPath: final) { continue }
                let attrs = try? fm.attributesOfItem(atPath: runDir.path)
                let modified = attrs?[.modificationDate] as? Date ?? .distantPast
                let last = ForgeOrphanRun.lastWrittenPhase(in: runDir)
                found.append(ForgeOrphanRun(
                    runID: name,
                    runDir: runDir.path,
                    lastModified: modified,
                    lastWrittenPhase: last
                ))
            }
        }
        // Most-recent first so the Resume banner names the latest one.
        orphanRuns = found.sorted { $0.lastModified > $1.lastModified }
    }

    /// Hard-discards every orphan run dir (recursive delete). Called
    /// when the user picks "Discard all" on the resume banner.
    public func discardOrphanRuns() {
        let fm = FileManager.default
        for run in orphanRuns {
            try? fm.removeItem(atPath: run.runDir)
        }
        orphanRuns = []
    }

    /// Hard-discards a single orphan run dir. The banner's expanded
    /// list calls this per-row so the user can clean up the noise
    /// without losing the rest.
    public func discardOrphanRun(_ run: ForgeOrphanRun) {
        let fm = FileManager.default
        try? fm.removeItem(atPath: run.runDir)
        orphanRuns.removeAll { $0.id == run.id }
    }

    /// Reveals an orphan's run directory in Finder so the user can
    /// inspect the partial artifact (useful while the backend doesn't
    /// support resume — they can at least see what's there).
    public func revealOrphanRun(_ run: ForgeOrphanRun) {
        #if canImport(AppKit)
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: run.runDir)])
        #endif
    }

    // MARK: - Cleanup

    public func cancelAll() {
        hardwareTask?.cancel()
        probeTask?.cancel()
        buildTask?.cancel()
        publishTask?.cancel()
        isProbing = false
        isBuilding = false
        isPublishing = false
    }
}

// MARK: - ForgeOrphanRun

public struct ForgeOrphanRun: Identifiable, Equatable, Sendable {
    public var id: String { runDir }
    public var runID: String
    public var runDir: String
    public var lastModified: Date
    public var lastWrittenPhase: ForgePhase?

    public init(runID: String, runDir: String, lastModified: Date, lastWrittenPhase: ForgePhase? = nil) {
        self.runID = runID
        self.runDir = runDir
        self.lastModified = lastModified
        self.lastWrittenPhase = lastWrittenPhase
    }

    /// Mirrors ForgeBuilder.lastWrittenPhase. Public here so the
    /// resume banner can surface "died during Calibrate" instead of
    /// a generic "abandoned" message.
    static func lastWrittenPhase(in runDir: URL) -> ForgePhase? {
        let fm = FileManager.default
        var best: (ForgePhase, Date)?
        for phase in ForgePhase.allCases {
            let path = runDir.appendingPathComponent(phase.rawValue + ".json").path
            if let attrs = try? fm.attributesOfItem(atPath: path),
               let modified = attrs[.modificationDate] as? Date
            {
                if best == nil || modified > best!.1 {
                    best = (phase, modified)
                }
            }
        }
        return best?.0
    }
}

// MARK: - Sampler parse helper

extension MTPLXForgeProvenance {
    /// Parses the `sampler` sub-dict (shape `{ temperature, top_p,
    /// top_k }`) from an arbitrary runtime metadata blob.
    static func parseSampler(_ value: Any?) -> ForgeSampler {
        guard let dict = value as? [String: Any] else { return ForgeSampler() }
        let temperature = (dict["temperature"] as? Double) ?? 0.6
        let topP = (dict["top_p"] as? Double) ?? 0.95
        let topK = (dict["top_k"] as? Int) ?? 20
        return ForgeSampler(temperature: temperature, topP: topP, topK: topK)
    }
}
