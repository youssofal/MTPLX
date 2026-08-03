import Foundation

// MARK: - OnboardingStep
//
// The six-step linear flow. Order is canonical; `goNext` / `goBack`
// walk the case array in declaration order so the enum doubles as the
// progress indicator's source of truth.
//
// `runtimeSetup` sits before download on purpose: the download and
// tune steps both shell the `mtplx` CLI, so the runtime (plus fan
// control and any global-CLI sync) must be in place before either —
// previously the install hid inside the download/tune fallbacks and
// "Skip tune" could finish onboarding with no runtime installed.

public enum OnboardingStep: String, CaseIterable, Equatable, Sendable {
    case welcome
    case hardwareScan = "hardware_scan"
    case modelPick = "model_pick"
    case runtimeSetup = "runtime_setup"
    case download
    case tune
}

// MARK: - ModelPickChoice
//
// What the user picked on the model-pick step. The Speed and Quality
// cases name the curated catalog entry directly; M1/M2 Speed → FP16
// routing is resolved by `OnboardingFeatureState.resolvedModel`, not
// by changing the case value itself, so the user's intent ("I want
// Speed") survives the swap.

public enum ModelPickChoice: Equatable, Sendable, Hashable {
    case none
    case curatedQwen35FourBit
    case curatedQwen35NineBSpeed
    case curatedSpeedV2
    case curatedSpeed
    case curatedQwen35BSpeed
    case curatedQwen35BBalance
    case curatedQuality
    case curatedGemmaSpeed
    case curatedStepFlash
    case other(hfRepo: String)
    case local(path: String)
}

// MARK: - OtherModelProbe
//
// Outcome of probing a user-supplied HuggingFace repository for MTP
// compatibility BEFORE downloading multi-gigabyte weights. Mirrors the
// daemon's four-tier compatibility verdict in
// `mtplx/ui/onboarding.py:_classify_scanned_model` but for remote
// repositories: we read `config.json` and the file tree, never the
// safetensors themselves.

public struct OtherModelProbe: Equatable, Sendable {
    public enum Verdict: String, Equatable, Sendable {
        /// Architecture declares MTP heads AND `mtp.safetensors`
        /// (or alternate sidecar path) is published in the repo.
        case ready
        /// Architecture declares MTP but the sidecar weights are
        /// not in the published file tree. Speed will drop to
        /// standard autoregressive decoding.
        case missingSidecar
        /// Architecture has no MTP heads. Loses speculative
        /// decoding entirely; requires user confirmation.
        case noMTP
        /// Probe failed (network, 404, auth, malformed config).
        case probeFailed
    }

    public var verdict: Verdict
    public var hfRepo: String
    /// One-sentence explanation to render under the result strip.
    public var message: String
    /// Surfaced when the probe failed (network error / 404 / 401).
    public var diagnostic: String?

    public init(
        verdict: Verdict,
        hfRepo: String,
        message: String,
        diagnostic: String? = nil
    ) {
        self.verdict = verdict
        self.hfRepo = hfRepo
        self.message = message
        self.diagnostic = diagnostic
    }
}

// MARK: - LocalModelProbe
//
// Outcome of validating a user-supplied local model directory before
// onboarding tries to launch or tune it. Local folders must satisfy the
// same "complete MTPLX install" contract as catalog installs: config,
// tokenizer/runtime metadata, full trunk weights, and an MTP sidecar.

public struct LocalModelProbe: Equatable, Sendable {
    public enum Verdict: String, Equatable, Sendable {
        case ready
        case notFound
        case incomplete
    }

    public var verdict: Verdict
    public var path: String
    public var message: String
    public var diagnostic: String?

    public init(
        verdict: Verdict,
        path: String,
        message: String,
        diagnostic: String? = nil
    ) {
        self.verdict = verdict
        self.path = path
        self.message = message
        self.diagnostic = diagnostic
    }
}

// MARK: - OnboardingFeatureState
//
// Pure value state machine for the onboarding flow. No `ObservableObject`,
// no `Combine`, no service dependencies — trivial to construct, mutate
// from a test, and compare for equality.
//
// Live progress (download bytes, tune candidate landings) is intentionally
// NOT held here; that lives on `OnboardingOrchestrator` as `@Published`
// properties because it changes on a sub-second cadence and would
// invalidate state equality checks otherwise.

public struct OnboardingFeatureState: Equatable, Sendable {
    public var step: OnboardingStep
    public var hardware: DetectedHardware?
    public var pick: ModelPickChoice
    public var otherProbe: OtherModelProbe?
    public var localProbe: LocalModelProbe?
    /// User explicitly opted to continue past a `.noMTP` warning.
    /// Resets to false whenever `pick` changes.
    public var hasAcknowledgedOtherWarning: Bool

    public init(
        step: OnboardingStep = .welcome,
        hardware: DetectedHardware? = nil,
        pick: ModelPickChoice = .none,
        otherProbe: OtherModelProbe? = nil,
        localProbe: LocalModelProbe? = nil,
        hasAcknowledgedOtherWarning: Bool = false
    ) {
        self.step = step
        self.hardware = hardware
        self.pick = pick
        self.otherProbe = otherProbe
        self.localProbe = localProbe
        self.hasAcknowledgedOtherWarning = hasAcknowledgedOtherWarning
    }

    // MARK: Derived

    /// The catalog entry the user should actually download, after
    /// applying chip-aware swaps. Speed on M1/M2 resolves to the FP16
    /// variant; everything else passes through unchanged. Returns
    /// `nil` for `.other` (the user-pasted repo isn't a catalog entry).
    public var resolvedModel: MTPLXModelOption? {
        let catalog = MTPLXModelOption.officialCatalog
        switch pick {
        case .none:
            return nil
        case .curatedQwen35FourBit:
            return catalog.first { $0.id == "qwen35-4b-optimized-speed" }
        case .curatedQwen35NineBSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen35-9b-optimized-speed-fp16" : "qwen35-9b-optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedSpeedV2:
            return catalog.first { $0.id == "optimized-speed-v2" }
        case .curatedSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "optimized-speed-fp16" : "optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedQwen35BSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen36-35b-a3b-optimized-speed-fp16" : "qwen36-35b-a3b-optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedQwen35BBalance:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen36-35b-a3b-optimized-balance-fp16" : "qwen36-35b-a3b-optimized-balance"
            return catalog.first { $0.id == id }
        case .curatedQuality:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "optimized-quality-fp16" : "optimized-quality"
            return catalog.first { $0.id == id }
        case .curatedGemmaSpeed:
            return catalog.first { $0.id == "gemma4-optimized-speed" }
        case .curatedStepFlash:
            return nil
        case .other, .local:
            return nil
        }
    }

    /// Repo identifier ultimately sent to `mtplx pull`. For curated
    /// picks this is `resolvedModel?.hfModelID`; for `.other` it is
    /// the raw user input. `nil` only if the state is impossibly
    /// inconsistent (e.g. `.other("")`).
    public var resolvedRepoID: String? {
        switch pick {
        case .none:
            return nil
        case .curatedQwen35FourBit,
             .curatedQwen35NineBSpeed,
             .curatedSpeedV2,
             .curatedSpeed,
             .curatedQwen35BSpeed,
             .curatedQwen35BBalance,
             .curatedQuality,
             .curatedGemmaSpeed,
             .curatedStepFlash:
            return resolvedModel?.hfModelID
        case .other(let repo):
            let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        case .local(let path):
            let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
    }

    public var resolvedModelFamily: String {
        if let resolvedModel {
            return resolvedModel.modelFamily
        }
        guard let repo = resolvedRepoID else { return "unknown" }
        return MTPLXModelOption.modelFamily(for: repo)
    }

    public var supportsTune: Bool {
        MTPLXModelOption.supportsOnboardingTune(family: resolvedModelFamily)
    }

    public var tuneCandidates: [TuneCandidate] {
        TuneCandidate.candidates(forFamily: resolvedModelFamily)
    }

    // MARK: Transitions

    public mutating func goNext() {
        let all = OnboardingStep.allCases
        guard let i = all.firstIndex(of: step), i < all.count - 1 else { return }
        step = all[i + 1]
    }

    public mutating func goBack() {
        let all = OnboardingStep.allCases
        guard let i = all.firstIndex(of: step), i > 0 else { return }
        step = all[i - 1]
    }

    public mutating func select(_ choice: ModelPickChoice) {
        guard pick != choice else { return }
        pick = choice
        otherProbe = nil
        localProbe = nil
        hasAcknowledgedOtherWarning = false
    }

    public mutating func record(_ probe: OtherModelProbe) {
        otherProbe = probe
        // A fresh probe always invalidates a previous acknowledgement
        // so the user has to consciously re-confirm the warning.
        hasAcknowledgedOtherWarning = false
    }

    public mutating func record(_ probe: LocalModelProbe) {
        localProbe = probe
    }

    /// Whether the `Next` (or step-specific primary) button should be
    /// enabled on the current step. Service-driven steps (download,
    /// tune) gate themselves via orchestrator state and ignore this
    /// flag — it covers the user-input-driven cases only.
    public var canAdvance: Bool {
        switch step {
        case .welcome:
            return true
        case .hardwareScan:
            return hardware != nil
        case .modelPick:
            switch pick {
            case .none:
                return false
            case .curatedQwen35FourBit,
                 .curatedQwen35NineBSpeed,
                 .curatedSpeedV2,
                 .curatedSpeed,
                 .curatedQwen35BSpeed,
                 .curatedQwen35BBalance,
                 .curatedQuality,
                 .curatedGemmaSpeed,
                 .curatedStepFlash:
                return resolvedModel != nil
            case .other:
                guard let probe = otherProbe else { return false }
                switch probe.verdict {
                case .ready, .missingSidecar:
                    return true
                case .noMTP:
                    return hasAcknowledgedOtherWarning
                case .probeFailed:
                    return false
                }
            case .local:
                guard let probe = localProbe else { return false }
                return probe.verdict == .ready
            }
        case .runtimeSetup, .download, .tune:
            return false
        }
    }
}
