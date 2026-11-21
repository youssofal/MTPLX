import Foundation

// MARK: - OnboardingStep
//
// The seven-step linear flow. Order is canonical; `goNext` / `goBack`
// walk the case array in declaration order so the enum doubles as the
// progress indicator's source of truth.
//
// `language` comes first so every later step already renders in the
// language the user picked; the choice persists through LanguageStore
// and can be changed again in Settings.
//
// `runtimeSetup` sits before download on purpose: the download and
// tune steps both shell the `mtplx` CLI, so the runtime (plus fan
// control and any global-CLI sync) must be in place before either —
// previously the install hid inside the download/tune fallbacks and
// "Skip tune" could finish onboarding with no runtime installed.

public enum OnboardingStep: String, CaseIterable, Equatable, Sendable {
    case language
    case welcome
    case hardwareScan = "hardware_scan"
    case modelPick = "model_pick"
    case runtimeSetup = "runtime_setup"
    case download
    case tune

    /// Position in the canonical flow, for the "Step N of M" capsule.
    public var index: Int {
        Self.allCases.firstIndex(of: self) ?? 0
    }
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
    case curatedQwen35FourBQuality
    case curatedBonsaiOptimizedSpeed
    case curatedFlashNextOptimizedQuality
    case curatedMiMoQwen9BOptimizedSpeed
    case curatedQwen35NineBSpeed
    case curatedQwen38OptimizedSpeed
    case curatedQwen38BareSpeed
    case curatedQwen38OptimizedQuality
    case curatedFlashNextBareSpeed
    case curatedFlashNextOptimizedSpeed
    case curatedSpeedV2
    case curatedSpeed
    case curatedQwen35BSpeed
    case curatedQwen35BBalance
    case curatedQuality
    case curatedGemmaSpeed
    case curatedInklingBareSpeed
    case curatedInklingOptimizedSpeed
    case curatedInklingOptimizedQuality
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

    public init(
        step: OnboardingStep = .language,
        hardware: DetectedHardware? = nil,
        pick: ModelPickChoice = .none,
        otherProbe: OtherModelProbe? = nil,
        localProbe: LocalModelProbe? = nil
    ) {
        self.step = step
        self.hardware = hardware
        self.pick = pick
        self.otherProbe = otherProbe
        self.localProbe = localProbe
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
        case .curatedQwen35FourBQuality:
            return catalog.first { $0.id == "qwen35-4b-optimized-quality" }
        case .curatedBonsaiOptimizedSpeed:
            return catalog.first { $0.id == "bonsai-2-27b-optimized-speed" }
        case .curatedFlashNextOptimizedQuality:
            return catalog.first { $0.id == "flash-next-optimized-quality" }
        case .curatedMiMoQwen9BOptimizedSpeed:
            // Modern-tier only with no FP16 sibling, so no legacy swap.
            return catalog.first { $0.id == "mimo-v26-qwen-9b-optimized-speed" }
        case .curatedQwen35NineBSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen35-9b-optimized-speed-fp16" : "qwen35-9b-optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedQwen38OptimizedSpeed:
            // The 3.8 trio has FP16 precision siblings (2026-08-15): same
            // quantized packs, 16-bit tensors cast to fp16, so M1/M2 Macs
            // get the identical model on the chip's native precision.
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen38-27b-optimized-speed-fp16" : "qwen38-27b-optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedQwen38BareSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen38-27b-bare-speed-fp16" : "qwen38-27b-bare-speed"
            return catalog.first { $0.id == id }
        case .curatedQwen38OptimizedQuality:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "qwen38-27b-optimized-quality-fp16" : "qwen38-27b-optimized-quality"
            return catalog.first { $0.id == id }
        case .curatedFlashNextBareSpeed:
            // Flash-Next is modern-tier only with no FP16 sibling, so
            // there is no legacy swap — the pair passes through unchanged.
            return catalog.first { $0.id == "flash-next-bare-speed" }
        case .curatedFlashNextOptimizedSpeed:
            return catalog.first { $0.id == "flash-next-optimized-speed" }
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
        case .curatedInklingBareSpeed:
            // The Inkling trio has FP16 precision siblings, offered on M1/M2 Macs
            // (MTPLXModelOption.inklingLegacyOffer), as the Qwen 3.8 trio does.
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "inkling2-mini-bare-speed-fp16" : "inkling2-mini-bare-speed"
            return catalog.first { $0.id == id }
        case .curatedInklingOptimizedSpeed:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "inkling2-mini-optimized-speed-fp16" : "inkling2-mini-optimized-speed"
            return catalog.first { $0.id == id }
        case .curatedInklingOptimizedQuality:
            let useFP16 = hardware?.tier == .legacyApple
            let id = useFP16 ? "inkling2-mini-optimized-quality-fp16" : "inkling2-mini-optimized-quality"
            return catalog.first { $0.id == id }
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
                 .curatedQwen35FourBQuality,
                 .curatedBonsaiOptimizedSpeed,
                 .curatedFlashNextOptimizedQuality,
                 .curatedMiMoQwen9BOptimizedSpeed,
             .curatedQwen35NineBSpeed,
             .curatedQwen38OptimizedSpeed,
             .curatedQwen38BareSpeed,
             .curatedQwen38OptimizedQuality,
             .curatedFlashNextBareSpeed,
             .curatedFlashNextOptimizedSpeed,
             .curatedSpeedV2,
             .curatedSpeed,
             .curatedQwen35BSpeed,
             .curatedQwen35BBalance,
             .curatedQuality,
             .curatedGemmaSpeed,
             .curatedInklingBareSpeed,
             .curatedInklingOptimizedSpeed,
             .curatedInklingOptimizedQuality,
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
    }

    public mutating func record(_ probe: OtherModelProbe) {
        otherProbe = probe
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
        case .language, .welcome:
            return true
        case .hardwareScan:
            return hardware != nil
        case .modelPick:
            switch pick {
            case .none:
                return false
            case .curatedQwen35FourBit,
                 .curatedQwen35FourBQuality,
                 .curatedBonsaiOptimizedSpeed,
                 .curatedFlashNextOptimizedQuality,
                 .curatedMiMoQwen9BOptimizedSpeed,
                 .curatedQwen35NineBSpeed,
                 .curatedQwen38OptimizedSpeed,
                 .curatedQwen38BareSpeed,
                 .curatedQwen38OptimizedQuality,
                 .curatedFlashNextBareSpeed,
                 .curatedFlashNextOptimizedSpeed,
                 .curatedSpeedV2,
                 .curatedSpeed,
                 .curatedQwen35BSpeed,
                 .curatedQwen35BBalance,
                 .curatedQuality,
                 .curatedGemmaSpeed,
                 .curatedInklingBareSpeed,
                 .curatedInklingOptimizedSpeed,
                 .curatedInklingOptimizedQuality,
                 .curatedStepFlash:
                return resolvedModel != nil
            case .other:
                guard let probe = otherProbe else { return false }
                switch probe.verdict {
                case .ready, .missingSidecar, .noMTP:
                    // MTP unavailable is informational, never a gate
                    // (founder directive 2026-08-26): the engine serves
                    // MTP-less checkpoints autoregressive, so the app
                    // advances the same way the CLI does.
                    return true
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
