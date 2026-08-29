import SwiftUI
import MTPLXAppCore

// MARK: - OnboardingExperienceView
//
// Root of the first-launch onboarding flow. Renders the current step
// inside the shared `OnboardingStepContainer` with a horizontal
// slide-in / slide-out transition between steps so the flow feels
// linear instead of a flicker-cut.
//
// Owns the `OnboardingOrchestrator` as `@StateObject` — it lives and
// dies with this view. `cancelAll()` on disappear tears down any
// in-flight subprocess (download / tune) so the user can quit
// onboarding mid-flow without leaving zombie processes.
//
// The final "Start chatting" handler (`completeOnboarding`) is
// declared here and threaded into `TuneStep`. It persists the
// onboarding-completed timestamp and the tuned depth, flips the
// router into chat mode, then asks the backend to start the daemon
// for `.chat`.

struct OnboardingExperienceView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var themeStore: ThemeStore
    @StateObject private var orchestrator = OnboardingOrchestrator()

    var body: some View {
        ZStack {
            Brand.bgOuter.ignoresSafeArea()
            Group {
                switch orchestrator.state.step {
                case .welcome:
                    WelcomeStep(orchestrator: orchestrator)
                        .transition(stepTransition)
                case .hardwareScan:
                    HardwareScanStep(orchestrator: orchestrator)
                        .transition(stepTransition)
                case .modelPick:
                    ModelPickStep(orchestrator: orchestrator)
                        .transition(stepTransition)
                case .runtimeSetup:
                    RuntimeSetupStep(orchestrator: orchestrator)
                        .transition(stepTransition)
                case .download:
                    DownloadStep(orchestrator: orchestrator)
                        .transition(stepTransition)
                case .tune:
                    TuneStep(orchestrator: orchestrator) {
                        Task { await completeOnboarding() }
                    }
                    .transition(stepTransition)
                }
            }
            .animation(stepAnimation, value: orchestrator.state.step)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .focusable(true)
        // Keyboard focus is only here so onKeyPress(.escape) fires;
        // without this, macOS draws its focus ring around the whole
        // window the moment the view is clicked.
        .focusEffectDisabled()
        .onKeyPress(.escape) {
            // Esc steps backwards through the flow — never closes the
            // onboarding window since that would leave the app in an
            // unrecoverable "no model, no daemon" state.
            if orchestrator.state.step != .welcome
                && !orchestrator.isRunningRuntimeSetup
                && !orchestrator.isDownloading
                && !orchestrator.isTuning
            {
                orchestrator.goBack()
                return .handled
            }
            return .ignored
        }
        .onAppear {
            orchestrator.restoreDownloadPreferences(
                modelDirectory: backend.configuration.modelDirectory,
                hfMirrorEndpoint: backend.configuration.hfEndpoint
            )
        }
        .onDisappear { orchestrator.cancelAll() }
    }

    /// Step transition: outgoing card eases out with a slight scale-down
    /// + fade, incoming card eases in scaled-up from 0.96 with a fade.
    /// Feels like a card being lifted away and a new one rising in,
    /// instead of the original PowerPoint slide.
    private var stepTransition: AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .scale(scale: 0.96, anchor: .center)),
            removal: .opacity.combined(with: .scale(scale: 1.02, anchor: .center))
        )
    }

    private var stepAnimation: Animation? {
        themeStore.reduceMotionPreference
            ? nil
            : .spring(response: 0.42, dampingFraction: 0.86)
    }

    // MARK: - Completion handoff

    /// "Open Dashboard" handler — persists the onboarding-completed
    /// flag, the tuned depth, and the resolved model path, then drops
    /// the user onto the existing dashboard shell. We do NOT auto-
    /// start the daemon: the user clicks the Play button themselves
    /// (matches the rest of the app where the daemon launch is always
    /// an explicit action). `lastLaunchTarget = .chat` so when they
    /// do click Play, Chat is pre-selected in the LaunchOverlay —
    /// they're one click from chatting if that's what they want.
    private func completeOnboarding() async {
        var config = backend.configuration
        config.onboardingCompletedAt = Date()
        config.lastLaunchTarget = LaunchTarget.chat.rawValue
        config.modelDirectory = MTPLXAppConfiguration.normalizedModelDirectory(
            orchestrator.modelDirectory
        )
        if MTPLXAppConfiguration.hfMirrorEnvironment(orchestrator.hfMirrorEndpoint) != nil {
            config.hfEndpoint = orchestrator.hfMirrorEndpoint
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let tuned = orchestrator.tuneResult, tuned.bestDepth >= 1 {
            let tunedAt = Date()
            let modelID = orchestrator.state.resolvedModel?.hfModelID
                ?? orchestrator.state.resolvedRepoID
                ?? config.model
            let modelFamily = orchestrator.state.resolvedModelFamily
            let controlField = Self.tuneControlField(for: modelFamily)
            let backendID = Self.backendID(for: modelFamily)
            if MTPLXModelOption.supportsTune(family: modelFamily) {
                config.lastTunedDepth = tuned.bestDepth
            } else {
                config.lastTunedDepth = nil
            }
            if !tuned.allCandidates.isEmpty {
                config.lastTunedAt = tunedAt
                config.tunedControlRecord = TunedControlRecord(
                    modelID: modelID,
                    modelFamily: modelFamily,
                    backendID: backendID,
                    controlField: controlField,
                    controlValue: tuned.bestDepth,
                    candidates: orchestrator.tuneCandidates.map(\.displayLabel),
                    tunedAt: tunedAt
                )
            }
        }
        if let downloaded = orchestrator.downloadProgress,
           downloaded.isComplete,
           MTPLXModelOption.hasCompleteInstall(at: downloaded.destinationPath) {
            config.model = downloaded.destinationPath
            if let repo = orchestrator.state.resolvedRepoID,
               orchestrator.state.resolvedModel == nil {
                config.rememberCustomModel(repoID: repo)
            }
        } else if let model = orchestrator.state.resolvedModel {
            // Use the local path ONLY when the install completeness
            // check actually succeeds. `installedLocalPath` returns
            // the first candidate dir that EXISTS — a metadata-only
            // hf-staging stub would otherwise win and feed the daemon
            // an unloadable path (saw this with the Quality stub:
            // `mtp.safetensors` missing, daemon went degraded). Fall
            // back to the HF id so the daemon's `resolve_model_path`
            // surfaces a clear "Model not cached. Run: mtplx pull"
            // error instead of a silent failure to load weights.
            if let local = orchestrator.installedModelPath(for: model) {
                config.model = local
            } else {
                config.model = model.hfModelID
            }
        } else if let repo = orchestrator.state.resolvedRepoID {
            config.model = repo
            config.rememberCustomModel(repoID: repo)
        }
        try? backend.saveSettings(config)

        router.onboardingPhase = .completed
        router.primaryMode = .dashboard
    }

    private static func tuneControlField(for family: String) -> String {
        family == "gemma4" ? "draft_block_size" : "depth"
    }

    private static func backendID(for family: String) -> String {
        switch family {
        case "gemma4": return "gemma4_assistant"
        case "step": return "step3p5_mtp"
        default: return "qwen3_next"
        }
    }
}
