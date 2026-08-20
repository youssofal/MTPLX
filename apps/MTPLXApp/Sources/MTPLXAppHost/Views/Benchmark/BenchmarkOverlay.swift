import SwiftUI
import MTPLXAppCore

// MARK: - BenchmarkOverlay
//
// Full-screen surface for the AIME 2026 benchmark. Mounted in
// ContentView's `appShell` ZStack (above the dashboard, below the
// drop-down launcher) and gated on `router.benchmarkOverlayPresented`.
// State lives on the `BenchmarkOrchestrator` hoisted to `MTPLXApp`
// root, so closing this overlay does NOT cancel an in-flight run.
//
// Jet Chrome pass: the 1,818-line monolith was split into 18 sibling
// files under Benchmark/ (BenchHeader, BenchHeaderStat,
// BenchPrimaryCTA, BenchQuestionGrid, BenchQuestionTile,
// BenchTilePulseModifier, BenchLiveCard, BenchTelemetryCell,
// MathProblemRender, TextMathRuns, ReasoningStreamView,
// BenchEmptyState, BenchHistoryRow, BenchSummaryCard,
// BenchConfettiView, BenchConfettiParticle, BenchPausePendingBanner,
// BenchErrorBanner) so each type honours the swiftui-pro views.md
// "each type in its own file" rule. This file is now the shell only:
// the surface chrome (routed through the shared `PanelChrome`
// primitive), the entrance / exit choreography, and the
// state-driven body switch.

struct BenchmarkOverlay: View {
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var orchestrator: BenchmarkOrchestrator
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    /// Stroke-trim progress 0..1 for the rounded-rect outline. Drives
    /// the "border draws in clockwise" reveal pattern shared with
    /// LaunchOverlay / InferenceParamsOverlay / ModelPickerOverlay.
    @State private var borderProgress: CGFloat = 0
    @State private var contentVisible: Bool = false
    @State private var startPending: Bool = false
    /// Finished question the user tapped to review (nil = no review open).
    @State private var selectedResult: BenchQuestionResult?

    private var motionEnabled: Bool {
        !backend.configuration.performanceLock && !themeStore.reduceMotionPreference
    }

    /// The benchmark fills the whole app window with a small inset on
    /// every edge. The previous "centred mini-panel inside the window"
    /// treatment wasted ~30% of the available area and forced every
    /// piece of content to fight for space; running a 30-problem grid
    /// + a live reasoning stream + a hero answer + telemetry strip
    /// inside an 880pt-wide box was always going to clip something.
    private let edgeInset: CGFloat = 16
    /// Extra top inset so the panel sits clear of the macOS traffic
    /// lights. The window is `.hiddenTitleBar`, so the red/yellow/green
    /// dots float over the top-left of whatever is on screen — with a
    /// flush panel they collided with the overlay's own close button.
    /// Pushing the panel down drops the whole top-left zone (wordmark +
    /// controls) below the OS dots.
    private let topInset: CGFloat = 36
    private var panelCornerRadius: CGFloat { Brand.Radii.panel }
    private static let quickQuestionLimit = 3

    var body: some View {
        GeometryReader { geo in
            let panelWidth = max(0, geo.size.width - edgeInset * 2)
            let panelHeight = max(0, geo.size.height - topInset - edgeInset)

            ZStack(alignment: .top) {
                backdrop

                panel(width: panelWidth, height: panelHeight)
                    .frame(width: panelWidth, height: panelHeight)
                    .background(panelChrome)
                    .clipShape(RoundedRectangle(cornerRadius: panelCornerRadius, style: .continuous))
                    .opacity(contentVisible ? 1 : 0)
                    .scaleEffect(contentVisible ? 1.0 : 0.99)
                    .animation(motionEnabled ? .spring(response: 0.42, dampingFraction: 0.86) : nil,
                               value: contentVisible)
                    .padding(.top, topInset)
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .ignoresSafeArea()
        .onAppear { runEnter() }
        .onDisappear { runExit() }
        .onChange(of: router.benchmarkOverlayPresented) { _, presented in
            if presented {
                runEnter()
            } else {
                runExit()
            }
        }
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.escape) {
            handleClose()
            return .handled
        }
        .sheet(item: $selectedResult) { result in
            BenchQuestionDetail(result: result, total: orchestrator.total) {
                selectedResult = nil
            }
        }
    }

    // MARK: - Layers

    @ViewBuilder
    private var backdrop: some View {
        Rectangle()
            .fill(Brand.bgOuter.opacity(0.78))
            .background(.ultraThinMaterial)
            .ignoresSafeArea()
            .contentShape(Rectangle())
            .onTapGesture { handleClose() }
            .opacity(contentVisible ? 1 : 0)
            .animation(motionEnabled ? .smooth(duration: 0.22) : nil, value: contentVisible)
    }

    /// Returns the panel chrome with an explicit `contentWidth`
    /// threaded through so child views can size themselves against
    /// the actual rendered width instead of relying on SwiftUI's
    /// `maxWidth:` soft cap (which the LazyVGrid and FlowLayout
    /// happily ignore, expanding to their intrinsic widest layout
    /// and overflowing the panel edge).
    @ViewBuilder
    private func panel(width: CGFloat, height: CGFloat) -> some View {
        let horizontalPadding: CGFloat = 18
        let contentWidth = width - horizontalPadding * 2
        VStack(spacing: 0) {
            BenchHeader(
                state: orchestrator.state,
                elapsed: orchestrator.elapsed(),
                resolved: orchestrator.resolved,
                total: orchestrator.total,
                score: orchestrator.score,
                accuracy: orchestrator.accuracy,
                pausePending: orchestrator.pausePending,
                skipPending: orchestrator.skipPending,
                startTitle: startButtonTitle,
                startIcon: startPending ? "hourglass" : "play.fill",
                startEnabled: !startPending,
                performanceLock: backend.configuration.performanceLock,
                availableWidth: contentWidth,
                onClose: handleClose,
                onStart: { startBenchmark(resetFirst: orchestrator.state.isTerminal) },
                onQuickStart: {
                    startBenchmark(
                        questionLimit: Self.quickQuestionLimit,
                        resetFirst: orchestrator.state.isTerminal
                    )
                },
                onPause: { Task { await orchestrator.pause() } },
                onSkip: { Task { await orchestrator.skipCurrent() } },
                onResume: { Task { await orchestrator.resume() } },
                onCancel: { Task { await orchestrator.cancel(reason: "benchmark_header_cancel") } }
            )
            .frame(width: contentWidth, alignment: .leading)
            .padding(.horizontal, horizontalPadding)
            .padding(.top, 14)
            .padding(.bottom, 12)

            Divider().overlay(Brand.separator)

            // A live run FILLS the panel — the question grid takes its
            // natural height and the live card consumes the rest, so the
            // whole benchmark fits on one screen with NO outer scroll (the
            // reasoning trace scrolls inside its own box). Static states
            // (idle / summary) keep a scroll view because their history
            // list can legitimately exceed the panel.
            panelBody(contentWidth: contentWidth, horizontalPadding: horizontalPadding)
                .frame(width: width, alignment: .top)
                .frame(maxHeight: .infinity)
        }
        .frame(width: width, height: height)
        .clipped()
    }

    @ViewBuilder
    private func panelBody(contentWidth: CGFloat, horizontalPadding: CGFloat) -> some View {
        switch orchestrator.state {
        case .running, .paused:
            VStack(alignment: .leading, spacing: 12) {
                banners
                BenchQuestionGrid(results: orchestrator.results, onSelect: { selectedResult = $0 })
                    .frame(width: contentWidth, alignment: .leading)
                BenchLiveCard(
                    currentIdx: orchestrator.currentIdx,
                    total: orchestrator.total,
                    currentProblem: orchestrator.currentProblem,
                    extractedAnswer: orchestrator.liveExtractedAnswer,
                    answerIsGraded: orchestrator.liveAnswerIsGraded,
                    state: orchestrator.state,
                    currentRequestID: orchestrator.currentRequestID,
                    verificationState: orchestrator.currentVerification,
                    availableWidth: contentWidth,
                    liveTelemetry: orchestrator.liveTelemetry,
                    reasoningDocument: orchestrator.liveReasoningDocument,
                    answerDocument: orchestrator.liveAnswerDocument
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
            .frame(width: contentWidth, alignment: .leading)
            .padding(.horizontal, horizontalPadding)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        default:
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 12) {
                    banners
                    body(for: orchestrator.state, contentWidth: contentWidth)
                }
                .frame(width: contentWidth, alignment: .leading)
                .padding(.horizontal, horizontalPadding)
                .padding(.vertical, 14)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    /// Error + pause banners shared by the live and static layouts.
    @ViewBuilder
    private var banners: some View {
        if let error = orchestrator.lastError {
            BenchErrorBanner(message: error) {
                orchestrator.dismissError()
            }
        }
        if orchestrator.state == .paused || orchestrator.pausePending {
            BenchPausePendingBanner(
                pendingProblemIdx: orchestrator.currentIdx,
                onCancelPause: {
                    Task { await orchestrator.resume() }
                }
            )
        }
    }

    /// Panel chrome: the shared PanelChrome primitive (was a duplicated
    /// inline ZStack of LinearGradient + 3-stop chrome stroke). The
    /// border trim from 0 to `borderProgress` is layered on top so the
    /// "draws in clockwise" reveal stays in sync with the four other
    /// overlays (LaunchOverlay / InferenceParamsOverlay /
    /// ModelPickerOverlay) that already share `Motion.overlayBorder`.
    @ViewBuilder
    private var panelChrome: some View {
        ZStack {
            PanelChrome(cornerRadius: panelCornerRadius, elevation: Brand.Elevation.hi)
            RoundedRectangle(cornerRadius: panelCornerRadius, style: .continuous)
                .trim(from: 0, to: borderProgress)
                .stroke(
                    LinearGradient(
                        stops: [
                            .init(color: Color.white.opacity(0.16), location: 0.0),
                            .init(color: Color.white.opacity(0.05), location: 0.45),
                            .init(color: Color.white.opacity(0.03), location: 1.0)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: Brand.hairlineStrong
                )
        }
    }

    /// Static (non-live) state bodies. Running / paused are rendered by
    /// `panelBody`'s fill layout instead, so they never reach here.
    @ViewBuilder
    private func body(for state: BenchRunState, contentWidth: CGFloat) -> some View {
        switch state {
        case .idle:
            BenchEmptyState(
                history: orchestrator.history,
                runTitle: startButtonTitle,
                runIcon: startPending ? "hourglass" : "play.fill",
                isRunEnabled: !startPending,
                onRun: { startBenchmark() },
                onQuickCheck: { startBenchmark(questionLimit: Self.quickQuestionLimit) }
            )
            .frame(width: contentWidth, alignment: .leading)
        case .done, .cancelled, .error:
            VStack(alignment: .leading, spacing: 14) {
                BenchQuestionGrid(results: orchestrator.results, onSelect: { selectedResult = $0 })
                    .frame(width: contentWidth, alignment: .leading)
                BenchSummaryCard(
                    state: state,
                    score: orchestrator.score,
                    resolved: orchestrator.resolved,
                    total: orchestrator.total,
                    accuracy: orchestrator.accuracy,
                    elapsed: orchestrator.elapsed(),
                    model: orchestrator.model,
                    onQuickCheck: {
                        startBenchmark(
                            questionLimit: Self.quickQuestionLimit,
                            resetFirst: true
                        )
                    },
                    onRunFull: {
                        startBenchmark(resetFirst: true)
                    },
                    onClear: clearPresentedRun
                )
                .frame(width: contentWidth, alignment: .leading)
            }
        case .running, .paused:
            EmptyView()
        }
    }

    // MARK: - Lifecycle

    private func handleClose() {
        guard !orchestrator.state.isTerminal else {
            clearPresentedRun()
            return
        }
        router.closeBenchmark()
    }

    private func clearPresentedRun() {
        orchestrator.discardPresentedRun()
        Task { await orchestrator.refreshHistory() }
    }

    private func startBenchmark(questionLimit: Int? = nil, resetFirst: Bool = false) {
        guard !startPending else { return }
        startPending = true
        Task { @MainActor in
            if resetFirst {
                orchestrator.reset()
            }
            await orchestrator.start(questionLimit: questionLimit)
            startPending = false
        }
    }

    private func runEnter() {
        if motionEnabled {
            // Border-stroke draw-in mirrors LaunchOverlay /
            // InferenceParamsOverlay choreography.
            withAnimation(Motion.overlayBorder) {
                borderProgress = 1
            }
            // Content pops slightly after the border begins drawing so
            // the eye reads the frame first, then the contents.
            withAnimation(Motion.overlayHeaderSpring.delay(Motion.overlayHeaderDelay)) {
                contentVisible = true
            }
        } else {
            borderProgress = 1
            contentVisible = true
        }
        orchestrator.prepareForPresentation()
        Task {
            // Refresh order matters: history first (cheap, populates
            // empty-state quickly); active-run second (potentially
            // hydrates a live run if one exists daemon-side).
            await orchestrator.refreshHistory()
            await orchestrator.refreshActiveRun()
        }
    }

    private func runExit() {
        if motionEnabled {
            withAnimation(Motion.overlayExit) {
                contentVisible = false
                borderProgress = 0
            }
        } else {
            contentVisible = false
            borderProgress = 0
        }
    }

    private var startButtonTitle: String {
        if startPending {
            return startPendingTitle
        }
        if backend.health?.ok == true || backend.daemonState == .running {
            return "Run AIME 2026"
        }
        return "Start"
    }

    private var startPendingTitle: String {
        switch backend.startupPhase {
        case .launching:
            return "Starting runtime..."
        case .waitingForOwnedHealth:
            return "Loading model..."
        case .rampingFans:
            return "Preparing fans..."
        case .warming:
            return "Warming model..."
        case .ready:
            return "Starting AIME..."
        case .failed:
            return "Start failed"
        case .idle:
            if backend.health?.ok == true || backend.daemonState == .running {
                return "Starting AIME..."
            }
            return "Starting runtime..."
        }
    }
}
