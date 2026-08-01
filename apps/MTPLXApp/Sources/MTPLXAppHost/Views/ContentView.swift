import SwiftUI
import MTPLXAppCore

// MARK: - ContentView
//
// MTPLX V1 shell, redesigned: the tab bar and Settings are ALWAYS
// accessible whether the daemon is running or not. The old
// welcome/warming takeover screens are gone. Daemon state shows as
// a state pill in the top strip + an empty-state on the Live tab
// when nothing is running yet. Start/Stop lives in the top strip.

struct ContentView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        Group {
            switch router.onboardingPhase {
            case .onboarding:
                OnboardingExperienceView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .completed:
                appShell
            }
        }
        // Allow the window to shrink to a thin bar. The dashboard reflows
        // (gauge shrinks, tiles wrap) below the old 1080×720 floor; tabs
        // are scroll views, so they degrade gracefully when narrow.
        // WindowSizingTuner enforces the same floor via
        // `window.contentMinSize` while turning OFF the hosting view's
        // content-derived extrema (the streaming minSize storm); this
        // .frame stays as the source of truth for the floor value and
        // as the fallback when the tuner is disabled.
        .frame(minWidth: 420, minHeight: 540)
        .background(WindowSizingTuner())
        .sheet(isPresented: $router.logsSheetPresented) {
            LogsSheet()
                .environmentObject(backend)
                .environmentObject(themeStore)
        }
        .sheet(isPresented: $router.aboutSheetPresented) {
            AboutSheet()
                .environmentObject(backend)
        }
        .sheet(isPresented: modelDownloadSheetPresented) {
            ModelDownloadSheet()
                .environmentObject(backend)
                .environmentObject(themeStore)
                .interactiveDismissDisabled(backend.isModelDownloading || backend.isModelTuning)
        }
        .appliesBrand()
    }

    private var modelDownloadSheetPresented: Binding<Bool> {
        Binding(
            get: { backend.pendingModelDownload != nil },
            set: { isPresented in
                if !isPresented {
                    backend.dismissModelDownloadPrompt()
                }
            }
        )
    }

    /// The pre-existing app shell — extracted from `body` verbatim so
    /// the onboarding branch can replace it cleanly without duplicating
    /// chrome / overlay setup.
    @ViewBuilder
    private var appShell: some View {
        ZStack(alignment: .top) {
            Brand.bgOuter
                .ignoresSafeArea()

            if router.benchmarkOverlayPresented {
                // AIME is a benchmark, so it gets a quiet foreground surface:
                // no background LiveTab charts, gauge animation, tab bar, or
                // hidden chat/hermes overlays burning SwiftUI work underneath.
                BenchmarkOverlay()
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.98, anchor: .center)),
                        removal: .opacity
                    ))
                    .zIndex(15)
            } else {
                VStack(spacing: 0) {
                    // The chrome strips are fixed-height anchors at the
                    // top and bottom of the window. They get a higher
                    // `layoutPriority` than the body so SwiftUI always
                    // gives them their intrinsic size first and the body
                    // only ever gets whatever's left over.
                    TopChromeStrip()
                        .layoutPriority(2)
                    ConnectionIssueBanner(state: backend.connectionState)
                        .layoutPriority(2)

                    // Dashboard + BottomTabBar are rendered for the normal
                    // app shell. Benchmark mode intentionally unmounts this
                    // subtree above so hidden charts cannot keep diffing while
                    // the model is solving AIME. Chat and Hermes get the same
                    // quiet background treatment below: keep chrome/navigation
                    // mounted, but do not keep Live charts or dashboard tabs
                    // diffing underneath a foreground work surface.
                    ZStack(alignment: .top) {
                        dashboardLayer
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .clipped()

                        if router.primaryMode == .chat {
                            ChatOverlay {
                                withAnimation(chatOverlayAnimation) {
                                    router.showDashboard()
                                }
                            }
                            .transition(chatOverlayTransition)
                            .zIndex(1)
                        } else if router.primaryMode == .hermes {
                            HermesOverlay {
                                withAnimation(chatOverlayAnimation) {
                                    router.showDashboard()
                                }
                            }
                            .transition(chatOverlayTransition)
                            .zIndex(1)
                        } else if backend.daemonState.kind == .running {
                            // Expand-chat tab only renders when the
                            // daemon is running — chat needs a daemon
                            // to talk to, so a "pull chat up" handle
                            // would be misleading when the model is
                            // stopped. The tab slides down out of view
                            // on the same chatOverlayTransition the
                            // moment the daemon transitions out of
                            // .running, and slides back up when it
                            // returns.
                            VStack(spacing: 0) {
                                Spacer(minLength: 0)
                                SurfaceExpandTab(surface: router.expandableSurface) {
                                    withAnimation(chatOverlayAnimation) {
                                        router.reopenExpandableSurface()
                                    }
                                }
                                .padding(.bottom, 12)
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .allowsHitTesting(true)
                            .transition(chatOverlayTransition)
                            .zIndex(0)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
                    .layoutPriority(0)
                    .animation(chatOverlayAnimation, value: chatSlotKey)

                    BottomTabBar()
                        .layoutPriority(2)
                }

                // Floating layers above the dashboard. NewMaxToast is the
                // pre-existing top-right toast; the two overlays are the
                // Play-button picker and the inference-params dropdown —
                // all anchored to the window, not to any particular tab.
                NewMaxToast()
                LaunchOverlay(presented: $router.launchPickerPresented)
                ModelPickerOverlay(presented: $router.modelPickerPresented)
            }

            // Reachable from both the normal shell and the benchmark header.
            InferenceParamsOverlay(presented: $router.inferenceParamsPresented)
                .zIndex(20)
        }
        .animation(.smooth(duration: 0.32), value: router.benchmarkOverlayPresented)
    }

    // MARK: - Chat overlay choreography
    //
    // Shared transition + animation between the chat overlay panel
    // and the bottom-centre expand tab so opening / closing reads as
    // one continuous slide rather than two unrelated animations.

    private var chatOverlayTransition: AnyTransition {
        .move(edge: .bottom).combined(with: .opacity)
    }

    /// Spring tuned for the chat drawer slide. Response 0.42 gives a
    /// long enough window-close feel that the close direction reads
    /// as a glide rather than a snap; damping 0.86 lands the panel
    /// without overshoot.
    private var chatOverlayAnimation: Animation? {
        guard !backend.configuration.performanceLock, !themeStore.reduceMotionPreference else {
            return nil
        }
        return .spring(response: 0.42, dampingFraction: 0.86)
    }

    /// Animation trigger key for the work-surface slot. Combines primaryMode
    /// + daemon-running state + selected collapsed surface so the implicit .animation modifier
    /// fires on either transition (open / close chat, OR daemon
    /// stopped while on the dashboard so the expand tab needs to
    /// slide away).
    private var chatSlotKey: String {
        let mode = router.primaryMode == .chat ? "chat" : "dash"
        let surface = router.primaryMode == .hermes ? "hermes" : mode
        let daemon = backend.daemonState.kind == .running ? "on" : "off"
        return "\(surface)|\(daemon)|\(router.expandableSurface.rawValue)"
    }

    @ViewBuilder
    private var dashboardLayer: some View {
        if router.primaryMode == .dashboard {
            DashboardSurface()
        } else {
            DashboardBackdropSurface()
        }
    }
}

// MARK: - DashboardSurface

struct DashboardSurface: View {
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var backend: MTPLXBackendStore

    /// The floating "Expand chat" handle hovers at the bottom-centre of
    /// this surface whenever it's visible (daemon running, dashboard
    /// mode). Reserve matching bottom clearance so a tab's bottom-most
    /// control — e.g. the KV-Quant option at the end of Settings — scrolls
    /// clear of it instead of sitting hidden under the pill.
    private var expandChatPillVisible: Bool {
        router.primaryMode == .dashboard && backend.daemonState.kind == .running
    }

    var body: some View {
        Group {
            switch router.selection {
            case .live: LiveTab()
            case .activity: ActivityTab()
            case .system: SystemTab()
            case .forge: ForgeTab()
            case .settings: SettingsTab()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if expandChatPillVisible {
                Color.clear.frame(height: 52)
            }
        }
    }
}

private struct DashboardBackdropSurface: View {
    var body: some View {
        Brand.bgOuter
            .ignoresSafeArea()
            .accessibilityHidden(true)
    }
}

// MARK: - ModelDownloadSheet

struct ModelDownloadSheet: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            details
            if pendingTune != nil {
                tuneBlock
            } else {
                progressBlock
            }
            if let failure = backend.modelDownloadFailure {
                failureBanner(failure, title: "Download failed")
            }
            if let failure = backend.modelTuneFailure {
                failureBanner(failure, title: "Tuning didn't finish")
            }
            actionRow
        }
        .padding(24)
        .frame(width: 520, alignment: .leading)
        .background(Brand.bgInner)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(
                headerTitle,
                systemImage: headerIcon
            )
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
            Text(headerSubtitle)
                .font(.system(size: 12.5))
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var details: some View {
        VStack(alignment: .leading, spacing: 10) {
            detailRow(label: "Model", value: pendingTune?.repoID ?? request?.repoID ?? "Unknown")
            detailRow(label: "Destination", value: pendingTune?.installedPath ?? progress?.destinationPath ?? request?.destinationPath ?? "")
            if let total = progress?.totalBytes ?? request?.totalBytes {
                detailRow(label: "Size", value: formatBytesShort(total))
            }
            if let pendingTune {
                detailRow(label: "Candidates", value: pendingTune.candidates.map(\.displayLabel).joined(separator: ", "))
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private var progressBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            progressBar
            statusRow
            telemetryRow
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var progressBar: some View {
        let fraction = progress?.fraction ?? 0
        return GeometryReader { proxy in
            let width = max(0, proxy.size.width)
            let filled = min(width, max(backend.isModelDownloading ? 10 : 0, width * CGFloat(fraction)))
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Brand.separator.opacity(0.55))
                Capsule()
                    .fill(backend.isModelDownloading ? Brand.accentChrome : Brand.typeBody)
                    .frame(width: filled, height: 8)
                    .animation(themeStore.reduceMotionPreference ? nil : .easeInOut(duration: 0.25), value: fraction)
            }
        }
        .frame(height: 8, alignment: .leading)
        .frame(maxWidth: .infinity)
        .clipShape(Capsule())
        .clipped()
        .accessibilityLabel("Download progress")
        .accessibilityValue("\(Int(fraction * 100)) percent")
    }

    private var statusRow: some View {
        let bytes = progress?.bytesOnDisk ?? 0
        let total = progress?.totalBytes ?? request?.totalBytes
        let fraction = progress?.fraction ?? 0
        return HStack(spacing: 10) {
            Text(formatBytesShort(bytes))
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
            if let total {
                Text("of \(formatBytesShort(total))")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .monospacedDigit()
            }
            Spacer()
            Text(String(format: "%.0f%%", fraction * 100))
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .monospacedDigit()
        }
    }

    private var telemetryRow: some View {
        let rate = progress?.bytesPerSecond ?? 0
        let eta = progress?.etaSeconds
        let stalled = progress?.stalledSeconds ?? 0
        return HStack(spacing: 16) {
            if let message = progress?.statusMessage, !message.isEmpty {
                Text(message)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(stalled >= 30 ? Brand.warning : Brand.typeSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Text(formatRate(rate))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(rate > 50_000 ? Brand.typeSecondary : Brand.typeTertiary)
                .monospacedDigit()
            if stalled == 0, let eta, eta > 0 {
                Text("ETA \(formatDuration(eta))")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .monospacedDigit()
            }
            if stalled >= 30 {
                Label("Stalled for \(stalled)s", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(Brand.warning)
            }
            Spacer()
        }
    }

    private var tuneBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let result = backend.modelTuneResult {
                tuneResultCard(result)
            } else {
                if let status = backend.modelTuneStatusMessage {
                    tuneStatusBanner(status)
                }
                ForEach(pendingTune?.candidates ?? [], id: \.self) { candidate in
                    candidateRow(for: candidate)
                }
                if !backend.isModelTuning && backend.modelTuneFailure == nil {
                    Text("Run tuning now, or skip and use the default MTP setting.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func tuneStatusBanner(_ status: String) -> some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
            Text(status)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func candidateRow(for candidate: TuneCandidate) -> some View {
        let landed = backend.modelTuneCandidatesLanded[candidate]
        let isRunning = backend.isModelTuning
            && backend.modelTuneStatusMessage == nil
            && landed == nil
            && previousCandidatesDone(before: candidate)
        return HStack(spacing: 12) {
            Group {
                if landed != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Brand.success)
                } else if isRunning {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "circle.dotted")
                        .foregroundStyle(Brand.typeTertiary)
                }
            }
            .font(.system(size: 15, weight: .medium))
            .frame(width: 18, height: 18)

            Text(candidate.displayLabel)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(Brand.typeHi)
                .lineLimit(1)
            Spacer(minLength: 10)
            if let landed {
                Text(formatTokS(landed.tokS))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Brand.typeBody)
                    .monospacedDigit()
            } else if isRunning {
                Text("running")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
            } else {
                Text("—")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func tuneResultCard(_ result: TuneResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Brand.success)
                Text("\(savedCandidateLabel(for: result)) saved")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Spacer()
                if result.bestTokS > 0 {
                    Text(formatTokS(result.bestTokS))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                        .monospacedDigit()
                }
            }
            ForEach(pendingTune?.candidates ?? [], id: \.self) { candidate in
                candidateRow(for: candidate)
            }
        }
    }

    @ViewBuilder
    private var actionRow: some View {
        if pendingTune != nil {
            tuneActionRow
        } else {
            downloadActionRow
        }
    }

    private var downloadActionRow: some View {
        HStack(spacing: 10) {
            Button {
                if backend.isModelDownloading {
                    backend.cancelModelDownload()
                } else {
                    backend.dismissModelDownloadPrompt()
                }
            } label: {
                Label(
                    backend.isModelDownloading
                        ? "Stop"
                        : "Cancel",
                    systemImage: backend.isModelDownloading ? "stop.fill" : "xmark"
                )
            }
            .buttonStyle(ModelDownloadSecondaryButtonStyle())
            .keyboardShortcut(.cancelAction)

            Spacer()

            if !backend.isModelDownloading {
                Button {
                    backend.downloadPendingModelAndStart()
                } label: {
                    Label(primaryTitle, systemImage: "arrow.down.circle.fill")
                }
                .buttonStyle(ModelDownloadPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
    }

    private var tuneActionRow: some View {
        HStack(spacing: 10) {
            Button {
                if backend.isModelTuning {
                    backend.cancelPendingModelTune()
                } else {
                    backend.dismissModelDownloadPrompt()
                }
            } label: {
                Label(
                    backend.isModelTuning ? "Stop" : "Cancel",
                    systemImage: backend.isModelTuning ? "stop.fill" : "xmark"
                )
            }
            .buttonStyle(ModelDownloadSecondaryButtonStyle())
            .keyboardShortcut(.cancelAction)

            Spacer()

            if !backend.isModelTuning {
                if backend.modelTuneResult != nil {
                    Button {
                        backend.startPendingTunedModel()
                    } label: {
                        Label(startTitle, systemImage: "play.fill")
                    }
                    .buttonStyle(ModelDownloadPrimaryButtonStyle())
                    .keyboardShortcut(.defaultAction)
                } else {
                    Button {
                        backend.skipPendingModelTune()
                    } label: {
                        Label(
                            backend.modelTuneFailure == nil ? "Skip" : "Skip and Use Default",
                            systemImage: "forward.fill"
                        )
                    }
                    .buttonStyle(ModelDownloadSecondaryButtonStyle())

                    Button {
                        backend.runPendingModelTune()
                    } label: {
                        Label(backend.modelTuneFailure == nil ? "Run Tuning" : "Retry", systemImage: "speedometer")
                    }
                    .buttonStyle(ModelDownloadPrimaryButtonStyle())
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
    }

    private func detailRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .frame(width: 92, alignment: .leading)
            Text(value.isEmpty ? "—" : value)
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(2)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func failureBanner(_ message: String, title: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Brand.danger)
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(message)
                    .font(.system(size: 11))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(4)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.danger.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.danger.opacity(0.4), lineWidth: 0.5)
                )
        )
    }

    private var request: PendingModelDownload? {
        backend.pendingModelDownload
    }

    private var progress: DownloadProgressSnapshot? {
        backend.modelDownloadProgress
    }

    private var pendingTune: PendingModelTune? {
        backend.pendingModelTune
    }

    private var headerTitle: String {
        if let pendingTune {
            if backend.isModelTuning {
                return "Tuning \(pendingTune.shortName)"
            }
            if backend.modelTuneResult != nil {
                return "\(pendingTune.shortName) tuned"
            }
            return "Tune \(pendingTune.shortName)?"
        }
        return "Download \(request?.shortName ?? "model")?"
    }

    private var headerSubtitle: String {
        if pendingTune != nil {
            if backend.modelTuneResult != nil {
                return "MTPLX saved the measured setting for this model."
            }
            return "MTPLX can test this model on your Mac and save the fastest MTP setting. Fans may ramp for a few minutes."
        }
        return "MTPLX found only a partial model folder. The full weights need to download before this model can start."
    }

    private var headerIcon: String {
        if pendingTune != nil {
            if backend.modelTuneResult != nil { return "checkmark.circle.fill" }
            return "speedometer"
        }
        return "arrow.down.circle.fill"
    }

    private var primaryTitle: String {
        if backend.modelDownloadFailure != nil {
            return "Retry"
        }
        if progress?.isComplete == true {
            return request?.launchAction == .restart ? "Restart" : "Start"
        }
        if request?.launchAction == .restart {
            return "Download & Restart"
        }
        return "Download & Start"
    }

    private var startTitle: String {
        pendingTune?.launchAction == .restart ? "Restart" : "Start"
    }

    private func previousCandidatesDone(before candidate: TuneCandidate) -> Bool {
        let candidates = pendingTune?.candidates ?? []
        guard let index = candidates.firstIndex(of: candidate), index > 0 else { return true }
        for earlier in candidates[..<index] {
            if backend.modelTuneCandidatesLanded[earlier] == nil {
                return false
            }
        }
        return true
    }

    private func savedCandidateLabel(for result: TuneResult) -> String {
        if result.bestCandidate != .ar {
            return result.bestCandidate.displayLabel
        }
        if let bestMTP = result.allCandidates
            .filter({ $0.candidate != .ar })
            .max(by: { $0.tokS < $1.tokS })
        {
            return bestMTP.candidate.displayLabel
        }
        return pendingTune?.modelFamily == "gemma4" ? TuneCandidate.block6.displayLabel : TuneCandidate.d2.displayLabel
    }

    private func formatBytesShort(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        if gib >= 1 { return String(format: "%.2f GB", gib) }
        let mib = Double(bytes) / 1_048_576.0
        return String(format: "%.0f MB", mib)
    }

    private func formatRate(_ bytesPerSecond: Double) -> String {
        if bytesPerSecond < 1024 { return "—" }
        let mbps = bytesPerSecond / 1_048_576.0
        return String(format: "%.1f MB/s", mbps)
    }

    private func formatDuration(_ seconds: Double) -> String {
        let total = Int(max(0, seconds))
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }

    private func formatTokS(_ value: Double) -> String {
        guard value > 0 else { return "—" }
        return String(format: "%.1f tps", value)
    }
}

private struct ModelDownloadPrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .labelStyle(.titleAndIcon)
            .foregroundStyle(Brand.bgOuter)
            .padding(.horizontal, 14)
            .frame(height: 34)
            .background(
                Capsule()
                    .fill(Brand.typeBody.opacity(configuration.isPressed ? 0.82 : 1))
            )
    }
}

private struct ModelDownloadSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .labelStyle(.titleAndIcon)
            .foregroundStyle(Brand.typeSecondary)
            .padding(.horizontal, 12)
            .frame(height: 34)
            .background(
                Capsule()
                    .fill(Brand.cardSurface.opacity(configuration.isPressed ? 0.7 : 1))
                    .overlay(Capsule().stroke(Brand.separator, lineWidth: 0.5))
            )
    }
}
