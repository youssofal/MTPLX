import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - DownloadStep
//
// Live progress while `mtplx pull` runs. Polls the destination dir
// size; the orchestrator does the smoothing.
//
// Fast-paths:
//   • If the model is already installed and complete on disk, the
//     step auto-advances to Tune on appear.
//   • If a previous run was interrupted, `mtplx pull` resumes via
//     huggingface_hub's native byte-range support and we show
//     resume-aware progress.

struct DownloadStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var hasAutoStarted = false
    @State private var showingAlreadyInstalledFlash = false

    var body: some View {
        OnboardingStepContainer(
            title: title,
            subtitle: subtitle,
            stepIndex: 4,
            stepCount: OnboardingStep.allCases.count,
            onBack: (orchestrator.isDownloading || showingAlreadyInstalledFlash) ? nil : { orchestrator.returnToModelPick() },
            primary: { primaryButton },
            content: {
                if showingAlreadyInstalledFlash {
                    alreadyInstalledFlash
                        .transition(.opacity)
                } else {
                    body(for: orchestrator)
                        .transition(.opacity)
                }
            }
        )
        .onAppear { autoAdvanceIfAlreadyInstalled() }
        .animation(.easeOut(duration: 0.25), value: showingAlreadyInstalledFlash)
    }

    private var alreadyInstalledFlash: some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 36, weight: .medium))
                .foregroundStyle(Brand.success)
            Text("Already on your Mac")
                .font(.system(size: 16, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
            Text("Skipping download. Moving on.")
                .font(.system(size: 12))
                .foregroundStyle(Brand.typeSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Model already installed. Skipping download.")
    }

    private var title: String {
        let verb: String
        if orchestrator.downloadProgress?.isComplete == true {
            verb = "Downloaded"
        } else if orchestrator.isDownloading {
            verb = "Downloading"
        } else {
            verb = "Download"
        }
        if let shortName = orchestrator.state.resolvedModel?.shortName {
            return "\(verb) \(shortName)"
        }
        if let repo = orchestrator.state.resolvedRepoID {
            return "\(verb) \(repo)"
        }
        return "Download"
    }

    private var subtitle: String {
        if let progress = orchestrator.downloadProgress, !progress.destinationPath.isEmpty {
            return progress.destinationPath
        }
        return "Files land in \(orchestrator.effectiveModelDirectory). Resume is automatic."
    }

    @ViewBuilder
    private var primaryButton: some View {
        if showingAlreadyInstalledFlash {
            OnboardingPrimaryButton("Skipping...", isEnabled: false) {}
        } else if orchestrator.downloadProgress?.isComplete == true {
            OnboardingPrimaryButton("Continue") { orchestrator.goNext() }
        } else if orchestrator.isDownloading {
            OnboardingPrimaryButton("Stop") { orchestrator.cancelDownload() }
        } else if orchestrator.downloadFailure != nil {
            OnboardingPrimaryButton("Retry") { orchestrator.startDownload() }
        } else {
            OnboardingPrimaryButton("Start download") { orchestrator.startDownload() }
        }
    }

    // MARK: - Body

    @ViewBuilder
    private func body(for orchestrator: OnboardingOrchestrator) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            if !orchestrator.isDownloading,
               orchestrator.downloadProgress?.isComplete != true {
                downloadLocationControl
            }
            progressBar
            statusRow
            telemetryRow
            if let failure = orchestrator.downloadFailure {
                failureBanner(failure)
            }
            Spacer(minLength: 0)
            footnote
        }
    }

    private var downloadLocationControl: some View {
        HStack(spacing: 8) {
            Image(systemName: "internaldrive")
                .foregroundStyle(Brand.typeSecondary)
            Text(orchestrator.effectiveModelDirectory)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            Button("Change…") { chooseModelDirectory() }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    private func chooseModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Use Folder"
        panel.message = "Choose where MTPLX should download model packs."
        let current = orchestrator.effectiveModelDirectory
        if FileManager.default.fileExists(atPath: current) {
            panel.directoryURL = URL(fileURLWithPath: current, isDirectory: true)
        }
        if panel.runModal() == .OK, let url = panel.url {
            orchestrator.setModelDirectory(url.path)
        }
    }

    private var progressBar: some View {
        let fraction = orchestrator.downloadProgress?.fraction ?? 0
        return ZStack(alignment: .leading) {
            Capsule()
                .fill(Brand.separator.opacity(0.4))
                .frame(height: 8)
            Capsule()
                .fill(Brand.typeBody)
                .frame(width: max(8, 476 * CGFloat(fraction)), height: 8)
                .animation(themeStore.reduceMotionPreference ? nil : .easeInOut(duration: 0.3), value: fraction)
        }
        .accessibilityLabel("Download progress")
        .accessibilityValue("\(Int(fraction * 100)) percent")
    }

    private var statusRow: some View {
        let progress = orchestrator.downloadProgress
        let bytes = progress?.bytesOnDisk ?? 0
        let total = progress?.totalBytes
        let percent = progress?.fraction ?? 0
        return HStack(spacing: 12) {
            Text(formatBytesShort(bytes))
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
            if let total {
                Text("of \(formatBytesShort(total))")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .monospacedDigit()
            }
            Spacer()
            Text(String(format: "%.0f%%", percent * 100))
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .monospacedDigit()
        }
    }

    private var telemetryRow: some View {
        let progress = orchestrator.downloadProgress
        let rate = progress?.bytesPerSecond ?? 0
        let eta = progress?.etaSeconds
        let stalled = progress?.stalledSeconds ?? 0
        return HStack(spacing: 16) {
            if let message = progress?.statusMessage, !message.isEmpty {
                Text(message)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(stalled >= 30 ? Brand.warning : Brand.typeSecondary)
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
                Label(
                    "Stalled for \(stalled)s, checking Hugging Face…",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.caption2)
                .foregroundStyle(Brand.warning)
            }
            Spacer()
        }
    }

    private func failureBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Brand.danger)
            VStack(alignment: .leading, spacing: 4) {
                Text("Download failed")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(message)
                    .font(.system(size: 11))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(4)
                    .fixedSize(horizontal: false, vertical: true)
                TextField(
                    "Optional mirror, e.g. https://hf-mirror.com",
                    text: $orchestrator.hfMirrorEndpoint
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 11))
                .padding(.top, 4)
                Text("Downloads use the mirror instead of huggingface.co. Your HF token is never sent to a mirror.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.danger.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.danger.opacity(0.4), lineWidth: 0.5)
                )
        )
    }

    private var footnote: some View {
        let diskFree = orchestrator.freeDiskGiB()
        let text: String = orchestrator.isDownloading
            ? String(format: "You can stop anytime. %.0f GB free on this Mac.", diskFree)
            : String(format: "%.0f GB free on this Mac. Resume is automatic.", diskFree)
        return Text(text)
            .font(.caption2)
            .foregroundStyle(Brand.typeTertiary)
    }

    // MARK: - Auto-advance for already-installed models

    private func autoAdvanceIfAlreadyInstalled() {
        guard !hasAutoStarted else { return }
        hasAutoStarted = true
        guard let model = orchestrator.state.resolvedModel,
            orchestrator.isModelInstalled(model)
        else { return }
        // Flash a brief "Already installed" confirmation so the skip
        // doesn't feel like the step glitched past. ~900 ms is long
        // enough to read, short enough not to annoy the user.
        showingAlreadyInstalledFlash = true
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(900))
            orchestrator.goNext()
        }
    }

    // MARK: - Formatting helpers

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
}
