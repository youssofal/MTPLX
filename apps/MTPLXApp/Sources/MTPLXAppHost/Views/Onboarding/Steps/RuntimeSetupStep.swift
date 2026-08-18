import SwiftUI
import MTPLXAppCore

// MARK: - RuntimeSetupStep
//
// "Setting up MTPLX" — runs before download/tune so every later step
// can shell the CLI unconditionally (previously the runtime install
// hid inside the download/tune fallbacks and "Skip tune" could finish
// onboarding with no runtime installed at all).
//
// Three checklist rows driven by `RuntimeSetupService` snapshots:
//   • MTPLX engine — bundled-wheel install into the app-owned venv
//     (blocking; Retry on failure)
//   • Fan control — ThermalForge via `mtplx max --install`
//     (warning-only)
//   • Terminal command line — detects a pre-existing global CLI and
//     upgrades Homebrew installs in place (best-effort, never blocks)
//
// Auto-starts on appear; re-runs are instant no-ops because every
// phase is idempotent.

struct RuntimeSetupStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        OnboardingStepContainer(
            title: "Setting up MTPLX",
            subtitle: "Engine, fan control, and your command line — a one-time setup.",
            stepIndex: 3,
            stepCount: OnboardingStep.allCases.count,
            onBack: orchestrator.isRunningRuntimeSetup ? nil : { orchestrator.goBack() },
            primary: { primaryButton },
            content: {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(rows) { row in
                        RuntimeSetupRowView(row: row)
                    }
                    Spacer(minLength: 0)
                }
                .animation(
                    themeStore.reduceMotionPreference ? nil : .easeOut(duration: 0.2),
                    value: rows
                )
            }
        )
        .onAppear { orchestrator.startRuntimeSetup() }
    }

    private var rows: [RuntimeSetupRow] {
        if orchestrator.runtimeSetupRows.isEmpty {
            return RuntimeSetupRowID.allCases.map { RuntimeSetupRow(id: $0) }
        }
        return orchestrator.runtimeSetupRows
    }

    @ViewBuilder
    private var primaryButton: some View {
        if orchestrator.isRunningRuntimeSetup {
            OnboardingPrimaryButton("Setting up…", isEnabled: false) {}
        } else if orchestrator.runtimeSetupComplete {
            OnboardingPrimaryButton("Continue") { orchestrator.goNext() }
        } else if orchestrator.runtimeSetupFailure != nil {
            OnboardingPrimaryButton("Retry") { orchestrator.retryRuntimeSetup() }
        } else {
            OnboardingPrimaryButton("Set up") { orchestrator.startRuntimeSetup() }
        }
    }

}

// MARK: - RuntimeSetupRowView

private struct RuntimeSetupRowView: View {
    let row: RuntimeSetupRow
    @State private var copied = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            stateIcon
                .frame(width: 20, height: 20)
            VStack(alignment: .leading, spacing: 3) {
                Text(row.title.mtplxLocalized)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                if !row.detail.isEmpty {
                    Text(row.detail.mtplxLocalized)
                        .font(.system(size: 11))
                        .foregroundStyle(detailColor)
                        .lineLimit(4)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let command = row.command, !command.isEmpty {
                    commandChip(command)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(backgroundFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(borderStroke, lineWidth: 0.5)
                )
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            String(
                format: "%1$@: %2$@. %3$@".mtplxLocalized,
                row.title.mtplxLocalized,
                accessibilityState.mtplxLocalized,
                row.detail.mtplxLocalized
            )
        )
    }

    @ViewBuilder
    private var stateIcon: some View {
        switch row.state {
        case .pending:
            Image(systemName: "circle")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Brand.typeTertiary)
        case .running:
            ProgressView()
                .controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Brand.success)
        case .warning:
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Brand.warning)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(Brand.danger)
        }
    }

    private var detailColor: Color {
        switch row.state {
        case .failed: return Brand.danger
        case .warning: return Brand.warning
        default: return Brand.typeSecondary
        }
    }

    private var backgroundFill: Color {
        switch row.state {
        case .failed: return Brand.danger.opacity(0.08)
        case .warning: return Brand.warning.opacity(0.06)
        default: return Brand.separator.opacity(0.18)
        }
    }

    private var borderStroke: Color {
        switch row.state {
        case .failed: return Brand.danger.opacity(0.4)
        case .warning: return Brand.warning.opacity(0.3)
        default: return Brand.separator.opacity(0.6)
        }
    }

    private var accessibilityState: String {
        switch row.state {
        case .pending: return "Pending"
        case .running: return "In progress"
        case .done: return "Done"
        case .warning: return "Warning"
        case .failed: return "Failed"
        }
    }

    private func commandChip(_ command: String) -> some View {
        HStack(spacing: 6) {
            Text(command)
                .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(command, forType: .string)
                copied = true
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(1.6))
                    copied = false
                }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 9.5, weight: .semibold))
                    .foregroundStyle(copied ? Brand.success : Brand.typeTertiary)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(copied ? "Copied" : "Copy command")
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 8)
        .background(
            Capsule()
                .fill(Brand.separator.opacity(0.35))
        )
        .padding(.top, 2)
    }
}
